"""
Background atmosphere and PICASO/photochem coupling utilities.

This file contains:
- Kzz profile utilities
- manual photochem coupling using photochem.extensions.gasgiants
- photochemical haze source construction
- conversion from photochem output into a ptchem_df-like object
- temporary context construction for the current Caligo scaffold
"""

from __future__ import annotations

import time as walltime

import numpy as np
import pandas as pd
import yaml
from photochem.extensions import gasgiants


def safe_log10(x):
    return np.log10(np.maximum(np.asarray(x, dtype=float), 1e-300))


def build_faces(zc):
    """
    Build cell faces from cell centers.
    """
    zc = np.asarray(zc, dtype=float)

    if zc.size < 2:
        raise ValueError("Need at least two grid centers to build faces.")

    zf = np.empty(zc.size + 1)
    zf[1:-1] = 0.5 * (zc[:-1] + zc[1:])
    zf[0] = zc[0] - 0.5 * (zc[1] - zc[0])
    zf[-1] = zc[-1] + 0.5 * (zc[-1] - zc[-2])

    return zf


def kzz_profile_from_settings(P_bar, settings):
    """
    Smooth three-zone Kzz profile.

    Parameters
    ----------
    P_bar : array-like
        Pressure in bar.
    settings : dict
        Should contain upper, mid, deep, break_top_bar, break_deep_bar,
        and optionally width_dex.

    Returns
    -------
    np.ndarray
        Kzz profile in cm^2/s.
    """
    P_bar = np.asarray(P_bar, dtype=float)

    upper = settings.get("upper", 1e9)
    mid = settings.get("mid", 1e8)
    deep = settings.get("deep", 1e8)
    break_top_bar = settings.get("break_top_bar", 1e-5)
    break_deep_bar = settings.get("break_deep_bar", 30.0)
    width_dex = settings.get("width_dex", 0.4)

    logP = safe_log10(P_bar)
    logP_top = np.log10(max(break_top_bar, 1e-300))
    logP_deep = np.log10(max(break_deep_bar, 1e-300))

    w_top = 0.5 * (1.0 - np.tanh((logP - logP_top) / width_dex))
    w_deep = 0.5 * (1.0 + np.tanh((logP - logP_deep) / width_dex))
    w_mid = np.clip(1.0 - w_top - w_deep, 0.0, 1.0)

    return w_top * upper + w_mid * mid + w_deep * deep


def _get_case_climate_profile(case):
    """
    Get the current climate pressure/temperature profile from a CaligoCase.

    Preferred source:
        case._climate_pressure and case._climate_temperature,
    which are cached when case.inputs_climate(...) is called.

    Fallback:
        case._last_pt, cached from case.guillot_pt(...).
    """
    if getattr(case, "_climate_pressure", None) is not None:
        Pbar = np.asarray(case._climate_pressure, dtype=float)
        T = np.asarray(case._climate_temperature, dtype=float)
        return Pbar, T

    if getattr(case, "_last_pt", None) is not None:
        pt = case._last_pt
        Pbar = np.asarray(pt["pressure"].values, dtype=float)
        T = np.asarray(pt["temperature"].values, dtype=float)
        return Pbar, T

    raise RuntimeError(
        "Could not find a climate P-T profile. "
        "Run pt = case.guillot_pt(...) and case.inputs_climate(...) first."
    )


def _get_case_kzz_profile(case, Pbar):
    """
    Build or retrieve the Kzz profile on the supplied pressure grid.
    """
    Pbar = np.asarray(Pbar, dtype=float)
    settings = case.kzz_settings

    if settings.get("profile") is not None:
        Kzz = np.asarray(settings["profile"], dtype=float)
        if Kzz.shape != Pbar.shape:
            raise ValueError(
                "case.kzz(profile=...) has a different shape than the pressure grid."
            )
        return Kzz

    if settings.get("value") is not None:
        return np.ones_like(Pbar, dtype=float) * float(settings["value"])

    return kzz_profile_from_settings(Pbar, settings)


def _get_case_cto_for_photochem(case):
    """
    Get a valid C/O value for photochem initialization.

    Priority:
        1. PICASO atmosphere cto_absolute, if present and valid
        2. PICASO atmosphere cto, if present and valid
        3. Caligo photochem setting cto_relative
        4. Default solar-ish relative value 1.0

    Important:
        PICASO may create keys like cto_absolute but leave them as None.
        We must not pass None into photochem because photochem checks CtoO <= 0.
    """

    def _valid_cto(value):
        try:
            if value is None:
                return False

            value = float(value)

            if not np.isfinite(value):
                return False

            if value <= 0:
                return False

            return True

        except Exception:
            return False

    try:
        atm = case.picaso.inputs.get("atmosphere", {})

        for key in ("cto_absolute", "cto"):
            if key in atm and _valid_cto(atm[key]):
                return float(atm[key])

    except Exception:
        pass

    cto_setting = case.photochem_settings.get("cto_relative", 1.0)

    if _valid_cto(cto_setting):
        return float(cto_setting)

    return 1.0


def run_photochem_from_case(case, find_steady_state=True):
    """
    Manually run photochem from the current Caligo/PICASO case.

    This mirrors the working path in the original monolithic code:

        P_dyn_cl = (P_bar_climate * 1e6)[::-1]
        T_cl = T_guess[::-1]
        Kzz_cl = kzz_profile(P_bar_climate[::-1])

        pc = gasgiants.EvoAtmosphereGasGiant(...)
        pc.initialize_to_climate_equilibrium_PT(...)
        pc.find_steady_state()

    Returns
    -------
    dict
        Contains pc, Pbar_pc, z_pc, T_pc, Pbar_climate, T_climate, Kzz_climate.
    """
    if not case.photochem_settings:
        raise RuntimeError(
            "No photochem settings found. Call case.photochem(...) before "
            "case.run_photochem()."
        )

    if getattr(case, "_planet_mass_cgs", None) is None:
        raise RuntimeError(
            "Planet mass was not cached. Call case.gravity(...) before "
            "case.run_photochem()."
        )

    if getattr(case, "_planet_radius_cgs", None) is None:
        raise RuntimeError(
            "Planet radius was not cached. Call case.gravity(...) before "
            "case.run_photochem()."
        )

    t0 = walltime.time()

    settings = case.photochem_settings

    Pbar_climate, T_climate = _get_case_climate_profile(case)
    Kzz_climate = _get_case_kzz_profile(case, Pbar_climate)

    # Photochem wants pressure in dyn/cm^2. This follows the original code.
    # The reversal preserves the original bottom/top order convention.
    P_dyn_cl = (Pbar_climate * 1e6)[::-1]
    T_cl = T_climate[::-1]
    Kzz_cl = Kzz_climate[::-1]

    pc = gasgiants.EvoAtmosphereGasGiant(
        settings["mechanism_file"],
        settings["stellar_flux_file"],
        planet_mass=case._planet_mass_cgs,
        planet_radius=case._planet_radius_cgs,
        thermo_file=settings["thermo_file"],
    )

    pc.gdat.P_ref = settings.get("P_ref", 1e7)
    pc.gdat.TOA_pressure_avg = settings.get("TOA_pressure", 1e-4)
    pc.gdat.BOA_pressure_factor = settings.get("BOA_pressure_factor", 1.0)

    metallicity = settings.get("metallicity", 100.0)
    CtoO = _get_case_cto_for_photochem(case)

    print("\n" + "=" * 70, flush=True)
    print("CALIGO: PHOTOCHEM STEADY STATE", flush=True)
    print("=" * 70, flush=True)
    print(
        f"Photochem input P range = "
        f"[{Pbar_climate.min():.3e}, {Pbar_climate.max():.3e}] bar",
        flush=True,
    )
    print(
        f"Photochem input T range = "
        f"[{T_climate.min():.1f}, {T_climate.max():.1f}] K",
        flush=True,
    )
    print(
        f"Photochem input Kzz range = "
        f"[{Kzz_climate.min():.3e}, {Kzz_climate.max():.3e}] cm^2/s",
        flush=True,
    )
    print(f"metallicity = {metallicity}", flush=True)
    print(f"C/O passed to photochem = {CtoO}", flush=True)

    pc.initialize_to_climate_equilibrium_PT(
        P_dyn_cl,
        T_cl,
        Kzz_cl,
        metallicity=metallicity,
        CtoO=CtoO,
    )

    if find_steady_state:
        print("Running pc.find_steady_state()...", flush=True)
        pc.find_steady_state()
    else:
        print("Initialized photochem but did not run steady state.", flush=True)

    Pbar_pc = np.asarray(pc.wrk.pressure, dtype=float) / 1e6
    z_pc = np.asarray(pc.var.z, dtype=float)
    T_pc = np.asarray(pc.var.temperature, dtype=float)

    print(
        f"Photochem output P range = "
        f"[{Pbar_pc.min():.3e}, {Pbar_pc.max():.3e}] bar",
        flush=True,
    )
    print(f"Photochem nz = {len(Pbar_pc)}", flush=True)
    print(f"Photochem finished in {walltime.time() - t0:.2f} s.", flush=True)

    return {
        "pc": pc,
        "Pbar_pc": Pbar_pc,
        "z_pc": z_pc,
        "T_pc": T_pc,
        "Pbar_climate": Pbar_climate,
        "T_climate": T_climate,
        "Kzz_climate": Kzz_climate,
    }


# ----------------------------------------------------------------------
# Photochemical haze source utilities
# ----------------------------------------------------------------------


def get_species_mw(thermo_path):
    """
    Read species molecular weights and elemental compositions from
    photochem_thermo.yaml.
    """
    with open(thermo_path) as f:
        data = yaml.safe_load(f)

    atom_mw = {a["name"]: a["mass"] for a in data["atoms"]}

    species_mw = {}
    species_atoms = {}

    for sp in data["species"]:
        comp = sp["composition"]
        species_atoms[sp["name"]] = dict(comp)
        species_mw[sp["name"]] = sum(atom_mw[at] * n for at, n in comp.items())

    return species_mw, species_atoms, atom_mw


def get_photolysis_parents(rxn_path):
    """
    Return all species that appear on the left side of photolysis reactions.
    """
    with open(rxn_path) as f:
        data = yaml.safe_load(f)

    parents = set()

    for rx in data.get("reactions", data):
        if str(rx.get("type", "")).lower() != "photolysis":
            continue

        left = rx["equation"].split("=>")[0]

        for token in left.split("+"):
            sp = token.strip()
            if sp and sp.lower() != "hv":
                parents.add(sp)

    return sorted(parents)


def L_phot(pc, species):
    """
    Total photolysis loss rate profile for one species.

    Returns a profile in units consistent with photochem's loss array.
    In the original prototype this is used as the number of photolyzed
    molecules per cm^3 per second.
    """
    pl = pc.production_and_loss(species, pc.wrk.usol)

    labels = [str(x).lower() for x in pl.loss_rx]
    hv_idx = [i for i, rx in enumerate(labels) if "hv" in rx or "photon" in rx]

    if len(hv_idx) == 0:
        return np.zeros(pl.loss.shape[0])

    return pl.loss[:, hv_idx].sum(axis=1)


def haze_mass_per_photolyzed_molecule(species, mw_map, species_atoms, atom_mw, source_settings):
    """
    Convert one photolyzed molecule into an effective haze mass.

    Supported yield models:
        parent_molecular_mass
        carbon_mass_only
        heavy_atom_mass
        fixed_literature_flux
    """
    if species not in mw_map:
        return 0.0

    comp = species_atoms.get(species, {})
    parent_mw = mw_map[species]

    yield_model = source_settings.get("yield_model", "carbon_mass_only")
    yield_haze = source_settings.get("yield_haze", 0.10)

    if yield_model == "parent_molecular_mass":
        effective_mw = parent_mw

    elif yield_model == "carbon_mass_only":
        effective_mw = comp.get("C", 0) * atom_mw.get("C", 12.011)

    elif yield_model == "heavy_atom_mass":
        effective_mw = sum(atom_mw[at] * n for at, n in comp.items() if at != "H")

    elif yield_model == "fixed_literature_flux":
        effective_mw = comp.get("C", 0) * atom_mw.get("C", 12.011)

    else:
        raise ValueError(
            "yield_model must be parent_molecular_mass, carbon_mass_only, "
            "heavy_atom_mass, or fixed_literature_flux."
        )

    # molecular weight / Avogadro = grams per molecule
    NA = 6.02214076e23
    return yield_haze * effective_mw / NA


def build_photochem_haze_source_from_case(case, photochem_result=None):
    """
    Build a raw photochemical haze mass source profile from a Caligo case.

    Uses:
        - case.haze_source_settings["parents"], usually ["HCN", "C2H2"]
        - case.haze_source_settings["yield_model"]
        - case.haze_source_settings["yield_haze"]

    Returns
    -------
    dict
        Contains z_src_sorted, Pbar_src_sorted, q_raw, F_photochem_raw,
        species_flux, species_effective_mass, and precursor metadata.
    """
    if photochem_result is None:
        photochem_result = getattr(case, "_photochem_result", None)

    if photochem_result is None:
        raise ValueError(
            "No photochem result found. Run case.run_photochem() before "
            "building a photochemical haze source."
        )

    pc = photochem_result["pc"]
    Pbar_pc = np.asarray(photochem_result["Pbar_pc"], dtype=float)
    z_pc = np.asarray(photochem_result["z_pc"], dtype=float)

    source_settings = case.haze_source_settings
    photochem_settings = case.photochem_settings

    rxn_path = photochem_settings["mechanism_file"]
    thermo_path = photochem_settings["thermo_file"]

    order_z_pc = np.argsort(z_pc)
    z_src_sorted = z_pc[order_z_pc]
    Pbar_src_sorted = Pbar_pc[order_z_pc]

    zf_src_sorted = build_faces(z_src_sorted)
    dz_src_sorted = np.diff(zf_src_sorted)

    parents_all = get_photolysis_parents(rxn_path)
    mw_map, species_atoms, atom_mw = get_species_mw(thermo_path)

    use_all_photolysis = source_settings.get("use_all_photolysis", False)
    requested_parents = set(source_settings.get("parents", ["HCN", "C2H2"]))

    if use_all_photolysis:
        precursors = [s for s in parents_all if s in mw_map]
    else:
        precursors = [s for s in parents_all if s in requested_parents and s in mw_map]

    if not precursors:
        raise RuntimeError(
            "No haze precursor photolysis parents found. "
            "Check case.haze_source(parents=...) and the reaction file."
        )

    print("\n" + "=" * 70, flush=True)
    print("CALIGO: PHOTOCHEM HAZE SOURCE", flush=True)
    print("=" * 70, flush=True)
    print("Photolysis haze precursors:", precursors, flush=True)
    print(
        f"Source yield model = {source_settings.get('yield_model', 'carbon_mass_only')}",
        flush=True,
    )
    print(
        f"Yield haze = {source_settings.get('yield_haze', 0.10)}",
        flush=True,
    )

    q_raw = np.zeros_like(z_src_sorted)
    species_flux = {}
    species_effective_mass = {}

    for sp in precursors:
        g_per_photolyzed = haze_mass_per_photolyzed_molecule(
            sp,
            mw_map,
            species_atoms,
            atom_mw,
            source_settings,
        )

        q_sp = L_phot(pc, sp)[order_z_pc] * g_per_photolyzed
        q_raw += q_sp

        species_flux[sp] = float(np.sum(q_sp * dz_src_sorted))
        species_effective_mass[sp] = float(g_per_photolyzed)

    F_photochem_raw = float(np.sum(q_raw * dz_src_sorted))

    print(f"Raw haze source flux = {F_photochem_raw:.3e} g cm^-2 s^-1", flush=True)

    for sp, Fsp in species_flux.items():
        frac = Fsp / max(F_photochem_raw, 1e-300)
        print(
            f"  {sp:>8s}: {Fsp:.3e} ({frac:.2f}), "
            f"haze mass per photolyzed molecule = "
            f"{species_effective_mass[sp]:.3e} g",
            flush=True,
        )

    if np.any(q_raw > 0):
        i_peak = int(np.argmax(q_raw))
        print(
            f"Source peak P = {Pbar_src_sorted[i_peak]:.3e} bar, "
            f"z = {z_src_sorted[i_peak]:.3e} cm",
            flush=True,
        )
    else:
        print("Warning: q_raw is zero everywhere.", flush=True)

    return {
        "z_src_sorted": z_src_sorted,
        "Pbar_src_sorted": Pbar_src_sorted,
        "dz_src_sorted": dz_src_sorted,
        "q_raw": q_raw,
        "F_photochem_raw": F_photochem_raw,
        "species_flux": species_flux,
        "species_effective_mass": species_effective_mass,
        "precursors": precursors,
        "parents_all": parents_all,
        "source_yield_model": source_settings.get("yield_model", "carbon_mass_only"),
        "yield_haze": source_settings.get("yield_haze", 0.10),
    }


# ----------------------------------------------------------------------
# Temporary scaffold output conversion
# ----------------------------------------------------------------------


def photochem_result_to_climate_out(case, photochem_result):
    """
    Convert a manual photochem result into the ptchem_df-style structure
    expected by case.haze(...).

    This is a temporary bridge while Caligo's real haze context builder
    is being moved in.
    """
    Pbar_pc = np.asarray(photochem_result["Pbar_pc"], dtype=float)
    T_pc = np.asarray(photochem_result["T_pc"], dtype=float)

    Kzz_pc = _get_case_kzz_profile(case, Pbar_pc)

    return {
        "ptchem_df": pd.DataFrame(
            {
                "pressure": Pbar_pc,
                "temperature": T_pc,
                "Kzz": Kzz_pc,
            }
        )
    }


def _get_ptchem_df(climate_out):
    """
    Extract the PICASO/Caligo ptchem dataframe from climate output.
    """
    if not isinstance(climate_out, dict):
        raise TypeError(
            "Expected climate_out to be a dictionary returned by PICASO climate() "
            "or by case.photochem_climate_out()."
        )

    if "ptchem_df" not in climate_out:
        raise KeyError(
            "Expected climate_out['ptchem_df']. "
            "Use case.photochem_climate_out() or pass a dict with ptchem_df."
        )

    return climate_out["ptchem_df"]


def _extract_pressure_temperature(ptchem_df):
    """
    Extract pressure and temperature arrays from a ptchem dataframe.
    """
    if "pressure" not in ptchem_df:
        raise KeyError("ptchem_df does not contain a 'pressure' column.")

    if "temperature" not in ptchem_df:
        raise KeyError("ptchem_df does not contain a 'temperature' column.")

    Pbar = np.asarray(ptchem_df["pressure"], dtype=float)
    T = np.asarray(ptchem_df["temperature"], dtype=float)

    return Pbar, T


def _extract_kzz(case, ptchem_df, Pbar):
    """
    Extract or reconstruct Kzz.
    """
    for key in ("Kzz", "kz", "kzz"):
        if key in ptchem_df:
            return np.asarray(ptchem_df[key], dtype=float)

    settings = case.kzz_settings

    if settings.get("profile") is not None:
        return np.asarray(settings["profile"], dtype=float)

    if settings.get("value") is not None:
        return np.ones_like(Pbar, dtype=float) * float(settings["value"])

    return kzz_profile_from_settings(Pbar, settings)


def build_context_from_case(case, climate_out):
    """
    Build the context dictionary that Caligo's haze solver will use.

    Current scaffold version:
        extracts pressure, temperature, Kzz, and stored settings.

    Later:
        this will be replaced/expanded with the full microphysics context
        builder from the original Caligo prototype.
    """
    ptchem_df = _get_ptchem_df(climate_out)
    Pbar, T = _extract_pressure_temperature(ptchem_df)
    Kzz = _extract_kzz(case, ptchem_df, Pbar)

    ctx = {
        "case": case,
        "climate_out": climate_out,
        "ptchem_df": ptchem_df,

        # Raw PICASO / photochem profiles.
        "Pbar_raw": Pbar,
        "T_raw": T,
        "Kzz_raw": Kzz,

        # Stored settings.
        "photochem_settings": dict(case.photochem_settings),
        "kzz_settings": dict(case.kzz_settings),
        "microphysics_grid_settings": dict(case.microphysics_grid_settings),
        "haze_source_settings": dict(case.haze_source_settings),
        "particle_settings": dict(case.particle_settings),
        "aggregate_settings": dict(case.aggregate_settings),
        "coagulation_settings": dict(case.coagulation_settings),
        "transport_settings": dict(case.transport_settings),
        "sink_settings": dict(case.sink_settings),
        "solver_settings": dict(case.solver_settings),
    }

    if getattr(case, "_source_info", None) is not None:
        ctx["source_info"] = case._source_info

    return ctx