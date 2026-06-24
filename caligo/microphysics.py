"""
Caligo haze microphysics.

This module contains the time-dependent haze microphysics solver used by
Caligo. The current implementation includes:

- construction of a microphysics context from photochem output
- conservative remapping of photochemical or Gaussian haze sources
- particle-bin source injection
- compact-particle and fractal-aggregate geometry
- VIRGA settling velocities
- Brownian coagulation with the Fuchs transition-regime correction
- optional electrostatic suppression of coagulation
- optional differential-settling coagulation
- eddy diffusion
- particle settling and lower-boundary removal
- parameterized thermal destruction
- optional deep removal and passive recycled-gas bookkeeping
- adaptive time integration to a steady or quasi-steady state

Important scope note:
The thermal destruction sink is parameterized rather than a physical
material-specific ablation model. The recycled-gas tracer is passive and is
not chemically coupled back into photochem.
"""
from __future__ import annotations

import time as walltime

import numpy as np

try:
    from virga.root_functions import vfall
except Exception as exc:  # pragma: no cover
    vfall = None
    _VFALL_IMPORT_ERROR = exc
else:
    _VFALL_IMPORT_ERROR = None


# Physical constants in cgs
k_B = 1.380649e-16
m_H = 1.6735575e-24
G_cgs = 6.67430e-8


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------


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


def integrate_source_over_interval(zf_src, q_src, z_left, z_right):
    """
    Integrate a piecewise-constant source profile over an arbitrary z interval.
    """
    zf_src = np.asarray(zf_src, dtype=float)
    q_src = np.asarray(q_src, dtype=float)

    if z_right <= z_left:
        return 0.0

    total = 0.0

    for i in range(q_src.size):
        a = max(z_left, zf_src[i])
        b = min(z_right, zf_src[i + 1])

        if b > a:
            total += q_src[i] * (b - a)

    return float(total)


def conservative_remap(zf_src, q_src, zf_dst):
    """
    Conservatively remap a piecewise-constant source profile to a new grid.
    """
    zf_src = np.asarray(zf_src, dtype=float)
    q_src = np.asarray(q_src, dtype=float)
    zf_dst = np.asarray(zf_dst, dtype=float)

    q_dst = np.zeros(zf_dst.size - 1)

    for j in range(q_dst.size):
        a_dst = zf_dst[j]
        b_dst = zf_dst[j + 1]
        width_dst = b_dst - a_dst

        if width_dst <= 0:
            raise ValueError("Destination grid faces must increase.")

        integral = integrate_source_over_interval(zf_src, q_src, a_dst, b_dst)
        q_dst[j] = integral / width_dst

    return q_dst


# ---------------------------------------------------------------------
# Gas-property approximations
# ---------------------------------------------------------------------


def mean_free_path(Pdyn, T, molecular_diameter_cm=2.8e-8):
    """
    Gas mean free path in cm.

    Direct translation of the old code:
        n = P / kT
        lambda = 1 / (sqrt(2) pi d^2 n)
    """
    Pdyn = np.asarray(Pdyn, dtype=float)
    T = np.asarray(T, dtype=float)

    n_gas = Pdyn / (k_B * np.maximum(T, 1.0))
    return 1.0 / (
        np.sqrt(2.0)
        * np.pi
        * molecular_diameter_cm**2
        * np.maximum(n_gas, 1.0)
    )


def gas_viscosity(T):
    """
    H2-dominated dynamic viscosity in g cm^-1 s^-1.

    Direct translation of the old code:
        8.9e-6 * (T/300)^0.70 * 10
    """
    return 8.9e-6 * (np.asarray(T, dtype=float) / 300.0) ** 0.70 * 10.0


def radius_from_mass_density(mass, rho):
    """
    Compact-sphere radius from particle mass and material density.
    """
    mass = np.maximum(np.asarray(mass, dtype=float), 1e-300)
    return (3.0 * mass / (4.0 * np.pi * max(rho, 1e-300))) ** (1.0 / 3.0)




def compute_mu_profile_from_photochem(case, Pbar_target):
    """
    Estimate mean molecular weight on the target pressure grid.

    This follows the original prototype logic more closely:

        species_index = {sp: i for i, sp in enumerate(pc.dat.species_names)}
        gas_sp = list(pc.dat.species_names[pc.dat.np:(-2 - pc.dat.nsl)])

    The photochem solution array pc.wrk.usol can have fewer rows than
    pc.dat.species_names because photochem includes bookkeeping species
    such as particles/background/surface species in species_names.
    """
    default_mu = case.microphysics_grid_settings.get("mu_default", 2.3)

    photochem_result = getattr(case, "_photochem_result", None)
    if photochem_result is None:
        return np.ones_like(Pbar_target, dtype=float) * default_mu

    pc = photochem_result.get("pc")
    Pbar_pc = np.asarray(photochem_result.get("Pbar_pc"), dtype=float)

    if pc is None or Pbar_pc.size == 0:
        return np.ones_like(Pbar_target, dtype=float) * default_mu

    try:
        thermo_path = case.photochem_settings["thermo_file"]

        from .background import get_species_mw

        mw_map, _, _ = get_species_mw(thermo_path)

        species_names = np.asarray(pc.dat.species_names)
        species_index = {sp: i for i, sp in enumerate(species_names)}

        usol = np.asarray(pc.wrk.usol, dtype=float)

        if usol.ndim != 2:
            raise ValueError(f"Expected pc.wrk.usol to be 2D, got shape {usol.shape}")

        if usol.shape[1] == Pbar_pc.size:
            usol_species_by_layer = usol
        elif usol.shape[0] == Pbar_pc.size:
            usol_species_by_layer = usol.T
        else:
            raise ValueError(
                "Could not infer photochem usol layer axis. "
                f"usol shape={usol.shape}, Pbar_pc size={Pbar_pc.size}"
            )

        nz_local = usol_species_by_layer.shape[1]
        n_species_usol = usol_species_by_layer.shape[0]

        npart = int(getattr(pc.dat, "np", 0))
        nsl = int(getattr(pc.dat, "nsl", 0))

        stop = -2 - nsl
        if stop == 0:
            gas_sp = list(species_names[npart:])
        else:
            gas_sp = list(species_names[npart:stop])

        n_tot = np.zeros(nz_local)
        mu_num = np.zeros(nz_local)

        used_species = []

        for sp in gas_sp:
            if sp not in mw_map:
                continue

            if sp not in species_index:
                continue

            idx = species_index[sp]

            if idx < 0 or idx >= n_species_usol:
                continue

            n_sp = np.asarray(usol_species_by_layer[idx, :], dtype=float)
            n_sp = np.maximum(n_sp, 0.0)

            n_tot += n_sp
            mu_num += n_sp * mw_map[sp]
            used_species.append(sp)

        mu_pc = mu_num / np.maximum(n_tot, 1e-300)

        if not np.any(np.isfinite(mu_pc)) or np.nanmax(mu_pc) <= 0:
            raise ValueError("Mean molecular weight calculation produced no usable values.")

        sP = np.argsort(safe_log10(Pbar_pc))

        mu_target = np.interp(
            safe_log10(Pbar_target),
            safe_log10(Pbar_pc[sP]),
            mu_pc[sP],
        )

        if not np.all(np.isfinite(mu_target)):
            raise ValueError("Non-finite interpolated mu profile.")

        print(
            "Mean molecular weight computed from photochem: "
            f"mu range = [{mu_target.min():.3f}, {mu_target.max():.3f}], "
            f"using {len(used_species)} gas species.",
            flush=True,
        )

        return mu_target

    except Exception as exc:
        print(
            "Warning: could not compute mean molecular weight from photochem; "
            f"using mu_default={default_mu}. Reason: {exc}",
            flush=True,
        )
        return np.ones_like(Pbar_target, dtype=float) * default_mu


# ---------------------------------------------------------------------
# Particle geometry
# ---------------------------------------------------------------------



def aggregate_weight_from_pressure(Pbar, aggregate_settings):
    """
    Smooth switch controlling where particles behave like aggregates.
    """
    Pbar = np.asarray(Pbar, dtype=float)

    use_aggregates = aggregate_settings.get("use", True)
    pressure_dependent = aggregate_settings.get("pressure_dependent", True)

    if not use_aggregates:
        return np.zeros_like(Pbar)

    if not pressure_dependent:
        return np.ones_like(Pbar)

    transition_p = aggregate_settings.get("transition_p_bar", 1e-6)
    width_dex = aggregate_settings.get("transition_width_dex", 0.5)

    x = (safe_log10(Pbar) - np.log10(max(transition_p, 1e-300))) / max(width_dex, 1e-6)

    return 0.5 * (1.0 + np.tanh(x))


def df_layer_from_pressure(Pbar, aggregate_settings):
    """
    Pressure-dependent fractal dimension.

    Low pressure: Df approaches 3, compact-like.
    High pressure: Df approaches df_fixed.
    """
    Pbar = np.asarray(Pbar, dtype=float)

    df_fixed = aggregate_settings.get("df_fixed", 2.0)
    w_agg = aggregate_weight_from_pressure(Pbar, aggregate_settings)

    return (1.0 - w_agg) * 3.0 + w_agg * df_fixed

def build_particle_geometry(nz, Pbar, particle_settings, aggregate_settings):
    """
    Build particle size/mass grid and aggregate geometry.

    Direct translation of the old code design:
      - r_monomer_nm defines the smallest compact mass bin.
      - r_primary_nm defines the primary aggregate monomer.
      - bins smaller than one primary remain compact.
      - Df may vary with pressure.
      - aggregate collision radius and drag radius are controlled separately.
    """
    n_bin = int(particle_settings.get("n_bin", 40))
    rho_haze = float(particle_settings.get("rho_haze", 1.0))
    r_monomer_nm = float(particle_settings.get("r_monomer_nm", 1.0))
    bin_mass_ratio = float(particle_settings.get("bin_mass_ratio", 2.0))

    use_aggregates = bool(aggregate_settings.get("use", True))
    df_fixed = float(aggregate_settings.get("df_fixed", 2.0))
    r_primary_nm = float(aggregate_settings.get("r_primary_nm", 10.0))
    k0_agg = float(aggregate_settings.get("k0", aggregate_settings.get("k0_agg", 1.0)))

    use_agg_in_settling = bool(
        aggregate_settings.get(
            "use_agg_in_settling",
            aggregate_settings.get("use_in_settling", True),
        )
    )
    use_agg_in_coag = bool(
        aggregate_settings.get(
            "use_agg_in_coag",
            aggregate_settings.get("use_in_coag", True),
        )
    )

    cap_factor = aggregate_settings.get(
        "collision_radius_cap_factor",
        aggregate_settings.get("agg_collision_radius_cap_factor", None),
    )

    r_monomer_cm = r_monomer_nm * 1e-7
    m_monomer = (4.0 / 3.0) * np.pi * r_monomer_cm**3 * rho_haze

    m_bin = m_monomer * bin_mass_ratio ** np.arange(n_bin, dtype=float)

    r_compact_cm = radius_from_mass_density(m_bin, rho_haze)
    r_compact_nm = r_compact_cm * 1e7

    r_primary_cm = r_primary_nm * 1e-7
    m_primary = (4.0 / 3.0) * np.pi * r_primary_cm**3 * rho_haze

    N_primary_raw = m_bin / max(m_primary, 1e-300)
    can_be_aggregate = N_primary_raw >= 1.0
    N_primary = np.maximum(N_primary_raw, 1.0)

    Pbar_layer = np.asarray(Pbar, dtype=float)

    if Pbar_layer.size != nz:
        raise ValueError("Pbar passed to build_particle_geometry must have length nz.")

    if use_aggregates:
        Df_layer = df_layer_from_pressure(Pbar_layer, aggregate_settings)
        w_agg_layer = aggregate_weight_from_pressure(Pbar_layer, aggregate_settings)
    else:
        Df_layer = np.full(nz, df_fixed)
        w_agg_layer = np.zeros(nz)

    r_fractal_layer_cm = np.zeros((nz, n_bin))
    r_collision_layer_cm = np.zeros((nz, n_bin))
    r_drag_layer_cm = np.zeros((nz, n_bin))
    rho_eff_layer = np.zeros((nz, n_bin))

    for k in range(nz):
        Dfk = float(Df_layer[k])

        for i in range(n_bin):
            if use_aggregates and can_be_aggregate[i]:
                r_f = r_primary_cm * (N_primary[i] / max(k0_agg, 1e-300)) ** (
                    1.0 / max(Dfk, 1e-6)
                )

                r_f = max(r_f, r_compact_cm[i])

                rho_i = m_bin[i] / (
                    (4.0 / 3.0) * np.pi * max(r_f, 1e-300) ** 3
                )

            else:
                r_f = r_compact_cm[i]
                rho_i = rho_haze

            r_fractal_layer_cm[k, i] = r_f
            rho_eff_layer[k, i] = max(rho_i, 1e-300)

            if use_aggregates and use_agg_in_coag and can_be_aggregate[i]:
                r_collision_layer_cm[k, i] = r_f
            else:
                r_collision_layer_cm[k, i] = r_compact_cm[i]

            if use_aggregates and use_agg_in_settling and can_be_aggregate[i]:
                r_drag_layer_cm[k, i] = r_f
            else:
                r_drag_layer_cm[k, i] = r_compact_cm[i]

    if cap_factor is not None:
        cap = float(cap_factor) * r_compact_cm[None, :]
        r_collision_layer_cm = np.minimum(r_collision_layer_cm, cap)

    if Pbar is not None:
        rep_idx = int(np.argmax(Pbar_layer))
    else:
        rep_idx = 0

    return {
        "m_bin": m_bin,
        "m_monomer": m_monomer,
        "m_primary": m_primary,
        "N_primary": N_primary,
        "N_primary_raw": N_primary_raw,
        "can_be_aggregate": can_be_aggregate,

        "r_compact_cm": r_compact_cm,
        "r_compact_nm": r_compact_nm,

        "r_primary_cm": r_primary_cm,

        "r_fractal_cm": r_fractal_layer_cm[rep_idx, :].copy(),
        "r_fractal_nm": r_fractal_layer_cm[rep_idx, :].copy() * 1e7,

        "r_mass_cm": r_compact_cm.copy(),
        "r_collision_cm": r_collision_layer_cm[rep_idx, :].copy(),
        "r_drag_cm": r_drag_layer_cm[rep_idx, :].copy(),
        "rho_eff": rho_eff_layer[rep_idx, :].copy(),

        "r_fractal_layer_cm": r_fractal_layer_cm,
        "r_collision_layer_cm": r_collision_layer_cm,
        "r_drag_layer_cm": r_drag_layer_cm,
        "rho_eff_layer": rho_eff_layer,

        "Df_layer": Df_layer,
        "w_agg_layer": w_agg_layer,

        "n_bin": n_bin,
        "rho_haze": rho_haze,
    }


def _case_from_ctx(ctx):
    raw_ctx = ctx.get("raw_ctx", {})
    return raw_ctx.get("case", None)


def _settings_from_ctx(ctx, attr_name):
    case = _case_from_ctx(ctx)

    if case is not None:
        return getattr(case, attr_name, {})

    raw_ctx = ctx.get("raw_ctx", {})
    return raw_ctx.get(attr_name, {})




def _transport_settings_from_ctx(ctx):
    return _settings_from_ctx(ctx, "transport_settings")


def _sink_settings_from_ctx(ctx):
    return _settings_from_ctx(ctx, "sink_settings")


def _solver_settings_from_ctx(ctx):
    return _settings_from_ctx(ctx, "solver_settings")


def _coag_settings_from_ctx(ctx):
    return _settings_from_ctx(ctx, "coagulation_settings")


#----------------------------------------------------------------
# Source distribution across particle bins
# ---------------------------------------------------------------------


def source_bin_weights_from_radii(r_compact_nm, source_settings):
    """
    Convert source-bin settings into normalized bin weights.

    Modes:
    - single_bin
    - lognormal
    - custom_weights
    """
    r_compact_nm = np.asarray(r_compact_nm, dtype=float)
    n_bin = r_compact_nm.size

    mode = source_settings.get("bin_mode", source_settings.get("source_bin_mode", "single_bin"))

    if mode == "single_bin":
        idx = int(source_settings.get("bin_index", source_settings.get("source_bin_index", 0)))
        idx = int(np.clip(idx, 0, n_bin - 1))

        weights = np.zeros(n_bin)
        weights[idx] = 1.0
        return weights

    if mode == "lognormal":
        r0 = float(source_settings.get("r0_nm", source_settings.get("source_r0_nm", 2.0)))
        sigma_dex = float(
            source_settings.get("sigma_dex", source_settings.get("source_sigma_dex", 0.20))
        )

        log_r = safe_log10(r_compact_nm)
        shape = np.exp(-0.5 * ((log_r - np.log10(max(r0, 1e-300))) / max(sigma_dex, 1e-6)) ** 2)

        if np.sum(shape) <= 0:
            raise RuntimeError("Lognormal source-bin weights sum to zero.")

        return shape / np.sum(shape)

    if mode == "custom_weights":
        custom = source_settings.get("bin_weights", source_settings.get("source_bin_weights", None))

        if custom is None:
            raise ValueError("custom_weights mode requires bin_weights.")

        weights = np.asarray(custom, dtype=float)

        if weights.size != n_bin:
            raise ValueError("Custom source bin weights must have length n_bin.")

        weights = np.maximum(weights, 0.0)

        if np.sum(weights) <= 0:
            raise ValueError("Custom source bin weights sum to zero.")

        return weights / np.sum(weights)

    raise ValueError(
        "source bin mode must be 'single_bin', 'lognormal', or 'custom_weights'."
    )


# ---------------------------------------------------------------------
# Thermal sink scaffold
# ---------------------------------------------------------------------


def smooth_step_T(T, T_on, width):
    """
    Smooth activation from 0 to 1 around T_on.
    """
    T = np.asarray(T, dtype=float)
    return 0.5 * (1.0 + np.tanh((T - T_on) / max(width, 1e-6)))



def thermal_sink_rate_profile(T, Pbar, sink_settings):
    """
    Parameterized thermal-destruction sink profile.

    This is a smooth first-order loss prescription used to represent haze
    removal in hot deep atmospheric layers. It is not a material-specific
    physical ablation model.

    The adopted rate is

        k_therm(T) =
            [1 / tau_therm]
            * 0.5
            * [1 + tanh((T - T_on) / width)].
    """
    model = sink_settings.get("thermal_model", "finite_rate_smooth_T")

    if model in (None, "none", "off", False):
        return (
            np.zeros_like(T, dtype=float),
            "parameterized thermal destruction off",
        )

    T_on = float(
        sink_settings.get("therm_destroy_t_on_k", 650.0)
    )

    width = float(
        sink_settings.get("therm_destroy_width_k", 100.0)
    )

    tau = float(
        sink_settings.get("therm_destroy_timescale", 1e5)
    )

    shape = smooth_step_T(T, T_on, width)
    rate = shape / max(tau, 1e-300)

    note = (
        "parameterized thermal destruction active: "
        f"T_on={T_on:.1f} K, "
        f"width={width:.1f} K, "
        f"tau={tau:.3e} s"
    )

    return rate, note
# ---------------------------------------------------------------------
# Settling velocities
# ---------------------------------------------------------------------


def compute_settling_velocity_matrix(ctx, case):
    """
    Compute layer/bin settling velocities with VIRGA's vfall.

    Returns
    -------
    np.ndarray
        v_set[k, i] in cm/s, positive downward.
    """
    transport_settings = case.transport_settings
    settling_on = bool(transport_settings.get("settling", True))
    settling_scale = float(transport_settings.get("settling_scale", 1.0))

    nz = ctx["nz"]
    n_bin = ctx["n_bin"]

    v_set = np.zeros((nz, n_bin), dtype=float)

    if not settling_on:
        print("Settling disabled; v_set is zero.", flush=True)
        return v_set

    if vfall is None:
        raise ImportError(
            "Could not import virga.root_functions.vfall. "
            f"Original import error: {_VFALL_IMPORT_ERROR}"
        )

    for k in range(nz):
        for i in range(n_bin):
            r_i = float(ctx["r_drag_layer_cm"][k, i])
            rho_i = float(ctx["rho_eff_layer"][k, i])
            df_i = float(ctx["Df_layer"][k])

            try:
                vraw = vfall(
                    r_i,
                    float(ctx["g_planet"]),
                    float(ctx["mu"][k]),
                    float(ctx["mfp"][k]),
                    float(ctx["visc"][k]),
                    float(ctx["T"][k]),
                    float(ctx["Pdyn"][k]),
                    rho_i,
                    False,
                    df_i,
                    1.0,
                    r_i,
                    1.0,
                )

                v_set[k, i] = abs(vraw) * settling_scale if np.isfinite(vraw) else 0.0

            except Exception as exc:
                raise RuntimeError(
                    f"VIRGA vfall failed at layer k={k}, bin i={i}, "
                    f"P={ctx['Pbar'][k]:.3e} bar, r={r_i:.3e} cm."
                ) from exc

    return v_set


def settling_diagnostics(ctx):
    """
    Basic settling diagnostics.
    """
    v_set = ctx["v_set"]
    H = np.maximum(ctx["H"], 1e-300)

    v_max = float(np.nanmax(v_set))
    v_min_positive = float(np.nanmin(v_set[v_set > 0])) if np.any(v_set > 0) else 0.0

    t_set_bin = H[:, None] / np.maximum(v_set, 1e-300)
    t_set_min = float(np.nanmin(t_set_bin)) if np.any(np.isfinite(t_set_bin)) else np.inf
    t_set_max_finite = float(np.nanmax(t_set_bin[np.isfinite(t_set_bin)]))

    t_mix = H**2 / np.maximum(ctx["Kzz"], 1e-300)

    return {
        "v_max": v_max,
        "v_min_positive": v_min_positive,
        "t_set_bin": t_set_bin,
        "t_set_min": t_set_min,
        "t_set_max_finite": t_set_max_finite,
        "t_mix": t_mix,
        "t_mix_min": float(np.nanmin(t_mix)),
        "t_mix_max": float(np.nanmax(t_mix)),
    }


# ---------------------------------------------------------------------
# Microphysics context builder
# ---------------------------------------------------------------------


def build_microphysics_context(raw_ctx, case):
    """
    Build the real Caligo microphysics context.
    """
    print("\n" + "=" * 70, flush=True)
    print("CALIGO: MICROPHYSICS CONTEXT", flush=True)
    print("=" * 70, flush=True)

    t0 = walltime.time()

    source_info = raw_ctx.get("source_info", None)

    if source_info is None:
        source_info = getattr(case, "_source_info", None)

    if source_info is None:
        raise RuntimeError(
            "No source_info found. Run source_info = case.build_haze_source() "
            "before case.haze(...)."
        )

    photochem_result = getattr(case, "_photochem_result", None)

    if photochem_result is None:
        raise RuntimeError(
            "No photochem result found. Run case.run_photochem() before "
            "building the microphysics context."
        )

    Pbar_pc = np.asarray(photochem_result["Pbar_pc"], dtype=float)
    z_pc = np.asarray(photochem_result["z_pc"], dtype=float)
    T_pc = np.asarray(photochem_result["T_pc"], dtype=float)

    Kzz_pc = np.asarray(raw_ctx["Kzz_raw"], dtype=float)

    if Kzz_pc.size != Pbar_pc.size:
        Kzz_pc = np.ones_like(Pbar_pc) * np.nanmedian(Kzz_pc)

    grid_settings = case.microphysics_grid_settings
    p_top_micro = float(grid_settings.get("p_top_micro", 1e-10))
    p_bot_micro = float(grid_settings.get("p_bot_micro", 10.0))
    nz_target = int(grid_settings.get("nz_target", 50))
    kzz_micro_scale = float(grid_settings.get("kzz_micro_scale", 1.0))

    mask_micro = (Pbar_pc >= p_top_micro) & (Pbar_pc <= p_bot_micro)

    P_c = Pbar_pc[mask_micro]
    z_c = z_pc[mask_micro]
    T_c = T_pc[mask_micro]
    Kzz_c = Kzz_pc[mask_micro]

    if len(P_c) < 4:
        raise RuntimeError(
            "Microphysics pressure domain selected too few photochem grid cells."
        )

    order_z = np.argsort(z_c)
    P_c = P_c[order_z]
    z_c = z_c[order_z]
    T_c = T_c[order_z]
    Kzz_c = Kzz_c[order_z]

    good = np.concatenate(([True], np.diff(z_c) > 0))
    P_c = P_c[good]
    z_c = z_c[good]
    T_c = T_c[good]
    Kzz_c = Kzz_c[good]

    mu_c = compute_mu_profile_from_photochem(case, P_c)

    sp_ord = np.argsort(safe_log10(P_c))

    logP_fine = np.linspace(
        safe_log10(P_c[sp_ord])[0],
        safe_log10(P_c[sp_ord])[-1],
        nz_target,
    )

    Pbar = 10.0 ** logP_fine
    z = np.interp(logP_fine, safe_log10(P_c[sp_ord]), z_c[sp_ord])
    T = np.interp(logP_fine, safe_log10(P_c[sp_ord]), T_c[sp_ord])
    Kzz = np.interp(logP_fine, safe_log10(P_c[sp_ord]), Kzz_c[sp_ord])
    mu = np.interp(logP_fine, safe_log10(P_c[sp_ord]), mu_c[sp_ord])

    order_z_micro = np.argsort(z)
    Pbar = Pbar[order_z_micro]
    z = z[order_z_micro]
    T = T[order_z_micro]
    Kzz = Kzz[order_z_micro] * kzz_micro_scale
    mu = mu[order_z_micro]

    if not np.all(np.diff(z) > 0):
        raise RuntimeError("Microphysics z grid must increase upward.")

    if not np.all(np.diff(Pbar) < 0):
        raise RuntimeError("Microphysics Pbar grid must decrease upward.")

    Pdyn = Pbar * 1e6

    zf = build_faces(z)
    dz = np.diff(zf)
    dz_c = z[1:] - z[:-1]

    mfp = mean_free_path(Pdyn, T)
    visc = gas_viscosity(T)

    g_planet = G_cgs * case._planet_mass_cgs / max(case._planet_radius_cgs, 1e-300) ** 2

    H = k_B * T / (mu * m_H * max(g_planet, 1e-300))
    N_gas = Pdyn / (k_B * np.maximum(T, 1.0))
    rho_gas = mu * m_H * N_gas

    nz = len(Pbar)

    print(f"Microphysics grid nz = {nz}", flush=True)
    print(f"Microphysics P range = [{Pbar[0]:.3e}, {Pbar[-1]:.3e}] bar", flush=True)
    print(f"Microphysics T range = [{T.min():.1f}, {T.max():.1f}] K", flush=True)
    print(f"Bottom P/T = {Pbar[0]:.3e} bar / {T[0]:.1f} K", flush=True)
    print(f"Top P/T    = {Pbar[-1]:.3e} bar / {T[-1]:.1f} K", flush=True)

    geom = build_particle_geometry(
        nz,
        Pbar=Pbar,
        particle_settings=case.particle_settings,
        aggregate_settings=case.aggregate_settings,
    )

    m_bin = geom["m_bin"]
    r_compact_nm = geom["r_compact_nm"]
    n_bin = geom["n_bin"]

    z_src_sorted = np.asarray(source_info["z_src_sorted"], dtype=float)
    q_raw = np.asarray(source_info["q_raw"], dtype=float)

    good_src = np.concatenate(([True], np.diff(z_src_sorted) > 0))
    z_src_good = z_src_sorted[good_src]
    q_raw_good = q_raw[good_src]

    zf_src_good = build_faces(z_src_good)

    q_micro = np.maximum(conservative_remap(zf_src_good, q_raw_good, zf), 0.0)
    F_micro_raw = float(np.sum(q_micro * dz))

    F_inside_raw = F_micro_raw
    F_above_raw = integrate_source_over_interval(
        zf_src_good,
        q_raw_good,
        zf[-1],
        zf_src_good[-1],
    )
    F_below_raw = integrate_source_over_interval(
        zf_src_good,
        q_raw_good,
        zf_src_good[0],
        zf[0],
    )

    source_settings = case.haze_source_settings

    vertical_mode = source_settings.get("vertical_mode", "photochem")
    coupling_mode = source_settings.get("coupling_mode", "split_top_flux")
    yield_model = source_settings.get("yield_model", "carbon_mass_only")

    use_fixed_flux = bool(source_settings.get("use_fixed_flux", True))
    target_haze_flux = float(source_settings.get("target_haze_flux", 1e-13))
    source_flux_scale = float(source_settings.get("source_flux_scale", 1.0))

    F_total_raw = max(float(source_info["F_photochem_raw"]), 1e-300)

    if use_fixed_flux or yield_model == "fixed_literature_flux":
        source_scale = target_haze_flux / F_total_raw
    else:
        source_scale = source_flux_scale

    if vertical_mode == "gaussian":
        gauss_peak_bar = float(source_settings.get("gauss_peak_bar", 1e-7))
        gauss_sigma_dex = float(source_settings.get("gauss_sigma_dex", 0.50))

        logP = safe_log10(Pbar)
        shape = np.exp(
            -0.5
            * ((logP - np.log10(max(gauss_peak_bar, 1e-300))) / max(gauss_sigma_dex, 1e-6))
            ** 2
        )

        norm = float(np.sum(shape * dz))

        if norm <= 0:
            raise RuntimeError("Gaussian source normalization is zero.")

        target_flux = (
            target_haze_flux
            if use_fixed_flux
            else source_flux_scale * source_info["F_photochem_raw"]
        )

        S_mass_density = shape * target_flux / norm
        F_top_source = 0.0
        F_inside_source = float(np.sum(S_mass_density * dz))
        F_above_source = 0.0
        F_below_source = 0.0

    elif vertical_mode == "photochem":
        if coupling_mode == "split_top_flux":
            S_mass_density = q_micro.copy() * source_scale
            F_inside_source = F_inside_raw * source_scale
            F_above_source = F_above_raw * source_scale
            F_below_source = F_below_raw * source_scale
            F_top_source = F_above_source

        elif coupling_mode == "in_domain_only":
            S_mass_density = q_micro.copy() * source_scale
            F_inside_source = F_inside_raw * source_scale
            F_above_source = F_above_raw * source_scale
            F_below_source = F_below_raw * source_scale
            F_top_source = 0.0

        elif coupling_mode == "renormalize_in_domain":
            raw_flux = float(np.sum(q_micro * dz))

            if raw_flux <= 0:
                raise RuntimeError(
                    "Photochem source remapped to zero flux on microphysics grid."
                )

            target_flux = (
                target_haze_flux
                if use_fixed_flux
                else source_flux_scale * source_info["F_photochem_raw"]
            )

            S_mass_density = q_micro.copy() * target_flux / raw_flux
            F_inside_source = float(np.sum(S_mass_density * dz))
            F_above_source = F_above_raw * source_scale
            F_below_source = F_below_raw * source_scale
            F_top_source = 0.0

        else:
            raise ValueError(
                "Bad source coupling mode. Use split_top_flux, "
                "in_domain_only, or renormalize_in_domain."
            )

    else:
        raise ValueError("source vertical_mode must be photochem or gaussian.")

    weights = source_bin_weights_from_radii(r_compact_nm, source_settings)

    S_number = np.zeros((nz, n_bin))

    for i in range(n_bin):
        if weights[i] > 0:
            S_number[:, i] = S_mass_density * weights[i] / np.maximum(m_bin[i], 1e-300)

    top_flux_number = np.zeros(n_bin)

    for i in range(n_bin):
        if weights[i] > 0:
            top_flux_number[i] = F_top_source * weights[i] / np.maximum(m_bin[i], 1e-300)

    F_vol_source = float(np.sum(S_number * m_bin[None, :] * dz[:, None]))
    F_src = F_vol_source + F_top_source

    source_peak_index = int(np.argmax(S_mass_density)) if np.any(S_mass_density > 0) else nz - 1

    print(f"Raw total photochem source = {source_info['F_photochem_raw']:.3e} g cm^-2 s^-1", flush=True)
    print(f"Raw in-domain source       = {F_inside_raw:.3e} g cm^-2 s^-1", flush=True)
    print(f"Raw above-domain source    = {F_above_raw:.3e} g cm^-2 s^-1", flush=True)
    print(f"Raw below-domain source    = {F_below_raw:.3e} g cm^-2 s^-1", flush=True)
    print(f"Source scale factor        = {source_scale:.3e}", flush=True)
    print(f"Final volumetric source    = {F_vol_source:.3e} g cm^-2 s^-1", flush=True)
    print(f"Final top-boundary source  = {F_top_source:.3e} g cm^-2 s^-1", flush=True)
    print(f"Final total source         = {F_src:.3e} g cm^-2 s^-1", flush=True)
    print(f"Source peak pressure       = {Pbar[source_peak_index]:.3e} bar", flush=True)

    thermal_sink_rate, recycling_note = thermal_sink_rate_profile(
        T,
        Pbar,
        case.sink_settings,
    )

    print(recycling_note, flush=True)
    print(f"Max thermal destruction rate = {np.max(thermal_sink_rate):.3e} s^-1", flush=True)

    ctx = {
        "Pbar": Pbar,
        "Pdyn": Pdyn,
        "z": z,
        "T": T,
        "mu": mu,
        "Kzz": Kzz,
        "zf": zf,
        "dz": dz,
        "dz_c": dz_c,
        "mfp": mfp,
        "visc": visc,
        "H": H,
        "N_gas": N_gas,
        "rho_gas": rho_gas,
        "g_planet": g_planet,
        "nz": nz,
        "n_bin": n_bin,
        "S_mass_density": S_mass_density,
        "S_number": S_number,
        "top_flux_number": top_flux_number,
        "F_src": F_src,
        "F_vol_source": F_vol_source,
        "F_top_source": F_top_source,
        "F_inside_source": F_inside_source,
        "F_above_source": F_above_source,
        "F_below_source": F_below_source,
        "q_micro": q_micro,
        "F_micro_raw": F_micro_raw,
        "source_scale": source_scale,
        "source_weights": weights,
        "thermal_sink_rate": thermal_sink_rate,
        "recycling_note": recycling_note,
        "source_info": source_info,
        "raw_ctx": raw_ctx,
    }

    ctx.update(geom)

    v_set = compute_settling_velocity_matrix(ctx, case)
    ctx["v_set"] = v_set

    sett = settling_diagnostics(ctx)
    ctx.update(sett)

    print(f"Max settling speed = {sett['v_max']:.3e} cm/s", flush=True)
    print(f"Min positive settling speed = {sett['v_min_positive']:.3e} cm/s", flush=True)
    print(f"Mixing time range = [{sett['t_mix_min']:.3e}, {sett['t_mix_max']:.3e}] s", flush=True)
    print(f"Minimum settling time = {sett['t_set_min']:.3e} s", flush=True)

    print(f"Microphysics context finished in {walltime.time() - t0:.2f} s.", flush=True)

    return ctx


def run_haze_model(raw_ctx, case):
    """
    Main haze model entry point.

    Current behavior:
        build and return the microphysics context.

    Later behavior:
        run the time-dependent haze microphysics solver.
    """
    ctx = build_microphysics_context(raw_ctx, case)

    haze_out = {
        "Pbar": ctx["Pbar"],
        "T": ctx["T"],
        "Kzz": ctx["Kzz"],
        "z": ctx["z"],
        "S_mass_density": ctx["S_mass_density"],
        "S_number": ctx["S_number"],
        "m_bin": ctx["m_bin"],
        "r_compact_nm": ctx["r_compact_nm"],
        "v_set": ctx["v_set"],
        "F_src": ctx["F_src"],
        "F_vol_source": ctx["F_vol_source"],
        "F_top_source": ctx["F_top_source"],

        # Real solver outputs will replace these later.
        "n": None,
        "q_recycled": None,
        "time": None,
        "converged": False,
        "history": [],

        "ctx": ctx,
        "message": (
            "Caligo microphysics context built successfully, including settling velocities. "
            "The time-dependent haze solver has not been moved in yet."
        ),
    }

    return haze_out




# ---------------------------------------------------------------------
# Single-step transport machinery
# ---------------------------------------------------------------------


def initialize_haze_distribution(ctx, fill_value=0.0):
    """
    Initialize particle number density n[z, bin].

    Parameters
    ----------
    ctx : dict
        Caligo microphysics context.
    fill_value : float
        Initial number density in cm^-3.

    Returns
    -------
    n : ndarray
        Number density array with shape (nz, n_bin).
    q_recycled : float
        Placeholder recycled gas column [g cm^-2].
    """
    n = np.ones((ctx["nz"], ctx["n_bin"]), dtype=float) * float(fill_value)
    q_recycled = 0.0
    return n, q_recycled


def column_haze_mass(n, ctx):
    """
    Total haze column mass [g cm^-2].
    """
    n = np.asarray(n, dtype=float)
    return float(np.sum(n * ctx["m_bin"][None, :] * ctx["dz"][:, None]))


def layer_haze_mass(n, ctx):
    """
    Haze mass per layer [g cm^-2 per layer].
    """
    n = np.asarray(n, dtype=float)
    return np.sum(n * ctx["m_bin"][None, :] * ctx["dz"][:, None], axis=1)




def _implicit_diffuse_density_1d(q_density, dt, ctx):
    """
    Implicit vertical diffusion of a particle density field using gas density
    as the carrier.

    This is the old-code operator:
        f = q_density / rho_gas
        diffuse f with Kzz * rho_gas
        q_new = f_new * rho_gas
    """
    from scipy.linalg import solve_banded

    q_density = np.maximum(np.asarray(q_density, dtype=float), 0.0)

    nz = int(ctx["nz"])

    if nz < 2 or dt <= 0:
        return q_density.copy()

    carrier = np.maximum(np.asarray(ctx["rho_gas"], dtype=float), 1e-300)
    f_old = q_density / carrier

    Kzz = np.asarray(ctx["Kzz"], dtype=float)
    dz = np.asarray(ctx["dz"], dtype=float)
    dz_c = np.asarray(ctx["dz_c"], dtype=float)

    carrier_face = 0.5 * (carrier[:-1] + carrier[1:])
    K_face = 0.5 * (Kzz[:-1] + Kzz[1:])
    G_face = K_face * carrier_face

    M = carrier * dz

    lo = np.zeros(nz)
    di = M.copy()
    up = np.zeros(nz)
    rhs = M * f_old

    for k in range(1, nz - 1):
        C_lower = dt * G_face[k - 1] / max(dz_c[k - 1], 1e-300)
        C_upper = dt * G_face[k] / max(dz_c[k], 1e-300)

        lo[k] = -C_lower
        di[k] = M[k] + C_lower + C_upper
        up[k] = -C_upper

    C_upper = dt * G_face[0] / max(dz_c[0], 1e-300)
    di[0] = M[0] + C_upper
    up[0] = -C_upper

    C_lower = dt * G_face[-1] / max(dz_c[-1], 1e-300)
    lo[-1] = -C_lower
    di[-1] = M[-1] + C_lower

    ab = np.zeros((3, nz))
    ab[0, 1:] = up[:-1]
    ab[1, :] = di
    ab[2, :-1] = lo[1:]

    f_new = solve_banded((1, 1), ab, rhs)

    return np.maximum(f_new * carrier, 0.0)


def implicit_diffuse_density(n, dt, ctx):
    """
    Diffuse every particle bin using the old gas-carrier diffusion operator.
    """
    n = np.maximum(np.asarray(n, dtype=float), 0.0)

    if dt <= 0:
        return n.copy()

    if n.ndim == 1:
        return _implicit_diffuse_density_1d(n, dt, ctx)

    n_new = np.zeros_like(n)

    for i in range(n.shape[1]):
        n_new[:, i] = _implicit_diffuse_density_1d(n[:, i], dt, ctx)

    return np.maximum(n_new, 0.0)




def settle_density_upwind(n, dt, ctx, cfl_limit=0.5, max_substeps=200000):
    """
    Upwind settling update for number density.

    Direct translation of the old transport logic, generalized to all bins:
      - z increases upward
      - settling velocity is positive downward
      - CFL is computed only over active particle cells
      - bottom boundary can be zero_flux, open_flux, or deposition_velocity
    """
    n = np.maximum(np.asarray(n, dtype=float), 0.0).copy()
    dt = float(dt)

    transport_settings = _transport_settings_from_ctx(ctx)

    bottom_bc = transport_settings.get(
        "bottom_bc",
        transport_settings.get("bottom_bc_particles", "open_flux"),
    )

    cfl_limit = float(
        transport_settings.get(
            "sedimentation_cfl",
            transport_settings.get("sed_cfl", cfl_limit),
        )
    )

    max_substeps = int(
        transport_settings.get(
            "max_sedimentation_substeps",
            transport_settings.get("max_sed_substeps", max_substeps),
        )
    )

    active_frac = float(
        transport_settings.get(
            "transport_active_frac",
            transport_settings.get("active_frac", 1e-24),
        )
    )

    active_floor = float(transport_settings.get("transport_active_floor", 1e-300))

    diag_empty = {
        "sed_substeps": 0,
        "sed_cfl_max": 0.0,
        "hit_max_substeps": False,
        "bottom_bc": bottom_bc,
        "bottom_v_eff_max": 0.0,
    }

    if dt <= 0 or not np.any(n > 0):
        return n, 0.0, diag_empty

    v_set = np.maximum(np.asarray(ctx["v_set"], dtype=float), 0.0)
    dz = np.asarray(ctx["dz"], dtype=float)
    m_bin = np.asarray(ctx["m_bin"], dtype=float)

    nz, n_bin = n.shape

    if bottom_bc == "zero_flux":
        v_bottom = np.zeros(n_bin, dtype=float)

    elif bottom_bc == "deposition_velocity":
        v_dep = float(ctx["Kzz"][0] / max(ctx["H"][0], 1e-300))
        v_bottom = np.maximum(v_set[0, :], v_dep)

    elif bottom_bc == "open_flux":
        v_bottom = v_set[0, :].copy()

    else:
        v_bottom = v_set[0, :].copy()

    v_cfl = v_set.copy()
    v_cfl[0, :] = v_bottom

    # Old behavior: do not let empty high-speed cells control the timestep.
    bin_max = np.max(n, axis=0, keepdims=True)
    active = n > np.maximum(bin_max * active_frac, active_floor)

    if not np.any(active):
        diag_empty["bottom_v_eff_max"] = float(np.max(v_bottom))
        return n, 0.0, diag_empty

    sed_cfl = v_cfl * dt / np.maximum(dz[:, None], 1e-300)
    sed_cfl_max = float(np.nanmax(sed_cfl[active]))

    if sed_cfl_max <= 0:
        diag_empty["bottom_v_eff_max"] = float(np.max(v_bottom))
        return n, 0.0, diag_empty

    n_sub_requested = int(np.ceil(sed_cfl_max / max(cfl_limit, 1e-12)))
    n_sub = max(1, n_sub_requested)
    hit_max = False

    if n_sub > max_substeps:
        n_sub = int(max_substeps)
        hit_max = True

    dt_sub = dt / max(n_sub, 1)
    bottom_loss_mass = 0.0

    for _ in range(n_sub):
        U = n * dz[:, None]

        Fdown = np.zeros((nz + 1, n_bin), dtype=float)

        # Bottom flux out of the domain.
        Fdown[0, :] = v_bottom * n[0, :]

        # Interior downward flux from cell k into k-1.
        for k in range(1, nz):
            v_face = 0.5 * (v_set[k, :] + v_set[k - 1, :])
            Fdown[k, :] = v_face * n[k, :]

        # No incoming top flux from settling.
        Fdown[nz, :] = 0.0

        U_new = U.copy()

        for k in range(nz):
            U_new[k, :] += dt_sub * (Fdown[k + 1, :] - Fdown[k, :])

        bottom_loss_mass += float(np.sum(Fdown[0, :] * dt_sub * m_bin))

        n = np.maximum(U_new / dz[:, None], 0.0)

    diagnostics = {
        "sed_substeps": int(n_sub),
        "sed_cfl_max": sed_cfl_max,
        "hit_max_substeps": hit_max,
        "bottom_bc": bottom_bc,
        "bottom_v_eff_max": float(np.max(v_bottom)),
        "sed_substeps_requested": int(n_sub_requested),
    }

    return n, bottom_loss_mass, diagnostics



def take_transport_step(
    n,
    q_recycled,
    dt,
    ctx,
    include_source=True,
    include_diffusion=True,
    include_settling=True,
    include_thermal=True,
):
    """
    Take one source + diffusion + settling + thermal-destruction step.

    This is the transport-only diagnostic stepping function. It intentionally
    excludes coagulation and deep recycling.

    Returns
    -------
    n : ndarray
        Updated particle number-density distribution.
    q_recycled : float or ndarray
        Updated passive recycled-gas tracer.
    diagnostics : dict
        Step-level mass-budget diagnostics.
    """
    n = np.asarray(n, dtype=float).copy()
    dt = float(dt)

    if dt <= 0:
        raise ValueError("dt must be positive.")

    mass_initial = column_haze_mass(n, ctx)
    recycled_initial = recycled_gas_column(q_recycled, ctx)

    source_mass = 0.0
    bottom_loss_mass = 0.0
    thermal_loss_mass = 0.0

    sed_diag = {
        "sed_substeps": 0,
        "sed_cfl_max": 0.0,
        "hit_max_substeps": False,
    }

    if include_source:
        n += ctx["S_number"] * dt

        if ctx.get("F_top_source", 0.0) > 0:
            n[-1, :] += (
                ctx["top_flux_number"]
                * dt
                / max(ctx["dz"][-1], 1e-300)
            )

        source_mass = ctx["F_src"] * dt

    if include_diffusion:
        n = implicit_diffuse_density(n, dt, ctx)

    if include_settling:
        n, bottom_loss_mass, sed_diag = settle_density_upwind(n, dt, ctx)

    if include_thermal:
        n, q_recycled, thermal_loss_mass = apply_thermal_destruction(
            n,
            dt,
            ctx,
            q_recycled=q_recycled,
        )

    mass_final = column_haze_mass(n, ctx)
    recycled_final = recycled_gas_column(q_recycled, ctx)

    expected_final = (
        mass_initial
        + source_mass
        - bottom_loss_mass
        - thermal_loss_mass
    )

    mass_error = mass_final - expected_final

    diagnostics = {
        "dt": dt,
        "mass_initial": mass_initial,
        "source_mass": source_mass,
        "bottom_loss_mass": bottom_loss_mass,
        "thermal_loss_mass": thermal_loss_mass,
        "mass_final": mass_final,
        "expected_final": expected_final,
        "mass_error": mass_error,
        "relative_mass_error": (
            mass_error / max(abs(expected_final), 1e-300)
        ),
        "recycled_gas_initial": recycled_initial,
        "recycled_gas_final": recycled_final,
        "recycled_gas_gain": recycled_final - recycled_initial,
        **sed_diag,
    }

    return np.maximum(n, 0.0), q_recycled, diagnostics

# ---------------------------------------------------------------------
# Chunked integration loop: source + diffusion + settling + thermal sink
# ---------------------------------------------------------------------


def diagnostic_peak_pressure(n, ctx):
    """
    Pressure where the haze layer mass is largest.

    Old-code behavior: if the column is empty, return a middle pressure
    instead of nan.
    """
    M = layer_haze_mass(n, ctx)

    if not np.any(M > 0):
        return float(ctx["Pbar"][ctx["nz"] // 2])

    idx = int(np.argmax(M))
    return float(ctx["Pbar"][idx])


def profile_change(n_old, n_new):
    """
    Backward-compatible wrapper. If ctx is unavailable, use number-density norm.

    run_full_microphysics_to_steady_state below uses profile_change_mass(...)
    instead.
    """
    n_old = np.asarray(n_old, dtype=float)
    n_new = np.asarray(n_new, dtype=float)

    denom = max(np.sum(np.abs(n_old)), np.sum(np.abs(n_new)), 1e-300)

    return float(np.sum(np.abs(n_new - n_old)) / denom)


def profile_change_mass(n_old, n_new, ctx):
    """
    Old-code profile change diagnostic using layer mass.
    """
    M0 = layer_haze_mass(n_old, ctx)
    M1 = layer_haze_mass(n_new, ctx)

    denom = max(np.sum(np.abs(M0)), np.sum(np.abs(M1)), 1e-300)

    return float(np.sum(np.abs(M1 - M0)) / denom)





def microphysics_step_with_retry(
    n,
    q_recycled,
    dt,
    ctx,
    max_retries=8,
    retry_shrink=0.5,
    max_relative_mass_error=1e-8,
    max_column_jump_frac=10.0,
):
    """
    Take one transport step, shrinking dt if the step becomes numerically bad.

    This mirrors the spirit of the original retry wrapper, but currently checks
    only transport/source/sink properties. Coagulation checks will be added when
    coagulation is migrated.
    """
    dt_try = float(dt)

    last_diag = None

    for retry in range(max_retries + 1):
        n_trial, q_trial, diag = take_transport_step(
            n.copy(),
            q_recycled,
            dt_try,
            ctx,
        )

        diag["retries"] = retry

        finite_ok = np.all(np.isfinite(n_trial))
        nonnegative_ok = np.all(n_trial >= 0)
        mass_error_ok = abs(diag.get("relative_mass_error", 0.0)) <= max_relative_mass_error

        M0 = max(diag.get("mass_initial", 0.0), 1e-300)
        M1 = diag.get("mass_final", 0.0)

        if diag.get("mass_initial", 0.0) <= 0:
            jump_ok = True
        else:
            jump_ok = (M1 / M0) <= (1.0 + max_column_jump_frac)

        substep_ok = not diag.get("hit_max_substeps", False)

        good = finite_ok and nonnegative_ok and mass_error_ok and jump_ok and substep_ok

        if good:
            return n_trial, q_trial, dt_try, diag

        last_diag = diag
        dt_try *= retry_shrink

    raise RuntimeError(
        "microphysics_step_with_retry failed after retries. "
        f"last_dt={dt_try:.3e}, "
        f"last_relative_mass_error={last_diag.get('relative_mass_error', np.nan):.3e}, "
        f"hit_max_substeps={last_diag.get('hit_max_substeps', None)}, "
        f"sed_substeps={last_diag.get('sed_substeps', None)}"
    )


def run_transport_to_steady_state(
    ctx,
    dt_init=None,
    dt_growth=None,
    dt_max=None,
    chunk_time=None,
    max_chunks=None,
    max_step_retries=None,
    retry_shrink=None,
    ss_tol_storage=None,
    ss_tol_balance=None,
    ss_tol_profile=None,
    ss_tol_peak_dex=None,
    ss_min_stable_chunks=None,
    verbose=True,
):
    """
    Run the current transport-only haze model.

    Included physics:
        source injection
        diffusion
        settling
        thermal destruction

    Not yet included:
        coagulation
        deep recycling
        gas recycling

    Returns
    -------
    n : ndarray
        Final particle number density [cm^-3], shape (nz, n_bin).
    q_recycled : float
        Placeholder recycled gas column.
    t : float
        Final integration time [s].
    converged : bool
        Whether chunk-level steady-state criteria were satisfied.
    hist : list[dict]
        Per-chunk history.
    """
    case = ctx.get("raw_ctx", {}).get("case", None)

    solver_settings = {}
    if case is not None:
        solver_settings = getattr(case, "solver_settings", {})

    dt_init = float(dt_init if dt_init is not None else solver_settings.get("dt_init", 1e2))
    dt_growth = float(dt_growth if dt_growth is not None else solver_settings.get("dt_growth", 1.03))
    dt_max = float(dt_max if dt_max is not None else solver_settings.get("dt_max", 300.0))
    chunk_time = float(chunk_time if chunk_time is not None else solver_settings.get("chunk_time", 1.0e5))
    max_chunks = int(max_chunks if max_chunks is not None else solver_settings.get("max_chunks", 200))

    max_step_retries = int(
        max_step_retries
        if max_step_retries is not None
        else solver_settings.get("max_step_retries", 8)
    )
    retry_shrink = float(
        retry_shrink
        if retry_shrink is not None
        else solver_settings.get("retry_shrink", 0.5)
    )

    ss_tol_storage = float(
        ss_tol_storage
        if ss_tol_storage is not None
        else solver_settings.get("ss_tol_storage", 0.08)
    )
    ss_tol_balance = float(
        ss_tol_balance
        if ss_tol_balance is not None
        else solver_settings.get("ss_tol_balance", 0.25)
    )
    ss_tol_profile = float(
        ss_tol_profile
        if ss_tol_profile is not None
        else solver_settings.get("ss_tol_profile", 0.04)
    )
    ss_tol_peak_dex = float(
        ss_tol_peak_dex
        if ss_tol_peak_dex is not None
        else solver_settings.get("ss_tol_peak_dex", 0.25)
    )
    ss_min_stable_chunks = int(
        ss_min_stable_chunks
        if ss_min_stable_chunks is not None
        else solver_settings.get("ss_min_stable_chunks", 3)
    )

    if verbose:
        print("\n" + "=" * 70, flush=True)
        print("CALIGO: TRANSPORT-ONLY INTEGRATION", flush=True)
        print("=" * 70, flush=True)
        print(f"dt_init = {dt_init:.3e} s", flush=True)
        print(f"dt_growth = {dt_growth:.3e}", flush=True)
        print(f"dt_max = {dt_max:.3e} s", flush=True)
        print(f"chunk_time = {chunk_time:.3e} s", flush=True)
        print(f"max_chunks = {max_chunks}", flush=True)

    n, q_recycled = initialize_haze_distribution(ctx)

    t = 0.0
    dt = dt_init
    stable_chunks = 0
    converged = False

    prev_n = n.copy()
    prev_peak = diagnostic_peak_pressure(n, ctx)

    hist = []

    for ichunk in range(1, max_chunks + 1):
        t_start = t
        M_before = column_haze_mass(n, ctx)

        chunk = {
            "chunk": ichunk,
            "t_start": t_start,
            "t_end": None,
            "dt_last": None,
            "steps": 0,
            "source": 0.0,
            "bottom_loss": 0.0,
            "thermal_loss": 0.0,
            "total_sink_loss": 0.0,
            "mass_error": 0.0,
            "max_rel_mass_error": 0.0,
            "retries": 0,
            "max_sed_substeps": 0,
            "max_sed_cfl": 0.0,
            "hit_max_substeps_count": 0,
        }

        if verbose:
            print(f"\n--- Chunk {ichunk}/{max_chunks} ---", flush=True)

        while t - t_start < chunk_time:
            dt_step = min(dt, chunk_time - (t - t_start))

            n, q_recycled, dt_used, diag = microphysics_step_with_retry(
                n,
                q_recycled,
                dt_step,
                ctx,
                max_retries=max_step_retries,
                retry_shrink=retry_shrink,
            )

            t += dt_used
            dt = min(dt_max, max(dt_init, dt_used * dt_growth))

            chunk["steps"] += 1
            chunk["source"] += diag.get("source_mass", 0.0)
            chunk["bottom_loss"] += diag.get("bottom_loss_mass", 0.0)
            chunk["thermal_loss"] += diag.get("thermal_loss_mass", 0.0)
            chunk["mass_error"] += diag.get("mass_error", 0.0)
            chunk["max_rel_mass_error"] = max(
                chunk["max_rel_mass_error"],
                abs(diag.get("relative_mass_error", 0.0)),
            )
            chunk["retries"] += diag.get("retries", 0)
            chunk["max_sed_substeps"] = max(
                chunk["max_sed_substeps"],
                diag.get("sed_substeps", 0),
            )
            chunk["max_sed_cfl"] = max(
                chunk["max_sed_cfl"],
                diag.get("sed_cfl_max", 0.0),
            )

            if diag.get("hit_max_substeps", False):
                chunk["hit_max_substeps_count"] += 1

        M_after = column_haze_mass(n, ctx)

        chunk["t_end"] = t
        chunk["dt_last"] = dt
        chunk["column_mass"] = M_after
        chunk["storage"] = M_after - M_before
        chunk["total_sink_loss"] = chunk["bottom_loss"] + chunk["thermal_loss"]

        elapsed = max(t - t_start, 1e-300)

        F_src_chunk = chunk["source"] / elapsed
        F_sink_chunk = chunk["total_sink_loss"] / elapsed
        F_storage_chunk = chunk["storage"] / elapsed

        chunk["F_source"] = F_src_chunk
        chunk["F_sink"] = F_sink_chunk
        chunk["F_bottom"] = chunk["bottom_loss"] / elapsed
        chunk["F_thermal"] = chunk["thermal_loss"] / elapsed
        chunk["F_storage"] = F_storage_chunk

        source_norm = max(abs(ctx["F_src"]), 1e-300)

        storage_ratio = abs(F_storage_chunk) / source_norm
        balance_ratio = abs(F_src_chunk - F_sink_chunk) / source_norm
        prof_change = profile_change(prev_n, n)

        peak = diagnostic_peak_pressure(n, ctx)

        if np.isfinite(prev_peak) and np.isfinite(peak) and prev_peak > 0 and peak > 0:
            peak_shift_dex = abs(np.log10(peak) - np.log10(prev_peak))
        else:
            peak_shift_dex = np.inf

        chunk["storage_ratio"] = storage_ratio
        chunk["balance_ratio"] = balance_ratio
        chunk["profile_change"] = prof_change
        chunk["peak_pressure"] = peak
        chunk["peak_shift_dex"] = peak_shift_dex

        stable_now = (
            storage_ratio < ss_tol_storage
            and balance_ratio < ss_tol_balance
            and prof_change < ss_tol_profile
            and peak_shift_dex < ss_tol_peak_dex
        )

        if stable_now:
            stable_chunks += 1
        else:
            stable_chunks = 0

        chunk["stable_now"] = stable_now
        chunk["stable_chunks"] = stable_chunks

        hist.append(chunk)

        if verbose:
            print(f"t = {t:.3e} s ({t / 86400.0:.3e} d)", flush=True)
            print(f"steps in chunk = {chunk['steps']}", flush=True)
            print(f"column mass = {M_after:.3e} g cm^-2", flush=True)
            print(f"F_source = {F_src_chunk:.3e} g cm^-2 s^-1", flush=True)
            print(f"F_sink = {F_sink_chunk:.3e} g cm^-2 s^-1", flush=True)
            print(f"F_storage = {F_storage_chunk:.3e} g cm^-2 s^-1", flush=True)
            print(f"storage_ratio = {storage_ratio:.3e}", flush=True)
            print(f"balance_ratio = {balance_ratio:.3e}", flush=True)
            print(f"profile_change = {prof_change:.3e}", flush=True)
            print(f"peak_pressure = {peak:.3e} bar", flush=True)
            print(f"peak_shift_dex = {peak_shift_dex:.3e}", flush=True)
            print(f"max_sed_substeps = {chunk['max_sed_substeps']}", flush=True)
            print(f"stable_chunks = {stable_chunks}", flush=True)

        if stable_chunks >= ss_min_stable_chunks:
            converged = True

            if verbose:
                print("\nTransport-only steady state reached.", flush=True)

            break

        prev_n = n.copy()
        prev_peak = peak

    if verbose:
        print("\n" + "=" * 70, flush=True)
        print("TRANSPORT-ONLY INTEGRATION COMPLETE", flush=True)
        print("=" * 70, flush=True)
        print(f"converged = {converged}", flush=True)
        print(f"t_final = {t:.3e} s = {t / 86400.0:.3e} days", flush=True)
        print(f"final column mass = {column_haze_mass(n, ctx):.3e} g cm^-2", flush=True)

    return n, q_recycled, t, converged, hist




# ---------------------------------------------------------------------
# Coagulation remap tables and kernel construction
# ---------------------------------------------------------------------


def cunningham_slip_correction(r_cm, mfp_cm):
    """
    Cunningham slip correction.

    Parameters
    ----------
    r_cm : array-like
        Particle radius in cm.
    mfp_cm : float
        Gas mean free path in cm.
    """
    r_cm = np.asarray(r_cm, dtype=float)
    Kn = np.maximum(mfp_cm / np.maximum(r_cm, 1e-300), 1e-300)
    return 1.0 + Kn * (1.257 + 0.4 * np.exp(-1.1 / Kn))


def fuchs_jump_distance_delta(r_cm, lambda_p_cm):
    """
    Fuchs jump distance used in the transition-regime Brownian kernel.
    """
    r = np.maximum(np.asarray(r_cm, dtype=float), 1e-60)
    lp = np.maximum(np.asarray(lambda_p_cm, dtype=float), 1e-60)

    num = (2.0 * r + lp) ** 3 - (4.0 * r**2 + lp**2) ** 1.5
    delta = num / (6.0 * r * lp) - 2.0 * r

    return np.maximum(delta, 0.0)





def charge_factor(r_i_cm, r_j_cm, T, coag_settings):
    """
    Electrostatic suppression factor for coagulation between particles
    carrying charges of the same sign.

    The dimensionless electrostatic barrier is

        tau = E_elec / (k_B T)

    and the coagulation-rate modification factor is

        f_e = tau / (exp(tau) - 1).

    This follows the Fuchs-style expression used by Lavvas et al. (2010).

    Parameters
    ----------
    r_i_cm, r_j_cm : array-like
        Collision radii of the two particles in cm.
    T : float
        Local atmospheric temperature in K.
    coag_settings : dict
        Coagulation settings. The relevant entries are:
        - charge_suppression or use_charge_in_coag
        - charge_density_e_per_um

    Returns
    -------
    ndarray
        Multiplicative coagulation suppression factor in the range [0, 1].
    """
    use_charge = bool(
        coag_settings.get(
            "charge_suppression",
            coag_settings.get("use_charge_in_coag", True),
        )
    )

    r_i_cm = np.asarray(r_i_cm, dtype=float)
    r_j_cm = np.asarray(r_j_cm, dtype=float)

    if not use_charge:
        return np.ones(np.broadcast(r_i_cm, r_j_cm).shape, dtype=float)

    charge_density = float(
        coag_settings.get("charge_density_e_per_um", 15.0)
    )

    # Elementary charge in electrostatic CGS units.
    e_cgs = 4.803204712e-10  # statCoulomb

    kT = k_B * max(float(T), 1.0)

    # Convert particle radii from cm to microns because the adopted charge
    # density is expressed as elementary charges per micron of radius.
    r_i_um = r_i_cm * 1e4
    r_j_um = r_j_cm * 1e4

    Zi = charge_density * r_i_um
    Zj = charge_density * r_j_um

    a_contact = np.maximum(r_i_cm + r_j_cm, 1e-300)

    # Electrostatic repulsion energy for particles carrying charges
    # of the same sign.
    E_elec = Zi * Zj * e_cgs**2 / a_contact

    tau = np.maximum(E_elec / max(kT, 1e-300), 0.0)

    # Numerically stable evaluation of tau / [exp(tau) - 1].
    # The series expansion avoids cancellation when tau is tiny.
    f_e = np.empty_like(tau, dtype=float)

    small = tau < 1e-6
    large = ~small

    tau_small = tau[small]
    f_e[small] = 1.0 - 0.5 * tau_small + tau_small**2 / 12.0

    tau_large = np.minimum(tau[large], 700.0)
    f_e[large] = tau_large / np.expm1(tau_large)

    return np.clip(f_e, 0.0, 1.0)






def build_remap_tables(ctx):
    """
    Build mass-bin remapping tables for coagulation.

    When particles in bins i and j coagulate, their combined mass generally
    falls between two existing bins. This table stores the lower/upper target
    bins and interpolation weight.
    """
    m = np.asarray(ctx["m_bin"], dtype=float)
    n_bin = int(ctx["n_bin"])

    m_new = m[:, None] + m[None, :]

    k_lo = np.full((n_bin, n_bin), -1, dtype=int)
    k_hi = np.full((n_bin, n_bin), -1, dtype=int)
    w_hi = np.zeros((n_bin, n_bin), dtype=float)
    overflow = np.zeros((n_bin, n_bin), dtype=bool)

    for i in range(n_bin):
        for j in range(n_bin):
            mn = m_new[i, j]

            if mn >= m[-1]:
                overflow[i, j] = True
                k_lo[i, j] = n_bin - 1
                k_hi[i, j] = n_bin - 1
                w_hi[i, j] = 0.0
                continue

            hi = int(np.searchsorted(m, mn, side="right"))
            lo = max(hi - 1, 0)
            hi = min(hi, n_bin - 1)

            if hi == lo:
                wh = 0.0
            else:
                wh = (mn - m[lo]) / max(m[hi] - m[lo], 1e-300)
                wh = float(np.clip(wh, 0.0, 1.0))

            k_lo[i, j] = lo
            k_hi[i, j] = hi
            w_hi[i, j] = wh

    ii, jj = np.triu_indices(n_bin, k=0)
    diag = ii == jj

    ctx.update(
        {
            "k_lo": k_lo,
            "k_hi": k_hi,
            "w_hi": w_hi,
            "overflow": overflow,
            "m_new": m_new,
            "tri_ii": ii,
            "tri_jj": jj,
            "tri_diag": diag,
            "tri_klo": k_lo[ii, jj],
            "tri_khi": k_hi[ii, jj],
            "tri_whi": w_hi[ii, jj],
            "tri_overflow": overflow[ii, jj],
            "tri_mnew": m_new[ii, jj],
        }
    )

    return ctx


def brownian_fuchs_kernel_layer(ctx, k, coag_settings):
    """
    Brownian coagulation kernel with Fuchs correction and optional charge suppression.
    """
    m = np.asarray(ctx["m_bin"], dtype=float)

    r_drag = np.maximum(ctx["r_drag_layer_cm"][k], 1e-60)
    r_coll = np.maximum(ctx["r_collision_layer_cm"][k], 1e-60)

    T_layer = float(ctx["T"][k])
    mfp_layer = float(ctx["mfp"][k])
    visc_layer = float(ctx["visc"][k])

    Cc = cunningham_slip_correction(r_drag, mfp_layer)

    D = k_B * T_layer * Cc / (
        6.0 * np.pi * max(visc_layer, 1e-300) * r_drag
    )

    c = np.sqrt(8.0 * k_B * T_layer / (np.pi * np.maximum(m, 1e-60)))
    lambda_p = 8.0 * D / (np.pi * np.maximum(c, 1e-60))
    delta = fuchs_jump_distance_delta(r_drag, lambda_p)

    Dij = D[:, None] + D[None, :]
    cij = np.sqrt(c[:, None] ** 2 + c[None, :] ** 2)
    delta_ij = np.sqrt(delta[:, None] ** 2 + delta[None, :] ** 2)
    Rij = np.maximum(r_coll[:, None] + r_coll[None, :], 1e-60)

    beta_inv = (
        Rij / np.maximum(Rij + delta_ij, 1e-60)
        + 4.0 * Dij / np.maximum(cij * Rij, 1e-60)
    )

    K_raw = 4.0 * np.pi * Rij * Dij / np.maximum(beta_inv, 1e-60)

    fe = charge_factor(
        r_coll[:, None],
        r_coll[None, :],
        T_layer,
        coag_settings,
    )

    return np.maximum(K_raw * fe, 0.0)


def kawashima_collection_efficiency_layer(ctx, k):
    """
    Kawashima & Ikoma (2018) gravitational-collection efficiency.

    Based on their Equations (25)-(30), following Jacobson (2005).

    Parameters
    ----------
    ctx : dict
        Caligo microphysics context.
    k : int
        Vertical-layer index.

    Returns
    -------
    np.ndarray
        Symmetric E_coll[i, j] matrix with shape (n_bin, n_bin).

    Notes
    -----
    Kawashima & Ikoma use one spherical-particle radius s.

    Caligo separately tracks:
        r_collision_layer_cm : geometric collision radius
        r_drag_layer_cm      : aerodynamic drag radius

    Here, the drag radius is used for Reynolds and Stokes numbers
    because those quantities describe the interaction with the gas.
    For compact spheres, the collision and drag radii are identical.
    """
    r_hydro = np.maximum(
        np.asarray(ctx["r_drag_layer_cm"][k], dtype=float),
        1e-60,
    )

    v = np.maximum(
        np.asarray(ctx["v_set"][k], dtype=float),
        0.0,
    )

    eta_gas = max(float(ctx["visc"][k]), 1e-300)
    rho_gas = max(float(ctx["rho_gas"][k]), 1e-300)
    g_layer = max(float(ctx["g_planet"]), 1e-300)

    # Kinematic viscosity [cm^2 s^-1].
    nu_gas = eta_gas / rho_gas

    # Pairwise matrices.
    r_i = r_hydro[:, None]
    r_j = r_hydro[None, :]

    v_i = v[:, None]
    v_j = v[None, :]

    dv = np.abs(v_i - v_j)

    # Identify the larger aerodynamic collector particle.
    r_large = np.maximum(r_i, r_j)

    larger_is_i = r_i > r_j
    tied_radius = np.isclose(r_i, r_j, rtol=1e-12, atol=0.0)

    v_large = np.where(larger_is_i, v_i, v_j)
    v_small = np.where(larger_is_i, v_j, v_i)

    # Preserve symmetry if two particles have the same drag radius.
    v_large = np.where(tied_radius, np.maximum(v_i, v_j), v_large)
    v_small = np.where(tied_radius, np.minimum(v_i, v_j), v_small)

    # Kawashima & Ikoma (2018), Eq. (28).
    Re_large = (
        2.0
        * r_large
        * v_large
        / max(nu_gas, 1e-300)
    )

    # Kawashima & Ikoma (2018), Eq. (30).
    St = (
        v_small
        * dv
        / np.maximum(r_large * g_layer, 1e-300)
    )

    # Kawashima & Ikoma (2018), Eq. (26).
    E_V = np.zeros_like(St)

    mask = St > 1.214

    E_V[mask] = (
        1.0
        + 0.75
        * np.log(2.0 * St[mask])
        / np.maximum(St[mask] - 1.214, 1e-300)
    ) ** (-2.0)

    # Kawashima & Ikoma (2018), Eq. (27).
    E_A = (
        St
        / np.maximum(St + 0.5, 1e-300)
    ) ** 2

    # Kawashima & Ikoma (2018), Eq. (25).
    E_coll = (
        60.0 * E_V
        + E_A * Re_large
    ) / np.maximum(60.0 + Re_large, 1e-300)

    E_coll = np.nan_to_num(
        E_coll,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return np.clip(E_coll, 0.0, 1.0)

def differential_settling_kernel_layer(ctx, k, coag_settings=None):
    """
    Differential-settling gravitational-collection kernel.

    Available collection-efficiency treatments:
        "unity"      : E_coll = 1
        "kawashima"  : Kawashima & Ikoma (2018), Equations (25)-(30)

    Returns
    -------
    np.ndarray
        Gravitational-collection kernel [cm^3 s^-1].
    """
    if coag_settings is None:
        coag_settings = {}

    r_coll = np.maximum(
        np.asarray(ctx["r_collision_layer_cm"][k], dtype=float),
        1e-60,
    )

    v = np.maximum(
        np.asarray(ctx["v_set"][k], dtype=float),
        0.0,
    )

    area = np.pi * (r_coll[:, None] + r_coll[None, :]) ** 2
    dv = np.abs(v[:, None] - v[None, :])

    mode = str(
        coag_settings.get("collection_efficiency", "kawashima")
    ).lower()

    if mode in {"unity", "one", "geometric"}:
        E_coll = np.ones_like(area)

    elif mode in {"kawashima", "kawashima_ikoma", "jacobson"}:
        E_coll = kawashima_collection_efficiency_layer(ctx, k)

    else:
        raise ValueError(
            "Unknown collection_efficiency mode. "
            "Use 'unity' or 'kawashima'."
        )

    return np.maximum(E_coll * area * dv, 0.0)



def build_coagulation_kernel(ctx, case=None):
    """
    Build K_coag[z, i, j] in cm^3/s.

    Uses:
        case.coagulation_settings if case is supplied,
        otherwise falls back to default settings.
    """
    if case is not None:
        coag_settings = getattr(case, "coagulation_settings", {})
    else:
        coag_settings = {}

    use_coag = bool(coag_settings.get("use", True))
    use_brownian = bool(coag_settings.get("brownian", True))
    use_diff_settling = bool(coag_settings.get("differential_settling", False))

    kernel_scale = float(
        coag_settings.get(
            "kernel_scale",
            coag_settings.get("coag_kernel_scale", 1.0),
        )
    )

    nz = int(ctx["nz"])
    n_bin = int(ctx["n_bin"])

    K = np.zeros((nz, n_bin, n_bin), dtype=float)

    if not use_coag:
        ctx["K_coag"] = K
        print("Coagulation disabled; K_coag is zero.", flush=True)
        return ctx

    for k in range(nz):
        K_layer = np.zeros((n_bin, n_bin), dtype=float)

        if use_brownian:
            K_layer += brownian_fuchs_kernel_layer(ctx, k, coag_settings)

            if use_diff_settling:
                K_layer += differential_settling_kernel_layer(
                    ctx,
                    k,
                    coag_settings=coag_settings,
                )

        K[k] = kernel_scale * np.maximum(K_layer, 0.0)

    ctx["K_coag"] = K
 
    Kmax = float(np.max(K))
    print(f"Max coag kernel = {Kmax:.3e} cm^3/s", flush=True)

    if Kmax > 0:
        kmax, imax, jmax = np.unravel_index(np.argmax(K), K.shape)

        print("[coag diagnostic] Max K location:", flush=True)
        print(f"  layer k = {kmax}", flush=True)
        print(f"  P = {ctx['Pbar'][kmax]:.3e} bar", flush=True)
        print(f"  bins i,j = {imax}, {jmax}", flush=True)
        print(f"  Kmax = {K[kmax, imax, jmax]:.3e} cm^3/s", flush=True)
        print(
            f"  r_compact_i,j = "
            f"{ctx['r_compact_nm'][imax]:.3e}, "
            f"{ctx['r_compact_nm'][jmax]:.3e} nm",
            flush=True,
        )
        print(
            f"  r_collision_i,j = "
            f"{ctx['r_collision_layer_cm'][kmax, imax] * 1e7:.3e}, "
            f"{ctx['r_collision_layer_cm'][kmax, jmax] * 1e7:.3e} nm",
            flush=True,
        )

        if "Df_layer" in ctx:
            print(
                f"  Df_layer = {ctx['Df_layer'][kmax]:.3f}, "
                f"w_agg = {ctx['w_agg_layer'][kmax]:.3f}",
                flush=True,
            )

    return ctx


def prepare_coagulation_context(ctx, case=None):
    """
    Add remap tables and coagulation kernel to an existing context.

    This lets us test coagulation setup without editing build_microphysics_context yet.
    """
    ctx = build_remap_tables(ctx)
    ctx = build_coagulation_kernel(ctx, case=case)
    return ctx




# ---------------------------------------------------------------------
# Coagulation gain/loss and adaptive coagulation stepping
# ---------------------------------------------------------------------


def coag_gain_loss(n, ctx):
    """
    Compute coagulation gain and loss terms.

    Parameters
    ----------
    n : ndarray
        Particle number density [cm^-3], shape (nz, n_bin).
    ctx : dict
        Microphysics context containing K_coag and remap tables.

    Returns
    -------
    P : ndarray
        Coagulation production term [cm^-3 s^-1].
    L : ndarray
        Coagulation loss coefficient [s^-1], where loss is n * L.
    overflow_mass_col_rate : float
        Mass column rate sent to overflow pairs [g cm^-2 s^-1].
    active_layers : int
        Number of layers with active coagulation.
    """
    if "K_coag" not in ctx:
        raise RuntimeError(
            "ctx does not contain K_coag. Run cm.prepare_coagulation_context(ctx, case=case) first."
        )

    settings = _coag_settings_from_ctx(ctx)

    skip_ntot = float(settings.get("skip_ntot", settings.get("coag_skip_ntot", 1e2)))
    skip_mass_dens = float(
        settings.get("skip_mass_dens", settings.get("coag_skip_mass_dens", 1e-30))
    )
    active_rel = float(settings.get("active_rel", settings.get("coag_active_rel", 1e-8)))
    overflow_mode = settings.get(
        "top_overflow_mode",
        settings.get("coag_top_overflow_mode", "mass_conserve"),
    )

    n = np.maximum(np.asarray(n, dtype=float), 0.0)

    nz = int(ctx["nz"])
    n_bin = int(ctx["n_bin"])

    P = np.zeros((nz, n_bin), dtype=float)
    L = np.zeros((nz, n_bin), dtype=float)

    overflow_mass_col_rate = 0.0
    active_layers = 0

    ii = ctx["tri_ii"]
    jj = ctx["tri_jj"]
    diag = ctx["tri_diag"]

    klo = ctx["tri_klo"]
    khi = ctx["tri_khi"]
    whi = ctx["tri_whi"]

    overflow = ctx["tri_overflow"]
    mnew = ctx["tri_mnew"]
    m_bin = ctx["m_bin"]

    for k in range(nz):
        nk = n[k]

        ntot = float(np.sum(nk))
        mdens = float(np.sum(nk * m_bin))

        if ntot < skip_ntot or mdens < skip_mass_dens:
            continue

        active_bins = nk > max(np.max(nk) * active_rel, 1e-300)

        if np.count_nonzero(active_bins) == 0:
            continue

        nk_eff = nk.copy()
        nk_eff[~active_bins] = 0.0

        Kk = ctx["K_coag"][k]

        # Loss coefficient L_i = sum_j K_ij n_j.
        L[k] = Kk @ nk_eff

        pair_active = active_bins[ii] & active_bins[jj]

        rates = np.zeros_like(ii, dtype=float)

        if np.any(pair_active):
            rates[pair_active] = (
                Kk[ii[pair_active], jj[pair_active]]
                * nk_eff[ii[pair_active]]
                * nk_eff[jj[pair_active]]
            )

        # Same-bin pairs are counted once, not twice.
        rates[diag] *= 0.5
        rates = np.maximum(rates, 0.0)

        if not np.any(rates > 0):
            continue

        gain = np.zeros(n_bin, dtype=float)

        keep = ~overflow
        keep_active = keep & (rates > 0)

        if np.any(keep_active):
            np.add.at(
                gain,
                klo[keep_active],
                (1.0 - whi[keep_active]) * rates[keep_active],
            )
            np.add.at(
                gain,
                khi[keep_active],
                whi[keep_active] * rates[keep_active],
            )

        ov = overflow & (rates > 0)

        if np.any(ov):
            if overflow_mode == "mass_conserve":
                gain[-1] += np.sum(rates[ov] * mnew[ov] / max(m_bin[-1], 1e-300))
            else:
                gain[-1] += np.sum(rates[ov])

            overflow_mass_col_rate += float(np.sum(rates[ov] * mnew[ov]) * ctx["dz"][k])

        P[k] = gain
        active_layers += 1

    return P, L, overflow_mass_col_rate, active_layers




def deep_recycling_rate_profile(Pbar, sink_settings):
    """
    Parameterized unresolved lower-atmosphere haze recycling/removal.

    Off by default, matching the clean thermal-sink baseline.
    """
    Pbar = np.asarray(Pbar, dtype=float)

    use_deep = bool(
        sink_settings.get(
            "use_deep_recycling",
            sink_settings.get("deep_recycling", False),
        )
    )

    if not use_deep:
        return np.zeros_like(Pbar), "deep recycling disabled"

    p_on = float(sink_settings.get("deep_recycling_p_on_bar", 1e-3))
    width = float(sink_settings.get("deep_recycling_width_dex", 0.5))
    tau = float(sink_settings.get("deep_recycling_timescale", 1.0e6))

    logP = safe_log10(Pbar)
    logP_on = np.log10(max(p_on, 1e-300))

    shape = 0.5 * (1.0 + np.tanh((logP - logP_on) / max(width, 1e-6)))
    rate = shape / max(tau, 1e-300)

    note = (
        "deep recycling active: unresolved lower-atmosphere removal/recycling "
        f"turns on near P > {p_on:.3e} bar with tau = {tau:.3e} s"
    )

    return rate, note


def ensure_deep_recycling_in_context(ctx):
    """
    Add deep recycling rate to ctx if missing.
    """
    if "deep_recycling_rate" in ctx:
        return ctx

    sink_settings = _sink_settings_from_ctx(ctx)
    rate, note = deep_recycling_rate_profile(ctx["Pbar"], sink_settings)

    ctx["deep_recycling_rate"] = rate
    ctx["deep_recycling_note"] = note

    return ctx


def _ensure_q_recycled_array(q_recycled, ctx):
    """
    Old code treats recycled gas as a mass mixing ratio-like profile.
    Keep scalar zero compatible, but convert to an nz array when needed.
    """
    if np.isscalar(q_recycled):
        return np.zeros(ctx["nz"], dtype=float) + float(q_recycled)

    q = np.asarray(q_recycled, dtype=float)

    if q.shape != (ctx["nz"],):
        return np.zeros(ctx["nz"], dtype=float)

    return q.copy()


def recycled_gas_column(q_recycled, ctx):
    """
    Compute the passive recycled-gas column mass [g cm^-2].

    q_recycled is treated as a mass-mixing-ratio-like vertical profile.
    A scalar zero remains supported for backward compatibility.
    """
    if q_recycled is None:
        return 0.0

    if np.isscalar(q_recycled):
        return float(q_recycled)

    q = np.asarray(q_recycled, dtype=float)

    if q.shape != (ctx["nz"],):
        raise ValueError(
            "q_recycled must be a scalar or an array with shape (nz,)."
        )

    rho_gas = np.asarray(ctx["rho_gas"], dtype=float)
    dz = np.asarray(ctx["dz"], dtype=float)

    return float(np.sum(q * rho_gas * dz))

def _mass_sink_exponential(n, dt, rate, ctx):
    """
    Apply a first-order local mass sink to all bins.

    Returns:
        n_after
        loss_mass_column
        loss_mass_density_layer [g cm^-3]
    """
    n = np.maximum(np.asarray(n, dtype=float), 0.0)
    rate = np.maximum(np.asarray(rate, dtype=float), 0.0)
    dt = float(dt)

    if dt <= 0 or np.max(rate) <= 0:
        return n.copy(), 0.0, np.zeros(ctx["nz"], dtype=float)

    decay = np.exp(-rate[:, None] * dt)

    n_after = n * decay
    n_loss = np.maximum(n - n_after, 0.0)

    loss_mass_density = np.sum(n_loss * ctx["m_bin"][None, :], axis=1)
    loss_mass_column = float(np.sum(loss_mass_density * ctx["dz"]))

    return np.maximum(n_after, 0.0), loss_mass_column, loss_mass_density


def apply_thermal_destruction(n, dt, ctx, q_recycled=0.0):
    """
    Apply thermal destruction.

    If enable_recycling_to_gas is true, add destroyed haze mass to q_recycled.
    Default remains off, matching the clean baseline.
    """
    sink_settings = _sink_settings_from_ctx(ctx)

    n_after, loss_mass, loss_density = _mass_sink_exponential(
        n,
        dt,
        ctx["thermal_sink_rate"],
        ctx,
    )

    q_rec = q_recycled

    recycle = bool(
        sink_settings.get(
            "enable_recycling_to_gas",
            sink_settings.get("recycling_to_gas", False),
        )
    )

    if recycle:
        q_arr = _ensure_q_recycled_array(q_recycled, ctx)
        q_arr += loss_density / np.maximum(ctx["rho_gas"], 1e-300)
        q_rec = q_arr

    return n_after, q_rec, loss_mass


def apply_deep_recycling(n, dt, ctx, q_recycled=0.0):
    """
    Apply optional deep recycling/removal.
    """
    ctx = ensure_deep_recycling_in_context(ctx)
    sink_settings = _sink_settings_from_ctx(ctx)

    rate = ctx.get("deep_recycling_rate", np.zeros(ctx["nz"]))

    n_after, loss_mass, loss_density = _mass_sink_exponential(n, dt, rate, ctx)

    q_rec = q_recycled

    recycle = bool(
        sink_settings.get(
            "deep_recycling_to_gas",
            sink_settings.get("deep_recycle_to_gas", False),
        )
    )

    if recycle:
        q_arr = _ensure_q_recycled_array(q_recycled, ctx)
        q_arr += loss_density / np.maximum(ctx["rho_gas"], 1e-300)
        q_rec = q_arr

    return n_after, q_rec, loss_mass



def coagulation_step(n_in, dt, ctx):
    """
    Adaptive coagulation update using the old fixed-point/semi-implicit logic.

    Old-code structure:
      - recompute P, L inside each substep
      - choose dt_sub from max coagulation loss rate
      - fixed-point iterate n_next = (n_start + dt_sub P)/(1 + dt_sub L)
      - conserve mass through remap tables
      - flag incomplete coagulation advancement if t_done < 0.999 dt
    """
    settings = _coag_settings_from_ctx(ctx)

    use_coag = bool(settings.get("use", True))
    substep_frac = float(
        settings.get("substep_frac", settings.get("coag_substep_frac", 0.03))
    )
    max_substeps = int(
        settings.get("max_substeps", settings.get("coag_max_substeps", 10000))
    )
    fixed_point_iters = int(
        settings.get("fixed_point_iters", settings.get("coag_fixed_point_iters", 4))
    )
    skip_dtl = float(settings.get("skip_dtl", settings.get("coag_skip_dtl", 1e-3)))
    active_rel = float(settings.get("active_rel", settings.get("coag_active_rel", 1e-8)))
    mass_err_warn = float(
        settings.get("mass_err_warn", settings.get("coag_mass_err_warn", 1e-6))
    )
    max_substep_retries = int(
        settings.get("substep_retries", settings.get("coag_substep_retries", 8))
    )

    diag = {
        "coag_substeps": 0,
        "coag_active_layers": 0,
        "coag_overflow_mass": 0.0,
        "coag_mass_error": 0.0,
        "coag_rel_mass_error": 0.0,
        "coag_number_loss_col": 0.0,
        "coag_max_dtl": 0.0,
        "coag_t_done_frac": 1.0,
        "coag_hit_substep_cap": False,
        "coag_min_sub_dt": np.inf,
        "coag_max_sub_dt": 0.0,
        "coag_substep_retries": 0,
        "coag_failed_substep": False,
    }

    if not use_coag or dt <= 0:
        return np.maximum(n_in, 0.0), diag

    if "K_coag" not in ctx:
        raise RuntimeError(
            "ctx does not contain K_coag. Run prepare_coagulation_context first."
        )

    n = np.maximum(np.asarray(n_in, dtype=float), 0.0).copy()

    m_bin = np.asarray(ctx["m_bin"], dtype=float)
    dz = np.asarray(ctx["dz"], dtype=float)

    mass0 = float(np.sum(n * m_bin[None, :] * dz[:, None]))
    num0 = float(np.sum(n * dz[:, None]))

    if mass0 <= 0:
        return n, diag

    P0, L0, _, active_layers0 = coag_gain_loss(n, ctx)

    active0 = n > np.maximum(np.max(n, axis=1, keepdims=True) * active_rel, 1e-300)
    max_loss0 = float(np.max(np.where(active0, L0, 0.0))) if np.any(active0) else 0.0

    diag["coag_max_dtl"] = float(dt * max_loss0)
    diag["coag_active_layers"] = int(active_layers0)

    if max_loss0 * dt < skip_dtl:
        return n, diag

    t_done = 0.0
    substeps = 0

    while t_done < dt:
        if substeps >= max_substeps:
            diag["coag_hit_substep_cap"] = True
            break

        P, L, overflow_rate, active_layers = coag_gain_loss(n, ctx)

        active = n > np.maximum(np.max(n, axis=1, keepdims=True) * active_rel, 1e-300)
        max_loss = float(np.max(np.where(active, L, 0.0))) if np.any(active) else 0.0

        if max_loss <= 0:
            break

        remaining = dt - t_done

        if max_loss * remaining < skip_dtl:
            break

        dt_sub = min(remaining, substep_frac / max(max_loss, 1e-300))

        accepted = False

        for retry in range(max_substep_retries + 1):
            n_start = n.copy()
            mass_before = float(np.sum(n_start * m_bin[None, :] * dz[:, None]))

            n_next = n_start.copy()

            for _ in range(max(fixed_point_iters, 1)):
                P_iter, L_iter, _, _ = coag_gain_loss(n_next, ctx)
                n_next = (n_start + dt_sub * P_iter) / (
                    1.0 + dt_sub * np.maximum(L_iter, 0.0)
                )
                n_next = np.maximum(n_next, 0.0)

            if np.any(~np.isfinite(n_next)) or np.any(n_next < -1e-30):
                dt_sub *= 0.5
                diag["coag_substep_retries"] += 1
                continue

            n_trial = np.maximum(n_next, 0.0)

            mass_after = float(np.sum(n_trial * m_bin[None, :] * dz[:, None]))
            rel_err = (mass_after - mass_before) / max(abs(mass_before), 1e-300)

            if abs(rel_err) > 10.0 * mass_err_warn:
                dt_sub *= 0.5
                diag["coag_substep_retries"] += 1
                continue

            # Old-code behavior: if error is small but nonzero, renormalize
            # to preserve particle mass exactly.
            if abs(rel_err) > 0 and abs(rel_err) <= 10.0 * mass_err_warn:
                n_trial *= mass_before / max(mass_after, 1e-300)
                mass_after = float(np.sum(n_trial * m_bin[None, :] * dz[:, None]))

            accepted = True
            break

        if not accepted:
            diag["coag_failed_substep"] = True
            break

        n = n_trial
        t_done += dt_sub
        substeps += 1

        diag["coag_overflow_mass"] += overflow_rate * dt_sub
        diag["coag_active_layers"] = max(diag["coag_active_layers"], int(active_layers))
        diag["coag_min_sub_dt"] = min(diag["coag_min_sub_dt"], float(dt_sub))
        diag["coag_max_sub_dt"] = max(diag["coag_max_sub_dt"], float(dt_sub))
        diag["coag_max_dtl"] = max(diag["coag_max_dtl"], float(dt_sub * max_loss))

    mass1 = float(np.sum(n * m_bin[None, :] * dz[:, None]))
    num1 = float(np.sum(n * dz[:, None]))

    diag["coag_substeps"] = int(substeps)
    diag["coag_t_done_frac"] = float(t_done / max(dt, 1e-300))
    diag["coag_mass_error"] = float(mass1 - mass0)
    diag["coag_rel_mass_error"] = float((mass1 - mass0) / max(abs(mass0), 1e-300))
    diag["coag_number_loss_col"] = float(num0 - num1)

    if not np.isfinite(diag["coag_min_sub_dt"]):
        diag["coag_min_sub_dt"] = 0.0

    if t_done < 0.999 * dt:
        diag["coag_hit_substep_cap"] = True

    return np.maximum(n, 0.0), diag

# ---------------------------------------------------------------------
# Full microphysics step and full chunked integration
# Includes: source + coagulation + diffusion + settling + thermal sink
# ---------------------------------------------------------------------
def take_full_microphysics_step(
    n,
    q_recycled,
    dt,
    ctx,
    include_source=True,
    include_coagulation=True,
    include_diffusion=True,
    include_settling=True,
    include_thermal=True,
    include_deep_recycling=True,
):
    """
    Take one full microphysics step.

    Direct old-code order:
        1. source injection
        2. coagulation
        3. diffusion
        4. settling
        5. parameterized thermal destruction
        6. optional deep recycling

    The recycled-gas tracer is passive bookkeeping only. It is not coupled
    back into photochem.
    """
    n = np.asarray(n, dtype=float).copy()
    dt = float(dt)

    if dt <= 0:
        raise ValueError("dt must be positive.")

    ctx = ensure_deep_recycling_in_context(ctx)

    mass_initial = column_haze_mass(n, ctx)
    recycled_initial = recycled_gas_column(q_recycled, ctx)

    source_mass = 0.0
    bottom_loss_mass = 0.0
    thermal_loss_mass = 0.0
    deep_loss_mass = 0.0

    coag_diag = {
        "coag_substeps": 0,
        "coag_active_layers": 0,
        "coag_overflow_mass": 0.0,
        "coag_mass_error": 0.0,
        "coag_rel_mass_error": 0.0,
        "coag_number_loss_col": 0.0,
        "coag_max_dtl": 0.0,
        "coag_t_done_frac": 1.0,
        "coag_hit_substep_cap": False,
        "coag_min_sub_dt": 0.0,
        "coag_max_sub_dt": 0.0,
        "coag_substep_retries": 0,
        "coag_failed_substep": False,
    }

    sed_diag = {
        "sed_substeps": 0,
        "sed_cfl_max": 0.0,
        "hit_max_substeps": False,
    }

    if include_source:
        n += ctx["S_number"] * dt

        if ctx.get("F_top_source", 0.0) > 0:
            n[-1, :] += (
                ctx["top_flux_number"]
                * dt
                / max(ctx["dz"][-1], 1e-300)
            )

        source_mass = ctx["F_src"] * dt

    if include_coagulation:
        if "K_coag" not in ctx:
            raise RuntimeError(
                "Coagulation requested, but ctx does not contain K_coag. "
                "Run ctx = prepare_coagulation_context(ctx, case=case) first."
            )

        n, coag_diag = coagulation_step(n, dt, ctx)

    if include_diffusion:
        n = implicit_diffuse_density(n, dt, ctx)

    if include_settling:
        n, bottom_loss_mass, sed_diag = settle_density_upwind(n, dt, ctx)

    if include_thermal:
        n, q_recycled, thermal_loss_mass = apply_thermal_destruction(
            n,
            dt,
            ctx,
            q_recycled=q_recycled,
        )

    if include_deep_recycling:
        n, q_recycled, deep_loss_mass = apply_deep_recycling(
            n,
            dt,
            ctx,
            q_recycled=q_recycled,
        )

    mass_final = column_haze_mass(n, ctx)
    recycled_final = recycled_gas_column(q_recycled, ctx)

    coag_mass_error = coag_diag.get("coag_mass_error", 0.0)

    expected_final = (
        mass_initial
        + source_mass
        + coag_mass_error
        - bottom_loss_mass
        - thermal_loss_mass
        - deep_loss_mass
    )

    mass_error = mass_final - expected_final

    diagnostics = {
        "dt": dt,
        "mass_initial": mass_initial,
        "source_mass": source_mass,
        "bottom_loss_mass": bottom_loss_mass,
        "thermal_loss_mass": thermal_loss_mass,
        "deep_loss_mass": deep_loss_mass,
        "mass_final": mass_final,
        "expected_final": expected_final,
        "mass_error": mass_error,
        "relative_mass_error": (
            mass_error / max(abs(expected_final), 1e-300)
        ),

        # Original-style aliases
        "source": source_mass,
        "vol_source": (
            ctx.get("F_vol_source", ctx["F_src"]) * dt
            if include_source
            else 0.0
        ),
        "top_source": (
            ctx.get("F_top_source", 0.0) * dt
            if include_source
            else 0.0
        ),
        "bottom_loss": bottom_loss_mass,
        "thermal_loss": thermal_loss_mass,
        "deep_loss": deep_loss_mass,
        "total_sink_loss": (
            bottom_loss_mass
            + thermal_loss_mass
            + deep_loss_mass
        ),

        # Passive recycled-gas bookkeeping
        "recycled_gas_initial": recycled_initial,
        "recycled_gas_final": recycled_final,
        "recycled_gas_gain": recycled_final - recycled_initial,

        **coag_diag,
        **sed_diag,
    }

    return np.maximum(n, 0.0), q_recycled, diagnostics






def full_microphysics_step_with_retry(
    n,
    q_recycled,
    dt,
    ctx,
    max_retries=8,
    retry_shrink=0.5,
    max_relative_mass_error=1e-8,
    max_column_jump_frac=1.0,
):
    """
    Take one full microphysics step, shrinking dt if the step becomes bad.

    Direct old-code controls:
      - strict column jump control
      - retry on sedimentation cap
      - retry on coagulation substep cap if requested
    """
    solver_settings = _solver_settings_from_ctx(ctx)
    coag_settings = _coag_settings_from_ctx(ctx)

    max_column_jump_frac = float(
        solver_settings.get("max_column_jump_frac", max_column_jump_frac)
    )

    retry_on_coag_cap = bool(
        coag_settings.get(
            "retry_on_coag_substep_cap",
            solver_settings.get("retry_on_coag_substep_cap", True),
        )
    )

    dt_try = float(dt)
    last_diag = None

    for retry in range(max_retries + 1):
        n_trial, q_trial, diag = take_full_microphysics_step(
            n.copy(),
            q_recycled,
            dt_try,
            ctx,
        )

        diag["retries"] = retry

        finite_ok = np.all(np.isfinite(n_trial))
        nonnegative_ok = np.all(n_trial >= 0)
        mass_error_ok = abs(diag.get("relative_mass_error", 0.0)) <= max_relative_mass_error

        M0 = max(diag.get("mass_initial", 0.0), 1e-300)
        M1 = diag.get("mass_final", 0.0)

        if diag.get("mass_initial", 0.0) <= 0:
            jump_ok = True
        else:
            jump_ok = (M1 / M0) <= (1.0 + max_column_jump_frac)

        sed_ok = not diag.get("hit_max_substeps", False)

        coag_failed = diag.get("coag_failed_substep", False)
        coag_capped = diag.get("coag_hit_substep_cap", False)

        if retry_on_coag_cap:
            coag_ok = (not coag_failed) and (not coag_capped)
        else:
            coag_ok = not coag_failed

        good = (
            finite_ok
            and nonnegative_ok
            and mass_error_ok
            and jump_ok
            and sed_ok
            and coag_ok
        )

        if good:
            return n_trial, q_trial, dt_try, diag

        last_diag = diag
        dt_try *= retry_shrink

    raise RuntimeError(
        "full_microphysics_step_with_retry failed after retries. "
        f"last_dt={dt_try:.3e}, "
        f"last_relative_mass_error={last_diag.get('relative_mass_error', np.nan):.3e}, "
        f"hit_max_substeps={last_diag.get('hit_max_substeps', None)}, "
        f"coag_hit_substep_cap={last_diag.get('coag_hit_substep_cap', None)}, "
        f"coag_failed_substep={last_diag.get('coag_failed_substep', None)}, "
        f"sed_substeps={last_diag.get('sed_substeps', None)}, "
        f"coag_substeps={last_diag.get('coag_substeps', None)}"
    )

def run_full_microphysics_to_steady_state(
    ctx,
    dt_init=None,
    dt_growth=None,
    dt_max=None,
    chunk_time=None,
    max_chunks=None,
    max_step_retries=None,
    retry_shrink=None,
    ss_tol_storage=None,
    ss_tol_balance=None,
    ss_tol_profile=None,
    ss_tol_peak_dex=None,
    ss_min_stable_chunks=None,
    verbose=True,
):
    """
    Run the full haze microphysics model.

    Direct old-code solver structure:
      - adaptive dt growth
      - chunk-level diagnostics
      - source/sink/storage balance
      - strict source-sink steady-state criteria
      - no quasi-steady shortcut unless explicitly enabled
    """
    ctx = ensure_deep_recycling_in_context(ctx)

    solver_settings = _solver_settings_from_ctx(ctx)

    dt_init = float(dt_init if dt_init is not None else solver_settings.get("dt_init", 1e2))
    dt_growth = float(dt_growth if dt_growth is not None else solver_settings.get("dt_growth", 1.03))
    dt_max = float(dt_max if dt_max is not None else solver_settings.get("dt_max", 300.0))
    chunk_time = float(chunk_time if chunk_time is not None else solver_settings.get("chunk_time", 1.0e5))
    max_chunks = int(max_chunks if max_chunks is not None else solver_settings.get("max_chunks", 200))

    max_step_retries = int(
        max_step_retries
        if max_step_retries is not None
        else solver_settings.get("max_step_retries", 10)
    )
    retry_shrink = float(
        retry_shrink
        if retry_shrink is not None
        else solver_settings.get("retry_shrink", 0.5)
    )

    max_column_jump_frac = float(solver_settings.get("max_column_jump_frac", 1.0))

    ss_tol_storage = float(
        ss_tol_storage
        if ss_tol_storage is not None
        else solver_settings.get("ss_tol_storage", 0.08)
    )
    ss_tol_balance = float(
        ss_tol_balance
        if ss_tol_balance is not None
        else solver_settings.get("ss_tol_balance", 0.25)
    )
    ss_tol_profile = float(
        ss_tol_profile
        if ss_tol_profile is not None
        else solver_settings.get("ss_tol_profile", 0.04)
    )
    ss_tol_peak_dex = float(
        ss_tol_peak_dex
        if ss_tol_peak_dex is not None
        else solver_settings.get("ss_tol_peak_dex", 0.25)
    )
    ss_min_stable_chunks = int(
        ss_min_stable_chunks
        if ss_min_stable_chunks is not None
        else solver_settings.get("ss_min_stable_chunks", 3)
    )

    allow_quasi = bool(solver_settings.get("allow_quasi_steady_stop", False))
    quasi_min_chunk = int(solver_settings.get("quasi_steady_min_chunk", 40))
    quasi_storage_min = float(solver_settings.get("quasi_steady_storage_min", 0.0))

    enable_diagnostic_stop = bool(solver_settings.get("enable_diagnostic_stop", False))
    diagnostic_stop_min_chunk = int(solver_settings.get("diagnostic_stop_min_chunk", 40))
    diagnostic_stop_storage_min = float(solver_settings.get("diagnostic_stop_storage_min", 0.5))

    if "K_coag" not in ctx:
        raise RuntimeError(
            "ctx does not contain K_coag. Run ctx = prepare_coagulation_context(ctx, case=case) first."
        )

    if verbose:
        print("\n" + "=" * 70, flush=True)
        print("CALIGO: FULL MICROPHYSICS INTEGRATION", flush=True)
        print("=" * 70, flush=True)
        print(f"dt_init = {dt_init:.3e} s", flush=True)
        print(f"dt_growth = {dt_growth:.3e}", flush=True)
        print(f"dt_max = {dt_max:.3e} s", flush=True)
        print(f"chunk_time = {chunk_time:.3e} s", flush=True)
        print(f"max_chunks = {max_chunks}", flush=True)
        print(f"max_column_jump_frac = {max_column_jump_frac:.3e}", flush=True)

    n, q_recycled = initialize_haze_distribution(ctx)

    t = 0.0
    dt = dt_init
    stable_chunks = 0
    converged = False

    prev_n = n.copy()
    prev_peak = diagnostic_peak_pressure(n, ctx)

    hist = []

    for ichunk in range(1, max_chunks + 1):
        t_start = t
        M_before = column_haze_mass(n, ctx)

        chunk = {
            "chunk": ichunk,
            "t_start": t_start,
            "t_end": None,
            "dt_last": None,
            "steps": 0,

            "source": 0.0,
            "vol_source": 0.0,
            "top_source": 0.0,

            "bottom_loss": 0.0,
            "thermal_loss": 0.0,
            "deep_loss": 0.0,
            "total_sink_loss": 0.0,

            "mass_error": 0.0,
            "max_rel_mass_error": 0.0,

            "retries": 0,

            "max_sed_substeps": 0,
            "max_sed_cfl": 0.0,
            "hit_max_substeps_count": 0,

            "coag_substeps": 0,
            "max_coag_substeps": 0,
            "coag_active_layers": 0,
            "coag_overflow_mass": 0.0,
            "coag_number_loss_col": 0.0,
            "max_coag_max_dtl": 0.0,
            "min_coag_t_done_frac": 1.0,
            "coag_hit_substep_cap_count": 0,
            "coag_failed_substep_count": 0,
        }

        if verbose:
            print(f"\n--- Chunk {ichunk}/{max_chunks} ---", flush=True)

        while t - t_start < chunk_time:
            dt_step = min(dt, chunk_time - (t - t_start))

            n, q_recycled, dt_used, diag = full_microphysics_step_with_retry(
                n,
                q_recycled,
                dt_step,
                ctx,
                max_retries=max_step_retries,
                retry_shrink=retry_shrink,
                max_column_jump_frac=max_column_jump_frac,
            )

            t += dt_used

            # If the step was retried and shrunk, grow from the accepted dt.
            dt = min(dt_max, max(dt_init, dt_used * dt_growth))

            chunk["steps"] += 1

            chunk["source"] += diag.get("source_mass", 0.0)
            chunk["vol_source"] += diag.get("vol_source", 0.0)
            chunk["top_source"] += diag.get("top_source", 0.0)

            chunk["bottom_loss"] += diag.get("bottom_loss_mass", 0.0)
            chunk["thermal_loss"] += diag.get("thermal_loss_mass", 0.0)
            chunk["deep_loss"] += diag.get("deep_loss_mass", 0.0)

            chunk["mass_error"] += diag.get("mass_error", 0.0)
            chunk["max_rel_mass_error"] = max(
                chunk["max_rel_mass_error"],
                abs(diag.get("relative_mass_error", 0.0)),
            )

            chunk["retries"] += diag.get("retries", 0)

            chunk["max_sed_substeps"] = max(
                chunk["max_sed_substeps"],
                diag.get("sed_substeps", 0),
            )
            chunk["max_sed_cfl"] = max(
                chunk["max_sed_cfl"],
                diag.get("sed_cfl_max", 0.0),
            )

            if diag.get("hit_max_substeps", False):
                chunk["hit_max_substeps_count"] += 1

            chunk["coag_substeps"] += diag.get("coag_substeps", 0)
            chunk["max_coag_substeps"] = max(
                chunk["max_coag_substeps"],
                diag.get("coag_substeps", 0),
            )
            chunk["coag_active_layers"] = max(
                chunk["coag_active_layers"],
                diag.get("coag_active_layers", 0),
            )
            chunk["coag_overflow_mass"] += diag.get("coag_overflow_mass", 0.0)
            chunk["coag_number_loss_col"] += diag.get("coag_number_loss_col", 0.0)
            chunk["max_coag_max_dtl"] = max(
                chunk["max_coag_max_dtl"],
                diag.get("coag_max_dtl", 0.0),
            )
            chunk["min_coag_t_done_frac"] = min(
                chunk["min_coag_t_done_frac"],
                diag.get("coag_t_done_frac", 1.0),
            )

            if diag.get("coag_hit_substep_cap", False):
                chunk["coag_hit_substep_cap_count"] += 1

            if diag.get("coag_failed_substep", False):
                chunk["coag_failed_substep_count"] += 1

        M_after = column_haze_mass(n, ctx)

        chunk["t_end"] = t
        chunk["dt_last"] = dt
        chunk["column_mass"] = M_after
        chunk["storage"] = M_after - M_before
        chunk["total_sink_loss"] = (
            chunk["bottom_loss"]
            + chunk["thermal_loss"]
            + chunk["deep_loss"]
        )

        elapsed = max(t - t_start, 1e-300)

        F_src_chunk = chunk["source"] / elapsed
        F_sink_chunk = chunk["total_sink_loss"] / elapsed
        F_storage_chunk = chunk["storage"] / elapsed

        chunk["F_source"] = F_src_chunk
        chunk["F_vol_source"] = chunk["vol_source"] / elapsed
        chunk["F_top_source"] = chunk["top_source"] / elapsed

        chunk["F_sink"] = F_sink_chunk
        chunk["F_bottom"] = chunk["bottom_loss"] / elapsed
        chunk["F_thermal"] = chunk["thermal_loss"] / elapsed
        chunk["F_deep"] = chunk["deep_loss"] / elapsed
        chunk["F_storage"] = F_storage_chunk

        source_norm = max(abs(ctx["F_src"]), 1e-300)

        storage_ratio = abs(F_storage_chunk) / source_norm
        balance_ratio = abs(F_src_chunk - F_sink_chunk) / source_norm
        prof_change = profile_change_mass(prev_n, n, ctx)

        peak = diagnostic_peak_pressure(n, ctx)

        if np.isfinite(prev_peak) and np.isfinite(peak) and prev_peak > 0 and peak > 0:
            peak_shift_dex = abs(np.log10(peak) - np.log10(prev_peak))
        else:
            peak_shift_dex = np.inf

        chunk["storage_ratio"] = storage_ratio
        chunk["balance_ratio"] = balance_ratio
        chunk["profile_change"] = prof_change
        chunk["peak_pressure"] = peak
        chunk["peak_shift_dex"] = peak_shift_dex

        stable_now = (
            storage_ratio < ss_tol_storage
            and balance_ratio < ss_tol_balance
            and prof_change < ss_tol_profile
            and peak_shift_dex < ss_tol_peak_dex
        )

        if stable_now:
            stable_chunks += 1
        else:
            stable_chunks = 0

        chunk["stable_now"] = stable_now
        chunk["stable_chunks"] = stable_chunks

        hist.append(chunk)

        if verbose:
            print(f"t = {t:.3e} s ({t / 86400.0:.3e} d)", flush=True)
            print(f"steps in chunk = {chunk['steps']}", flush=True)
            print(f"column mass = {M_after:.3e} g cm^-2", flush=True)
            print(f"F_source = {F_src_chunk:.3e} g cm^-2 s^-1", flush=True)
            print(f"F_sink = {F_sink_chunk:.3e} g cm^-2 s^-1", flush=True)
            print(f"F_bottom = {chunk['F_bottom']:.3e} g cm^-2 s^-1", flush=True)
            print(f"F_thermal = {chunk['F_thermal']:.3e} g cm^-2 s^-1", flush=True)
            print(f"F_deep = {chunk['F_deep']:.3e} g cm^-2 s^-1", flush=True)
            print(f"F_storage = {F_storage_chunk:.3e} g cm^-2 s^-1", flush=True)
            print(f"storage_ratio = {storage_ratio:.3e}", flush=True)
            print(f"balance_ratio = {balance_ratio:.3e}", flush=True)
            print(f"profile_change = {prof_change:.3e}", flush=True)
            print(f"peak_pressure = {peak:.3e} bar", flush=True)
            print(f"peak_shift_dex = {peak_shift_dex:.3e}", flush=True)
            print(f"max_sed_substeps = {chunk['max_sed_substeps']}", flush=True)
            print(f"coag_substeps total = {chunk['coag_substeps']}", flush=True)
            print(f"max_coag_substeps = {chunk['max_coag_substeps']}", flush=True)
            print(f"coag_active_layers = {chunk['coag_active_layers']}", flush=True)
            print(f"max_coag_max_dtl = {chunk['max_coag_max_dtl']:.3e}", flush=True)
            print(f"stable_chunks = {stable_chunks}", flush=True)

        if stable_chunks >= ss_min_stable_chunks:
            converged = True

            if verbose:
                print("\nFull microphysics steady state reached.", flush=True)

            break

        if allow_quasi and ichunk >= quasi_min_chunk:
            if storage_ratio <= quasi_storage_min and prof_change < ss_tol_profile:
                converged = True

                if verbose:
                    print("\nQuasi-steady diagnostic stop reached.", flush=True)

                break

        if enable_diagnostic_stop and ichunk >= diagnostic_stop_min_chunk:
            if storage_ratio <= diagnostic_stop_storage_min:
                if verbose:
                    print("\nDiagnostic stop reached.", flush=True)
                break

        prev_n = n.copy()
        prev_peak = peak

    if verbose:
        print("\n" + "=" * 70, flush=True)
        print("FULL MICROPHYSICS INTEGRATION COMPLETE", flush=True)
        print("=" * 70, flush=True)
        print(f"converged = {converged}", flush=True)
        print(f"t_final = {t:.3e} s = {t / 86400.0:.3e} days", flush=True)
        print(f"final column mass = {column_haze_mass(n, ctx):.3e} g cm^-2", flush=True)

    return n, q_recycled, t, converged, hist