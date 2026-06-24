"""
Caligo diagnostics and plotting utilities.
"""

from __future__ import annotations

import numpy as np


def _get_ctx(case, haze_out=None):
    """
    Retrieve the microphysics context.
    """
    if haze_out is not None and isinstance(haze_out, dict) and "ctx" in haze_out:
        return haze_out["ctx"]

    if getattr(case, "_result", None) is not None:
        result = case._result
        if isinstance(result, dict) and "ctx" in result:
            return result["ctx"]

    if getattr(case, "_ctx", None) is not None:
        return case._ctx

    if getattr(case, "_haze_out", None) is not None:
        hout = case._haze_out
        if isinstance(hout, dict) and "ctx" in hout:
            return hout["ctx"]

    return None


def _get_solution(case, haze_out=None):
    """
    Retrieve n, q_recycled, t, converged, and history.
    """
    n = None
    q_recycled = None
    t = None
    converged = None
    hist = None

    if haze_out is not None and isinstance(haze_out, dict):
        n = haze_out.get("n", haze_out.get("number_density_bin", None))
        q_recycled = haze_out.get("q_recycled", None)
        t = haze_out.get("time", haze_out.get("t_final", None))
        converged = haze_out.get("converged", None)
        hist = haze_out.get("history", None)

    if n is None and getattr(case, "_result", None) is not None:
        n = case._result.get("n", None)

    if q_recycled is None and getattr(case, "_result", None) is not None:
        q_recycled = case._result.get("q_recycled", None)

    if t is None and getattr(case, "_result", None) is not None:
        t = case._result.get("time", case._result.get("t_final", None))

    if converged is None and getattr(case, "_result", None) is not None:
        converged = case._result.get("converged", None)

    if hist is None and getattr(case, "_result", None) is not None:
        hist = case._result.get("history", None)

    if n is None:
        n = getattr(case, "_n", None)

    if q_recycled is None:
        q_recycled = getattr(case, "_q_recycled", None)

    if t is None:
        t = getattr(case, "_t", None)

    if converged is None:
        converged = getattr(case, "_converged", None)

    if hist is None:
        hist = getattr(case, "_hist", None)

    return n, q_recycled, t, converged, hist


def _column_haze_mass(n, ctx):
    if n is None or ctx is None:
        return None

    return float(np.sum(n * ctx["m_bin"][None, :] * ctx["dz"][:, None]))


def _mass_weighted_mean_radius_nm(n, ctx):
    if n is None or ctx is None:
        return None

    mass_by_bin = np.sum(n * ctx["m_bin"][None, :] * ctx["dz"][:, None], axis=0)
    total = float(np.sum(mass_by_bin))

    if total <= 0:
        return None

    return float(np.sum(mass_by_bin * ctx["r_compact_nm"]) / total)


def _number_weighted_mean_radius_nm(n, ctx):
    if n is None or ctx is None:
        return None

    num_by_bin = np.sum(n * ctx["dz"][:, None], axis=0)
    total = float(np.sum(num_by_bin))

    if total <= 0:
        return None

    return float(np.sum(num_by_bin * ctx["r_compact_nm"]) / total)


def _peak_haze_pressure(n, ctx):
    if n is None or ctx is None:
        return None

    mass_density_layer = np.sum(n * ctx["m_bin"][None, :], axis=1)

    if not np.any(mass_density_layer > 0):
        return None

    k = int(np.argmax(mass_density_layer))
    return float(ctx["Pbar"][k])


def _active_bin_range(n, ctx, rel_threshold=1e-8):
    if n is None or ctx is None:
        return None

    mass_by_bin = np.sum(n * ctx["m_bin"][None, :] * ctx["dz"][:, None], axis=0)

    if not np.any(mass_by_bin > 0):
        return None

    threshold = np.max(mass_by_bin) * rel_threshold
    active = np.where(mass_by_bin > threshold)[0]

    if active.size == 0:
        return None

    return (
        int(active[0]),
        int(active[-1]),
        float(ctx["r_compact_nm"][active[0]]),
        float(ctx["r_compact_nm"][active[-1]]),
    )


def _format_bool(x):
    if x is None:
        return "not run"
    return str(bool(x))


def _print_settings_block(title, settings):
    if not settings:
        return

    print(f"\n{title}:")
    for key, value in settings.items():
        print(f"  {key}: {value}")

        
def _recycled_gas_column(q_recycled, ctx):
    """
    Compute passive recycled-gas column mass [g cm^-2].

    q_recycled may be a scalar zero for backward compatibility or an
    altitude-dependent mass-mixing-ratio-like profile when recycling is active.
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

    return float(
        np.sum(
            q
            * np.asarray(ctx["rho_gas"], dtype=float)
            * np.asarray(ctx["dz"], dtype=float)
        )
    )

def print_summary(case, haze_out=None):
    """
    Print a concise Caligo summary.
    """
    ctx = _get_ctx(case, haze_out)
    n, q_recycled, t, converged, hist = _get_solution(case, haze_out)

    print("\n" + "=" * 70)
    print("CALIGO SUMMARY")
    print("=" * 70)

    if ctx is None:
        print("No microphysics context found.")
        return None

    Pbar = np.asarray(ctx["Pbar"], dtype=float)
    T = np.asarray(ctx["T"], dtype=float)
    Kzz = np.asarray(ctx["Kzz"], dtype=float)

    print("Context:")
    print(f"  n_pressure = {ctx.get('nz', len(Pbar))}")
    print(f"  n_bin      = {ctx.get('n_bin', 'unknown')}")
    print(f"  P range    = [{Pbar.min():.3e}, {Pbar.max():.3e}] bar")
    print(f"  T range    = [{T.min():.1f}, {T.max():.1f}] K")
    print(f"  Kzz range  = [{Kzz.min():.3e}, {Kzz.max():.3e}] cm^2/s")

    if "mu" in ctx and ctx["mu"] is not None:
        mu = np.asarray(ctx["mu"], dtype=float)
        print(f"  mu range   = [{mu.min():.3f}, {mu.max():.3f}]")

    if "v_set" in ctx:
        v = np.asarray(ctx["v_set"], dtype=float)
        print(f"  v_set max  = {np.nanmax(v):.3e} cm/s")

    if "K_coag" in ctx:
        K = np.asarray(ctx["K_coag"], dtype=float)
        print(f"  K_coag max = {np.nanmax(K):.3e} cm^3/s")

    print("\nSource:")
    print(f"  F_src        = {ctx.get('F_src', np.nan):.3e} g cm^-2 s^-1")
    print(f"  F_vol_source = {ctx.get('F_vol_source', np.nan):.3e} g cm^-2 s^-1")
    print(f"  F_top_source = {ctx.get('F_top_source', np.nan):.3e} g cm^-2 s^-1")

    if "source_info" in ctx:
        sinfo = ctx["source_info"]
        if sinfo is not None:
            if "F_photochem_raw" in sinfo:
                print(f"  F_photochem_raw = {sinfo['F_photochem_raw']:.3e} g cm^-2 s^-1")
            if "precursors" in sinfo:
                print(f"  precursors = {sinfo['precursors']}")

    print("\nSolver:")
    print(f"  converged = {_format_bool(converged)}")

    if t is not None:
        print(f"  t_final   = {float(t):.3e} s = {float(t) / 86400.0:.3e} days")

        if q_recycled is not None:
            recycled_column = _recycled_gas_column(q_recycled, ctx)

            print(
                f"  recycled gas column = "
                f"{recycled_column:.3e} g cm^-2"
            )

    col_mass = None
    if haze_out is not None and isinstance(haze_out, dict):
        col_mass = haze_out.get("column_mass", None)

    if col_mass is None:
        col_mass = _column_haze_mass(n, ctx)

    if col_mass is not None:
        print(f"  haze column mass    = {col_mass:.3e} g cm^-2")

    p_peak = None
    if haze_out is not None and isinstance(haze_out, dict):
        p_peak = haze_out.get("peak_pressure", None)

    if p_peak is None:
        p_peak = _peak_haze_pressure(n, ctx)

    if p_peak is not None and np.isfinite(p_peak):
        print(f"  peak haze pressure  = {p_peak:.3e} bar")

    r_num = None
    r_mass = None

    if haze_out is not None and isinstance(haze_out, dict):
        r_num = haze_out.get("number_weighted_radius_nm", None)
        r_mass = haze_out.get("mass_weighted_radius_nm", None)

    if r_num is None:
        r_num = _number_weighted_mean_radius_nm(n, ctx)

    if r_mass is None:
        r_mass = _mass_weighted_mean_radius_nm(n, ctx)

    if r_num is not None and np.isfinite(r_num):
        print(f"  number-weighted radius = {r_num:.3e} nm")

    if r_mass is not None and np.isfinite(r_mass):
        print(f"  mass-weighted radius   = {r_mass:.3e} nm")

    active = _active_bin_range(n, ctx)
    if active is not None:
        i0, i1, r0, r1 = active
        print(f"  active bins = {i0}–{i1} ({r0:.3e}–{r1:.3e} nm)")

    if hist:
        last = hist[-1]

        print("\nLast chunk:")
        print(f"  chunk              = {last.get('chunk')}")
        print(f"  column_mass        = {last.get('column_mass', np.nan):.3e} g cm^-2")
        print(f"  F_source           = {last.get('F_source', np.nan):.3e} g cm^-2 s^-1")
        print(f"  F_sink             = {last.get('F_sink', np.nan):.3e} g cm^-2 s^-1")
        print(f"  F_bottom           = {last.get('F_bottom', np.nan):.3e} g cm^-2 s^-1")
        print(f"  F_thermal          = {last.get('F_thermal', np.nan):.3e} g cm^-2 s^-1")
        print(f"  F_storage          = {last.get('F_storage', np.nan):.3e} g cm^-2 s^-1")
        print(f"  storage_ratio      = {last.get('storage_ratio', np.nan):.3e}")
        print(f"  balance_ratio      = {last.get('balance_ratio', np.nan):.3e}")
        print(f"  profile_change     = {last.get('profile_change', np.nan):.3e}")
        print(f"  peak_pressure      = {last.get('peak_pressure', np.nan):.3e} bar")
        print(f"  max_sed_substeps   = {last.get('max_sed_substeps', 0)}")
        print(f"  coag_substeps      = {last.get('coag_substeps', 0)}")
        print(f"  max_coag_substeps  = {last.get('max_coag_substeps', 0)}")
        print(f"  coag_active_layers = {last.get('coag_active_layers', 0)}")
        print(f"  max_coag_max_dtl   = {last.get('max_coag_max_dtl', np.nan):.3e}")

    _print_settings_block("Photochem settings", getattr(case, "photochem_settings", {}))
    _print_settings_block("Kzz settings", getattr(case, "kzz_settings", {}))
    _print_settings_block("Microphysics grid settings", getattr(case, "microphysics_grid_settings", {}))
    _print_settings_block("Haze source settings", getattr(case, "haze_source_settings", {}))
    _print_settings_block("Particle settings", getattr(case, "particle_settings", {}))
    _print_settings_block("Aggregate settings", getattr(case, "aggregate_settings", {}))
    _print_settings_block("Coagulation settings", getattr(case, "coagulation_settings", {}))
    _print_settings_block("Transport settings", getattr(case, "transport_settings", {}))
    _print_settings_block("Sink settings", getattr(case, "sink_settings", {}))
    _print_settings_block("Solver settings", getattr(case, "solver_settings", {}))

    return None


def history_table(case=None, hist=None, result=None):
    """
    Return a pandas DataFrame from a run history.
    """
    if hist is None and result is not None:
        hist = result.get("history", None)

    if hist is None and case is not None:
        if getattr(case, "_result", None) is not None:
            hist = case._result.get("history", None)

    if hist is None and case is not None:
        hist = getattr(case, "_hist", None)

    if hist is None:
        raise ValueError("No history found.")

    import pandas as pd

    return pd.DataFrame(hist)


def plot_results(case, haze_out=None):
    """
    Basic diagnostic plots.
    """
    ctx = _get_ctx(case, haze_out)
    n, _, _, _, hist = _get_solution(case, haze_out)

    if ctx is None:
        raise ValueError("No context found to plot.")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.semilogy(ctx["T"], ctx["Pbar"])
    ax.invert_yaxis()
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_title("P-T profile")
    ax.grid(alpha=0.3)
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(ctx["Kzz"], ctx["Pbar"])
    ax.invert_yaxis()
    ax.set_xlabel(r"Kzz [cm$^2$ s$^{-1}$]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_title("Kzz profile")
    ax.grid(alpha=0.3)
    plt.show()

    if n is not None:
        mass_density = np.sum(n * ctx["m_bin"][None, :], axis=1)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.loglog(np.maximum(mass_density, 1e-300), ctx["Pbar"])
        ax.invert_yaxis()
        ax.set_xlabel(r"Haze mass density [g cm$^{-3}$]")
        ax.set_ylabel("Pressure [bar]")
        ax.set_title("Final haze mass density")
        ax.grid(alpha=0.3)
        plt.show()

        mass_by_bin = np.sum(n * ctx["m_bin"][None, :] * ctx["dz"][:, None], axis=0)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(ctx["r_compact_nm"], np.maximum(mass_by_bin, 1e-300), marker="o")
        ax.set_xlabel("Compact radius [nm]")
        ax.set_ylabel(r"Column mass by bin [g cm$^{-2}$]")
        ax.set_title("Final particle size distribution")
        ax.grid(alpha=0.3)
        plt.show()

    if hist:
        t_days = np.array([h["t_end"] / 86400.0 for h in hist])
        mass = np.array([h["column_mass"] for h in hist])
        f_storage = np.array([h["F_storage"] for h in hist])
        f_sink = np.array([h["F_sink"] for h in hist])

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_days, mass, marker="o")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel(r"Column mass [g cm$^{-2}$]")
        ax.set_title("Column mass history")
        ax.grid(alpha=0.3)
        plt.show()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_days, f_storage, marker="o", label="Storage")
        ax.plot(t_days, f_sink, marker="o", label="Sink")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel(r"Flux [g cm$^{-2}$ s$^{-1}$]")
        ax.set_title("Flux balance history")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.show()

    return None




# ---------------------------------------------------------------------
# Result export helpers
# ---------------------------------------------------------------------


def save_result_npz(result, filename):
    """
    Save the main Caligo result arrays to a compressed .npz file.

    This intentionally saves numeric arrays and simple scalar diagnostics,
    not the full ctx dictionary, because ctx may contain nested objects.
    """
    import numpy as np

    required = [
        "n",
        "Pbar",
        "z",
        "T",
        "Kzz",
        "m_bin",
        "r_compact_nm",
        "mass_density",
        "number_density",
        "mass_by_bin",
        "number_by_bin",
    ]

    missing = [key for key in required if key not in result]

    if missing:
        raise KeyError(f"Result is missing required keys: {missing}")

    np.savez_compressed(
        filename,
        n=result["n"],
        Pbar=result["Pbar"],
        z=result["z"],
        T=result["T"],
        Kzz=result["Kzz"],
        mu=result["mu"] if result.get("mu") is not None else np.array([]),
        dz=result["dz"],
        m_bin=result["m_bin"],
        r_compact_nm=result["r_compact_nm"],
        mass_density=result["mass_density"],
        number_density=result["number_density"],
        mass_by_bin=result["mass_by_bin"],
        number_by_bin=result["number_by_bin"],
        column_mass=np.array(result["column_mass"]),
        column_number=np.array(result["column_number"]),
        t_final=np.array(result["t_final"]),
        converged=np.array(result["converged"]),
        peak_pressure=np.array(result["peak_pressure"]),
        mass_weighted_radius_nm=np.array(result["mass_weighted_radius_nm"]),
        number_weighted_radius_nm=np.array(result["number_weighted_radius_nm"]),
        F_src=np.array(result["F_src"]),
        F_vol_source=np.array(result["F_vol_source"]),
        F_top_source=np.array(result["F_top_source"]),
    )

    return filename


def save_history_csv(result=None, case=None, filename="caligo_history.csv"):
    """
    Save the run history to CSV.
    """
    df = history_table(case=case, result=result)
    df.to_csv(filename, index=False)
    return filename


def save_summary_txt(case, result=None, filename="caligo_summary.txt"):
    """
    Save a text summary of the latest Caligo result.
    """
    import contextlib
    import io

    if result is None:
        result = getattr(case, "_result", None)

    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        print_summary(case, result)

    with open(filename, "w") as f:
        f.write(buffer.getvalue())

    return filename


def load_result_npz(filename):
    """
    Load a saved Caligo .npz result into a lightweight dictionary.

    Note:
        This does not restore the full ctx object or case object. It is for
        post-processing saved arrays.
    """
    import numpy as np

    data = np.load(filename, allow_pickle=True)

    result = {key: data[key] for key in data.files}

    # Convert common scalar arrays back to Python scalars.
    scalar_keys = [
        "column_mass",
        "column_number",
        "t_final",
        "converged",
        "peak_pressure",
        "mass_weighted_radius_nm",
        "number_weighted_radius_nm",
        "F_src",
        "F_vol_source",
        "F_top_source",
    ]

    for key in scalar_keys:
        if key in result and np.asarray(result[key]).shape == ():
            result[key] = result[key].item()

    return result