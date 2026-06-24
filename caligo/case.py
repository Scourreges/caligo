"""
The main Caligo case object.

CaligoCase wraps a real picaso.justdoit.inputs object and adds haze-specific
settings and run methods.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import picaso.justdoit as jdi
from astropy import units as u

from . import background
from . import diagnostics
from . import microphysics


def _to_cgs_value(value, unit, target_unit):
    """
    Convert a value/unit pair to a CGS float.

    Handles:
        value=number, unit=astropy Unit
        value=Quantity, unit=None
        value=number, unit=None
    """
    if hasattr(value, "to"):
        return value.to(target_unit).value

    if unit is None:
        return float(value)

    return (value * unit).to(target_unit).value


class CaligoCase:
    """
    PICASO-style input object with additional haze microphysics controls.

    Normal PICASO methods are forwarded to an internal PICASO inputs object.
    Haze-specific methods store settings that are later used by Caligo.
    """

    def __init__(self, *args, **kwargs):
        self.picaso = jdi.inputs(*args, **kwargs)

        # Stored user settings.
        self.photochem_settings: dict[str, Any] = {}
        self.kzz_settings: dict[str, Any] = {}
        self.microphysics_grid_settings: dict[str, Any] = {}
        self.haze_source_settings: dict[str, Any] = {}
        self.particle_settings: dict[str, Any] = {}
        self.aggregate_settings: dict[str, Any] = {}
        self.coagulation_settings: dict[str, Any] = {}
        self.transport_settings: dict[str, Any] = {}
        self.sink_settings: dict[str, Any] = {}
        self.solver_settings: dict[str, Any] = {}

        # Track whether settings have been attached to PICASO.
        self._photochem_attached = False
        self._kzz_attached = False

        # Cached run objects.
        self._opacity = None
        self._climate_out = None
        self._ctx = None
        self._haze_out = None

        # Final solver state.
        self._n = None
        self._q_recycled = None
        self._t = None
        self._converged = None
        self._hist = None
        self._result = None

        # Cached physical inputs needed by manual photochem.
        self._planet_radius_cgs = None
        self._planet_mass_cgs = None
        self._gravity_kwargs = None
        self._last_pt = None
        self._climate_pressure = None
        self._climate_temperature = None
        self._photochem_result = None
        self._source_info = None

    # ------------------------------------------------------------------
    # PICASO pass-through methods
    # ------------------------------------------------------------------

    def gravity(self, *args, **kwargs):
        """
        Forward gravity to PICASO and cache planet mass/radius in CGS for photochem.
        """
        self._gravity_kwargs = dict(kwargs)

        if "radius" in kwargs:
            self._planet_radius_cgs = _to_cgs_value(
                kwargs["radius"],
                kwargs.get("radius_unit", None),
                u.cm,
            )

        if "mass" in kwargs:
            self._planet_mass_cgs = _to_cgs_value(
                kwargs["mass"],
                kwargs.get("mass_unit", None),
                u.g,
            )

        return self.picaso.gravity(*args, **kwargs)

    def effective_temp(self, *args, **kwargs):
        return self.picaso.effective_temp(*args, **kwargs)

    def star(self, opacity, *args, **kwargs):
        self._opacity = opacity
        return self.picaso.star(opacity, *args, **kwargs)

    def guillot_pt(self, *args, **kwargs):
        """
        Forward guillot_pt to PICASO and cache the resulting P-T profile.
        """
        pt = self.picaso.guillot_pt(*args, **kwargs)
        self._last_pt = pt
        return pt

    def inputs_climate(self, *args, **kwargs):
        """
        Forward inputs_climate to PICASO and cache the pressure/temperature
        profile used for manual photochem.
        """
        if "pressure" in kwargs:
            self._climate_pressure = np.asarray(kwargs["pressure"], dtype=float)

        if "temp_guess" in kwargs:
            self._climate_temperature = np.asarray(kwargs["temp_guess"], dtype=float)

        out = self.picaso.inputs_climate(*args, **kwargs)

        # Attach Kzz if it was already specified.
        if self.kzz_settings and not self._kzz_attached:
            self._attach_kzz_to_picaso()

        return out

    def atmosphere(self, *args, **kwargs):
        return self.picaso.atmosphere(*args, **kwargs)

    def spectrum(self, *args, **kwargs):
        return self.picaso.spectrum(*args, **kwargs)

    def climate(self, opacity=None, **kwargs):
        """
        Run PICASO climate through the wrapped PICASO inputs object.

        For the current Caligo development path, this is usually not used.
        We generally use case.run_photochem() manually instead.
        """
        if opacity is None:
            opacity = self._opacity

        if opacity is None:
            raise ValueError(
                "No opacity object found. Pass opacity to case.climate(opacity), "
                "or call case.star(opacity, ...) first."
            )

        if self.photochem_settings and not self._photochem_attached:
            self._attach_photochem_to_picaso()

        if self.kzz_settings and not self._kzz_attached:
            self._attach_kzz_to_picaso()

        self._climate_out = self.picaso.climate(opacity, **kwargs)
        return self._climate_out

    def __getattr__(self, name):
        """
        Fall back to the wrapped PICASO object for methods/attributes not
        explicitly defined here.
        """
        return getattr(self.picaso, name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_picaso_atmosphere_profile(self):
        """
        Check whether PICASO has a real atmosphere profile.
        """
        try:
            profile = self.picaso.inputs["atmosphere"]["profile"]
        except Exception:
            return False

        if profile is None:
            return False

        if hasattr(profile, "columns"):
            cols = set(profile.columns)
            return "pressure" in cols and "temperature" in cols

        if isinstance(profile, dict):
            return "pressure" in profile and "temperature" in profile

        return False

    def _attach_photochem_to_picaso(self):
        """
        Attach stored photochem settings to the internal PICASO object.

        This is only needed for the full PICASO climate disequilibrium route.
        It is not needed for Caligo's manual photochem path.
        """
        if not self.photochem_settings:
            return

        if not self._has_picaso_atmosphere_profile():
            raise RuntimeError(
                "Cannot attach photochem to PICASO yet because no atmosphere "
                "profile exists. Call case.inputs_climate(...) before "
                "case.climate(...), or use case.run_photochem() for the manual "
                "photochem route."
            )

        self.picaso.atmosphere(
            mh=self.photochem_settings["metallicity"],
            cto_relative=self.photochem_settings["cto_relative"],
            chem_method=self.photochem_settings["chem_method"],
            photochem_init_args=self.photochem_settings["photochem_init_args"],
        )

        try:
            atm = self.picaso.inputs["atmosphere"]
            if "cto_absolute" in atm:
                atm["cto"] = atm["cto_absolute"]
        except Exception:
            pass

        self._photochem_attached = True

    def _attach_kzz_to_picaso(self):
        """
        Attach stored Kzz settings to the internal PICASO object.
        """
        if not self.kzz_settings:
            return

        if "atmosphere" not in self.picaso.inputs:
            return

        try:
            self.picaso.inputs["atmosphere"]["profile"]
        except Exception:
            return

        if self.kzz_settings.get("profile") is not None:
            kz = np.asarray(self.kzz_settings["profile"], dtype=float)
        elif self.kzz_settings.get("value") is not None:
            kz = float(self.kzz_settings["value"])
        else:
            pressure = self._get_pressure_for_kzz()
            kz = background.kzz_profile_from_settings(pressure, self.kzz_settings)

        self.picaso.inputs["atmosphere"]["profile"]["kz"] = kz
        self._kzz_attached = True

    # ------------------------------------------------------------------
    # Caligo setup methods
    # ------------------------------------------------------------------

    def photochem(
        self,
        mechanism_file: str,
        thermo_file: str,
        stellar_flux_file: str,
        nz: int = 120,
        P_ref: float = 1e7,
        TOA_pressure: float = 1e-4,
        metallicity: float = 100.0,
        cto_relative: float = 1.0,
        chem_method: str = "photochem+visscher",
        attach: bool = False,
        **kwargs,
    ):
        """
        Store photochem settings.
        """
        photochem_init_args = {
            "mechanism_file": mechanism_file,
            "thermo_file": thermo_file,
            "stellar_flux_file": stellar_flux_file,
            "nz": nz,
            "P_ref": P_ref,
            "TOA_pressure": TOA_pressure,
        }
        photochem_init_args.update(kwargs)

        self.photochem_settings = {
            "mechanism_file": mechanism_file,
            "thermo_file": thermo_file,
            "stellar_flux_file": stellar_flux_file,
            "nz": nz,
            "P_ref": P_ref,
            "TOA_pressure": TOA_pressure,
            "metallicity": metallicity,
            "cto_relative": cto_relative,
            "chem_method": chem_method,
            "photochem_init_args": photochem_init_args,
        }

        self._photochem_attached = False

        if attach:
            self._attach_photochem_to_picaso()

        return self

    def kzz(self, value=None, profile=None, attach: bool = True, **kwargs):
        """
        Store Kzz settings.
        """
        self.kzz_settings = {
            "value": value,
            "profile": profile,
        }
        self.kzz_settings.update(kwargs)

        self._kzz_attached = False

        if attach:
            self._attach_kzz_to_picaso()

        return self

    def _get_pressure_for_kzz(self):
        """
        Try to find the current pressure grid for building a Kzz profile.
        """
        possible_paths = [
            ("climate", "pressure"),
            ("atmosphere", "profile", "pressure"),
        ]

        for path in possible_paths:
            current = self.picaso.inputs
            try:
                for key in path:
                    current = current[key]
                return np.asarray(current, dtype=float)
            except Exception:
                continue

        if self._climate_pressure is not None:
            return np.asarray(self._climate_pressure, dtype=float)

        if self._last_pt is not None:
            return np.asarray(self._last_pt["pressure"].values, dtype=float)

        raise RuntimeError(
            "Could not find a pressure grid for Kzz. "
            "Call case.inputs_climate(...) before using piecewise case.kzz(...), "
            "or pass case.kzz(value=...) / case.kzz(profile=...)."
        )

    def microphysics_grid(self, **kwargs):
        self.microphysics_grid_settings.update(kwargs)
        return self

    def haze_source(self, **kwargs):
        self.haze_source_settings.update(kwargs)
        return self

    def particles(self, **kwargs):
        self.particle_settings.update(kwargs)
        return self

    def aggregates(self, **kwargs):
        self.aggregate_settings.update(kwargs)
        return self

    def coagulation(self, **kwargs):
        self.coagulation_settings.update(kwargs)
        return self

    def transport(self, **kwargs):
        self.transport_settings.update(kwargs)
        return self

    def sinks(self, **kwargs):
        self.sink_settings.update(kwargs)
        return self

    def solver(self, **kwargs):
        self.solver_settings.update(kwargs)
        return self

    # ------------------------------------------------------------------
    # Caligo run methods
    # ------------------------------------------------------------------

    def run_photochem(self, find_steady_state=True):
        """
        Manually run photochem using the current Caligo/PICASO background.
        """
        self._photochem_result = background.run_photochem_from_case(
            self,
            find_steady_state=find_steady_state,
        )

        return (
            self._photochem_result["pc"],
            self._photochem_result["Pbar_pc"],
            self._photochem_result["z_pc"],
            self._photochem_result["T_pc"],
        )

    def build_haze_source(self, photochem_result=None):
        """
        Build the photochemical haze source from the manual photochem result.
        """
        if photochem_result is None:
            photochem_result = self._photochem_result

        if photochem_result is None:
            raise ValueError(
                "No photochem result found. Run case.run_photochem() first."
            )

        self._source_info = background.build_photochem_haze_source_from_case(
            self,
            photochem_result=photochem_result,
        )

        return self._source_info

    def photochem_climate_out(self, photochem_result=None):
        """
        Convert a manual photochem result into the climate_out-style object
        that case.haze(...) currently expects.
        """
        if photochem_result is None:
            photochem_result = self._photochem_result

        if photochem_result is None:
            raise ValueError(
                "No photochem result found. Run case.run_photochem() first."
            )

        return background.photochem_result_to_climate_out(self, photochem_result)

    def haze(self, climate_out=None):
        """
        Build a microphysics context from a PICASO/photochem-like output.

        This does not run the time-dependent solver. It prepares the physical
        context and returns the context/scaffold haze object.
        """
        if climate_out is None:
            climate_out = self._climate_out

        if climate_out is None:
            raise ValueError(
                "No climate output found. Run out = case.climate(...), "
                "or pass case.haze(out), or use case.photochem_climate_out()."
            )

        self._ctx = background.build_context_from_case(self, climate_out)
        self._haze_out = microphysics.run_haze_model(self._ctx, self)

        return self._haze_out

    def prepare_microphysics(self, climate_out=None, with_coagulation=True):
        """
        Build and cache the microphysics context.
        """
        if climate_out is None:
            climate_out = self.photochem_climate_out()

        haze_out = self.haze(climate_out)
        ctx = haze_out["ctx"]

        if with_coagulation:
            ctx = microphysics.prepare_coagulation_context(ctx, case=self)
            self._ctx = ctx
            self._haze_out["ctx"] = ctx

        return ctx

    def _make_microphysics_result(self, n, q_recycled, t, converged, hist, ctx):
        """
        Create a clean result dictionary from a completed Caligo run.
        """
        n = np.asarray(n, dtype=float)

        mass_density = np.sum(n * ctx["m_bin"][None, :], axis=1)
        number_density = np.sum(n, axis=1)

        mass_by_bin = np.sum(
            n * ctx["m_bin"][None, :] * ctx["dz"][:, None],
            axis=0,
        )
        number_by_bin = np.sum(n * ctx["dz"][:, None], axis=0)

        column_mass = float(np.sum(mass_by_bin))
        column_number = float(np.sum(number_by_bin))

        recycled_gas_column = microphysics.recycled_gas_column(
            q_recycled,
            ctx,
        )


        if np.any(mass_density > 0):
            peak_index = int(np.argmax(mass_density))
            peak_pressure = float(ctx["Pbar"][peak_index])
        else:
            peak_index = None
            peak_pressure = np.nan

        if column_mass > 0:
            mass_weighted_radius_nm = float(
                np.sum(mass_by_bin * ctx["r_compact_nm"]) / column_mass
            )
        else:
            mass_weighted_radius_nm = np.nan

        if column_number > 0:
            number_weighted_radius_nm = float(
                np.sum(number_by_bin * ctx["r_compact_nm"]) / column_number
            )
        else:
            number_weighted_radius_nm = np.nan

        if np.any(mass_by_bin > 0):
            active_threshold = np.max(mass_by_bin) * 1e-8
            active_bins = np.where(mass_by_bin > active_threshold)[0]
        else:
            active_bins = np.array([], dtype=int)

        if active_bins.size > 0:
            active_bin_min = int(active_bins[0])
            active_bin_max = int(active_bins[-1])
            active_radius_min_nm = float(ctx["r_compact_nm"][active_bin_min])
            active_radius_max_nm = float(ctx["r_compact_nm"][active_bin_max])
        else:
            active_bin_min = None
            active_bin_max = None
            active_radius_min_nm = np.nan
            active_radius_max_nm = np.nan

        last_chunk = hist[-1] if hist else None

        result = {
            # Core solution
            "n": n,
            "number_density_bin": n,
            "q_recycled": q_recycled,
            "recycled_gas_column": recycled_gas_column,
            "time": t,
            "t_final": t,
            "converged": converged,
            "history": hist,
            "last_chunk": last_chunk,

            # Context
            "ctx": ctx,
            "Pbar": ctx["Pbar"],
            "Pdyn": ctx["Pdyn"],
            "z": ctx["z"],
            "T": ctx["T"],
            "Kzz": ctx["Kzz"],
            "mu": ctx.get("mu", None),
            "dz": ctx["dz"],

            # Particle grid
            "m_bin": ctx["m_bin"],
            "r_compact_nm": ctx["r_compact_nm"],
            "n_bin": ctx["n_bin"],
            "nz": ctx["nz"],

            # Derived haze fields
            "mass_density": mass_density,
            "number_density": number_density,
            "mass_by_bin": mass_by_bin,
            "number_by_bin": number_by_bin,
            "column_mass": column_mass,
            "column_number": column_number,

            # Compact diagnostics
            "peak_index": peak_index,
            "peak_pressure": peak_pressure,
            "mass_weighted_radius_nm": mass_weighted_radius_nm,
            "number_weighted_radius_nm": number_weighted_radius_nm,
            "active_bin_min": active_bin_min,
            "active_bin_max": active_bin_max,
            "active_radius_min_nm": active_radius_min_nm,
            "active_radius_max_nm": active_radius_max_nm,

            # Source/sink summary
            "F_src": ctx.get("F_src", np.nan),
            "F_vol_source": ctx.get("F_vol_source", np.nan),
            "F_top_source": ctx.get("F_top_source", np.nan),
            "source_info": ctx.get("source_info", None),

            # Settings snapshot
            "settings": {
                "photochem": dict(self.photochem_settings),
                "kzz": dict(self.kzz_settings),
                "microphysics_grid": dict(self.microphysics_grid_settings),
                "haze_source": dict(self.haze_source_settings),
                "particles": dict(self.particle_settings),
                "aggregates": dict(self.aggregate_settings),
                "coagulation": dict(self.coagulation_settings),
                "transport": dict(self.transport_settings),
                "sinks": dict(self.sink_settings),
                "solver": dict(self.solver_settings),
            },

            "message": "Caligo microphysics run completed.",
        }

        return result

    def run_microphysics(
        self,
        climate_out=None,
        prepare_context=True,
        with_coagulation=True,
        verbose=True,
        **solver_kwargs,
    ):
        """
        Run the full Caligo microphysics solver.

        Returns
        -------
        result : dict
            Clean result object containing the final number density, context,
            derived mass/size diagnostics, history, and settings snapshot.
        """
        if prepare_context or self._ctx is None:
            ctx = self.prepare_microphysics(
                climate_out=climate_out,
                with_coagulation=with_coagulation,
            )
        else:
            ctx = self._ctx

            if with_coagulation and "K_coag" not in ctx:
                ctx = microphysics.prepare_coagulation_context(ctx, case=self)
                self._ctx = ctx

        self._n, self._q_recycled, self._t, self._converged, self._hist = (
            microphysics.run_full_microphysics_to_steady_state(
                ctx,
                verbose=verbose,
                **solver_kwargs,
            )
        )

        self._result = self._make_microphysics_result(
            self._n,
            self._q_recycled,
            self._t,
            self._converged,
            self._hist,
            ctx,
        )

        self._haze_out = self._result

        return self._result

    def result(self):
        """
        Return the most recent Caligo microphysics result.
        """
        if self._result is None:
            raise ValueError("No result found. Run case.run_microphysics(...) first.")

        return self._result

    def summary(self, haze_out=None):
        if haze_out is None:
            haze_out = self._result if self._result is not None else self._haze_out
        return diagnostics.print_summary(self, haze_out)

    def plot(self, haze_out=None):
        if haze_out is None:
            haze_out = self._result if self._result is not None else self._haze_out
        return diagnostics.plot_results(self, haze_out)

    def copy(self):
        """
        Make a copy of the case.
        """
        return copy.deepcopy(self)