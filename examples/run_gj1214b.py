"""
Example Caligo run for a GJ 1214b-like setup.

Run from the Caligo project root:

    python examples/run_gj1214b.py
"""

import caligo.justdoit as cdi
import picaso.photochem as picasochem

from astropy import constants as const
from astropy import units as u


# -------------------------------------------------------------------
# Photochem files
# -------------------------------------------------------------------

picasochem.generate_photochem_rx_and_thermo_files()


# -------------------------------------------------------------------
# Opacity setup
# -------------------------------------------------------------------

gases_fly = [
    "H2O",
    "NH3",
    "CO2",
    "N2",
    "HCN",
    "H2",
    "C2H2",
    "C2H4",
    "C2H6",
]

opacity_ck = cdi.opannection(
    method="resortrebin",
    preload_gases=gases_fly,
)


# -------------------------------------------------------------------
# Caligo / PICASO case
# -------------------------------------------------------------------

case = cdi.inputs(calculation="planet", climate=True)

case.gravity(
    radius=(0.2438 * const.R_jup).to(u.cm).value,
    radius_unit=u.cm,
    mass=(0.0265 * const.M_jup).to(u.g).value,
    mass_unit=u.g,
)

case.effective_temp(30.0)

case.star(
    opacity_ck,
    temp=3101.0,
    metal=0.24,
    logg=5.0286,
    radius=0.2162,
    radius_unit=u.R_sun,
    semi_major=0.01505,
    semi_major_unit=u.AU,
    database="phoenix",
)

pt = case.guillot_pt(
    567.0,
    nlevel=141,
    T_int=30.0,
    p_bottom=3.0,
    p_top=-7.0,
)

case.inputs_climate(
    temp_guess=pt["temperature"].values,
    pressure=pt["pressure"].values,
    nstr=[0, 120, 139, 0, 0, 0],
    nofczns=1,
    rfacv=0.5,
)

case.photochem(
    mechanism_file="photochem_rxns.yaml",
    thermo_file="photochem_thermo.yaml",
    stellar_flux_file="data/stellar_flux/GJ1214_test.txt",
    nz=120,
    P_ref=1e7,
    TOA_pressure=1e-4,
    metallicity=100.0,
    cto_relative=1.0,
)

case.kzz(
    upper=1e9,
    mid=1e8,
    deep=1e8,
    break_top_bar=1e-5,
    break_deep_bar=30.0,
)

case.microphysics_grid(
    nz_target=50,
    p_top_micro=1e-10,
    p_bot_micro=10.0,
)

case.haze_source(
    vertical_mode="photochem",
    parents=["HCN", "C2H2"],
    yield_model="carbon_mass_only",
    yield_haze=0.10,
    target_haze_flux=1e-13,
)

case.particles(
    n_bin=40,
    rho_haze=1.0,
    r_monomer_nm=1.0,
    bin_mass_ratio=2.0,
)

case.aggregates(
    use=True,
    df_fixed=2.0,
    r_primary_nm=10.0,
    pressure_dependent=True,
    transition_p_bar=1e-6,
)

case.coagulation(
    use=True,
    brownian=True,
    differential_settling=False,
    charge_suppression=True,
    charge_density_e_per_um=15.0,
)

case.transport(
    settling=True,
    bottom_bc="deposition_velocity",
)

case.sinks(
    thermal_model="finite_rate_smooth_T",
    therm_destroy_t_on_k=650.0,
    therm_destroy_width_k=100.0,
    therm_destroy_timescale=1e5,
)

import pandas as pd
import numpy as np

pressure = pt["pressure"].values
temperature = pt["temperature"].values

fake_out = {
    "ptchem_df": pd.DataFrame(
        {
            "pressure": pressure,
            "temperature": temperature,
            "Kzz": case.picaso.inputs["atmosphere"]["profile"]["kz"],
        }
    )
}

haze = case.haze(fake_out)



case.plot(haze)
case.summary(haze)