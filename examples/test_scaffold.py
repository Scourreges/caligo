import numpy as np
import pandas as pd

import caligo.justdoit as cdi


case = cdi.inputs(calculation="planet", climate=True)
case.photochem(
    mechanism_file="photochem_rxns.yaml",
    thermo_file="photochem_thermo.yaml",
    stellar_flux_file="data/stellar_flux/GJ1214_test.txt",
    nz=120,
    P_ref=1e7,
    TOA_pressure=1e-4,
    metallicity=100.0,
    cto_relative=1.0,
    attach=False,
)
case.kzz(value=1e8, attach=False)

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

# Fake PICASO-like climate output, just to test Caligo plumbing.
pressure = np.logspace(1, -8, 80)
temperature = 600 + 100 * np.tanh(np.log10(pressure))

fake_out = {
    "ptchem_df": pd.DataFrame(
        {
            "pressure": pressure,
            "temperature": temperature,
            "Kzz": np.ones_like(pressure) * 1e8,
        }
    )
}

haze = case.haze(fake_out)

case.summary(haze)
case.plot(haze)