# Caligo

Caligo is a Python package for haze microphysics and optical post-processing in exoplanet atmospheres.

The package is designed to work in a PICASO-like workflow while adding a time-dependent haze microphysics model. A typical Caligo run can:

1. Build a background pressure-temperature profile with PICASO.
2. Run photochem to compute disequilibrium chemistry and photolysis rates.
3. Convert photochemical or prescribed haze production into a particle source.
4. Evolve haze particles through coagulation, diffusion, settling, boundary loss, and thermal recycling.
5. Post-process the final particle distribution into wavelength-dependent optical properties.
6. Feed those optical properties back into PICASO for transmission spectra.

## Current status

Caligo is research code under active development. The current release should be treated as an alpha development release, not a polished public package.

The current implementation includes:

* photochemical and Gaussian haze source options
* sectional particle mass bins
* compact-particle and fractal-aggregate geometry
* VIRGA-based particle settling velocities
* Brownian coagulation with optional electrostatic suppression
* optional differential-settling coagulation
* eddy diffusion and particle settling
* parameterized thermal destruction
* passive recycled-gas bookkeeping
* direct-Mie optical post-processing on the Caligo particle grid
* PICASO cloud-table generation for transmission spectra

Important scope notes:

* The thermal destruction sink is parameterized and is not a material-specific ablation model.
* The recycled-gas tracer is passive and is not chemically coupled back into photochem.
* `aggregate_effective_mie` uses a porous-sphere Maxwell-Garnett approximation. It is not a full fractal aggregate scattering calculation.
* Full PICASO and photochem workflows require external reference data and local environment variables.

## Installation

Create and activate a clean conda environment:

```bash
conda create -n caligo python=3.12
conda activate caligo
```

Install Caligo in editable mode from the repository root:

```bash
cd /path/to/Caligo
pip install -e .
```

Optional notebook/development tools can be installed with:

```bash
pip install -e ".[notebooks,dev]"
```

## External data setup

PICASO and stellar-spectrum tools may require local reference data. Before running the full notebooks, set:

```bash
export picaso_refdata="/path/to/picaso/reference"
export PYSYN_CDBS="/path/to/grp_Phoenix/redcat/trds"
```

The full GJ 1214 b tutorial also uses local photochemistry, stellar-flux, and optical-constant inputs. Paths in the notebooks should be edited for each user’s machine.

## Dependencies

The main direct dependencies are:

* `numpy`
* `scipy`
* `pandas`
* `matplotlib`
* `astropy`
* `pyyaml`
* `miepython`
* `picaso`
* `photochem`

Caligo also uses VIRGA for particle settling velocities through `virga.root_functions.vfall`.

VIRGA is not pinned directly in `pyproject.toml` because `picaso==4.0` declares its own VIRGA dependency. In a standard editable install, `picaso==4.0` installs `virga-exo==1.0`. This is expected.

The current import-tested environment uses:

* `numpy==1.26.4`
* `scipy==1.13.1`
* `pandas==2.2.3`
* `matplotlib==3.10.3`
* `astropy==7.0.1`
* `picaso==4.0`
* `photochem==0.6.5`
* `miepython==3.0.2`
* `virga==1.0.0`

The warning about a missing Vega spectrum comes from the local stellar/reference-data setup, not from Caligo itself.

## Minimal import test

After installation, run:

```bash
python - <<'PY'
import caligo
import caligo.justdoit as cdi
from caligo import background
from caligo import diagnostics
from caligo import microphysics
from caligo import optics

print("Caligo import test passed.")
print("Caligo module:", caligo.__file__)
print("Optics module:", optics.__file__)
PY
```

## Basic usage sketch

```python
from astropy import units as u
import caligo.justdoit as cdi

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

case = cdi.inputs(calculation="planet", climate=True)

case.gravity(...)
case.effective_temp(...)
case.star(opacity_ck, ...)
case.inputs_climate(...)

case.kzz(...)
case.photochem(...)

pc, Pbar_pc, z_pc, T_pc = case.run_photochem(
    find_steady_state=True,
)

case.microphysics_grid(...)
case.particles(...)
case.haze_source(...)
case.aggregates(...)
case.coagulation(...)
case.transport(...)
case.sinks(...)
case.solver(...)

source_info = case.build_haze_source()

result = case.run_microphysics(
    prepare_context=True,
    with_coagulation=True,
    verbose=True,
)
```

## Direct-Mie optical post-processing

Caligo can compute optical properties directly on the Caligo particle radius grid. This avoids forcing particles onto a narrow precomputed VIRGA `.mieff` radius grid.

```python
from pathlib import Path
from caligo import optics as copt

cloud_df, optics_out = copt.build_picaso_cloud_from_caligo(
    result=result,
    case=case,
    refractive_index_file=Path("/path/to/material.refrind"),
    optics_mode="compact_mie",
    n_wavelength=180,
    attach_photochem_atmosphere=True,
    use_virga_format=False,
    verbose=True,
)

case.clouds(df=cloud_df)

spectrum = case.spectrum(
    opacity_ck,
    calculation="transmission",
    full_output=True,
)
```

Available optical modes:

* `compact_mie`: spherical Mie using compact-equivalent Caligo radii.
* `aggregate_effective_mie`: porous-sphere effective-medium approximation.
* `aggregate_table`: planned future mode for true aggregate-scattering tables.

## Repository layout

```text
caligo/
  __init__.py
  background.py
  case.py
  diagnostics.py
  justdoit.py
  microphysics.py
  optics.py

examples/
  notebooks/
  scripts/

data/
  stellar_flux/
  chemistry/

runs/
  README.md
```

## Citation

If you use Caligo, please cite the repository and the underlying tools used in your workflow, including PICASO, photochem, VIRGA, and any optical constants used in the Mie calculations.

A formal citation file will be added before the first archived release.
