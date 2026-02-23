# %%
root = r"/zhome/2f/7/202918/phc_nzi"
src = r"/zhome/2f/7/202918/phc_nzi/src"
import sys
sys.path.append(root)
sys.path.append(src)
import meep as mp

sys.path.append("/zhome/2f/7/202918/phc_nzi/src")
from phc_nzi.photonic_crystal_maker import BaseDielectricDistribution, PhotonicCrystal, SquareLattice
from phc_nzi.mpb_configurator import MPBSchemeConfigurator
from phc_nzi.simulation_handler import Simulation
from phc_nzi.field_analyzer import FieldAnalyzer
from phc_nzi.simulation_viewer import SimulationViewer
from phc_nzi.lsf_job_configurator import LSFJobConfiguration



## Define the materials
from phc_nzi.photonic_crystal_maker import (Material, Geometry, PhotonicCrystal, 
                                        ScriptParam, ScriptParamVector3, BaseDielectricDistribution,
                                        HexagonalLattice)

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as colors


T = 4
n_InP = 3.075*(1+2.7e-5*T)
print(n_InP)
eps = round(n_InP**2,2)
print("Epsilon of InP: ", eps)
InP = Material(epsilon = eps)
air = Material(epsilon = 1)
## Simulation Name: 
name = "effective_params"

## Define the geometry
radius1 = ScriptParam("r1", 0.3)
radius2 = ScriptParam("r2", 0.2)
height_supercell = 3
height_slab = ScriptParam("h", 0.4)
slab_block_size = ScriptParamVector3(1e20, 1e20, "h", z_def = 0.4)

## Define the lattice

lattice_2D = SquareLattice()
lattice_slab = SquareLattice(supercell_height= 4)

k_points = {
    "k_points_values": [
        mp.Vector3(0.013, 0, 0),
        mp.Vector3(0.2, 0, 0),
    ],
    "k_points_label": [
        "$\Gamma$",
        "$k_x$"
    ],
}
print(k_points)

geom_2D = BaseDielectricDistribution(eps_bulk = eps).make_C4v_diatomic_B()

photonic_crystal_2D = PhotonicCrystal(lattice=lattice_2D, atoms=geom_2D)



configuration_options_2D = dict(
    resolution = 64,
    num_bands=8,
    k_points=k_points["k_points_values"],
    k_points_interpolation_factor = 20,
    extra_runner_command = "fix-hfield-phase fix-efield-phase output-hfield output-nonbloch-efield-y output-nonbloch-hfield-z",
)





mpb_configuration_2D = MPBSchemeConfigurator(photonic_crystal_2D, ["te"], **configuration_options_2D)

script_2D = mpb_configuration_2D.get_scheme_config(join_newline=True)


name_2D = name  + "_2Db"



data_rooth = "/work3/enrva/phc_nzi_data/MPB_data/"

simulation_2D = Simulation(
    simulation_name=name_2D,
    script = script_2D,
    directory= os.path.join(data_rooth, name_2D)
    )

numproc= 32
pop = numproc // 2
lsf_config = LSFJobConfiguration(num_processors=numproc, span_option="block", span_value=8, queue="fotonano")


# %%
r1 = 0.24
r2 = 0.23444
simulation_2D.run_hpc(mpb_command_line_params=dict(r1 = r1, r2 = r2), mpi = True)

# %%

analyzer = FieldAnalyzer(simulation_2D, [4, 6], "te", "x")
data = analyzer.get_eps_mu_impedance_neff("y", "z", plot = True, enforce_continuity=True)



# %%
plt.figure(figsize=(20, 8)) 
plt.subplot(1, 3, 1)
analyzer.plot_eps_vs_freqs()
analyzer.plot_mu_vs_freqs()
plt.subplot(1, 3, 2)
analyzer.plot_neff_vs_freqs()
plt.subplot(1, 3, 3)
analyzer.plot_impedance_vs_freqs()
plt.tight_layout()  
plt.savefig("effective_params.png")