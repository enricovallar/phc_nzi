#%%

root = r"/zhome/2f/7/202918/phc_nzi"
src = r"/zhome/2f/7/202918/phc_nzi/src"
import sys
sys.path.append(root)
sys.path.append(src)

## Define the materials
from src.photonic_crystal_maker import (Material, Geometry, PhotonicCrystal, 
                                        ScriptParam, ScriptParamVector3, BaseDielectricDistribution,
                                        HexagonalLattice, SquareLattice)
import meep as mp
from src.mpb_configurator import *
from src.mpi_differential_evolution import MPIdeOptimizator
from src.simulation_handler import LSFJobConfiguration, MPBDataOptions, Simulation
from src.optimization_data_analyzer import OptimizationDataAnalyzer
from src.simulation_viewer import SimulationViewer
from src.lsf_sweeper import LSFSweeper

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as colors
plt.rcParams["font.size"] = 15

T = 4
n_InP = 3.075*(1+2.7e-5*T)
print(n_InP)
eps = round(n_InP**2,2)
print("Epsilon of InP: ", eps)
InP = Material(epsilon = eps)
air = Material(epsilon = 1)
## Simulation Name: 
name = "C4v_diatomic_T4"

## Define the geometry
radius1 = ScriptParam("r1", 0.3)
radius2 = ScriptParam("r2", 0.2)
height_supercell = 3
height_slab = ScriptParam("h", 0.4)
slab_block_size = ScriptParamVector3(1e20, 1e20, "h", z_def = 0.4)

## Define the lattice

lattice_2D = SquareLattice()



center = mp.Vector3(0,0,0)
geom = BaseDielectricDistribution(eps_bulk = eps).make_C4v_diatomic_B()
photonic_crystal = PhotonicCrystal(lattice=lattice_2D, atoms=geom)



import numpy as np

def make_kpoints_vector3(kpoints):
    """
    Convert a list of k-points to a list of Vector3 objects.

    Parameters
    ----------
    kpoints : (N, 2) ndarray
        Array of (k_x, k_y) coordinates in reduced units.

    Returns
    -------
    kpoints_vector3 : list of mp.Vector3
        List of Vector3 objects representing the k-points.
    """
    return [mp.Vector3(k[0], k[1], 0) for k in kpoints]

def generate_ibz_kpoints(nx, ny, centered=True):
    """
    Generate k-points in the irreducible Brillouin zone (IBZ) 
    of a 2D square (C4v) lattice, ensuring (0,0) is included.

    Parameters
    ----------
    nx, ny : int
        Number of divisions along k_x and k_y in the full BZ grid.
    centered : bool, default=True
        If True, use midpoints of each cell: k_i = (i+0.5)/N - 0.5.
        If False, use endpoints:       k_i = i/N       - 0.5.

    Returns
    -------
    ibz_kpts : (M, 2) ndarray
        Array of (k_x, k_y) coordinates in reduced units 
        lying in the IBZ wedge 0 ≤ k_y ≤ k_x ≤ 0.5, 
        with (0,0) always present.
    """
    # 1) Build full grid in the square BZ
    if centered:
        kx_vals = (np.arange(nx) + 0.5)/nx - 0.5
        ky_vals = (np.arange(ny) + 0.5)/ny - 0.5
    else:
        kx_vals = np.arange(nx)/nx - 0.5
        ky_vals = np.arange(ny)/ny - 0.5

    kxg, kyg = np.meshgrid(kx_vals, ky_vals, indexing='ij')
    all_pts = np.vstack([kxg.ravel(), kyg.ravel()]).T

    # 2) Select only the IBZ wedge: 0 ≤ ky ≤ kx ≤ 0.5
    mask = (all_pts[:,1] >= 0) & (all_pts[:,0] >= all_pts[:,1]) & (all_pts[:,0] <=  0.5)
    ibz_kpts = all_pts[mask]

  

    # 3) Sort (first by kx, then ky) for consistency
    ibz_kpts = ibz_kpts[np.lexsort((ibz_kpts[:,1], ibz_kpts[:,0]))]
    print("shape of kpoints: ", ibz_kpts.shape) 
    return ibz_kpts

# Generate k-points in the irreducible Brillouin zone
nx, ny = 10, 10
kpoints_array = generate_ibz_kpoints(nx, ny, centered=False)
kpoints = make_kpoints_vector3(kpoints_array)

# plot the k-points 
plt.figure(figsize=(8, 8))
plt.scatter(kpoints_array[:, 0], kpoints_array[:, 1], marker='o', color='blue', s=10)
plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)
plt.xlabel(r'$k_x$')
plt.ylabel(r'$k_y$')
plt.title('K-points in the First Brillouin Zone')
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


## Define the simulation
configuration_options= dict(
    resolution = 32, 
    num_bands=12,
    k_points=kpoints,
    extra_runner_command = "fix-hfield-phase output-hfield output-efield",
)



mpb_configuration = MPBSchemeConfigurator(photonic_crystal, ["te", "tm"], **configuration_options)
script = mpb_configuration.get_scheme_config(join_newline=True)

name = __name__

data_rooth = "/work3/s232699/phc_nzi/data/"

simulation = Simulation(
    simulation_name=name,
    script = script,
    directory= os.path.join(data_rooth, name)
    )


numproc= 32
pop = numproc // 2
lsf_config = LSFJobConfiguration(num_processors=numproc, span_option="block", span_value=8, queue="fotonano")


print(script)

simulation.run_hpc_lsf(LSFOptions=lsf_config)

viewer = SimulationViewer(simulation)
viewer.plot_epsilon_2d()


