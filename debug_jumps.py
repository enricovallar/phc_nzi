import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import meep as mp

sys.path.append("/zhome/2f/7/202918/phc_nzi/src")
from phc_nzi.photonic_crystal_maker import BaseDielectricDistribution, PhotonicCrystal, SquareLattice
from phc_nzi.mpb_configurator import MPBSchemeConfigurator
from phc_nzi.simulation_handler import Simulation
from phc_nzi.field_analyzer import FieldAnalyzer
from phc_nzi.simulation_viewer import SimulationViewer

lattice_2D = SquareLattice()
data_root = "/work3/enrva/phc_nzi_data/MPB_data/retrieval_debug"
T = 4
n_InP = 3.075 * (1 + 2.7e-5 * T)
eps_inp = round(n_InP**2, 2)
RESOLUTION = 32

def get_analyzer_and_df(case_idx, r1, r2, name):
    geom_2D = BaseDielectricDistribution(eps_bulk=eps_inp, eps_atoms=1).make_C4v_diatomic_B()
    photonic_crystal_2D = PhotonicCrystal(lattice=lattice_2D, atoms=geom_2D)
    sim_name = f"C4v_debug_case_{case_idx}_{name.replace(' ', '_')}"
    
    configuration_options = dict(
        resolution=RESOLUTION, num_bands=8,
        k_points=[mp.Vector3(0.1, 0), mp.Vector3(0,0)],
        k_points_interpolation_factor=50,
        extra_runner_command=("display-group-velocities fix-hfield-phase fix-efield-phase "
                              "output-hfield output-nonbloch-efield-y output-nonbloch-hfield-z")
    )
    mpb_config = MPBSchemeConfigurator(photonic_crystal_2D, ["te"], **configuration_options)
    script = mpb_config.get_scheme_config(join_newline=True)
    simulation = Simulation(simulation_name=sim_name, script=script, directory=os.path.join(data_root, sim_name))
    simulation.run_hpc(mpb_command_line_params={"r1": r1, "r2": r2})
    visualizer = SimulationViewer(simulation)
    plt.figure()
    plt.subplot(1, 2, 1)
    visualizer.plot_epsilon_2d()
    plt.subplot(1, 2, 2)
    visualizer.plot_band_diagram("te", k_points_path=["k_x", "$\Gamma$"])
    plt.savefig(f"{sim_name}_band_diagram.png")
    plt.show()
    
    analyzer = FieldAnalyzer(simulation, bands=[4, 5, 6], polarization="te", k_direction="x")
    # Call directly with true to use the updated internal field_analyzer.py logic!
    df = analyzer.get_eps_mu_impedance_neff(component_i="y", component_j="z", plot=False, enforce_continuity=True)
    return analyzer, df

# 0.2537, r2=0.3616
# 0.2179, r2=0.3153


an, df_raw = get_analyzer_and_df(0, 0.2179, 0.3153, "Valley")
df_new = df_raw.copy()

# Plot Band 4 and Band 6
fig, axs = plt.subplots(4, 1, figsize=(10, 16))

bands = [4, 6]
colors = {4: 'blue', 6: 'orange'}
markers = {4: 'o', 6: 'x'}

for band in bands:
    mask = df_new['band'] == band
    if not mask.any():
        continue
    x = df_new.loc[mask, 'frequency']

    label_re = f'Re(Band {band})'
    label_im = f'Im(Band {band})'

    # eps
    axs[0].plot(x, df_new.loc[mask, 'eps'].apply(np.real), color=colors[band], marker=markers[band], label=label_re)
    axs[0].plot(x, df_new.loc[mask, 'eps'].apply(np.imag), color=colors[band], marker=markers[band], linestyle='--', label=label_im)
    axs[0].set_title('Epsilon (eps)')
    axs[0].set_ylabel('Epsilon')

    # mu
    axs[1].plot(x, df_new.loc[mask, 'mu'].apply(np.real), color=colors[band], marker=markers[band], label=label_re)
    axs[1].plot(x, df_new.loc[mask, 'mu'].apply(np.imag), color=colors[band], marker=markers[band], linestyle='--', label=label_im)
    axs[1].set_title('Permeability (mu)')
    axs[1].set_ylabel('Mu')

    # n_eff
    axs[2].plot(x, df_new.loc[mask, 'n_eff'].apply(np.real), color=colors[band], marker=markers[band], label=label_re)
    axs[2].plot(x, df_new.loc[mask, 'n_eff'].apply(np.imag), color=colors[band], marker=markers[band], linestyle='--', label=label_im)
    axs[2].set_title('Effective Index (n_eff)')
    axs[2].set_ylabel('n_eff')

    # Z
    axs[3].plot(x, df_new.loc[mask, 'impedance'].apply(np.real), color=colors[band], marker=markers[band], label=label_re)
    axs[3].plot(x, df_new.loc[mask, 'impedance'].apply(np.imag), color=colors[band], marker=markers[band], linestyle='--', label=label_im)
    axs[3].set_title('Impedance (Z)')
    axs[3].set_ylabel('Z')
    axs[3].set_xlabel('Frequency')

for ax in axs:
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('debug_jumps.png')
print("Saved debug_jumps.png")

