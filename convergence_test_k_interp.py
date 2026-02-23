# %%
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

root = r"/zhome/2f/7/202918/phc_nzi"
src = r"/zhome/2f/7/202918/phc_nzi/src"
sys.path.append(root)
sys.path.append(src)
import meep as mp

from phc_nzi.photonic_crystal_maker import BaseDielectricDistribution, PhotonicCrystal, SquareLattice, Material, ScriptParam, ScriptParamVector3
from phc_nzi.mpb_configurator import MPBSchemeConfigurator
from phc_nzi.simulation_handler import Simulation
from phc_nzi.field_analyzer import FieldAnalyzer
from phc_nzi.lsf_job_configurator import LSFJobConfiguration

T = 4
n_InP = 3.075*(1+2.7e-5*T)
print("n_InP:", n_InP)
eps = round(n_InP**2, 2)
print("Epsilon of InP: ", eps)

lattice_2D = SquareLattice()

k_points = {
    "k_points_values": [
        mp.Vector3(0.001, 0, 0),
        mp.Vector3(0.2, 0, 0),
    ],
    "k_points_label": [
        r"$\Gamma$",
        "$k_x$"
    ],
}
print("k_points:", k_points)

geom_2D = BaseDielectricDistribution(eps_bulk = eps).make_C4v_diatomic_B()
photonic_crystal_2D = PhotonicCrystal(lattice=lattice_2D, atoms=geom_2D)

# [0] Left 50%        | r1=0.2922, r2=0.3721
# [1] Peak Linearity  | r1=0.2537, r2=0.3616
# [2] Right 50%       | r1=0.2312, r2=0.3438
# [3] Baseline Tail   | r1=0.2179, r2=0.3153
# [4] Thesis          | r1 = 0.24, r2 = 0.23444

param_dictionary = {
    "left_50%": (0.2922, 0.3721),
    "peak_linearity": (0.2537, 0.3616),
    "right_50%": (0.2312, 0.3438),
    "baseline_tail": (0.2179, 0.3153),
    "thesis": (0.24, 0.23444),
}

k_interp_factors = [100]
data_root = "/work3/enrva/phc_nzi_data/MPB_data/"
base_name = "convergence_k_interp"

results = {}

for k_factor in k_interp_factors:
    print(f"\n{'='*50}")
    print(f"Running for k_points_interpolation_factor = {k_factor}")
    print(f"{'='*50}")
    
    configuration_options_2D = dict(
        resolution = 32,
        num_bands=8,
        k_points=k_points["k_points_values"],
        k_points_interpolation_factor = k_factor,
        extra_runner_command = "fix-hfield-phase fix-efield-phase output-hfield output-nonbloch-efield-y output-nonbloch-hfield-z",
    )

    mpb_configuration_2D = MPBSchemeConfigurator(photonic_crystal_2D, ["te"], **configuration_options_2D)
    script_2D = mpb_configuration_2D.get_scheme_config(join_newline=True)
    
    point = "right_50%"
    r1 = param_dictionary[point][0]
    r2 = param_dictionary[point][1]
    print(f"Running for {point}: r1={r1}, r2={r2}") 
    sim_name = f"{base_name}_{point}_2Db_f{k_factor}"
    
    simulation_2D = Simulation(
        simulation_name=sim_name,
        script = script_2D,
        directory= os.path.join(data_root, sim_name)
    )
    
    # Run simulation
    simulation_2D.run_hpc(mpb_command_line_params=dict(r1 = r1, r2 = r2), mpi = True)
    
    # Analyze fields
    analyzer = FieldAnalyzer(simulation_2D, [4, 5, 6], "te", "x")
    data = analyzer.get_eps_mu_impedance_neff("y", "z", plot=False, enforce_continuity=False, overwrite=True)
    
    results[k_factor] = (analyzer, data)

# %%
print("Plotting convergence results...")
plt.figure(figsize=(20, 16))

bands_to_plot = [4, 5, 6]
colors_bands = {4: "blue", 5: "green", 6: "red"}
styles_k = ["o", "s", "^", "D"]

for band in bands_to_plot:
    # Plot Effective Permittivity
    plt.subplot(2, 2, 1)
    for i, (k_factor, (analyzer, data)) in enumerate(results.items()):
        band_data = data[data["band"] == band]
        if not band_data.empty:
            plt.plot(band_data["frequency"], band_data["eps"].values.real, 
                     label=f'Band {band}, k={k_factor}', 
                     marker=styles_k[i%len(styles_k)], color=colors_bands[band], linestyle='-', alpha=0.7)
            # plot imaginary part
            plt.plot(band_data["frequency"], [x if abs(x) > 1e-8 else None for x in band_data["eps"].values.imag], 
                     label=f'Band {band}, k={k_factor}', 
                     marker=styles_k[i%len(styles_k)], color=colors_bands[band], linestyle='--', alpha=0.7)
    plt.title("Effective Permittivity (Real)")
    plt.xlabel("Frequency")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.grid(True)
    plt.ylim(-1,1)
    
    # Plot Effective Permeability
    plt.subplot(2, 2, 2)
    for i, (k_factor, (analyzer, data)) in enumerate(results.items()):
        band_data = data[data["band"] == band]
        if not band_data.empty:
            plt.plot(band_data["frequency"], band_data["mu"].values.real, 
                     label=f'Band {band}, k={k_factor}', 
                     marker=styles_k[i%len(styles_k)], color=colors_bands[band], linestyle='-', alpha=0.7)
            # plot imaginary part
            plt.plot(band_data["frequency"], [x if abs(x) > 1e-8 else None for x in band_data["mu"].values.imag], 
                     label=f'Band {band}, k={k_factor}', 
                     marker=styles_k[i%len(styles_k)], color=colors_bands[band], linestyle='--', alpha=0.7)
    plt.title("Effective Permeability (Real)")
    plt.xlabel("Frequency")
    plt.ylabel("Mu")
    plt.legend()
    plt.grid(True)
    plt.ylim(-1,1)  
    
    # Plot Refractive Index
    plt.subplot(2, 2, 3)
    for i, (k_factor, (analyzer, data)) in enumerate(results.items()):
        band_data = data[data["band"] == band]
        if not band_data.empty:
            plt.plot(band_data["frequency"], band_data["n_eff"].values.real, 
                     label=f'Band {band}, k={k_factor}', 
                     marker=styles_k[i%len(styles_k)], color=colors_bands[band], linestyle='-', alpha=0.7)
    plt.title("Refractive Index (Real)")
    plt.xlabel("Frequency")
    plt.ylabel("n_eff")
    plt.legend()
    plt.grid(True)
    plt.ylim(-1,1)  
    
    # Plot Impedance
    plt.subplot(2, 2, 4)
    for i, (k_factor, (analyzer, data)) in enumerate(results.items()):
        band_data = data[data["band"] == band]
        if not band_data.empty:
            plt.plot(band_data["frequency"], band_data["impedance"].values.real, 
                     label=f'Band {band}, k={k_factor}', 
                     marker=styles_k[i%len(styles_k)], color=colors_bands[band], linestyle='-', alpha=0.7)
    plt.title("Impedance (Real)")
    plt.xlabel("Frequency")
    plt.ylabel("Z")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 10)

plt.tight_layout()  
plt.savefig(f"{base_name}.png")
print(f"Saved {base_name}.png")
