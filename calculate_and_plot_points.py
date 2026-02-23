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

from phc_nzi.photonic_crystal_maker import BaseDielectricDistribution, PhotonicCrystal, SquareLattice
from phc_nzi.mpb_configurator import MPBSchemeConfigurator
from phc_nzi.simulation_handler import Simulation
from phc_nzi.field_analyzer import FieldAnalyzer
from phc_nzi.simulation_viewer import SimulationViewer

fig_folder = "FIGS_RETRIVED_PARAMS"
os.makedirs(fig_folder, exist_ok=True)

T = 4
n_InP = 3.075*(1+2.7e-5*T)
print("n_InP:", n_InP)
eps = round(n_InP**2, 2)
print("Epsilon of InP: ", eps)

lattice_2D = SquareLattice()

k_points = {
    "k_points_values": [
        mp.Vector3(0.0, 0, 0),
        mp.Vector3(0.05, 0, 0),
    ],
    "k_points_label": [
        r"$\Gamma$",
        "$k_x$"
    ],
}
print("k_points:", k_points)

geom_2D = BaseDielectricDistribution(eps_bulk = eps).make_C4v_diatomic_B()
photonic_crystal_2D = PhotonicCrystal(lattice=lattice_2D, atoms=geom_2D)

# Parameters to test
param_dictionary = {
    # "Left 50%": (0.2922, 0.3721),
    # "Peak Linearity": (0.2537, 0.3616),
    "Right 50%": (0.2312, 0.3438),
    # "Baseline Tail": (0.2179, 0.3153),
    # "Thesis": (0.24, 0.23444),
}

k_factor = 50
data_root = "/work3/enrva/phc_nzi_data/MPB_data/"
base_name = "eff_params_compare_points"

results = {}

configuration_options_2D = dict(
    resolution = 32,
    num_bands=8,
    k_points=k_points["k_points_values"],
    k_points_interpolation_factor = k_factor,
    extra_runner_command = "fix-hfield-phase fix-efield-phase output-hfield output-nonbloch-efield-y output-nonbloch-hfield-z",
)

mpb_configuration_2D = MPBSchemeConfigurator(photonic_crystal_2D, ["te", "tm"], **configuration_options_2D)
script_2D = mpb_configuration_2D.get_scheme_config(join_newline=True)

for point_name, (r1, r2) in param_dictionary.items():
    print(f"\n{'='*50}")
    print(f"Running for {point_name}: r1={r1}, r2={r2}")
    print(f"{'='*50}")
    
    sim_name = f"{base_name}_{point_name.replace(' ', '_').replace('%', 'm')}_2Db"
    
    simulation_2D = Simulation(
        simulation_name=sim_name,
        script = script_2D,
        directory= os.path.join(data_root, sim_name)
    )
    
    # Run simulation
    #simulation_2D.run_hpc(mpb_command_line_params=dict(r1=r1, r2=r2), mpi = True)

    # Visualize the plot of the bands
    viewer = SimulationViewer(simulation_2D)  
    plt.figure()
    viewer.plot_band_diagram("te", color = "red", k_points_path =  ["\Gamma", "k_x"])
    viewer.plot_band_diagram("tm", color = "blue", k_points_path =  ["\Gamma", "k_x"])
    df = simulation_2D.load_frequency_data("te")
    ymin, ymax = df["te band 4"].min(), df["te band 6"].max()
    plt.ylim(ymin, ymax)
    n_xticks = 5
    k_indices = np.linspace(0, len(df)-1, n_xticks)
    plt.xticks([x + 1 for x in k_indices], df["k1"][k_indices.astype(int)])
    plt.xlabel("$k_x$")
    plt.ylabel("Frequency") 
    plt.title("Band Diagram")
    plt.grid(True)


    plt.savefig(f"{fig_folder}/{sim_name}_band_diagram.png")


    plt.figure()
    viewer.plot_epsilon_2d()
    plt.savefig(f"{fig_folder}/{sim_name}_epsilon_2d.png")
    
    # Find the Dirac frequency
    fdirac = df[df["kmag/2pi"]==0.0]["te band 5"].values[0]
    # Analyze fields
    analyzer = FieldAnalyzer(simulation_2D, [4, 5, 6], "te", "x")
    data = analyzer.get_eps_mu_impedance_neff("y", "z", plot=False, enforce_continuity=False, overwrite=True)
    
    results[point_name] = (analyzer, data, fdirac)

# %%
print("Plotting results...")
plt.figure(figsize=(20, 16))

bands_to_plot = [4, 5, 6]
colors_bands = {4: "blue", 5: "green", 6: "red"}
styles_pt = ["o", "s", "^", "D", "v"]
point_names = list(param_dictionary.keys())
markersize = 2

plt.subplot(2, 2, 1)
for band in bands_to_plot:
    for i, point_name in enumerate(point_names):
        analyzer, data, fdirac = results[point_name]
        band_data = data[data["band"] == band]
        if not band_data.empty:
            plt.plot(band_data["frequency"]- fdirac, band_data["eps"].values.real, 
                     label=f'{point_name} (Band {band})', 
                     marker=styles_pt[i%len(styles_pt)], color=colors_bands[band], linestyle='-', alpha=0.7, 
                     markersize = markersize)
            plt.plot(band_data["frequency"]- fdirac, [x if abs(x) > 1e-8 else None for x in band_data["eps"].values.imag], 
                     label=f'{point_name} (Band {band})', 
                     marker=styles_pt[i%len(styles_pt)], color="orange", linestyle='--', alpha=0.7, 
                     markersize = markersize)
plt.title("Effective Permittivity (Real)")
plt.xlabel("$\omega - \omega_{dirac}$")
plt.ylabel("Epsilon")
plt.legend(fontsize='small', ncol=2)
plt.grid(True)
ylims = plt.ylim()
plt.ylim(max(ylims[0], -1), min(ylims[1], 1))

plt.subplot(2, 2, 2)
for band in bands_to_plot:
    for i, point_name in enumerate(point_names):
        analyzer, data, fdirac = results[point_name]
        band_data = data[data["band"] == band]
        if not band_data.empty:
            plt.plot(band_data["frequency"] - fdirac, band_data["mu"].values.real, 
                     label=f'{point_name} (Band {band})', 
                     marker=styles_pt[i%len(styles_pt)], color=colors_bands[band], linestyle='-', alpha=0.7 , 
                     markersize = markersize)
            plt.plot(band_data["frequency"]- fdirac, [x if abs(x) > 1e-8 else None for x in band_data["mu"].values.imag], 
                     label=f'{point_name} (Band {band})', 
                     marker=styles_pt[i%len(styles_pt)], color="orange", linestyle='--', alpha=0.7, 
                     markersize = markersize)
plt.title("Effective Permeability (Real)")
plt.xlabel("$\omega - \omega_{dirac}$")
plt.ylabel("Mu")
plt.legend(fontsize='small', ncol=2)
plt.grid(True)
ylims = plt.ylim()
plt.ylim(max(ylims[0], -1), min(ylims[1], 1))

plt.subplot(2, 2, 3)
for band in bands_to_plot:
    for i, point_name in enumerate(point_names):
        analyzer, data, fdirac = results[point_name]
        band_data = data[data["band"] == band]
        if not band_data.empty:
            plt.plot(band_data["frequency"] - fdirac, band_data["n_eff"].values.real, 
                     label=f'{point_name} (Band {band})', 
                     marker=styles_pt[i%len(styles_pt)], color=colors_bands[band], linestyle='-', alpha=0.7, 
                     markersize = markersize )
plt.title("Refractive Index (Real)")
plt.xlabel("$\omega - \omega_{dirac}$")
plt.ylabel("n_eff")
plt.legend(fontsize='small', ncol=2)
plt.grid(True)
ymin, ymax = plt.ylim()
plt.ylim(max(ymin, -1), min(ymax, 1))


plt.subplot(2, 2, 4)
for band in bands_to_plot:
    for i, point_name in enumerate(point_names):
        analyzer, data, fdirac = results[point_name]
        band_data = data[data["band"] == band]
        if not band_data.empty:
            plt.plot(band_data["frequency"] - fdirac, band_data["impedance"].values.real, 
                     label=f'{point_name} (Band {band})', 
                     marker=styles_pt[i%len(styles_pt)], color=colors_bands[band], linestyle='-', alpha=0.7, 
                     markersize = markersize)
plt.title("Impedance (Real)")
plt.xlabel("$\omega - \omega_{dirac}$")
plt.ylabel("Z")
plt.legend(fontsize='small', ncol=2)
plt.grid(True)
plt.ylim(0, min(plt.ylim()[1], 20))

plt.tight_layout()  
plt.savefig(f"{base_name}.png")
print(f"Saved {base_name}.png")
