#%% Configure
## Define the materials
from phc_nzi.photonic_crystal_maker import (Material, Geometry, PhotonicCrystal, 
                                        ScriptParam, ScriptParamVector3, BaseDielectricDistribution,
                                        HexagonalLattice)
import phc_nzi.photonic_crystal_maker as phc_mkr
import meep as mp
from phc_nzi.mpb_configurator import *
from phc_nzi.simulation_handler import LSFJobConfiguration, MPBDataOptions, Simulation
from phc_nzi.optimization_data_analyzer import OptimizationDataAnalyzer
from phc_nzi.simulation_viewer import SimulationViewer
from phc_nzi.lsf_sweeper import LSFSweeper

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import FancyArrow
from matplotlib.legend_handler import HandlerPatch
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import joblib
from matplotlib.ticker import FormatStrFormatter, LogFormatterSciNotation, LogLocator
import matplotlib.colors as mcolors
import pandas as pd
from phc_nzi.mpi_bayesian_optimization import MPIBayesianOptimizator
import matplotlib.pyplot as plt
import phc_nzi.simulation_viewer as sv

# import tqdm for progress bars
from tqdm import tqdm
plt.rcParams["font.size"] = 15

T = 4
n_InP = 3.075*(1+2.7e-5*T)
print(n_InP)
eps_inp = round(n_InP**2,2)
print("Epsilon of InP: ", eps_inp)



# Simulation Name: 
name = "C4v_InP_NEW"
data_rooth = "/work3/enrva/phc_nzi_data/MPB_data"
data_path = os.path.join(data_rooth, name)
if not os.path.exists(data_path ):
    os.makedirs(data_path)


# Define the geometry
radius1 = ScriptParam("r1", 0.3)
radius2 = ScriptParam("r2", 0.2)
height_supercell = 3
height_slab = ScriptParam("h", 0.4)
slab_block_size = ScriptParamVector3(1e20, 1e20, "h", z_def=0.4)

# Define the lattice

lattice_2D = phc_mkr.SquareLattice()
lattice_slab = phc_mkr.SquareLattice(supercell_height=4)

k_points = lattice_2D.get_high_symmetry_k_points(centered_in_gamma=True)
distance = 0.1
k_points_around_gamma = lattice_2D.get_k_points_around_gamma(distance=distance)

print(k_points)

center = mp.Vector3(0,0,0)
dist = BaseDielectricDistribution(eps_bulk=eps_inp, eps_atoms=1)
r0 = phc_mkr.ScriptParam("r0", 0.3)
rd = phc_mkr.ScriptParam("rd", 0.05)
m = phc_mkr.ScriptParam("m", 6)

atoms_1 = dist.make_C4v_1a(radius=dist._radius1)
atoms_2 = dist.make_C4v_4e(radius=dist._radius2, x_dist = phc_mkr.ScriptParam("x_dist", 0.2))
                           
molecule_geometry = dist.make_superposition([atoms_1, atoms_2])
photonic_crystal_2D = PhotonicCrystal(lattice=lattice_2D, atoms=molecule_geometry)
crystal_symmetries = r"$C_{6v}:\;2b \oplus 6d$"

import numpy as np

def calculate_filling_factor_from_distribution(eps_grid, eps_th):
    """
    Calculates the filling factor from a dielectric distribution grid.
    
    Parameters:
    eps_grid (np.array): 2D array of epsilon values.
    eps_th (float): Threshold value (e.g., 1.5 if dielectric is 12 and air is 1).
    
    Returns:
    float: The filling factor f.
    """
    # Create a boolean mask where epsilon is greater than the threshold
    dielectric_mask = eps_grid > eps_th
    
    # Count the number of 'True' values and divide by total grid size
    filling_factor = np.sum(dielectric_mask) / eps_grid.size
    
    return filling_factor


dist_slab = BaseDielectricDistribution(eps_bulk=eps_inp  , eps_atoms=1, height_slab=0.5)
atoms_1_slab = dist_slab.make_C4v_1a(radius=dist_slab._radius1)
atoms_2_slab = dist_slab.make_C4v_1b(radius=dist_slab._radius2)
molecule_geometry_slab = dist_slab.make_superposition([atoms_1_slab, atoms_2_slab])
photonic_crystal_slab = PhotonicCrystal(lattice=lattice_slab, atoms=molecule_geometry_slab)

RESOLUTION = 16
PARITY_2D = "te"
POINT_GROUP = "C4v"
delta_k = 0.1
k_points_2D_optimized_values = [
    mp.Vector3(delta_k, 0, 0),                       
    mp.Vector3(0, 0, 0),                             
    mp.Vector3(0, delta_k,0)
]
k_points_2D_optimized_labels = [r"K $\leftarrow$", r"$\Gamma$", r"$\rightarrow$ M"]
k_points_2D_optimized = {"k_points_values": k_points_2D_optimized_values, "k_points_labels": k_points_2D_optimized_labels}

figure_path = os.path.abspath("/zhome/2f/7/202918/InP/pics")  
if not os.path.exists(figure_path):
    os.makedirs(figure_path)


# Dictionary of standard scientific plotting parameters
SINGLE_COLUMN_WIDTH = 3.3 # inches
DOUBLE_COLUMN_WIDTH = 6.9
GOLDEN_RATIO = 1.618  # Standard proportion for scientific plots
ROW_HEIGHT = SINGLE_COLUMN_WIDTH / GOLDEN_RATIO  # Approx 2.16 inches
publication_params = {
    # --- Figure Dimensions & Output ---
    "figure.figsize": (SINGLE_COLUMN_WIDTH, ROW_HEIGHT), # 3.5 inches is standard for a single-column width
    "figure.dpi": 300,              # High resolution for rasterized rendering
    "savefig.bbox": "tight",        # Prevents clipping of axis labels
    "savefig.pad_inches": 0.05,     # Removes excess white space margins
    
    # --- Fonts & Text ---
    "font.family": "serif",            # Standard for academic journals
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 8,                    # Set base font size to 9pt (perfect mid-range for 8-10pt rule)
    "axes.labelsize": 9,               # Axis label size
    "axes.titlesize": 9,               # Plot title size
    "xtick.labelsize": 8,              # X-axis tick label size (matches 8pt minimum)
    "ytick.labelsize": 8,              # Y-axis tick label size (matches 8pt minimum)
    "legend.fontsize": 8,              # Legend text size
    "axes.labelsize": 9,



    
    # --- Axes & Lines ---
    "axes.linewidth": 1.0,          # Thicker bounding box
    "lines.linewidth": 1.1,         # Thicker lines for visibility when printed
    "lines.markersize": 4,          # Clear, readable data markers
    
    # --- Ticks ---
    "xtick.direction": "in",        # Ticks point inward (standard in science)
    "ytick.direction": "in",
    "xtick.top": True,              # Top axis ticks visible
    "ytick.right": True,            # Right axis ticks visible
    "xtick.minor.visible": True,    # Enable minor ticks for precision
    "ytick.minor.visible": True,
    
    # --- Legend ---
    "legend.frameon": False,        # Remove the box around the legend to save ink
    
    # --- Mathtext / LaTeX ---
    "mathtext.fontset": "cm",       # Computer Modern for math rendering
    # "text.usetex": True,          # Uncomment if you have a local LaTeX engine installed!
}

# Apply the configurations globally
plt.rcParams.update(publication_params)


#%% 2D Unoptimized
SIMULATE = True
r1 = 0.0
r2 = 0.15


## Define the simulation
configuration_options_2D_unoptimized = dict(
    resolution = 25,
    num_bands=10,
    k_points=k_points["k_points_values"],
    k_points_interpolation_factor = 10,
    extra_runner_command = "fix-hfield-phase output-hfield output-efield display-symmetries-c4v",
)

mpb_configuration_2D_unoptimized=MPBSchemeConfigurator(
    photonic_crystal_2D, ["te"], **configuration_options_2D_unoptimized)

# Run or load the unoptimized 2D simulation to get the data for plotting
script =  mpb_configuration_2D_unoptimized.get_scheme_config(join_newline=True)
# make a new simulation just for this purpose, to avoid overwriting the original one with the new parameters
folder = os.path.join(data_path,  "2D", "unoptimized")
id = f"r1_{r1:0.3f}_r2_{r2:0.3f}"
simulation_2D_unoptimized = Simulation(
    simulation_name =id,
    script = script,
    directory= os.path.join(folder, id)
    )
if SIMULATE is True:
    print("2D simulation started")
    simulation_2D_unoptimized.run_hpc(mpb_command_line_params=dict(r1 = r1, r2 =r2, x_dist = 0.3), mpi = False, version = "mpb-dev/1.11.2-dev")
    print("Done.")
print("Collecting results forom ", simulation_2D_unoptimized.directory)



# -- 2. TOP RIGHT PANEL: Crystal Structure
opt_3x3 = MPBDataOptions(rectify=True, periods=1, resolution=65)
eps_3x3 = simulation_2D_unoptimized.load_and_convert_epsilon_data(options=opt_3x3)


plt.figure(dpi = 100)
# Plotting the dielectric distribution directly to avoid automatic colorbars/ticks
sv.plot_epsilon_on_ax(eps_3x3, lattice_type="square",padding=1, ws_lw = 1.5, interpolation = "spline36")

plt.xticks([]),
ax = plt.gca()
plt.xlabel([])
plt.ylabel([])  
ax.set_box_aspect(1)
ax.axis('off')


## Plot the band diagram and the unit cell
df = simulation_2D_unoptimized.load_frequency_data(PARITY_2D)
df_target_idx = df.loc[(df["k2"] == 0) & (df["k1"] >0), "k1"].idxmin()
k_idx_for_field_plot = df.loc[df_target_idx, "k index"]
plot_options = dict(linestyle='', marker='o', markersize=1)

# Create a figure with a custom grid layout
fig = plt.figure(figsize=(7, 6),constrained_layout=True)
gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3,2])
gs_top = gridspec.GridSpecFromSubplotSpec(2,3, subplot_spec=gs[0, :], wspace=0.1)
band_indices = [2,3, 4, 5,6,7,8]

gs_bottom = gridspec.GridSpecFromSubplotSpec(2,len(band_indices), 
                                             subplot_spec=gs[1, :], 
                                             wspace=0.1, 
                                             height_ratios = [1,0.2])

# Add 4 axis for fields in the gs_bottom first row
axes_fields = []
for i in range(len(band_indices)):
    ax_field = fig.add_subplot(gs_bottom[0, i])
    axes_fields.append(ax_field)
cax = fig.add_subplot(gs_bottom[1,1:3])

# -- 1. LEFT PANEL: Band Diagram --
ax_band = fig.add_subplot(gs_top[:, 0:2])
plt.sca(ax_band) # Set current axis for the viewer
plot_options = dict(linestyle='', marker='o', markersize=1)

viewer_2D = sv.SimulationViewer(simulation_2D_unoptimized)
# viewer_2D.plot_band_diagram("tm", k_points_path=k_points, color="blue", plot_options=plot_options)
viewer_2D.plot_band_diagram(PARITY_2D, k_points_path=k_points, color="red", plot_options=plot_options)
ax_band.set_ylabel(r"$\omega a/2\pi c$")
ax_band.set_title("Band diagram")
plt.ylim(0,0.8)
ymin, ymax = ax_band.get_ylim()
# ax_band.vlines(k_idx_for_field_plot, ymin, ymax, colors="gray", linestyles="--", label=None)


# -- 2. TOP RIGHT PANEL: Crystal Structure
ax_lat = fig.add_subplot(gs_top[0, 2])
plt.sca(ax_lat)
sv.plot_epsilon_on_ax(eps_3x3, lattice_type="square", ws_lw = 1.5, padding = 1)
plt.xticks([])
plt.yticks([])
plt.title(crystal_symmetries)
plt.xlabel([])
plt.ylabel([])  
ax_lat.set_box_aspect(1)
ax_lat.axis('off')


# --3. BOTTOM LEFT PANEL: Brillouin Zone with IBZ highlighted
eps = simulation_2D_unoptimized.load_and_convert_epsilon_data(options=MPBDataOptions(rectify=False))
calculate_filling_factor_from_distribution(eps, (1 + eps_inp)/2)


# --4. Field Plots

# Plot the eigen fields 
for ax in axes_fields: 
    ax.set_axis_off()


# data loading
conv_opt = MPBDataOptions(rectify=True, periods=1, transpose=False, resolution=65)
loaded_fields = {}
global_vmax = 0
try:
    eps_data = simulation_2D_unoptimized.load_and_convert_epsilon_data(options=conv_opt)
except Exception:
    eps_data = None
for b_idx in band_indices:
    fz = simulation_2D_unoptimized.load_and_convert_field_data(
        k_idx_for_field_plot, b_idx, "z", PARITY_2D, "h" if PARITY_2D == "te" else "e", conversion_options=conv_opt, overwrite=True)
    loaded_fields[b_idx] = {'z': fz, 'x': None, 'y': None}
    
    local_max = np.max(np.abs(np.real(fz)))
    if local_max > global_vmax:
        global_vmax = local_max

# Identify the Irrep of these modes
irreps = simulation_2D_unoptimized.identify_irrep_by_band_indices(band_indices, PARITY_2D, POINT_GROUP)

# Plot fields 
for ax, b_idx, irrep in zip(axes_fields, band_indices, irreps):
    data = loaded_fields[b_idx]
    im = sv.plot_field_quiver_on_ax(        
        field_z=data['z'],
        field_x=data['x'],
        field_y=data['y'],
        eps_data=eps_data,
        ax = ax, 
        lattice_type="square",
        step = 5,
        ws_lw = 1,
        padding = 1,
    )
    ax.set_title(f"${PARITY_2D.upper()}_{b_idx}$: ${irrep}$")
    ax.set_box_aspect(1)



cbar = fig.colorbar(im, cax=cax, orientation='horizontal') 
cbar.set_label(r"$H_{z}$ (real part, a.u.)")
plt.show()

# %% Optimization
def run_simulation_task(entry, which_sim = "sim", which_param_1 = "r2", which_param_2 = "x_dist", mpi = False, version = "mpb-dev/1.11.2-dev"):
    return entry[which_sim].run_hpc(
        mpb_command_line_params=dict(r1=0.0, r2=entry[which_param_1], x_dist=entry[which_param_2]), 
        mpi=mpi, 
        version=version
    )

# Gamma-only configuration (for optimization)
config_gamma_degeneracy = dict(
    resolution=RESOLUTION,
    num_bands=8,
    k_points=[mp.Vector3(0, 0, 0)],
    extra_runner_command = "display-symmetries-c4v",
)
mpb_config_gamma_degeneracy= MPBSchemeConfigurator(photonic_crystal_2D, [PARITY_2D], **config_gamma_degeneracy)
script_gamma_degeneracy = mpb_config_gamma_degeneracy.get_scheme_config(join_newline=True)


def run_sweeper_and_analyze(entry, n_r2_sweep, b_idx_top, b_idx_bot):
    
    try:
        data_old = entry["sim"].load_frequency_data(PARITY_2D)
        # Calculate gaps at Gamma (k=0)
        # Note: Ensuring we only look at k=0 if your sweep contains multiple k-points
        gamma_data = data_old[(data_old['k1'] == 0) & (data_old['k2'] == 0)]
        gap = (gamma_data[f'{PARITY_2D} band {b_idx_top}'] - gamma_data[f'{PARITY_2D} band {b_idx_bot}']).abs().values[0]
        entry["gap"] = gap
        sw = entry['sweeper']
        
        # Load or Run
        if SIMULATE:
            sw.run() 
            
        # Data Extraction
        data = sw.data if len(sw.data) > 0 else sw.load_df()
        data = sw.data if len(sw.data) > 0 else sw.load_df()

        gaps = (data[f'band_{b_idx_top}'] - data[f'band_{b_idx_bot}']).abs().values

        best = int(np.argmin(gaps))
        r2_corr = entry['r2_sweep'][best]

        # Update entry with results
        entry['gap_corrected'] = gaps[best]
        entry['r2_corrected'] = r2_corr
        entry['delta_r2'] = r2_corr - entry['r2']
        entry['edge_warning'] = (best == 0 or best == n_r2_sweep - 1)

        
        return None, None
    except Exception as e:
        return None, f"Error at r1={entry['r1']}: {str(e)}"


# Configurations for linearity analysis
configuration_options = dict(
    resolution = RESOLUTION,
    num_bands=8,
    k_points=k_points_2D_optimized["k_points_values"],
    k_points_interpolation_factor = 20,
    extra_runner_command = "fix-hfield-phase output-hfield output-efield display-symmetries-c4v display-group-velocities",
)

mpb_configuration=MPBSchemeConfigurator(
    photonic_crystal_2D, [PARITY_2D], **configuration_options)

script = mpb_configuration.get_scheme_config(join_newline=True)

# Run Bayesian Optimization
N_INIT = 10
N_BO = 0
OBJECTIVE_MODE = "log" # log or linear
USE_LSF = False
NUMPROC = 30
lsf_config = LSFJobConfiguration(num_processors=NUMPROC, span_option="ptile", span_value=7, queue="hpc",   mem = "1GB")

p1_range = (0.1, 0.2)
p2_range = (0.1, 0.3)
bo_options = {
    "dimensions": [p1_range, p2_range],
    "acq_func": "LCB",
    "acq_func_kwargs": {"kappa": 6},
    "n_initial_points": NUMPROC * N_INIT,
    "initial_point_generator": "sobol",
    "random_state": 42,
}
target_irreps = ["A_2", "E", "E"]
id = target_irreps[0] + "_" + "_".join(target_irreps[1:]) 
optimizer = MPIBayesianOptimizator(
    simulation_name=id, 
    scheme_script=script_gamma_degeneracy,
    directory=os.path.join(data_path, "optimization", id),
    maxiter=N_INIT+N_BO, 
    batch_size=NUMPROC,
    param_names=["r2", "x_dist"],
    polarization=PARITY_2D,
    symmetry_group=POINT_GROUP,
    target_irreps=target_irreps, 
    irrep_occurrences=[1, 1, 1],
    bo_options=bo_options,
    fixed_params = {"r1": 0.0},
    objective_mode = OBJECTIVE_MODE,
    target_cost = 0.0025,
    strategy = "cl_min",
    degeneracy_tol = 5e-3,
    use_mpi = USE_LSF, 
    local_workers = NUMPROC,
)


if SIMULATE is True:
    optimizer.erease_data_file()
    if USE_LSF:
        optimizer.submit_lsf_job(lsf_config, conda_env_name="mpb-nzi-env")
    else: 
        optimizer.optimize_parameters()

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.ticker import FormatStrFormatter, LogLocator, LogFormatterSciNotation
import joblib
import os


import phc_nzi.simulation_viewer as sv


# 1. Load Data and Optimizer
analyzer = OptimizationDataAnalyzer(optimizer.data_file)
analyzer.load_data()
loaded_optimizer = joblib.load(optimizer.model_file)

# Prepare predictions from the GP
gp_model = loaded_optimizer.models[-1]
valid = (analyzer.cost_vals > 0) & (~np.isnan(analyzer.cost_vals))
X_train = np.c_[analyzer.param1_vals[valid], analyzer.param2_vals[valid]]
fom_train = 1.0 / analyzer.cost_vals[valid]  # Calculate 1/C for explored points

# Setup grids for the map
r1_grid = np.linspace(p1_range[0], p1_range[1], 200)
r2_grid = np.linspace(p2_range[0], p2_range[1], 200)
R1, R2 = np.meshgrid(r1_grid, r2_grid)
grid_points = np.c_[R1.ravel(), R2.ravel()]

grid_points_transformed = loaded_optimizer.space.transform(grid_points)
gp_predictions = gp_model.predict(grid_points_transformed)

# --- DYNAMIC SCALE CONVERSION ---
if OBJECTIVE_MODE == "log":
    predicted_cost = 10**gp_predictions
else:
    predicted_cost = gp_predictions

# Calculate the Figure of Merit cleanly on a linear scale
predicted_cost_safe = np.clip(predicted_cost, a_min=1e-6, a_max=None)
Predicted_FOM_2D = (1.0 / predicted_cost_safe).reshape(R1.shape)

# --- FIND 10 OPTIMAL SAMPLES FOR 0.05 < r1 < 0.14 ---
r1_samples_1 = np.linspace(p1_range[0], p1_range[1], 5)
optimal_r2_points_1 = []
r2_search = np.linspace(p2_range[0], p2_range[1], 500)

for r1_val in r1_samples_1:
    test_points = np.c_[np.full(len(r2_search), r1_val), r2_search]
    test_transformed = loaded_optimizer.space.transform(test_points)
    
    # np.argmin works identically for both linear cost and log10(cost)
    # because log10 is a strictly monotonic function!
    costs_pred = gp_model.predict(test_transformed)
    best_r2 = r2_search[np.argmin(costs_pred)]
    optimal_r2_points_1.append([r1_val, best_r2])

branch_1 = np.array(optimal_r2_points_1)

# VMIN AND VMAX FROM PERCENTILES
vmin, vmax = np.percentile(Predicted_FOM_2D, [5, 99])
vmax = 1e3
log_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

# ---------------------------------------------------------
# Initialize Figure with 2 Subplots (Spatial 2D layouts)
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained")
fig.suptitle(f"{target_irreps}", fontsize=12)

# --- LEFT PLOT: GP Surrogate Map ---
levels = np.logspace(np.log10(vmin), np.log10(vmax), 200)
heatmap = ax1.contourf(R1, R2, Predicted_FOM_2D, levels=levels, cmap="cool", norm=log_norm, extend='both')

# Plot Sampled Points (Training data placeholder dots)
ax1.scatter(X_train[:, 0], X_train[:, 1], c='white', edgecolors='black', 
           s=10, alpha=0.4, label="Explored points")

# Plot the Optimal Samples extracted from the GP
ax1.scatter(branch_1[:, 0], branch_1[:, 1], 
           color='yellow', marker='d', s=20, edgecolors='black', 
           label=None, zorder=15)

ax1.set_xlabel("$r_1/a$")
ax1.set_ylabel("$r_2/a$")
ax1.set_title("GP Surrogate Map")
ax1.grid(alpha=0.3)
ax1.set_xlim(p1_range[0], p1_range[1])
ax1.set_ylim(p2_range[0], p2_range[1])

ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=1, framealpha=0.95, edgecolor='black')

# --- RIGHT PLOT: 2D Scatter of Explored Points (Same Map Space) ---
# X_train[:, 0] is r1, X_train[:, 1] is r2. Colored by fom_train (1/C)
scatter2 = ax2.scatter(X_train[:, 0], X_train[:, 1], c=fom_train, 
                       cmap="cool", norm=log_norm, edgecolors='black', 
                       linewidths=0.4, s=15, alpha=0.9)

ax2.set_xlabel("$r_1/a$")
ax2.set_ylabel("$r_2/a$")
ax2.set_title("Explored Points Actual $1/\mathrm{C}$")
ax2.grid(alpha=0.3)
ax2.set_xlim(p1_range[0], p1_range[1])
ax2.set_ylim(p2_range[0], p2_range[1])  # Matches the y-limits of the meshgrid/left plot

# Formatting ticks for right plot
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


# --- COLORBAR SETUP (Attached to ax2 on the far right) ---
cbar = fig.colorbar(scatter2, ax=ax2, extend='both', fraction=0.046, pad=0.05, shrink=0.8, format=LogFormatterSciNotation())
cbar.set_label(r"$\mathbf{E}[\mathrm{C}]^{-1}$")
cbar.locator = LogLocator(base=10.0, numticks=6)
cbar.update_ticks()
cbar.ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=12))

plt.show()
# %%
