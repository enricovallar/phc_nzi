import meep as mp
import argparse
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
mp.verbosity(0)

# --- DTU HPC MPI SETUP ---
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    rank = 0
    size = 1

def print_master(text):
    if rank == 0:
        print(text)
# --------------------------

param_dictionary = {
    "Peak Linearity": (0.2537, 0.3616, 0.684),
    "Right 50%": (0.2312, 0.3438, 0.5265),
    "Thesis": (0.24, 0.23444, 0.5265),
}

def run_single_sim(point_name, r1, r2, w_monitor, padding, N_periods, resolution, autoshutoff):
    """Runs a single normalization + PhC simulation and returns the target metrics."""
    T = 4
    n_InP = 3.075 * (1 + 2.7e-5 * T)
    eps_val = round(n_InP**2, 2)
    air = mp.Medium(epsilon=1)

    fcen = 0.684
    df = 0.5
    nfreq = 300
    
    d_mat = N_periods       
    d_buffer = padding      
    d_src_gap = 0.5         
    d_pml_gap = 0.5         
    dpml = 1.0              
    
    sx = dpml + d_pml_gap + d_src_gap + d_buffer + d_mat + d_buffer + d_pml_gap + dpml
    sy = 1.0                
    cell = mp.Vector3(sx, sy, 0)
    
    pml_layers = [mp.PML(dpml, direction=mp.X)] 
    pol = mp.Hz

    x_start = -d_mat / 2.0
    x_end = d_mat / 2.0
    
    refl_x = x_start - d_buffer
    trans_x = x_end + d_buffer
    src_x = refl_x - d_src_gap
    
    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=pol,
            center=mp.Vector3(src_x, 0, 0),
            size=mp.Vector3(0, sy, 0) 
        )
    ]
    
    # ---------------------------------------------------------
    # 1. NORMALIZATION RUN
    # ---------------------------------------------------------
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=[],
        sources=sources,
        resolution=resolution,
        k_point=mp.Vector3(0,0,0),
        default_material=mp.Medium(epsilon=eps_val)
    )
    
    trans_mon = sim.add_dft_fields([mp.Ey], fcen, df, nfreq, center=mp.Vector3(trans_x, 0, 0), size=mp.Vector3(0, sy, 0))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, pol, mp.Vector3(trans_x, 0, 0), 1e-6)) # Norm run doesn't need deep decay
    
    Ey_inc_trans = np.zeros(nfreq, dtype=complex)
    for i in range(nfreq):
        Ey_inc_trans[i] = np.mean(sim.get_dft_array(trans_mon, mp.Ey, i))
        
    freqs = np.linspace(fcen - df/2.0, fcen + df/2.0, nfreq)
    
    # ---------------------------------------------------------
    # 2. METASURFACE RUN
    # ---------------------------------------------------------
    sim.reset_meep()
    
    geometry = []
    for i in range(N_periods):
        cx = -N_periods/2 + i + 0.5
        geometry.append(mp.Cylinder(radius=r1, material=air, axis=mp.Vector3(0, 0, 1), center=mp.Vector3(cx, 0, 0)))
        geometry.append(mp.Cylinder(radius=r2, material=air, axis=mp.Vector3(0, 0, 1), center=mp.Vector3(cx+0.5, 0.5, 0)))
        geometry.append(mp.Cylinder(radius=r2, material=air, axis=mp.Vector3(0, 0, 1), center=mp.Vector3(cx+0.5, -0.5, 0)))
        geometry.append(mp.Cylinder(radius=r2, material=air, axis=mp.Vector3(0, 0, 1), center=mp.Vector3(cx-0.5, 0.5, 0)))
        geometry.append(mp.Cylinder(radius=r2, material=air, axis=mp.Vector3(0, 0, 1), center=mp.Vector3(cx-0.5, -0.5, 0)))
    
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        resolution=resolution,
        k_point=mp.Vector3(0,0,0),
        default_material=mp.Medium(epsilon=eps_val)
    )
    
    trans_mon = sim.add_dft_fields([mp.Ey], fcen, df, nfreq, center=mp.Vector3(trans_x, 0, 0), size=mp.Vector3(0, sy, 0))

    # Add point monitors for phase tracking
    pt_monitors = []
    x_coords = []
    for i in range(N_periods):
        cx = -N_periods/2 + i + 0.5
        mon = sim.add_dft_fields([mp.Hz], fcen, df, nfreq, center=mp.Vector3(cx, 0, 0), size=mp.Vector3(0,0,0))
        pt_monitors.append(mon)
        x_coords.append(cx)

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, pol, mp.Vector3(trans_x, 0, 0), autoshutoff))
    
    # ---------------------------------------------------------
    # 3. EXTRACT METRICS
    # ---------------------------------------------------------
    # Transmission (S21)
    Ey_tot_trans = np.zeros(nfreq, dtype=complex)
    for i in range(nfreq):
        Ey_tot_trans[i] = np.mean(sim.get_dft_array(trans_mon, mp.Ey, i))
    
    S21_mag = np.abs(Ey_tot_trans / Ey_inc_trans)

    # Point Monitor Phase
    H_pts = np.zeros((N_periods, nfreq), dtype=complex)
    for i in range(N_periods):
        for ifreq in range(nfreq):
            H_pts[i, ifreq] = sim.get_dft_array(pt_monitors[i], mp.Hz, ifreq).flat[0]
            
    # Metric A: Spatial Phase at target Dirac frequency
    target_idx = np.argmin(np.abs(freqs - w_monitor))
    spatial_phase = np.unwrap(np.angle(H_pts[:, target_idx]))
    
    # Metric B: Phase vs Frequency at the Central Monitor
    center_mon_idx = N_periods // 2
    central_freq_phase = np.unwrap(np.angle(H_pts[center_mon_idx, :]))

    return freqs, x_coords, spatial_phase, central_freq_phase, S21_mag


def run_convergence_tests(point_name, r1, r2, w_monitor, out_dir="Convergence_Results"):
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        
    # Baseline Parameters (N=5 ensures we have enough points for the spatial plot)
    base_res = 32
    base_shutoff = 1e-5
    base_pad = 3
    base_N = 7

    # Define the parameter sweeps
    sweeps = {
        "Resolution": {"param_list": [32, 48], "args": lambda val: (val, base_shutoff, base_pad)},
        "Autoshutoff": {"param_list": [1e-7, 1e-10], "args": lambda val: (base_res, val, base_pad)},
        "Padding": {"param_list": [1.0, 3.0, 5.0], "args": lambda val: (base_res, base_shutoff, val)}
    }

    for sweep_name, config in sweeps.items():
        print_master(f"\n=========================================")
        print_master(f" Starting Convergence Sweep: {sweep_name}")
        print_master(f"=========================================")
        
        results = {}
        for val in config["param_list"]:
            print_master(f" ---> Running {sweep_name} = {val}")
            
            res, shutoff, pad = config["args"](val)
            freqs, x_coords, spatial_phase, central_phase, S21 = run_single_sim(
                point_name, r1, r2, w_monitor, padding=pad, N_periods=base_N, resolution=res, autoshutoff=shutoff
            )
            
            results[val] = {
                "x_coords": x_coords,
                "spatial_phase": spatial_phase,
                "central_phase": central_phase,
                "S21": S21
            }

        # ---------------------------------------------------------
        # PLOTTING (Master Core Only)
        # ---------------------------------------------------------
        if rank == 0:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            for val in config["param_list"]:
                label_str = f"{sweep_name} = {val}"
                
                # Plot 1: Unwrapped Phase vs Space
                axes[0].plot(results[val]["x_coords"], results[val]["spatial_phase"], marker='o', label=label_str)
                
                # Plot 2: Central Phase vs Frequency
                axes[1].plot(freqs, results[val]["central_phase"], label=label_str)
                
                # Plot 3: Transmission Magnitude vs Frequency
                axes[2].plot(freqs, results[val]["S21"], label=label_str)

            axes[0].set_title(f"Spatial Phase @ f={w_monitor}")
            axes[0].set_xlabel("Unit Cell Center (X-coord)")
            axes[0].set_ylabel("Unwrapped Phase (radians)")
            axes[0].legend()
            axes[0].grid(True)

            axes[1].set_title("Central Monitor Phase vs Freq")
            axes[1].set_xlabel("Frequency")
            axes[1].set_ylabel("Unwrapped Phase (radians)")
            axes[1].legend()
            axes[1].grid(True)

            axes[2].set_title("Transmission (|S21|) vs Freq")
            axes[2].set_xlabel("Frequency")
            axes[2].set_ylabel("|S21|")
            axes[2].legend()
            axes[2].grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"Convergence_{sweep_name}.png"))
            plt.close()
            print(f"Saved Convergence_{sweep_name}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", type=str, default="Thesis", help="Name of the point to simulate")
    args = parser.parse_args()
    
    if args.point not in param_dictionary:
        raise ValueError(f"Point '{args.point}' not in param dictionary")
        
    r1, r2, w_dict_val = param_dictionary[args.point]
    
    run_convergence_tests(args.point, r1, r2, w_dict_val)