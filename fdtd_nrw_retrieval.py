import meep as mp
import argparse
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd

# The dictionary of design points
param_dictionary = {
    "Peak Linearity": (0.2537, 0.3616, 0.684),
    "Right 50%": (0.2312, 0.3438, 0.5265),
    "Thesis": (0.24, 0.23444, 0.5265),
}

def run_simulation(point_name, r1, r2, w_monitor, padding=1.5, N_periods=1, resolution=32, autoshutoff=1e-5, out_dir="FDTD_NRW"):
    os.makedirs(out_dir, exist_ok=True)
    
    T = 4
    n_InP = 3.075 * (1 + 2.7e-5 * T)
    eps_val = round(n_InP**2, 2)
    air = mp.Medium(epsilon=1)

    fcen = 0.65
    df = 0.1
    nfreq = 50
    
    d_mat = N_periods       
    d_buffer = padding      
    d_src_gap = 0.5         
    d_pml_gap = 0.5         
    dpml = 1.0              
    
    sx = dpml + d_pml_gap + d_src_gap + d_buffer + d_mat + d_buffer + d_pml_gap + dpml
    sy = 1.0                
    cell = mp.Vector3(sx, sy, 0)
    
    pml_layers = [mp.PML(dpml, direction=mp.X)] 
    pol = mp.Hz # TE polarization implies Ey and Ex fields

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
    
    # ==========================================
    # 1. NORMALIZATION RUN (Raw DFT Ey Fields)
    # ==========================================
    if mp.am_master():
        print(f"--- Running Normalization for '{point_name}' ---")
    
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=[],
        sources=sources,
        resolution=resolution,
        k_point=mp.Vector3(0,0,0),
        default_material=mp.Medium(epsilon=eps_val)
    )
    
    # Using raw DFT fields instead of mode monitors
    refl_mon = sim.add_dft_fields([mp.Ey], fcen, df, nfreq, center=mp.Vector3(refl_x, 0, 0), size=mp.Vector3(0, sy, 0))
    trans_mon = sim.add_dft_fields([mp.Ey], fcen, df, nfreq, center=mp.Vector3(trans_x, 0, 0), size=mp.Vector3(0, sy, 0))
    
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, pol, mp.Vector3(trans_x, 0, 0), 1e-15))
    
    # Extract incident Ey field by spatially averaging along the line monitor
    Ey_inc_refl = np.zeros(nfreq, dtype=complex)
    Ey_inc_trans = np.zeros(nfreq, dtype=complex)
    for i in range(nfreq):
        Ey_inc_refl[i] = np.mean(sim.get_dft_array(refl_mon, mp.Ey, i))
        Ey_inc_trans[i] = np.mean(sim.get_dft_array(trans_mon, mp.Ey, i))
        
    freqs = np.linspace(fcen - df/2.0, fcen + df/2.0, nfreq)
    
    # ==========================================
    # 2. METASURFACE RUN
    # ==========================================
    if mp.am_master():
        print(f"--- Running PhC Simulation for '{point_name}' ---")
    sim.reset_meep()
    
    pml_layers = [mp.PML(dpml, direction=mp.X)]
    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=pol,
            center=mp.Vector3(src_x, 0, 0),
            size=mp.Vector3(0, sy, 0)
        )
    ]
    
    geometry = []
    #geometry.append(mp.Block(center=mp.Vector3(0, 0, 0), size=mp.Vector3(d_mat, sy, mp.inf), material=mp.Medium(epsilon=eps_val)))
    
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
    
    refl_mon = sim.add_dft_fields([mp.Ey], fcen, df, nfreq, center=mp.Vector3(refl_x, 0, 0), size=mp.Vector3(0, sy, 0))
    trans_mon = sim.add_dft_fields([mp.Ey], fcen, df, nfreq, center=mp.Vector3(trans_x, 0, 0), size=mp.Vector3(0, sy, 0))
    hz_dft = sim.add_dft_fields([mp.Hz], w_monitor, w_monitor, 1, center=mp.Vector3(0, 0, 0), size=cell)

    pt_monitors = []
    for i in range(N_periods):
        cx = -N_periods/2 + i + 0.5
        mon = sim.add_dft_fields([mp.Hz], fcen, df, nfreq, center=mp.Vector3(cx, 0, 0), size=mp.Vector3(0,0,0))
        pt_monitors.append(mon)

    plt.figure(figsize=(12, 3))
    sim.plot2D()
    plt.title(f"Supercell for {point_name}")
    if mp.am_master():
        plt.savefig(os.path.join(out_dir, f"supercell_{point_name.replace(' ', '_')}.png"))
    plt.close()
    
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, pol, mp.Vector3(trans_x, 0, 0), autoshutoff))
    
    Ey_tot_refl = np.zeros(nfreq, dtype=complex)
    Ey_tot_trans = np.zeros(nfreq, dtype=complex)
    for i in range(nfreq):
        Ey_tot_refl[i] = np.mean(sim.get_dft_array(refl_mon, mp.Ey, i))
        Ey_tot_trans[i] = np.mean(sim.get_dft_array(trans_mon, mp.Ey, i))
        
    k0_array = 2 * np.pi * freqs
    
    # ---------------------------------------------------------
    # POINT MONITORS (Observational Only)
    # ---------------------------------------------------------
    H_pts = np.zeros((N_periods, nfreq), dtype=complex)
    for i in range(N_periods):
        for ifreq in range(nfreq):
            H_pts[i, ifreq] = sim.get_dft_array(pt_monitors[i], mp.Hz, ifreq).flat[0]

    if N_periods > 1:
        delta_phases = np.zeros((N_periods - 1, nfreq))
        for i in range(N_periods - 1):
            ratio = H_pts[i+1, :] / H_pts[i, :]
            delta_phases[i, :] = np.unwrap(np.angle(ratio))
        avg_delta_phase = np.mean(delta_phases, axis=0)
        n_eff_pt = avg_delta_phase / (k0_array * 1.0) 
    else:
        n_eff_pt = np.zeros(nfreq)

    # ---------------------------------------------------------
    # 2D Fields Plotting
    # ---------------------------------------------------------
    hz_array = sim.get_dft_array(hz_dft, mp.Hz, 0)
    hz_real = np.real(hz_array).T
    hz_imag = np.imag(hz_array).T
    hz_phase = np.angle(hz_array).T
    
    extent = [-sx/2, sx/2, -sy/2, sy/2]
    
    fig_fields, ax_fields = plt.subplots(3, 1, figsize=(12, 8))
    
    im_r = ax_fields[0].imshow(hz_real, extent=extent, origin='lower', cmap='RdBu', alpha=0.9)
    ax_fields[0].set_title(f"Re(Hz) @ w={w_monitor}")
    plt.colorbar(im_r, ax=ax_fields[0], fraction=0.046, pad=0.04)
    
    im_i = ax_fields[1].imshow(hz_imag, extent=extent, origin='lower', cmap='RdBu', alpha=0.9)
    ax_fields[1].set_title(f"Im(Hz) @ w={w_monitor}")
    plt.colorbar(im_i, ax=ax_fields[1], fraction=0.046, pad=0.04)
    
    im_p = ax_fields[2].imshow(hz_phase, extent=extent, origin='lower', cmap='hsv', alpha=0.9)
    ax_fields[2].set_title(f"Phase(Hz) @ w={w_monitor}")
    plt.colorbar(im_p, ax=ax_fields[2], fraction=0.046, pad=0.04)
    
    for ax in ax_fields:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(-sx/2, sx/2)
        ax.set_ylim(-sy/2, sy/2)

    plt.tight_layout()
    if mp.am_master():
        plt.savefig(os.path.join(out_dir, f"hz_fields_2D_{point_name.replace(' ', '_')}.png"))
    plt.close(fig_fields)

    # ==========================================
    # 3. NRW PARAMETER RETRIEVAL
    # ==========================================
    
    # Calculate Raw S-parameters from Ey fields
    # Scattered field at reflection = Total - Incident
    Ey_scat_refl = Ey_tot_refl - Ey_inc_refl
    
    S11_sim = Ey_scat_refl / Ey_inc_refl
    S21_sim = Ey_tot_trans / Ey_inc_refl
    
    d1 = x_start - refl_x  
    d2 = trans_x - x_end   
    
    S11 = S11_sim * np.exp(1j * 2 * k0_array * d1)
    S21 = S21_sim * np.exp(1j * k0_array * (d1 + d2))
    
    term1 = (1 + S11)**2 - S21**2
    term2 = (1 - S11)**2 - S21**2
    Z_raw = np.sqrt(term1 / term2)
    Z = np.where(Z_raw.real >= 0, Z_raw, -Z_raw)
    
    P_val1 = (1 - S11**2 + S21**2 + np.sqrt((1 - S11**2 + S21**2)**2 - 4*S21**2 + 0j)) / (2*S21)
    P_val2 = (1 - S11**2 + S21**2 - np.sqrt((1 - S11**2 + S21**2)**2 - 4*S21**2 + 0j)) / (2*S21)
    
    P = np.where(np.abs(P_val1) <= 1.0, P_val1, P_val2)
    P = np.where(np.abs(P) > 1.001, np.where(np.abs(P_val1) < np.abs(P_val2), P_val1, P_val2), P)
    
    n_raw = np.log(P) / (1j * k0_array * d_mat)
    n_eff = n_raw
    
    # # PURE CASCADING PHASE UNWRAPPING
    # n_real = n_raw.real.copy()
    # m_array = np.zeros_like(freqs)
    
    # if len(freqs) > 1:
    #     for i in range(1, len(freqs)):
    #         if np.isnan(n_real[i-1]) or np.isnan(n_real[i]):
    #             continue
            
    #         branch_shift = 1.0 / (freqs[i] * d_mat)
    #         n_prev = n_real[i-1]
    #         n_raw_curr = n_real[i]
            
    #         n_curr_guess = n_raw_curr + m_array[i-1] * branch_shift
    #         diff = n_prev - n_curr_guess
    #         m_step = round(diff / branch_shift)
            
    #         m_array[i] = m_array[i-1] + m_step
    #         n_real[i] = n_raw_curr + m_array[i] * branch_shift
            
    # n_eff = n_real + 1j * n_raw.imag
    # eps_eff = n_eff / Z
    # mu_eff = n_eff * Z
    
    # ==========================================
    # 4. SAVING AND PLOTTING
    # ==========================================
    df_res = pd.DataFrame({
        "frequency": freqs,
        "n_eff_real_nrw": n_eff.real,
        "n_eff_imag_nrw": n_eff.imag,
        "n_eff_real_dft": n_eff_pt,
        "Z_eff_real": Z.real,
        "Z_eff_imag": Z.imag,
        "eps_eff_real": eps_eff.real,
        "eps_eff_imag": eps_eff.imag,
        "mu_eff_real": mu_eff.real,
        "mu_eff_imag": mu_eff.imag,
        "S11_real": S11.real,
        "S11_imag": S11.imag,
        "S21_real": S21.real,
        "S21_imag": S21.imag,
        "branch_m": m_array
    })
    
    csv_path = os.path.join(out_dir, f"nrw_results_{point_name.replace(' ', '_')}.csv")
    if mp.am_master():
        df_res.to_csv(csv_path, index=False)
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
    axes[0,0].scatter(freqs, eps_eff.real, label="Re(eps)")
    axes[0,0].scatter(freqs, eps_eff.imag, '--', label="Im(eps)")
    axes[0,0].set_title("Effective Permittivity")
    axes[0,0].legend()
    
    axes[0,1].scatter(freqs, mu_eff.real, label="Re(mu)")
    axes[0,1].scatter(freqs, mu_eff.imag, '--', label="Im(mu)")
    axes[0,1].set_title("Effective Permeability")
    axes[0,1].legend()
    
    axes[1,0].scatter(freqs, n_eff.real, label="Re(n) [NRW]")
    axes[1,0].scatter(freqs, n_eff.imag, '--', label="Im(n) [NRW]")
    if N_periods > 1:
        axes[1,0].scatter(freqs, n_eff_pt, 'g:', linewidth=2.5, label="Re(n) [DFT Phase]")
    axes[1,0].set_title("Effective Refractive Index")
    axes[1,0].legend()
    
    axes[1,1].scatter(freqs, Z.real, label="Re(Z)")
    axes[1,1].scatter(freqs, Z.imag, '--', label="Im(Z)")
    axes[1,1].set_title("Effective Impedance")
    axes[1,1].legend()
    
    axes[2,0].scatter(freqs, np.abs(S11), label="|S11|")
    axes[2,0].scatter(freqs, np.abs(S21), '--', label="|S21|")
    axes[2,0].set_title("S-Parameters Magnitude")
    axes[2,0].legend()
    
    axes[2,1].scatter(freqs, np.angle(S11), label="Phase(S11)")
    axes[2,1].scatter(freqs, np.angle(S21), '--', label="Phase(S21)")
    axes[2,1].set_title("S-Parameters Phase")
    axes[2,1].legend()
    
    for ax in axes.flat:
        ax.set_xlabel("Frequency")
        ax.grid(True)
        
    plt.tight_layout()
    if mp.am_master():
        plt.savefig(os.path.join(out_dir, f"eff_params_{point_name.replace(' ', '_')}.png"))
    plt.close()
    
    if mp.am_master():
        print(f"--- Finished {point_name} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", type=str, default="Thesis", help="Name of the point to simulate")
    
    # CHANGED DEFAULT N_PERIODS TO 1 TO FIX ENZ RETRIEVAL MATH
    parser.add_argument("--n-periods", type=int, default=1, help="Number of unit cells in propagation direction")
    
    parser.add_argument("--resolution", type=int, default=64, help="Resolution for Meep FDTD")
    parser.add_argument("--autoshutoff", type=float, default=1e-7, help="Autoshutoff for Meep FDTD")    
    parser.add_argument("--w_monitor", type=float, default=None, help="Frequency of monitor")   
    parser.add_argument("--padding", type=float, default=5, help="Vacuum buffer distance between material and monitors")
    args = parser.parse_args()
    
    if args.point not in param_dictionary:
        raise ValueError(f"Point '{args.point}' not in param dictionary")
        
    r1, r2, w_dict_val = param_dictionary[args.point]
    w_monitor = args.w_monitor if args.w_monitor is not None else w_dict_val
    
    run_simulation(args.point, r1, r2, w_monitor, 
                    padding=args.padding,
                    N_periods=args.n_periods, 
                    resolution=args.resolution, 
                    autoshutoff=args.autoshutoff)