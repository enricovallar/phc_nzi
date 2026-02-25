import meep as mp
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd

mp.verbosity(0)

# -------------------------------------------------------------------------
# 1. PARAMETERS & DICTIONARY
# -------------------------------------------------------------------------
param_dictionary = {
    "Peak Linearity": (0.2537, 0.3616, 0.684),
    "Right 50%": (0.2312, 0.3438, 0.5265),
    "Thesis": (0.24, 0.23444, 0.5265),
}

# =========================================================================
# Point-Monitor Extraction Function
# =========================================================================
def extract_parameters_point_monitors(freqs, S11, S21, d_mat, n_points, n_bg=1.0):
    """
    Parameter extraction using center-cell point monitors for branch selection.
    Enforces Re(Z) > 0 and Im(n) > 0.
    """
    k0 = 2 * np.pi * freqs
    kd = k0 * d_mat

    # 1. Wave Impedance strictly enforcing Passivity (Real(z) >= 0)
    z_norm_sq = ((1 + S11)**2 - S21**2) / ((1 - S11)**2 - S21**2)
    z_norm = np.sqrt(z_norm_sq)
    z_norm = np.where(np.real(z_norm) < 0, -z_norm, z_norm)
    z_eff = z_norm / n_bg

    # 2. Principal Branch Refractive Index (m=0) 
    arg_val = S21 / (1 - S11 * ((z_norm - 1) / (z_norm + 1)))
    phase_arg = np.angle(arg_val)
    n_principal = (1.0 / kd) * phase_arg
    
    # 3. Imaginary part of n enforcing Im(n) > 0
    n_imag = (-1.0 / kd) * np.real(np.log(arg_val + 0j))
    n_imag = np.abs(n_imag)  # Force positive imaginary part

    # 4. Branch Selection using n_points from DFT monitors
    # We find the integer 'm' that makes n_corrected closest to n_points
    m_float = (n_points - n_principal) * kd / (2 * np.pi)
    m_arr = np.round(m_float).astype(int)

    # 5. Reconstruct Constitutive Parameters
    n_corrected = n_principal + (2 * m_arr * np.pi) / kd
    n_eff_complex = n_corrected + 1j * n_imag
    
    eps_eff = n_eff_complex / z_eff
    mu_eff = n_eff_complex * z_eff
    
    return n_principal, n_corrected, m_arr, eps_eff, mu_eff, z_eff


def run_simulation(point_name, r1, r2, w_monitor, padding=2.0, N_periods=5, resolution=64, autoshutoff=1e-8, out_dir="FDTD_NRW_V3"):
    os.makedirs(out_dir, exist_ok=True)
    
    # Material setup
    n_bg = 3.075 
    eps_bg = n_bg**2
    air = mp.Medium(epsilon=1)

    # Frequency setup
    fcen = w_monitor
    df = 0.05
    nfreq = 100
    freqs = np.linspace(fcen - df/2, fcen + df/2, nfreq)
    
    # Geometry spacing
    d_mat = N_periods       
    d_buffer = padding      
    d_src_gap = 0.5         
    d_pml_gap = 0.5         
    dpml = 1.0              
    
    sx = 2*dpml + 2*d_pml_gap + 2*d_src_gap + 2*d_buffer + d_mat
    sy = 1.0                
    cell = mp.Vector3(sx, sy, 0)
    pml_layers = [mp.PML(dpml, direction=mp.X)] 
    
    x_start = -d_mat / 2.0
    x_end = d_mat / 2.0
    refl_x = x_start - d_buffer
    trans_x = x_end + d_buffer
    src_x = refl_x - d_src_gap

    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), 
                         component=mp.Ey,
                         center=mp.Vector3(src_x, 0, 0), 
                         size=mp.Vector3(0, sy, 0))]

    # =========================================================================
    # 2. NORMALIZATION RUN (Reference incident phase and flux)
    # =========================================================================
    if mp.am_master(): print(f"--- Running Normalization for '{point_name}' ---")
    
    sim = mp.Simulation(
        cell_size=cell, boundary_layers=pml_layers, geometry=[],
        sources=sources, resolution=resolution, default_material=mp.Medium(epsilon=eps_bg),
        k_point=mp.Vector3(0,0,0) 
    )
    
    # Monitors
    refl_fr = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(refl_x, 0, 0), size=mp.Vector3(0, sy, 0)))
    trans_fr = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(trans_x, 0, 0), size=mp.Vector3(0, sy, 0)))
    refl_mon_ref = sim.add_dft_fields([mp.Ey], fcen, df, nfreq, center=mp.Vector3(refl_x, 0, 0), size=mp.Vector3(0, sy, 0))
    trans_mon_ref = sim.add_dft_fields([mp.Ey], fcen, df, nfreq, center=mp.Vector3(trans_x, 0, 0), size=mp.Vector3(0, sy, 0))
    
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ey, mp.Vector3(trans_x, 0, 0), 1e-12))
    
    inc_flux = np.array(mp.get_fluxes(trans_fr))
    
    Ey_inc_refl = np.array([np.mean(sim.get_dft_array(refl_mon_ref, mp.Ey, i)) for i in range(nfreq)])
    Ey_inc_trans = np.array([np.mean(sim.get_dft_array(trans_mon_ref, mp.Ey, i)) for i in range(nfreq)])

    # =========================================================================
    # 3. METASURFACE RUN
    # =========================================================================
    if mp.am_master(): print(f"--- Running PhC Simulation for '{point_name}' ---")
    sim.reset_meep()
    
    geometry = []
    for i in range(N_periods):
        cx = -N_periods/2 + i + 0.5
        geometry.append(mp.Cylinder(radius=r1, material=air, center=mp.Vector3(cx, 0, 0)))
        for dy in [-0.5, 0.5]:
            geometry.append(mp.Cylinder(radius=r2, material=air, center=mp.Vector3(cx+0.5, dy, 0)))
            geometry.append(mp.Cylinder(radius=r2, material=air, center=mp.Vector3(cx-0.5, dy, 0)))

    sim = mp.Simulation(
        cell_size=cell, boundary_layers=pml_layers, geometry=geometry,
        sources=sources, resolution=resolution, default_material=mp.Medium(epsilon=eps_bg),
        k_point=mp.Vector3(0,0,0)
    )

    refl_fr = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(refl_x, 0, 0), size=mp.Vector3(0, sy, 0)))
    trans_fr = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(trans_x, 0, 0), size=mp.Vector3(0, sy, 0)))
    refl_mon = sim.add_dft_fields([mp.Ey], fcen, df, nfreq, center=mp.Vector3(refl_x, 0, 0), size=mp.Vector3(0, sy, 0))
    trans_mon = sim.add_dft_fields([mp.Ey], fcen, df, nfreq, center=mp.Vector3(trans_x, 0, 0), size=mp.Vector3(0, sy, 0))

    # --- Add DFT Point Monitors at the center of each unit cell ---
    cell_center_monitors = []
    for i in range(N_periods):
        cx = -N_periods/2 + i + 0.5
        mon = sim.add_dft_fields([mp.Ey], fcen, df, nfreq, center=mp.Vector3(cx, 0, 0), size=mp.Vector3(0, 0, 0))
        cell_center_monitors.append(mon)

    # --- 2D DFT Monitor for Hz Phase Mapping ---
    dft_2d_size = mp.Vector3(d_mat + 2*d_buffer, sy, 0)
    dft_resolution = 32
    decimation_factor = int(resolution / dft_resolution)
    hz_2d_mon = sim.add_dft_fields([mp.Hz], fcen, 0, 1, center=mp.Vector3(0,0,0), size=dft_2d_size, decimation_factor=decimation_factor)

    # meep plots
    mp.plot2D(sim)
    plt.savefig(os.path.join(out_dir, f"geometry_{point_name}.png"), dpi=150)
    plt.close()

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ey, mp.Vector3(trans_x, 0, 0), autoshutoff))

    # =========================================================================
    # 4. GRATING PROJECTION & LOCAL PHASE EXTRACTION
    # =========================================================================
    T_total = np.array(mp.get_fluxes(trans_fr)) / inc_flux
    R_total = -np.array(mp.get_fluxes(refl_fr)) / inc_flux

    S11_zeroth = np.zeros(nfreq, dtype=complex)
    S21_zeroth = np.zeros(nfreq, dtype=complex)
    
    # Extract Point Monitor fields (N_periods x nfreq)
    E_centers = np.zeros((N_periods, nfreq), dtype=complex)

    for i in range(nfreq):
        # Grating projection
        S11_zeroth[i] = (np.mean(sim.get_dft_array(refl_mon, mp.Ey, i)) - Ey_inc_refl[i]) / Ey_inc_refl[i]
        S21_zeroth[i] = np.mean(sim.get_dft_array(trans_mon, mp.Ey, i)) / Ey_inc_trans[i]
        
        # Point monitors
        for j in range(N_periods):
            # Safe extraction: grab the single field point as a scalar
            E_centers[j, i] = sim.get_dft_array(cell_center_monitors[j], mp.Ey, i).flatten()[0]

    T_zeroth = np.abs(S21_zeroth)**2
    Diffraction_Loss = T_total - T_zeroth

    # --- Calculate n_points from local unit cell phase ---
    n_points = np.zeros(nfreq)
    for i in range(nfreq):
        # Unwrap phase along the propagation direction to handle > pi jumps robustly
        spatial_phases = np.unwrap(np.angle(E_centers[:, i]))
        # Average phase difference between consecutive cells. Since dx=1, average dphi is just:
        avg_dphase = (spatial_phases[-1] - spatial_phases[0]) / (N_periods - 1)
        k0_dx = 2 * np.pi * freqs[i] * 1.0  # Period a = 1.0
        n_points[i] = avg_dphase / k0_dx

    # =========================================================================
    # 5. PARAMETER RETRIEVAL (Phase Roll-back & Monitor Extraction)
    # =========================================================================
    k_bg = 2 * np.pi * freqs * n_bg
    S11 = S11_zeroth * np.exp(2j * k_bg * d_buffer)
    S21 = S21_zeroth * np.exp(1j * k_bg * (2 * d_buffer))

    n_m0, n_eff_final, m_arr, eps_eff, mu_eff, Z_eff = extract_parameters_point_monitors(
        freqs, S11, S21, d_mat, n_points, n_bg
    )

    # =========================================================================
    # 6. SAVE & PLOT
    # =========================================================================
    if mp.am_master():
        hz_data = sim.get_dft_array(hz_2d_mon, mp.Hz, 0)
        hz_phase = np.angle(hz_data)

        plt.figure(figsize=(12, 4))
        plt.imshow(hz_phase.transpose(), interpolation='spline36', cmap='RdBu', 
                   extent=[-dft_2d_size.x/2, dft_2d_size.x/2, -sy/2, sy/2])
        plt.axvline(x=-d_mat/2, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=d_mat/2, color='k', linestyle='--', alpha=0.5)
        plt.colorbar(label='Phase (rad)')
        plt.title(f"Hz Phase Profile: {point_name} at f={fcen:.4f}")
        plt.xlabel("x (a)")
        plt.ylabel("y (a)")
        plt.savefig(os.path.join(out_dir, f"hz_phase_map_{point_name}.png"), dpi=150)
        plt.close()

    res_path = os.path.join(out_dir, f"results_{point_name}.csv")
    pd.DataFrame({
        "freq": freqs, "T_total": T_total, "T_zeroth": T_zeroth,
        "n_points": n_points, "n_m0": n_m0, "n_eff": n_eff_final, "m": m_arr,
        "eps_real": eps_eff.real, "eps_imag": eps_eff.imag,
        "mu_real": mu_eff.real, "mu_imag": mu_eff.imag,
        "z_real": Z_eff.real, "z_imag": Z_eff.imag,
    }).to_csv(res_path)

    # Extraction Plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"PhC Extraction (Point Monitors): {point_name}")

    axs[0,0].plot(freqs, n_m0, label='Principal (m=0)')
    axs[0,0].plot(freqs, n_points, 'rx', label='n_points (Monitors)', markersize=5)
    axs[0,0].plot(freqs, n_eff_final, 'k--', label='Final n_eff')
    axs[0,0].set_ylabel("Real(n_eff)")
    axs[0,0].set_xlabel("Frequency (c/a)")
    axs[0,0].legend(); axs[0,0].grid(True)
    
    axs[0,1].plot(freqs, m_arr, 'r-o', label='Selected Branch (m)')
    axs[0,1].set_ylabel("Branch Index, m")
    axs[0,1].legend(); axs[0,1].grid(True)
    
    axs[1,0].plot(freqs, eps_eff.real, 'b-', label='Real(eps)')
    axs[1,0].plot(freqs, eps_eff.imag, 'b--', label='Imag(eps) (>0)')
    axs[1,0].set_ylabel("Epsilon Effective")
    axs[1,0].legend(); axs[1,0].grid(True)
    
    axs[1,0].plot(freqs, mu_eff.real, 'g-', label='Real(mu)')
    axs[1,0].plot(freqs, mu_eff.imag, 'g--', label='Imag(mu)')
    axs[1,0].set_ylabel("Mu Effective")
    axs[1,0].legend(); axs[1,0].grid(True)

    axs[1,1].plot(freqs, Z_eff.real, 'r-', label='Real(Z)')
    axs[1,1].plot(freqs, Z_eff.imag, 'r--', label='Imag(Z)')
    axs[1,1].set_ylabel("Impedance Effective")
    axs[1,1].legend(); axs[1,1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"monitor_extraction_{point_name}.png"))
    plt.close()

if __name__ == "__main__":
    point = "Peak Linearity"  
    r1, r2, w_mon = param_dictionary[point]
    run_simulation(point, r1, r2, w_mon, N_periods=11, autoshutoff=1e-7, resolution=32, padding=10)