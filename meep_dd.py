import meep as mp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
mp.verbosity(0)
# # # matplotlib agg backend for environments without display (e.g., Jupyter notebooks or HPC)
# matplotlib.use('TkAgg')

# ==========================================
# 1. Physical Parameters and Setup
# ==========================================
c_speed = 299792458.0  # m/s
a_m = 1e-6  # a = 1 micron
f_scale = (1e15 * a_m) / c_speed  # Conversion factor from PHz to MEEP units

# Simulation frequency range [cite: 167]
f_min_PHz = 0.05
f_max_PHz = 1.5
fc_meep = (f_min_PHz + f_max_PHz) / 2.0 * f_scale
df_meep = (f_max_PHz - f_min_PHz) * f_scale
nfreq = 500
# Quick-run debug mode: set True to run a low-res single-thickness test and
# produce the `dd_debug_d{d}nm.png` figure quickly. Set False for full runs.
FAST_TEST = False
if FAST_TEST:
    nfreq = 128

# Drude-Lorentz Parameters (from Table 1) [cite: 165, 177]
eps_inf = 1.8  # [cite: 178, 179]
mu_inf = 2.2   # [cite: 180, 186]

# Table 1 (Aladadi & Alkanhal 2019) parameters (SI units shown in table):
# eps_inf, mu_inf, omega_p = 2π×0.8e15 rad/s, omega_0 = 2π×0.4e15 rad/s,
# gamma_c = 80e15 s^-1, delta = 0.05e15 s^-1, mu_s = 2.6
eps_inf = 1.8
mu_inf = 2.2
mu_s = 2.6

# Frequencies in Hz (cycle frequency): table gives omega_p = 2π*0.8e15 (rad/s),
# so f_p = 0.8e15 Hz. Similarly f0 = 0.4e15 Hz.
f_wp_hz = 0.8e15
f_w0_hz = 0.4e15

# Damping rates (given in s^-1 in the table)
gamma_c_hz = 0.1e15
delta_hz = 0.05e15

# Convert to MEEP frequency units: value_in_meep = (value_in_Hz / 1e15) * f_scale
# Use cycle frequencies (this mapping matches the original script and avoids instability).
f_wp_meep = (f_wp_hz / 1e15) * f_scale
f_w0_meep = (f_w0_hz / 1e15) * f_scale
# gamma/delta in table are given in s^-1; convert to cycle frequency by dividing by 2π
gamma_meep = (gamma_c_hz / (2.0 * np.pi * 1e15)) * f_scale
delta_meep = (delta_hz / (2.0 * np.pi * 1e15)) * f_scale

# Susceptibilities: use Drude frequency=1.0 (as before) and sigma = f_wp_meep**2
susceptibilities = [
    mp.DrudeSusceptibility(frequency=1.0, sigma=f_wp_meep**2, gamma=gamma_meep),
    mp.LorentzianSusceptibility(frequency=f_w0_meep, gamma=delta_meep, sigma=(mu_s - mu_inf))
]
material = mp.Medium(epsilon=eps_inf, mu=mu_inf, E_susceptibilities=[susceptibilities[0]], H_susceptibilities=[susceptibilities[1]])

# Slab thicknesses to simulate [cite: 156]
thicknesses_nm = [100] 
if FAST_TEST:
    thicknesses_nm = [200]

# ==========================================
# 2. DD Method Extraction Functions
# ==========================================
def extract_parameters_dd(f_PHz, S11, S21, d_nm):
    d_m = d_nm * 1e-9
    c_speed = 299792458.0
    k = 2 * np.pi * (f_PHz * 1e15) / c_speed
    kd = k * d_m

    # 1. REVERT: Calculate Wave Impedance strictly enforcing Passivity (Real(z) >= 0) 
    z_eff_sq = ((1 + S11)**2 - S21**2) / ((1 - S11)**2 - S21**2)
    z_eff = np.sqrt(z_eff_sq)
    z_eff = np.where(np.real(z_eff) < 0, -z_eff, z_eff)

    # 2. Principal Branch Refractive Index (m=0) 
    arg_val = S21 / (1 - S11 * ((z_eff - 1) / (z_eff + 1)))
    phase_arg = np.angle(arg_val)
    n_principal = (1.0 / (kd + 1e-18)) * phase_arg
    n_imag = (-1.0 / (kd + 1e-18)) * np.real(np.log(arg_val + 0j))

    # 3. Data-Driven Discontinuity Detection [cite: 118]
    n_corrected = np.zeros_like(n_principal)
    m_arr = np.zeros_like(f_PHz, dtype=int)
    m_total = 0 
    n_corrected[0] = n_principal[0] 

    for i in range(1, len(f_PHz)):
        df = f_PHz[i] - f_PHz[i-1]
        
        # Calculate D(f) and q(f) on the m=0 branch [cite: 124, 147, 150]
        D_fi = (n_principal[i] - n_principal[i-1]) / df
        q_fi = (2 * n_principal[i-1]) / df
        
        # THE GUARDRAIL: A true branch jump MUST be accompanied by a phase wrap
        phase_jump = np.abs(phase_arg[i] - phase_arg[i-1])
        
        # Check if the jump condition is met AND it's a true phase wrap [cite: 128]
        if np.isclose(np.abs(D_fi), np.abs(q_fi), rtol=0.2) and phase_jump > np.pi:
            # Calculate the incremental branch change [cite: 130]
            dm = int(np.round((n_principal[i-1] - n_principal[i]) * kd[i] / (2 * np.pi)))
            m_total += dm # Accumulate the total branch index [cite: 141]
        
        # Apply the accumulated branch index [cite: 91, 121]
        m_arr[i] = m_total 
        n_corrected[i] = n_principal[i] + (2 * m_total * np.pi) / kd[i]

    # 4. Reconstruct Constitutive Parameters [cite: 97, 98]
    n_eff_complex = n_corrected + 1j * n_imag
    eps_eff = n_eff_complex / z_eff
    mu_eff = n_eff_complex * z_eff
    
    return n_principal, n_corrected, m_arr, eps_eff, mu_eff, z_eff

def get_analytical_models(f_PHz, ):
    w = 2 * np.pi * f_PHz * 1e15
    wp = 2 * np.pi * 0.8e15
    w0 = 2 * np.pi * 0.4e15
    gamma_c = 0.1e15
    delta = 0.05e15
    
    eps_ana = eps_inf - (wp**2) / (w**2 + 1j * gamma_c * w) # [cite: 159]
    mu_ana = mu_inf - ((mu_s - mu_inf) * w0**2) / (w**2 + 1j * delta * w - w0**2) # [cite: 159]
    return eps_ana, mu_ana

# ==========================================
# 3. FAST MEEP Simulation Loop
# ==========================================
dpml = 1.0 # PML thickness
cell_length = 10.0 # Total cell length in um
resolution = 500 # pixels/um
if FAST_TEST:
    resolution = 30

cell = mp.Vector3(0, 0, cell_length)
pml_layers = [mp.PML(dpml)]

# Normal incidence plane wave [cite: 65]
sources = [mp.Source(mp.GaussianSource(fc_meep, fwidth=df_meep),
                     component=mp.Ex, center=mp.Vector3(0, 0, -cell_length/2 + dpml + 0.5))]

# Measurement points (using single pixels)
pt_refl = mp.Vector3(0, 0, -cell_length/2 + dpml + 1.0)
pt_trans = mp.Vector3(0, 0, cell_length/2 - dpml - 1.0)
pt_size = mp.Vector3(0, 0, 0) # MEEP requires a size vector even for single points

# Precompute frequency array (used for all runs)
freqs_meep = np.linspace(fc_meep - df_meep/2.0, fc_meep + df_meep/2.0, nfreq)
f_PHz_arr = freqs_meep / f_scale

# --- Run 2: Simulation with Slabs ---
results = {}
for d_nm in thicknesses_nm:
    print(f"\n--- Running simulation for d = {d_nm} nm ---")
    d_um = d_nm / 1000.0

    # --- Normalization run for this slab thickness (empty cell) ---
    sim_norm = mp.Simulation(cell_size=cell, resolution=resolution, boundary_layers=pml_layers, dimensions=1, sources=sources)
    dft_norm_refl = sim_norm.add_dft_fields([mp.Ex], fc_meep, df_meep, nfreq, center=pt_refl, size=pt_size)
    dft_norm_trans = sim_norm.add_dft_fields([mp.Ex], fc_meep, df_meep, nfreq, center=pt_trans, size=pt_size)
    sim_norm.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, pt_trans, 1e-12))

    # Extract incident fields (squeeze 0-D arrays)
    Ex_inc = np.array([np.squeeze(sim_norm.get_dft_array(dft_norm_refl, mp.Ex, i)) for i in range(nfreq)], dtype=complex)
    Ex_norm_trans = np.array([np.squeeze(sim_norm.get_dft_array(dft_norm_trans, mp.Ex, i)) for i in range(nfreq)], dtype=complex)

    # Plot normalization region permittivity (helps verify geometry/setup)
    try:
        eps_norm = np.array(sim_norm.get_array(center=mp.Vector3(0,0,0), size=mp.Vector3(0,0,cell_length), component=mp.Dielectric)).flatten()
        z_norm = np.linspace(-cell_length/2.0, cell_length/2.0, eps_norm.size)
        plt.figure(figsize=(8,3))
        plt.plot(z_norm, eps_norm, '-k')
        plt.title(f'Simulation region permittivity (normalization, d = {d_nm} nm)')
        plt.xlabel('z (um)')
        plt.ylabel('epsilon')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'meep_region_norm_d{d_nm}nm.png')
        plt.close()
    except Exception:
        pass

    sim_norm.reset_meep()
    geometry = [mp.Block(mp.Vector3(mp.inf, mp.inf, d_um), center=mp.Vector3(0,0,0), material=material)]
    
    sim = mp.Simulation(cell_size=cell, resolution=resolution, boundary_layers=pml_layers, dimensions=1, sources=sources, geometry=geometry)
    
    dft_refl = sim.add_dft_fields([mp.Ex], fc_meep, df_meep, nfreq, center=pt_refl, size=pt_size)
    dft_trans = sim.add_dft_fields([mp.Ex], fc_meep, df_meep, nfreq, center=pt_trans, size=pt_size)
    
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, pt_trans, 1e-12))
    
    # Extract the full arrays (squeeze 0-D results to scalars)
    Ex_total_refl = np.array([np.squeeze(sim.get_dft_array(dft_refl, mp.Ex, i)) for i in range(nfreq)], dtype=complex)
    Ex_trans = np.array([np.squeeze(sim.get_dft_array(dft_trans, mp.Ex, i)) for i in range(nfreq)], dtype=complex)
    
    # Calculate raw S-parameters
    S11_raw = (Ex_total_refl - Ex_inc) / Ex_inc 
    S21_raw = Ex_trans / Ex_norm_trans

    # Save S-parameters for diagnostics
    results.setdefault(d_nm, {})
    results[d_nm]['S11_raw'] = S11_raw
    results[d_nm]['S21_raw'] = S21_raw
    
    # Shift phase reference planes from the DFT coordinates to the actual slab boundaries
    d_m = d_nm * 1e-9
    z_refl = pt_refl.z
    z_front = -d_um / 2.0
    L1_m = (z_front - z_refl) * 1e-6
    k = 2 * np.pi * (f_PHz_arr * 1e15) / c_speed
    
    S11 = S11_raw * np.exp(-1j * 2 * k * L1_m)
    S21 = S21_raw * np.exp(1j * k * d_m)
    
    # Post-process (enable debug plotting for FAST_TEST)

    n_m0, n_dd, m_arr, eps_eff, mu_eff, z_eff = extract_parameters_dd(f_PHz_arr, S11, S21, d_nm)
    results[d_nm].update({'n_m0': n_m0, 'n_dd': n_dd, 'm': m_arr, 'eps': eps_eff, 'mu': mu_eff, 'z_eff': z_eff})
    sim.reset_meep() 

# ==========================================
# 4. Plotting Results
# ==========================================
eps_ana, mu_ana = get_analytical_models(f_PHz_arr)

for d_nm in thicknesses_nm:
    res = results[d_nm]
    # Diagnostic S-parameter plots
    S11 = res.get('S11_raw')
    S21 = res.get('S21_raw')
    if S11 is not None and S21 is not None:
        fig_s, ax_s = plt.subplots(2, 2, figsize=(10, 8))
        ax_s[0,0].plot(f_PHz_arr, np.abs(S11))
        ax_s[0,0].set_ylabel('|S11|')
        ax_s[0,1].plot(f_PHz_arr, np.angle(S11))
        ax_s[0,1].set_ylabel('arg(S11)')
        ax_s[1,0].plot(f_PHz_arr, np.abs(S21))
        ax_s[1,0].set_ylabel('|S21|')
        ax_s[1,1].plot(f_PHz_arr, np.angle(S21))
        ax_s[1,1].set_ylabel('arg(S21)')
        for ax_row in ax_s:
            for ax in ax_row:
                ax.set_xlabel('Frequency (PHz)')
                ax.grid(True)
        plt.tight_layout()
        plt.savefig(f'meep_sparams_d{d_nm}nm.png')
        plt.close()
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Homogeneous Metamaterial Slab Extraction (d = {d_nm} nm)")

    # Identify the exact indices where a branch jump occurred
    # np.diff(m) != 0 gives True where the value changes. +1 aligns it with the step index.
    jump_indices = np.where(np.diff(res['m']) != 0)[0] + 1
    jump_freqs = f_PHz_arr[jump_indices]
    jump_n_m0_vals = res['n_m0'][jump_indices]
    
    # Plot Real n [cite: 297]
    axs[0,0].plot(f_PHz_arr, res['n_m0'], label='m=0')
    axs[0,0].plot(f_PHz_arr, res['n_dd'], 'k--', label='DD Method')
    # Overlay the detected jumps
    if len(jump_indices) > 0:
        axs[0,0].plot(jump_freqs, jump_n_m0_vals, 'rx', markersize=10, mew=2, label='Detected Jumps')
    axs[0,0].set_ylim(-10, 10)
    axs[0,0].set_ylabel("Real(n_eff)")
    axs[0,0].set_xlabel("Frequency (PHz)")
    axs[0,0].legend()
    axs[0,0].grid(True)
    
    # Plot Branch Index m [cite: 297]
    axs[0,1].plot(f_PHz_arr, np.zeros_like(f_PHz_arr), label='m=0')
    axs[0,1].plot(f_PHz_arr, res['m'], 'r--', label='DD Method')
    # Overlay the detected jumps
    if len(jump_indices) > 0:
        axs[0,1].plot(jump_freqs, res['m'][jump_indices], 'rx', markersize=10, mew=2, label='Detected Jumps')
    axs[0,1].set_ylabel("Branch Index, m")
    axs[0,1].set_xlabel("Frequency (PHz)")
    axs[0,1].legend()
    axs[0,1].grid(True)
    
    # Plot Epsilon Effective [cite: 359]
    axs[1,0].plot(f_PHz_arr, np.real(eps_ana), 'b-', label='Defined Real(eps)')
    axs[1,0].plot(f_PHz_arr, np.imag(eps_ana), 'b--', label='Defined Imag(eps)')
    # Diagnostic: report number of finite extracted epsilon points
    n_eps_finite = np.sum(np.isfinite(np.real(res['eps'])))
    print(f"d={d_nm} nm: extracted eps finite points = {n_eps_finite}/{len(f_PHz_arr)}")
    axs[1,0].plot(f_PHz_arr, np.real(res['eps']), 'k--', label='Extracted DD Real(eps)')
    axs[1,0].set_ylim(-10, 25)
    axs[1,0].set_ylabel("Epsilon Effective")
    axs[1,0].set_xlabel("Frequency (PHz)")
    axs[1,0].legend()
    axs[1,0].grid(True)
    
    # Plot Mu Effective [cite: 359]
    axs[1,0].plot(f_PHz_arr, np.real(mu_ana), 'g-', label='Defined Real(mu)')
    axs[1,0].plot(f_PHz_arr, np.imag(mu_ana), 'g--', label='Defined Imag(mu)')
    # Diagnostic: report number of finite extracted mu points
    n_mu_finite = np.sum(np.isfinite(np.real(res['mu'])))
    print(f"d={d_nm} nm: extracted mu finite points = {n_mu_finite}/{len(f_PHz_arr)}")
    axs[1,0].plot(f_PHz_arr, np.real(res['mu']), 'k--', label='Extracted DD Real(mu)')
    axs[1,0].set_ylim(-10, 25)
    axs[1,0].set_ylabel("Mu Effective")
    axs[1,0].set_xlabel("Frequency (PHz)")
    axs[1,0].legend()
    axs[1,0].grid(True)

    # Plot Impedance Effective [cite: 359]
    axs[1,1].plot(f_PHz_arr, np.real(z_eff), 'm-', label='Extracted Real(z)')
    axs[1,1].plot(f_PHz_arr, np.imag(z_eff), 'm--', label='Extracted Imag(z)')
    axs[1,1].set_ylim(-10, 25)
    axs[1,1].set_ylabel("Impedance Effective")
    axs[1,1].set_xlabel("Frequency (PHz)")
    axs[1,1].legend()
    axs[1,1].grid(True)
    
    
    plt.tight_layout()

    plt.savefig(f"meep_dd2_results_d{d_nm}nm.png")
    plt.show()
    