import numpy as np
from phc_nzi.simulation_handler import MPBDataOptions

def load_mpb_data(simulation, gamma_idx, b_idxes, parity):
    """Extracts the correct field components and epsilon from MPB."""
    # Keep rectify=False to preserve exact mode orthogonality and grid geometry
    opt_math = MPBDataOptions(rectify=False, periods=1)
    
    # Define MPB fetch parameters based on parity
    parity_str = parity.lower()
    field_str = "e" if parity_str == "tm" else "h"
    
    fields = []
    for b_idx in b_idxes:
        f = simulation.load_and_convert_field_data(
            gamma_idx, b_idx, "z", parity_str, field_str, conversion_options=opt_math, overwrite=True)
        fields.append(f)
        
    eps = simulation.load_and_convert_epsilon_data(options=opt_math)
    return np.array(fields), eps

def normalize_fields(fields, eps, parity):
    """Normalizes fields. TM is eps-weighted, TE is unweighted."""
    fields_normed = np.copy(fields)
    
    # TM norm involves integral of (eps * |E|^2)
    # TE norm involves integral of (|H|^2)
    norm_weight = eps if parity.lower() == 'tm' else np.ones_like(eps)
    
    for i in range(fields_normed.shape[0]):
        norm_factor = np.sqrt(np.mean(norm_weight * np.abs(fields_normed[i])**2)) 
        fields_normed[i] /= norm_factor
        
    return fields_normed

def compute_cartesian_gradients(fields, lattice_type):
    """Calculates spatial gradients and maps them to standard Cartesian (x, y)."""
    res_u1, res_u2 = fields.shape[1], fields.shape[2] 
    du1, du2 = 1.0 / res_u1, 1.0 / res_u2
    
    # Raw gradients in fractional grid space
    grad_u1, grad_u2 = np.gradient(fields, du1, du2, axis=(1, 2))
    
    # Apply Inverse Jacobian based on lattice geometry
    if lattice_type.lower() == 'hexagonal':
        grad_x = grad_u1
        grad_y = (2.0 / np.sqrt(3)) * grad_u2 - (1.0 / np.sqrt(3)) * grad_u1
    elif lattice_type.lower() == 'square':
        grad_x = grad_u1
        grad_y = grad_u2
    else:
        raise ValueError("lattice_type must be 'square' or 'hexagonal'")
        
    return np.array([grad_x, grad_y]), res_u1, res_u2

def calculate_pt_matrices(fields, eps, parity, lattice_type, a=1.0):
    """Master function to compute P_ij and Q matrices."""
    # 1. Normalize
    fields = normalize_fields(fields, eps, parity)
    
    # 2. Get Cartesian gradients
    grad_cart, res_u1, res_u2 = compute_cartesian_gradients(fields, lattice_type)
    
    # 3. Define integration weight for the P/Q equations
    # TM matrix terms use a weight of 1. TE matrix terms use a weight of 1/eps.
    int_weight = np.ones_like(eps) if parity.lower() == 'tm' else (1.0 / eps)
    
    # 4. Compute P Matrix
    p_matrix = np.zeros((fields.shape[0], fields.shape[0], 2), dtype=complex)
    for dim in [0, 1]:
        # Using the unified integration weight makes the einsum identical for TE/TM
        term1 = np.einsum('lyx, yx, jyx -> lj', np.conj(fields), int_weight, grad_cart[dim]) / (res_u1 * res_u2)
        term2 = np.einsum('jyx, yx, lyx -> lj', fields, int_weight, np.conj(grad_cart[dim])) / (res_u1 * res_u2)
        p_matrix[:, :, dim] = (1j / a**2) * (term1 - term2)
        
    # 5. Compute Q Matrix
    q_matrix = np.einsum('iyx, yx, jyx -> ij', np.conj(fields), int_weight, fields) / (res_u1 * res_u2)
    
    return p_matrix, q_matrix

def get_cartesian_k(k1, k2, lattice_type):
    """Converts fractional MPB k-vectors to pure Cartesian components."""
    if lattice_type.lower() == 'hexagonal':
        kx_cart = k1 
        ky_cart = (2 * k2 - k1) / np.sqrt(3)
    elif lattice_type.lower() == 'square':
        kx_cart = k1
        ky_cart = k2
    else:
        raise ValueError("lattice_type must be 'square' or 'hexagonal'")
    return kx_cart, ky_cart

def calculate_perturbed_bands(p_matrix, q_matrix, omega_0s, k1_arr, k2_arr, kmag_arr, lattice_type):
    """Solves the eigenvalue problem to return perturbed band frequencies."""
    kx_cart, ky_cart = get_cartesian_k(k1_arr, k2_arr, lattice_type)
    
    bands_calc = np.zeros((len(k1_arr), len(omega_0s)))
    Omega_0_sq = np.diag((2 * np.pi * np.array(omega_0s))**2)
    
    for idx in range(len(k1_arr)):
        # Physical wavevectors
        kx_phys = kx_cart[idx] * 2 * np.pi
        ky_phys = ky_cart[idx] * 2 * np.pi
        k2_phys = (kmag_arr[idx] * 2 * np.pi)**2 
        
        # Assemble Matrix: M = W_0^2 - (kx*Px + ky*Py) + k^2*Q
        M = Omega_0_sq - (kx_phys * p_matrix[:, :, 0] + ky_phys * p_matrix[:, :, 1]) + k2_phys * q_matrix
        
        # Solve eigenvalues
        bands_calc[idx, :] = np.sqrt(np.sort(np.real(np.linalg.eigvals(M)))) / (2 * np.pi)
        
    return bands_calc


def calculate_reduced_bands(p_matrix, omega_0s, k1_arr, k2_arr, kmag_arr, lattice_type, target_indices):
    """
    Computes the band dispersion using only the reduced NxN subspace.
    Uses Cartesian k-components to ensure the dot product is physically correct.
    """
    # Map fractional k to Cartesian (kx, ky)
    kx_cart, ky_cart = get_cartesian_k(k1_arr, k2_arr, lattice_type)
    
    bands_calc = np.zeros((len(k1_arr), len(target_indices)))
    
    # Isolate the NxN subspace and unperturbed frequencies
    omega_0_red = np.array([omega_0s[i] for i in target_indices])
    Omega_0_sq_red = np.diag((2 * np.pi * omega_0_red)**2)
    p_red = p_matrix[np.ix_(target_indices, target_indices)]
    
    for idx in range(len(k1_arr)):
        # Physical wavevectors (scaling by 2pi)
        kx_phys = kx_cart[idx] * 2 * np.pi
        ky_phys = ky_cart[idx] * 2 * np.pi
        
        # M_red = W_0^2 - (kx*Px + ky*Py)
        M_red = Omega_0_sq_red - (kx_phys * p_red[:, :, 0] + ky_phys * p_red[:, :, 1])
        
        # Eigenvalues of the matrix are (omega * 2pi)^2
        eigs = np.linalg.eigvals(M_red)
        bands_calc[idx, :] = np.sqrt(np.sort(np.real(eigs))) / (2 * np.pi)
        
    return bands_calc

def calculate_group_velocity_tensor(p_matrix, omega_0s, target_indices):
    """
    Calculates the Cartesian group velocity matrix elements at Gamma.
    
    Returns:
    - vg_x: NxN matrix of velocities along the Cartesian x-axis.
    - vg_y: NxN matrix of velocities along the Cartesian y-axis.
    
    Units: Returns dimensionless velocity (v/c).
    """
    # Isolate the subspace
    p_red = p_matrix[np.ix_(target_indices, target_indices)]
    
    # Use the average frequency of the degenerate triplet for the denominator
    # (In a true degeneracy, these are all identical)
    w0_phys = omega_0s[target_indices[0]] * 2 * np.pi
    
    # vg = P / (2 * w0)
    # The 2*pi factors from P and w0 cancel out, but we need 
    # to account for the k-scaling. P was defined as d(w^2)/dk.
    # Therefore vg = P / (2 * w0 * 2pi)
    vg_x = p_red[:, :, 0] / (2 * w0_phys * 2 * np.pi)
    vg_y = p_red[:, :, 1] / (2 * w0_phys * 2 * np.pi)
    
    return vg_x, vg_y
