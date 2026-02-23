from phc_nzi.simulation_handler import Simulation
from phc_nzi.simulation_viewer import SimulationViewer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from phc_nzi.simulation_handler import MPBDataOptions, MPBDataConverter
import concurrent.futures

def _retrieve_single_point(analyzer, k_idx, band, component_i, component_j, average_type, nonbloch, conversion_options, reference_masks, overwrite):
    eps, mu = analyzer.get_eps_mu(k_idx, band, component_i, component_j, average_type, nonbloch, conversion_options, reference_masks=reference_masks, overwrite=overwrite)
    impedance = analyzer.get_impedance(eps, mu)
    neff = analyzer.get_refractive_index(eps, mu)
    kmag, freq = analyzer.get_kmag_and_freq(k_idx, band)
    return {
        'k_index': k_idx,
        'band': band,
        'frequency': freq,
        'eps': eps,
        'mu': mu,
        'impedance': impedance,
        'n_eff': neff,
        'kmag': kmag
    }





class FieldAnalyzer:
    def __init__(self,simulation: Simulation,  bands, polarization, k_direction):
        self.bands = bands
        self.polarization = polarization
        self.simulation = simulation 
        self.df = self.simulation.load_frequency_data(self.polarization)  
        self.result_df = None
        self.k_direction = k_direction

    def load_field_data(self, k_index, band_index, field_type, nonbloch = True, comp = "z", conversion_options: MPBDataOptions = None, overwrite=False):
        field =  self.simulation.load_and_convert_field_data(k_index, band_index, comp, self.polarization, field_type, 
                                                           nonbloch = nonbloch, conversion_options=conversion_options,
                                                           file_comp=comp, overwrite=overwrite)  
        if field is None:
            raise ValueError("Field data is None")
        return self.make_2D(field)
    
    def load_hfield_data(self, k_index, band_index, comp, nonbloch = True, conversion_options: MPBDataOptions = None, overwrite=False): 
        return self.load_field_data(k_index, band_index, "h", nonbloch, comp, conversion_options, overwrite=overwrite)
    
    def load_efield_data(self, k_index, band_index, comp, nonbloch = True, conversion_options: MPBDataOptions = None, overwrite=False): 
        return self.load_field_data(k_index, band_index, "e", nonbloch, comp ,conversion_options, overwrite=overwrite)    
    
    def make_2D(self, field_data):
        if field_data.ndim == 2:
            return field_data
        elif field_data.ndim == 3:
            return field_data[:,:,field_data.shape[2]//2]
        else: 
            raise ValueError("Field data must be 2D or 3D")
        
    def field_less_than_zero(self, field_data):
        """Returns a boolean mask for values less than zero"""
        return field_data < 0
    
    def field_greater_than_zero(self, field_data):
        """Returns a boolean mask for values greater than zero or equal to zero"""
        return field_data >= 0
    
    
    def get_mask(self, field_data, mask_func):
        return mask_func(field_data)
    
    def mask_field_data(self, field_data, mask):
        """Mask field data by setting values where mask is False to None/np.nan
        while preserving the shape of the array"""
        masked = field_data.copy()
        masked[~mask] = 0
        return masked

    def average_2D(self, field_data):
        if field_data.ndim == 3:
            raise ValueError("Field data must be 2D")
        shape = field_data.shape
        return field_data.sum() / (shape[0] * shape[1])
    
    def average_1D(self, field_data, axis):
        if field_data.ndim == 3:
            raise ValueError("Field data must be 2D")
        if axis not in [0, 1]:
            raise ValueError("Axis must be 0 or 1")
        if axis == 0:
            return field_data[0,:].sum() / field_data.shape[1]
        elif axis == 1:
            return field_data[:,0].sum() / field_data.shape[0]
        
    def get_kmag_and_freq(self, k_idx, b_idx):
        kmag, freq = self.simulation.get_kmag_and_freq(self.df, k_idx,  b_idx, self.polarization,)
        return kmag, freq
    
    def get_freq(self, k_idx, b_idx):
        return self.simulation.get_freq(self.df, k_idx, b_idx, self.polarization,)

    def compute_reference_mask(self, band_index, component_j, nonbloch=True, conversion_options=None, overwrite=False):
        """Compute a handedness mask from the k-point with the largest |k|.

        The mode pattern is best defined far from Gamma where the fields
        have clear spatial structure.  The returned mask is reused for all
        k-points so that the handedness regions do not jump when H_z is
        nearly zero near the Dirac point.

        Returns
        -------
        mask_I : ndarray[bool]   – pixels where Re(H_j) >= 0 at reference k
        mask_II : ndarray[bool]  – pixels where Re(H_j) <  0 at reference k
        ref_k : int              – the k-index used as reference
        """
        k_indices = list(self.get_k_indices())
        ref_k = max(k_indices)  # largest k → best-defined mode
        h_ref = self.load_hfield_data(ref_k, band_index, component_j,
                                      nonbloch=nonbloch,
                                      conversion_options=conversion_options,
                                      overwrite=overwrite)
        mask_I = h_ref.real >= 0
        mask_II = h_ref.real < 0
        return mask_I, mask_II, ref_k

    def get_eps_mu(self, k_index, band_index, component_i, component_j, average_type = "2D", nonbloch = True, conversion_options: MPBDataOptions = None, reference_masks = None, overwrite=False):
        e_field_i = self.load_efield_data(k_index, band_index, component_i,  nonbloch = nonbloch, conversion_options=conversion_options, overwrite=overwrite)
        h_field_j = self.load_hfield_data(k_index, band_index, component_j,  nonbloch = nonbloch, conversion_options=conversion_options, overwrite=overwrite)
        
        if self.k_direction == "x":
            if self.polarization == "tm" or self.polarization == "zodd":
                sgn_mu = 1
                sgn_eps = 1

            elif self.polarization == "te" or self.polarization == "zeven":
                sgn_mu = 1
                sgn_eps = -1
        if self.k_direction == "y":
            if self.polarization == "tm" or self.polarization == "zodd":
                sgn_mu = -1
                sgn_eps = -1

            elif self.polarization == "te" or self.polarization == "zeven":
                sgn_mu = -1
                sgn_eps = 1
        

        if reference_masks is not None:
            mask_I, mask_II = reference_masks
        else:
            mask_I = self.get_mask(h_field_j.real, self.field_greater_than_zero)
            mask_II = self.get_mask(h_field_j.real, self.field_less_than_zero)

        # Create the masked complex fields for both regions
        e_field_i_masked_I = self.mask_field_data(e_field_i, mask_I)
        e_field_i_masked_II = self.mask_field_data(e_field_i, mask_II)
        h_field_j_masked_I = self.mask_field_data(h_field_j, mask_I)
        h_field_j_masked_II = self.mask_field_data(h_field_j, mask_II)

        
        if average_type == "2D":
            e_field_i_masked_I_avg = self.average_2D(e_field_i_masked_I)
            e_field_i_masked_II_avg = self.average_2D(e_field_i_masked_II)
            h_field_j_masked_I_avg = self.average_2D(h_field_j_masked_I)
            h_field_j_masked_II_avg = self.average_2D(h_field_j_masked_II)

        elif average_type == "1D":
            e_field_i_masked_I_avg = self.average_1D(e_field_i_masked_I, 0)
            e_field_i_masked_II_avg = self.average_1D(e_field_i_masked_II, 0)
            h_field_j_masked_I_avg = self.average_1D(h_field_j_masked_I, 0)
            h_field_j_masked_II_avg = self.average_1D(h_field_j_masked_II, 0)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e_over_h_I = e_field_i_masked_I_avg.real / h_field_j_masked_I_avg.real
            e_over_h_II = e_field_i_masked_II_avg.real / h_field_j_masked_II_avg.real

        if np.isnan(e_over_h_I) or np.isnan(e_over_h_II) or np.isinf(e_over_h_I) or np.isinf(e_over_h_II) or e_over_h_I == 0 or e_over_h_II == 0:
            return np.nan + 0j, np.nan + 0j

        mu_I = e_over_h_I
        mu_II = e_over_h_II
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eps_I = 1/e_over_h_I
            eps_II = 1/e_over_h_II

        k_mag, freq = self.get_kmag_and_freq(k_index, band_index)
        k_mag_over_freq = k_mag / freq
        mu = np.emath.sqrt(mu_I)*np.emath.sqrt(mu_II)
        eps = np.emath.sqrt(eps_I)*np.emath.sqrt(eps_II)
        
        eps = eps*k_mag_over_freq
        mu = mu*k_mag_over_freq

        return  eps, mu
    
    def get_impedance(self, eps, mu):
        if pd.isna(eps) or pd.isna(mu) or eps == 0:
            return np.nan + 0j
        return np.emath.sqrt(mu/eps)
    
    def get_refractive_index(self, eps, mu):
        if pd.isna(eps) or pd.isna(mu):
            return np.nan + 0j
        return np.emath.sqrt(eps)*np.emath.sqrt(mu)
    
    def get_k_indices(self):
        return self.simulation.get_kpoints_indices(self.df)
    
    def get_eps_mu_impedance_neff(self, component_i, component_j , average_type = "2D", nonbloch = True, conversion_options: MPBDataOptions = None, plot = False, enforce_continuity=True, overwrite=False):
        k_indices = self.get_k_indices()
        bands_freqs = []
        for band in self.bands:
            freqs = []
            for k_idx in k_indices:
                freq = self.get_freq(k_idx, band)
                freqs.append(freq)
            bands_freqs.append(freqs)
        if plot is True:
            plt.figure()
            plt.title("Selected Points")
            for idx, freqs in enumerate(bands_freqs):
                plt.plot(k_indices, freqs, "ro", label=f"Band {idx}")
                plt.xlabel("K-Index")
                plt.ylabel("Frequency")
                plt.grid(True)
            plt.legend()

        # Dynamic masking is now forced to prevent phase discontinuities and unphysical divisions
        # Create dataframe to store results
        results = []
        requests = []
        for k_idx in k_indices:
            for band in self.bands:
                masks = None # Always use dynamic masks
                requests.append((self, k_idx, band, component_i, component_j, average_type, nonbloch, conversion_options, masks, overwrite))
        
        # Parallel execution
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(_retrieve_single_point, *req) for req in requests]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    results.append(res)
                    # print(f"Finished K-Index: {res['k_index']}, Band: {res['band']}") # Reduced verbosity for speed
                except Exception as e:
                    print(f"Error processing item: {e}")
        
        # Sort results because parallel execution might shuffle them
        results.sort(key=lambda x: (x['band'], x['k_index']))
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        if enforce_continuity:
            results_df = self._enforce_phase_continuity(results_df)

        self.result_df = results_df
        return results_df

    def _enforce_phase_continuity(self, df):
        """
        Enforces continuity of n_eff, eps, mu, and impedance by choosing the branch 
        (positive or negative) that minimizes the distance to the previous point.
        """
        bands = df['band'].unique()
        df_corrected = df.copy()
        
        def enforce_sign_continuity(values, initial_sign=1.0):
            new_values = np.zeros_like(values)
            new_values[0] = values[0] * initial_sign
            valid_prev = new_values[0]
            for i in range(1, len(values)):
                curr = values[i]
                if pd.isna(curr) or np.isinf(curr):
                    new_values[i] = curr
                else:
                    if pd.isna(valid_prev) or np.isinf(valid_prev):
                        # If we lacked a valid previous point, adopt the branch that matches current principal root * initial_sign
                        # or just stick to the principal root. We'll pick principal * initial_sign.
                        new_values[i] = curr * initial_sign
                        valid_prev = new_values[i]
                    else:
                        cand1 = curr
                        cand2 = -curr
                        if abs(cand2 - valid_prev) < abs(cand1 - valid_prev):
                            new_values[i] = cand2
                            valid_prev = cand2
                        else:
                            new_values[i] = cand1
                            valid_prev = cand1
            return new_values
            
        for band in bands:
            band_mask = df_corrected['band'] == band
            band_indices = df_corrected[band_mask].index
            
            if len(band_indices) < 2:
                continue
                
            # We need to determine if this band is a positive or negative index band.
            # We can do this robustly by checking the macroscopic group velocity d(omega) / dk!
            # If the frequency decreases as the wave vector increases, the band is backward-wave (negative index).
            freqs = df_corrected.loc[band_indices, 'frequency'].values
            
            # Simple slope over the band to get the macroscopic v_g sign
            # We assume k_points are ordered by increasing wavevector magnitude in the DataFrame
            if len(freqs) > 1:
                # Average slope over the first few points or the entire band
                delta_freq = freqs[-1] - freqs[0]
                # If frequency goes down as k goes up, group and phase velocity are anti-parallel
                physical_sign = -1.0 if delta_freq < 0 else 1.0
            else:
                physical_sign = 1.0
                
            print(f"DEBUG: band={band}, delta_freq={delta_freq:.5f} -> physical_sign={physical_sign}")
            
            eps = df_corrected.loc[band_indices, 'eps'].values
            mu = df_corrected.loc[band_indices, 'mu'].values
            
            eps_cont = enforce_sign_continuity(eps, initial_sign=physical_sign)
            mu_cont = enforce_sign_continuity(mu, initial_sign=physical_sign)
            
            # Recompute Z and n_eff from the continuous eps and mu
            # We must also enforce continuity on Z because of the sqrt branch cut
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Z_raw = np.emath.sqrt(mu_cont / eps_cont)
            
            # Z is fundamentally a positive quantity for passive media real parts
            # We enforce standard continuity on it without forcing a negative initial sign
            Z_cont = enforce_sign_continuity(Z_raw, initial_sign=1.0)
            
            # n_eff is uniquely determined by eps and Z without sign ambiguity
            n_eff_cont = eps_cont * Z_cont
            
            df_corrected.loc[band_indices, 'eps'] = eps_cont
            df_corrected.loc[band_indices, 'mu'] = mu_cont
            df_corrected.loc[band_indices, 'impedance'] = Z_cont
            df_corrected.loc[band_indices, 'n_eff'] = n_eff_cont
                
        return df_corrected
    
    def plot_eps_vs_freqs(self):
        results_df = self.result_df
        if results_df is None:
            print("No results found. Run get_eps_mu_impedance_neff() first.")
            return
        plt.plot(results_df["frequency"], results_df["eps"], "ro", label="$\\varepsilon_{eff}$")
        plt.xlabel("Frequency")
        plt.legend()
        plt.grid(True)
    
    def plot_mu_vs_freqs(self):
        results_df = self.result_df
        if results_df is None:
            print("No results found. Run get_eps_mu_impedance_neff() first.")
            return
        plt.plot(results_df["frequency"], results_df["mu"], "bo", label="$\\mu_{eff}$")
        plt.xlabel("Frequency")
        plt.legend()    
        plt.grid(True)

    def plot_impedance_vs_freqs(self):
        results_df = self.result_df
        if results_df is None:
            print("No results found. Run get_eps_mu_impedance_neff() first.")
            return  
        plt.plot(results_df["frequency"], results_df["impedance"], "go", label="$Z_{eff}$")
        plt.xlabel("Frequency")
        plt.legend()
        plt.grid(True)

    def plot_neff_vs_freqs(self):
        results_df = self.result_df
        if self.result_df is None:
            print("No results found. Run get_eps_mu_impedance_neff() first.")
            return
        plt.plot(results_df["frequency"], results_df["n_eff"], "yo", label="$n_{eff}$")
        plt.xlabel("Frequency")
        plt.legend()
        plt.grid(True)
