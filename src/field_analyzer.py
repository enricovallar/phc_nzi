from simulation_handler import Simulation
from simulation_viewer import SimulationViewer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.simulation_handler import MPBDataOptions, MPBDataConverter





class FieldAnalyzer:
    def __init__(self,simulation: Simulation,  bands, polarization, k_direction):
        self.bands = bands
        self.polarization = polarization
        self.simulation = simulation 
        self.df = self.simulation.load_frequency_data(self.polarization)  
        self.result_df = None
        self.k_direction = k_direction

    def load_field_data(self, k_index, band_index, field_type, nonbloch = True, comp = "z", conversion_options: MPBDataOptions = None):
        field =  self.simulation.load_and_convert_field_data(k_index, band_index, comp, self.polarization, field_type, 
                                                           nonbloch = nonbloch, conversion_options=conversion_options,
                                                           file_comp=comp)  
        if field is None:
            raise ValueError("Field data is None")
        return self.make_2D(field)
    
    def load_hfield_data(self, k_index, band_index, comp, nonbloch = True, conversion_options: MPBDataOptions = None): 
        return self.load_field_data(k_index, band_index, "h", nonbloch, comp, conversion_options)
    
    def load_efield_data(self, k_index, band_index, comp, nonbloch = True, conversion_options: MPBDataOptions = None): 
        return self.load_field_data(k_index, band_index, "e", nonbloch, comp ,conversion_options)    
    
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
        
    
    def get_eps_mu(self, k_index, band_index, component_i, component_j, average_type = "2D", nonbloch = True, conversion_options: MPBDataOptions = None):
        e_field_i = self.load_efield_data(k_index, band_index, component_i,  nonbloch = nonbloch, conversion_options=conversion_options)
        h_field_j = self.load_hfield_data(k_index, band_index, component_j,  nonbloch = nonbloch, conversion_options=conversion_options)
        
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

            
        e_over_h_I = e_field_i_masked_I_avg.real / h_field_j_masked_I_avg.real
        e_over_h_II = e_field_i_masked_II_avg.real / h_field_j_masked_II_avg.real

        mu_I = e_over_h_I
        mu_II = e_over_h_II
        eps_I = 1/e_over_h_I
        eps_II = 1/e_over_h_II

        k_mag, freq = self.get_kmag_and_freq(k_index, band_index)
        k_mag_over_freq = k_mag / freq
        mu= np.emath.sqrt(mu_I)*np.emath.sqrt(mu_II)
        eps = np.emath.sqrt(eps_I)*np.emath.sqrt(eps_II)
        
        eps = eps*k_mag_over_freq
        mu = mu*k_mag_over_freq

        return  eps, mu
    
    def get_impedance(self, eps, mu):        
        return np.emath.sqrt(eps/mu)
    
    def get_refractive_index(self, eps, mu):
        return np.emath.sqrt(eps)*np.emath.sqrt(mu)
    
    def get_k_indices(self):
        return self.simulation.get_kpoints_indices(self.df)
    
    def get_eps_mu_impedance_neff(self, component_i, component_j , average_type = "2D", nonbloch = True, conversion_options: MPBDataOptions = None, plot = False):
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

        # Create dataframe to store results
        results = []
        for k_idx in k_indices:
            for band in self.bands:
                eps, mu = self.get_eps_mu(k_idx, band, component_i, component_j, average_type, nonbloch, conversion_options)
                impedance = self.get_impedance(eps, mu)
                neff = self.get_refractive_index(eps, mu)
                kmag, freq = self.get_kmag_and_freq(k_idx, band)
                
                results.append({
                    'k_index': k_idx,
                    'band': band,
                    'frequency': freq,
                    'eps': eps,
                    'mu': mu,
                    'impedance': impedance,
                    'n_eff': neff,
                    'kmag': kmag
                })
                
                print(f"K-Index: {k_idx}, Band: {band}, Eps: {eps}, Mu: {mu}, Impedance: {impedance}, N_eff: {neff}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        self.result_df = results_df
        return results_df
    
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
