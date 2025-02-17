import numpy as np
import matplotlib.pyplot as plt
import inspect
from photonic_crystal_maker import PhotonicCrystal, Geometry, Lattice, Material
from mpb_configurator import MPBSchemeConfigurator
from simulation_handler import Simulation, SimulationViewer
from pandas import DataFrame    
import os
from tqdm import tqdm


class PhCBatch:
    def __init__(self, phc: PhotonicCrystal, configurator_options: dict, runner_options: dict): 
        """
        Constructor for the PhCBatch class.
        :param phc: Instance of the PhotonicCrystal class.
        :param configurator_options: Dictionary with the options for the MPBSchemeConfigurator class.
        :param runner_options: Dictionary with the options for the Simulation class.
        """
        self._phc = phc
        self._scheme_configurator = MPBSchemeConfigurator(phc, **configurator_options)
        self._simulation = Simulation(**runner_options)
        self._root_dir = self._simulation.directory
        self._df_sweep_list = []
        os.makedirs(self._root_dir, exist_ok=True)

    @staticmethod
    def print_available_configuration_options():
        sig = inspect.signature(MPBSchemeConfigurator.__init__)
        print("Configuration options:")
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            print(f"    {name}: default={param.default}")

    @staticmethod
    def print_available_runner_options():
        sig = inspect.signature(Simulation.__init__)
        print("Runner options:")
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            print(f"    {name}: default={param.default}")

    def print_script_params(self):
        print(self._phc.print_script_params())

    @property
    def phc(self):
        return self._phc

    @property
    def scheme_configurator(self):
        return self._scheme_configurator

    @property
    def simulation(self):
        return self._simulation 

    @property
    def root_dir(self):
        return self._root_dir

class PhCSweeper(PhCBatch):
    def __init__(self, phc: PhotonicCrystal, configurator_options: dict, runner_options: dict):
        super().__init__(phc, configurator_options, runner_options)
        self._params = {}          # Dictionary to store sweep parameter values
        self._df_sweep_list = []   # List to store DataFrame results for each sweep value
        self._loaded_param = None
        self._loaded_polarization = None  # New property to store the polarization used


    def sweep_parameter(self, param_name, values):
        self._params[param_name] = values
        with open(os.path.join(self._root_dir, f"parameters_{param_name}.txt"), "w") as f:
            f.write(f"{param_name}: " + ", ".join(map(str, values)) + "\n")
        
        # Use tqdm to wrap the iteration over values.
        for value in tqdm(values, desc=f"Sweeping {param_name}", unit="value"):
            self._simulation.directory = os.path.join(self._root_dir, f"{param_name}_{value:.4f}")
            self._simulation.run_hpc(command_line_params={param_name: value})
        
        self._simulation.directory = self._root_dir
        self._loaded_param = None
        self._df_sweep_list = []
        self._loaded_polarization = None


    def load_bands(self, param_name, polarization):
        """
        Loads frequency data for the given parameter and polarization.
        
        :param param_name: The parameter name whose sweep values have been recorded.
        :param polarization: The polarization to use (e.g., "te" or "tm").
        """
        values = self._params.get(param_name, [])
        df_list = []
        for value in values:
            self._simulation.directory = self._root_dir + f"/{param_name}_{value:.4f}"
            df_list.append(self._simulation.load_frequency_data(polarization))
        self._simulation.directory = self._root_dir
        self._df_sweep_list = df_list
        self._loaded_param = param_name
        self._loaded_polarization = polarization

    def plot_sweep_results(self, bands, k_point: tuple):
        """
        Plot the frequency versus sweep parameter results for the given bands.
        
        For each sweep simulation, the method finds the row with k-point values (k1, k2, k3)
        closest to the provided k_point. It then extracts the frequency from the corresponding band
        column (e.g. "te band 1" if polarization is "te"). The sweep parameter values are used as the x-axis.

        :param bands: Iterable of band indices (e.g., [1, 2, 3]) to plot.
        :param k_point: Tuple (k1, k2, k3) to locate a specific k-point in the data.
        """
        if self._loaded_param is None or self._loaded_polarization is None:
            print("No parameter or polarization loaded.")
            return

        sweep_values = self._params.get(self._loaded_param, [])
        if not self._df_sweep_list or len(self._df_sweep_list) != len(sweep_values):
            print("Data not loaded properly.")
            return

        # Dictionary to accumulate frequency values for each band across sweep values
        frequencies_by_band = {band: [] for band in bands}

        # Iterate over each simulation result (DataFrame) corresponding to a sweep value
        for df in self._df_sweep_list:
            frequencies_by_band_partial = self.simulation.get_frequencies_by_band(df, self._loaded_polarization, bands,  k_point)
            for band, freq in frequencies_by_band_partial.items():
                frequencies_by_band[band].append(freq)
        

        # Plot frequency versus sweep parameter value for each band
        plt.figure(figsize=(10, 6))
        for band in bands:
            plt.plot(sweep_values, frequencies_by_band[band],
                     marker='o', linestyle='-', label=f"{self._loaded_polarization} band {band}")

        plt.xlabel(f"Sweep Parameter ({self._loaded_param})")
        plt.ylabel("Frequency")
        plt.title(f"Sweep Results for {self._loaded_polarization.upper()} Polarization")
        plt.legend()
        plt.tight_layout()
        plt.show()


