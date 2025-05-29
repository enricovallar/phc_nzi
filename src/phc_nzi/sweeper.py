import functools
import tempfile
from simulation_handler import Simulation
import pandas as pd


def use_nested_temp_directory(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Save the current simulation directory (if it exists)
        original_directory = getattr(self.simulation, "directory", None)
        with tempfile.TemporaryDirectory() as tempdir:
            # Set the simulation directory to the temporary directory
            self.simulation.directory = tempdir
            result = func(self, *args, **kwargs)
        # Optionally restore the original directory after the call
        if original_directory is not None:
            self.simulation.directory = original_directory
        return result
    return wrapper

class ParamSweeper():
    def __init__(self, simulation: Simulation, param_name: str, values: list, 
                 polarization: str, bands: list, other_command_line_params: dict = {}):
        self.param_name = param_name
        self.values = values
        self.simulation = simulation
        self.data_file = simulation.simulation_name + "_sweep.data"
        self.polarization = polarization
        self.bands = bands
        self.data = pd.DataFrame()  
        self.other_command_line_params = other_command_line_params

    
    @use_nested_temp_directory
    def temp_folder_operations(self, mpb_command_line_params):
        self.simulation.write_scheme_script()
        self.simulation.run_hpc(mpb_command_line_params)
        df = self.simulation.load_frequency_data(self.polarization)
        freqs = self.simulation.get_frequencies_by_band(df, self.polarization, bands=self.bands)
        return freqs
    


    def _process_step(self, param_value, command_line_params):
        """Execute a single parameter sweep step and return the row data."""
        freqs = self.temp_folder_operations(command_line_params)
        # Prepare the data for this step
        row = {'param_name': self.param_name, 'param_value': param_value}
        for band, frequency in freqs.items():
            row[f'band_{band}'] = frequency
        return row

    def run(self):
        total_steps = len(self.values)
        
        for i, param_value in enumerate(self.values):
            print(f"step {i+1}/{total_steps}: {self.param_name} = {param_value}")
            command_line_params = {self.param_name: param_value}
            command_line_params.update(self.other_command_line_params)
            row = self._process_step(param_value, command_line_params)
            # Append the row to the internal DataFrame property and persist it
            self.data = pd.concat([self.data, pd.DataFrame([row])], ignore_index=True)
            self._save_df(self.data)

    
    def _save_df(self, df):
        df.to_csv(self.data_file, index=False)
        return df
    
    def load_df(self, data_file: str | None =  None):  
        data_file_to_load = self.data_file if data_file is None else data_file
        return pd.read_csv(data_file_to_load)
    
    def plot(self, data_file: str | None = None, figsize=(10, 6)):
        import matplotlib.pyplot as plt
        
        data = self.load_df(data_file)
        
        # Create a single figure for all bands
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each band
        for band in self.bands:
            ax.plot(data['param_value'], data[f'band_{band}'], marker='o', label=f'Band {band}')
        
        # Set labels and title
        ax.set_xlabel(self.param_name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Band frequencies vs {self.param_name}')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return data
    





    


