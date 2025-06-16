from phc_nzi.lsf_job_submitter import LSFJob, use_nested_temp_directory
import argparse
import os
from schwimmbad import MPIPool
import pandas as pd
from phc_nzi.simulation_handler import Simulation, logging
import time


class LSFSweeper(LSFJob):
    def __init__(self, 
                 simulation_name: str,
                 scheme_script: str,     
                 directory: str,
                 polarization: str,
                 bands: list,
                 param_name: str, 
                 param_values: list,
                 others_command_line_params: dict = {}
                 ):
        super().__init__(simulation_name, scheme_script, directory)
        self.simulation_name = simulation_name
        self.directory = directory
        self.scheme_script = scheme_script
        self.polarization = polarization
        self.bands = bands
        self.param_name = param_name
        self.param_values = param_values
        self.others_command_line_params = others_command_line_params
        # Define the results CSV file.
        self.data_file = os.path.join(directory, simulation_name + "_results.csv")

    def run(self):
        with MPIPool() as pool:
            if not pool.is_master():
                pool.wait()
                return None

            # Clear the CSV file before starting the sweep.
            with open(self.data_file, "w") as f:
                pass

            # Create a temporary directory for each worker.
            tempdirs = []
            scripts = []
            names = []
            for param_value in self.param_values:
                random_unique_id = os.urandom(4).hex()
                tempdir = os.path.join(self.directory, f"temp_{random_unique_id}")
                os.makedirs(tempdir, exist_ok=True)
                tempdirs.append(tempdir)
                scripts.append(self.scheme_script)
                with open(os.path.join(tempdir, f"{random_unique_id}.ctl"), "w") as f:
                    f.write(self.scheme_script)
                    time.sleep(0.1)  # Optional: sleep to
                names.append(random_unique_id)
                time.sleep(0.1)  # Optional: sleep to avoid


            

            # Map the work to the pool: each worker returns a row.
            rows = pool.map(self._process_step, zip(self.param_values, tempdirs, names, scripts))

            # Now that all workers have finished, the master writes the rows to CSV.
            self._write_rows_to_csv(rows)

            with self.print_lock:
                print("Sweep completed")

    def _process_step(self, task):
        """Execute a single parameter sweep step and return a row (as a dict)."""
        # Create the command-line parameter set for this sweep.
        param_value, tempdir, name, script = task
        command_line_params = {self.param_name: param_value}
        command_line_params.update(self.others_command_line_params)
        simulation = Simulation(
            name, 
            script, 
            tempdir,
            write_script=False,
            log_level=logging.DEBUG,
        )

        print(f"Setup simulation in {tempdir} with parameters: {command_line_params}")
        

        freqs = self.workers_operations(
            simulation,
            command_line_params,
            self.polarization,
            self.bands
            
        )

        row = self._prepare_row(command_line_params, freqs)
        return row

    def _prepare_row(self, command_line_params, freqs):
        row = {}
        # Add sweep parameters.
        row.update(command_line_params)
        # Add frequency data for each band.
        for band, frequency in freqs.items():
            row[f'band_{band}'] = frequency
        return row

    def _write_rows_to_csv(self, rows):
        """Write all rows to the CSV file using a pandas DataFrame."""
        df = pd.DataFrame(rows)
        df.to_csv(self.data_file, index=False)

    def load_results(self):
        """Load the CSV data into a DataFrame when needed."""
        return pd.read_csv(self.data_file)

    @use_nested_temp_directory
    def temp_folder_operations(self, mpb_command_line_params):
        # These methods are assumed to be implemented in the base class or within your simulation.
        self.simulation.write_scheme_script()  
        self.simulation.run_hpc(mpb_command_line_params)
        df = self.simulation.load_frequency_data(self.polarization)
        freqs = self.simulation.get_frequencies_by_band(df, self.polarization, bands=self.bands)
        return freqs
    
    def workers_operations(self, simulation: Simulation, mpb_command_line_params, polarization: str, bands: list):
        # These methods are assumed to be implemented in the base class or within your simulation.
        simulation.logger.info(f"Running simulation with parameters: {mpb_command_line_params} in {simulation.directory}")
        simulation.run_hpc(mpb_command_line_params)
        df = simulation.load_frequency_data(polarization)
        freqs = simulation.get_frequencies_by_band(df, polarization, bands=bands)
        return freqs
    
    def _get_script_abs_path(self):
        return os.path.abspath(__file__)

    def _prepare_command_line_args(self):
        cmd_args = [
            f"--simulation_name={self.simulation_name}",
            f"--directory={self.directory}",
            f"--param_to_sweep={self.param_name}",
            f"--sweep_values={','.join(map(str, self.param_values))}",
            f"--cmd_params_names={','.join(self.others_command_line_params.keys())}",
            f"--cmd_params_values={','.join(map(str, self.others_command_line_params.values()))}",
            f"--polarization={self.polarization}",
            f"--bands={','.join(map(str, self.bands))}"
        ]
        return " ".join(cmd_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a parameter sweep on an MPB simulation")
    parser.add_argument("--run_sweep", type=str, default="true", help="Whether to run the sweep")
    parser.add_argument("--simulation_name", type=str, required=True, help="Name of the simulation")
    parser.add_argument("--param_to_sweep", type=str, required=True, help="Name of the parameter to sweep")
    parser.add_argument("--sweep_values", type=str, required=True, help="Comma-separated list of parameter values")
    parser.add_argument("--cmd_params_names", type=str, required=True, help="Comma-separated list of command line parameter names")
    parser.add_argument("--cmd_params_values", type=str, required=True, help="Comma-separated list of command line parameter values")
    parser.add_argument("--polarization", type=str, required=True, help="Polarization to consider")
    parser.add_argument("--bands", type=str, required=True, help="Comma-separated list of bands to consider")
    parser.add_argument("--directory", type=str, required=True, help="Directory to store the simulation data")
    args = parser.parse_args()

    run_sweep = args.run_sweep.lower() in ['true', '1', 'yes']
    sweep_values = list(map(float, args.sweep_values.split(',')))
    bands = list(map(int, args.bands.split(',')))
    others_command_line_params = dict(zip(args.cmd_params_names.split(','), args.cmd_params_values.split(',')))

    # Read the scheme script from the provided path.
    scheme_script_path = os.path.join(args.directory, args.simulation_name + ".ctl")
    with open(scheme_script_path, "r") as f:
        scheme_script = f.read()

    # Create the LSFSweeper instance.
    sweeper = LSFSweeper(
        simulation_name=args.simulation_name,
        scheme_script=scheme_script,
        directory=args.directory,
        polarization=args.polarization,
        bands=bands,
        param_name=args.param_to_sweep,
        param_values=sweep_values,
        others_command_line_params=others_command_line_params
    )

    # Run the sweep if specified.
    if run_sweep:
        sweeper.run()
    else:
        print("Sweep not executed. Set --run_sweep to 'true' to run the sweep.")



