"""
MPIOptimization for NZI Optimization (Bayesian Optimization Version)
"""

import os
import re
import subprocess
import time
import sys
import tempfile
import argparse
import functools
import threading
import joblib
import json

import numpy as np
from schwimmbad import MPIPool

# Ensure scikit-optimize is installed
try:
    from skopt import Optimizer
except ImportError:
    raise ImportError("scikit-optimize is required. Please install it using: pip install scikit-optimize")

# Import your pre-implemented Simulation class from simulation_handler.
from phc_nzi.simulation_handler import Simulation
from phc_nzi.lsf_job_configurator import LSFJobConfiguration

# Global lock to synchronize printing among threads.
print_lock = threading.Lock()

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

class MPIBayesianOptimizator:
    DEFAULT_MINICONDA_SOURCE = "/zhome/2f/7/202918/miniconda3/etc/profile.d/conda.sh"
    DEFAULT_CONDA_ENV_NAME = "nzi-mp"

    def __init__(self, 
                 simulation_name: str,
                 scheme_script: str,
                 param_names: list,
                 polarization: str = "te", 
                 maxiter: int = 5,
                 batch_size: int = 15,
                 bo_options: dict = None,  
                 bands: list = [2,3,4], 
                 height_slab: float = None, 
                 directory: str = None,
                 fixed_params: dict = None): # <-- ADDED fixed_params
        """
        Parameters:
          simulation_name : str
              Name of the simulation.
          scheme_script : str
              The scheme script to be used by the simulation.
          param_names : list
              Names of the parameters to be optimized (e.g., ["R1", "R2"]).
          polarization : str, optional
              The polarization mode to use when extracting frequency data.
          maxiter : int, optional
              Maximum number of Bayesian Optimization generations (batches).
          batch_size : int, optional
              Number of simulations to run in each batch.
          bo_options : dict, optional
              Options for Bayesian optimization (e.g., {"acq_func": "EI"}).
          bands : list, optional,
                List of band indices to consider for the cost function.
          height_slab : float, optional
                Height of the slab, if applicable.
          directory : str, optional
              Directory to run the simulation in. If None, it will use the current working directory.
          fixed_params : dict, optional
              Parameters to keep constant during the optimization (e.g., {"R1": 0.25}).
        """
        self.simulation = Simulation(simulation_name=simulation_name,
                                     script=scheme_script,
                                     directory=directory,
                                     write_script=False)
        self.param_names = param_names
        self.polarization = polarization
        self.maxiter = maxiter
        self.batch_size = batch_size
        self.scheme_script = scheme_script
        
        self.bo_options = bo_options if bo_options is not None else {
            "random_state": 42,
            "base_estimator": "GP",
            "initial_point_generator": "lhs"
        }
        
        self.data_file = os.path.join(self.simulation.directory, f"{simulation_name}.bo.data")
        self.model_file = os.path.join(self.simulation.directory, f"{simulation_name}_bo_model.pkl")
        self.bands = bands
        
        # --- NEW: Handle fixed parameters and maintain backward compatibility ---
        self.fixed_params = fixed_params if fixed_params is not None else {}
        self.height_slab = height_slab
        if self.height_slab is not None:
            self.fixed_params["h"] = self.height_slab

    def erease_data_file(self):
        with open(self.data_file, "w") as f:
            f.write("")
        with open(self.model_file, "wb") as f:
            f.write(b"")

    @use_nested_temp_directory
    def temp_folder_operations(self, mpb_command_line_params):
        self.simulation.write_scheme_script()
        self.simulation.run_hpc(mpb_command_line_params)
        df = self.simulation.load_frequency_data(self.polarization)
        freqs = self.simulation.get_frequencies_by_band(df, self.polarization, bands=self.bands)
        cost = self._calculate_cost(freqs)
        freq_central_band = self._get_central_band_freq(freqs)
        
        # Guard against division by zero in case of anomalous freq_central_band
        if freq_central_band != 0:
            cost = cost / freq_central_band
            
        return cost, freq_central_band

    def _calculate_cost(self, freqs):
        idx_high = np.max(self.bands)
        idx_low = np.min(self.bands)
        cost = abs(freqs[idx_high] - freqs[idx_low])
        return cost

    def _get_central_band_freq(self, freqs):
        idx_low = np.min(self.bands)
        idx_central = idx_low + 1
        return freqs[idx_central]

    def objective(self, params, current_gen):
        # Handle 1D optimization safely (skopt sometimes passes scalars if 1D)
        if not isinstance(params, (list, tuple, np.ndarray)):
            params = [params]
            
        # Map dynamic parameters
        command_line_params = dict(zip(self.param_names, params))
        
        # Merge in the fixed parameters (including height_slab if it was set)
        command_line_params.update(self.fixed_params)

        cost, freq_central_band = self.temp_folder_operations(command_line_params)

        # Write directly to data log
        with open(self.data_file, "a") as f:
            line = f"Gen: {current_gen}, " + ", ".join(f"{name}: {value}" for name, value in command_line_params.items())
            line += f", cost: {cost}"
            line += f", freq_dirac: {freq_central_band}\n"
            f.write(line)
            
        return cost
    
    def _objective_wrapper(self, args):
        # Unpack the bundled tuple and pass it to the objective
        current_gen, params = args
        return self.objective(params, current_gen)

    def optimize_parameters(self):
        """
        Run Bayesian optimization with MPI parallelism using the ask/tell interface.
        """
        with print_lock:
            print(f"Optimizing parameters {self.param_names} with MPI parallelism using Bayesian Optimization...")
            if self.fixed_params:
                print(f"Fixed parameters: {self.fixed_params}")
            
        pool = MPIPool()
        with pool as mpi_pool:
            if not mpi_pool.is_master():
                mpi_pool.wait()
                return None
            
            # Initialize the Bayesian Optimizer
            optimizer = Optimizer(**self.bo_options)
            
            for gen in range(self.maxiter):
                # Ask the GP for the next batch using the explicit class variable
                x_batch = optimizer.ask(n_points=self.batch_size)
                
                # Bundle the current generation number with each set of parameters
                x_batch_with_gen = [(gen, x) for x in x_batch]
                
                # Evaluate the batch in parallel using the wrapper!
                y_batch = list(mpi_pool.map(self._objective_wrapper, x_batch_with_gen))
                
                # Tell the GP the actual results to update the surrogate model
                optimizer.tell(x_batch, y_batch)
                
                with print_lock:
                    best_cost = min(optimizer.yi)
                    print(f"Generation {gen} complete. Best cost so far: {best_cost:.6f}")

            with print_lock:
                print("Optimization complete.")
                joblib.dump(optimizer, self.model_file)
                best_idx = np.argmin(optimizer.yi)
                best_params = optimizer.Xi[best_idx]
                print("Optimal dynamic parameters found:", best_params)
            
            # Return a simple mock OptimizeResult object to match your previous interface
            class OptimizeResult:
                pass
            result = OptimizeResult()
            result.x = best_params
            result.fun = optimizer.yi[best_idx]
            return result

    def wait_for_job(self, submission_output, poll_interval=10):
        match = re.search(r"Job <(\d+)>", submission_output)
        if match:
            job_id = match.group(1)
            with print_lock:
                print(f"Waiting for job {job_id} to finish...")
            start_time = time.time()
            while True:
                try:
                    out = subprocess.check_output(f"bstat {job_id}", shell=True, universal_newlines=True)
                    lines = out.strip().splitlines()
                    if len(lines) < 2:
                        with print_lock:
                            print(f"Job {job_id} not found in bstat, assuming finished.")
                        break
                    job_line = lines[1]
                    tokens = job_line.split()
                    status = tokens[5] if len(tokens) > 5 else ""
                    elapsed = int(time.time() - start_time)
                    with print_lock:
                        print(f"Job {job_id} status: {status}, elapsed time: {elapsed} sec")
                    if status not in ["RUN", "PEND"]:
                        with print_lock:
                            print(f"Job {job_id} finished.")
                        break
                except subprocess.CalledProcessError:
                    with print_lock:
                        print(f"bstat command failed for job {job_id}; assuming job is finished.")
                    break
                time.sleep(poll_interval)
        else:
            with print_lock:
                print("Could not parse job ID from submission output. Not waiting.")

    def submit_lsf_job(self, 
                       lsf_config: LSFJobConfiguration = LSFJobConfiguration(),
                       conda_source: str = DEFAULT_MINICONDA_SOURCE,    
                       conda_env_name: str = DEFAULT_CONDA_ENV_NAME):
        python_script_name = os.path.abspath(__file__)
        lsf_commands = lsf_config.prepare_lsf_preamble(self.simulation.simulation_name)
        
        output_filepath = os.path.join(self.simulation.directory, f"{self.simulation.simulation_name}.out")
        error_filepath = os.path.join(self.simulation.directory, f"{self.simulation.simulation_name}.err")
        
        lsf_commands.extend([
            f"#BSUB -oo {output_filepath}",
            f"#BSUB -eo {error_filepath}",
            "module purge",
            f"source {conda_source}",
            f"conda activate {conda_env_name}"
        ])

        lsf_commands.append(self._prepare_main_command(lsf_config.num_processors, python_script_name))
        lsf_script = self._merge_lsf_commands(lsf_commands)
        job_script_path = self._write_lsf_script(lsf_script, self.simulation.directory, self.simulation.simulation_name)
        submission_output = self._submit_job(job_script_path)
        return submission_output
        
    def _prepare_main_command(self, nprocs: int, python_script_name: str) -> str:
        cmd = f"mpirun -np {nprocs} python {python_script_name} "
        cmd += self._prepare_command_line_args()
        return cmd

    def _prepare_command_line_args(self) -> str:
        import json
        
        cmd_args = [
            "--run_opt",
            f"--param_names=\"{','.join(self.param_names)}\"",
            f"--simulation_name=\"{self.simulation.simulation_name}\"",
            f"--directory=\"{self.simulation.directory}\"",
            f"--maxiter={self.maxiter}",
            f"--polarization=\"{self.polarization}\"",
            f"--batch_size={self.batch_size}", 
            f"--bands {' '.join(map(str, self.bands))}",
        ]
        if self.height_slab is not None:
            cmd_args.append(f"--height_slab={self.height_slab}")
            
        if self.bo_options:
            bo_options_json = json.dumps(self.bo_options)
            cmd_args.append(f"--bo_options='{bo_options_json}'")
            
        # --- NEW: Append fixed params to CLI args ---
        if self.fixed_params:
            fixed_params_json = json.dumps(self.fixed_params)
            cmd_args.append(f"--fixed_params='{fixed_params_json}'")
            
        return " ".join(cmd_args)
    
    def _merge_lsf_commands(self, lsf_commands) -> str:
        return "\n".join(lsf_commands) + "\n"
    
    def _write_lsf_script(self, lsf_script: str, directory: str, simulation_name: str) -> str:
        job_script_path = os.path.join(f"{directory}", f"{simulation_name}.sh")
        with open(job_script_path, 'w') as script_file:
            script_file.write(lsf_script)
        with print_lock:
            print("LSF job script written to:", job_script_path)
        return job_script_path

    def _submit_job(self, job_script_path: str):
        try:
            submission_output = subprocess.check_output(
                f"bsub < {job_script_path}",
                shell=True,
                universal_newlines=True
            )
            with print_lock:
                print("Job submitted successfully. Submission output:")
                print(submission_output)
            # Wait for the job to finish.
            self.wait_for_job(submission_output, poll_interval=60)
        except subprocess.CalledProcessError as e:
            with print_lock:
                print("Job submission failed:")
                print(e.output)
            submission_output = None
        return submission_output


# Main block:
if __name__ == '__main__':
    import json
    
    parser = argparse.ArgumentParser(description="MPIOptimization with Bayesian Optimization")
    parser.add_argument("--run_opt", action="store_true", help="Run the optimization")
    parser.add_argument("--param_names", type=str, nargs='+', help="Parameter names to optimize (e.g., R1 R2)")
    parser.add_argument("--simulation_name", type=str, help="Simulation name", required=True)
    parser.add_argument("--maxiter", type=int, default=100, help="Maximum number of BO iterations (generations)")
    parser.add_argument("--batch_size", type=int, default=15, help="Batch size per BO iteration")
    parser.add_argument("--bo_options", type=str, default="{}", help="JSON string of skopt Optimizer options")
    parser.add_argument("--polarization", type=str, default="te", help="Polarization mode (e.g., te or tm)")
    parser.add_argument("--bands", type=int, nargs='+', help="Bands to optimize (e.g., 1 2 3)")
    parser.add_argument("--height_slab", type=float, required=False, help="Height of the slab (legacy fixed param)")
    parser.add_argument("--directory", type=str, help="Directory to run the simulation in")
    
    # --- NEW: Argparse for fixed parameters ---
    parser.add_argument("--fixed_params", type=str, default="{}", help="JSON string of fixed parameters (e.g., '{\"R1\": 0.25}')")
    
    args = parser.parse_args()
    
    scheme_script_path = os.path.join(args.directory, args.simulation_name + ".ctl")
    with open(scheme_script_path, "r") as f:
        scheme_script = f.read()

    # Test the scheme script by saving it to a test file.
    with open("test.ctl", "w") as f:
        f.write(scheme_script)  

    if args.run_opt:
        if not args.param_names:
            raise ValueError("Parameter names must be provided")
        param_names = [name.strip() for item in args.param_names for name in item.split(',')]
        height_slab = args.height_slab if hasattr(args, 'height_slab') and args.height_slab is not None else None
        directory = args.directory if hasattr(args, 'directory') and args.directory is not None else None
        
        try:
            parsed_bo_options = json.loads(args.bo_options)
        except json.JSONDecodeError:
            print("Warning: Could not parse bo_options JSON. Using empty dict.")
            parsed_bo_options = {}

        # --- NEW: Parse the JSON string into a python dictionary ---
        try:
            parsed_fixed_params = json.loads(args.fixed_params)
        except json.JSONDecodeError:
            print("Warning: Could not parse fixed_params JSON. Using empty dict.")
            parsed_fixed_params = {}

        optimizer = MPIBayesianOptimizator(simulation_name=args.simulation_name,
                                           scheme_script=scheme_script,
                                           param_names=param_names,
                                           maxiter=args.maxiter,
                                           batch_size=args.batch_size, 
                                           polarization=args.polarization,
                                           bo_options=parsed_bo_options, 
                                           bands=args.bands,
                                           height_slab=height_slab, 
                                           directory=directory,
                                           fixed_params=parsed_fixed_params) # Pass parsed variables

        result = optimizer.optimize_parameters()
        if result is not None:
            with print_lock:
                print("Optimal dynamic parameters found:", result.x)
                print("Minimum frequency difference:", result.fun)
    else:
        with print_lock:
            print("Interactive mode.")
    os.chdir("..")