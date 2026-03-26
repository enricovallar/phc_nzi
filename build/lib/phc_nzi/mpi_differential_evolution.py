#!/usr/bin/env python
"""
MPIOptimization for NZI Optimization 
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

import numpy as np
from scipy.optimize import differential_evolution
from schwimmbad import MPIPool

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

# --- Custom Map Wrapper to satisfy SciPy's workers argument ---
class CustomMapWrapper:
    def __init__(self, pool, tracking_file):
        self.pool = pool
        self.tracking_file = tracking_file
        self.generation = 0

    def __call__(self, func, iterable):
        # Master process writes the current generation to the tracking file
        with open(self.tracking_file, "w") as f:
            f.write(str(self.generation))
            
        result = self.pool.map(func, iterable)
        self.generation += 1
        return result

    def __int__(self):
        # Return 0 so that int(worker)==-1 check in differential_evolution passes.
        return 0

class MPIdeOptimizator:
    DEFAULT_MINICONDA_SOURCE = "/zhome/2f/7/202918/miniconda3/etc/profile.d/conda.sh"
    DEFAULT_CONDA_ENV_NAME = "nzi-mp"

    def __init__(self, 
                 simulation_name: str,
                 scheme_script: str,
                 param_names: list,
                 polarization: str = "te", 
                 maxiter: int = 5, 
                 de_options: dict = {"strategy": "rand1bin", "popsize": 15}, 
                 param_bounds: list = None, 
                 bands: list = [2,3,4], 
                 height_slab: float = None, 
                 directory: str = None):
        """
        Parameters:
          simulation_name : str
              Name of the simulation.
          scheme_script : str
              The scheme script to be used by the simulation.
          param_names : list
              Names of the parameters to be optimized (e.g., ["R1", "R2"]).
          polarization : str, optional
              The polarization mode to use when extracting frequency data (e.g., "te" or "tm").
              Default is "te".
          maxiter : int, optional
              Maximum number of DE iterations.
          de_options : dict, optional
              Additional keyword options to pass to SciPy's differential_evolution.
              For example, {"popsize": 15, "strategy": "rand1bin"}.
          param_bounds : list, optional
              List of bounds for each parameter, in the form of tuples, e.g. [(lb, ub), ...].
        """
        self.simulation = Simulation(simulation_name=simulation_name,
                                     script=scheme_script,
                                     directory=directory,
                                     write_script=False)
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.polarization = polarization
        self.maxiter = maxiter
        self.de_options = de_options if de_options is not None else {}
        self.scheme_script = scheme_script
        self.data_file = os.path.join(self.simulation.directory, f"{simulation_name}.de.data")
        self.tracking_file = os.path.join(self.simulation.directory, f"{simulation_name}.gen.txt")
        self.bands = bands
        self.height_slab = height_slab


    def erease_data_file(self):
        with open(self.data_file, "w") as f:
            f.write("")

    @use_nested_temp_directory
    def temp_folder_operations(self, mpb_command_line_params):
        self.simulation.write_scheme_script()
        self.simulation.run_hpc(mpb_command_line_params)
        df = self.simulation.load_frequency_data(self.polarization)
        freqs = self.simulation.get_frequencies_by_band(df, self.polarization, bands=self.bands)
        cost = self._calculate_cost(freqs)
        freq_central_band = self._get_central_band_freq(freqs)
        cost = cost/freq_central_band
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

    def objective(self, params):
        command_line_params = dict(zip(self.param_names, params))
        if self.height_slab is not None:
            command_line_params["h"] = self.height_slab
        cost, freq_central_band = self.temp_folder_operations(command_line_params)
        
        # Read current generation from the tracking file
        try:
            with open(self.tracking_file, "r") as f:
                current_gen = f.read().strip()
        except Exception:
            current_gen = "unknown"

        with open(self.data_file, "a") as f:
            # Prepend the generation to your output string
            line = f"Gen: {current_gen}, {self.param_names[0]}: {params[0]}, {self.param_names[1]}: {params[1]}"
            line += f", cost: {cost}"
            line += f", freq_dirac: {freq_central_band}\n"
            f.write(line)
        return cost

    def optimize_parameters(self):
        """
        Run DE optimization with MPI parallelism.
        
        Returns:
          result : OptimizeResult from SciPy's differential_evolution.
        """
        with print_lock:
            print("Optimizing parameters with MPI parallelism...")
        pool = MPIPool()
        with pool as mpi_pool:
            if not mpi_pool.is_master():
                mpi_pool.wait()
                return None
            bounds = self.param_bounds
            custom_map = CustomMapWrapper(mpi_pool, self.tracking_file)
            result = differential_evolution(self.objective, bounds,
                                            workers=custom_map,
                                            maxiter=self.maxiter,
                                            **self.de_options)
            with print_lock:
                print("Optimization complete.")
                print("Optimal parameters found:", result)
            return result

    def wait_for_job(self, submission_output, poll_interval=10):
        """
        Wait until the submitted LSF job is finished by polling with 'bstat'.
        This version does not use a progress bar but still waits until the job is finished.
        """
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
        cmd_args = [
            "--run_opt",
            f"--param_names=\"{','.join(self.param_names)}\"",
            f"--simulation_name=\"{self.simulation.simulation_name}\"",
            f"--directory=\"{self.simulation.directory}\"",
            f"--maxiter={self.maxiter}",
            f"--polarization=\"{self.polarization}\"",
            f"--param_bounds " + " ".join([f"{lb},{ub}" for (lb, ub) in self.param_bounds]),
            f"--popsize={self.de_options.get('popsize', 15)}",
            f"--strategy=\"{self.de_options.get('strategy', 'rand1bin')}\"",
            f"--bands {' '.join(map(str, self.bands))}",

        ]
        if self.height_slab is not None:
            cmd_args.append(f"--height_slab={self.height_slab}")
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
            self.wait_for_job(submission_output, poll_interval=10)
        except subprocess.CalledProcessError as e:
            with print_lock:
                print("Job submission failed:")
                print(e.output)
            submission_output = None
        return submission_output


# Main block:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MPIOptimization")
    parser.add_argument("--run_opt", action="store_true", help="Run the optimization")
    parser.add_argument("--param_names", type=str, nargs='+', help="Parameter names (e.g., R1 R2)")
    parser.add_argument("--simulation_name", type=str, help="Simulation name", required=True)
    parser.add_argument("--maxiter", type=int, default=100, help="Maximum number of DE iterations")
    parser.add_argument("--popsize", type=int, default=15, help="Population size for DE")
    parser.add_argument("--strategy", type=str, default="rand1bin", help="DE strategy")
    parser.add_argument("--polarization", type=str, default="te", help="Polarization mode (e.g., te or tm)")
    parser.add_argument("--param_bounds", type=str, nargs='+', help="Parameter bounds (e.g., 0.1,0.9 0.1,0.9)", required=True)
    parser.add_argument("--bands", type=int, nargs='+', help="Bands to optimize (e.g., 1 2 3)")
    parser.add_argument("--height_slab", type=float, required=False, help="Height of the slab")
    parser.add_argument("--directory", type=str, help="Directory to run the simulation in")
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
        simulation_name = args.simulation_name
        maxiter = args.maxiter
        polarization = args.polarization
        bands = args.bands
        height_slab = args.height_slab if hasattr(args, 'height_slab') and args.height_slab is not None else None
        directory = args.directory if hasattr(args, 'directory') and args.directory is not None else None
        
        param_bounds = []
        for b in args.param_bounds:
            parts = b.split(',')
            if len(parts) != 2:
                raise ValueError(f"Bound {b} is not in the format 'lower,upper'")
            lb, ub = parts
            param_bounds.append((float(lb), float(ub)))
        
        de_options = {
            "popsize": args.popsize,
            "strategy": args.strategy
        }
        optimizer = MPIdeOptimizator(simulation_name=simulation_name,
                                     scheme_script=scheme_script,
                                     param_names=param_names,
                                     maxiter=maxiter,
                                     polarization=polarization,
                                     param_bounds=param_bounds, 
                                     de_options=de_options,
                                     bands=bands,
                                     height_slab=height_slab, 
                                     directory=directory)    

        result = optimizer.optimize_parameters()
        if result is not None:
            with print_lock:
                print("Optimal parameters found:", result.x)
                print("Minimum frequency difference:", result.fun)
    else:
        with print_lock:
            print("Interactive mode.")
    os.chdir("..")
