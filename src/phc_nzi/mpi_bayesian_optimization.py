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
                 bands: list = None, 
                 target_irreps: list = None,       
                 irrep_occurrences: list = None,   
                 symmetry_group: str = "C6v",      
                 height_slab: float = None, 
                 directory: str = None,
                 fixed_params: dict = None):
        """
        Initializes the MPI Bayesian Optimizer with dynamic irrep tracking capabilities. 
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
        self.irrep_log_file = os.path.join(self.simulation.directory, f"{simulation_name}.irreps.log")
        
        # Fallback to [2,3,4] only if neither bands nor irreps are provided
        if bands is None and target_irreps is None:
            self.bands = [2, 3, 4]
        else:
            self.bands = bands

        self.target_irreps = target_irreps
        self.irrep_occurrences = irrep_occurrences
        self.symmetry_group = symmetry_group
        
        # Handle fixed parameters and maintain backward compatibility 
        self.fixed_params = fixed_params if fixed_params is not None else {}
        self.height_slab = height_slab
        if self.height_slab is not None:
            self.fixed_params["h"] = self.height_slab

    def erease_data_file(self):
        """Clears previous data and log files before starting a new run. """
        with open(self.data_file, "w") as f:
            f.write("")
        with open(self.model_file, "wb") as f:
            f.write(b"")
        with open(self.irrep_log_file, "w") as f:
            f.write("")

    def _find_bands_from_irreps(self):
        """Dynamically maps requested irreps to actual band indices based on current simulation output. [cite: 1, 2]"""
        # Search far enough up the bands to find the required modes
        search_max = max(15, len(self.target_irreps) * 3) 
        bands_to_check = list(range(2, search_max + 1))
        
        identified_irreps = self.simulation.identify_irrep_by_band_indices(
            which_bands=bands_to_check, 
            which_parity=self.polarization, 
            group=self.symmetry_group
        ) 
        
        # Create a full mapping for the log file: {band: "Irrep"}
        full_irrep_map = {b: i for b, i in zip(bands_to_check, identified_irreps)}
        
        # Group bands into mode clusters
        modes_by_irrep = {} 
        current_irrep = None
        current_cluster = []
        
        for band, irrep in zip(bands_to_check, identified_irreps):
            if irrep is None:
                continue
            if irrep == current_irrep:
                current_cluster.append(band)
            else:
                if current_irrep is not None:
                    if current_irrep not in modes_by_irrep:
                        modes_by_irrep[current_irrep] = []
                    modes_by_irrep[current_irrep].append(current_cluster)
                current_irrep = irrep
                current_cluster = [band]
                
        if current_irrep is not None:
            if current_irrep not in modes_by_irrep:
                modes_by_irrep[current_irrep] = []
            modes_by_irrep[current_irrep].append(current_cluster)
            
        occurrences = self.irrep_occurrences or [1] * len(self.target_irreps)
        dynamic_bands = []
        used_bands = set()
        
        for irrep, occ in zip(self.target_irreps, occurrences):
            if irrep not in modes_by_irrep or len(modes_by_irrep[irrep]) < occ:
                error_msg = f"Missing occurrence {occ} for {irrep}. Found: {modes_by_irrep}"
                return None, full_irrep_map, error_msg
            
            cluster = modes_by_irrep[irrep][occ - 1]
            
            assigned_band = None
            for b in cluster:
                if b not in used_bands:
                    assigned_band = b
                    break
            
            if assigned_band is None:
                error_msg = f"Not enough bands in occurrence {occ} of {irrep}. Cluster: {cluster}"
                return None, full_irrep_map, error_msg
                
            dynamic_bands.append(assigned_band)
            used_bands.add(assigned_band)
            
        return dynamic_bands, full_irrep_map, None

    @use_nested_temp_directory
    def temp_folder_operations(self, mpb_command_line_params, current_gen=None):
        """Handles the simulation execution and dynamic band mapping. """
        self.simulation.write_scheme_script()
        self.simulation.run_hpc(mpb_command_line_params, mpi = False, version = "mpb-dev/1.11.2-dev")
        df = self.simulation.load_frequency_data(self.polarization)
        
        full_irrep_map = {}
        error_msg = None
        
        # Determine current band indices dynamically and create a log label
        if self.target_irreps is not None:
            current_bands, full_irrep_map, error_msg = self._find_bands_from_irreps()
            
            if error_msg:
                # Continuous penalty (1.0) regardless of the generation to avoid GP tearing. 
                tracking_label = f"FAILED: {error_msg}"
                return 1.0, 0.0, tracking_label, full_irrep_map
                
            tracking_label = f"Mapped to bands {current_bands}"
        else:
            current_bands = self.bands
            tracking_label = f"Static bands {current_bands}"
            
        freqs = self.simulation.get_frequencies_by_band(df, self.polarization, bands=current_bands) 
        
        cost = self._calculate_cost(freqs, current_bands)
        freq_central_band = self._get_central_band_freq(freqs, current_bands)
        
        if freq_central_band != 0:
            cost = cost / freq_central_band
            
        return cost, freq_central_band, tracking_label, full_irrep_map

    def _calculate_cost(self, freqs, current_bands):
        """Computes the difference between high and low band indices. """
        idx_high = np.max(current_bands)
        idx_low = np.min(current_bands)
        cost = abs(freqs[idx_high] - freqs[idx_low])
        return cost

    def _get_central_band_freq(self, freqs, current_bands):
        """Returns the frequency of the middle band in the set. """
        sorted_bands = sorted(current_bands)
        if len(sorted_bands) >= 3:
            idx_central = sorted_bands[1]
        else:
            idx_central = sorted_bands[0]
        return freqs[idx_central]

    def objective(self, params, current_gen):
        """Objective function called by the Bayesian Optimizer. """
        if not isinstance(params, (list, tuple, np.ndarray)):
            params = [params]
            
        command_line_params = dict(zip(self.param_names, params))
        command_line_params.update(self.fixed_params)

        cost, freq_central_band, tracking_label, full_irrep_map = self.temp_folder_operations(
            command_line_params, 
            current_gen=current_gen
        )

        with open(self.data_file, "a") as f:
            line = f"Gen: {current_gen}, " + ", ".join(f"{name}: {value}" for name, value in command_line_params.items())
            line += f", cost: {cost:.6f}"
            line += f", freq_dirac: {freq_central_band:.6f}"
            line += f", Tracking: [{tracking_label}]\n"
            f.write(line)
            
        with open(self.irrep_log_file, "a") as f:
            params_str = ", ".join(f"{name}: {value}" for name, value in command_line_params.items())
            irrep_str = str(full_irrep_map) if full_irrep_map else "None"
            f.write(f"Gen: {current_gen} | Params: [{params_str}] | Map: {irrep_str} | Status: {tracking_label}\n")
            
        return cost
    
    def _objective_wrapper(self, args):
        current_gen, params = args
        return self.objective(params, current_gen)

    def optimize_parameters(self):
        """Runs the Bayesian optimization loop using MPI parallelism. """
        with print_lock:
            print(f"Optimizing parameters {self.param_names} with MPI parallelism using Bayesian Optimization...")
            if self.fixed_params:
                print(f"Fixed parameters: {self.fixed_params}")
            if self.target_irreps:
                print(f"Tracking irreps: {self.target_irreps} (occurrences: {self.irrep_occurrences}) in {self.symmetry_group}")
            else:
                print(f"Tracking static bands: {self.bands}")
            
        pool = MPIPool()
        with pool as mpi_pool:
            if not mpi_pool.is_master():
                mpi_pool.wait()
                return None
            
            optimizer = Optimizer(**self.bo_options) 
            
            for gen in range(self.maxiter):
                x_batch = optimizer.ask(n_points=self.batch_size)
                x_batch_with_gen = [(gen, x) for x in x_batch]
                
                y_batch = list(mpi_pool.map(self._objective_wrapper, x_batch_with_gen))
                
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
            
            class OptimizeResult:
                pass
            result = OptimizeResult()
            result.x = best_params
            result.fun = optimizer.yi[best_idx]
            return result

    def wait_for_job(self, submission_output, poll_interval=10):
        """Polls LSF for job completion. """
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
        """Prepares and submits an LSF job script. """
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
        ]
        
        if self.bands:
            cmd_args.append(f"--bands {' '.join(map(str, self.bands))}")
        if self.target_irreps:
            cmd_args.append(f"--target_irreps {' '.join(self.target_irreps)}")
        if self.irrep_occurrences:
            cmd_args.append(f"--irrep_occurrences {' '.join(map(str, self.irrep_occurrences))}")
        if self.symmetry_group:
            cmd_args.append(f"--symmetry_group {self.symmetry_group}")
        
        if self.height_slab is not None:
            cmd_args.append(f"--height_slab={self.height_slab}")
            
        if self.bo_options:
            bo_options_json = json.dumps(self.bo_options)
            cmd_args.append(f"--bo_options='{bo_options_json}'")
            
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
            self.wait_for_job(submission_output, poll_interval=60)
        except subprocess.CalledProcessError as e:
            with print_lock:
                print("Job submission failed:")
                print(e.output)
            submission_output = None
        return submission_output

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
    parser.add_argument("--target_irreps", type=str, nargs='+', help="Target irreps (e.g., A_1 E_2 E_2)")
    parser.add_argument("--irrep_occurrences", type=int, nargs='+', help="Which occurrence to use (e.g., 1 1 1)")
    parser.add_argument("--symmetry_group", type=str, default="C6v", help="Symmetry group (e.g., C6v)")
    
    parser.add_argument("--height_slab", type=float, required=False, help="Height of the slab (legacy fixed param)")
    parser.add_argument("--directory", type=str, help="Directory to run the simulation in")
    parser.add_argument("--fixed_params", type=str, default="{}", help="JSON string of fixed parameters (e.g., '{\"R1\": 0.25}')")
    
    args = parser.parse_args()
    
    scheme_script_path = os.path.join(args.directory, args.simulation_name + ".ctl")
    with open(scheme_script_path, "r") as f:
        scheme_script = f.read()

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
                                           target_irreps=args.target_irreps,
                                           irrep_occurrences=args.irrep_occurrences,
                                           symmetry_group=args.symmetry_group,
                                           height_slab=height_slab, 
                                           directory=directory,
                                           fixed_params=parsed_fixed_params) 

        result = optimizer.optimize_parameters()
        if result is not None:
            with print_lock:
                print("Optimal dynamic parameters found:", result.x)
                print("Minimum frequency difference:", result.fun)
    else:
        with print_lock:
            print("Interactive mode.")
    os.chdir("..")