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
from schwimmbad import MPIPool, MultiPool, SerialPool

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

def _majority(iterable):
    # Convert to a tuple to handle generators (which don't have a length)
    items = tuple(iterable)
    
    # Handle empty iterables (any() returns False, all() returns True)
    if not items:
        return False 
        
    # Count how many items evaluate to True
    truthy_count = sum(bool(x) for x in items)
    
    # Return True if the count is strictly greater than half the total length
    return truthy_count > len(items) / 2

def failsafe_irrep_mapping(target_irreps, degeneracy_tol, full_irrep_map, bands_to_check, log_file=None):
    """
    FAILSAFE FOR MIXED MODES: 
    # Look for a triplet of modes with the same frequency within a certain tolerance
    # If this triplet:
    # 1) exists
    # 2) contains modes from the target irreps (in any number)
    # 3) Before the triplet occurrence, an even number of E modes exist
    # Then we assume that this is a triple degeneracy where the solver mixed the states, 
    # and we change the irreps of the three modes to match exactly the target irreps
    # Example: if A_2, E_1, A_2 are very close in frequency, and we are looking for A_2, E_1, E_1, 
    # we assume the second A_2 is actually an E_1 that got mixed up, and we relabel it as such.

    Parameters:
    - target_irreps: list of irreps we are targeting (e.g. ['A_2', 'E_1', 'E_1'])
    - degeneracy_tol: frequency tolerance to consider modes as degenerate (e.g. 1e-3)
    - full_irrep_map: dict mapping band indices to (irrep, frequency) tuples
    - bands_to_check: list of band indices that were checked for irreps
    - log_file: optional file path to log any corrections made for transparency
    """
    if degeneracy_tol is not None and target_irreps is not None:
        freqs = np.array([freq for _, _, freq in full_irrep_map.values()])
        n_targets = len(target_irreps)

        # PRE-CHECK: Skip correction if a valid consecutive set already exists
        if len(freqs) >= n_targets:
            for i in range(len(freqs) - n_targets + 1):
                # Verify if these consecutive modes are within the degeneracy tolerance
                is_degenerate = all(abs(freqs[i+k] - freqs[i+k+1]) < degeneracy_tol for k in range(n_targets - 1))
                
                if is_degenerate:
                    current_irreps = [full_irrep_map[bands_to_check[i+k]][0] for k in range(n_targets)]
                    if sorted(current_irreps) == sorted(target_irreps):
                        # A perfectly matched set exists in the data. Abort failsafe.
                        return

        # Original correction logic for triplets
        for i in range(len(freqs)-2):
            if abs(freqs[i] - freqs[i+1]) < degeneracy_tol and abs(freqs[i+1] - freqs[i+2]) < degeneracy_tol:
                triplet_irreps = [full_irrep_map[bands_to_check[i]][0], 
                                  full_irrep_map[bands_to_check[i+1]][0], 
                                  full_irrep_map[bands_to_check[i+2]][0]]
                
                # Apply correction if not all of them are already correct independently of order
                # if at least one of the triplet irreps has a low confidence. 
                condition_1 = sorted(triplet_irreps) != sorted(target_irreps)
                condition_2 = _majority(full_irrep_map[bands_to_check[j]][1] < 0.85 for j in range(i, i+3))

                if condition_1 and condition_2:

                    # If there is a mode that is not in the target irreps
                    # And one of the modes has a high confidence, 
                    # we skip the correction because it is likely that the solver got it right and the degeneracy is just a coincidence.
                    condition_3 = any(full_irrep_map[bands_to_check[j]][0] not in target_irreps for j in range(i, i+3))
                    condition_4 = any(full_irrep_map[bands_to_check[j]][1] >= 0.85 for j in range(i, i+3))
                    if condition_3 and condition_4:
                        continue

                    # Check E mode count before this triplet
                    e_count_before = sum(1 for j in range(i) if full_irrep_map[bands_to_check[j]][0] is not None and full_irrep_map[bands_to_check[j]][0].startswith('E'))
                    if e_count_before % 2 == 0:
                        # Log this correction in the log file for transparency
                        msg = f"Degeneracy correction applied at bands {bands_to_check[i:i+3]} with freqs {freqs[i:i+3]} and irreps {triplet_irreps} relabeled to {target_irreps}\n"
                        if log_file is not None:
                            with open(log_file, "a") as f:
                                f.write(msg)
                        else:
                            print(msg)
                            
                        # Relabel the triplet to match target irreps
                        for j, target_irrep in enumerate(target_irreps):
                            full_irrep_map[bands_to_check[i+j]] = (target_irrep, full_irrep_map[bands_to_check[i+j]][1], full_irrep_map[bands_to_check[i+j]][2])
                        break

def find_bands_from_irreps(simulation: Simulation, parity, symmetry_group, target_irreps, irrep_occurrences, degeneracy_tol, irrep_log_file):
        """Dynamically maps requested irreps to actual band indices based on current simulation output.

        Parameters:
        - simulation: Simulation object to extract frequency and irrep data
        - parity: polarization parity to consider (e.g. "te" or "tm")
        - symmetry_group: symmetry group of the system (e.g. "C6v")
        - target_irreps: list of irreps we want to track (e.g. ['A_2', 'E_1', 'E_1'])
        - irrep_occurrences: list of which occurrence of each irrep to track (e.g. [1, 1, 1] for first occurrence)
        - degeneracy_tol: frequency tolerance to consider modes as degenerate for the failsafe correction
        - irrep_log_file: file path to log the irrep mapping and any corrections for transparency
        """
        # Search far enough up the bands to find the required modes
        
        df = simulation.load_frequency_data(parity)
        # find the number of columns starting with the polarization prefix to determine how many bands are available
        search_max = len([col for col in df.columns if col.startswith(parity)])

        bands_to_check = list(range(2, search_max+1))
        
        identified_irreps = simulation.identify_irrep_by_band_indices_with_confidence(
            which_bands=bands_to_check, 
            which_parity=parity, 
            group=symmetry_group
        ) 
        corresponding_freqs = simulation.get_frequencies_by_band(df, parity, bands=bands_to_check)
        full_irrep_map = {
            b: (irrep, confidence, corresponding_freqs[b]) 
            for b, (irrep, confidence) in zip(bands_to_check, identified_irreps)
        }
        
        
        failsafe_irrep_mapping(target_irreps, degeneracy_tol, full_irrep_map, bands_to_check, log_file=irrep_log_file)

        # Handle band crossings and degeneracies by grouping bands into their irreps first.
        global_bands_by_irrep = {}
        for band, (irrep, confidence, freq) in full_irrep_map.items():
            if irrep is None:
                continue
            if irrep not in global_bands_by_irrep:
                global_bands_by_irrep[irrep] = []
            
            global_bands_by_irrep[irrep].append(band) 
        
        # Chunk global bands into distinct occurrence containers based on symmetry rules
        modes_by_irrep = {}
        
        for irrep, bands in global_bands_by_irrep.items(): 
            sorted_bands = sorted(bands)
            # 'E' representations are always 2D (degenerate pairs), 'A'/'B' are 1D
            dim = 2 if irrep.startswith('E') else 1
            
            # Slice sorted bands into chunks matching the representation dimension
            modes_by_irrep[irrep] = [sorted_bands[i:i + dim] for i in range(0, len(sorted_bands), dim)]
            
        occurrences = irrep_occurrences or [1] * len(target_irreps)
        dynamic_bands = []
        used_bands = set()
        
        for irrep, occ in zip(target_irreps, occurrences):
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
                 fixed_params: dict = None,
                 objective_mode: str = "linear",
                 target_cost: float = None,
                 strategy: str = "cl_min",
                 degeneracy_tol: float = None,
                 use_mpi: bool = True,
                 local_workers: int = 1): 
        """
        Initializes the MPI Bayesian Optimizer with dynamic irrep tracking capabilities. 
        """
        self.simulation = Simulation(simulation_name=simulation_name,
                                     script=scheme_script,
                                     directory=directory,
                                     write_script=True) 
        self.param_names = param_names
        self.polarization = polarization
        self.maxiter = maxiter
        self.batch_size = batch_size
        self.scheme_script = scheme_script
        self.objective_mode = objective_mode  
        self.target_cost = target_cost
        self.strategy = strategy
        self.use_mpi = use_mpi
        self.local_workers = local_workers
        self.degeneracy_tol = degeneracy_tol
        
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
            current_bands, full_irrep_map, error_msg = find_bands_from_irreps(
                self.simulation,
                self.polarization,
                self.symmetry_group,
                self.target_irreps,
                self.irrep_occurrences,
                self.degeneracy_tol,
                self.irrep_log_file
            )
            
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

        # Keep logging the RAW un-transformed cost here so your plots don't break
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
            
        # Apply optional flat floor active learning metric
        effective_cost = self.target_cost if (self.target_cost is not None and cost < self.target_cost) else cost

        # --- THE FIX: Transform what the GP model actually receives ---
        if self.objective_mode == "log":
            # Safety clip at 1e-12 to prevent math errors if cost hits absolute 0
            return float(np.log10(max(effective_cost, 1e-12)))
            
        return effective_cost
    
    def _objective_wrapper(self, args):
        current_gen, params = args
        return self.objective(params, current_gen)

    def optimize_parameters(self):
        """Runs the Bayesian optimization loop using MPI parallelism. """
        with print_lock:
            print(f"Optimizing parameters {self.param_names} with MPI parallelism using Bayesian Optimization...")
            print(f"Objective Mode: {self.objective_mode}") 
            if self.target_cost is not None:
                print(f"Target Cost Floor: {self.target_cost}")
            if self.fixed_params:
                print(f"Fixed parameters: {self.fixed_params}")
            if self.target_irreps:
                print(f"Tracking irreps: {self.target_irreps} (occurrences: {self.irrep_occurrences}) in {self.symmetry_group}")
            else:
                print(f"Tracking static bands: {self.bands}")
            
        if self.use_mpi:
            pool = MPIPool()
        else:
            if self.local_workers > 1:
                pool = MultiPool(processes=self.local_workers)
            else:
                pool = SerialPool()
                
        with pool as compute_pool:
            if self.use_mpi and not compute_pool.is_master():
                compute_pool.wait()
                return None
            
            optimizer = Optimizer(**self.bo_options) 
            
            for gen in range(self.maxiter):
                x_batch = optimizer.ask(n_points=self.batch_size, strategy=self.strategy)
                x_batch_with_gen = [(gen, x) for x in x_batch]
                
                y_batch = list(compute_pool.map(self._objective_wrapper, x_batch_with_gen))
                
                optimizer.tell(x_batch, y_batch)
                
                with print_lock:
                    best_cost = min(optimizer.yi)
                    print(f"Generation {gen} complete. Best surrogate score so far: {best_cost:.6f}")

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
            f"--objective_mode=\"{self.objective_mode}\"",  
            f"--strategy=\"{self.strategy}\"",
        ]
        
        if self.target_cost is not None:
            cmd_args.append(f"--target_cost={self.target_cost}")
        if self.degeneracy_tol is not None:
            cmd_args.append(f"--degeneracy_tol={self.degeneracy_tol}")
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
    
    # --- ADDED NEW ARGUMENT ---
    parser.add_argument("--objective_mode", type=str, choices=["linear", "log"], default="linear",
                        help="Scale mapping of target optimization. Linear (default) or log space.")
    parser.add_argument("--target_cost", type=float, default=None,
                        help="Optional target cost to create a flat floor below this value.")
    parser.add_argument("--strategy", type=str, default="cl_min",
                        help="Strategy to use for ask() in Bayesian Optimization.")
    parser.add_argument("--degeneracy_tol", type=float, default=None,
                        help="Tolerance for frequency-proximity degeneracy failsafe (e.g. 1e-3).")
    parser.add_argument("--local", action="store_true", help="Run locally using multiprocessing instead of MPI")
    parser.add_argument("--local_workers", type=int, default=1, help="Number of local workers if running locally")
    
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
                                           fixed_params=parsed_fixed_params,
                                           objective_mode=args.objective_mode,
                                           target_cost=args.target_cost,
                                           strategy=args.strategy,
                                           degeneracy_tol=args.degeneracy_tol,
                                           use_mpi=not args.local,
                                           local_workers=args.local_workers) 

        result = optimizer.optimize_parameters()
        if result is not None:
            with print_lock:
                print("Optimal dynamic parameters found:", result.x)
                print("Minimum objective function score:", result.fun)
    else:
        with print_lock:
            print("Interactive mode.")
    os.chdir("..")


