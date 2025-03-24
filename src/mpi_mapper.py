#!/usr/bin/env python
"""
MPI Gamma Mapping for Photonic Crystal Bands at Gamma Point

This script uses MPI to evaluate a grid of parameter values in parallel.
For each grid point the simulation is run (using the provided Simulation class)
and the frequency bands at the gamma point (k=0) are extracted.
This is useful when you wish to study how the gamma point bands change
as you vary parameters (e.g. radii or other design parameters).

Usage:
    python MPIGammaMapper.py --run_mapping --param_names "R1" "R2" --simulation_name MySimulation \
       --polarization te --grid_bounds 0.1,0.9 0.1,0.9 --resolution 50,50 --band_indices 0,1,2,3,4
"""

import os, re, subprocess, time, sys, tempfile, argparse
import numpy as np
import matplotlib.pyplot as plt
from schwimmbad import MPIPool
from tqdm import tqdm
import functools
from src.simulation_handler import LSFJobConfiguration

# Import your pre-implemented Simulation class from simulation_handler.
from simulation_handler import Simulation
import threading

# Global lock to synchronize printing among threads.
print_lock = threading.Lock()

# --- Custom Map Wrapper (for SciPy's worker compatibility) ---
class CustomMapWrapper:
    def __init__(self, pool):
        self.pool = pool
    def __call__(self, func, iterable):
        return self.pool.map(func, iterable)
    def __int__(self):
        return 0

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

class MPIGammaMapper:

    DEFAULT_MINICONDA_SOURCE = "/zhome/2f/7/202918/miniconda3/etc/profile.d/conda.sh"
    DEFAULT_CONDA_ENV_NAME = "nzi-mp"


    def __init__(self, 
                 simulation_name: str,
                 scheme_script: str,
                 param_names: list,
                 grid_bounds: list,
                 grid_points: tuple,
                 polarization: str = "te",
                 band_indices: list = None):
        """
        Parameters:
        -----------
        simulation_name : str
            The simulation folder/name.
        scheme_script : str
            Contents of the control (.ctl) file.
        param_names : list
            Names of the parameters (e.g. ["R1", "R2"]).
        grid_bounds : list
            List of tuples [(lower, upper), (lower, upper)] for each parameter.
        grid_points : tuple
            Number of grid points (Nx, Ny).
        polarization : str, optional
            Polarization mode ("te" or "tm").
        band_indices : list, optional
            List of band indices to extract (e.g., [0,1,2,3,4]). If None, all bands are returned.
        """
        self.simulation = Simulation(simulation_name=simulation_name,
                                     script=scheme_script)
        self.simulation_name = simulation_name
        self.scheme_script = scheme_script
        self.param_names = param_names
        self.grid_bounds = grid_bounds
        self.grid_points = grid_points
        self.polarization = polarization
        self.band_indices = band_indices
        self.mapping_log = os.path.join(simulation_name, f"{simulation_name}_gamma_mapping.log")

    @use_nested_temp_directory
    def _temp_folder_operations(self, mpb_command_line_params):
        self.simulation.write_scheme_script()
        self.simulation.run_hpc(mpb_command_line_params=mpb_command_line_params)
        self._get_modes_symmetryes()

    def _get_modes_symmetryes(self):
        pass

    def _write_mapping_data(self):
        pass



    def evaluate_grid_point(self, task):
        """
        Evaluate one grid point.
        The task is a tuple: (i, j, param0, param1).
        This method:
          - Sets the simulation parameters,
          - Writes a temporary control file,
          - Runs the simulation,
          - Loads the frequency data,
          - Extracts the gamma (k=0) band frequencies,
          - Logs the result,
          - Returns (i, j, selected_freqs).
        """
        i, j, param0, param1 = task
        params = [param0, param1]
        cmd_params = dict(zip(self.param_names, params))
        
        self._temp_folder_operations(cmd_params)
        self._write_mapping_data(i, j, cmd_params)
        
        
    def _make_mesh(self):
        p0_vals = np.linspace(self.grid_bounds[0][0], self.grid_bounds[0][1], self.grid_points[0])
        p1_vals = np.linspace(self.grid_bounds[1][0], self.grid_bounds[1][1], self.grid_points[1])
        X, Y = np.meshgrid(p0_vals, p1_vals)
        return X, Y
    
    def _prepare_tasks(self, X, Y):
        tasks = []
        for i in range(self.grid_points[0]):
            for j in range(self.grid_points[1]):
                param0 = X[j, i]
                param1 = Y[j, i]
                tasks.append((i, j, param0, param1))
        return tasks

    def run_mapping(self):
        """
        Runs the simulation over a grid of parameter values using MPI,
        and collects the gamma point band frequencies.
        
        Returns:
        --------
        X, Y : 2D arrays (meshgrid of parameter values)
        freq_dict : dict
            Dictionary mapping each band index to a 2D array (grid) of frequencies.
        """
        X, Y = self._make_mesh()           
        tasks = self._prepare_tasks(X, Y)
        
        with MPIPool() as mpi_pool:
            if not mpi_pool.is_master():
                mpi_pool.wait()
                return None
            custom_map = CustomMapWrapper(mpi_pool)
            results = custom_map(self.evaluate_grid_point, tasks)

    
    def _prepare_main_command(self, nprocs: int, python_script_name: str) -> str: 
        cmp = f"mpirun -np {nprocs} python {python_script_name}"
        cmd += self._prepare_command_line_args()
        return cmd
    
    def _prepare_command_line_args(self) -> str:
        cmd_args = [
            f"--run_mapping",
            f"--param_names {' '.join(self.param_names)}",
            f"--simulation_name {self.simulation_name}",
            f"--polarization {self.polarization}",
            f"--grid_bounds {' '.join([f'{x[0]},{x[1]}' for x in self.grid_bounds])}",
            f"--grid_points {self.grid_points[0]},{self.grid_points[1]}"
        ]
        
    def _merge_lsf_commands(self, lsf_commands: list) -> str:
        return "\n".join(lsf_commands) + "\n"
    
    def _write_lsf_script(self, lsf_script: str, simulation_name: str) -> str:
        job_script_path = os.path.join(simulation_name, f"{simulation_name}.sh")
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
        
        output_filepath = os.path.join(self.simulation.simulation_name, f"{self.simulation.simulation_name}.out")
        error_filepath = os.path.join(self.simulation.simulation_name, f"{self.simulation.simulation_name}.err")
        
        lsf_commands.extend([
            f"#BSUB -oo {output_filepath}",
            f"#BSUB -eo {error_filepath}",
            "module purge",
            f"source {conda_source}",
            f"conda activate {conda_env_name}"
        ])

        lsf_commands.append(self._prepare_main_command(lsf_config.num_processors, python_script_name))
        lsf_script = self._merge_lsf_commands(lsf_commands)
        job_script_path = self._write_lsf_script(lsf_script, self.simulation.simulation_name)
        submission_output = self._submit_job(job_script_path)
        return submission_output

# Main block:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MPI Gamma Mapping for Photonic Bands at Gamma Point")
    parser.add_argument("--run_mapping", action="store_true", help="Run the gamma mapping")
    parser.add_argument("--param_names", type=str, nargs='+', help="Parameter names (e.g., R1 R2)", required=True)
    parser.add_argument("--simulation_name", type=str, help="Simulation name", required=True)
    parser.add_argument("--polarization", type=str, default="te", help="Polarization mode (te or tm)")
    parser.add_argument("--grid_bounds", type=str, nargs='+',
                        help="Grid bounds for each parameter (e.g., 0.1,0.9 0.1,0.9)", required=True)
    parser.add_argument("--grid_points", type=str, help="Resolution for each parameter as Nx,Ny", required=True)
    parser.add_argument("--band_indices", type=str, default=None,
                        help="Comma separated band indices to extract (e.g., 0,1,2,3,4)")
    args = parser.parse_args()

    if not args.param_names:
        raise ValueError("Parameter names must be provided")
    param_names = [name.strip() for item in args.param_names for name in item.split(',')]

    grid_bounds = []
    for b in args.grid_bounds:
        parts = b.split(',')
        if len(parts) != 2:
            raise ValueError(f"Bound {b} is not in the format 'lower,upper'")
        grid_bounds.append((float(parts[0]), float(parts[1])))

    res_parts = args.resolution.split(',')
    if len(res_parts) != 2:
        raise ValueError("Resolution must be provided as Nx,Ny")
    resolution = (int(res_parts[0]), int(res_parts[1]))

    if args.band_indices:
        band_indices = [int(x.strip()) for x in args.band_indices.split(',')]
    else:
        band_indices = None

    scheme_script_path = os.path.join(args.simulation_name, args.simulation_name + ".ctl")
    with open(scheme_script_path, "r") as f:
        scheme_script = f.read()

    gamma_mapper = MPIGammaMapper(simulation_name=args.simulation_name,
                                  scheme_script=scheme_script,
                                  param_names=param_names,
                                  grid_bounds=grid_bounds,
                                  resolution=resolution,
                                  polarization=args.polarization,
                                  band_indices=band_indices)
    if args.run_mapping:
        gamma_mapper.run_mapping()
    else:
        print("Interactive mode.")
    os.chdir("..")
