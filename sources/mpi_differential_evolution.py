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

import numpy as np
from scipy.optimize import differential_evolution
from schwimmbad import MPIPool

# Import your pre-implemented Simulation class from simulation_handler.
from simulation_handler import Simulation


# --- Custom Map Wrapper to satisfy SciPy's workers argument ---
class CustomMapWrapper:
    def __init__(self, pool):
        self.pool = pool

    def __call__(self, func, iterable):
        return self.pool.map(func, iterable)

    def __int__(self):
        # Return 0 so that int(worker)==-1 check in differential_evolution passes.
        return 0


class MPIDiffEvoSimulation:
    def __init__(self, 
                 simulation_name: str,
                 scheme_script: str,
                 param_names: list,
                 polarization: str = "te", 
                 maxiter: int = 100, 
                 de_options: dict = None, 
                 param_bounds: list = None):
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
          param_bounds : list, optional
              List of bounds for each parameter, in the form of tuples, e.g. [(lb, ub), ...].
        """
        self.simulation = Simulation(simulation_name=simulation_name,
                                     script=scheme_script)
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.polarization = polarization
        self.maxiter = maxiter
        self.de_options = de_options if de_options is not None else {}
        self.scheme_script = scheme_script

    def objective(self, params):
        """
        Objective function for DE.
        Sets the simulation's command-line parameters using the user-specified parameter names,
        runs the simulation (via run_hpc()), then parses the output for "Frequency 1" and 
        "Frequency 3" (using the selected polarization) and returns their absolute difference.
        
        Parameters:
          params : array_like
              Candidate parameter values.
        
        Returns:
          float : The cost value |Frequency 1 - Frequency 3| or a penalty on error.
        """
        # Build command-line parameters using the user-provided names.
        cmd_params = dict(zip(self.param_names, params))
        old_dir = self.simulation.directory
        # Create a temporary directory for this simulation evaluation.
        self.simulation.directory = os.path.join(old_dir, f"tmp_{time.time_ns()}")
        os.mkdir(self.simulation.directory)
        ctl_file = os.path.join(old_dir, f"{self.simulation.simulation_name}.ctl")
        new_ctl_file = os.path.join(self.simulation.directory, f"{self.simulation.simulation_name}.ctl")
        with open(new_ctl_file, "w") as f:
            f.write(self.scheme_script)
        self.simulation.run_hpc(mpb_command_line_params=cmd_params)
        # Wait briefly for output to be written.
        time.sleep(0.5)
        df = self.simulation.load_frequency_data(self.polarization)
        freqs = self.simulation.get_frequencies_by_band(df, self.polarization)
        cost = abs(freqs[4] - freqs[2])
        # Optionally: restore the directory.
        self.simulation.directory = old_dir
        return cost

    def optimize_parameters(self):
        """
        Run DE optimization with MPI parallelism.
        
        Returns:
          result : OptimizeResult from SciPy's differential_evolution.
        """
        print("Optimizing parameters with MPI parallelism...")
        pool = MPIPool()
        with pool as mpi_pool:
            if not mpi_pool.is_master():
                mpi_pool.wait()
                return None
            bounds = self.param_bounds
            custom_map = CustomMapWrapper(mpi_pool)
            result = differential_evolution(self.objective, bounds,
                                            workers=custom_map,
                                            maxiter=self.maxiter,
                                            **self.de_options)
            print("Optimization complete.")
            print("Optimal parameters found:", result)
            return result
        
    def prepare_lsf_preamble(self, 
                             simulation_name: str,
                             queue: str = "fotonano",
                             num_procs: int = 4,
                             walltime: str = "24:00",
                             mem: str = "4GB",
                             extra_options: list[str] | None = None,
                             user_email: str = None,
                             span_option: str = "hosts",
                             span_value: int = 1) -> list[str]:
        """
        Prepare the preamble lines for an LSF job submission script with span support and
        output/error file directives.
        """
        # Delegate to the Simulation instance's method.
        preamble = self.simulation.prepare_lsf_preamble(
            simulation_name, queue, num_procs, walltime, mem,
            extra_options, user_email, span_option, span_value
        )
        return preamble
        

    def submit_lsf_job(self, nprocs: int = 5, walltime: str = "00:30", queue: str = "normal",
                       user_mail: str = "s232699@dtu.dk", span_option: str = "ptile", span_value: int = 1, 
                       miniconda_source: str = "/zhome/2f/7/202918/miniconda3/etc/profile.d/conda.sh", 
                       conda_env_name: str = "nzi-mp"):
        """
        Submit an LSF job.
        The LSF job script will include the parameter bounds passed via the command-line.
        """
        python_script_name = os.path.basename(__file__)
        preamble = self.prepare_lsf_preamble(self.simulation.simulation_name, queue, nprocs, walltime, user_email=user_mail, span_option=span_option, span_value=span_value)
        lsf_commands = []
        lsf_commands += preamble
        output_path = os.path.join(self.simulation.simulation_name, f"{self.simulation.simulation_name}.out")
        error_path  = os.path.join(self.simulation.simulation_name, f"{self.simulation.simulation_name}.err")
        lsf_commands += [f"#BSUB -oo {output_path}",
                         f"#BSUB -eo {error_path}"]
        lsf_commands.append("module purge")
        lsf_commands.append(f"source {miniconda_source}")
        lsf_commands.append(f"conda activate {conda_env_name}")

        # Convert bounds into strings of the form "lower,upper"
        bound_strings = [f"{lb},{ub}" for (lb, ub) in self.param_bounds]
        cmd = (f"mpirun -np {nprocs} python {python_script_name} "
               f"--run_opt --param_names=\"{','.join(self.param_names)}\" "
               f"--simulation_name=\"{self.simulation.simulation_name}\" "
               f"--maxiter={self.maxiter} --polarization=\"{self.polarization}\" "
               f"--param_bounds " + " ".join(bound_strings))
        lsf_commands.append(cmd)
        lsf_script = "\n".join(lsf_commands) + "\n"
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as tmp_file:
            tmp_file.write(lsf_script)
            job_script_path = tmp_file.name

        print("LSF job script written to:", job_script_path)
        try:
            submission_output = subprocess.check_output(
                f"bsub < {job_script_path}",
                shell=True,
                universal_newlines=True
            )
            print("Job submitted successfully. Submission output:")
            print(submission_output)
        except subprocess.CalledProcessError as e:
            print("Job submission failed:")
            print(e.output)
            submission_output = None

        os.remove(job_script_path)
        return submission_output


# Main block:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MPIOptimization")
    parser.add_argument("--run_opt", action="store_true", help="Run the optimization")
    parser.add_argument("--param_names", type=str, nargs='+', help="Parameter names (e.g., R1 R2)")
    parser.add_argument("--simulation_name", type=str, help="Simulation name", required=True)
    parser.add_argument("--maxiter", type=int, default=100, help="Maximum number of DE iterations")
    parser.add_argument("--polarization", type=str, default="te", help="Polarization mode (e.g., te or tm)")
    parser.add_argument("--param_bounds", type=str, nargs='+', help="Parameter bounds (e.g., 0.1,0.9 0.1,0.9)", required=True)
    args = parser.parse_args()
    
    # Construct the path to the scheme script (assumed to be in a folder named after the simulation)
    scheme_script_path = os.path.join(args.simulation_name, args.simulation_name + ".ctl")
    with open(scheme_script_path, "r") as f:
        scheme_script = f.read()

    if args.run_opt:
        if not args.param_names:
            raise ValueError("Parameter names must be provided")
        param_names = args.param_names  # Already a list
        simulation_name = args.simulation_name
        maxiter = args.maxiter
        polarization = args.polarization

        # Process param_bounds: each bound is provided as "lower,upper"
        param_bounds = []
        for b in args.param_bounds:
            parts = b.split(',')
            if len(parts) != 2:
                raise ValueError(f"Bound {b} is not in the format 'lower,upper'")
            lb, ub = parts
            param_bounds.append((float(lb), float(ub)))
            
        optimizer = MPIDiffEvoSimulation(simulation_name=simulation_name,
                                          scheme_script=scheme_script,
                                          param_names=param_names,
                                          maxiter=maxiter,
                                          polarization=polarization,
                                          param_bounds=param_bounds)
        result = optimizer.optimize_parameters()
        if result is not None:
            print("Optimal parameters found:", result.x)
            print("Minimum frequency difference:", result.fun)
    else:
        print("Interactive mode.")
    os.chdir("..")
