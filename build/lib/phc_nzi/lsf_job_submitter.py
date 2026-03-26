from abc import ABC, abstractmethod
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



class LSFJob(ABC):

    DEFAULT_MINICONDA_SOURCE = "/zhome/2f/7/202918/miniconda3/etc/profile.d/conda.sh"
    DEFAULT_CONDA_ENV_NAME = "nzi-mp"
    print_lock = threading.Lock()

    def __init__(self, 
                 simulation_name: str,
                 scheme_script: str,     
                 directory: str,
                 ):
        self.simulation = Simulation(simulation_name, 
                                     scheme_script, 
                                     directory)
        self.scheme_script = scheme_script
        self.data_file = os.path.join(self.simulation.directory, self.simulation.simulation_name + ".data")
        
    
    @abstractmethod
    def temp_folder_operations(self, mpb_command_line_params):
        pass

    @abstractmethod
    def _prepare_command_line_args(self) -> str:
        pass
    
    def erease_data_file(self):
        with open(self.data_file, 'w') as f:
            f.write('')

    @abstractmethod
    def _get_script_abs_path(self):
        pass

    def wait_for_job(self, submission_output, poll_interval=10):
        """
        Wait until the submitted LSF job is finished by polling with 'bstat'.
        This version does not use a progress bar but still waits until the job is finished.
        """
        match = re.search(r"Job <(\d+)>", submission_output)
        if match:
            job_id = match.group(1)
            with self.print_lock:
                print(f"Waiting for job {job_id} to finish...")
            start_time = time.time()
            while True:
                try:
                    out = subprocess.check_output(f"bstat {job_id}", shell=True, universal_newlines=True)
                    lines = out.strip().splitlines()
                    if len(lines) < 2:
                        with self.print_lock:
                            print(f"Job {job_id} not found in bstat, assuming finished.")
                        break
                    job_line = lines[1]
                    tokens = job_line.split()
                    status = tokens[5] if len(tokens) > 5 else ""
                    elapsed = int(time.time() - start_time)
                    with self.print_lock:
                        print(f"Job {job_id} status: {status}, elapsed time: {elapsed} sec")
                    if status not in ["RUN", "PEND"]:
                        with self.print_lock:
                            print(f"Job {job_id} finished.")
                        break
                except subprocess.CalledProcessError:
                    with self.print_lock:
                        print(f"bstat command failed for job {job_id}; assuming job is finished.")
                    break
                time.sleep(poll_interval)
        else:
            with self.print_lock:
                print("Could not parse job ID from submission output. Not waiting.")
    
    def submit_lsf_job(self, 
                       lsf_config: LSFJobConfiguration = LSFJobConfiguration(),
                       conda_source: str = DEFAULT_MINICONDA_SOURCE,    
                       conda_env_name: str = DEFAULT_CONDA_ENV_NAME):
        python_script_name = self._get_script_abs_path()
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
    

    def _merge_lsf_commands(self, lsf_commands) -> str:
        return "\n".join(lsf_commands) + "\n"
    
    def _write_lsf_script(self, lsf_script: str, directory: str, simulation_name: str) -> str:
        job_script_path = os.path.join(f"{directory}", f"{simulation_name}.sh")
        with open(job_script_path, 'w') as script_file:
            script_file.write(lsf_script)
        with self.print_lock:
            print("LSF job script written to:", job_script_path)
        return job_script_path
    
    def _submit_job(self, job_script_path: str):
        try:
            submission_output = subprocess.check_output(
                f"bsub < {job_script_path}",
                shell=True,
                universal_newlines=True
            )
            with self.print_lock:
                print("Job submitted successfully. Submission output:")
                print(submission_output)
            # Wait for the job to finish.
            self.wait_for_job(submission_output, poll_interval=10)
        except subprocess.CalledProcessError as e:
            with self.print_lock:
                print("Job submission failed:")
                print(e.output)
            submission_output = None
        return submission_output
