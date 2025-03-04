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
                 param_bounds: list = None, 
                 bands: list = [2,3,4]):
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
                                     script=scheme_script)
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.polarization = polarization
        self.maxiter = maxiter
        self.de_options = de_options if de_options is not None else {}
        self.scheme_script = scheme_script
        self.log_file = os.path.join(simulation_name, f"{simulation_name}.log")
        self.bands = bands

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
        freqs = self.simulation.get_frequencies_by_band(df, self.polarization, bands=self.bands)
        idx_high = np.max(self.bands)
        idx_low = np.min(self.bands)
        idx_central = idx_low + 1

        cost = abs(freqs[idx_high] - freqs[idx_low])
        # Optionally: restore the directory.
        self.simulation.directory = old_dir

        with open(self.log_file, "a") as f:
            line = f"{self.param_names[0]}: {params[0]}, {self.param_names[1]}: {params[1]}"
            line += f", cost: {cost}"
            line += f", freq_dirac: {freqs[idx_central]}\n"
            f.write(line)
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

    def wait_for_job(self, submission_output, poll_interval=10):
        """
        Wait until the submitted LSF job is finished by polling with 'bstat'.
        An unlimited (indeterminate) tqdm progress bar is used to indicate elapsed time.
        """
        from tqdm import tqdm

        match = re.search(r"Job <(\d+)>", submission_output)
        if match:
            job_id = match.group(1)
            print(f"Waiting for job {job_id} to finish...")
            start_time = time.time()
            with tqdm(desc="Waiting for LSF job", unit="sec", dynamic_ncols=True) as pbar:
                while True:
                    try:
                        out = subprocess.check_output(f"bstat {job_id}", shell=True, universal_newlines=True)
                        lines = out.strip().splitlines()
                        if len(lines) < 2:
                            pbar.write("Job not found in bstat, assuming finished.")
                            break
                        job_line = lines[1]
                        tokens = job_line.split()
                        status = tokens[5] if len(tokens) > 5 else ""
                        elapsed = int(time.time() - start_time)
                        pbar.set_postfix_str(f"Job status: {status}, Elapsed: {elapsed} sec")
                        if status not in ["RUN", "PEND"]:
                            pbar.write("Job finished.")
                            break
                    except subprocess.CalledProcessError:
                        pbar.write("bstat command failed; assuming job is finished.")
                        break
                    time.sleep(poll_interval)
                    pbar.update(poll_interval)
        else:
            print("Could not parse job ID from submission output. Not waiting.")

    def submit_lsf_job(self, nprocs: int = 5, walltime: str = "00:30", queue: str = "normal",
                       user_mail: str = "s232699@dtu.dk", span_option: str = "ptile", span_value: int = 1, 
                       miniconda_source: str = "/zhome/2f/7/202918/miniconda3/etc/profile.d/conda.sh", 
                       conda_env_name: str = "nzi-mp"):
        """
        Submit an LSF job.
        The LSF job script will include the parameter bounds passed via the command-line.
        This method now waits for the job to finish, displaying an unlimited progress bar.
        """
        python_script_name = os.path.abspath(__file__)
        preamble = self.prepare_lsf_preamble(self.simulation.simulation_name, queue, nprocs, walltime,
                                               mem="4GB", extra_options=None,
                                               user_email=user_mail, span_option=span_option, span_value=span_value)
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
               f"--param_bounds " + " ".join(bound_strings) + " " +
               f"--popsize={self.de_options.get('popsize', 15)} "
               f"--strategy=\"{self.de_options.get('strategy', 'rand1bin')}\" "
               f"--bands {' '.join(map(str, self.bands))}")
             
        lsf_commands.append(cmd)
        lsf_script = "\n".join(lsf_commands) + "\n"
        
        # Create a job script in the simulation directory
        job_script_path = os.path.join(self.simulation.simulation_name, f"{self.simulation.simulation_name}_lsf_job.sh")
        with open(job_script_path, 'w') as script_file:
            script_file.write(lsf_script)

        print("LSF job script written to:", job_script_path)
        try:
            submission_output = subprocess.check_output(
                f"bsub < {job_script_path}",
                shell=True,
                universal_newlines=True
            )
            print("Job submitted successfully. Submission output:")
            print(submission_output)
            # Wait for the job to finish using an unlimited progress bar.
            self.wait_for_job(submission_output, poll_interval=10)
        except subprocess.CalledProcessError as e:
            print("Job submission failed:")
            print(e.output)
            submission_output = None
        return submission_output

    def plot_optimization_points(self, 
                        log_file_path=None, 
                        use_logscale=False, 
                        levels=50, 
                        points_only=False,
                        plot_inverse_cost=False,
                        custom_title=None):
        """
        Reads lines from the .log file, extracting arbitrary parameter names and 
        their values, along with 'cost'. Produces a 2D heat map (or scatter plot) 
        of 'cost' (or 1/cost) vs. two selected parameters.
        """
        import re
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        if log_file_path is None:
            log_file_path = self.log_file

        # We'll store the param data in separate arrays for plotting:
        x_vals = []
        y_vals = []
        cost_vals = []

        # Track the parameter names for labeling
        param_x_name = None
        param_y_name = None

        if not os.path.isfile(log_file_path):
            print(f"Log file not found: {log_file_path}")
            return

        with open(log_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # skip empty lines

                # Regex to find all "paramName: value" pairs
                pattern = r"(\w+)\s*:\s*([\d.+\-eE]+)"
                matches = re.findall(pattern, line)
                if not matches:
                    continue

                # Convert captures to a dictionary: { paramName: floatValue }
                param_dict = {}
                for (pname, pval_str) in matches:
                    try:
                        val = float(pval_str)
                    except ValueError:
                        continue
                    param_dict[pname] = val

                # Remove the cost
                cost = param_dict.pop('cost', None)
                if cost is None:
                    continue

                # Remove freq_dirac if present
                param_dict.pop('freq_dirac', None)

                # If we want 1/cost, handle that now
                if plot_inverse_cost:
                    if cost == 0:
                        continue
                    cost = 1.0 / cost

                # We need exactly 2 parameters (after removing cost and freq_dirac)
                if len(param_dict) != 2:
                    continue

                # Sort param names so that we always pick them in a stable order.
                sorted_params = sorted(param_dict.keys())
                p1, p2 = sorted_params[0], sorted_params[1]

                # Lock in param names once we see the first valid line.
                if param_x_name is None and param_y_name is None:
                    param_x_name, param_y_name = p1, p2
                else:
                    if {p1, p2} != {param_x_name, param_y_name}:
                        continue

                x_val = param_dict[param_x_name]
                y_val = param_dict[param_y_name]

                x_vals.append(x_val)
                y_vals.append(y_val)
                cost_vals.append(cost)

        if not x_vals:
            print("No valid lines with exactly two parameters + cost found.")
            return

        if use_logscale:
            min_cost = min(cost_vals)
            if min_cost <= 0:
                print("Cannot use log scale because min cost <= 0. Switching to linear scale.")
                use_logscale = False

        norm = LogNorm(vmin=min(cost_vals), vmax=max(cost_vals)) if use_logscale else None

        scale_title = " (Log Scale)" if use_logscale else " (Linear Scale)"
        extra_title = "1/Cost" if plot_inverse_cost else "Cost"

        def make_title(prefix):
            if custom_title is not None:
                return custom_title
            else:
                return f"{prefix}{scale_title} ({extra_title})"

        if points_only:
            plt.figure(figsize=(7, 6))
            scatter = plt.scatter(x_vals, y_vals, c=cost_vals, cmap="viridis", norm=norm)
            plt.colorbar(scatter, label=extra_title)
            plt.xlabel(param_x_name)
            plt.ylabel(param_y_name)
            plt.title(make_title("Parameter Space Scatter"))
            plt.tight_layout()
            plt.show()
            return

        try:
            import matplotlib.tri as mtri
            triang = mtri.Triangulation(x_vals, y_vals)

            plt.figure(figsize=(7, 6))

            if levels is None:
                pc = plt.tripcolor(
                    triang,
                    cost_vals,
                    shading="gouraud",
                    cmap="viridis",
                    norm=norm
                )
                plt.colorbar(pc, label=extra_title)
                plot_desc = "Tripcolor (Gouraud Shading)"
            else:
                cntr = plt.tricontourf(
                    triang,
                    cost_vals,
                    levels=levels,
                    cmap="viridis",
                    norm=norm
                )
                plt.colorbar(cntr, label=extra_title)
                plot_desc = f"Tricontourf (levels={levels})"

            plt.xlabel(param_x_name)
            plt.ylabel(param_y_name)
            plt.title(make_title(f"Parameter Space Heatmap\n{plot_desc}"))
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print("Falling back to scatter plot due to:", e)
            plt.figure(figsize=(7, 6))
            scatter = plt.scatter(x_vals, y_vals, c=cost_vals, cmap="viridis", norm=norm)
            plt.colorbar(scatter, label=extra_title)
            plt.xlabel(param_x_name)
            plt.ylabel(param_y_name)
            plt.title(make_title("Parameter Space Scatter"))
            plt.tight_layout()
            plt.show()


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
    args = parser.parse_args()
    
    scheme_script_path = os.path.join(args.simulation_name, args.simulation_name + ".ctl")
    with open(scheme_script_path, "r") as f:
        scheme_script = f.read()

    # test the scheme script saving it to a test file
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
        optimizer = MPIDiffEvoSimulation(simulation_name=simulation_name,
                                          scheme_script=scheme_script,
                                          param_names=param_names,
                                          maxiter=maxiter,
                                          polarization=polarization,
                                          param_bounds=param_bounds, 
                                          de_options=de_options,
                                          bands=bands)

        result = optimizer.optimize_parameters()
        if result is not None:
            print("Optimal parameters found:", result.x)
            print("Minimum frequency difference:", result.fun)
    else:
        print("Interactive mode.")
    os.chdir("..")
