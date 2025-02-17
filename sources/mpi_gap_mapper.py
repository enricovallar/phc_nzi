#!/usr/bin/env python
"""
MPI Gap Mapping for Photonic Crystal using LSF (DTU Cluster)

This script uses MPI to evaluate a grid of parameter values in parallel.
It submits an LSF job to run on multiple nodes and cores, waits until the
job is finished (using bstat), and then plots a heatmap of the computed gap
(i.e. the absolute difference between two specified frequency bands).

Usage:
    python MPIGapMapper.py --run_mapping --param_names "R1" "R2" --simulation_name MySimulation \
       --polarization te --grid_bounds 0.1,0.9 0.1,0.9 --resolution 50,50 --band_indices 2,4
"""

import os
import re
import subprocess
import time
import sys
import tempfile
import argparse

import numpy as np
import matplotlib.pyplot as plt
from schwimmbad import MPIPool
from tqdm import tqdm

# Import your pre-implemented Simulation class from simulation_handler.
from simulation_handler import Simulation


# --- Custom Map Wrapper to satisfy SciPy's workers argument ---
class CustomMapWrapper:
    def __init__(self, pool):
        self.pool = pool

    def __call__(self, func, iterable):
        return self.pool.map(func, iterable)

    def __int__(self):
        return 0


class MPIGapMapper:
    def __init__(self, 
                 simulation_name: str,
                 scheme_script: str,
                 param_names: list,
                 grid_bounds: list,
                 resolution: tuple,
                 polarization: str = "te",
                 band_indices: tuple = (2, 4)):
        """
        Parameters
        ----------
        simulation_name : str
            Name of the simulation.
        scheme_script : str
            The scheme (.ctl) script contents.
        param_names : list
            Names of the two parameters to be varied (e.g., ["R1", "R2"]).
        grid_bounds : list
            List of tuples specifying the boundaries for each parameter,
            e.g. [(lb1, ub1), (lb2, ub2)].
        resolution : tuple
            A tuple (Nx, Ny) giving the number of grid points for each parameter.
        polarization : str, optional
            Polarization mode used when extracting frequency data. Default is "te".
        band_indices : tuple, optional
            Tuple specifying which bands to compare.
            For example, (2, 4) computes gap = |freqs[4] - freqs[2]|.
        """
        self.simulation = Simulation(simulation_name=simulation_name,
                                     script=scheme_script)
        self.simulation_name = simulation_name
        self.scheme_script = scheme_script
        self.param_names = param_names
        self.grid_bounds = grid_bounds
        self.resolution = resolution
        self.polarization = polarization
        self.band_indices = band_indices
        self.mapping_log = os.path.join(simulation_name, f"{simulation_name}_mapping.log")
        self.orig_dir = self.simulation.directory

    def evaluate_grid_point(self, task):
        """
        Evaluate one grid point.
        'task' is a tuple: (i, j, param0, param1).
        This method sets the parameters, runs the simulation, computes the gap,
        logs the evaluation, and returns (i, j, gap).
        """
        i, j, param0, param1 = task
        params = [param0, param1]
        cmd_params = dict(zip(self.param_names, params))
        
        temp_dir = os.path.join(self.orig_dir, f"tmp_{time.time_ns()}")
        os.mkdir(temp_dir)
        self.simulation.directory = temp_dir
        ctl_file = os.path.join(temp_dir, f"{self.simulation.simulation_name}.ctl")
        with open(ctl_file, "w") as f:
            f.write(self.scheme_script)
        
        self.simulation.run_hpc(mpb_command_line_params=cmd_params)
        time.sleep(0.5)
        
        df = self.simulation.load_frequency_data(self.polarization)
        freqs = self.simulation.get_frequencies_by_band(df, self.polarization)
        gap = abs(freqs[self.band_indices[1]] - freqs[self.band_indices[0]])
        
        with open(self.mapping_log, "a") as f:
            line = f"{self.param_names[0]}: {param0}, {self.param_names[1]}: {param1}, gap: {gap}\n"
            f.write(line)
        
        self.simulation.directory = self.orig_dir
        
        return (i, j, gap)

    def map_gap(self):
        """
        Run the simulation on a grid of parameter values in parallel using MPI
        (if more than one process is available) and compute the gap (absolute
        difference between the two specified bands) at each grid point.
        
        Returns
        -------
        X, Y : 2D arrays (meshgrid of parameter values)
        Z : 2D array (gap computed as |freq[band2] - freqs[band1]|)
        """
        p0_vals = np.linspace(self.grid_bounds[0][0], self.grid_bounds[0][1], self.resolution[0])
        p1_vals = np.linspace(self.grid_bounds[1][0], self.grid_bounds[1][1], self.resolution[1])
        X, Y = np.meshgrid(p0_vals, p1_vals)
        Z = np.zeros_like(X)
        
        tasks = []
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                param0 = X[j, i]
                param1 = Y[j, i]
                tasks.append((i, j, param0, param1))
        
        results = []
        with MPIPool() as pool:
            if pool.size == 1:
                print("Only one MPI process available; running mapping serially.")
                results = list(tqdm(map(self.evaluate_grid_point, tasks),
                                    total=len(tasks), desc="Mapping Grid Points"))
            else:
                if not hasattr(pool, "imap"):
                    if hasattr(pool, "imap_unordered"):
                        imap_func = pool.imap_unordered
                    else:
                        imap_func = pool.map
                else:
                    imap_func = pool.imap
                if not pool.is_master():
                    pool.wait()
                    sys.exit(0)
                for r in tqdm(imap_func(self.evaluate_grid_point, tasks),
                              total=len(tasks), desc="Mapping Grid Points"):
                    results.append(r)
        
        for (i, j, gap) in results:
            Z[j, i] = gap
        
        return X, Y, Z

    def plot_mapping(self, X, Y, Z, custom_title: str = None):
        """
        Plot a heatmap of the computed gap.
        """
        plt.figure(figsize=(8, 6))
        cp = plt.contourf(X, Y, Z, levels=50, cmap="viridis")
        plt.colorbar(cp, label="Gap")
        plt.xlabel(self.param_names[0])
        plt.ylabel(self.param_names[1])
        if custom_title is None:
            plt.title(f"Gap |band {self.band_indices[1]} - band {self.band_indices[0]}|")
        else:
            plt.title(custom_title)
        plt.tight_layout()
        plt.show()

    def plot_mapping_from_log(self, log_file_path=None, use_logscale=False, levels=50, 
                              points_only=False, custom_title=None, plot_inverse_gap=False):
        """
        Reads lines from the mapping log file, extracting parameter names and values,
        and plots a heatmap (or scatter plot) of gap versus the two parameters.
        
        Each valid log line is expected to contain key-value pairs in the form:
            <param1>: <value>, <param2>: <value>, gap: <value>
        The method extracts the parameter names dynamically from the first valid line.
        """
        from matplotlib.colors import LogNorm
        
        if log_file_path is None:
            log_file_path = self.mapping_log

        param1_vals = []
        param2_vals = []
        gap_vals = []
        x_label, y_label = None, None
        
        if not os.path.exists(log_file_path):
            print(f"Log file {log_file_path} not found.")
            return
        
        with open(log_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pairs = re.findall(r"(\w[\w\d_]*)\s*:\s*([\d\.\-eE]+)", line)
                if not pairs:
                    continue
                data = {}
                for key, val in pairs:
                    try:
                        data[key] = float(val)
                    except ValueError:
                        continue
                if "gap" not in data or len(data) - 1 != 2:
                    continue
                gap_val = data.pop("gap")
                keys = list(data.keys())
                if x_label is None and y_label is None:
                    x_label, y_label = keys[0], keys[1]
                if set(keys) != {x_label, y_label}:
                    continue
                param1_vals.append(data[x_label])
                param2_vals.append(data[y_label])
                gap_vals.append(gap_val)
        
        if not param1_vals:
            print("No valid data found in the log file.")
            return

        if plot_inverse_gap:
            gap_vals = [1.0/g if g != 0 else 0 for g in gap_vals]
            gap_label = "1/Gap"
        else:
            gap_label = "Gap"
        
        norm = LogNorm(vmin=min(gap_vals), vmax=max(gap_vals)) if use_logscale else None
        
        if points_only:
            plt.figure(figsize=(8,6))
            sc = plt.scatter(param1_vals, param2_vals, c=gap_vals, cmap="viridis", norm=norm)
            plt.colorbar(sc, label=gap_label)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            title = custom_title if custom_title else "Mapping Data (Scatter)"
            plt.title(title)
            plt.tight_layout()
            plt.show()
        else:
            try:
                import matplotlib.tri as mtri
                triang = mtri.Triangulation(param1_vals, param2_vals)
                plt.figure(figsize=(8,6))
                if levels is None:
                    pc = plt.tripcolor(triang, gap_vals, shading="gouraud", cmap="viridis", norm=norm)
                    plt.colorbar(pc, label=gap_label)
                    plot_desc = "Tripcolor (Gouraud Shading)"
                else:
                    cntr = plt.tricontourf(triang, gap_vals, levels=levels, cmap="viridis", norm=norm)
                    plt.colorbar(cntr, label=gap_label)
                    plot_desc = f"Tricontourf (levels={levels})"
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                title = custom_title if custom_title else f"Mapping Data ({plot_desc})"
                plt.title(title)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print("Contour plot failed; falling back to scatter plot:", e)
                plt.figure(figsize=(8,6))
                sc = plt.scatter(param1_vals, param2_vals, c=gap_vals, cmap="viridis", norm=norm)
                plt.colorbar(sc, label=gap_label)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                title = custom_title if custom_title else "Mapping Data (Scatter)"
                plt.title(title)
                plt.tight_layout()
                plt.show()

    def wait_for_job(self, submission_output, poll_interval=10):
        """
        Wait until the submitted LSF job is finished by polling with 'bstat'.
        Also read and print the contents of the error and output files.
        """
        match = re.search(r"Job <(\d+)>", submission_output)
        if match:
            job_id = match.group(1)
            print(f"Waiting for job {job_id} to finish...")
            last_status = None
            while True:
                try:
                    out = subprocess.check_output(f"bstat {job_id}", shell=True, universal_newlines=True)
                    lines = out.strip().splitlines()
                    if len(lines) < 2:
                        print("Job not found in bstat, assuming finished.")
                        break
                    job_line = lines[1]
                    tokens = job_line.split()
                    status = tokens[5] if len(tokens) > 5 else ""
                    if status != last_status:
                        print(f"Job {job_id} status: {status}")
                        last_status = status
                    if status not in ["RUN", "PEND"]:
                        print("Job finished.")
                        break
                except subprocess.CalledProcessError:
                    print("bstat command failed; assuming job is finished.")
                    break
                time.sleep(poll_interval)
            base_dir = self.simulation.simulation_name
            out_file = os.path.join(base_dir, f"{self.simulation.simulation_name}.out")
            err_file = os.path.join(base_dir, f"{self.simulation.simulation_name}.err")
            if os.path.exists(out_file):
                with open(out_file, "r") as f:
                    print("==== Output File ====")
                    print(f.read())
            if os.path.exists(err_file):
                with open(err_file, "r") as f:
                    print("==== Error File ====")
                    print(f.read())
        else:
            print("Could not parse job ID from submission output. Not waiting.")

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
        Prepare the preamble for an LSF job submission script.
        Delegates to the Simulation instance's method.
        """
        preamble = self.simulation.prepare_lsf_preamble(
            simulation_name, queue, num_procs, walltime, mem,
            extra_options, user_email, span_option, span_value
        )
        return preamble

    def submit_lsf_job(self, nprocs: int = 5, walltime: str = "00:30", queue: str = "normal",
                       user_mail: str = "s232699@dtu.dk", span_option: str = "ptile", span_value: int = 1,
                       miniconda_source: str = "/zhome/2f/7/202918/miniconda3/etc/profile.d/conda.sh",
                       conda_env_name: str = "nzi-mp", store_sh_file: bool = False):
        """
        Submit an LSF job for the gap mapping.
        The LSF job script includes the grid boundaries, resolution, and band indices.
        Waits until the job is finished before returning.
        """
        python_script_name = os.path.abspath(__file__)
        preamble = self.prepare_lsf_preamble(self.simulation.simulation_name, queue, nprocs, walltime, mem="4GB",
                                               extra_options=None, user_email=user_mail,
                                               span_option=span_option, span_value=span_value)
        lsf_commands = []
        lsf_commands += preamble
        base_dir = self.simulation.simulation_name
        output_path = os.path.join(base_dir, f"{self.simulation.simulation_name}.out")
        error_path  = os.path.join(base_dir, f"{self.simulation.simulation_name}.err")
        lsf_commands += [f"#BSUB -oo {output_path}",
                         f"#BSUB -eo {error_path}"]
        lsf_commands.append("module purge")
        lsf_commands.append(f"source {miniconda_source}")
        lsf_commands.append(f"conda activate {conda_env_name}")
        grid_bound_strings = [f"{lb},{ub}" for (lb, ub) in self.grid_bounds]
        res_string = f"{self.resolution[0]},{self.resolution[1]}"
        cmd = (f"mpirun -np {nprocs} python {python_script_name} "
               f"--run_mapping --param_names=\"{','.join(self.param_names)}\" "
               f"--simulation_name=\"{self.simulation.simulation_name}\" "
               f"--polarization=\"{self.polarization}\" "
               f"--grid_bounds " + " ".join(grid_bound_strings) + " " +
               f"--resolution {res_string} "
               f"--band_indices {self.band_indices[0]},{self.band_indices[1]}")
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
            self.wait_for_job(submission_output)
        except subprocess.CalledProcessError as e:
            print("Job submission failed:")
            print(e.output)
            submission_output = None

        if not store_sh_file:
            os.remove(job_script_path)
        else:
            print("Keeping the shell script file at:", job_script_path)
        return submission_output

    def run_mapping(self):
        """
        Convenience method to compute the gap mapping and plot the heatmap.
        In an interactive run, this method computes the mapping immediately.
        When using LSF submission, call submit_lsf_job() and then run mapping after job completion.
        """
        X, Y, Z = self.map_gap()
        self.plot_mapping(X, Y, Z)


# Main block:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MPI Gap Mapping")
    parser.add_argument("--run_mapping", action="store_true", help="Run the gap mapping")
    parser.add_argument("--param_names", type=str, nargs='+', help="Parameter names (e.g., R1 R2)")
    parser.add_argument("--simulation_name", type=str, help="Simulation name", required=True)
    parser.add_argument("--polarization", type=str, default="te", help="Polarization mode (e.g., te or tm)")
    parser.add_argument("--grid_bounds", type=str, nargs='+',
                        help="Grid bounds for each parameter (e.g., 0.1,0.9 0.1,0.9)", required=True)
    parser.add_argument("--resolution", type=str, help="Resolution for each parameter as Nx,Ny", required=True)
    parser.add_argument("--band_indices", type=str, default="2,4", help="Band indices as i,j (default: 2,4)")
    args = parser.parse_args()

    if not args.param_names:
        raise ValueError("Parameter names must be provided")
    param_names = [name.strip() for item in args.param_names for name in item.split(',')]

    grid_bounds = []
    for b in args.grid_bounds:
        parts = b.split(',')
        if len(parts) != 2:
            raise ValueError(f"Bound {b} is not in the format 'lower,upper'")
        lb, ub = parts
        grid_bounds.append((float(lb), float(ub)))

    res_parts = args.resolution.split(',')
    if len(res_parts) != 2:
        raise ValueError("Resolution must be provided as Nx,Ny")
    resolution = (int(res_parts[0]), int(res_parts[1]))

    band_parts = args.band_indices.split(',')
    if len(band_parts) != 2:
        raise ValueError("Band indices must be provided as i,j")
    band_indices = (int(band_parts[0]), int(band_parts[1]))

    scheme_script_path = os.path.join(args.simulation_name, args.simulation_name + ".ctl")
    with open(scheme_script_path, "r") as f:
        scheme_script = f.read()

    gap_mapper = MPIGapMapper(simulation_name=args.simulation_name,
                              scheme_script=scheme_script,
                              param_names=param_names,
                              grid_bounds=grid_bounds,
                              resolution=resolution,
                              polarization=args.polarization,
                              band_indices=band_indices)
    if args.run_mapping:
        gap_mapper.run_mapping()
    else:
        print("Interactive mode.")
    os.chdir("..")
