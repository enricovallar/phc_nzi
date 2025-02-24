import os
import subprocess
import time
import re
import sqlite3
import tempfile
import logging

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import meep as mp
from meep import mpb
from mpb_configurator import MPBSchemeConfigurator

import plotly.graph_objects as go

# Configure module-level logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default level; change as needed
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class Simulation:
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    def __init__(self, simulation_name: str, script: str, 
                 directory: str = None, description: str = None, 
                 log_level: int = logging.INFO, save_script=True):
        self.simulation_name = simulation_name
        self.directory = directory or simulation_name
        os.makedirs(self.directory, exist_ok=True)
        if save_script is True:
            with open(os.path.join(self.directory, f"{simulation_name}.ctl"), "w") as f:
                f.write(script)
                
        self.scheme_filename = f"{simulation_name}.ctl"
        self.script = script
        self.output_filename = f"{simulation_name}.out"
        self.error_filename = f"{simulation_name}.err"
        self.epsilon = None
        self.lattice = None
        self.bands_df = {}  # For frequency database storage

        # Configure an instance-specific logger.
        self.logger = logging.getLogger(f"{__name__}.{self.simulation_name}")
        self.logger.setLevel(log_level)

        if description:
            desc_path = os.path.join(self.directory, f"{simulation_name}.txt")
            with open(desc_path, "w") as f:
                f.write(description)

    def save_script(self, scheme_script: str, print_script: bool = False) -> str:
        if scheme_script is None:
            raise ValueError("Scheme script must be provided.")
        elif not isinstance(scheme_script, str):
            raise ValueError("Scheme script must be a string.")
        else:
            with open(os.path.join(self.directory, self.scheme_filename), "w") as f:
                f.write(scheme_script)
            if print_script:
                print(scheme_script)

    def _execute_command(self, cmd, shell: bool = False,
                         print_output: bool = False, print_error: bool = False):
        self.logger.debug("Executing command: %s", cmd)
        result = subprocess.run(cmd, shell=shell, capture_output=True,
                                text=True, cwd=self.directory)
        out_path = os.path.join(self.directory, self.output_filename)
        err_path = os.path.join(self.directory, self.error_filename)
        with open(out_path, "w") as f_out:
            f_out.write(result.stdout)
        with open(err_path, "w") as f_err:
            f_err.write(result.stderr)
        if print_output:
            print(result.stdout)
        if print_error:
            print(result.stderr)
        return result

    def run_hpc(self, mpb_command_line_params: dict = {},
                load_epsilon: bool = True, extract_frequencies: bool = True,
                print_output: bool = False, print_error: bool = False): 
        filename = self.scheme_filename
        with open(filename, "w") as f:
            f.write(self.script)
        self.logger.debug("Current directory: %s", os.getcwd())
        params = " ".join(f"{k}={v}" for k, v in mpb_command_line_params.items())
        cmd = (f"source /dtu/sw/dcc/dcc-sw.bash && module load mpb/1.11.1 && "
               f"mpb-mpi {params} {filename}")
        self.logger.debug("Running HPC command: %s", cmd)
        self._execute_command(cmd, shell=True,
                              print_output=print_output, print_error=print_error)
        self.logger.debug("Simulation completed")
        if load_epsilon:
            self.load_epsilon()
        if extract_frequencies:
            self.extract_frequencies()

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
        Prepare the preamble lines for an LSF job submission script.
        """
        preamble = []
        preamble.append("#!/bin/bash")
        preamble.append(f"#BSUB -J {simulation_name}")
        preamble.append(f"#BSUB -q {queue}")
        preamble.append(f"#BSUB -n {num_procs}")
        preamble.append(f"#BSUB -W {walltime}")
        preamble.append(f"#BSUB -R \"rusage[mem={mem}]\"")
        
        span_str = {
            "hosts": f'span[hosts={span_value}]',
            "ptile": f'span[ptile={span_value}]',
            "block": f'span[block={span_value}]'
        }.get(span_option)
        if span_str:
            preamble.append(f"#BSUB -R \"{span_str}\"")
        
        preamble.append(f"#BSUB -oo {simulation_name}.out")
        preamble.append(f"#BSUB -eo {simulation_name}.err")
        
        if user_email:
            preamble.append(f"#BSUB -u {user_email}")
        if extra_options:
            for option in extra_options:
                preamble.append(option)
        return preamble

    def run_hpc_lsf(self, 
                    load_epsilon: bool = True, extract_frequencies: bool = True,
                    mpb_command_line_params: dict = {},
                    print_output: bool = False, print_error: bool = False,
                    queue: str = "fotonano", num_procs: int = 4,
                    initial_wait: int = 2, poll_interval: int = 5, output_timeout: int = 300,
                    span_option: str = "hosts", span_value: int = 1):
        filename = self.scheme_filename
        with open(filename, "w") as f:
            f.write(self.script)
        
        params = " ".join(f"{k}={v}" for k, v in mpb_command_line_params.items())
        preamble_lines = self.prepare_lsf_preamble(simulation_name=self.simulation_name, queue=queue,
                                                     num_procs=num_procs, span_option=span_option,
                                                     span_value=span_value)
        job_script = "\n".join(preamble_lines) + "\n\n"
        job_script += (
            f"source /dtu/sw/dcc/dcc-sw.bash && module load mpb/1.11.1 && "
            f"mpirun -np $LSB_DJOB_NUMPROC mpb-mpi {params} {filename}"
        )
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=self.directory, suffix=".sh") as job_tmp:
            job_tmp.write(job_script)
            job_script_file = job_tmp.name
        cmd = (f"bsub -oo {self.output_filename} -eo {self.error_filename} "
               f"< {job_script_file}")
        self.logger.debug("Running LSF job command: %s", cmd)
        result = self._execute_command(cmd, shell=True,
                                       print_output=print_output, print_error=print_error)
        job_id_match = re.search(r"<(\d+)>", result.stdout)
        if job_id_match:
            job_id = job_id_match.group(1)
            self.logger.info("Job submitted with ID: %s", job_id)
            self.logger.debug("Waiting %s seconds before first status check...", initial_wait)
            time.sleep(initial_wait)
            while True:
                status = subprocess.run("bstat", shell=True, capture_output=True, text=True)
                if "No unfinished job found" in status.stdout or job_id not in status.stdout:
                    self.logger.debug("Job has finished.")
                    break
                else:
                    self.logger.debug("Job %s is still running. Waiting %s seconds...", job_id, poll_interval)
                    time.sleep(poll_interval)
        else:
            self.logger.warning("Could not determine job ID; proceeding without waiting.")
        out_path = os.path.join(self.directory, self.output_filename)
        err_path = os.path.join(self.directory, self.error_filename)
        elapsed = 0
        while (((not os.path.exists(out_path) or os.path.getsize(out_path) == 0) or 
                (not os.path.exists(err_path) or os.path.getsize(err_path) == 0))
               and elapsed < output_timeout):
            self.logger.info("Waiting for simulation output files to be written...")
            time.sleep(5)
            elapsed += 5
        if elapsed >= output_timeout:
            self.logger.warning("Output files not found or empty after waiting.")
        else:
            self.logger.debug("Output files are now available.")
        self.logger.debug("Simulation completed")
        if load_epsilon:
            self.load_epsilon()
        if extract_frequencies:
            self.extract_frequencies()

    def extract_frequencies(self, remove_line_prefixes: bool = True):
        prefixes = ["tmfreqs:", "tefreqs:", "zevenfreqs:", "zoddfreqs:", "gaps:"]
        def strip_prefix(line: str) -> str:
            for prefix in prefixes:
                if line.startswith(prefix):
                    return line[len(prefix):].lstrip(" ,")
            return line
        output_path = os.path.join(self.directory, self.output_filename)
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output file {output_path} does not exist.")
        with open(output_path, "r") as f:
            lines = f.readlines()
        modes = {
            "tm": [strip_prefix(line) if remove_line_prefixes else line
                   for line in lines if "tmfreqs:" in line],
            "te": [strip_prefix(line) if remove_line_prefixes else line
                   for line in lines if "tefreqs:" in line],
            "zeven": [strip_prefix(line) if remove_line_prefixes else line
                      for line in lines if "zevenfreqs:" in line],
            "zodd": [strip_prefix(line) if remove_line_prefixes else line
                     for line in lines if "zoddfreqs:" in line],
            "gaps": [strip_prefix(line) if remove_line_prefixes else line
                     for line in lines if "gaps:" in line]
        }
        for mode, data in modes.items():
            file_path = os.path.join(self.directory, f"{self.simulation_name}.{mode}.dat")
            with open(file_path, "w") as f_mode:
                f_mode.writelines(data)
            self.logger.debug("Extracted %d lines of data for mode '%s'", len(data), mode)

    def load_frequency_data(self, mode: str = "te") -> pd.DataFrame:
        """
        Load frequency data for the given mode from a .dat file.
        
        The file is expected to be named:
        {simulation_name}.{mode}.dat
        The data is also stored in an SQLite database for future use.
        
        Parameters:
            mode (str): The polarization mode ('te', 'tm', etc.).
        
        Returns:
            pd.DataFrame: The loaded frequency data.
        
        Raises:
            FileNotFoundError: If the .dat file is not found.
        """
        filepath = os.path.join(self.directory, f"{self.simulation_name}.{mode}.dat")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found.")
        df = pd.read_csv(filepath, skipinitialspace=True)
        db_path = os.path.join(self.directory, f"{self.simulation_name}_frequencies.db")
        with sqlite3.connect(db_path) as conn:
            df.to_sql("frequencies", conn, if_exists="replace", index=False)
        self.bands_df[mode] = df
        self.logger.debug("Loaded frequency data for mode '%s'", mode)
        return df

    def load_epsilon(self, converted: bool = False):
        if converted:
            filepath = os.path.join(self.directory, f"{self.simulation_name}-epsilon.converted.h5")
        else:
            filepath = os.path.join(self.directory, f"{self.simulation_name}-epsilon.h5")
        data = self.load_h5_data(filepath)
        self.epsilon = data.get("data")
        self.lattice = data.get("lattice vectors")
        self.logger.debug("Loaded epsilon and lattice vectors using load_h5_data")
        return self.epsilon

    def _convert_array(self, md: mpb.MPBData, x: np.ndarray, periods=1, use_2d=True, is_fully_3d=False) -> np.ndarray:
        if x.ndim == 2:
            return md.convert(x)
        elif x.ndim == 3:
            if use_2d:
                mid = x.shape[2] // 2
                return md.convert(x[:, :, mid])
            elif not use_2d and not is_fully_3d:
                x_conv = md.convert(x)
                nz = x.shape[2]
                start = (x_conv.shape[2] - nz) // 2
                return x_conv[:, :, start:start+nz]
            else:
                return md.convert(x)
        else:
            raise ValueError("Invalid array dimensions")

    def convert_epsilon(self, periods=1, use_2d=True, is_fully_3d=False) -> np.ndarray:
        """
        Convert the loaded epsilon array using MPBData.
        """
        if self.lattice is None or self.epsilon is None:
            raise ValueError("Call load_epsilon() first.")
        mpb_data = mpb.MPBData(rectify=True, periods=periods, lattice=self.lattice)
        return self._convert_array(mpb_data, self.epsilon, periods, use_2d, is_fully_3d)

    def _run_mpb_data_conversion(self, input_file: str, output_file: str, options: dict) -> None:
        """
        Build and run an mpb-data conversion command.
        Common options (e.g., rectify, axis, resolution, periods, phase, transpose, pixellized, dataset)
        are passed via the options dict.
        """
        cmd = "source /dtu/sw/dcc/dcc-sw.bash && module load mpb/1.11.1 && mpb-data"
        if options.get("rectify", True):
            cmd += " -r"
        if "axis" in options and options["axis"]:
            cmd += f" -e {options['axis']}"
        if "resolution" in options and options["resolution"]:
            cmd += f" -n {options['resolution']}"
        if "periods" in options and options["periods"]:
            periods = options["periods"]
            if isinstance(periods, int):
                cmd += f" -m {periods}"
            elif isinstance(periods, (list, tuple)) and len(periods) == 3:
                cmd += f" -x {periods[0]} -y {periods[1]} -z {periods[2]}"
        if "phase" in options and options["phase"] is not None:
            cmd += f" -P {options['phase']}"
        if options.get("transpose", False):
            cmd += " -T"
        if options.get("pixellized", False):
            cmd += " -p"
        if "dataset" in options and options["dataset"]:
            cmd += f" -d {options['dataset']}"
        cmd += f" -o {output_file} {input_file}"
        self.logger.debug("Running mpb-data conversion command: %s", cmd)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            self.logger.debug(result.stderr)
            raise subprocess.CalledProcessError(result.returncode, cmd)

    def convert_field_data(self,
                           k_value: int,
                           b_value: int,
                           comp: str,
                           polarization: str,
                           field_type: str = "e",
                           conversion_options: dict = None) -> str:
        """
        Convert the field data file using mpb-data.
        
        Constructs the input filename from simulation metadata and uses conversion_options (a dict)
        to pass common options (e.g., rectangular, axis, resolution, periods, etc.).
        Returns the path to the converted file.
        """
        conversion_options = conversion_options or {}
        input_file = self.find_field_data(k_value, b_value, comp, polarization, field_type)
        output_filename = f"{self.simulation_name}-{field_type}.k{k_value:02d}.b{b_value:02d}.{comp}.{polarization}.converted.h5"
        output_filepath = os.path.join(self.directory, output_filename)
        self._run_mpb_data_conversion(input_file, output_filepath, conversion_options)
        return output_filepath

    def convert_epsilon_data(self,
                             conversion_options: dict = None) -> str:
        """
        Convert the epsilon file using mpb-data.
        
        The input epsilon file is assumed to be {simulation_name}-epsilon.h5.
        Returns the path to the converted epsilon file.
        """
        conversion_options = conversion_options or {}
        input_file = os.path.join(self.directory, f"{self.simulation_name}-epsilon.h5")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Epsilon file {input_file} not found.")
        output_filename = f"{self.simulation_name}-epsilon.converted.h5"
        output_filepath = os.path.join(self.directory, output_filename)
        self._run_mpb_data_conversion(input_file, output_filepath, conversion_options)
        return output_filepath

    def load_h5_data(self, filename: str) -> dict:
        """
        Load an HDF5 file and return its contents as a dictionary.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        with h5py.File(filename, 'r') as f:
            data = { key: f[key][...] for key in f.keys() }
        return data

    def get_frequencies_by_band(self, df, polarization: str = "te", bands: list[int] = [2, 3, 4], k_point: tuple = (0, 0, 0)) -> dict:
        """
        Get the frequencies for the specified bands and k-point.
        """
        if not {"k1", "k2", "k3"}.issubset(df.columns):
            raise ValueError("The database must contain 'k1', 'k2', and 'k3' columns for k-point matching.")
        k_coords = df[["k1", "k2", "k3"]].values
        target = np.array(k_point)
        distances = np.linalg.norm(k_coords - target, axis=1)
        closest_idx = distances.argmin()
        frequencies_by_band = {}
        for band in bands:
            band_col = f"{polarization} band {band}"
            if band_col not in df.columns:
                print(f"Band {band} not found in the database.")
                frequencies_by_band[band] = np.nan
            else:
                frequencies_by_band[band] = df[band_col].iloc[closest_idx]
        return frequencies_by_band

    def find_closest_k_point_row(self, df, target) -> object:
        """
        Find the row in the DataFrame corresponding to the closest k-point to the target.
        """
        keys = ["k1", "k2"]
        if "k3" in df.columns:
            keys.append("k3")
        target_arr = np.array(target)
        if target_arr.size < len(keys):
            target_arr = np.pad(target_arr, (0, len(keys) - target_arr.size), 'constant')
        distances = df.apply(
            lambda row: np.linalg.norm(np.array([row[k] for k in keys]) - target_arr),
            axis=1
        )
        return df.loc[distances.idxmin()]

    @property 
    def verbosity(self):
        return self.logger.level    
    
    @verbosity.setter
    def verbosity(self, level):
        self.logger.setLevel(level)
        print("Log level set to %s" % logging.getLevelName(int(level)))

    def set_verbosity(self, level):
        """Set the verbosity level of the logger."""
        if level not in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL):
            raise ValueError(f"Invalid verbosity level: {level}")
        self.verbosity = level
    
    def find_field_data(self, k_value: int, b_value: int, comp: str, polarization: str, field_type: str = "e"):
        """
        Find the field data file based on the given parameters.
        
        The file is expected to be named as:
        {simulation_name}-{field_type}.k{k_value:02d}.b{b_value:02d}.{comp}.{polarization}.h5
        
        Parameters:
            k_value (int): The k-point index (used both for the filename and for extracting the Bloch wavevector).
            b_value (int): The band index.
            comp (str): The field component specifier (e.g., 'x', 'y', or 'z').
            polarization (str): One of 'te', 'tm', 'zeven', or 'zodd'.
            field_type (str): The field type (default "e").
            
        Returns:
            str: The full path to the field data file.
        """
        filename = f"{self.simulation_name}-{field_type}.k{k_value:02d}.b{b_value:02d}.{comp}.{polarization}.h5"
        filepath = os.path.join(self.directory, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")
        return filepath
    
    def load_field_data(self, k_value: int, b_value: int, comp: str, polarization: str, field_type: str = "e"):
        """
        Load the field data file using h5py.
        
        The file is expected to be named as:
        {simulation_name}-{field_type}.k{k_value:02d}.b{b_value:02d}.{comp}.{polarization}.h5
        
        Parameters:
            k_value (int): The k-point index.
            b_value (int): The band index.
            comp (str): The field component specifier.
            polarization (str): The polarization mode.
            field_type (str): The field type (default "e").
        
        Returns:
            dict: The loaded data from the file.
        """
        filename = self.find_field_data(k_value, b_value, comp, polarization, field_type)
        return self.load_h5_data(filename)

class SimulationViewer:
    def __init__(self, simulation: Simulation):
        self.simulation = simulation

    def _apply_title(self, default_title: str, title: str | None):
        """
        Apply a title to the current matplotlib figure.
        """
        if title is False:
            return
        elif title is not None:
            new_title = title
        else:
            new_title = default_title
        plt.title(new_title)

    def plot_epsilon_2d(self, title: str | bool | None = None,
                        cmap: str = 'viridis', aspect_ratio: tuple = (1, 1), 
                        conversion_options: dict = {"rectify": True, "periods": 3}):
        """
        Plot the 2D epsilon data using convert_epsilon_data.
        """
        if self.simulation.epsilon is None:
            raise ValueError("Epsilon data not loaded. Call load_epsilon() on the simulation object.")
        # Convert the epsilon file using mpb-data conversion.
        converted_filepath = self.simulation.convert_epsilon_data(conversion_options)
        # Load the converted data from the file.
        data = self.simulation.load_h5_data(converted_filepath)
        eps = data.get("data")
        if eps is None:
            raise KeyError("Converted data not found in the file.")
        # If the data is three-dimensional, take the mid-plane.
        if eps.ndim == 3:
            mid_index = eps.shape[2] // 2
            eps = eps[:, :, mid_index]
        fig = plt.figure()
        plt.imshow(eps, interpolation='spline36', cmap=cmap)
        plt.colorbar()
        plt.gca().set_aspect(aspect_ratio[0] / aspect_ratio[1])
        self._apply_title(self.simulation.simulation_name, title)
        plt.show()
        return fig

    def plot_epsilon_3d(self, title: str | bool | None = None,
                          cmap: str = 'viridis', alpha: float = 0.3,
                          aspect_ratio: tuple = (1, 1, 1), 
                          conversion_options: dict = {"rectify": True, "periods": 3}):
        """
        Plot the 3D epsilon data as an isosurface.
        """
        filepath = self.simulation.convert_epsilon_data(conversion_options)
        data = self.simulation.load_h5_data(filepath)
        eps = data.get("data")
        iso = (np.min(eps) + np.max(eps)) / 2.0
        try:
            from skimage import measure
        except ImportError:
            raise ImportError("scikit-image is required for 3D plotting. Please install it.")
        verts, faces, normals, values = measure.marching_cubes(eps, level=iso)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        mesh = Poly3DCollection(verts[faces], alpha=alpha)
        colormap = plt.get_cmap(cmap)
        face_color = colormap(0.5)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        nx, ny, nz = eps.shape
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)
        try:
            ax.set_box_aspect(aspect_ratio)
        except Exception:
            pass
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        mappable = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=np.min(eps), vmax=np.max(eps)))
        mappable.set_array(eps)
        fig.colorbar(mappable, ax=ax, pad=0.1, label="Epsilon")
        default_title = f"{self.simulation.simulation_name} epsilon 3D"
        self._apply_title(default_title, title)
        plt.show()
        return fig

    def rotate_fig(self, fig: plt.Figure, azim: float, elev: float) -> plt.Figure:
        """
        Rotate the 3D view in the given figure.
        """
        ax = fig.axes[0] if fig.axes else None
        if ax is None:
            raise ValueError("The figure does not contain any axes.")
        ax.view_init(elev=elev, azim=azim)
        plt.draw()
        return fig

    def plot_field_data(self, k_value: int, b_value: int, comp: str, polarization: str, 
                        field_type: str = "e", plot_mode: str = "real", cmap: str = "RdBu",
                        conversion_options: dict = {"rectify": True, "periods": 3}):    
        """
        Plot the custom field data.
        
        If conversion is requested, the field data file is converted using mpb-data and then loaded.
        The complex field is reconstructed from 'z.r' and 'z.i' and then visualized based on plot_mode.
        """
        conversion_options = conversion_options or {}
        # Convert the field data file and get the converted file path.
        data = self.simulation.load_field_data(k_value, b_value, comp, polarization, field_type)
        description = data.get("description")
        converted_filepath = self.simulation.convert_field_data(k_value, b_value, comp, polarization, field_type, conversion_options)
        data = self.simulation.load_h5_data(converted_filepath)
        if "z.r" not in data or "z.i" not in data:
            raise KeyError("Field file must contain keys 'z.r' and 'z.i'.")
        field_complex = data["z.r"] + 1j * data["z.i"]
        if plot_mode == "real":
            field_to_plot = np.real(field_complex)
            title_mode = "Real"
        elif plot_mode == "imag":
            field_to_plot = np.imag(field_complex)
            title_mode = "Imaginary"
        elif plot_mode == "phase":
            field_to_plot = np.angle(field_complex)
            title_mode = "Phase"
        elif plot_mode == "abs":
            field_to_plot = np.abs(field_complex)
            title_mode = "Absolute"
        else:
            raise ValueError("plot_mode must be one of 'real', 'imag', 'phase', or 'abs'")
        plt.figure()
        im = plt.imshow(field_to_plot, interpolation='spline36', cmap=cmap)
        cbar = plt.colorbar(im)
        cbar.set_label(f"{field_type.upper()}{comp}, {title_mode}")

        plt.title(f"{self.simulation.simulation_name} {field_type} field: k{k_value:02d}, b{b_value:02d}, comp={comp}, {polarization}, {title_mode}")
        plt.suptitle(description)

    def plot_epsilon_contour(self, title: str | bool | None = None, conversion_options: dict = {"rectify": True, "periods": 3}):
        """
        Plot a contour of the 2D epsilon data.
        """
        output_filepath = self.simulation.convert_epsilon_data(conversion_options)
        data = self.simulation.load_h5_data(output_filepath)
        epsilon_converted = data.get("data")
        if epsilon_converted is None:
            raise KeyError("Converted data not found in the file.")
        iso = (np.min(epsilon_converted) + np.max(epsilon_converted)) / 2.0
        plt.contour(epsilon_converted, levels=[iso], colors='red', linewidths=3)
        self._apply_title(self.simulation.simulation_name, title)

    def plot_band_diagram(self, mode: str = "te", title: str | None = None, 
                          colors: list[str] | str | None = None, grid: bool = True, 
                          fig: plt.Figure | None = None,
                          k_points_path: dict | None = None) -> plt.Figure:
        """
        Plot the band diagram for the given mode.
        """
        fig = plt.figure() if fig is None else fig  
        if mode not in self.simulation.bands_df:
            df = self.simulation.load_frequency_data(mode)
        else:
            df = self.simulation.bands_df[mode]
        bands = df.columns[5:]
        plot_color = "C0"
        if isinstance(colors, list) and colors:
            plot_color = colors[0]
        elif isinstance(colors, str):
            plot_color = colors
        for i, col in enumerate(bands):
            if i == 0:
                plt.plot(df["k index"], df[col], label=f"{mode.upper()} bands", color=plot_color)
            else:
                plt.plot(df["k index"], df[col], color=plot_color)
        if k_points_path is not None and "k_points_values" in k_points_path and "k_points_labels" in k_points_path:
            k_points_values = k_points_path["k_points_values"]
            k_points_labels = k_points_path["k_points_labels"]
            custom_tick_positions = []
            n_custom = len(k_points_values)
            for idx, custom_k in enumerate(k_points_values):
                if idx == n_custom - 1:
                    tick_val = df["k index"].iloc[-1]
                else:
                    row = self.simulation.find_closest_k_point_row(df, custom_k)
                    tick_val = row["k index"]
                custom_tick_positions.append(tick_val)
            plt.xticks(ticks=custom_tick_positions, labels=k_points_labels)
        else:
            plt.xticks(df["k index"])
        plt.xlabel("k index")
        plt.ylabel("Frequency")
        self._apply_title(self.simulation.simulation_name, title)
        plt.legend()
        plt.grid(grid)
        return fig

    def plot_light_cone(self, df, fig: plt.Figure | None = None) -> plt.Figure:
        """
        Plot the light cone for the simulation.
        """
        fig = plt.figure() if fig is None else fig
        w = df['kmag/2pi']
        plt.plot(df['k index'], w, color='black', label='Light cone')
        return fig

    def show(self):
        plt.show()


# End of module.
