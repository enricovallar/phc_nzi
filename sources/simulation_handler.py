import os
import subprocess
import time
import re
import sqlite3
import tempfile
import logging
from typing import Optional, Union

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
                 directory: Optional[str] = None, description: Optional[str] = None, 
                 log_level: int = logging.INFO, save_script: bool = True):
        self.simulation_name = simulation_name
        self.directory = directory or simulation_name
        os.makedirs(self.directory, exist_ok=True)
        if save_script:
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
        if not scheme_script or not isinstance(scheme_script, str):
            raise ValueError("A valid scheme script must be provided as a string.")
        with open(os.path.join(self.directory, self.scheme_filename), "w") as f:
            f.write(scheme_script)
        if print_script:
            print(scheme_script)
        return scheme_script

    def _execute_command(self, cmd: str, shell: bool = False,
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
                             extra_options: Optional[list[str]] = None,
                             user_email: Optional[str] = None,
                             span_option: str = "hosts",
                             span_value: int = 1) -> list[str]:
        """
        Prepare the preamble lines for an LSF job submission script.
        """
        preamble = [
            "#!/bin/bash",
            f"#BSUB -J {simulation_name}",
            f"#BSUB -q {queue}",
            f"#BSUB -n {num_procs}",
            f"#BSUB -W {walltime}",
            f"#BSUB -R \"rusage[mem={mem}]\""
        ]
        
        span_str = {
            "hosts": f'span[hosts={span_value}]',
            "ptile": f'span[ptile={span_value}]',
            "block": f'span[block={span_value}]'
        }.get(span_option)
        if span_str:
            preamble.append(f"#BSUB -R \"{span_str}\"")
        
        preamble.extend([
            f"#BSUB -oo {simulation_name}.out",
            f"#BSUB -eo {simulation_name}.err"
        ])
        
        if user_email:
            preamble.append(f"#BSUB -u {user_email}")
        if extra_options:
            preamble.extend(extra_options)
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
            elif isinstance(periods, (list, tuple)) and len(periods) == 2:
                cmd += f" -x {periods[0]} -y {periods[1]}"
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
                           polarization: str,
                           field_type: str = "e",
                           conversion_options: Optional[dict] = None, 
                           file_comp: str|None = None,
                           nonbloch = True
                           ) -> str:
        """
        Convert the field data file using mpb-data.
        Returns the path to the converted file.
        """
        conversion_options = conversion_options or {}
        input_file = self.find_field_data(k_value, b_value, polarization, field_type, file_comp, nonbloch)
        field_type = f"{field_type}.v" if nonbloch is True else field_type
        if file_comp is not None:
            output_filename = f"{self.simulation_name}-{field_type}.k{k_value:02d}.b{b_value:02d}.{file_comp}.{polarization}.converted.h5"
        else:
            output_filename = f"{self.simulation_name}-{field_type}.k{k_value:02d}.b{b_value:02d}.{polarization}.converted.h5"
        output_filepath = os.path.join(self.directory, output_filename)
        self._run_mpb_data_conversion(input_file, output_filepath, conversion_options)
        return output_filepath

    def convert_epsilon_data(self,
                             conversion_options: Optional[dict] = None) -> str:
        """
        Convert the epsilon file using mpb-data.
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
    
    def find_field_data(self, k_value: int, b_value: int, polarization: str, field_type: str = "e", file_comp: str|None = None, nonbloch = True) -> str:
        """
        Find the field data file based on the given parameters.
        """
        if nonbloch is True:
            field_type = f"{field_type}.v"
        if file_comp is not None: 
            filepath = f"{self.simulation_name}-{field_type}.k{k_value:02d}.b{b_value:02d}.{file_comp}.{polarization}.h5"
        else:
            filepath = f"{self.simulation_name}-{field_type}.k{k_value:02d}.b{b_value:02d}.{polarization}.h5"
        filepath = os.path.join(self.directory, filepath)   
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")
        return filepath

    def load_field_data(self, k_value: int, b_value: int, polarization: str, field_type: str = "e", file_comp: str|None = None, nonbloch = True) -> dict:
        """
        Load the field data file using h5py.
        """
        filename = self.find_field_data(k_value, b_value, polarization, field_type, file_comp, nonbloch)
        return self.load_h5_data(filename)
    
    def load_and_convert_field_data(self,
                                     k_value: int,
                                     b_value: int,
                                     comp: str,
                                     polarization: str,
                                     field_type: str,
                                     conversion_options: dict,
                                     file_comp: str|None = None,
                                     nonbloch = True,
                                    ) -> tuple[np.ndarray, str]:
        """
        Load and convert the field data, then reconstruct the complex field.
        """
        data = self.load_field_data(k_value, b_value, polarization, field_type, file_comp, nonbloch)
        description = data.get("description", "")
        converted_filepath = self.convert_field_data(
            k_value, b_value, polarization, field_type, conversion_options, file_comp=file_comp, nonbloch = nonbloch
        )
        data = self.load_h5_data(converted_filepath)
        if f"{comp}.r" not in data or f"{comp}.i" not in data:
            raise KeyError(f"Field file must contain '{comp}.r' and '{comp}.i'.")
        field_complex = data[f"{comp}.r"] + 1j * data[f"{comp}.i"]
        return field_complex, description

class SimulationViewer:
    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation

    def newfig(self) -> None:
        """Create a new figure (clearing any existing figure)."""
        plt.figure()

    def _apply_title(self, ax: plt.Axes, main_title: str, subtitle: Optional[str] = None) -> None:
        """
        Apply a main title to the given Axes, and optionally a subtitle at the figure level.
        """
        if main_title:
            ax.set_title(main_title)
        if subtitle:
            plt.suptitle(subtitle)

    def plot_epsilon_2d(self,
                        title: Optional[str] = None,
                        cmap: str = 'viridis',
                        aspect_ratio: tuple[float, float] = (1, 1),
                        conversion_options: dict = {"rectify": True, "periods": 3}
                       ) -> None:
        """
        Plot the 2D epsilon data using the converted epsilon file on the current axes.
        Does not call plt.show().
        """
        if self.simulation.epsilon is None:
            raise ValueError("Epsilon data not loaded. Call load_epsilon() first.")
        converted_filepath = self.simulation.convert_epsilon_data(conversion_options)
        data = self.simulation.load_h5_data(converted_filepath)
        eps = data.get("data")
        if eps is None:
            raise KeyError("Converted epsilon data not found.")
        if eps.ndim == 3:
            mid_index = eps.shape[2] // 2
            eps = eps[:, :, mid_index]
        plt.imshow(eps, interpolation='spline36', cmap=cmap)
        plt.colorbar()
        plt.gca().set_aspect(aspect_ratio[0] / aspect_ratio[1])
        ax = plt.gca()
        self._apply_title(ax, main_title=self.simulation.simulation_name, subtitle=title)

    def plot_epsilon_3d(self,
                        title: Optional[str] = None,
                        cmap: str = 'viridis',
                        alpha: float = 0.3,
                        aspect_ratio: tuple[float, float, float] = (1, 1, 1),
                        conversion_options: dict = {"rectify": True, "periods": 3}
                       ) -> None:
        """
        Overlay the 3D epsilon isosurface on the current 3D axes. Does not call plt.show().
        """
        filepath = self.simulation.convert_epsilon_data(conversion_options)
        data = self.simulation.load_h5_data(filepath)
        eps = data.get("data")
        if eps is None:
            raise KeyError("Converted epsilon data not found.")
        iso = 0.5 * (np.min(eps) + np.max(eps))
        from skimage import measure
        verts, faces, normals, _ = measure.marching_cubes(eps, level=iso)
        fig = plt.gcf()
        old_ax = plt.gca()
        if not hasattr(old_ax, 'view_init'):
            fig.delaxes(old_ax)
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = old_ax
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        mesh = Poly3DCollection(verts[faces], alpha=alpha)
        face_color = plt.get_cmap(cmap)(0.5)
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
        plt.colorbar(mappable, ax=ax, pad=0.1, label="Epsilon")
        if title:
            ax.set_title(title)

    def plot_epsilon_contour(self,
                             conversion_options: dict = {"rectify": True, "periods": 3},
                             title: Optional[str] = None
                            ) -> None:
        """
        Overlay an epsilon contour on the current axes. Does not call plt.show().
        """
        output_filepath = self.simulation.convert_epsilon_data(conversion_options)
        data = self.simulation.load_h5_data(output_filepath)
        epsilon_converted = data.get("data")
        if epsilon_converted is None:
            raise KeyError("Converted epsilon data not found.")
        iso = 0.5 * (np.min(epsilon_converted) + np.max(epsilon_converted))
        plt.contour(epsilon_converted, levels=[iso], colors='red', linewidths=3)
        ax = plt.gca()
        self._apply_title(ax, main_title=self.simulation.simulation_name, subtitle=title)

    def rotate_fig(self, azim: float, elev: float) -> None:
        """
        Rotate the 3D view in the current axes.
        """
        ax = plt.gca()
        if not hasattr(ax, "view_init"):
            raise ValueError("Current axes is not 3D.")
        ax.view_init(elev=elev, azim=azim)

    def plot_field_data(self,
                        k_value: int,
                        b_value: int,
                        comp: str, 
                        polarization: str,
                        field_type: str = "e",
                        plot_mode: str = "real",
                        cmap: str = "RdBu",
                        conversion_options: dict = {"rectify": True, "periods": 3},
                        slice_axis: int = 0,
                        slice_index: Optional[int] = None,
                        overlay_epsilon: bool = False,
                        epsilon_cmap: str = "viridis",
                        epsilon_alpha: float = 0.3,
                        overlay_epsilon_slice_contour: bool = False, 
                        file_comp: str= None,
                        nonbloch = True
                       ) -> None:
        """
        Plot field data on the current axes. Does not call plt.show().
        """
        field_complex, description = self.simulation.load_and_convert_field_data(
            k_value, b_value, comp, polarization, field_type, conversion_options, file_comp, nonbloch
        )
        if field_complex.ndim == 3:
            self._plot_3d_field(
                field_complex, plot_mode, slice_axis, slice_index,
                field_type, comp, k_value, b_value, polarization, cmap,
                description, overlay_epsilon, epsilon_cmap, epsilon_alpha,
                conversion_options, overlay_epsilon_slice_contour
            )
        elif field_complex.ndim == 2:
            self._plot_2d_field(
                field_complex, plot_mode, field_type, comp, k_value, b_value,
                polarization, cmap, description, overlay_epsilon, conversion_options
            )
        else:
            raise ValueError("Unsupported field dimensions (expected 2 or 3).")

    

    def _extract_field_data(self, field_complex: np.ndarray, plot_mode: str) -> tuple[np.ndarray, str]:
        """
        Extract the desired 2D field array and a label for the title.
        """
        if plot_mode == "real":
            return np.real(field_complex), "Real"
        elif plot_mode == "imag":
            return np.imag(field_complex), "Imaginary"
        elif plot_mode == "phase":
            return np.angle(field_complex), "Phase"
        elif plot_mode == "abs":
            return np.abs(field_complex), "Absolute"
        else:
            raise ValueError("plot_mode must be 'real', 'imag', 'phase', or 'abs'")

    def _plot_2d_field(self,
                       field_complex: np.ndarray,
                       plot_mode: str,
                       field_type: str,
                       comp: str,
                       k_value: int,
                       b_value: int,
                       polarization: str,
                       cmap: str,
                       description: str,
                       overlay_epsilon: bool,
                       epsilon_conversion_options: dict
                      ) -> None:
        """
        Plot a 2D field using imshow. Optionally overlay an epsilon contour.
        """
        field_2d, title_mode = self._extract_field_data(field_complex, plot_mode)
        plt.imshow(field_2d, interpolation='spline36', cmap=cmap)
        cbar = plt.colorbar()
        cbar.set_label(f"{field_type.upper()}{comp}, {title_mode}")
        plt.title(f"{self.simulation.simulation_name} {field_type} field: k{k_value:02d}, b{b_value:02d}, "
                  f"comp={comp}, {polarization}, {title_mode}")
        plt.suptitle(description)
        if overlay_epsilon:
            self.plot_epsilon_contour(epsilon_conversion_options)

    def _plot_3d_field(self,
                       field_complex: np.ndarray,
                       plot_mode: str,
                       slice_axis: int,
                       slice_index: Optional[int],
                       field_type: str,
                       comp: str,
                       k_value: int,
                       b_value: int,
                       polarization: str,
                       cmap: str,
                       description: str,
                       overlay_epsilon: bool,
                       epsilon_cmap: str,
                       epsilon_alpha: float,
                       conversion_options: dict,
                       overlay_epsilon_slice_contour: bool
                      ) -> None:
        """
        Plot a 2D slice of a 3D field on a 3D axes, with orientation depending on slice_axis.
        Optionally overlay epsilon as a 3D isosurface or a 2D contour on the same slice.
        """
        field_data, title_mode = self._extract_field_data(field_complex, plot_mode)
        if slice_index is None:
            slice_index = field_data.shape[slice_axis] // 2
        if slice_axis == 0:
            field_slice = np.take(field_data, slice_index, axis=0)
            Y, Z = np.meshgrid(
                np.arange(field_slice.shape[0]),
                np.arange(field_slice.shape[1]),
                indexing='ij'
            )
            X = np.full_like(Y, slice_index)
            plane_label = "(yz plane)"
            zdir_for_contour = 'x'
            # Instead of offset_for_contour = slice_index, use the maximum x value
            offset_for_contour = field_data.shape[0] - 1
        elif slice_axis == 1:
            field_slice = np.take(field_data, slice_index, axis=1)
            X, Z = np.meshgrid(
                np.arange(field_slice.shape[0]),
                np.arange(field_slice.shape[1]),
                indexing='ij'
            )
            Y = np.full_like(X, slice_index)
            plane_label = "(xz plane)"
            zdir_for_contour = 'y'
            # Use the maximum y value
            offset_for_contour = field_data.shape[1] - 1
        elif slice_axis == 2:
            field_slice = np.take(field_data, slice_index, axis=2)
            X, Y = np.meshgrid(
                np.arange(field_slice.shape[0]),
                np.arange(field_slice.shape[1]),
                indexing='ij'
            )
            Z = np.full_like(X, slice_index)
            plane_label = "(xy plane)"
            zdir_for_contour = 'z'
            # Use the maximum z value
            offset_for_contour = field_data.shape[2] - 1
        else:
            raise ValueError("slice_axis must be 0, 1, or 2.")

        fig = plt.gcf()
        old_ax = plt.gca()
        if not hasattr(old_ax, 'view_init'):
            fig.delaxes(old_ax)
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = old_ax
        norm_field = plt.Normalize(vmin=field_slice.min(), vmax=field_slice.max())
        facecolors_field = plt.cm.get_cmap(cmap)(norm_field(field_slice))
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        facecolors=facecolors_field,
                        shade=False, antialiased=False)
        mappable_field = plt.cm.ScalarMappable(norm=norm_field, cmap=cmap)
        mappable_field.set_array(field_slice)
        plt.colorbar(mappable_field, ax=ax, shrink=0.5, aspect=5,
                     label=f"{field_type.upper()}{comp}, {title_mode}")
        if overlay_epsilon:
            self.plot_epsilon_3d(title=None,
                                 cmap=epsilon_cmap,
                                 alpha=epsilon_alpha,
                                 conversion_options=conversion_options)
        if overlay_epsilon_slice_contour:
            self._plot_epsilon_contour_on_slice(ax, slice_index, slice_axis, 
                                                X, Y, 
                                                zdir_for_contour, 
                                                offset_for_contour, 
                                                conversion_options)

        main_title = (f"{self.simulation.simulation_name} {field_type} field: "
                      f"k{k_value:02d}, b{b_value:02d}, comp={comp}, {polarization}, {title_mode}\n"
                      f"slice_axis={slice_axis}, index={slice_index} {plane_label}")
        self._apply_title(ax, main_title=main_title)

    def _plot_epsilon_contour_on_slice(self, ax: plt.Axes, slice_index: int, slice_axis: int, X, Y, zdir: str, offset: float, conversion_options: dict) -> None:
        try:
            eps_filepath = self.simulation.convert_epsilon_data(conversion_options)
            eps_data = self.simulation.load_h5_data(eps_filepath).get("data")
        except Exception as e:
            raise RuntimeError(f"Error loading epsilon data: {e}")
        if eps_data is None or eps_data.ndim != 3:
            return
        eps_slice = np.take(eps_data, slice_index, axis=slice_axis)
        # Adjust iso level to ensure the contour stands out
        if np.ptp(eps_slice) == 0:
            iso = np.mean(eps_slice)
        else:
            iso = np.mean(eps_slice) + 0.1 * np.ptp(eps_slice)
        ax.contour(X, Y, eps_slice, zdir=zdir, offset=offset, levels=[iso], colors='red', linewidths=2)

    def plot_band_diagram(self,
                          mode: str = "te",
                          title: Optional[str] = None,
                          colors: Optional[Union[list[str], str]] = None,
                          grid: bool = True,
                          k_points_path: Optional[dict] = None
                         ) -> None:
        """
        Plot the band diagram for the given mode on the current axes. Does not call plt.show().
        """
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
        if k_points_path and "k_points_values" in k_points_path and "k_points_labels" in k_points_path:
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
        ax = plt.gca()
        self._apply_title(ax, main_title=self.simulation.simulation_name, subtitle=title)
        plt.legend()
        plt.grid(grid)

    def plot_light_cone(self, df: pd.DataFrame) -> None:
        """
        Plot the light cone for the simulation on the current axes. Does not call plt.show().
        """
        plt.plot(df['k index'], df['kmag/2pi'], color='black', label='Light cone')
        plt.legend()

    def show(self) -> None:
        """Show the current figure."""
        plt.show()
