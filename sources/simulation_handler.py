import os
import subprocess
import time
import re
import sqlite3

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import meep as mp
from meep import mpb
from mpb_configurator import MPBSchemeConfigurator

import plotly.graph_objects as go
import logging
import tempfile

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
                 directory: str = None, description: str = None, log_level: int = logging.INFO, save_script = True):
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
        elif isinstance(scheme_script, str)==False:
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
        Prepare the preamble lines for an LSF job submission script with span support and
        output/error file directives.

        Parameters:
            simulation_name: Name of the simulation (used as job name).
            queue: LSF queue name.
            num_procs: Number of processors to request.
            walltime: Maximum wall-clock time (e.g., "24:00").
            mem: Memory per process (e.g., "4GB").
            extra_options: Additional BSUB options as a list of strings.
            user_email: User email to receive notifications.
            span_option: Option for span constraint ('hosts', 'ptile', or 'block').
            span_value: The value to use with the span option.

        Returns:
            A list of strings representing the preamble of the LSF script.
        """

        
        preamble = []
        preamble.append("#!/bin/bash")
        preamble.append(f"#BSUB -J {simulation_name}")
        preamble.append(f"#BSUB -q {queue}")
        preamble.append(f"#BSUB -n {num_procs}")
        preamble.append(f"#BSUB -W {walltime}")
        preamble.append(f"#BSUB -R \"rusage[mem={mem}]\"")
        
        # Add span support if span_option is provided.
        span_str = {
            "hosts": f'span[hosts={span_value}]',
            "ptile": f'span[ptile={span_value}]',
            "block": f'span[block={span_value}]'
        }.get(span_option)
        if span_str:
            preamble.append(f"#BSUB -R \"{span_str}\"")
        
        # Add output and error file directives.
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
       

        # Write the scheme script to a temporary file.
        filename = self.scheme_filename
        with open(filename, "w") as f:
            f.write(self.script)
        

        # Convert additional command-line parameters.
        params = " ".join(f"{k}={v}" for k, v in mpb_command_line_params.items())

        # Prepare the LSF preamble using the provided method.
        preamble_lines = self.prepare_lsf_preamble(simulation_name = self.simulation_name, queue=queue,
                                                     num_procs=num_procs, span_option=span_option,
                                                     span_value=span_value)
        # Build the job script that will be submitted.
        job_script = "\n".join(preamble_lines) + "\n\n"
        job_script += (
            f"source /dtu/sw/dcc/dcc-sw.bash && module load mpb/1.11.1 && "
            f"mpirun -np $LSB_DJOB_NUMPROC mpb-mpi {params} {filename}"
        )

        # Write the complete job script to a temporary file.
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=self.directory, suffix=".sh") as job_tmp:
            job_tmp.write(job_script)
            job_script_file = job_tmp.name

        # Submit the job via bsub; the job script is fed to bsub via input redirection.
        cmd = (f"bsub -oo {self.output_filename} -eo {self.error_filename} "
               f"< {job_script_file}")
        self.logger.debug("Running LSF job command: %s", cmd)
        result = self._execute_command(cmd, shell=True,
                                       print_output=print_output, print_error=print_error)

        # Extract job ID and poll for job completion.
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

        # Wait for output and error files to become available.
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

    def load_epsilon(self):
        filepath = os.path.join(self.directory, f"{self.simulation_name}-epsilon.h5")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"HDF5 file {filepath} not found.")
        with h5py.File(filepath, 'r') as f:
            self.epsilon = f['data'][...]
            self.lattice = f['lattice vectors'][...] if 'lattice vectors' in f else None
        self.logger.debug("Loaded epsilon and lattice vectors")

    def convert_epsilon(self, periods=1, use_2d=True, is_fully_3d=False) -> np.ndarray:
        if self.lattice is None or self.epsilon is None:
            raise ValueError("Call load_epsilon() first.")
        mpb_data = mpb.MPBData(rectify=True, periods=periods, lattice=self.lattice)
        return self._convert_array(mpb_data, self.epsilon, periods, use_2d, is_fully_3d)

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

    def load_frequency_data(self, mode: str = "te") -> pd.DataFrame:
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
    
    
    def get_frequencies_by_band(self, df, polarization: str = "te", bands: list[int] = [2, 3, 4], k_point: tuple = (0, 0, 0)) -> dict:
        """
        Get the frequencies for the specified bands and k-point.

        Parameters:
            polarization: The polarization mode (e.g., 'te', 'tm').
            bands: List of band numbers to retrieve.
            k_point: The k-point as a tuple (k1, k2, k3).
            df: The DataFrame containing the frequency data.

        Returns:
            frequencies_by_band: A dictionary mapping band numbers to their frequency at the given k-point.
        """

        # Ensure that the required k-point columns exist.
        if not {"k1", "k2", "k3"}.issubset(df.columns):
            raise ValueError("The database must contain 'k1', 'k2', and 'k3' columns for k-point matching.")

        # Find the row closest to the provided k_point using the k1, k2, k3 columns.
        k_coords = df[["k1", "k2", "k3"]].values
        target = np.array(k_point)
        distances = np.linalg.norm(k_coords - target, axis=1)
        closest_idx = distances.argmin()

        # Initialize a dictionary to store the frequencies by band.
        frequencies_by_band = {}

        # For each band, extract the frequency at the closest k-point.
        for band in bands:
            band_col = f"{polarization} band {band}"
            if band_col not in df.columns:
                print(f"Band {band} not found in the database.")
                frequencies_by_band[band] = np.nan
            else:
                frequencies_by_band[band] = df[band_col].iloc[closest_idx]

        return frequencies_by_band

    @property 
    def verbosity(self):
        return self.logger.level    
    
    @verbosity.setter
    def verbosity(self, level):
        self.logger.setLevel(level)
        print("Log level set to %s" % logging.getLevelName(int(level)))

    def set_verbosity(self, level):
        """ Set the verbosity level of the logger. """
        if level not in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL):
            raise ValueError(f"Invalid verbosity level: {level}")
        self.verbosity = level  


    



class SimulationViewer:
    def __init__(self, simulation: Simulation):
        self.simulation = simulation

    def _apply_title(self, default_title: str, title: str | None, fig: object = None):
        """
        Apply the title to the current figure. If a figure is passed and is a Plotly figure,
        update its layout; if it's a matplotlib figure (or if no figure is passed), use the usual
        title functions.
        """
        if title is False:
            return
        elif title is not None:
            new_title = title
        else:
            new_title = default_title

        # If fig is None, assume we are working with the current matplotlib axes.
        if fig is None:
            plt.title(new_title)
        elif isinstance(fig, go.Figure):
            fig.update_layout(title=dict(text=new_title))
        else:
            # Assume fig is a matplotlib figure
            if hasattr(fig, 'axes') and fig.axes:
                fig.axes[0].set_title(new_title)

    def plot_epsilon_2d(self, periods: int | tuple = 1, title: str | bool | None = None, 
                        converted: bool = True, cmap: str = 'viridis', aspect_ratio: tuple = (1, 1)):
        """
        Plot 2D epsilon data.
        
        Parameters:
            periods: number of periods to use in conversion.
            title: Title for the plot; if False, no title is set.
            converted: If True, use self.simulation.convert_epsilon() to convert the raw epsilon data.
            cmap: Colormap name.
            aspect_ratio: Tuple (rx, ry) for setting the aspect ratio (default (1,1) for equal scaling).
        """
        
        if self.simulation.epsilon is None:
            raise ValueError("Epsilon data not loaded. Call load_epsilon() on the simulation object.")
        fig = plt.figure()
        if converted:
            eps = self.simulation.convert_epsilon(periods, use_2d=True)
        else:
            eps = self.simulation.epsilon
            if eps.ndim == 3:
                mid_index = eps.shape[2] // 2
                eps = eps[:, :, mid_index]
        plt.imshow(eps, interpolation='spline36', cmap=cmap)
        plt.colorbar()
        # Set the aspect ratio; for matplotlib imshow we use the ratio of the two numbers.
        plt.gca().set_aspect(aspect_ratio[0] / aspect_ratio[1])
        self._apply_title(self.simulation.simulation_name, title)
        plt.show()
        return fig

        
        
    def plot_epsilon_3d(self, periods: int | tuple = 1, title: str | bool | None = None,
                        converted: bool = True, cmap: str = 'viridis', alpha: float = 0.3,
                        aspect_ratio: tuple = (1, 1, 1)) -> plt.Figure:
        """
        Plot the 3D dielectric constant data as an isosurface using matplotlib.
        The method uses the marching cubes algorithm (from scikit-image) to extract an isosurface
        (using the mid-value as the default isosurface level) from the 3D epsilon array.
        The surface is plotted with a semi-transparent face color, a colorbar is added,
        and the 3D axes are set to the specified aspect_ratio.
        
        Parameters:
            periods: Number of periods to use in conversion.
            title: Title for the plot; if False, no title is set.
            converted: If True, use self.simulation.convert_epsilon() to convert the raw epsilon data.
            cmap: Name of the matplotlib colormap to use.
            alpha: Opacity for the isosurface.
            aspect_ratio: Tuple (rx, ry, rz) for setting the 3D axis aspect ratio (default (1,1,1)).
            
        Returns:
            fig: The matplotlib Figure object.
        """
        # Ensure that epsilon data is available.
        if self.simulation.epsilon is None:
            raise ValueError("Epsilon data not loaded. Call load_epsilon() first.")
        
        # Retrieve the 3D epsilon array.
        if converted:
            eps = self.simulation.convert_epsilon(periods, use_2d=False, is_fully_3d=False)
        else:
            eps = self.simulation.epsilon

        # Compute the isosurface level as the midpoint of the data.
        iso = (np.min(eps) + np.max(eps)) / 2.0

        # Extract the isosurface using marching cubes.
        try:
            from skimage import measure
        except ImportError:
            raise ImportError("scikit-image is required for 3D plotting. Please install it (e.g., pip install scikit-image).")
        verts, faces, normals, values = measure.marching_cubes(eps, level=iso)
        
        # Create a new 3D matplotlib figure.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a Poly3DCollection from the extracted isosurface.
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        mesh = Poly3DCollection(verts[faces], alpha=alpha)
        
        # Set the face color using the given colormap (using the midpoint of the colormap).
        colormap = plt.get_cmap(cmap)
        face_color = colormap(0.5)  # Adjust this value as needed.
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        
        # Set the axis limits based on the epsilon data dimensions.
        nx, ny, nz = eps.shape
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)
        
        # Set the aspect ratio if possible (requires matplotlib >= 3.3).
        try:
            ax.set_box_aspect(aspect_ratio)
        except Exception:
            pass
        
        # Add a colorbar.
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        mappable = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=np.min(eps), vmax=np.max(eps)))
        mappable.set_array(eps)
        fig.colorbar(mappable, ax=ax, pad=0.1, label="Epsilon")
        
        # Apply the title.
        default_title = f"{self.simulation.simulation_name} epsilon 3D"
        self._apply_title(default_title, title)
        plt.show()
        return fig
    
    def plotly_epsilon_3d(self, periods: int | tuple = 1, title: str | bool | None = None,
                            converted: bool = True, cmap: str = 'viridis', alpha: float = 0.3,
                            width: int = 800, height: int = 600, aspect_ratio: tuple = (1, 1, 1),
                            fig: go.Figure | None = None) -> go.Figure:
        """
        Plot the 3D dielectric constant data as an isosurface using Plotly.
        The method uses a marching cubes algorithm to extract an isosurface (using the mid-value
        as a default isosurface level) from the 3D epsilon array. The surface is rendered with
        a semi-transparent face color, an attached colorbar, and the scene's aspect ratio is set 
        according to the provided aspect_ratio tuple.
        
        Parameters:
            periods: number of periods to use in conversion.
            title: Title for the plot; if False, no title is set.
            converted: If True, use self.simulation.convert_epsilon() to convert the raw epsilon data.
            cmap: Name of the matplotlib colormap to use; also used as a Plotly colorscale.
                (E.g., 'viridis' is supported.)
            alpha: Opacity for the isosurface.
            width: Figure width in pixels.
            height: Figure height in pixels.
            aspect_ratio: A 3-tuple (rx, ry, rz) to set the scene's aspect ratio (default (1,1,1)).
            fig: An existing Plotly figure to which the isosurface will be added. If None,
                a new figure is created.
        
        Returns:
            fig: The Plotly figure object.
        """
        # Ensure that epsilon data is available.
        if self.simulation.epsilon is None:
            raise ValueError("Epsilon data not loaded. Call load_epsilon() first.")
        
        # Get the 3D epsilon array.
        if converted:
            eps = self.simulation.convert_epsilon(periods, use_2d=False, is_fully_3d=False)
        else:
            eps = self.simulation.epsilon

        # Compute the isosurface level as the midpoint.
        iso = (np.min(eps) + np.max(eps)) / 2.0

        # Use marching cubes to extract the isosurface.
        try:
            from skimage import measure
        except ImportError:
            raise ImportError("scikit-image is required for 3D plotting. Please install it (e.g., pip install scikit-image).")
        verts, faces, normals, values = measure.marching_cubes(eps, level=iso)
        
        # If no figure is passed, create a new one with the specified width and height.
        if fig is None:
            fig = go.Figure(layout=go.Layout(width=width, height=height))
        
        # Create a Mesh3d trace.
        # Using 'intensity' equal to the marching cubes values so that a colorscale and colorbar are shown.
        mesh = go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=values,
            colorscale=cmap,
            opacity=alpha,
            showscale=True,
            colorbar=dict(title='Epsilon')
        )
        
        fig.add_trace(mesh)
        
        # Set axis limits based on the epsilon data dimensions.
        nx, ny, nz = eps.shape
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[0, nx]),
                yaxis=dict(range=[0, ny]),
                zaxis=dict(range=[0, nz]),
                aspectmode='manual',
                aspectratio=dict(x=aspect_ratio[0], y=aspect_ratio[1], z=aspect_ratio[2])
            )
        )
        
        # Set the title if requested. _apply_title is assumed to handle both matplotlib and Plotly figures.
        default_title = f"{self.simulation.simulation_name} epsilon 3D"
        self._apply_title(default_title, title, fig=fig)
        fig.show()
        return fig



    
    def rotate_fig(self, fig: plt.Figure, azim: float, elev: float) -> plt.Figure:
        """
        Rotate the 3D axes in the given figure to the specified azimuth and elevation angles.
        
        Parameters:
            fig: The matplotlib figure object containing the 3D axes.
            azim: The azimuth angle in degrees (rotation about the z-axis).
            elev: The elevation angle in degrees (rotation about the x-axis).
        
        Returns:
            fig: The updated matplotlib figure object with the new view.
        """
        # Get the current 3D axis. If there are multiple axes, this example uses the first one.
        ax = fig.axes[0] if fig.axes else None
        if ax is None:
            raise ValueError("The figure does not contain any axes.")
        
        # For a 3D axis, set the view using the view_init method.
        ax.view_init(elev=elev, azim=azim)
        
        # Optionally force a redraw of the figure.
        plt.draw()
        return fig


    def plot_epsilon(self, periods: int | tuple = 1, title: str | bool | None = None, converted: bool = True, cmap: str = 'viridis'):
        if self.simulation.epsilon.ndim == 2:   
            self.plot_epsilon_2d(periods, title, converted, cmap)
        elif self.simulation.epsilon.ndim == 3:
            self.plot_epsilon_3d(periods, title, converted, cmap)
        else:
            raise ValueError("Invalid epsilon data dimensions")
        

    def plot_epsilon_contour(self, periods: int | tuple = 1, title: str | bool | None = None):
        """
        Plot a contour of the converted 2D epsilon data.
        """
        epsilon_converted = self.simulation.convert_epsilon(periods, use_2d=True)
        plt.contour(epsilon_converted, cmap='binary')
        self._apply_title(self.simulation.simulation_name, title)
        
            

    def plot_band_diagram(self, mode: str = "te", title: str | None = None, 
                            colors: list[str] | str | None = None, decimation_label_factor: int = 1, grid: bool = True, 
                            fig: plt.Figure | None = None,
                            k_points_values: list | None = None, k_points_labels: list | None = None
                            ) -> plt.Figure:
        """
        Plot the band diagram for the given mode.
        All bands will use the same color.
        The x-axis tick labels are formatted as (kₓ, kᵧ) in LaTeX if decimation_label_factor > 1,
        with decimation controlled by decimation_label_factor.
        The y-axis label is always frequency.
        The legend includes the polarization mode.
        
        Parameters:
            k_points_values: List of custom k-point vectors (each an iterable representing [k1, k2, ...]).
                            For each custom value, the label will be assigned to the closest tick (based on the
                            Euclidean distance in the (k1,k2) plane) except for the last custom point.
                            For the last custom point, the tick is set to the last k-index in the database.
            k_points_labels: List of labels corresponding to k_points_values.
            
        Returns:
            fig: The matplotlib Figure object.
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

        # Default tick labels: if decimation_label_factor > 1 and k1/k2 exist.
        if decimation_label_factor > 1 and {"k1", "k2"}.issubset(df.columns):
            tickvals = df["k index"].values[::decimation_label_factor]
            ticklabels = [f"({row['k1']:.2f}, {row['k2']:.2f})" 
                        for _, row in df.iloc[::decimation_label_factor].iterrows()]
            plt.xticks(tickvals, ticklabels)
        else:
            plt.xticks(df["k index"])

        # If custom k-points are provided, override the ticks.
        if k_points_values is not None and k_points_labels is not None:
            # Build a dictionary mapping row indices to the 2D k-point as a numpy array.
            db_vectors = {}
            for i, row in df.iterrows():
                db_vectors[i] = np.array([row["k1"], row["k2"]])
            
            custom_tick_positions = []
            custom_tick_labels = []
            n_custom = len(k_points_values)
            for idx, (custom_k, label) in enumerate(zip(k_points_values, k_points_labels)):
                if idx == n_custom - 1:
                    # For the last custom k-point, assign the last "k index" in the database.
                    tick_val = df["k index"].iloc[-1]
                else:
                    custom_arr = np.array(custom_k)[:2]  # Only compare the k1, k2 components.
                    closest_index = min(db_vectors, key=lambda i: np.linalg.norm(db_vectors[i] - custom_arr))
                    tick_val = df.loc[closest_index, "k index"]
                custom_tick_positions.append(tick_val)
                custom_tick_labels.append(label)
            plt.xticks(ticks=custom_tick_positions, labels=custom_tick_labels)

        plt.xlabel("k index" if decimation_label_factor == 1 else r"$(k_x, k_y)$")
        plt.ylabel("Frequency")
        self._apply_title(self.simulation.simulation_name, title)
        plt.legend()
        plt.grid(grid)
        return fig


    def show(self):
        plt.show()
    
    TEXT_HUGE = {"family": "Arial", "size": 28, "color": "black"}  
    TEXT_BIG = {"family": "Arial", "size": 23, "color": "black"}
    TEXT_MEDIUM = {"family": "Arial", "size": 20, "color": "black"}
    TEXT_SMALL = {"family": "Arial", "size": 18, "color": "black"}
    
    @staticmethod
    def _font_config(font: dict | str | None | int):
        if font is None:
            return SimulationViewer.TEXT_MEDIUM
        if type(font) is str:
            if font == "small":
                return SimulationViewer.TEXT_SMALL
            elif font == "medium":
                return SimulationViewer.TEXT_MEDIUM
            elif font == "big":
                return SimulationViewer.TEXT_BIG
            elif font == "huge":
                return SimulationViewer.TEXT_HUGE
            else:
                raise ValueError("Invalid font size, select from 'small', 'medium', 'big', 'huge'")                   
        elif type(font) is int:
            return {"family": "Arial", "size": font, "color": "black"}
        elif type(font) is dict:
            return font
        else:
            raise ValueError("Invalid font size, select from 'small', 'medium', 'big', 'huge', int value or build your own dictionary.")
       


    from IPython.display import display, HTML
    import plotly
    plotly.offline.init_notebook_mode()
    display(HTML(
    '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js"></script>'
    ))

    def set_fig_size(self, fig: go.Figure, width: int = 800) -> go.Figure:
        """
        Auxiliary method to set figure size to almost a 4:3 ratio.
        By default, width=800 will result in height=600.
        """
        height = int(width * 3 / 4)
        fig.update_layout(width=width, height=height)
        return fig

    def plotly_band_diagram(self, mode: str = "te", title: str | None = None, fig: go.Figure | None = None, 
                            decimation_label_factor: int = 1, color: str = "red", 
                            title_font = "huge", legend_font = "medium", label_font = "huge", tick_font = "small"):
        
        title_font = SimulationViewer._font_config(title_font)
        legend_font = SimulationViewer._font_config(legend_font)
        label_font = SimulationViewer._font_config(label_font)
        tick_font = SimulationViewer._font_config(tick_font)

        # Load the data frame
        if mode not in self.simulation.bands_df:
            df = self.simulation.load_frequency_data(mode)
        else:
            df = self.simulation.bands_df[mode]

        bands = df.columns[5:]
        if fig is None:
            fig = go.Figure()
        
        # Prepare hover information if k1 and k2 are available
        if {"k1", "k2"}.issubset(df.columns):
            customdata = df[["k1", "k2"]].to_numpy()
            hovertemplate = " (%{customdata[0]:.2f}, %{customdata[1]:.2f}) : %{y}<extra></extra>"
        else:
            customdata = None
            hovertemplate = None

        for col in bands:
            if customdata is not None:
                fig.add_trace(go.Scatter(
                    x=df["k index"], 
                    y=df[col], 
                    mode='lines+markers', 
                    name=col, 
                    line=dict(color=color),
                    marker=dict(symbol='circle'),
                    customdata=customdata,
                    hovertemplate=hovertemplate
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=df["k index"], 
                    y=df[col], 
                    mode='lines+markers', 
                    name=col, 
                    line=dict(color=color),
                    marker=dict(symbol='circle')
                ))
            
        # Set the plot title: use the given title or fallback to simulation name.
        plot_title = title if title is not None else self.simulation.simulation_name

        # Configure x-axis tick labels using decimation factor and, if available, k1/k2 values
        if decimation_label_factor > 1:
            k_indices = df["k index"].tolist()
            tickvals = k_indices[::decimation_label_factor]
            if {"k1", "k2"}.issubset(df.columns):
                ticks = df.loc[::decimation_label_factor, ["k1", "k2"]]
                ticktext = [f"({row['k1']:.2f}, {row['k2']:.2f})" for _, row in ticks.iterrows()]
            else:
                ticktext = [f"k{i+1}" for i in range(len(tickvals))]
            xaxis_config = dict(tickmode="array", tickvals=tickvals, ticktext=ticktext, tickfont=tick_font)
        else:
            xaxis_config = dict(tickfont=tick_font)

        # Update the layout with the fonts for title, axis labels, tick labels, and legend
        fig.update_layout(
            title=dict(text=plot_title, font=title_font),
            xaxis=dict(
                title=dict(text="$\mathrm{(k_x,\; k_y)}$", font=label_font),
                **xaxis_config
            ),
            yaxis=dict(
                title=dict(text="frequency", font=label_font),
                tickfont=tick_font
            ),
            legend=dict(
                font=legend_font
            )
        )

        # Set the figure size to almost 4:3 using the auxiliary method.
        fig = self.set_fig_size(fig)

        return fig
