import os
import subprocess
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
import meep as mp
from meep import mpb
from mpb_configurator import MPBSchemeConfigurator
import plotly.graph_objects as go

class Simulation:
    def __init__(self, simulation_name: str, config: MPBSchemeConfigurator|None =None, directory: str | None = None, description: str | None = None):
        self.simulation_name = simulation_name
        self.config = config
        self.directory = directory if directory is not None else simulation_name
        self.scheme_filename = f"{simulation_name}.ctl"
        self.output_filename = f"{simulation_name}.out"
        self.error_filename = f"{simulation_name}.err"
        self.epsilon = None
        self.lattice = None
        self.bands_df = {}  # For frequency database storage

        if description is not None:
            desc_path = os.path.join(self.directory, f"{self.simulation_name}.txt")
            os.makedirs(self.directory, exist_ok=True)
            with open(desc_path, "w") as f:
                f.write(description)

    def run(self, print_config: bool = False, scheme_script: str | None = None, load_epsilon: bool = True, extract_frequencies: bool = True):
        """
        Run simulation by writing the scheme configuration (when config is provided)
        or using an existing scheme script.
        """
        os.makedirs(self.directory, exist_ok=True)
        if self.config is not None:
            scheme_path = os.path.join(self.directory, self.scheme_filename)
            scheme = self.config.generate_scheme_config(scheme_path)
            if print_config:
                print(scheme)
            with open(scheme_path, "w") as f:
                f.write(scheme)
            cmd_script = self.scheme_filename
        else:
            if scheme_script is None:
                raise ValueError("scheme_script must be provided if config is not set.")
            scheme_path = os.path.join(self.directory, os.path.basename(scheme_script))
            if not os.path.exists(scheme_path):
                with open(scheme_script, "r") as src, open(scheme_path, "w") as dst:
                    dst.write(src.read())
            cmd_script = scheme_script

        # Run the simulation within the simulation directory
        cmd = ["mpb", cmd_script]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.directory)

        # Save output and error files
        output_path = os.path.join(self.directory, self.output_filename)
        error_path = os.path.join(self.directory, self.error_filename)
        with open(output_path, "w") as f_out:
            f_out.write(result.stdout)
        with open(error_path, "w") as f_err:
            f_err.write(result.stderr)

        print("Simulation completed")

        if load_epsilon:
            self.load_epsilon()
        if extract_frequencies:
            self.extract_frequencies()


    def run_hpc(self, print_config: bool = False, scheme_script: str | None = None, load_epsilon: bool = True, extract_frequencies: bool = True, path_to_mpb: str = "mpb-mpi"):
        """
        Run simulation by writing the scheme configuration (when config is provided)
        or using an existing scheme script.
        """
        os.makedirs(self.directory, exist_ok=True)
        if self.config is not None:
            scheme_path = os.path.join(self.directory, self.scheme_filename)
            scheme = self.config.generate_scheme_config(scheme_path)
            if print_config:
                print(scheme)
            with open(scheme_path, "w") as f:
                f.write(scheme)
            cmd_script = self.scheme_filename
        else:
            if scheme_script is None:
                raise ValueError("scheme_script must be provided if config is not set.")
            scheme_path = os.path.join(self.directory, os.path.basename(scheme_script))
            if not os.path.exists(scheme_path):
                with open(scheme_script, "r") as src, open(scheme_path, "w") as dst:
                    dst.write(src.read())
            cmd_script = scheme_script

        # Run the simulation within the simulation directory using module load and mpb in one command
        cmd = f"source /dtu/sw/dcc/dcc-sw.bash && module load mpb/1.11.1 && {path_to_mpb} {cmd_script}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.directory)
        print(result.stdout)
        print(result.stderr)

        # Save output and error files
        output_path = os.path.join(self.directory, self.output_filename)
        error_path = os.path.join(self.directory, self.error_filename)
        with open(output_path, "w") as f_out:
            f_out.write(result.stdout)
        with open(error_path, "w") as f_err:
            f_err.write(result.stderr)

        print("Simulation completed")

        if load_epsilon:
            self.load_epsilon()
        if extract_frequencies:
            self.extract_frequencies()
        
        
        

    def extract_frequencies(self, remove_line_prefixes: bool = True):
        """
        Parse the output file, extract frequency data, and write them to separate files.
        """
        prefixes = ["tmfreqs:", "tefreqs:", "zevenfreqs:", "zoddfreqs:", "gaps:"]
        def strip_prefix(line: str, prefixes: list[str]) -> str:
            for prefix in prefixes:
                if line.startswith(prefix):
                    return line[len(prefix):].lstrip(" ,")
            return line

        output_path = os.path.join(self.directory, self.output_filename)
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output file {output_path} does not exist.")

        with open(output_path, "r") as f:
            lines = f.readlines()

        tm_lines = [strip_prefix(line, prefixes) if remove_line_prefixes else line
                    for line in lines if "tmfreqs:" in line]
        te_lines = [strip_prefix(line, prefixes) if remove_line_prefixes else line
                    for line in lines if "tefreqs:" in line]
        zeven_lines = [strip_prefix(line, prefixes) if remove_line_prefixes else line
                       for line in lines if "zevenfreqs:" in line]
        zodd_lines = [strip_prefix(line, prefixes) if remove_line_prefixes else line
                      for line in lines if "zoddfreqs:" in line]
        gaps_lines = [strip_prefix(line, prefixes) if remove_line_prefixes else line
                      for line in lines if "gaps:" in line]

        tm_file = os.path.join(self.directory, f"{self.simulation_name}.tm.dat")
        with open(tm_file, "w") as f_tm:
            f_tm.writelines(tm_lines)
        print(f"Extracted {len(tm_lines)} lines of data for the TM mode")

        te_file = os.path.join(self.directory, f"{self.simulation_name}.te.dat")
        with open(te_file, "w") as f_te:
            f_te.writelines(te_lines)
        print(f"Extracted {len(te_lines)} lines of data for the TE mode")
        
        zeven_file = os.path.join(self.directory, f"{self.simulation_name}.zeven.dat")
        with open(zeven_file, "w") as f_zeven:
            f_zeven.writelines(zeven_lines)
        print(f"Extracted {len(zeven_lines)} lines of data for the zeven frequencies")

        zodd_file = os.path.join(self.directory, f"{self.simulation_name}.zodd.dat")
        with open(zodd_file, "w") as f_zodd:
            f_zodd.writelines(zodd_lines)
        print(f"Extracted {len(zodd_lines)} lines of data for the zodd frequencies")

        gaps_file = os.path.join(self.directory, f"{self.simulation_name}.gaps.dat")
        with open(gaps_file, "w") as f_gaps:
            f_gaps.writelines(gaps_lines)
        print(f"Extracted {len(gaps_lines)} lines of data for the gaps")

    def load_epsilon(self):
        """
        Load epsilon and lattice vectors from an HDF5 file.
        Expects the file to be named `<simulation_name>-epsilon.h5` in the simulation directory.
        """
        filepath = os.path.join(self.directory, f"{self.simulation_name}-epsilon.h5")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"HDF5 file {filepath} not found.")
        with h5py.File(filepath, 'r') as f:
            self.epsilon = f['data'][...]
            if 'lattice vectors' in f:
                self.lattice = f['lattice vectors'][...]
        print("Loaded epsilon and lattice vectors")

    def convert_epsilon(self, periods: int | tuple = 1, use_2d: bool = True) -> np.ndarray:
        """
        Convert the epsilon data using MPB routines.
        For 2D, extract the middle slice of the 3D epsilon array.
        """
        if self.lattice is None:
            raise ValueError("Lattice is not defined. Call load_epsilon() first.")
        if self.epsilon is None:
            raise ValueError("Epsilon is not defined. Call load_epsilon() first.")
        mpb_data = mpb.MPBData(rectify=True, periods=periods, lattice=self.lattice)
        eps = self.epsilon
        if use_2d and self.epsilon.ndim == 3:
            mid_index = self.epsilon.shape[2] // 2
            eps = self.epsilon[:, :, mid_index]
        epsilon_converted = mpb_data.convert(eps)
        return epsilon_converted

    def load_frequency_data(self, mode: str = "te") -> pd.DataFrame:
        """
        Load frequency data from a CSV file and store it in a SQLite database.
        Expects a CSV file `<simulation_name>.<mode>.dat` in the simulation directory.
        """
        filepath = os.path.join(self.directory, f"{self.simulation_name}.{mode}.dat")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found.")
        df = pd.read_csv(filepath, skipinitialspace=True)
        db_path = os.path.join(self.directory, f"{self.simulation_name}_frequencies.db")
        conn = sqlite3.connect(db_path)
        df.to_sql("frequencies", conn, if_exists="replace", index=False)
        conn.close()
        self.bands_df[mode] = df
        print(f"Loaded frequency data for mode '{mode}'")
        return df


class SimulationViewer:
    def __init__(self, simulation: Simulation):
        self.simulation = simulation

    def _apply_title(self, default_title: str, title: str | None):
        if title is False:
            return
        elif title is not None:
            plt.title(title)
        else:
            plt.title(default_title)

    def plot_epsilon_2d(self, periods: int | tuple = 1, title: str | bool | None = None, converted: bool = True, cmap: str = 'viridis'):
        """
        Plot 2D epsilon data.
        """
        if self.simulation.epsilon is None:
            raise ValueError("Epsilon data not loaded. Call load_epsilon() on the simulation object.")
        if converted:
            eps = self.simulation.convert_epsilon(periods, use_2d=True)
        else:
            eps = self.simulation.epsilon
            if eps.ndim == 3:
                mid_index = eps.shape[2] // 2
                eps = eps[:, :, mid_index]
        plt.imshow(eps, interpolation='spline36', cmap=cmap)
        plt.colorbar()
        self._apply_title(self.simulation.simulation_name, title)
        
    
    def plot_epsilon_3d(self, periods: int | tuple = 1, title: str | bool | None = None, converted: bool = True, cmap: str = 'viridis'):
        print("3D plot not implemented yet")

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
                          colors: list[str] | str | None = None, decimation_label_factor: int = 1, grid: bool = True):
        """
        Plot the band diagram for the given mode.
        All bands will use the same color.
        The x-axis tick labels are formatted as (kₓ, kᵧ) in LaTeX if decimation_label_factor > 1,
        with decimation controlled by decimation_label_factor.
        The y-axis label is always frequency.
        The legend includes the polarization mode.
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

        first_line = True
        for col in bands:
            if first_line:
                plt.plot(df["k index"], df[col], label=f"{mode.upper()} bands", color=plot_color)
                first_line = False
            else:
                plt.plot(df["k index"], df[col], color=plot_color)

        # Set x-axis label
        if decimation_label_factor > 1:
            plt.xlabel(r"$(k_x, k_y)$")
        else:
            plt.xlabel("k index")
        plt.ylabel("Frequency")

        # Update x-axis tick labels to (kₓ, kᵧ) with decimation if available.
        if decimation_label_factor > 1 and {"k1", "k2"}.issubset(df.columns):
            tickvals = df["k index"].values[::decimation_label_factor]
            ticklabels = [
                f"({row['k1']:.2f}, {row['k2']:.2f})"
                for _, row in df.iloc[::decimation_label_factor].iterrows()
            ]
            plt.xticks(tickvals, ticklabels)
        else:
            plt.xticks(df["k index"])

        self._apply_title(self.simulation.simulation_name, title)
        plt.legend()
        plt.grid(grid)

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
