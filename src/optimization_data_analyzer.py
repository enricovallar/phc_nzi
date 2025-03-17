#!/usr/bin/env python3
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy.interpolate import griddata

try:
    from mpi_differential_evolution import MPIDiffEvoSimulation
except ImportError:
    from sources.mpi_differential_evolution import MPIDiffEvoSimulation

class OptimizationDataAnalyzer:
    # Precompile the regex for efficiency.
    PATTERN = re.compile(r"(\w[\w\d_]*)\s*:\s*([\d\.\+\-eE]+)")

    def __init__(self, data_file_path, use_inverse_cost=True):
        """
        Parameters:
            data_file_path : str
                Path to the log file generated during optimization.
            use_inverse_cost : bool, optional
                If True, use 1/cost (i.e. higher weight for lower cost) for fitting.
                Default is True.
        """
        self.data_file_path = data_file_path
        self.use_inverse_cost = use_inverse_cost
        # These properties will be populated by extract_all_data().
        self.param1_name = None
        self.param2_name = None
        self.param1_vals = None
        self.param2_vals = None
        self.cost_vals = None
        self.freq_dirac_vals = None

    def _parse_line(self, line):
        """
        Parse a single log file line into a dictionary of {parameter: value}.
        """
        matches = self.PATTERN.findall(line)
        param_dict = {}
        for name, val_str in matches:
            try:
                param_dict[name] = float(val_str)
            except ValueError:
                continue
        return param_dict

    def extract_all_data(self, debug=False):
        """
        Parse the log file once and extract all data.
        Expects each valid line to contain exactly two parameter keys (aside from 'cost' and 'freq_dirac').
        Sets the following properties:
          - self.param1_name, self.param2_name
          - self.param1_vals, self.param2_vals
          - self.cost_vals, self.freq_dirac_vals
        """
        if not os.path.isfile(self.data_file_path):
            raise ValueError(f"Log file not found: {self.data_file_path}")

        p1_list = []
        p2_list = []
        cost_list = []
        freq_list = []
        param1_name = None
        param2_name = None

        with open(self.data_file_path, "r") as f:
            for line in f:
                original_line = line.strip()
                if not original_line:
                    continue

                data = self._parse_line(original_line)
                # Remove 'cost' and 'freq_dirac' from the dictionary; if missing, use NaN.
                cost = data.pop("cost", np.nan)
                freq = data.pop("freq_dirac", np.nan)

                # Expect exactly two remaining parameters.
                if len(data) != 2:
                    if debug:
                        print(f"DEBUG: Skipping line (expected 2 parameters, got {len(data)}): {original_line}")
                    continue

                sorted_keys = sorted(data.keys())
                if param1_name is None and param2_name is None:
                    param1_name, param2_name = sorted_keys
                elif set(sorted_keys) != {param1_name, param2_name}:
                    if debug:
                        print(f"DEBUG: Skipping line (inconsistent parameter names): {original_line}")
                    continue

                p1_list.append(data[param1_name])
                p2_list.append(data[param2_name])
                cost_list.append(cost)
                freq_list.append(freq)

        if not p1_list:
            raise ValueError("No valid data points were found in the log file.")

        self.param1_name = param1_name
        self.param2_name = param2_name
        self.param1_vals = np.array(p1_list)
        self.param2_vals = np.array(p2_list)
        self.cost_vals = np.array(cost_list)
        self.freq_dirac_vals = np.array(freq_list)

    def load_data(self, debug=False):
        """
        Load data from the log file if not already loaded.
        """
        if self.param1_vals is None or self.param2_vals is None:
            self.extract_all_data(debug=debug)

    def _plot_data(self, x_vals, y_vals, values, value_label, custom_title,
                   use_logscale, levels, points_only, show_description=False):
        """
        Generic plotting routine that creates either a scatter plot or a heatmap.
        """
        if use_logscale:
            min_val = np.min(values)
            if min_val <= 0:
                print(f"Cannot use log scale because min {value_label} <= 0. Switching to linear scale.")
                use_logscale = False
        norm = LogNorm(vmin=np.min(values), vmax=np.max(values)) if use_logscale else None
        scale_title = " (Log Scale)" if use_logscale else " (Linear Scale)"
        title_suffix = f"{scale_title} ({value_label})"
        title = custom_title if custom_title is not None else f"Parameter Space {value_label}{title_suffix}"

        if points_only:
            plt.figure(figsize=(7, 6))
            scatter = plt.scatter(x_vals, y_vals, c=values, cmap="viridis", norm=norm)
            plt.colorbar(scatter, label=value_label)
            plt.xlabel(self.param1_name)
            plt.ylabel(self.param2_name)
            plt.title(title)
            plt.tight_layout()
            return

        try:
            import matplotlib.tri as mtri
            triang = mtri.Triangulation(x_vals, y_vals)
            plt.figure(figsize=(7, 6))
            if levels is None:
                pc = plt.tripcolor(triang, values, shading="gouraud", cmap="viridis", norm=norm)
                plt.colorbar(pc, label=value_label)
                plot_desc = "Tripcolor (Gouraud Shading)"
            else:
                cntr = plt.tricontourf(triang, values, levels=levels, cmap="viridis", norm=norm)
                plt.colorbar(cntr, label=value_label)
                plot_desc = f"Tricontourf (levels={levels})"
            plt.xlabel(self.param1_name)
            plt.ylabel(self.param2_name)
            if show_description:
                plt.title(f"{title}\n{plot_desc}")
            else:
                plt.title(title)
            plt.tight_layout()
        except Exception as e:
            print("Falling back to scatter plot due to:", e)
            plt.figure(figsize=(7, 6))
            scatter = plt.scatter(x_vals, y_vals, c=values, cmap="viridis", norm=norm)
            plt.colorbar(scatter, label=value_label)
            plt.xlabel(self.param1_name)
            plt.ylabel(self.param2_name)
            plt.title(title)
            plt.tight_layout()

    def weighted_linear_regression(self):
        """
        Perform weighted linear regression using param1_vals as x and param2_vals as y.
        The weights are computed from cost_vals.
        
        Returns:
            tuple: (m, b) where m is the slope and b is the intercept.
        """
        self.load_data()
        valid = ~np.isnan(self.cost_vals)
        x = self.param1_vals[valid]
        y = self.param2_vals[valid]

        if self.use_inverse_cost:
            # Avoid dividing by zero
            nonzero = self.cost_vals[valid] != 0
            x = x[nonzero]
            y = y[nonzero]
            weights = 1.0 / self.cost_vals[valid][nonzero]
        else:
            weights = self.cost_vals[valid]

        m, b = np.polyfit(x, y, 1, w=weights)
        self.fit_params = (m, b)
        return m, b

    def predict_second_parameter(self, first_param_value):
        """
        Given a value for the first parameter, predict the second parameter using the fitted model.
        """
        if not hasattr(self, "fit_params"):
            self.weighted_linear_regression()
        m, b = self.fit_params
        return m * first_param_value + b

    def plot_raw_data(self, use_logscale=False, levels=100, points_only=False,
                      plot_inverse_cost=False, custom_title=None):
        """
        Plot raw data using cost (or 1/cost if plot_inverse_cost is True) as the color.
        """
        self.load_data()
        if plot_inverse_cost:
            # Avoid division by zero.
            values = np.array([1.0 / c if c != 0 else np.nan for c in self.cost_vals])
            value_label = "1/Cost"
        else:
            values = self.cost_vals
            value_label = "Cost"

        self._plot_data(
            self.param1_vals,
            self.param2_vals,
            values,
            value_label,
            custom_title,
            use_logscale,
            levels,
            points_only
        )

    def plot_optimization_points_bandgap(self, use_logscale=False, levels=50,
                                         points_only=False, plot_inverse_cost=False,
                                         custom_title=None):
        """
        Plot optimization points using cost (or 1/cost if plot_inverse_cost is True) as the color.
        This method is provided for compatibility with bandgap analysis.
        """
        self.plot_raw_data(
            use_logscale=use_logscale,
            levels=levels,
            points_only=points_only,
            plot_inverse_cost=plot_inverse_cost,
            custom_title=custom_title
        )

    def plot_optimization_points_freq_dirac(self, use_logscale=False, levels=50,
                                            points_only=False, plot_inverse_freq=False,
                                            custom_title=None):
        """
        Plot optimization points using freq_dirac (or its inverse if plot_inverse_freq is True) as the color.
        """
        self.load_data()
        if plot_inverse_freq:
            values = np.array([1.0 / f if f != 0 else np.nan for f in self.freq_dirac_vals])
            value_label = "1/freq_dirac"
        else:
            values = self.freq_dirac_vals
            value_label = "freq_dirac"

        self._plot_data(
            self.param1_vals,
            self.param2_vals,
            values,
            value_label,
            custom_title,
            use_logscale,
            levels,
            points_only
        )

    def plot_fitted_line(self):
        """
        Plot the fitted weighted linear regression line on the current plot.
        """
        self.weighted_linear_regression()
        m, b = self.fit_params
        x_vals = np.array([np.min(self.param1_vals), np.max(self.param1_vals)])
        y_vals = m * x_vals + b
        plt.plot(x_vals, y_vals, 'r--', label="y = {:.3f}x + {:.3f}".format(m, b))
        plt.legend()

    # --------------------------------------------------------------------
    # NEW: TWO METHODS to handle computing vs. plotting freq_dirac
    #      along the fitted line param2 = m * param1 + b.
    # --------------------------------------------------------------------
    def compute_freq_dirac_along_fit(self, n_points=200, interp_method='linear'):
        """
        Compute freq_dirac along the line param2 = m * param1 + b by:
          1. Doing a weighted linear regression (to get m, b).
          2. Interpolating freq_dirac in 2D using griddata.
          3. Evaluating along r1_line and r2_line = m*r1_line + b.

        Parameters
        ----------
        n_points : int
            Number of points in param1 at which to sample freq_dirac along the line.
        interp_method : {'linear', 'nearest', 'cubic'}
            The interpolation method used by scipy.interpolate.griddata.

        Returns
        -------
        r1_line : np.ndarray
            The param1 values used for sampling.
        freq_line : np.ndarray
            Interpolated freq_dirac values corresponding to r1_line.
        m, b : float
            Slope and intercept from the weighted linear regression.
        """
        self.load_data()
        # 1) Weighted linear regression => param2 = m * param1 + b
        self.weighted_linear_regression()
        m, b = self.fit_params

        # 2) Filter out any NaNs in freq_dirac.
        valid = ~np.isnan(self.freq_dirac_vals)
        r1 = self.param1_vals[valid]
        r2 = self.param2_vals[valid]
        freq = self.freq_dirac_vals[valid]

        if len(r1) < 3:
            raise ValueError("Not enough valid data points to interpolate freq_dirac.")

        # 3) param1 range for sampling
        r1_line = np.linspace(r1.min(), r1.max(), n_points)
        r2_line = m * r1_line + b

        # 4) Interpolate freq_dirac in 2D with griddata
        points = np.column_stack((r1, r2))  # shape (N, 2)
        freq_line = griddata(points, freq, (r1_line, r2_line), method=interp_method)

        return r1_line, freq_line, m, b

    def plot_freq_dirac_vs_param1_along_fit(self, n_points=200, interp_method='linear',
                                            custom_title=None):
        """
        Plot freq_dirac vs. param1 by:
          1. Calling compute_freq_dirac_along_fit(...) to get the line data.
          2. Plotting freq_line vs. r1_line.

        Parameters
        ----------
        n_points : int
            Number of points in param1 at which to sample freq_dirac along the line.
        interp_method : {'linear', 'nearest', 'cubic'}
            The interpolation method used by scipy.interpolate.griddata.
        custom_title : str, optional
            If given, used as the plot title.
        """
        # 1) Compute the freq_dirac data along the fitted line
        r1_line, freq_line, m, b = self.compute_freq_dirac_along_fit(
            n_points=n_points,
            interp_method=interp_method
        )

        # 2) Plot freq_line vs. r1_line
        plt.figure(figsize=(7, 5))
        plt.plot(r1_line, freq_line, 'b-', label=f"{interp_method} interpolation")
        plt.xlabel(self.param1_name)
        plt.ylabel("freq_dirac")
        fitted_line_str = f"{self.param2_name} = {m:.3f} * {self.param1_name} + {b:.3f}"
        default_title = f"freq_dirac along fitted line: {fitted_line_str}"
        plt.title(custom_title if custom_title else default_title)
        plt.legend()
        plt.tight_layout()    
        plt.grid(True)

