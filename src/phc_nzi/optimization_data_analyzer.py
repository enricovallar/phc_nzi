#!/usr/bin/env python3
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
import plotly.graph_objects as go

from scipy.interpolate import griddata
import pandas as pd

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
        self.gen_vals = None

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
        Expects each valid line to contain exactly two parameter keys (aside from 'Gen', 'cost' and 'freq_dirac').
        """
        if not os.path.isfile(self.data_file_path):
            raise ValueError(f"Log file not found: {self.data_file_path}")

        p1_list = []
        p2_list = []
        cost_list = []
        freq_list = []
        gen_list = [] # <-- Added
        param1_name = None
        param2_name = None

        with open(self.data_file_path, "r") as f:
            for line in f:
                original_line = line.strip()
                if not original_line:
                    continue

                data = self._parse_line(original_line)
                
                # Remove known non-parameter keys; if missing, use NaN.
                cost = data.pop("cost", np.nan)
                freq = data.pop("freq_dirac", np.nan)
                # Pop 'Gen' (or lowercase 'gen' as a fallback)
                gen = data.pop("Gen", data.pop("gen", np.nan)) # <-- Added

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
                gen_list.append(gen) # <-- Added

        if not p1_list:
            raise ValueError("No valid data points were found in the log file.")

        self.param1_name = param1_name
        self.param2_name = param2_name
        self.param1_vals = np.array(p1_list)
        self.param2_vals = np.array(p2_list)
        self.cost_vals = np.array(cost_list)
        self.freq_dirac_vals = np.array(freq_list)
        self.gen_vals = np.array(gen_list) # <-- Added

    def load_data(self, debug=False):
        """
        Load data from the log file if not already loaded.
        """
        if self.param1_vals is None or self.param2_vals is None:
            self.extract_all_data(debug=debug)

    def _plot_data(self, x_vals, y_vals, values, value_label, custom_title,
                   use_logscale, levels, points_only, show_description=False, plot_options=None):
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
            scatter = plt.scatter(x_vals, y_vals, c=values, norm=norm, **(plot_options or {}))
            plt.colorbar(scatter, label=value_label)
            plt.xlabel(self.param1_name)
            plt.ylabel(self.param2_name)
            plt.title(title)
            plt.tight_layout()
            return

        try:
            import matplotlib.tri as mtri
            triang = mtri.Triangulation(x_vals, y_vals)
            if levels is None:
                pc = plt.tripcolor(triang, values, shading="gouraud", norm=norm, **(plot_options or {}))
                plt.colorbar(pc, label=value_label)
                plot_desc = "Tripcolor (Gouraud Shading)"
            else:
                cntr = plt.tricontourf(triang, values, levels=levels, norm=norm, **(plot_options or {}))
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
            scatter = plt.scatter(x_vals, y_vals, c=values, norm=norm, **(plot_options or {}))
            plt.colorbar(scatter, label=value_label)
            plt.xlabel(self.param1_name)
            plt.ylabel(self.param2_name)
            plt.title(title)
            plt.tight_layout()


    def plot_raw_data(self, use_logscale=False, levels=100, points_only=False,
                      plot_inverse_cost=False, custom_title=None, plot_options=None):
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
            points_only,
            plot_options=plot_options
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



    def get_points_above_treshold(self, threshold):
        """
        Get the points where 1/cost is above a certain threshold.
        """
        self.load_data()
        valid = ~np.isnan(self.cost_vals)
        valid &= (self.cost_vals != 0)  # Avoid division by zero
        valid &= (1.0 / self.cost_vals) > threshold
        
        param1_vals = self.param1_vals[valid]
        param2_vals = self.param2_vals[valid]
        cost_vals = self.cost_vals[valid]
        freq_dirac_vals = self.freq_dirac_vals[valid]
        gen_vals = self.gen_vals[valid] # <-- Added
        
        df = pd.DataFrame({
            'Gen': gen_vals, # <-- Added
            self.param1_name: param1_vals,
            self.param2_name: param2_vals,
            'cost': cost_vals, 
            'freq-dirac': freq_dirac_vals, 
        })
        
        return df
    
    def get_generation(self, gen_number):
        """
        Get the points corresponding to a specific generation number.
        """
        self.load_data()
        valid = ~np.isnan(self.gen_vals)
        valid &= (self.gen_vals == gen_number)
        
        param1_vals = self.param1_vals[valid]
        param2_vals = self.param2_vals[valid]
        cost_vals = self.cost_vals[valid]
        freq_dirac_vals = self.freq_dirac_vals[valid]
        gen_vals = self.gen_vals[valid]
        df = pd.DataFrame({
            'Gen': gen_vals,
            self.param1_name: param1_vals,
            self.param2_name: param2_vals,
            'cost': cost_vals, 
            'freq-dirac': freq_dirac_vals, 
        })
        return df

    def fit_ellipse(self, x, y, w=None):
        """
        Fit an ellipse to the given (x, y) points using a weighted direct least squares method.
        
        Parameters:
            x, y : 1D numpy arrays of the same length containing the coordinates of the points.
            w    : Optional 1D numpy array of weights for each point. If None, equal weights are used.
        
        Returns:
            a : 1D numpy array of ellipse conic coefficients (A, B, C, D, E, F) for the general conic:
                A*x**2 + B*x*y + C*y**2 + D*x + E*y + F = 0
                with the constraint that the fitted conic is an ellipse (B**2 - 4*A*C < 0).
                
        Raises:
            RuntimeError: If no valid ellipse (i.e. satisfying B**2 - 4*A*C < 0) is found.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        n = len(x)
        if w is None:
            w = np.ones(n)
        else:
            w = np.asarray(w)
        
        # Build the design matrix D with columns: x^2, x*y, y^2, x, y, 1.
        D = np.column_stack((x**2, x*y, y**2, x, y, np.ones(n)))
        
        # Build the weight matrix W (diagonal).
        W = np.diag(w)
        
        # Compute the weighted scatter matrix.
        S = D.T @ W @ D
        
        # Constraint matrix C for the ellipse condition: 4*A*C - B^2 > 0
        # Here we use the formulation as in Fitzgibbon et al. (1999).
        C_matrix = np.zeros((6, 6))
        C_matrix[0, 2] = 2
        C_matrix[2, 0] = 2
        C_matrix[1, 1] = -1
        
        # Solve the generalized eigenvalue problem S*a = lambda*C*a.
        # We solve it by converting to a standard eigenvalue problem:
        #    inv(S) * C * a = lambda * a
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S) @ C_matrix)
        
        # Find candidate eigenvectors that satisfy the ellipse condition: B^2 - 4*A*C < 0.
        valid_indices = []
        for i, vec in enumerate(eig_vecs.T):
            A_, B_, C_, D_, E_, F_ = vec
            if B_**2 - 4*A_*C_ < 0:
                valid_indices.append(i)
        
        if not valid_indices:
            raise RuntimeError("No valid ellipse found (B^2 - 4*A*C >= 0 for all solutions).")
        
        # Choose the candidate corresponding to the eigenvalue with the maximum absolute value.
        # (You may choose a different criterion if desired.)
        i_best = valid_indices[np.argmax(np.abs(eig_vals[valid_indices]))]
        a = eig_vecs[:, i_best].real
        
        # Optionally, normalize the parameters (for example, so that F = -1)
        if a[-1] != 0:
            a = -a / a[-1]
        
        return a




    def plot_ellipse_from_conic(self, A, B, C, D, E, F, ax=None, plot_kwds=None):
        """
        Plot an ellipse defined by the conic coefficients:
        
            A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0
            
        Assumes that the conic represents an ellipse (i.e. B^2 - 4*A*C < 0).

        Parameters:
        A, B, C, D, E, F : float
            Conic coefficients.
        ax : matplotlib.axes.Axes, optional
            An existing axes to plot on. If None, uses current axes.
        plot_kwds : dict, optional
            Additional keyword arguments to pass to the plot function.

        Returns:
        ax : matplotlib.axes.Axes
            The axes with the ellipse plotted.
        """
        if ax is None:
            ax = plt.gca()
        if plot_kwds is None:
            plot_kwds = {'color': 'red', 'linewidth': 2}
        
        # Compute the center of the ellipse
        denom = B**2 - 4*A*C
        if denom == 0:
            raise ValueError("Invalid ellipse parameters (denom==0).")
        x0 = (2*C*D - B*E) / denom
        y0 = (2*A*E - B*D) / denom
        
        # Compute the rotation angle (in radians)
        theta = 0.5 * np.arctan2(B, A - C)
        
        # Compute the axes lengths.
        # Plug the center (x0, y0) back into the conic equation:
        num = 2 * (A*x0**2 + B*x0*y0 + C*y0**2 - F)
        term = np.sqrt((A - C)**2 + B**2)
        
        # Semi-axis lengths (order them so that a_e is the semi-major axis)
        a_e = np.sqrt(num / (A + C - term))
        b_e = np.sqrt(num / (A + C + term))
        if b_e > a_e:
            a_e, b_e = b_e, a_e

        # Generate points along the ellipse using the parametric form:
        #   X(t) = x0 + a_e*cos(t)*cos(theta) - b_e*sin(t)*sin(theta)
        #   Y(t) = y0 + a_e*cos(t)*sin(theta) + b_e*sin(t)*cos(theta)
        t = np.linspace(0, 2*np.pi, 500)
        ellipse_x = x0 + a_e * np.cos(t) * np.cos(theta) - b_e * np.sin(t) * np.sin(theta)
        ellipse_y = y0 + a_e * np.cos(t) * np.sin(theta) + b_e * np.sin(t) * np.cos(theta)
        
        ax.plot(ellipse_x, ellipse_y, **plot_kwds)
        ax.set_aspect('equal', 'datalim')
        ax.set_xlabel(self.param1_name)
        ax.set_ylabel(self.param2_name)
        ax.set_title('Fitted Ellipse')
        
        return ax

    def compute_freq_dirac_along_ellipse(self, n_points=200, interp_method='linear', threshold=1000):
        """
        Compute freq_dirac along the fitted ellipse by:
          1. Using the full dataset (param1_vals, param2_vals, freq_dirac_vals) to
             fit an ellipse (via self.fit_ellipse).
          2. Parameterizing the ellipse (using 0 <= t < 2π) to obtain (param1, param2)
             points along its boundary.
          3. Interpolating freq_dirac at these ellipse points using griddata.
        
        Parameters:
            n_points : int
                Number of points to sample along the ellipse.
            interp_method : str, one of {'linear', 'nearest', 'cubic'}
                Interpolation method for griddata.
            threshold : float
                Threshold for 1/cost to filter out points.
        
        Returns:
            ellipse_param1 : np.ndarray
                Array of param1 coordinates along the ellipse.
            ellipse_param2 : np.ndarray
                Array of param2 coordinates along the ellipse.
            freq_ellipse : np.ndarray
                Interpolated freq_dirac values at the ellipse points.
            conic_params : np.ndarray
                The fitted ellipse conic coefficients (A, B, C, D, E, F).
        """

        self.load_data()
        freq_all = self.freq_dirac_vals
        param1_all = self.param1_vals
        param2_all = self.param2_vals
        cost_all = self.cost_vals

        df = self.get_points_above_treshold(threshold)
        param1_above_threshold = df[self.param1_name].values
        param2_above_threshold = df[self.param2_name].values
        freq_above_threshold= df['freq-dirac'].values
        weights_above_threshold = 1.0 / df['cost'].values
        
        if len(param1_above_threshold) < 5:
            raise ValueError("Not enough valid data points to fit an ellipse.")
        
        # Fit an ellipse to (param1, param2) using your existing method.
        conic_params = self.fit_ellipse(param1_above_threshold, param2_above_threshold, w=weights_above_threshold)
        A, B, C, D, E, F = conic_params
        
        # Compute the ellipse center.
        denom = B**2 - 4*A*C
        if denom == 0:
            raise ValueError("Invalid ellipse parameters (denom==0).")
        center1 = (2 * C * D - B * E) / denom
        center2 = (2 * A * E - B * D) / denom
        
        # Compute rotation angle.
        theta = 0.5 * np.arctan2(B, A - C)
        
        # Compute semi-axis lengths.
        num = 2 * (A * center1**2 + B * center1 * center2 + C * center2**2 - F)
        term = np.sqrt((A - C)**2 + B**2)
        a_e = np.sqrt(num / (A + C - term))
        b_e = np.sqrt(num / (A + C + term))
        if b_e > a_e:
            a_e, b_e = b_e, a_e
        
        # Parameterize the ellipse using parameter t.
        t = np.linspace(0, 2*np.pi, n_points)
        ellipse_param1 = center1 + a_e * np.cos(t) * np.cos(theta) - b_e * np.sin(t) * np.sin(theta)
        ellipse_param2 = center2 + a_e * np.cos(t) * np.sin(theta) + b_e * np.sin(t) * np.cos(theta)
        
        # Interpolate freq_dirac at the ellipse (param1, param2) points.
        points = np.column_stack((param1_all, param2_all))
        freq_ellipse = griddata(points, freq_all, (ellipse_param1, ellipse_param2), method=interp_method)
        
        return ellipse_param1, ellipse_param2, freq_ellipse, conic_params

    def plot_freq_dirac_along_ellipse(self, n_points=200, interp_method='linear',
                                            custom_title=None,
                                            threshold=1000,
                                            plt_kwds={},
                                            ):
        """
        Plot freq_dirac vs. param1 along the fitted ellipse, coloring the line
        according to the value of param2.

        Parameters
        ----------
        n_points : int
            Number of points in param1 at which to sample freq_dirac along the ellipse.
        interp_method : {'linear', 'nearest', 'cubic'}
            The interpolation method used by scipy.interpolate.griddata.
        custom_title : str, optional
            If given, used as the plot title.
        """
        # 1) Compute the freq_dirac data along the fitted ellipse
        ellipse_param1, ellipse_param2, freq_ellipse, _ = self.compute_freq_dirac_along_ellipse(
            n_points=n_points,
            interp_method=interp_method,
            threshold=threshold
        )

        colors = ellipse_param2  # Use ellipse_param2 directly as colors

        # 2) Plot freq_ellipse vs. ellipse_param1, coloring by ellipse_param2

        scatter = plt.scatter(ellipse_param1, freq_ellipse, c=colors, marker='o', s=10, **(plt_kwds or {}))

        # Add colorbar

        plt.xlabel(self.param1_name)
        plt.ylabel("freq_dirac")
        default_title = f"freq_dirac along fitted ellipse"
        plt.title(custom_title if custom_title else default_title)
        plt.tight_layout()
        plt.grid(True)

    



    def compute_param2_from_param1_on_ellipse(self, param1_value, conic_params=None, branch='upper', threshold=1000):
        """
        Compute the corresponding param2 value on the fitted ellipse for a given param1_value.
        
        The ellipse is defined by the conic equation:
            A*param1^2 + B*param1*param2 + C*param2^2 + D*param1 + E*param2 + F = 0.
        If conic_params is None, the ellipse is fitted using the data points for which 1/cost > threshold.
        
        Parameters:
            param1_value : float
                The value of param1 for which to compute param2 on the ellipse.
            conic_params : array-like of shape (6,), optional
                The conic coefficients (A, B, C, D, E, F) defining the ellipse.
            branch : str, optional
                Which branch of the quadratic solution to return ('upper' for larger param2, 'lower' for smaller).
            threshold : float, optional
                Threshold for 1/cost to filter data points when computing the ellipse.
        
        Returns:
            param2_value : float
                The computed param2 value corresponding to the given param1_value on the ellipse.
        
        Raises:
            ValueError: If there are not enough valid data points to fit an ellipse,
                        or if no real solution exists for the given param1_value.
        """
        # If conic_params is not provided, compute them using points above the threshold.
        if conic_params is None:
            df = self.get_points_above_treshold(threshold)
            param1_data = df[self.param1_name].values
            param2_data = df[self.param2_name].values
            weights = 1.0 / df['cost'].values
            if len(param1_data) < 5:
                raise ValueError("Not enough valid data points to fit an ellipse using the threshold.")
            conic_params = self.fit_ellipse(param1_data, param2_data, w=weights)
        
        A, B, C, D, E, F = conic_params
        
        # For the given param1_value, the ellipse conic becomes a quadratic in param2:
        #   C*param2^2 + (B*param1_value + E)*param2 + (A*param1_value**2 + D*param1_value + F) = 0.
        a_coef = C
        b_coef = B * param1_value + E
        c_coef = A * param1_value**2 + D * param1_value + F
        
        tol = 1e-12
        # Solve the quadratic equation
        if np.abs(a_coef) > tol:
            discriminant = b_coef**2 - 4 * a_coef * c_coef
            if discriminant < 0:
                raise ValueError("No real solution exists for the given param1_value on the ellipse.")
            sqrt_disc = np.sqrt(discriminant)
            sol1 = (-b_coef + sqrt_disc) / (2 * a_coef)
            sol2 = (-b_coef - sqrt_disc) / (2 * a_coef)
            # Choose the branch as specified: 'upper' returns the larger solution.
            param2_value = max(sol1, sol2) if branch == 'upper' else min(sol1, sol2)
        else:
            # Degenerate to a linear equation: b_coef * param2 + c_coef = 0.
            if np.abs(b_coef) < tol:
                raise ValueError("Degenerate equation; cannot solve for param2.")
            param2_value = -c_coef / b_coef
        
        return param2_value


    def compute_gradient_along_ellipse(self, n_points=200, interp_method='linear', threshold=1000):
        """
        Compute the gradient of freq_dirac along the fitted ellipse by calculating the derivative 
        with respect to the arc length of the ellipse.

        Parameters:
            n_points : int
                Number of points to sample along the ellipse.
            interp_method : str, one of {'linear', 'nearest', 'cubic'}
                Interpolation method for griddata.
            threshold : float
                Threshold for 1/cost to filter out points.

        Returns:
            ellipse_param1 : np.ndarray
                Array of param1 coordinates along the ellipse.
            ellipse_param2 : np.ndarray
                Array of param2 coordinates along the ellipse.
            gradient : np.ndarray
                The gradient of freq_dirac along the ellipse (d(freq_dirac)/ds), computed as the derivative 
                of freq_dirac with respect to the arc length.
            conic_params : np.ndarray
                The fitted ellipse conic coefficients (A, B, C, D, E, F).
        """
        # Get ellipse points and freq_dirac values along the ellipse.
        ellipse_param1, ellipse_param2, freq_ellipse, conic_params = self.compute_freq_dirac_along_ellipse(
            n_points=n_points, interp_method=interp_method, threshold=threshold
        )
        # Compute finite differences of the ellipse coordinates (i.e. dx/dt and dy/dt).
        d_param1 = np.gradient(ellipse_param1)
        d_param2 = np.gradient(ellipse_param2)
        # Compute the differential arc length along the curve.
        ds = np.sqrt(d_param1**2 + d_param2**2)
        # Compute the derivative of freq_dirac with respect to the parameter.
        d_freq = np.gradient(freq_ellipse)
        # Calculate the gradient (change in frequency per unit arc length).
        gradient = d_freq / ds
        return ellipse_param1, ellipse_param2, gradient, conic_params

    def plot_gradient_along_ellipse(self, n_points=200, interp_method='linear', custom_title=None, threshold=1000, abs = False,  cmap="inferno_r",plt_kwds=None):
        """
        Plot the gradient of freq_dirac along the fitted ellipse in the (param1, param2) space.

        Parameters:
            n_points : int
                Number of points to sample along the ellipse.
            interp_method : str, one of {'linear', 'nearest', 'cubic'}
                Interpolation method for griddata.
            custom_title : str, optional
                Custom title for the plot.
            threshold : float
                Threshold for 1/cost to filter out points.
            plt_kwds : dict, optional
                Additional keyword arguments to pass to the scatter plot.

        This method computes the gradient using compute_gradient_along_ellipse and then produces a scatter 
        plot of the ellipse points colored by the computed gradient.
        """
        ellipse_param1, ellipse_param2, gradient, _ = self.compute_gradient_along_ellipse(
            n_points=n_points, interp_method=interp_method, threshold=threshold
        )
        if plt_kwds is None:
            plt_kwds = {}
        scatter = plt.scatter(ellipse_param1, ellipse_param2, c=gradient if not abs else np.abs(gradient),
                              cmap=cmap, marker='o', s=10, **plt_kwds)
        c = plt.colorbar(scatter, label="Gradient of freq_dirac (d(freq_dirac)/ds)")
        plt.xlabel(self.param1_name)
        plt.ylabel(self.param2_name)
        title = custom_title if custom_title is not None else "Gradient of freq_dirac along fitted ellipse"
        plt.title(title)
        plt.tight_layout()
        return c