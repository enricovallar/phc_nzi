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
        self.param_names = None
        self.params_vals = None
        
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
        Expects a consistent set of parameters per line (aside from 'Gen', 'cost' and 'freq_dirac').
        """
        if not os.path.isfile(self.data_file_path):
            raise ValueError(f"Log file not found: {self.data_file_path}")

        cost_list = []
        freq_list = []
        gen_list = [] # <-- Added
        param_names = None
        params_lists = {}

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

                # Expect at least one parameter
                if len(data) < 1:
                    if debug:
                        print(f"DEBUG: Skipping line (expected at least 1 parameter): {original_line}")
                    continue

                sorted_keys = tuple(sorted(data.keys()))
                if param_names is None:
                    param_names = sorted_keys
                    for k in param_names:
                        params_lists[k] = []
                elif set(sorted_keys) != set(param_names):
                    if debug:
                        print(f"DEBUG: Skipping line (inconsistent parameter names): {original_line}")
                    continue

                for k in param_names:
                    params_lists[k].append(data[k])
                cost_list.append(cost)
                freq_list.append(freq)
                gen_list.append(gen) # <-- Added

        if not param_names or not params_lists[param_names[0]]:
            raise ValueError("No valid data points were found in the log file.")

        self.param_names = list(param_names)
        self.params_vals = {k: np.array(v) for k, v in params_lists.items()}

        # Backward compatibility for methods assuming two parameters
        p1_candidates = [p for p in self.param_names if str(p).endswith('1')]
        if p1_candidates:
            self.param1_name = p1_candidates[0]
            self.param1_vals = self.params_vals[self.param1_name]
        elif len(self.param_names) >= 1:
            self.param1_name = self.param_names[0]
            self.param1_vals = self.params_vals[self.param1_name]

        p2_candidates = [p for p in self.param_names if str(p).endswith('2')]
        if p2_candidates:
            self.param2_name = p2_candidates[0]
            self.param2_vals = self.params_vals[self.param2_name]
        elif len(self.param_names) >= 2:
            self.param2_name = self.param_names[1]
            self.param2_vals = self.params_vals[self.param2_name]

        self.cost_vals = np.array(cost_list)
        self.freq_dirac_vals = np.array(freq_list)
        self.gen_vals = np.array(gen_list) # <-- Added

    def load_data(self, debug=False):
        """
        Load data from the log file if not already loaded.
        """
        if self.param_names is None:
            self.extract_all_data(debug=debug)

    def _plot_data(self, x_vals, y_vals, values, value_label, custom_title,
                   use_logscale, levels, points_only, show_description=False, plot_options=None, xlabel=None, ylabel=None):
        """
        Generic plotting routine that creates either a scatter plot or a heatmap.
        """
        if xlabel is None:
            xlabel = self.param1_name
        if ylabel is None:
            ylabel = self.param2_name

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
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
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
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if show_description:
                plt.title(f"{title}\n{plot_desc}")
            else:
                plt.title(title)
            plt.tight_layout()
        except Exception as e:
            print("Falling back to scatter plot due to:", e)
            scatter = plt.scatter(x_vals, y_vals, c=values, norm=norm, **(plot_options or {}))
            plt.colorbar(scatter, label=value_label)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.tight_layout()


    def plot_raw_data(self, use_logscale=False, levels=100, points_only=False,
                      plot_inverse_cost=False, custom_title=None, plot_options=None, param_x=None, param_y=None):
        """
        Plot raw data using cost (or 1/cost if plot_inverse_cost is True) as the color.
        """
        self.load_data()
        
        px_name = param_x if param_x else self.param1_name
        py_name = param_y if param_y else self.param2_name
        
        if plot_inverse_cost:
            # Avoid division by zero.
            values = np.array([1.0 / c if c != 0 else np.nan for c in self.cost_vals])
            value_label = "1/Cost"
        else:
            values = self.cost_vals
            value_label = "Cost"

        self._plot_data(
            self.params_vals[px_name] if self.params_vals else self.param1_vals, # using fallback if needed
            self.params_vals[py_name] if self.params_vals else self.param2_vals,
            values,
            value_label,
            custom_title,
            use_logscale,
            levels,
            points_only,
            plot_options=plot_options,
            xlabel=px_name,
            ylabel=py_name
        )

    def plot_optimization_points_bandgap(self, use_logscale=False, levels=50,
                                         points_only=False, plot_inverse_cost=False,
                                         custom_title=None, param_x=None, param_y=None):
        """
        Plot optimization points using cost (or 1/cost if plot_inverse_cost is True) as the color.
        This method is provided for compatibility with bandgap analysis.
        """
        self.plot_raw_data(
            use_logscale=use_logscale,
            levels=levels,
            points_only=points_only,
            plot_inverse_cost=plot_inverse_cost,
            custom_title=custom_title,
            param_x=param_x,
            param_y=param_y
        )

    def plot_optimization_points_freq_dirac(self, use_logscale=False, levels=50,
                                            points_only=False, plot_inverse_freq=False,
                                            custom_title=None, param_x=None, param_y=None):
        """
        Plot optimization points using freq_dirac (or its inverse if plot_inverse_freq is True) as the color.
        """
        self.load_data()
        
        px_name = param_x if param_x else self.param1_name
        py_name = param_y if param_y else self.param2_name
        
        if plot_inverse_freq:
            values = np.array([1.0 / f if f != 0 else np.nan for f in self.freq_dirac_vals])
            value_label = "1/freq_dirac"
        else:
            values = self.freq_dirac_vals
            value_label = "freq_dirac"

        self._plot_data(
            self.params_vals[px_name] if self.params_vals else self.param1_vals,
            self.params_vals[py_name] if self.params_vals else self.param2_vals,
            values,
            value_label,
            custom_title,
            use_logscale,
            levels,
            points_only,
            xlabel=px_name,
            ylabel=py_name
        )



    def get_points_above_treshold(self, threshold):
        """
        Get the points where 1/cost is above a certain threshold.
        """
        self.load_data()
        valid = ~np.isnan(self.cost_vals)
        valid &= (self.cost_vals != 0)  # Avoid division by zero
        valid &= (1.0 / self.cost_vals) > threshold
        
        data_dict = {'Gen': self.gen_vals[valid]}
        for p_name in self.param_names:
            data_dict[p_name] = self.params_vals[p_name][valid]
        data_dict['cost'] = self.cost_vals[valid]
        data_dict['freq-dirac'] = self.freq_dirac_vals[valid]
        
        return pd.DataFrame(data_dict)
    
    def get_generation(self, gen_number):
        """
        Get the points corresponding to a specific generation number.
        """
        self.load_data()
        valid = ~np.isnan(self.gen_vals)
        valid &= (self.gen_vals == gen_number)
        
        data_dict = {'Gen': self.gen_vals[valid]}
        for p_name in self.param_names:
            data_dict[p_name] = self.params_vals[p_name][valid]
        data_dict['cost'] = self.cost_vals[valid]
        data_dict['freq-dirac'] = self.freq_dirac_vals[valid]
        
        return pd.DataFrame(data_dict)

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

    def compute_freq_dirac_along_ellipse_from_conic(self, conic_params, n_points=200, interp_method='linear', param_x=None, param_y=None):
        """
        Compute freq_dirac along a predefined ellipse using provided conic coefficients.
        
        Parameters:
            conic_params : tuple or np.ndarray
                (A, B, C, D, E, F) representing the conic section.
            n_points : int
                Resolution of the ellipse sampling.
            interp_method : str
                'linear', 'cubic', or 'nearest'.
        """
        self.load_data()
        
        px_name = param_x if param_x else self.param1_name
        py_name = param_y if param_y else self.param2_name
        
        A, B, C, D, E, F = conic_params
        
        # 1. Geometry extraction (Reused from plot_ellipse_from_conic logic)
        denom = B**2 - 4*A*C
        if denom == 0:
            raise ValueError("Invalid ellipse parameters (denom==0).")
        
        x0 = (2 * C * D - B * E) / denom
        y0 = (2 * A * E - B * D) / denom
        theta = 0.5 * np.arctan2(B, A - C)
        
        # Semi-axis lengths
        num = 2 * (A * x0**2 + B * x0 * y0 + C * y0**2 - F)
        term = np.sqrt((A - C)**2 + B**2)
        a_e = np.sqrt(num / (A + C - term))
        b_e = np.sqrt(num / (A + C + term))
        if b_e > a_e:
            a_e, b_e = b_e, a_e
            
        # 2. Parameterize the ellipse
        t = np.linspace(0, 2*np.pi, n_points)
        ellipse_x = x0 + a_e * np.cos(t) * np.cos(theta) - b_e * np.sin(t) * np.sin(theta)
        ellipse_y = y0 + a_e * np.cos(t) * np.sin(theta) + b_e * np.sin(t) * np.cos(theta)
        
        # 3. Interpolate freq_dirac at these coordinates
        source_points = np.column_stack((self.params_vals[px_name], self.params_vals[py_name]))
        freq_ellipse = griddata(source_points, self.freq_dirac_vals, (ellipse_x, ellipse_y), method=interp_method)
        
        return ellipse_x, ellipse_y, freq_ellipse

    def compute_freq_dirac_along_ellipse(self, n_points=200, interp_method='linear', threshold=1000, param_x=None, param_y=None):
        """
        Modified to use the new 'from_conic' method after fitting.
        """
        self.load_data()
        px_name = param_x if param_x else self.param1_name
        py_name = param_y if param_y else self.param2_name

        # Perform the fit
        df = self.get_points_above_treshold(threshold)
        if len(df) < 5:
            raise ValueError("Not enough valid data points to fit an ellipse.")
            
        conic_params = self.fit_ellipse(
            df[px_name].values, 
            df[py_name].values, 
            w=1.0 / df['cost'].values
        )
        
        # Delegate to the new function
        ex, ey, ef = self.compute_freq_dirac_along_ellipse_from_conic(
            conic_params, n_points, interp_method, px_name, py_name
        )
        
        return ex, ey, ef, conic_params

    def plot_freq_dirac_along_ellipse(self, n_points=200, interp_method='linear',
                                            custom_title=None,
                                            threshold=1000,
                                            plt_kwds=None,
                                            param_x=None, param_y=None):
        """
        Plot freq_dirac vs. param1 along the fitted ellipse, coloring the line
        according to the value of param2.
        """
        px_name = param_x if param_x else self.param1_name
        py_name = param_y if param_y else self.param2_name

        # 1) Compute the freq_dirac data along the fitted ellipse
        ellipse_param1, ellipse_param2, freq_ellipse, _ = self.compute_freq_dirac_along_ellipse(
            n_points=n_points,
            interp_method=interp_method,
            threshold=threshold,
            param_x=param_x,
            param_y=param_y
        )

        colors = ellipse_param2  # Use ellipse_param2 directly as colors

        # 2) Plot freq_ellipse vs. ellipse_param1, coloring by ellipse_param2

        scatter = plt.scatter(ellipse_param1, freq_ellipse, c=colors, marker='o', s=10, **(plt_kwds or {}))

        # Add colorbar

        plt.xlabel(px_name)
        plt.ylabel("freq_dirac")
        default_title = f"freq_dirac along fitted ellipse"
        plt.title(custom_title if custom_title else default_title)
        plt.tight_layout()
        plt.grid(True)

    



    def compute_param2_from_param1_on_ellipse(self, param1_value, conic_params=None, branch='upper', threshold=1000, param_x=None, param_y=None):
        """
        Compute the corresponding param2 value on the fitted ellipse for a given param1_value.
        """
        px_name = param_x if param_x else self.param1_name
        py_name = param_y if param_y else self.param2_name

        # If conic_params is not provided, compute them using points above the threshold.
        if conic_params is None:
            df = self.get_points_above_treshold(threshold)
            param1_data = df[px_name].values
            param2_data = df[py_name].values
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


    def compute_gradient_along_ellipse(self, n_points=200, interp_method='linear', threshold=1000, param_x=None, param_y=None):
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
            n_points=n_points, interp_method=interp_method, threshold=threshold, param_x=param_x, param_y=param_y
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