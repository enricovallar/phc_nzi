from typing import Optional, Union
from matplotlib import patches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from phc_nzi.simulation_handler import Simulation, MPBDataConverter, MPBDataOptions
from matplotlib.colorbar import Colorbar
import functools
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from matplotlib.legend_handler import HandlerPatch
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# --- Arrow colors ---
COLOR_RE =  "#83817E"  
COLOR_IM = "#FFD000"        
EDGE_RE  =  "#000000"  
EDGE_IM  = "#000000"  

class ArrowHandler(HandlerPatch):
    """Custom legend handler that draws an arrow instead of a rectangle."""
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        arrow = FancyArrow(
            xdescent, ydescent + height / 2,
            width * 0.9, 0,
            width=height * 0.35,
            head_width=height * 0.9,
            head_length=width * 0.25,
            fc=orig_handle.get_facecolor(),
            ec=orig_handle.get_edgecolor(),
            linewidth=0.8,
            transform=trans
        )
        return [arrow]


def add_arrow_legend(fig_or_ax,  field, is_figure=False, **kwargs):
    """Add arrow-style legend for Re/Im E_parallel."""
    # Instantiating default arrows inside the function to avoid Signature SyntaxErrors
    arrow_re = FancyArrow(0, 0.5, 0.5, 0, width=0.08, head_width=0.25, head_length=0.12, fc=COLOR_RE, ec=EDGE_RE)
    arrow_im = FancyArrow(0, 0.5, 0.5, 0, width=0.08, head_width=0.25, head_length=0.12, fc=COLOR_IM, ec=EDGE_IM)
    
    target = fig_or_ax.legend if is_figure else fig_or_ax.legend
    return target(
        [arrow_re, arrow_im],
        [rf'$\mathrm{{Re}}({field}_{{\parallel}})$', rf'$\mathrm{{Im}}({field}_{{\parallel}})$'],
        handler_map={FancyArrow: ArrowHandler()},
        ncol=2, 
        framealpha=0.95, facecolor='white', edgecolor='black',
        **kwargs
    )


def _get_hexagonal_ws_patch(nx, ny, lw = 3):
    """Helper function to calculate the Wigner-Seitz cell for a hexagonal lattice."""
    cx, cy = nx / 2.0, ny / 2.0
    a_pix = min(nx, ny) / 3.0 
    R = a_pix / np.sqrt(3)
    
    # 30-degree phase shift to align vertices correctly
    angles = np.deg2rad([30, 90, 150, 210, 270, 330, 390])
    hex_x = cx + R * np.cos(angles)
    hex_y = cy + R * np.sin(angles)
    
    ws_path = Path(np.column_stack([hex_x, hex_y]))
    ws_patch = PathPatch(ws_path, facecolor='none', edgecolor='black', lw=lw, zorder=10)
    
    pad = R * 0.02 
    xlims = (np.min(hex_x) - pad, np.max(hex_x) + pad)
    ylims = (np.min(hex_y) - pad, np.max(hex_y) + pad)
    
    return ws_patch, xlims, ylims


def plot_field_quiver_on_ax(field_z, field_x, field_y, eps_data=None, 
                            lattice_type="square", ax=None, step=2, global_vmax=None, ws_lw=3):
    """
    Plot field_z background with field_xy quiver arrows from pre-loaded numpy arrays.
    Supports both 'square' and 'hexagonal' lattice types.
    """
    if ax is None: 
        ax = plt.gca()
    
    # MPB arrays are (nx, ny). Extract dimensions.
    nx, ny = field_z.shape
    field_z_real = np.real(field_z)
    
    # Enforce global colorscale if provided
    vmax_h = global_vmax if global_vmax is not None else np.max(np.abs(field_z_real))
    
    # --- [ 1. Handle Lattice Types & Clipping ] ---
    clip_patch = None
    if lattice_type.lower() == "hexagonal":
        clip_patch, xlims, ylims = _get_hexagonal_ws_patch(nx, ny, ws_lw)
        ax.add_patch(clip_patch)
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
    else:
        # Standard Square Bounds
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)

    # --- [ 2. Plot Background Field ] ---
    # Transpose (.T) required to map (nx, ny) arrays to standard X/Y plots
    im = ax.imshow(field_z_real.T, interpolation='spline36', cmap="RdBu_r",
                   vmin=-vmax_h, vmax=vmax_h, origin='lower')
    if clip_patch:
        im.set_clip_path(clip_patch)
    
    if field_x is not None and field_y is not None:
        # --- [ 3. Plot Quivers ] ---
        x = np.arange(nx)
        y = np.arange(ny)
        
        # Use 'ij' indexing to ensure grid shapes match field shapes exactly
        X, Y = np.meshgrid(x, y, indexing='ij')
        X_sub = X[::step, ::step]
        Y_sub = Y[::step, ::step]

        field_x_r = np.real(field_x)[::step, ::step]
        field_y_r = np.real(field_y)[::step, ::step]
        field_x_i = np.imag(field_x)[::step, ::step]
        field_y_i = np.imag(field_y)[::step, ::step]

        mag_real = np.sqrt(field_x_r**2 + field_y_r**2)
        mag_imag = np.sqrt(field_x_i**2 + field_y_i**2)
        max_mag = max(mag_real.max(), mag_imag.max())
        
        # Avoid zero division scale errors
        arrow_scale = max_mag * 15 if max_mag > 0 else 1 

        q_real = ax.quiver(X_sub, Y_sub, field_x_r, field_y_r,
                        color=COLOR_RE, edgecolor=EDGE_RE, linewidth=0.15,
                        scale=arrow_scale, width=0.004, headwidth=4, headlength=4)
        if clip_patch: q_real.set_clip_path(clip_patch)

        offset = 0.3
        q_imag = ax.quiver(X_sub + offset, Y_sub + offset, field_x_i, field_y_i,
                        color=COLOR_IM, edgecolor=EDGE_IM, linewidth=0.15,
                        scale=arrow_scale, width=0.004, headwidth=4, headlength=4)
        if clip_patch: q_imag.set_clip_path(clip_patch)

    # --- [ 4. Plot Epsilon Contours ] ---
    if eps_data is not None:
        try:
            # Must use .T to map over the transposed imshow
            cont = ax.contour(eps_data.T, levels=[eps_data.max() * 0.5],
                              colors='gray', linewidths=1.5, linestyles='--',
                              alpha=0.7, origin='lower')
            
            # Apply clipping dynamically depending on Matplotlib version
            if clip_patch:
                if hasattr(cont, 'collections') and cont.collections:
                    for coll in cont.collections:
                        coll.set_clip_path(clip_patch)
                else:
                    cont.set_clip_path(clip_patch)
        except Exception as e:
            print(f"Warning: Failed to plot epsilon contours: {e}")

    # --- [ 5. Formatting ] ---
    ax.set_xlabel("x (grid pts)")
    ax.set_ylabel("y (grid pts)")
    ax.set_aspect('equal')
    
    return im

def plot_epsilon_on_ax(eps_data, lattice_type="square", ax=None, ws_lw = 3,  cmap = "managua_r", **kwargs):
    """
    Plot the epsilon distribution background from pre-loaded numpy arrays.
    Supports both 'square' and 'hexagonal' lattice types using Wigner-Seitz clipping.
    """
    if ax is None: 
        ax = plt.gca()
    
    # Extract dimensions from the epsilon array
    nx, ny = eps_data.shape
    
    # --- [ 1. Handle Lattice Types & Clipping ] ---
    clip_patch = None
    if lattice_type.lower() == "hexagonal":
        # Uses your exact helper function logic for Wigner-Seitz cells
        clip_patch, xlims, ylims = _get_hexagonal_ws_patch(nx, ny, ws_lw)
        ax.add_patch(clip_patch)
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
    else:
        clip_patch = patches.Rectangle((0, 0), nx, ny, 
                                       fill=False, 
                                       edgecolor='black', 
                                       linewidth=ws_lw,
                                       joinstyle='miter')
        ax.add_patch(clip_patch)
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)

    # --- [ 2. Plot Background Epsilon ] ---
    im = ax.imshow(eps_data.T, cmap=cmap, origin='lower', **kwargs)
    
    if clip_patch:
        im.set_clip_path(clip_patch)

    # --- [ 3. Formatting ] ---
    ax.set_xlabel("x (grid pts)")
    ax.set_ylabel("y (grid pts)")
    ax.set_aspect('equal')
    
    return im

class SimulationViewer:

    SQUARE_ASPECT_RATIO = (1,1,1)
    DEFAULT_TITLE_FONTSIZE = 12
    DEFAULT_ELEMENT_FONTSIZE = 11
    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation

    def figure(self) -> None:
        """Create a new figure (clearing any existing figure)."""
        plt.figure()

    def show(self) -> None:
        """Display the current figure."""
        plt.show()

    @staticmethod
    def _change_fontsizes(title_fontsize, element_fontsize, fig=None):
        
        if fig is None:
            fig = plt.gcf()
        
        # Change the overall figure title (suptitle) if it exists.
        if hasattr(fig, '_suptitle') and fig._suptitle is not None:
            fig._suptitle.set_fontsize(title_fontsize)
        
        # Iterate over all axes in the figure.
        for ax in fig.get_axes():
            # Set the axes title font size to title_fontsize.
            ax.title.set_fontsize(title_fontsize)
            
            # Set x and y axis labels font sizes to element_fontsize.
            ax.xaxis.label.set_fontsize(element_fontsize)
            ax.yaxis.label.set_fontsize(element_fontsize)
            
            
            # Set tick labels font sizes to element_fontsize.
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(element_fontsize)
            
            # Update legend text font size to element_fontsize, if a legend exists.
            legend = ax.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_fontsize(element_fontsize)
            

            imgs = ax.get_images()
            
            for img in imgs:
                try:
                    cbar = img.colorbar
                    cbar.ax.yaxis.label.set_fontsize(element_fontsize)
                except AttributeError:
                    pass
        
        # Redraw the canvas to update changes.
        fig.canvas.draw_idle()

    def change_fontsizes(self, title_fontsize, element_fontsize):
        SimulationViewer._change_fontsizes(title_fontsize, element_fontsize)


    @staticmethod
    def default_fontsize(func):
        @functools.wraps(func)
        def wrap(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            SimulationViewer._change_fontsizes(
                SimulationViewer.DEFAULT_TITLE_FONTSIZE,
                SimulationViewer.DEFAULT_ELEMENT_FONTSIZE
            )
            return result
        return wrap

    def _apply_title(self, ax: plt.Axes, main_title: str, subtitle: Optional[str] = None) -> None:
        
        if main_title:
            ax.set_title(main_title + "\n" + (subtitle if subtitle else ""), fontsize=14)
        

    def _make_2d_slice(self, data: np.ndarray, axis: int = 2, index: Optional[int] = None) -> np.ndarray:
        if data.ndim < 3:
            return data
        index = index if index is not None else data.shape[axis] // 2
        if axis == 0:
            return data[index, :, :]
        elif axis == 1:
            return data[:, index, :]
        elif axis == 2:
            return data[:, :, index]
        else:
            raise ValueError("Axis must be 0, 1, or 2.")
        

    def _plot_data_2d(self, data: np.ndarray, title: Optional[str] = None, subtitle: Optional[str] = None,   
                     cmap: str = 'viridis', axis: int = 2, index: Optional[int] = None)-> None:
        
        slice_data = self._make_2d_slice(data, axis, index)
        plt.imshow(slice_data.T, interpolation='spline36', cmap=cmap, origin='lower')
        plt.gca().set_aspect(self.SQUARE_ASPECT_RATIO[0]/self.SQUARE_ASPECT_RATIO[1])
        ax = plt.gca()
        self._apply_title(ax, main_title=title, subtitle=subtitle)
        

 
    def plot_epsilon_2d(self,
                        title: Optional[str] = "Dielectric Distribution",
                        subtitle: Optional[str] = None,
                        cmap: str = 'cividis',
                        conversion_options: MPBDataOptions = MPBDataOptions(), 
                        axis: int = 2,
                        index: Optional[int] = None                        
                       ) -> None:
        """
        Plot the 2D epsilon data using the converted epsilon file on the current axes.
        Does not call plt.show().
        """
        eps = self.simulation.load_and_convert_epsilon_data(conversion_options)
        if eps is None:
            raise KeyError("Converted epsilon data not found.")
        self._plot_data_2d(eps, title=title, cmap=cmap, axis=axis, index=index, subtitle=subtitle)
        colorbar = plt.colorbar()
        colorbar.set_label('$\\varepsilon$')

        
   
        

    def plot_epsilon_3d(self,
                        title: Optional[str] = "Dielectric Distribution",
                        subtitle: Optional[str] = None,
                        cmap: str = 'viridis',
                        alpha: float = 0.3,
                        aspect_ratio: tuple[float, float, float] = (1, 1, 1),
                        conversion_options: MPBDataOptions = MPBDataOptions(),

                    ) -> None:
        """
        Overlay the 3D epsilon isosurface on the current 3D axes. Does not call plt.show().
        """
        # Convert and load the epsilon data using MPBDataOptions.
        filepath = self.simulation.convert_epsilon_data(conversion_options)
        data = self.simulation.load_h5_data(filepath)
        eps = data.get("data")
        if eps is None:
            raise KeyError("Converted epsilon data not found.")
        
        # Determine the isosurface level as the midpoint of the epsilon range.
        iso = 0.5 * (np.min(eps) + np.max(eps))
        
        # Extract the isosurface via the marching cubes algorithm.
        from skimage import measure
        verts, faces, normals, _ = measure.marching_cubes(eps, level=iso)
        
        # Get the current figure and axes; if the current axes is not 3D, create a new 3D subplot.
        fig = plt.gcf()
        old_ax = plt.gca()
        if not hasattr(old_ax, 'view_init'):
            fig.delaxes(old_ax)
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = old_ax
        
        # Create and add the polygon mesh for the isosurface.
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        mesh = Poly3DCollection(verts[faces], alpha=alpha)
        face_color = plt.get_cmap(cmap)(0.5)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        
        # Set the plot limits based on the data dimensions.
        nx, ny, nz = eps.shape
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)
        
        # Attempt to set the aspect ratio of the 3D plot.
        try:
            ax.set_box_aspect(aspect_ratio)
        except Exception:
            pass
        
        # Create a colorbar mapping the epsilon values.
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        mappable = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=np.min(eps), vmax=np.max(eps)))
        mappable.set_array(eps)
        plt.colorbar(mappable, ax=ax, pad=0.1, label='$\\varepsilon$')
        
        # Apply the title and subtitle to the plot.
        self._apply_title(ax, main_title=title, subtitle=subtitle)



      
    def plot_epsilon_contour_2d(self,
                                title: Optional[str] = None,
                                cmap: str = 'cividis',
                                conversion_options: MPBDataOptions = MPBDataOptions(),
                                axis: int = 2,
                                index: Optional[int] = None
                                 ) -> None:
        """
        Plot a 2D contour of epsilon data using the converted epsilon file on the current axes.
        Does not call plt.show().
        """ 
        eps = self.simulation.load_and_convert_epsilon_data(conversion_options)
        if eps is None:
            raise KeyError("Converted epsilon data not found.")
        slice_eps = self._make_2d_slice(eps, axis, index)
        # Calculate a single level between min and max
        level = (np.min(slice_eps) + np.max(slice_eps)) / 2
        plt.contour(slice_eps, levels=[level], cmap=cmap)
        ax = plt.gca()
        self._apply_title(ax, main_title=title)
        

    def rotate(self, azim: float, elev: float) -> None:
        """
        Rotate the 3D view in the current axes.
        """
        ax = plt.gca()
        if not hasattr(ax, "view_init"):
            raise ValueError("Current axes is not 3D.")
        ax.view_init(elev=elev, azim=azim)


    
    def plot_field_2d(self,
                    k_idx: int,
                    b_idx: int,
                    field_type: str,
                    comp: str,
                    polarization: str,
                    nonbloch: bool = False,
                    operation: callable = np.real, 
                    cmap: str = "jet",
                    overlay_epsilon_contour: bool = False,
                    conversion_options: MPBDataOptions = MPBDataOptions(),
                    axis: int = 2,
                    index: Optional[int] = None,
                    filecomp = None, 
                    ) -> Colorbar:
        """
        Plot a 2D field of the given type, component, and polarization on the current axes.
        Does not call plt.show().
        """

        field_complex = self.simulation.load_and_convert_field_data(k_idx, b_idx, 
                                                                    comp, polarization,
                                                                    field_type, 
                                                                    conversion_options, filecomp, 
                                                                    nonbloch, overwrite=True)
        field_complex = self._make_2d_slice(field_complex, axis, index)
        title = f"{self.simulation.simulation_name} \n {field_type}-field: k{k_idx:02d}, b{b_idx:02d}, comp={comp}, {polarization}"
        subtitle = f"{operation.__name__.capitalize()}"
        field_after_operation = operation(field_complex)
        self._plot_data_2d(field_after_operation, title=title, subtitle=subtitle, cmap=cmap, axis=axis, index=index)
        c = plt.colorbar()
        c.set_label(f"{operation.__name__.capitalize()}(${field_type.upper()}_{comp}$)")
        if overlay_epsilon_contour:
            self.plot_epsilon_contour_2d(title=None, cmap='gray', conversion_options=conversion_options, axis=axis, index=index)
        return c
    

    
    def plot_band_diagram(self,
                          polarization: str = "te",
                          title: Optional[str] = "Band Diagram",
                          subtitle: Optional[str] = None,
                          color: Optional[Union[list[str], str]] = None,
                          grid: bool = True,
                          k_points_path: Optional[dict] = None,
                          custom_label: str | None = None, 
                          plot_options: Optional[dict] = None
                         ) -> None:
        """
        Plot the band diagram for the given mode on the current axes. Does not call plt.show().
        """
        df = self.simulation.load_frequency_data(polarization)
        bands = df.columns[5:]
        plot_color = "C0"
        if isinstance(color, list) and color:
            plot_color = color[0]
        elif isinstance(color, str):
            plot_color = color
        for i, col in enumerate(bands):
            if i == 0:
                if custom_label is not None:
                    label = custom_label
                else:
                    label = f"{polarization.upper()} bands"
            else:
                label = None
            plt.plot(df["k index"], df[col], color=plot_color, label=label, **(plot_options or {}))
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
        plt.xlabel("Wavevector")
        plt.ylabel("Frequency")
        ax = plt.gca()
        self._apply_title(ax, main_title=title, subtitle=subtitle)
        plt.legend(
            loc='best',
            frameon=True,              # ← Show frame
            facecolor='white',         # ← White background
            edgecolor='black',         # ← Black border
            framealpha=0.90            # ← Nearly opaque
        )
        plt.grid(grid)
    
    
    def plot_light_cone(self, df: pd.DataFrame, opts: str |None = None) -> None:
        """
        Plot the light cone for the simulation on the current axes. Does not call plt.show().
        """
        opts = "k--" if opts is None else opts
        plt.plot(df['k index'], df['kmag/2pi'], opts, label='Light cone')
        plt.legend()


    
    def savefig(self, filename: str, dpi: int = 300) -> None:
        """
        Save the current figure to a file.
        """
        plt.savefig(filename, dpi=dpi)

    
        
