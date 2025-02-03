from photonicCrystal import *
import meep as mp
from meep import mpb   

class MPBConfiguration():
    def __init__(self, simulation_type: str = "te", resolution: int | mp.Vector3 = 32, num_bands: int = 8, k_point_interpolation_factor = 4,
                 geometry_lattice:  Lattice = None, geometry: list = None):
        self._simulation_type = None
        self._resolution = None
        self._num_bands = None
        self._mesh_size = None
        self._target_freq = None
        self._tolerance = None
        self._geometry_lattice = None
        self._geometry = None
        self._k_points = None
        self._k_point_interpolation_factor = None
        
        
        
        self.simulation_type = simulation_type
        self.resolution = resolution
        self.num_bands = num_bands
        self.geometry_lattice = geometry_lattice
        self.geometry = geometry
        self.k_points_interpolation_factor = k_point_interpolation_factor
        

    

    @property
    def simulation_type(self):
        return self._simulation_type
    @simulation_type.setter
    def simulation_type(self, value):
        if value in ['te', 'tm', 'zeven', 'zodd']:
            self._simulation_type = value
        else:
            raise ValueError("Simulation type must be one of 'te', 'tm', 'zeven', 'zodd' or None.")

    @property
    def resolution(self):
        return self._resolution
    
    @resolution.setter
    def resolution(self, value):
        if isinstance(value, int):
            self._resolution = value
        elif isinstance(value, mp.Vector3):
            self._resolution = value 
        else:
            raise ValueError("Resolution must be an integer or a meep vector")  
    
    @property
    def num_bands(self):
        return self._num_bands
    
    @num_bands.setter
    def num_bands(self, value):
        if isinstance(value, int):
            self._num_bands = value
        else:
            raise ValueError("Number of bands must be an integer")
        
    @property
    def geometry_lattice(self):
        return self._geometry_lattice
    
    @geometry_lattice.setter
    def geometry_lattice(self, value):
        if isinstance(value, Lattice):
            self._geometry_lattice = value
        else:
            raise ValueError("Geometry lattice must be of type Lattice")
    
    @property
    def geometry(self):
        return self._geometry
    
    @geometry.setter
    def geometry(self, value: list):
        if all(isinstance(geom_element, Geometry) for geom_element in value):
            self._geometry = value
        else:
            raise ValueError("All geometry elements must be of type Geometry")
    
    @property
    def k_points(self):
        return self._k_points   
    
    @k_points.setter
    def k_points(self, value: list):
        if all(isinstance(k_point, mp.Vector3) for k_point in value):
            self._k_points = value
        else:
            raise ValueError("All k-points must be of type meep Vector3")  

    @property
    def mesh_size(self):
        return self._mesh_size

    @mesh_size.setter
    def mesh_size(self, value):
        if isinstance(value, int):
            self._mesh_size = value
        else:
            raise ValueError("Mesh size must be an integer")

    @property
    def target_freq(self):
        return self._target_freq

    @target_freq.setter
    def target_freq(self, value):
        if isinstance(value, float):
            self._target_freq = value
        else:
            raise ValueError("Target frequency must be a float")

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        if isinstance(value, float):
            self._tolerance = value
        else:
            raise ValueError("Tolerance must be a float")    
        
    @property
    def k_points_interpolation_factor(self):
        return self._k_points_interpolation_factor
    
    @k_points_interpolation_factor.setter    
    def k_points_interpolation_factor(self, value):
        if isinstance(value, int):
            self._k_points_interpolation_factor = value
        else:
            raise ValueError("K-point interpolation factor must be an integer") 
        
    
    def build_commands(self):
        commands = []

        # Set the number of bands
        commands += self.generate_number_of_bands_command()
        # Set the resolution
        commands += self.generate_resolution_command()
        # Set the mesh size if provided
        commands += self.generate_mesh_size_command()

        # Set k-points
        commands += self.generate_k_points_command()
        # Interpolate k-points 
        commands += self.generate_k_points_interpolation_command()
        
       
        
        # Set the geometry
        commands += self.generate_geometry_command()
        # Set the geometry lattice
        commands += self.generate_geometry_lattice_command()

        # Set the simulation type
        commands += self.generate_runner_command()

        return commands
    
    def generate_scheme_config(self, filename):
        """Generate the complete Scheme configuration string."""
        script = ""
        commands = self.build_commands()
        with open(filename, "w") as f:
            for command in commands:
                f.write(command)
                f.write("\n")
                script += command + "\n" 
        print(f"Scheme configuration written to {filename}")
        return script

        
        
    
    def generate_number_of_bands_command(self): 
        return [f"(set! num-bands {self.num_bands})"]
    
    def generate_resolution_command(self):
        if isinstance(self.resolution, int):
            return [f"(set! resolution {self.resolution})"]
        else:
            return [f"(set! resolution (vector3 {self.resolution[0]} {self.resolution[1]} {self.resolution[2]})"]
        
    def generate_mesh_size_command(self):
        if self.mesh_size:
            return [f"(set! mesh-size {self.mesh_size})"]
        else:
            return [""]
        
    def generate_geometry_lattice_command(self):
        return [f"(set! geometry-lattice {self.geometry_lattice.to_scheme()})"]
    
    def generate_geometry_command(self): 
        commands = []   
        if self._geometry:
            # Assume self.geometry is a list of valid Scheme strings for geometry objects.
            geom_str = f"(list\n"
            for geom in self._geometry:
                if isinstance(geom, Geometry):
                    geom_str += f"    {geom.to_scheme()}\n"
            geom_str += ")"
            commands.append(f"(set! geometry {geom_str})")
        else:
            raise ValueError("Geometry must be provided")
        return commands
    
    def generate_k_points_command(self):    
        if self._k_points:
            kpts_str = "(list " + " ".join([f"(vector3 {pt[0]} {pt[1]} {pt[2]})" for pt in self._k_points]) + ")"
            return [f"(set! k-points {kpts_str})"]
        else:
            raise ValueError("K-points must be provided")   
        
    def generate_k_points_interpolation_command(self): 
        if self._k_points:
            return [f"(set! k-points (interpolate {self._k_points_interpolation_factor} k-points))"]
        else:
            raise ValueError("K-points must be provided")
        
    def generate_runner_command(self):
        if self.simulation_type == "tm":
            return ["(run-tm)"]
        elif self.simulation_type == "te":
            return ["(run-te)"]
        elif self.simulation_type == "zeven":
            return ["(run-zeven)"]
        elif self.simulation_type == "zodd":
            return ["(run-zodd)"]
        else:
            return ["(run)"]
        

import h5py    

class H5Loader():

    def load_h5(self, filename):
        with h5py.File(filename, "r") as f:
            data = {}
            for key in f.keys():
                data[key] = f[key][()]
        