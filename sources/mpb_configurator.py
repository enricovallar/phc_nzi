from photonic_crystal_maker import *
import meep as mp
from meep import mpb   

class MPBSchemeConfigurator():
    def __init__(self, phc: PhotonicCrystal,  simulation_types: list = ["te"], resolution: int | mp.Vector3 = 32, num_bands: int = 8, 
                 k_point_interpolation_factor: int = 4, mesh_size: int | None = None, target_freq: float = None, tolerance: float = None, 
                 k_points: list = [mp.Vector3(0, 0, 0)]):
        # PhotonicCrystal object is used to generate the geometry and lattice
        if type(phc) == PhotonicCrystal:    
            self._phc = phc
        else:
            raise ValueError("Photonic crystal must be of type PhotonicCrystal")
         
        # Simulation internal parameters are set to None by default
        self._simulation_type = None 
        self._resolution = None
        self._num_bands = None
        self._k_points_interpolation_factor = None
        self._mesh_size = None
        self._target_freq = None
        self._tolerance = None
        self._k_points = None


        # Initializing the internal parameters checking the types
        self.simulation_types = simulation_types 
        self.resolution = resolution
        self.num_bands = num_bands
        self.k_points_interpolation_factor = k_point_interpolation_factor
        self.mesh_size = mesh_size
        self.target_freq = target_freq
        self.tolerance = tolerance
        self.k_points = k_points

        

    

    @property
    def simulation_types(self):
        return self._simulation_types

    @simulation_types.setter
    def simulation_types(self, value):
        allowed_types = ['te', 'tm', 'zeven', 'zodd']
        if isinstance(value, list) and all(item in allowed_types or item is None for item in value):
            self._simulation_types = value
        else:
            raise ValueError("Simulation types must be a list with each element being one of 'te', 'tm', 'zeven', 'zodd' or None.")

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
    def mesh_size(self):
        return self._mesh_size

    @mesh_size.setter
    def mesh_size(self, value):
        if isinstance(value, int):
            self._mesh_size = value
        elif value is None:
            self._mesh_size = value
        else:
            raise ValueError("Mesh size must be an integer or None")

    @property
    def target_freq(self):
        return self._target_freq

    @target_freq.setter
    def target_freq(self, value):
        if isinstance(value, float):
            self._target_freq = value
        elif value is None: 
            self._target_freq = value
        else:
            raise ValueError("Target frequency must be a float or None")

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        if isinstance(value, float):
            self._tolerance = value
        elif value is None:
            self._tolerance = value
        else:
            raise ValueError("Tolerance must be a float or None")   
        
    @property
    def k_points_interpolation_factor(self):
        return self._k_points_interpolation_factor
    
    @k_points_interpolation_factor.setter    
    def k_points_interpolation_factor(self, value):
        if isinstance(value, int):
            self._k_points_interpolation_factor = value
        elif value is None:
            self._k_points_interpolation_factor = value
        else:
            raise ValueError("K-point interpolation factor must be an integer or None")
        

    @property
    def k_points(self):
        return self._k_points   
    
    @k_points.setter
    def k_points(self, value: list):
        if all(isinstance(k_point, mp.Vector3) for k_point in value):
            self._k_points = value
        else:
            raise ValueError("All k-points must be of type meep Vector3")  


     
        
    
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
        
        # Set the lattice and geometry
        commands += self.generate_lattice_and_geometry_commands()

        # Set the simulation type
        commands += self.generate_runner_commands()

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
    
    def print_scheme_config(self):
        commands = self.build_commands()
        for command in commands:
            print(command)  

    def generate_number_of_bands_command(self): 
        return [f"(set! num-bands {self.num_bands})"]
    
    def generate_resolution_command(self):
        if isinstance(self.resolution, int):
            return [f"(set! resolution {self.resolution})"]
        else:
            return [f"(set! resolution (vector3 {self.resolution[0]} {self.resolution[1]} {self.resolution[2]}))"]
        
    def generate_mesh_size_command(self):
        if self.mesh_size:
            return [f"(set! mesh-size {self.mesh_size})"]
        else:
            return []
        
    def generate_lattice_and_geometry_commands(self):
        return [self._phc.to_scheme()]


    
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
            return []
        
    def generate_runner_commands(self):
        commands = []
        for sim_type in dict.fromkeys(self.simulation_types):
            if sim_type == "tm":
                commands.append("(run-tm)")
            elif sim_type == "te":
                commands.append("(run-te)")
            elif sim_type == "zeven":
                commands.append("(run-zeven)")
            elif sim_type == "zodd":
                commands.append("(run-zodd)")
            else:
                print(f"simulation type: {sim_type}")
                commands.append("(run)")
        return commands
        

# Example usage
if __name__ == "__main__":
    atom = Geometry(mp.Cylinder, {"radius": 0.2, "height": 0.5, "center": mp.Vector3(0, 0, 0), "material": Material(epsilon=12)})
    lattice = Lattice("SX", (1, 1, Lattice.NO_SIZE))

    phc = PhotonicCrystal([atom], lattice)
    mpb_config = MPBSchemeConfigurator(phc, simulation_types=["te", "tm"], resolution=32, num_bands=8, k_point_interpolation_factor=4, mesh_size=16, target_freq=0.5, tolerance=1e-6, k_points=[mp.Vector3(0, 0, 0)])
    mpb_config.print_scheme_config()