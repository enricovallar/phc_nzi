from phc_nzi.photonic_crystal_maker import *
import meep as mp
from meep import mpb   



class MPBSchemeConfigurator():

    VALID_SIMULATION_TYPES = ["te", "tm", "zeven", "zodd"]

    def __init__(self, phc: PhotonicCrystal,  simulation_types: list = ["te"], resolution: int | mp.Vector3 = 32, num_bands: int = 8, 
                 k_points_interpolation_factor: int = 4, mesh_size: int | None = None, target_freq: float = None, tolerance: float = None, 
                 k_points: list = [mp.Vector3(0, 0, 0)], extra_runner_command: str = ""):
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
        self.k_points_interpolation_factor = k_points_interpolation_factor
        self.mesh_size = mesh_size
        self.target_freq = target_freq
        self.tolerance = tolerance
        self.k_points = k_points

        self.extra_runner_command = extra_runner_command    

    def validate_simulation_type(self, simulation_type):
        if simulation_type in self.VALID_SIMULATION_TYPES:
            return True
        else:
            return False    


    @property
    def simulation_types(self):
        return self._simulation_types

    @simulation_types.setter
    def simulation_types(self, value):
        allowed_types = self.VALID_SIMULATION_TYPES
        if isinstance(value, list) and all(item in allowed_types or item is None for item in value):
            self._simulation_types = value
        else:
            raise ValueError(f"Simulation types must be a list with each element being one of {self.VALID_SIMULATION_TYPES} or None.")

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
            self._k_points_interpolation_factor = None
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

        # Extra run functions   
        commands += self.generate_extra_run_functions()

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
        #print(f"Scheme configuration written to {filename}")
        return script
    
    def print_scheme_config(self):
        commands = self.build_commands()
        for command in commands:
            print(command)

    def get_scheme_config(self, join_newline=False):    
        commands = self.build_commands()
        if join_newline:
            return "\n".join(commands)
        else:
            commands_cleaned = [command.replace("\n", "") for command in commands]
            return " ".join(commands_cleaned)

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
        if self._k_points_interpolation_factor is None:
            return [""]
        elif self._k_points:
            return [f"(set! k-points (interpolate {self._k_points_interpolation_factor} k-points))"]
        else:
            raise ValueError("Invalid k-points interpolation factor")
        
    def generate_extra_run_functions(self):
        functions = []
        
        # Define all function strings
        efield_nonbloch = """
(define (output-nonbloch-efield which-band)
    (get-efield which-band)
    (cvector-field-nonbloch! cur-field)
    (output-field-to-file -1 (string-append (get-filename-prefix)"e."))
)
"""
        
        hfield_nonbloch = """
(define (output-nonbloch-hfield which-band)
    (get-hfield which-band)
    (cvector-field-nonbloch! cur-field)
    (output-field-to-file -1 (string-append (get-filename-prefix)"h."))
)
"""
        
        efield_nonbloch_x = """
(define (output-nonbloch-efield-x which-band)
    (get-efield which-band)
    (cvector-field-nonbloch! cur-field)
    (output-field-to-file 0 (string-append (get-filename-prefix)"e."))
)
"""
        
        hfield_nonbloch_x = """
(define (output-nonbloch-hfield-x which-band)
    (get-hfield which-band)
    (cvector-field-nonbloch! cur-field)
    (output-field-to-file 0 (string-append (get-filename-prefix)"h."))
)
"""
        
        efield_nonbloch_y = """
(define (output-nonbloch-efield-y which-band)
    (get-efield which-band)   
    (cvector-field-nonbloch! cur-field)
    (output-field-to-file 1 (string-append (get-filename-prefix)"e."))
)
"""
        
        hfield_nonbloch_y = """
(define (output-nonbloch-hfield-y which-band)
    (get-hfield which-band)
    (cvector-field-nonbloch! cur-field)
    (output-field-to-file 1 (string-append (get-filename-prefix)"h."))
)
"""
        
        efield_nonbloch_z = """
(define (output-nonbloch-efield-z which-band)
    (get-efield which-band)
    (cvector-field-nonbloch! cur-field)
    (output-field-to-file 2 (string-append (get-filename-prefix)"e."))
)
"""
        
        hfield_nonbloch_z = """
(define (output-nonbloch-hfield-z which-band)
    (get-hfield which-band)
    (cvector-field-nonbloch! cur-field)
    (output-field-to-file 2 (string-append (get-filename-prefix)"h."))
)
"""

        display_symmetries_c4v = """
; --- C4v Definitions for Square Lattice: basis1=(1,0), basis2=(0,1) ---
(define C4-s (matrix3x3 (vector3 0 1 0) (vector3 -1 0 0) (vector3 0 0 1)))
(define C2-s (matrix3x3 (vector3 -1 0 0) (vector3 0 -1 0) (vector3 0 0 1)))
(define sv-s (matrix3x3 (vector3 1 0 0) (vector3 0 -1 0) (vector3 0 0 1)))  
(define sd-s (matrix3x3 (vector3 0 1 0) (vector3 1 0 0) (vector3 0 0 1)))  

(define (display-symmetries-c4v)
  (if (vector3= current-k (vector3 0 0 0))
      (begin
        (print "SYM_DATA_START_" parity "\n")
        (map (lambda (b)
               (print parity "," b 
                      ",C4=" (compute-symmetry b C4-s (vector3 0 0 0)) 
                      ",C2=" (compute-symmetry b C2-s (vector3 0 0 0)) 
                      ",sv=" (compute-symmetry b sv-s (vector3 0 0 0)) 
                      ",sd=" (compute-symmetry b sd-s (vector3 0 0 0)) "\n"))
             (arith-sequence 1 1 num-bands))
        (print "SYM_DATA_END_" parity "\n"))))
"""
        display_symmetries_c6v = """
; --- C6v Definitions for Hexagonal Lattice: basis1=(1,0), basis2=(0.5, 0.866) ---
(define C6-h (matrix3x3 (vector3 0 1 0) (vector3 -1 1 0) (vector3 0 0 1))) 
(define C3-h (matrix3x3 (vector3 -1 1 0) (vector3 -1 0 0) (vector3 0 0 1)))
(define C2-h (matrix3x3 (vector3 -1 0 0) (vector3 0 -1 0) (vector3 0 0 1))) 
(define sv-h (matrix3x3 (vector3 1 0 0) (vector3 1 -1 0) (vector3 0 0 1)))  
(define sd-h (matrix3x3 (vector3 0 1 0) (vector3 1 0 0) (vector3 0 0 1)))   

(define (display-symmetries-c6v)
  (if (vector3= current-k (vector3 0 0 0))
      (begin
        (print "SYM_DATA_START_" parity "\n") ; Unique start tag
        (map (lambda (b)
               (print parity "," b 
                      ",C6=" (compute-symmetry b C6-h (vector3 0 0 0)) 
                      ",C3=" (compute-symmetry b C3-h (vector3 0 0 0)) 
                      ",C2=" (compute-symmetry b C2-h (vector3 0 0 0)) 
                      ",sv=" (compute-symmetry b sv-h (vector3 0 0 0)) 
                      ",sd=" (compute-symmetry b sd-h (vector3 0 0 0)) "\n"))
             (arith-sequence 1 1 num-bands))
        (print "SYM_DATA_END_" parity "\n")))) ; Unique end tag
"""
        
        # Add all functions to the list
        functions.extend([
            efield_nonbloch, 
            hfield_nonbloch, 
            efield_nonbloch_x, 
            hfield_nonbloch_x, 
            efield_nonbloch_y, 
            hfield_nonbloch_y, 
            efield_nonbloch_z, 
            hfield_nonbloch_z,
            display_symmetries_c4v,
            display_symmetries_c6v
        ])
        
        return functions
    
    def generate_runner_commands(self, extra_commands=None):
        commands = []
        if extra_commands is None:
            extra_commands = self.extra_runner_command
        for sim_type in dict.fromkeys(self.simulation_types):
            if sim_type == "tm":
                commands.append(f"(run-tm {extra_commands})")
            elif sim_type == "te":
                commands.append(f"(run-te {extra_commands})")
            elif sim_type == "zeven":
                commands.append(f"(run-zeven {extra_commands})")
            elif sim_type == "zodd":
                commands.append(f"(run-zodd {extra_commands})")
            else:
                print(f"simulation type: {sim_type}")
                commands.append(f"(run {extra_commands})")
        return commands
    
        
