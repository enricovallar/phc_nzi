import meep as mp
from meep import mpb 
import numpy as np





class Lattice:
    LATTICE_TYPES = {
        
        # Simple lattices
        'SX': 'square lattice, atoms to be defined by user',
        'TX': 'triangular, atoms to be defined by user',

        # Diatomic lattices
        'SXY': 'diatomic square lattice, atoms to be defined by user',
        'TXY': 'diatomic triangular lattice, atoms to be defined by user',
    }

    NO_SIZE = "no-size"

    def __init__(self, type: str, size: tuple | int = (1, 1, 0)):
        """Create a lattice object. 
        
        Args:
            type (str): The type of the lattice. Must be one of the keys in Lattice.LATTICE_TYPES: 'SX', 'TX', 'SXY', 'TXY'.
            size (tuple | int): The size of the lattice. If the lattice is 2D, the size is a tuple (x, y, z). If the lattice is 1D, the size is an int.
        
        Raises:
            ValueError: If the lattice type is not valid.
            """
        self._type = type
        self._size = size
        self._make_lattice()

    def _make_lattice(self):
        if self._type in Lattice.LATTICE_TYPES:
            if self._type == 'SX':
                self._make_SX()
            elif self._type == 'TX':
                self._make_TX()
            elif self._type == 'SXY':
                self._make_SXY()
            elif self._type == 'TXY':
                self._make_TXY()
        else:
            raise ValueError(f"Invalid lattice type: {self._type}. Must be one of {Lattice.LATTICE_TYPES.keys()}")
        
    
    def _make_SX(self):
        self._mp_lattice = mp.Lattice(size=self._size, 
                                      basis1=mp.Vector3(1, 0, 0),
                                      basis2=mp.Vector3(0, 1, 0))

    def _make_TX(self): 
        self._mp_lattice = mp.Lattice(size=self._size,
                                      basis1=mp.Vector3(1, 0, 0),
                                      basis2=mp.Vector3(0.5, np.sqrt(3)/2, 0))

    def _make_SXY(self):
        self._mp_lattice = mp.Lattice(size=self._size,
                                        basis1=mp.Vector3(1, 0, 0),
                                        basis2=mp.Vector3(0, 1, 0))
        
    def _make_TXY(self):
        self._mp_lattice = mp.Lattice(size=self._size,
                                        basis1=mp.Vector3(1, 0, 0),
                                        basis2=mp.Vector3(0.5, np.sqrt(3)/2, 0))
        
    def cartesian_to_reciprocal(self, vector: mp.Vector3):
        return mp.cartesian_to_reciprocal(vector, self._mp_lattice)

        
    def get_centers(self):
        if self._type == "SX":
            return mp.Vector3(0, 0)
        
        if self._type == "TX":
            return mp.Vector3(0, 0)
       
        if self._type == "SXY":
            return (mp.Vector3(0, 0), mp.Vector3(0.5, 0.5))
        
        if self._type == "TXY": 
            return (mp.Vector3(1/3, 1/3), mp.Vector3(2/3, 2/3))
        
    def to_python(self) -> mp.Lattice:
        return self._mp_lattice 
    
    def to_scheme(self) -> str:
        size_0 = self._size[0] if self._size[0] != 0 else "no-size"
        size_1 = self._size[1] if self._size[1] != 0 else "no-size"
        size_2 = self._size[2] if self._size[2] != 0 else "no-size"
        return f"(make lattice (size {size_0} {size_1} {size_2}) " + \
               f"(basis1  (vector3 {self._mp_lattice.basis1[0]} {self._mp_lattice.basis1[1]} {self._mp_lattice.basis1[2]})) " + \
               f"(basis2  (vector3 {self._mp_lattice.basis2[0]} {self._mp_lattice.basis2[1]} {self._mp_lattice.basis2[2]})) " + \
                ")"
    
    def get_high_symmetry_k_points(self, centerd_in_gamma=False):
        """Get the high symmetry k points of the lattice.
        
        Returns:
            dict: A dictionary with the keys "k_points_values" and "k_points_labels".
        """
        if not centerd_in_gamma:
            if self._type in ("SX", "SXY"):
                # Square lattice
                k_points_values = [mp.Vector3(0, 0, 0),
                                mp.Vector3(0.5, 0, 0),
                                mp.Vector3(0.5, 0.5, 0),
                                mp.Vector3(0, 0, 0)]
                k_points_labels = ["Γ", "X", "M", "Γ"]
            elif self._type in ("TX", "TXY"):
                # Triangular (hexagonal) lattice
                k_points_values = [mp.Vector3(0, 0, 0),       # Γ
                                mp.Vector3(0, 0.5, 0),     # M
                                mp.Vector3(-1/3, 1/3, 0),   # K
                                mp.Vector3(0, 0, 0)]       # Γ
                k_points_labels = ["Γ", "M", "K", "Γ"]
            else:
                raise ValueError(f"Invalid lattice type: {self._type}")
        else:
            if self._type in ("SX", "SXY"):
                # Square lattice with gamma as the second value
                k_points_values = [mp.Vector3(0.5, 0, 0),
                                mp.Vector3(0, 0, 0),
                                mp.Vector3(0.5, 0.5, 0),
                                mp.Vector3(0.5, 0, 0)]
                k_points_labels = ["X", "Γ", "M", "X"]
            elif self._type in ("TX", "TXY"):
                # Triangular lattice with gamma as the second value
                k_points_values = [mp.Vector3(0.5, 0, 0),     # M
                                mp.Vector3(0, 0, 0),       # Γ
                                mp.Vector3(1/3, 1/3, 0),   # K
                                mp.Vector3(0.5, 0, 0)]     # M
                k_points_labels = ["M", "Γ", "K", "M"]
            else:
                raise ValueError(f"Invalid lattice type: {self._type}")
        return {"k_points_values": k_points_values, "k_points_labels": k_points_labels}
         

    def get_k_points_around_Gamma(self, distance: float):  
        """Get the k points around the Gamma point of the lattice.    
        Args:
            distance (float): The distance from the Gamma point.
        Returns:
            dict: A dictionary with the keys "k_points_values" and "k_points_labels".
        """
        K_X_cartesian = mp.Vector3(distance, 0, 0)   
        K_Y_cartesian = mp.Vector3(0, distance, 0)
        K_X_reciprocal = mp.cartesian_to_reciprocal(K_X_cartesian, self._mp_lattice)
        K_Y_reciprocal = mp.cartesian_to_reciprocal(K_Y_cartesian, self._mp_lattice)
        Gamma = mp.Vector3(0, 0, 0) 

        k_points_values = [K_X_reciprocal, Gamma, K_Y_reciprocal]
        k_points_labels = ["X", "Γ", "Y"]
        return {"k_points_values": k_points_values, "k_points_labels": k_points_labels}
        

class ScriptParam:
    def __init__(self, name, default_value): 
        self._names = [name]
        self._default_values = [default_value]
        self._scheme_string = self._names[0]

    def to_scheme(self):
        commands = []
        for name, default_value in zip(self._names, self._default_values):
            commands.append( f"(define-param {name} {default_value})" )
        return commands
    
    def to_scheme_by_name(self, name):
        
        if name in self._names:
            index = self._names.index(name)
            return [f"(define-param {self._names[index]} {self._default_values[index]})"]
        else:
            raise ValueError(f"Name {name} not found in script param names")

    
    def __str__(self):
        return self._scheme_string

class ScriptParamVector3(ScriptParam):
    def __init__(self, x= 1, y=1, z=1, x_def=1, y_def=1, z_def=1):
        self._validate(x, y, z, x_def, y_def, z_def)
        # Build the scheme string using the provided component values directly.
        self._scheme_string = f"(vector3 {x} {y} {z})"
        self._names = []
        self._default_values = []
        # Only add to the names list if the component is given as a string.
        if isinstance(x, str):
            self._names.append(x)
            self._default_values.append(x_def)
        if isinstance(y, str):
            self._names.append(y)
            self._default_values.append(y_def)
        if isinstance(z, str):
            self._names.append(z)
            self._default_values.append(z_def)

    def _validate(self, x, y, z, x_def, y_def, z_def):
        # For each component, if it's a string, its default must be int or float.
        # Otherwise, the component must be int or float.
        for comp, comp_def, label in ((x, x_def, "x"), (y, y_def, "y"), (z, z_def, "z")):
            if isinstance(comp, str):
                if not isinstance(comp_def, (int, float)):
                    raise ValueError(f"Default value for {label} must be an int or float")
            elif not isinstance(comp, (int, float)):
                raise ValueError(f"{label} must be either a string or an int/float")
                

class ScriptParams:
    def __init__(self, *script_params: ScriptParam):
        self._script_params = list(script_params)

    def __add__(self, other):   
        if isinstance(other, ScriptParams):
            return ScriptParams(*self._script_params, *other._script_params)    
        elif isinstance(other, ScriptParam):
            return ScriptParams(*self._script_params, other)
        else:
            raise ValueError(f"Invalid type for addition: {type(other)}")   
    
    def to_scheme(self):
        commands = []
        already_added = set()
        for script_param in self._script_params:
            for name in script_param._names:
                if name not in already_added:
                    commands.extend(script_param.to_scheme_by_name(name))
                    already_added.add(name)

        print("COMMANDS", commands) 
        return commands
        

    
class Geometry:
    """ This class is a wrapper for meep geometry objects. It can be used to create meep geometry objects and convert them to Scheme strings. """
    VALID_SCHEME_GEOMETRIES = {
        mp.Cylinder: "cylinder",
        mp.Sphere: "sphere",
        mp.Block: "block",
        mp.Prism: "prism",
        mp.Ellipsoid: "ellipsoid",
    }

    def __init__(self, geom_type, params: dict):   
        self.mp_geom_type = geom_type
        self.params = params
        self.script_params = ScriptParams()
        self.collect_script_params()

    
    def is_script_param(self, param_value):
        return isinstance(param_value, ScriptParam)
            
        
    def is_script_param_vector3(self, param_value):   
        if isinstance(param_value, ScriptParamVector3):
            return True
            


    def parse_script_param(self, param_value: str): 
  
        if self.is_script_vector3_param(param_value):
            param_name = self._parse_script_vector3_param(param_value)
        elif self.is_script_param(param_value):
            param_name = self._parse_script_param(param_value)
        else:
            raise ValueError(f"Invalid script param value: {param_value}")
        return param_name
    
    
    def _parse_script_vector3_param(self, param_value: str):
        
        if self.is_script_vector3_param(param_value):
            # Expect format: vector3-{name}=(x, y, z)
            # Remove the "vector3-" prefix
            stripped = param_value[len(Geometry.GEOM_PARAM_PREFIX + Geometry.VECTOR3_PREFIX):]
            try:
                # Split into name and tuple string based on the '='
                name, tuple_str = stripped.split("=", 1)
            except ValueError:
                raise ValueError(f"Invalid vector3 param format: {param_value}")
            tuple_str = tuple_str.strip()
            # Remove enclosing parentheses if present
            if tuple_str.startswith("(") and tuple_str.endswith(")"):
                tuple_str = tuple_str[1:-1]
            numbers = [num.strip() for num in tuple_str.split(",")]
            if len(numbers) != 3:
                raise ValueError(f"Invalid vector3 tuple: {param_value}")
            try:
                x, y, z = (float(n) for n in numbers)
            except Exception as e:
                raise ValueError(f"Could not parse numbers in vector3 param: {param_value}") from e
            
            # add {name}_x, {name}_y, {name}_z to the sript_params_default dictionary
            self.script_params_default[f"{name}_x"] = x
            self.script_params_default[f"{name}_y"] = y
            self.script_params_default[f"{name}_z"] = z
            return f"(vector3 {name}_x {name}_y {name}_z)" 
        else:
            raise ValueError(f"Invalid vector3 param value: {param_value}")
        
    def build(self):
        if self.mp_geom_type:
            return self.mp_geom_type(**self.params)
        else:
            raise ValueError(f"Invalid geometry type: {self.mp_geom_type}, use one of meep geometry types")
    
    def collect_script_params(self):
        for param, value in self.params.items():
            if self.is_script_param(value):
                print(f"Found script param: {param} with value: {value}")      
                self.script_params += value

                
    def _params_to_scheme(self): 
        commands = []
        for param, value in self.params.items():
            if param == "material":
                if isinstance(value, Material):
                    value = value.to_scheme()
                else:
                    raise ValueError("Material must be of type Material")
            elif isinstance(value, (int, float)):
                value = value
            elif isinstance(value, mp.Vector3):
                value = f"(vector3 {value.x} {value.y} {value.z})"
            elif self.is_script_param(value):
                value = str(value)
            else:
                raise ValueError(f"Invalid value type for {param}")
            commands.append(f"({param} {value})")
        return commands

    def to_scheme(self):
        params_commands = self._params_to_scheme()
        command = f"(make {self.to_valid_scheme_geometry_definition()} "
        command += "\n  ".join(params_commands)
        command += ")"
        return command
    
    def get_script_params(self)-> ScriptParams:
        return self.script_params
    
    def to_valid_scheme_geometry_definition(self):
        if self.mp_geom_type in Geometry.VALID_SCHEME_GEOMETRIES:
            return Geometry.VALID_SCHEME_GEOMETRIES[self.mp_geom_type]
        else:
            raise ValueError(f"Invalid geometry type: {self.mp_geom_type}, use one of meep geometry types available in the python interface")
        
    def to_python(self):
        return self.build()   
        
class GeometryGroup:
    def __init__(self, *geometries:  Geometry):
        self._geometries = geometries
        self._script_params = ScriptParams()

        for geometry in geometries:
            self._script_params += geometry.get_script_params()
            

    def to_scheme(self):
        commands = []
        for geometry in self._geometries:
            commands.append(geometry.to_scheme())
        commands_string =  "\n ".join(commands)
        commands_string = f"(list \n {commands_string}\n)"
        return commands_string
    
    def get_script_params(self):
        return self._script_params  
        





class Material: 
    """ This class is a wrapper for meep material objects. It can be used to create meep material objects and convert them to Scheme strings. """
    def __init__(self, epsilon: float):
        self._epsilon = epsilon

    def to_scheme(self):
        return f"(make dielectric (epsilon {self._epsilon}))"
    
    def to_python(self):
        return mp.Medium(epsilon=self._epsilon) 
    
    @property
    def epsilon(self):
        return self._epsilon
    


    
    

class PhotonicCrystal:
    """
    This class is a wrapper for  some of mpb objects. It can be used to create photonics crystal objects and convert them to Scheme strings.
    """
    def __init__(self, atoms: list, lattice: Lattice, background_material: Material = Material(epsilon=1)):
        """Create a photonic crystal object.

        Args:
            background_material (Material): The background material of the photonic crystal.
            atoms (list): A list of Geometry objects that represent the atoms in the photonic crystal.
            lattice (Lattice): The lattice of the photonic crystal.
        """
        
        if all(isinstance(atom, Geometry) for atom in atoms):
            self._atoms = atoms
            self._geometry_group = GeometryGroup(*atoms)
            self._script_params = self._geometry_group.get_script_params()
        else:
            raise ValueError("All atoms must be of type Geometry")
        
        if isinstance(lattice, Lattice):
            self._lattice = lattice
        else:
            raise ValueError("Lattice must be of type Lattice") 
        
        if isinstance(background_material, Material):
            self._background_material = background_material
        else:
            raise ValueError("Material must be of type Material")  

    def to_scheme_list(self) -> list:
        """Convert the photonic crystal to a Scheme string.
        
        Returns:
            list: A list of Scheme commands that define the photonic crystal. 
            The first command is the lattice definition, the second command is the geometry group definition.
        """
        commands = []
        commands.append(self._lattice.to_scheme())
        commands.append(self._geometry_group.to_scheme())
        return commands
    
    

    def _script_params_to_scheme_string(self) -> str:
        commands = self._script_params.to_scheme()
        if commands:
            return "\n".join(commands)
        else:
            return ""

        

    
    def to_scheme(self) -> str:
        """Convert the photonic crystal to a Scheme string.
        
        Returns:
            str: A Scheme string that defines the photonic crystal. 
        """
        commands_partial = self.to_scheme_list()
        command_lattice = f"(set! geometry-lattice {commands_partial[0]})"
        command_geometry = f"(set! geometry {commands_partial[1]})"
        
        command_script_params  = self._script_params_to_scheme_string()
        return "\n".join([command_script_params, command_lattice, command_geometry])
    

    def print_script_params(self):
        print(self._script_params)
    
    
          

#example usage of the classes
if __name__ == "__main__":

    atom_geometry = Geometry(mp.Cylinder, {"radius": 0.2, "height": ScriptParam(name="h", default_value=0.3), "center": mp.Vector3(0, 0, 0), "material": Material(epsilon=12)})
    atom_geometry2 = Geometry(mp.Block , {"size": ScriptParamVector3(2, 1, "sz", 1, 1, 0.5), "center": mp.Vector3(0.5, 0.5, 0.5), "material": Material(epsilon=12)})
    lattice = Lattice("SX", (1, 1, Lattice.NO_SIZE))
    material = Material(epsilon=12)
    photonic_crystal = PhotonicCrystal([atom_geometry, atom_geometry2], lattice)
    print(photonic_crystal.to_scheme())


