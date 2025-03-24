
from abc import ABC, abstractmethod
import meep as mp
from  typing import Optional
import numpy as np


class KPath:
    def __init__(self, values: Optional[list] = None, labels: Optional[list] = None):
        self.values = values
        self.labels = labels

    def to_dict(self):
        return {"k_points_values": self.values, "k_points_labels": self.labels}


class BravaisLattice(ABC):
    def __init__(self, supercell_height: Optional[int] = None):
        self._supercell_height = supercell_height   
        self._mp_lattice = None
        self._size = None
        self._G = mp.Vector3(0, 0, 0)
        self._path_starting_in_gamma = KPath()
        self._path_centered_in_gamma = KPath()

    
       
    @abstractmethod
    def _make_lattice(self) -> mp.Lattice:
        pass

    def cartesian_to_reciprocal(self, vector: mp.Vector3):
        return mp.cartesian_to_reciprocal(vector, self._mp_lattice)
    
    def reciprocal_to_cartesian(self, vector: mp.Vector3):
        return mp.reciprocal_to_cartesian(vector, self._mp_lattice)
    
    def lattice_to_cartesian(self, vector: mp.Vector3):
        return mp.lattice_to_cartesian(vector, self._mp_lattice)
    
    def cartesian_to_lattice(self, vector: mp.Vector3):
        return mp.cartesian_to_lattice(vector, self._mp_lattice)
    
    def lattice_to_reciprocal(self, vector: mp.Vector3):
        return mp.lattice_to_reciprocal(vector, self._mp_lattice)
    
    def reciprocal_to_lattice(self, vector: mp.Vector3):
        return mp.reciprocal_to_lattice(vector, self._mp_lattice)
    
  
    def get_high_symmetry_k_points(self, centered_in_gamma: bool = True)-> dict:
        path =  self._path_centered_in_gamma if centered_in_gamma else self._path_starting_in_gamma
        return path.to_dict()
    
    def get_k_points_around_gamma(self, distance: float)-> dict:
        Kx_cartesian = mp.Vector3(distance, 0, 0)
        Ky_cartesian = mp.Vector3(0, distance, 0)
        Kx_reciprocal = self.cartesian_to_reciprocal(Kx_cartesian)
        Ky_reciprocal = self.cartesian_to_reciprocal(Ky_cartesian)
        Gamma = mp.Vector3(0, 0, 0)
        k_path = KPath([Kx_reciprocal, Gamma, Ky_reciprocal], ["$k_x$", "$\Gamma", "$k_y$"])
        return k_path.to_dict()
        
    def _get_size_value(self,val):
        return val if val != 0 else "no-size"
    
    def _get_scheme_size(self)-> list:
        return [self._get_size_value(val) for val in self._size]
    
    def to_scheme(self) -> str:
        size = self._get_scheme_size()
        return f"(make lattice (size {size[0]} {size[1]} {size[2]}) " + \
               f"(basis1  (vector3 {self.basis1[0]} {self.basis1[1]} {self.basis1[2]})) " + \
               f"(basis2  (vector3 {self.basis2[0]} {self.basis2[1]} {self.basis2[2]})) " + \
                ")"
        

    @property
    def basis1(self):
        return self._mp_lattice.basis1
    
    @property
    def basis2(self):
        return self._mp_lattice.basis2
    
    @property
    def size(self):
        return self._mp_lattice.basis3
    

class SquareLattice(BravaisLattice):
    def __init__(self, supercell_height: Optional[float] = None):
        super().__init__(supercell_height)
        self._size = (1, 1, 0) if supercell_height is None else (1, 1, supercell_height)
        self._X = mp.Vector3(0.5, 0, 0)
        self._M = mp.Vector3(0.5, 0.5, 0)
        self._path_starting_in_gamma = KPath([self._G, self._X, self._M, self._G], 
                                                     ["$\Gamma$", "X", "M", "$\Gamma$"])
        
        self._path_centered_in_gamma = KPath([self._X, self._G, self._M, self._X],  
                                                             ["X", "$\Gamma$", "M", "X"])
        self._mp_lattice = self._make_lattice() 
                


    def _make_lattice(self) -> mp.Lattice:
        return mp.Lattice(size=self._size,
                            basis1=mp.Vector3(1, 0, 0), 
                            basis2=mp.Vector3(0, 1, 0))
    
    
    
class HexagonalLattice(BravaisLattice):
    def __init__(self, supercell_height: Optional[float] = None):
        super().__init__(supercell_height)
        self._size = (1, 1, 0) if supercell_height is None else (1, 1, supercell_height)
        self._M = mp.Vector3(0, 0.5, 0)
        self._K = mp.Vector3(-1/3, 1/3, 0)

        self._path_starting_in_gamma = KPath([self._G, self._K, self._M, self._G], ["$\Gamma$", "K", "M", "$\Gamma$"])
        self._path_centered_in_gamma = KPath([self._K, self._G, self._M, self._K], ["K", "$\Gamma$", "M", "K"])
        self._mp_lattice = self._make_lattice() 

    def _make_lattice(self) -> mp.Lattice:
        return mp.Lattice(size=self._size,
                            basis1=mp.Vector3(1, 0, 0), 
                            basis2=mp.Vector3(0.5, np.sqrt(3)/2, 0))   

    
        


class ObliqueLattice(BravaisLattice):
    def __init__(self, a1: tuple, a2: tuple, supercell_height: Optional[float] = None):
        super().__init__(supercell_height)
        self._size = (1, 1, 0) if supercell_height is None else (1, 1, supercell_height)
        self._M = mp.Vector3(0.5, 0.5, 0)
        self._K = mp.Vector3(0.5, 0, 0)
        self._path_starting_in_gamma = KPath([self._G, self._K, self._M, self._G], ["$\Gamma$", "K", "M", "$\Gamma$"])
        self._path_centered_in_gamma = KPath([self._K, self._G, self._M, self._K], ["K", "$\Gamma$", "M", "K"])
        
        self._a1 = a1 if type(a1) == tuple else  ValueError("a1 must be a tuple")
        self._a2 = a2 if type(a2) == tuple else  ValueError("a2 must be a tuple")
        
        self._mp_lattice = self._make_lattice() 

    def _make_lattice(self) -> mp.Lattice:
        return mp.Lattice(size=self._size,
                            basis1=mp.Vector3(self._a1[0], self._a1[0], self._a1[0]), 
                            basis2=mp.Vector3(self._a2[0], self._a2[0], self._a2[0]))
    

class RectangularLattice(BravaisLattice):
    def __init__(self, a1: int, a2: int, supercell_height: Optional[float] = None):
        super().__init__(supercell_height)
        self._size = (1, 1, 0) if supercell_height is None else (1, 1, supercell_height)
        self._X = mp.Vector3(0.5, 0, 0)
        self._Y = mp.Vector3(0, 0.5, 0)
        self._M = mp.Vector3(0.5, 0.5, 0)
        self._path_starting_in_gamma = KPath([self._G, self._X, self._M, self._Y, self._G], 
                                                     ["$\Gamma$", "X", "M", "Y", "$\Gamma$"])
        self._path_centered_in_gamma = KPath([self._X, self._G, self._M, self._Y, self._X],  
                                                             ["X", "$\Gamma$", "M", "Y", "X"])
        
        self._a1 = a1
        self._a2 = a2
        
        self._mp_lattice = self._make_lattice()

    def _make_lattice(self) -> mp.Lattice:
        return mp.Lattice(size=self._size,
                            basis1=mp.Vector3(self._a1, 0, 0), 
                            basis2=mp.Vector3(0, self._a2, 0))
    




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
    
        
    def build(self):
        if self.mp_geom_type:
            return self.mp_geom_type(**self.params)
        else:
            raise ValueError(f"Invalid geometry type: {self.mp_geom_type}, use one of meep geometry types")
    
    def collect_script_params(self):
        for param, value in self.params.items():
            if self.is_script_param(value):
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
    
 

class BaseDielectricDistribution:
    def __init__(self, eps_bulk = 3.1**2, eps_atoms = 1, eps_background = 1, 
                 radius1 = 0.1, radius2 = 0.2, 
                 height_slab = 1e20):
        self._material_background = Material(epsilon=eps_background)
        self._material_atoms = Material(epsilon=eps_atoms)
        self._material_bulk = Material(epsilon=eps_bulk)
        self._radius1 = ScriptParam(name="r1", default_value=radius1)
        self._radius2 = ScriptParam(name="r2", default_value=radius2)
        self._height = ScriptParam(name="h", default_value=height_slab)
        self._bulk_size = ScriptParamVector3(1, 1, "h", 1, 1, height_slab)
        self._bulk = Geometry(mp.Block, {"size": ScriptParamVector3(1, 1, "h", 1, 1, height_slab), 
                                         "center": mp.Vector3(), "material": self._material_bulk})
        

    def make_C6v_monoatomic(self):
        hole_1 = Geometry(mp.Cylinder, {"radius": self._radius1, "height": self._height, 
                                        "center": mp.Vector3(0, 0, 0), 
                                        "material": self._material_atoms})
        return [self._bulk, hole_1]
    
    def make_C6v_diatomic(self):
        hole_1 = Geometry(mp.Cylinder, {"radius": self._radius1, "height": self._height,
                                        "center": mp.Vector3(0, 0, 0), 
                                        "material": self._material_atoms})
        hole_2 = Geometry(mp.Cylinder, {"radius": self._radius2, "height": self._height,
                                        "center": mp.Vector3(1/3, 1/3, 0), 
                                        "material": self._material_atoms})
        hole_3 = Geometry(mp.Cylinder, {"radius": self._radius2, "height": self._height,
                                        "center": mp.Vector3(2/3, 2/3, 0), 
                                        "material": self._material_atoms})
        return [self._bulk, hole_1, hole_2, hole_3]
    

    def make_C3v_monoatomic(self):
        hole_1 = Geometry(mp.Cylinder, {"radius": self._radius1, "height": self._height,
                                        "center": mp.Vector3(0, 0, 0), 
                                        "material": self._material_atoms})
        
        hole_2 = Geometry(mp.Cylinder, {"radius": self._radius1, "height": self._height, 
                                        "center": mp.Vector3(1/3, 1/3, 0), 
                                        "material": self._material_atoms})
        return [self._bulk, hole_1, hole_2]
    
    def make_C3v_diatomic(self):
        hole_1 = Geometry(mp.Cylinder, {"radius": self._radius1, "height": self._height,
                                        "center": mp.Vector3(0, 0, 0), 
                                        "material": self._material_atoms})
        hole_2 = Geometry(mp.Cylinder, {"radius": self._radius2, "height": self._height,
                                        "center": mp.Vector3(1/3, 1/3, 0), 
                                        "material": self._material_atoms})
        return [self._bulk, hole_1, hole_2]
    

    def make_C4v_monoatomic(self):
        hole_1 = Geometry(mp.Cylinder, {"radius": self._radius1, "height": self._height,
                                        "center": mp.Vector3(0, 0, 0), 
                                        "material": self._material_atoms})
        
        return [self._bulk, hole_1]
    
    def make_C4v_diatomic(self):
        return self.make_C4v_diatomic_A()
    
    def make_C4v_diatomic_A(self):
        hole_1 = Geometry(mp.Cylinder, {"radius": self._radius1, "height": self._height,
                                        "center": mp.Vector3(0, 0, 0), 
                                        "material": self._material_atoms})
        
        hole_2 = Geometry(mp.Cylinder, {"radius": self._radius2, "height": self._height,
                                        "center": mp.Vector3(0, 1/2, 0), 
                                        "material": self._material_atoms})
        hole_3 = Geometry(mp.Cylinder, {"radius": self._radius2, "height": self._height,
                                        "center": mp.Vector3(1/2, 0, 0), 
                                        "material": self._material_atoms})
        return [self._bulk, hole_1, hole_2, hole_3]
    
    def make_C4v_diatomic_B(self):
        hole_1 = Geometry(mp.Cylinder, {"radius": self._radius1, "height": self._height,
                                        "center": mp.Vector3(0, 0, 0), 
                                        "material": self._material_atoms})
        hole_2 = Geometry(mp.Cylinder, {"radius": self._radius2, "height": self._height,
                                        "center": mp.Vector3(1/2, 1/2, 0), 
                                        "material": self._material_atoms})
        
        return [self._bulk, hole_1, hole_2]
    
    def make_C2v_monoatomic(self):
        hole_1 = Geometry(mp.Cylinder, {"radius": self._radius1, "height": self._height,
                                        "center": mp.Vector3(0, 0, 0), 
                                        "material": self._material_atoms})
        
        hole_2 = Geometry(mp.Cylinder, {"radius": self._radius1, "height": self._height,
                                        "center": mp.Vector3(1/2,0, 0), 
                                        "material": self._material_atoms})
        return [self._bulk, hole_1, hole_2]
    
    def make_C2v_diatomic(self):
        hole_1 = Geometry(mp.Cylinder, {"radius": self._radius1, "height": self._height,
                                        "center": mp.Vector3(0, 0, 0), 
                                        "material": self._material_atoms})
        hole_2 = Geometry(mp.Cylinder, {"radius": self._radius2, "height": self._height,
                                        "center": mp.Vector3(1/3, 1/3, 0), 
                                        "material": self._material_atoms})
        return [self._bulk, hole_1, hole_2]
    

class PhotonicCrystal:
    """
    This class is a wrapper for  some of mpb objects. It can be used to create photonics crystal objects and convert them to Scheme strings.
    """
    def __init__(self, atoms: list, lattice: BravaisLattice, background_material: Material = Material(epsilon=1)):
        """Create a photonic crystal object.

        Args:
            background_material (Material): The background material of the photonic crystal.
            atoms (list): A list of Geometry objects that represent the atoms in the photonic crystal.
            lattice (BravaisLattice): The lattice of the photonic crystal.
        """
        
        if all(isinstance(atom, Geometry) for atom in atoms):
            self._atoms = atoms
            self._geometry_group = GeometryGroup(*atoms)
            self._script_params = self._geometry_group.get_script_params()
        else:
            raise ValueError("All atoms must be of type Geometry")
        
        if isinstance(lattice, BravaisLattice):
            self._lattice = lattice
        else:
            raise ValueError("Lattice must be of type BravaisLattice") 
        
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

    pass


