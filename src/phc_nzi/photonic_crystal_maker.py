
from abc import ABC, abstractmethod
import meep as mp
from  typing import Optional
import numpy as np
import warnings


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
        k_path = KPath([Kx_reciprocal, Gamma, Ky_reciprocal], ["$k_x$", "$\Gamma$", "$k_y$"])
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
    def __init__(self, x=1, y=1, z=1, x_def=1, y_def=1, z_def=1):
        self._names = []
        self._default_values = []
        
        # Helper to parse and validate each component
        def parse_comp(comp, default_val, label):
            if isinstance(comp, ScriptParam):
                # Inherit the definitions of the nested ScriptParam
                self._names.extend(comp._names)
                self._default_values.extend(comp._default_values)
                return str(comp)  # Returns the math expression (e.g., "(* -1 x_val)")
            elif isinstance(comp, str):
                # Treat pure strings as new Scheme parameter names
                if not isinstance(default_val, (int, float)):
                    raise ValueError(f"Default value for {label} must be an int or float")
                self._names.append(comp)
                self._default_values.append(default_val)
                return comp
            elif isinstance(comp, (int, float)):
                return str(comp)
            else:
                raise ValueError(f"{label} must be a ScriptParam, string, or int/float")

        # Parse all three coordinates
        x_str = parse_comp(x, x_def, "x")
        y_str = parse_comp(y, y_def, "y")
        z_str = parse_comp(z, z_def, "z")

        # Build the final Scheme vector string
        self._scheme_string = f"(vector3 {x_str} {y_str} {z_str})"


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
            elif isinstance(value, list):
                vec_strings = []
                for v in value:
                    if isinstance(v, mp.Vector3):
                        vec_strings.append(f"(vector3 {v.x} {v.y} {v.z})")
                    elif self.is_script_param_vector3(v):
                        vec_strings.append(str(v))
                    else:
                        raise ValueError("List elements must be mp.Vector3 or ScriptParamVector3")
                value = "(list " + " ".join(vec_strings) + ")"
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
    def __init__(self, epsilon: float | int| tuple ):
        self._epsilon = epsilon
        

    def to_scheme(self):
        if isinstance(self._epsilon, float) or isinstance(self._epsilon, int):
            str = f"(make dielectric (epsilon {self._epsilon}))"
        elif isinstance(self._epsilon, tuple):
            str = f"(make dielectric-anisotropic (epsilon-diag (vector3 {self._epsilon[0]} {self._epsilon[1]} {self._epsilon[2]})))"
        else :
            raise ValueError(f"Invalid epsilon value: {self._epsilon}, must be float or tuple")
        return str
    
    def to_python(self):
        return mp.Medium(epsilon=self._epsilon) 
    
    @property
    def epsilon(self):
        return self._epsilon
    
class MaterialFunction(Material):
    def __init__(self, name, definition):
        self.name = name
        self.definition = definition # The Scheme code block
        
    def to_scheme(self):
        # Tells Meep/MPB to use a function
        return f"(make material-function (material-func {self.name}))"

class BaseDielectricDistribution:
    def __init__(self, eps_bulk = 3.1**2, eps_atoms = 1, eps_background = 1, 
                 radius1 = 0.1, radius2 = 0.2, 
                 height_slab = 1e20,
                 cz_sphere = 0, 
              ):
        self._material_background = Material(epsilon=eps_background)
        self._material_atoms = Material(epsilon=eps_atoms)
        self._material_bulk = Material(epsilon=eps_bulk)
        self._radius1 = ScriptParam(name="r1", default_value=radius1)
        self._radius2 = ScriptParam(name="r2", default_value=radius2)
        self._height = ScriptParam(name="h", default_value=height_slab)
        self._sphere_center_1 = ScriptParamVector3(0, 0, "cz_sphere", 0, 0, cz_sphere)
        self._sphere_center_2 = ScriptParamVector3(1/2, 0, "cz_sphere", 1/2, 0, cz_sphere)
        self._sphere_center_3 = ScriptParamVector3(0, 1/2, "cz_sphere", 0, 1/2, cz_sphere)
        self._bulk_size = ScriptParamVector3(1, 1, "h", 1, 1, height_slab)
        self._bulk = Geometry(mp.Block, {"size": ScriptParamVector3(1, 1, "h", 1, 1, height_slab), 
                                         "center": mp.Vector3(), "material": self._material_bulk})


    def _scale_coord(self, coord, factor):
        """Safely scales a coordinate for both numeric types and ScriptParams."""
        if factor == 1:
            return coord
            
        if isinstance(coord, ScriptParam):
            # Create proxy param inheriting names/defaults from the original
            scaled = ScriptParam(coord._names[0], coord._default_values[0])
            scaled._scheme_string = f"(* {factor} {str(coord)})"
            return scaled
            
        return coord * factor

    def _get_center_vector(self, u, v, w=0):
        """Returns the appropriate Vector3 object based on the inputs."""
        if any(isinstance(val, ScriptParam) for val in (u, v, w)):
            return ScriptParamVector3(u, v, w)
        return mp.Vector3(u, v, w)
    
    def make_1a_daisy(self, r0=0.35, rd=0.08, m=6):
        # Define the Scheme function definition
        # Note: We use the r0, rd, m parameters directly in the Scheme string
        daisy_def = f"""
(define (daisy-dielectric p)
  (let* ((cart-p (lattice->cartesian p))
         (x (vector3-x cart-p))
         (y (vector3-y cart-p))
         (z (vector3-z cart-p))
         (r (sqrt (+ (* x x) (* y y))))
         (phi (atan y x))
         (r-boundary (+ {r0} (* {rd} (cos (* {m} phi))))))
    (if (and (< (abs z) (/ h 2)) (< r r-boundary))
        (make dielectric (epsilon {self._material_atoms.epsilon}))
        (make dielectric (epsilon {self._material_bulk.epsilon})))))
"""
        daisy_mat = MaterialFunction("daisy-dielectric", daisy_def)
        
        # The block fills the whole unit cell; the material function logic "carves" the hole
        daisy_block = Geometry(mp.Block, {
            "size": self._bulk_size, 
            "center": mp.Vector3(0, 0, 0), 
            "material": daisy_mat
        })
        
        return [daisy_block]

    def make_C6v_1a(self, radius=None):
        """1a Wyckoff position for C6v (Origin). 1 atom."""
        r = radius if radius is not None else self._radius1
        hole = Geometry(mp.Cylinder, {"radius": r, "height": self._height, 
                                      "center": mp.Vector3(0, 0, 0), 
                                      "material": self._material_atoms})
        return [hole]

    def make_C6v_2b(self, radius=None):
        """2b Wyckoff position for C6v (Honeycomb). 2 atoms."""
        r = radius if radius is not None else self._radius2
        coords = [(1/3, 1/3), (2/3, 2/3)]
        return [Geometry(mp.Cylinder, {"radius": r, "height": self._height, 
                                       "center": mp.Vector3(u, v, 0), 
                                       "material": self._material_atoms}) for u, v in coords]

    def make_C6v_3c(self, radius=None):
        """3c Wyckoff position for C6v (Kagome). 3 atoms."""
        r = radius if radius is not None else self._radius1
        coords = [(1/2, 0), (0, 1/2), (1/2, 1/2)]
        return [Geometry(mp.Cylinder, {"radius": r, "height": self._height, 
                                       "center": mp.Vector3(u, v, 0), 
                                       "material": self._material_atoms}) for u, v in coords]

    def make_C6v_6d(self, radius=None, x_dist=0.25):
        """6d Wyckoff position for C6v (Primary Hexamer). 6 atoms on mirror axes."""
        r = radius if radius is not None else self._radius1
        
        # Define scaled variables modularly
        x = x_dist
        neg_x = self._scale_coord(x, -1)
        
        coords = [
            (x, 0), 
            (0, x), 
            (neg_x, x), 
            (neg_x, 0), 
            (0, neg_x), 
            (x, neg_x)
        ]
        
        return [
            Geometry(mp.Cylinder, {
                "radius": r, 
                "height": self._height, 
                "center": self._get_center_vector(u, v, 0), 
                "material": self._material_atoms
            }) for u, v in coords
        ]

    def make_C6v_6e(self, radius=None, x_dist=0.15):
        """6e Wyckoff position for C6v (Secondary Hexamer). 6 atoms off mirror axes."""
        r = radius if radius is not None else self._radius1
        
        # Define scaled variables modularly
        x = x_dist
        neg_x = self._scale_coord(x, -1)
        two_x = self._scale_coord(x, 2)
        neg_two_x = self._scale_coord(x, -2)
        
        coords = [
            (x, x), 
            (neg_x, two_x), 
            (neg_two_x, x), 
            (neg_x, neg_x), 
            (x, neg_two_x), 
            (two_x, neg_x)
        ]
        
        return [
            Geometry(mp.Cylinder, {
                "radius": r, 
                "height": self._height, 
                "center": self._get_center_vector(u, v, 0), 
                "material": self._material_atoms
            }) for u, v in coords
        ]

    # ==============================================================================
    # Standard C4v (Square / p4mm) Wyckoff Positions
    # ==============================================================================

    def make_C4v_1a(self, radius=None):
        """1a Wyckoff position for C4v (Origin). 1 atom."""
        r = radius if radius is not None else self._radius1
        hole = Geometry(mp.Cylinder, {"radius": r, "height": self._height, 
                                      "center": mp.Vector3(0, 0, 0), 
                                      "material": self._material_atoms})
        return [hole]

    def make_C4v_1b(self, radius=None):
        """1b Wyckoff position for C4v (Center). 1 atom."""
        r = radius if radius is not None else self._radius2
        hole = Geometry(mp.Cylinder, {"radius": r, "height": self._height, 
                                      "center": mp.Vector3(1/2, 1/2, 0), 
                                      "material": self._material_atoms})
        return [hole]

    def make_C4v_2c(self, radius=None):
        """2c Wyckoff position for C4v (Edges). 2 atoms."""
        r = radius if radius is not None else self._radius2
        coords = [(1/2, 0), (0, 1/2)]
        return [Geometry(mp.Cylinder, {"radius": r, "height": self._height, 
                                       "center": mp.Vector3(u, v, 0), 
                                       "material": self._material_atoms}) for u, v in coords]

    def make_C4v_4d(self, radius=None, x_dist=0.25):
        """4d Wyckoff position for C4v (Diagonals). 4 atoms."""
        r = radius if radius is not None else self._radius2
        x = x_dist
        neg_x = self._scale_coord(x, -1)
        coords = [(x, x), (neg_x, x), (neg_x, neg_x), (x, neg_x)]
        return [Geometry(mp.Cylinder, {"radius": r, "height": self._height, 
                                       "center": mp.Vector3(u, v, 0), 
                                       "material": self._material_atoms}) for u, v in coords]

    def make_C4v_4e(self, radius=None, x_dist=0.25):
        """4e Wyckoff position for C4v (Axes). 4 atoms."""
        r = radius if radius is not None else self._radius2
        x = x_dist
        neg_x = self._scale_coord(x, -1)
        coords = [(x, 0), (0, x), (neg_x, 0), (0, neg_x)]
        return [Geometry(mp.Cylinder, {"radius": r, "height": self._height, 
                                       "center": mp.Vector3(u, v, 0), 
                                       "material": self._material_atoms}) for u, v in coords]

    # ----------------------------------------------------------------------
    # Linear Superposition Method
    # ----------------------------------------------------------------------

    def make_superposition(self, wyckoff_lists, include_bulk=True):
        """
        Combines multiple lists of Wyckoff geometries into a single list.
        
        Args:
            wyckoff_lists (list of lists): E.g. [make_C6v_1a(), make_C6v_6d(x_dist=0.3)]
            include_bulk (bool): If True, prepends self._bulk to the returned list.
        
        Returns:
            list: A flat list of Geometry objects ready for PhotonicCrystal.
        """
        geometries = [self._bulk] if include_bulk else []
        for w_list in wyckoff_lists:
            geometries.extend(w_list)
        return geometries
    

    @property
    def material_background(self):
        return self._material_background
    
    @property
    def material_atoms(self):
        return self._material_atoms
    
    @property
    def material_bulk(self):
        return self._material_bulk
    

    @material_background.setter
    def material_background(self, value):
        if isinstance(value, Material):
            self._material_background = value
        else:
            raise ValueError("Material must be of type Material")
        
    @material_atoms.setter
    def material_atoms(self, value):
        if isinstance(value, Material):
            self._material_atoms = value
        else:
            raise ValueError("Material must be of type Material")
    
    @material_bulk.setter
    def material_bulk(self, value):
        if isinstance(value, Material):
            self._material_bulk = value
        else:
            raise ValueError("Material must be of type Material")
        


    

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
        
    def _get_material_function_defs(self) -> str:
        defs = []
        for atom in self._atoms:
            mat = atom.params.get("material")
            if isinstance(mat, MaterialFunction):
                defs.append(mat.definition)
        return "\n".join(set(defs))
    def to_scheme(self) -> str:
        """Convert the photonic crystal to a Scheme string.
        
        Returns:
            str: A Scheme string that defines the photonic crystal. 
        """
        mat_defs = self._get_material_function_defs()

        commands_partial = self.to_scheme_list()
        command_lattice = f"(set! geometry-lattice {commands_partial[0]})"
        command_geometry = f"(set! geometry {commands_partial[1]})"
        
        command_script_params  = self._script_params_to_scheme_string()
        return "\n".join([command_script_params, mat_defs, command_lattice, command_geometry])
    

    def print_script_params(self):
        print(self._script_params)
    
    
        
#example usage of the classes
if __name__ == "__main__":

    pass


