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

    def __init__(self, type: str, size: tuple | int = (1, 1, "no-size")):
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
            raise ValueError(f"Invalid lattice type: {self._type}")
        
    
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
        return f"(make lattice (size {self._size[0]} {self._size[1]} {self._size[2]})  " + \
               f"(basis1  (vector3 {self._mp_lattice.basis1[0]} {self._mp_lattice.basis1[1]} {self._mp_lattice.basis1[2]})) " + \
               f"(basis2  (vector3 {self._mp_lattice.basis2[0]} {self._mp_lattice.basis2[1]} {self._mp_lattice.basis2[2]})) " + \
                ")"
    
       



class Geometry:
    """ This class is a wrapper for meep geometry objects. It can be used to create meep geometry objects and convert them to Scheme strings. """
    valid_scheme_geometries = {
        mp.Cylinder: "cylinder",
        mp.Sphere: "sphere",
        mp.Block: "block",
        mp.Prism: "prism",
        mp.Ellipsoid: "ellipsoid",
    }

    def __init__(self, geom_type, params: dict):   
        self.mp_geom_type = geom_type
        self.params = params

    def build(self):
        if self.mp_geom_type:
            return self.mp_geom_type(**self.params)
        else:
            raise ValueError(f"Invalid geometry type: {self.mp_geom_type}, use one of meep geometry types")
    
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
    

    def to_valid_scheme_geometry_definition(self):
        if self.mp_geom_type in Geometry.valid_scheme_geometries:
            return Geometry.valid_scheme_geometries[self.mp_geom_type]
        else:
            raise ValueError(f"Invalid geometry type: {self.mp_geom_type}, use one of meep geometry types available in the python interface")
        
    def to_python(self):
        return self.build()
    
        
class GeometryGroup:
    def __init__(self, *geometries):
        self._geometries = geometries


    def to_scheme(self):
        commands = []
        for geometry in self._geometries:
            commands.append(geometry.to_scheme())
        commands_string =  "\n".join(commands)
        commands_string = f"(list {commands_string}\n)"
        return commands_string



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
    
    def to_scheme(self) -> str:
        """Convert the photonic crystal to a Scheme string.
        
        Returns:
            str: A Scheme string that defines the photonic crystal. 
        """
        commands_partial = self.to_scheme_list()
        command_lattice = f"(set! geometry-lattice {commands_partial[0]})"
        command_geometry = f"(set! geometry {commands_partial[1]})"
        return "\n".join([command_lattice, command_geometry])
    


#example usage of the classes
if __name__ == "__main__":
    atom_geometry = Geometry(mp.Cylinder, {"radius": 0.2, "height": 0.5, "center": mp.Vector3(0, 0, 0), "material": Material(epsilon=12)})
    lattice = Lattice("SX", (1, 1, Lattice.NO_SIZE))
    material = Material(epsilon=12)
    photonic_crystal = PhotonicCrystal([atom_geometry], lattice)
    print(photonic_crystal.to_scheme())


