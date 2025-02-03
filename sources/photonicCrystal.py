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

    def __init__(self, type: str, size: tuple | int):
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
        
    def to_mp(self):
        return self._mp_lattice 
    
    def to_scheme(self):
        return f"(make lattice (size {self._size[0]} {self._size[1]}) " + \
               f"(basis1  (vector3 {self._mp_lattice.basis1[0]} {self._mp_lattice.basis1[1]} {self._mp_lattice.basis1[2]})) " + \
               f"(basis2  (vector3 {self._mp_lattice.basis2[0]} {self._mp_lattice.basis2[1]} {self._mp_lattice.basis2[2]})) " + \
                ")"
    
       



class Geometry:

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
            else:
                if isinstance(value, int):
                    value = value
                elif isinstance(value, float):
                    value = value
                elif isinstance(value, mp.Vector3):
                    value = f"(vector3 {value[0]} {value[1]} {value[2]})"
                else:
                    raise ValueError(f"Please check the value of {param}") 
            command = f"({param} {value})"
            commands.append(command)
        return commands

    def to_scheme(self):
        params_commands = self._params_to_scheme()

        command = f"(make {self.to_valid_scheme_geometry()} "
        for param in params_commands:
            command += param
        command += ")"
        return command
    

    def to_valid_scheme_geometry(self):
        if self.mp_geom_type in Geometry.valid_scheme_geometries:
            return Geometry.valid_scheme_geometries[self.mp_geom_type]
        else:
            raise ValueError(f"Invalid geometry type: {self.mp_geom_type}, use one of meep geometry types available in the python interface")
        
    
    

        

class Material: 
    def __init__(self, epsilon: float):
        self._epsilon = epsilon

    def to_scheme(self):
        return f"(make dielectric (epsilon {self._epsilon}))"
    
    @property
    def epsilon(self):
        return self._epsilon
    
    
    
    

    

class PhotonicCrystal:
    def __init__(self, atoms: list, lattice: Lattice, material: Material):

        if all(isinstance(atom, Geometry) for atom in atoms):
            self._atoms = atoms
        else:
            raise ValueError("All atoms must be of type Geometry")
        
        if isinstance(lattice, Lattice):
            self._lattice = lattice
        else:
            raise ValueError("Lattice must be of type Lattice") 
        
        if isinstance(material, Material):
            self._material = material
        else:
            raise ValueError("Material must be of type Material")  
        
        self._make_mpb_geometry()

        
        
    def _make_mpb_geometry(self):
        geometry = []
        for atom in self._atoms:
            geometry.append(atom.build())   

    