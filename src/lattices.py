
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
    
  
    def get_high_symmetry_k_points(self, centered_in_gamma: bool = True)-> KPath:
        path =  self._path_centered_in_gamma if centered_in_gamma else self._path_starting_in_gamma
        return path.to_dict()
    
    def get_k_points_around_gamma(self, distance: float):
        Kx_cartesian = mp.Vector3(distance, 0, 0)
        Ky_cartesian = mp.Vector3(0, distance, 0)
        Kx_reciprocal = self.cartesian_to_reciprocal(Kx_cartesian)
        Ky_reciprocal = self.cartesian_to_reciprocal(Ky_cartesian)
        Gamma = mp.Vector3(0, 0, 0)
        k_path = KPath([Kx_reciprocal, Gamma, Ky_reciprocal], ["$k_x$", "$\Gamma", "$k_y$"])
        return k_path.to_dict()
        
    def _get_size_value(val):
        return val if val != 0 else "no-size"
    
    def _get_scheme_size(self)-> list:
        return [self._get_size_value(val) for val in self._size]
    
    def to_scheme(self) -> str:
        size = self._get_scheme_size()
        return f"(make lattice (size {size[0]} {size[1]} {size[2]}) " + \
               f"(basis1  (vector3 {self.basis1[0]} {self.basis1[1]} {self.basis1[2]})) " + \
               f"(basis2  (vector3 {self.basis2[0]} {self.basis2[1]} {self.basis2[2]})) " + \
                ")"
    
    @abstractmethod
    def get_centers(self):
        pass

    @property
    def basis1(self):
        return self._mp_lattice.basis1
    
    @property
    def basis2(self):
        return self._mp_lattice.basis2
    
    @property
    def size(self):
        return self._mp_lattice.basis3
    

class TetragonalLattice(BravaisLattice):
    def __init__(self, supercell_height: Optional[int] = None):
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
    
    
    def get_centers(self):
        centers = [
            mp.Vector3(0, 0),
            mp.Vector3(0.5, 0.5)
        ]
        return centers
    
class TrigonalLattice(BravaisLattice):
    def __init__(self, supercell_height: Optional[int] = None):
        super().__init__(supercell_height)
        self._size = (1, 1, 0) if supercell_height is None else (1, 1, supercell_height)
        self._M = mp.Vector3(0, 0.5, 0)
        self._K = mp.Vector3(-1/3, 1/3, 0)

        self._path_starting_in_gamma = KPath([self._G, self._K, self._M, self._G]),
        self._path_centered_in_gamma = KPath([self._K, self._G, self._M, self._K])
        self._mp_lattice = self._make_lattice() 

    def _make_lattice(self) -> mp.Lattice:
        return mp.Lattice(size=self._size,
                            basis1=mp.Vector3(1, 0, 0), 
                            basis2=mp.Vector3(0.5, 1/np.sqrt(3), 0))   

    def get_centers(self): 
        centers = [
            mp.Vector3(0, 0),
            mp.Vector3(1/3, 1/3),
            mp.Vector3(2/3, 2/3)    
        ]
        return centers 
        
class HexagonalLattice(TrigonalLattice):
    def get_centers(self):
        centers = [  
            mp.Vector3(0, 0), 
        ]
        return centers

class ObliqueLattice(BravaisLattice):
    def __init__(self, supercell_height: Optional[int] = None):
        super().__init__(supercell_height)
        self._size = (1, 1, 0) if supercell_height is None else (1, 1, supercell_height)
        self._M = mp.Vector3(0.5, 0.5, 0)
        self._K = mp.Vector3(0.5, 0, 0)
        self._path_starting_in_gamma = KPath([self._G, self._K, self._M, self._G]),
        self._path_centered_in_gamma = KPath([self._K, self._G, self._M, self._K])
        self._mp_lattice = self._make_lattice() 

    def _make_lattice(self) -> mp.Lattice:
        return mp.Lattice(size=self._size,
                            basis1=mp.Vector3(1, 0, 0), 
                            basis2=mp.Vector3(0.5, 1, 0))   

    def get_centers(self):
        centers = [
            mp.Vector3(0, 0),
            mp.Vector3(0.5, 0.5)
        ]
        return centers

        



    

    


    