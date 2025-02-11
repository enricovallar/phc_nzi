from photonic_crystal_maker import PhotonicCrystal, Geometry, Lattice, Material
from mpb_configurator import MPBSchemeConfigurator
from simulation_handler import Simulation, SimulationViewer

class Phc_optimizer:
    def __init__(self, phc, configurator_options): 
        self._phc
        self._scheme_configurator = MPBSchemeConfigurator(phc, **configurator_options)


    def print_configuration_options(self):
        print(MPBSchemeConfigurator.__dict__())
        


    



