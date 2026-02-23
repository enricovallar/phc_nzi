import sys
import os
import pandas as pd
root = r"/zhome/2f/7/202918/phc_nzi"
src = r"/zhome/2f/7/202918/phc_nzi/src"
sys.path.append(root)
sys.path.append(src)
from phc_nzi.simulation_handler import Simulation
from phc_nzi.field_analyzer import FieldAnalyzer

data_root = "/work3/enrva/phc_nzi_data/MPB_data/"

sim_name = "convergence_k_interp_2Db_f30"
sim = Simulation(simulation_name=sim_name, script="", directory=os.path.join(data_root, sim_name), write_script=False)
analyzer = FieldAnalyzer(sim, [4, 6], "te", "x")
df_raw = analyzer.get_eps_mu_impedance_neff("y", "z", plot=False, enforce_continuity=False)

print("\n--- k_factor=30 Band 4 First 5 Rows RAW ---")
print(df_raw[df_raw["band"]==4].head(5)[['k_index', 'eps', 'mu', 'impedance']])

