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

for k_factor in [10, 30]:
    sim_name = f"convergence_k_interp_2Db_f{k_factor}"
    sim = Simulation(simulation_name=sim_name, script="", directory=os.path.join(data_root, sim_name), write_script=False)
    analyzer = FieldAnalyzer(sim, [4, 6], "te", "x")
    df = analyzer.get_eps_mu_impedance_neff("y", "z", plot=False, enforce_continuity=True)
    
    b4_df = df[df["band"] == 4].head(5)
    print(f"\n--- k_factor={k_factor} Band 4 First 5 Rows ---")
    print(b4_df[["k_index", "frequency", "eps", "mu", "n_eff", "impedance"]])

