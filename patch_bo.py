import re

with open("src/phc_nzi/mpi_bayesian_optimization.py", "r") as f:
    text = f.read()

# 1. Update __init__ signature
init_target = """                 objective_mode: str = "linear",
                 target_cost: float = None,
                 strategy: str = "cl_min"):"""
init_replace = """                 objective_mode: str = "linear",
                 target_cost: float = None,
                 strategy: str = "cl_min",
                 degeneracy_tol: float = None):"""
text = text.replace(init_target, init_replace)

# 2. Update __init__ body
init_body_target = """        self.objective_mode = objective_mode
        self.target_cost = target_cost
        self.strategy = strategy"""
init_body_replace = """        self.objective_mode = objective_mode
        self.target_cost = target_cost
        self.strategy = strategy
        self.degeneracy_tol = degeneracy_tol"""
text = text.replace(init_body_target, init_body_replace)

# 3. Update argparse
argparse_target = """    parser.add_argument("--strategy", type=str, default="cl_min",
                        help="Strategy to use for ask() in Bayesian Optimization.")
    
    parser.add_argument("--bands", type=int, nargs='+', help="Bands to optimize (e.g., 1 2 3)")"""
argparse_replace = """    parser.add_argument("--strategy", type=str, default="cl_min",
                        help="Strategy to use for ask() in Bayesian Optimization.")
    parser.add_argument("--degeneracy_tol", type=float, default=None,
                        help="Tolerance for frequency-proximity degeneracy failsafe (e.g. 1e-3).")
    
    parser.add_argument("--bands", type=int, nargs='+', help="Bands to optimize (e.g., 1 2 3)")"""
text = text.replace(argparse_target, argparse_replace)

# 4. Pass parameter to object
obj_call_target = """                                           target_cost=args.target_cost,
                                           strategy=args.strategy)"""
obj_call_replace = """                                           target_cost=args.target_cost,
                                           strategy=args.strategy,
                                           degeneracy_tol=args.degeneracy_tol)"""
text = text.replace(obj_call_target, obj_call_replace)

# 5. Add the failsafe to _find_bands_from_irreps
find_target = """        identified_irreps = self.simulation.identify_irrep_by_band_indices(
            which_bands=bands_to_check, 
            which_parity=self.polarization, 
            group=self.symmetry_group
        ) 
        
        # Create a full mapping for the log file: {band: "Irrep"}"""
find_replace = """        identified_irreps = self.simulation.identify_irrep_by_band_indices(
            which_bands=bands_to_check, 
            which_parity=self.polarization, 
            group=self.symmetry_group
        ) 
        
        # --- FAILSAFE: Frequency-Proximity Degeneracy Sanity Check ---
        # When perfect degeneracy occurs, MPB returns linear combinations of eigenfunctions
        # resulting in misidentified 'irreps'. We patch these local misidentifications 
        # by checking frequency distances at Gamma.
        if getattr(self, 'degeneracy_tol', None) is not None:
            try:
                df = self.simulation.load_frequency_data(self.polarization)
                # Ensure we capture Gamma point frequencies
                gamma_cond = (df["k1"] == 0) & (df["k2"] == 0)
                if gamma_cond.any():
                    gamma_idx = df.loc[gamma_cond, "k index"].values[0]
                    freqs = {b: df.loc[df["k index"] == gamma_idx, f"{self.polarization} band {b}"].values[0] for b in bands_to_check}
                    
                    # 1. Cluster bands by frequency proximity
                    clusters = []
                    current_cluster = [bands_to_check[0]]
                    for b in bands_to_check[1:]:
                        if abs(freqs[b] - freqs[current_cluster[0]]) < self.degeneracy_tol:
                            current_cluster.append(b)
                        else:
                            clusters.append(current_cluster)
                            current_cluster = [b]
                    clusters.append(current_cluster)
                    
                    # 2. Re-label clusters that match dimensions & features of target multiplet
                    req_modes = len(self.target_irreps)
                    expected_labels_set = set([i for i in self.target_irreps if i])
                    
                    for cluster in clusters:
                        if len(cluster) == req_modes:
                            cluster_labels = set([identified_irreps[bands_to_check.index(b)] for b in cluster if identified_irreps[bands_to_check.index(b)]])
                            # If the cluster shares at least some irrep labels with our target
                            if expected_labels_set.intersection(cluster_labels):
                                # Overwrite the misidentified labels to match the pristine multiplet pattern
                                for b, target_label in zip(cluster, self.target_irreps):
                                    identified_irreps[bands_to_check.index(b)] = target_label
            except Exception as e:
                pass # Silently fallback to strict logic
                
        # Create a full mapping for the log file: {band: "Irrep"}"""
text = text.replace(find_target, find_replace)

with open("src/phc_nzi/mpi_bayesian_optimization.py", "w") as f:
    f.write(text)
