from phc_nzi.lsf_job_submitter import LSFJob
import argparse
import os
from scipy.optimize import minimize_scalar, minimize
import pandas as pd
from phc_nzi.simulation_handler import Simulation, logging
import time
import tempfile
import shutil

class LSFScalarOptimizer(LSFJob):
    def __init__(self, 
                 simulation_name: str,
                 scheme_script: str,     
                 directory: str,
                 polarization: str,
                 bands: list,
                 param_name: str, 
                 param_bounds: tuple,
                 initial_guess: float = None,
                 others_command_line_params: dict = None,
                 method: str = "bounded",
                 maxiter: int = 20
                 ):
        super().__init__(simulation_name, scheme_script, directory)
        self.simulation_name = simulation_name
        self.directory = directory
        self.scheme_script = scheme_script
        self.polarization = polarization
        self.bands = bands
        self.param_name = param_name
        self.param_bounds = param_bounds
        self.initial_guess = initial_guess
        self.others_command_line_params = others_command_line_params or {}
        self.method = method
        self.maxiter = maxiter
        
        self.data_file = os.path.join(directory, f"{simulation_name}_opt_results.csv")
        self.history = []

    def temp_folder_operations(self, mpb_command_line_params):
        pass # Implemented directly in _objective to handle sequential creation

    def run(self):
        # Create CSV file with headers
        headers = [self.param_name, "cost"] + [f"band_{b}" for b in self.bands]
        with open(self.data_file, "w") as f:
            f.write(",".join(headers) + "\n")

        print(f"Starting {self.method} optimization for {self.param_name} in range {self.param_bounds}")
        
        if self.method.lower() in ["l-bfgs-b", "nelder-mead"]:
            x0 = self.initial_guess if self.initial_guess is not None else sum(self.param_bounds)/2.0
            bounds = [self.param_bounds] if self.method.lower() == "l-bfgs-b" else None
            res = minimize(
                self._objective_wrapper, 
                x0=[x0], 
                method=self.method, 
                bounds=bounds,
                options={"maxiter": self.maxiter, "disp": True}
            )
        else:
            # SciPy's bounded scalar method (Brent's method)
            res = minimize_scalar(
                self._objective_wrapper_scalar, 
                bounds=self.param_bounds, 
                method='bounded',
                options={"maxiter": self.maxiter, "disp": 3}
            )

        print("\nOptimization completed.")
        best_val = res.x[0] if isinstance(res.x, (list, tuple)) or hasattr(res.x, "__iter__") else res.x
        print(f"Best {self.param_name}: {best_val}")
        print(f"Minimum cost: {res.fun}")
        
        return res

    def _objective_wrapper(self, x):
        return self._objective(x[0])
        
    def _objective_wrapper_scalar(self, x):
        return self._objective(x)
        
    def _objective(self, param_value):
        # Convert scientific notation or numpy types back to float
        param_value = float(param_value)
        command_line_params = {self.param_name: param_value}
        command_line_params.update(self.others_command_line_params)
        
        if not self.scheme_script:
            raise ValueError(f"scheme_script is empty! The script was not loaded properly from {self.directory}")
            
        random_unique_id = os.urandom(4).hex()
        # Use a temporary directory that cleans up automatically
        with tempfile.TemporaryDirectory(dir=self.directory, prefix=f"temp_{random_unique_id}_") as tempdir:
            
            # Write control script
            script_path = os.path.join(tempdir, f"{random_unique_id}.ctl")
            with open(script_path, "w") as f:
                f.write(self.scheme_script)
                
            simulation = Simulation(
                random_unique_id, 
                self.scheme_script, 
                tempdir,
                write_script=True,  # script already written, but ensures handler expects it
                log_level=logging.WARNING,
            )

            try:
                freqs = self.workers_operations(
                    simulation,
                    command_line_params,
                    self.polarization,
                    self.bands
                )
                
                # Degeneracy -> cost is difference between largest and smallest band requested
                if len(self.bands) >= 2:
                    cost = abs(freqs[self.bands[1]] - freqs[self.bands[0]])
                else:
                    cost = 0.0
                    
                row = {
                    self.param_name: param_value,
                    'cost': cost
                }
                for b in self.bands:
                    row[f'band_{b}'] = freqs[b]
                    
                self.history.append(row)
                
                # Update CSV
                df = pd.DataFrame(self.history)
                df.to_csv(self.data_file, index=False)
                
                print(f"Evaluated {self.param_name}={param_value:.6f} -> cost={cost:.6e}")
                return cost
                
            except Exception as e:
                print(f"Simulation failed for {self.param_name}={param_value}: {e}")
                return 1e6 # Return a large penalty cost on failure

    def workers_operations(self, simulation: Simulation, mpb_command_line_params, polarization: str, bands: list):
        simulation.run_hpc(mpb_command_line_params)
        df = simulation.load_frequency_data(polarization)
        freqs = simulation.get_frequencies_by_band(df, polarization, bands=bands)
        return freqs
        
    def _get_script_abs_path(self):
        return os.path.abspath(__file__)

    def _prepare_command_line_args(self):
        cmd_args = [
            f"--simulation_name={self.simulation_name}",
            f"--directory={self.directory}",
            f"--param_to_opt={self.param_name}",
            f"--bounds={self.param_bounds[0]},{self.param_bounds[1]}"
        ]
        
        # Format the cmd_params dictionaries
        if self.others_command_line_params:
            cmd_args.append(f"--cmd_params_names={','.join(self.others_command_line_params.keys())}")
            cmd_args.append(f"--cmd_params_values={','.join(map(str, self.others_command_line_params.values()))}")
        else:
            cmd_args.append("--cmd_params_names=NONE")
            cmd_args.append("--cmd_params_values=NONE")

        cmd_args.extend([
            f"--polarization={self.polarization}",
            f"--bands={','.join(map(str, self.bands))}",
            f"--method={self.method}",
            f"--maxiter={self.maxiter}"
        ])
        
        if self.initial_guess is not None:
            cmd_args.append(f"--initial_guess={self.initial_guess}")
            
        return " ".join(cmd_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a scalar optimization on an MPB simulation")
    parser.add_argument("--run_opt", type=str, default="true", help="Whether to run the optimization")
    parser.add_argument("--simulation_name", type=str, required=True, help="Name of the simulation")
    parser.add_argument("--param_to_opt", type=str, required=True, help="Name of the parameter to optimize")
    parser.add_argument("--bounds", type=str, required=True, help="Comma-separated min,max bounds")
    parser.add_argument("--initial_guess", type=float, default=None, help="Initial guess within bounds")
    parser.add_argument("--cmd_params_names", type=str, default="NONE", help="Comma-separated list of command line parameter names")
    parser.add_argument("--cmd_params_values", type=str, default="NONE", help="Comma-separated list of command line parameter values")
    parser.add_argument("--polarization", type=str, required=True, help="Polarization to consider")
    parser.add_argument("--bands", type=str, required=True, help="Comma-separated list of bands to consider (e.g. '2,3')")
    parser.add_argument("--directory", type=str, required=True, help="Directory to store the simulation data")
    parser.add_argument("--method", type=str, default="bounded", help="Optimization method (bounded, L-BFGS-B)")
    parser.add_argument("--maxiter", type=int, default=20, help="Max iterations")
    
    args = parser.parse_args()

    run_opt = args.run_opt.lower() in ['true', '1', 'yes']
    try:
        bounds = tuple(map(float, args.bounds.split(',')))
    except ValueError:
        bounds = (0.0, 1.0) # Fallback

    bands = list(map(int, args.bands.split(',')))
    
    others_command_line_params = {}
    if args.cmd_params_names and args.cmd_params_names != "NONE" and args.cmd_params_values and args.cmd_params_values != "NONE":
        others_command_line_params = dict(zip(args.cmd_params_names.split(','), args.cmd_params_values.split(',')))

    # Fix cross-platform pathing issues (Windows submission -> Linux execution)
    args.directory = args.directory.replace('\\', '/')
    
    # Fix: simulation.ctl handling
    scheme_script_path = os.path.join(args.directory, args.simulation_name + ".ctl")
    
    # Wait for NFS sync if the file doesn't appear immediately (up to 30 seconds)
    import time
    for _ in range(15):
        if os.path.exists(scheme_script_path) and os.path.getsize(scheme_script_path) > 0:
            break
        print(f"Waiting for {scheme_script_path} to synchronize over NFS...")
        time.sleep(2)
        
    if os.path.exists(scheme_script_path):
        with open(scheme_script_path, "r") as f:
            scheme_script = f.read()
    else:
        scheme_script = ""
        print(f"WARNING: Could not find control file at {scheme_script_path} even after waiting!")

    optimizer = LSFScalarOptimizer(
        simulation_name=args.simulation_name,
        scheme_script=scheme_script,
        directory=args.directory,
        polarization=args.polarization,
        bands=bands,
        param_name=args.param_to_opt,
        param_bounds=bounds,
        initial_guess=args.initial_guess,
        others_command_line_params=others_command_line_params,
        method=args.method,
        maxiter=args.maxiter
    )

    if run_opt:
        optimizer.run()
    else:
        print("Optimization not executed. Set --run_opt to 'true' to run.")
