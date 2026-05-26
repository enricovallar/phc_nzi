import os
import subprocess
import tempfile
import time
import re
import sqlite3
import logging
import time
from typing import Optional, Union, List, Dict, Any, Tuple
from phc_nzi.lsf_job_configurator import LSFJobConfiguration
import h5py
import numpy as np
import pandas as pd
import re

# Module-level logger configuration.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)




class MPBDataOptions:
    """Options for MPB data conversion."""
    
    # Define valid flags with their corresponding command-line arguments
    FLAG_MAP = {
        "rectify": "-r",
        "transpose": "-T",  
        "pixellized": "-p",
    }
    
    # Define options that require values with their flags
    VALUE_MAP = {
        "axis": "-e",
        "resolution": "-n",
        "phase": "-P",
        "dataset": "-d",
    }
    
    def __init__(self, 
                 rectify: bool = True, 
                 axis: Optional[int] = None, 
                 resolution: Optional[int] = None,
                 periods: Optional[Union[int, Tuple[int, ...], List[int]]] = (3,3,1),
                 phase: Optional[float] = None, 
                 transpose: bool = False, 
                 pixellized: bool = False,
                 dataset: Optional[str] = None) -> None:
        """
        Initialize MPB data conversion options.
        
        Args:
            rectify: Whether to rectify the data (default: True)
            axis: Axis to extract (optional)
            resolution: Resolution to use (optional)
            periods: Number of periods in each direction (optional)
                     Can be a single integer or a 2/3-element tuple/list
            phase: Phase to use for complex fields (optional)
            transpose: Whether to transpose the data (default: False)
            pixellized: Whether to use pixellized output (default: False)
            dataset: Dataset name to extract (optional)
        """
        self.rectify = rectify
        self.axis = axis
        self.resolution = resolution
        self.periods = periods
        self.phase = phase
        self.transpose = transpose
        self.pixellized = pixellized
        self.dataset = dataset
    
    def to_command_args(self) -> List[str]:
        """
        Convert options to command line arguments.
        
        Returns:
            List of command line arguments for mpb-data.
        """
        cmd = []
        
        # Add boolean flags
        for attr, flag in self.FLAG_MAP.items():
            if getattr(self, attr):
                cmd.append(flag)
        
        # Add value options
        for attr, flag in self.VALUE_MAP.items():
            value = getattr(self, attr)
            if value is not None:
                cmd.extend([flag, str(value)])
        
        # Handle periods specially due to multiple formats
        if self.periods is not None:
            if isinstance(self.periods, int):
                cmd.extend(["-m", str(self.periods)])
            elif isinstance(self.periods, (list, tuple)):
                if len(self.periods) >= 3:
                    cmd.extend(["-x", str(self.periods[0]), 
                              "-y", str(self.periods[1]), 
                              "-z", str(self.periods[2])])
                elif len(self.periods) == 2:
                    cmd.extend(["-x", str(self.periods[0]), 
                              "-y", str(self.periods[1])])
            
        return cmd


class MPBDataConverter:
    """Handles conversion of MPB data files using the mpb-data utility."""
    
    def __init__(self, input_file: str, output_file: str, options: Optional[MPBDataOptions] = None) -> None:
        """
        Initialize the converter with input/output files and options.
        
        Args:
            input_file: Path to input HDF5 file
            output_file: Path for output HDF5 file
            options: MPBDataOptions object with conversion parameters (default: empty options)
        """
        self.input_file = input_file
        self.output_file = output_file
        self.options = options or MPBDataOptions()
        self.logger = logger

    def build_command(self) -> List[str]:
        """
        Build the complete command for mpb-data conversion.
        
        Returns:
            List of command components to execute.
        """
        cmd = ["source /dtu/sw/dcc/dcc-sw.bash &&", 
               "module load mpb/1.11.1 &&", 
               "mpb-data"]
        
        # Add options
        cmd.extend(self.options.to_command_args())
        
        # Add input and output files
        cmd.extend(["-o", self.output_file, self.input_file])
        
        return cmd

    def run_conversion(self) -> str:
        """
        Execute the mpb-data conversion command.
        
        Raises:
            FileNotFoundError: If input file doesn't exist
            subprocess.CalledProcessError: If the command execution fails
        """
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
            
        cmd = self.build_command()
        full_cmd = " ".join(cmd)
        
        self.logger.debug("Running mpb-data conversion command: %s", full_cmd)
        
        result = subprocess.run(
            full_cmd, 
            shell=True, 
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            self.logger.error("MPB data conversion failed: %s", result.stderr)
            raise subprocess.CalledProcessError(result.returncode, full_cmd, 
                                              result.stdout, result.stderr)
            
        self.logger.debug("MPB data conversion completed successfully")
        self.logger.debug("Output file: %s",result.stdout)
        self.logger.debug("Output file: %s",result.stderr)

        return self.output_file


        
    
class Simulation:
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    def __init__(self, simulation_name: str, script: str, 
                 directory: Optional[str] = None, 
                 description: Optional[str] = "Photonic Crystal Simulation", 
                 log_level: int = logging.INFO, write_script = True) -> None:
        self.simulation_name = simulation_name
        self.directory = directory or simulation_name
        os.makedirs(self.directory, exist_ok=True)
        self.scheme_filename = f"{simulation_name}.ctl"
        self.output_filename = f"{simulation_name}.out"
        self.error_filename = f"{simulation_name}.err"
        self.script = script
        self.description = description
        self.epsilon = None
        self.lattice = None
        self.bands_df: Dict[str, pd.DataFrame] = {}
         
        if write_script is True:
            self._write_scheme_script()
        self._write_description()

        self.logger = logging.getLogger(f"{__name__}.{self.simulation_name}")
        self.logger.setLevel(log_level)

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'logger' in state:
            del state['logger']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = logging.getLogger(f"{__name__}.{self.simulation_name}")
        # Default level info if not found, though usually it wouldn't persist level in state unless stored separately
        self.logger.setLevel(logging.INFO)

    # Helper method for writing content to a file.
    def _write_to_file(self, filepath: str, content: str) -> None:
        with open(filepath, "w") as f:
            f.write(content)

    # Helper method for reading content from a file.    
    def _read_from_file(self, filepath: str) -> str:
        with open(filepath, "r") as f:
            return f.read()
        
    # Help method to check if a file have been correctly written.
    def _check_file_written(self, filepath: str) -> bool:
        return os.path.exists(filepath) and os.path.getsize(filepath) > 0

    # Write a given scheme script.
    def _write_scheme_script(self) -> None:
        if not self.script or not isinstance(self.script, str):
            return  # Do nothing if script is invalid
        filepath = os.path.join(self.directory, self.scheme_filename)
        os.makedirs(self.directory, exist_ok=True)
        self._write_to_file(filepath, self.script)
    
    def write_scheme_script(self) -> None: 
        self._write_scheme_script()

    # Read the scheme script. 
    def read_scheme_script(self) -> str:
        return self._read_from_file(os.path.join(self.directory, self.scheme_filename))
    
    # Check if the scheme script has been written.
    def check_scheme_script(self) -> bool:
        return self._check_file_written(os.path.join(self.directory, self.scheme_filename))
    
    # Write a description to a text file.
    def _write_description(self) -> None:
        if not self.description or not isinstance(self.description, str):
            return  # Do nothing if description is invalid
        self._write_to_file(os.path.join(self.directory, f"{self.simulation_name}.txt"), self.description)
    
    # Read the description from a text file.
    def read_description(self) -> str:
        return self._read_from_file(os.path.join(self.directory, f"{self.simulation_name}.txt"))
   
    # Write error message to a text file.
    def _write_error(self, error_message: str) -> None:
        if not error_message or not isinstance(error_message, str):
            error_message = "No error message."
        self._write_to_file(os.path.join(self.directory, f"{self.simulation_name}.err"), error_message)
    
    # Write output message to a text file.
    def _write_output(self, output_message: str) -> None:   
        if not output_message or not isinstance(output_message, str):
            output_message = "No output message."
        self._write_to_file(os.path.join(self.directory, f"{self.simulation_name}.out"), output_message)

    # A common method for executing commands.
    def _execute_command(self, cmd: str, shell: bool = False) -> subprocess.CompletedProcess:
        self.logger.debug("Executing command: %s", cmd)
        result = subprocess.run(cmd, shell=shell, capture_output=True,
                                text=True, cwd=self.directory)
        self._write_error(result.stderr)
        self._write_output(result.stdout)
        return result
    
    def _make_sure_scheme_script_exists(self) -> None:
        filepath = os.path.join(self.directory, self.scheme_filename)
        if self.check_scheme_script():
            return
        # First attempt: write the script
        self._write_scheme_script()
        if self.check_scheme_script():
            return
        # Retry with increasing delays for filesystem sync
        for i in range(10):
            self.logger.warning(
                "Scheme script not found at '%s' (dir exists: %s, script valid: %s, attempt %d/10). Retrying...",
                filepath, os.path.exists(self.directory), 
                bool(self.script and isinstance(self.script, str)), i + 1
            )
            self._write_scheme_script()
            time.sleep(0.5 * (i + 1))
            if self.check_scheme_script():
                return
        raise FileNotFoundError(
            f"Could not write the scheme script to '{filepath}'. "
            f"Directory exists: {os.path.exists(self.directory)}, "
            f"Script length: {len(self.script) if self.script else 0}"
        )
        
    def run_hpc(self, mpb_command_line_params: dict = {},
                load_epsilon: bool = True, extract_frequencies: bool = True, 
                mpi: bool = True, cores: int = 4, version: str = "mpb/1.11.1") -> None:
        
        self._make_sure_scheme_script_exists()
        params = " ".join(f"{k}={v}" for k, v in mpb_command_line_params.items())
        if mpi:
            mpb_cmd = f"mpirun -np {cores} mpb-mpi"
        else:
            mpb_cmd = "mpb"
            
        cmd = (
            f"source /dtu/sw/dcc/dcc-sw.bash && module load {version} && "
            f"{mpb_cmd} {params} {self.scheme_filename}"
        )
        
        self.logger.debug("Running HPC command: %s", cmd)
        self._execute_command(cmd, shell=True)
        self.logger.debug("Simulation completed")
        
        if load_epsilon:
            self.load_epsilon_data()
        if extract_frequencies:
            self.extract_frequencies()

    
    # Submit the LSF job using the given job script.
    def _submit_lsf_job(self, job_script_file: str) -> subprocess.CompletedProcess:
        cmd = (f"bsub -oo {self.output_filename} -eo {self.error_filename} "
               f"< {job_script_file}")
        self.logger.debug("Submitting LSF job with command: %s", cmd)
        return self._execute_command(cmd, shell=True)

    # Wait until the job has completed.
    def _wait_for_job_completion(self, job_id: str, initial_wait: int, poll_interval: int) -> None:
        self.logger.debug("Waiting %s seconds before first status check...", initial_wait)
        time.sleep(initial_wait)
        while True:
            status = subprocess.run("bstat", shell=True, capture_output=True, text=True)
            if "No unfinished job found" in status.stdout or job_id not in status.stdout:
                self.logger.debug("Job has finished.")
                break
            else:
                self.logger.debug("Job %s is still running. Waiting %s seconds...", job_id, poll_interval)
                time.sleep(poll_interval)

    # Wait for the simulation output files to appear.
    def _wait_for_output_files(self, timeout: int = 300, interval: int = 5) -> None:
        out_path = os.path.join(self.directory, self.output_filename)
        err_path = os.path.join(self.directory, self.error_filename)
        elapsed = 0
        while (((not os.path.exists(out_path) or os.path.getsize(out_path) == 0) or 
                (not os.path.exists(err_path) or os.path.getsize(err_path) == 0))
               and elapsed < timeout):
            self.logger.info("Waiting for simulation output files to be written...")
            time.sleep(interval)
            elapsed += interval
        if elapsed >= timeout:
            self.logger.warning("Output files not found or empty after waiting.")
        else:
            self.logger.debug("Output files are now available.")





    # Run the simulation in an HPC LSF environment.
    def run_hpc_lsf(self, LSFOptions: LSFJobConfiguration = LSFJobConfiguration(), 
                    initial_wait: int = 5, poll_interval: int = 5,
                    output_timeout: int = 300,
                    load_epsilon: bool = True, extract_frequencies: bool = True,
                    mpb_command_line_params: dict = {},
                    version: str = "mpb/1.11.1"
                    ) -> None:
        self._make_sure_scheme_script_exists()

        params = " ".join(f"{k}={v}" for k, v in mpb_command_line_params.items())
        preamble = LSFOptions.prepare_lsf_preamble(self.simulation_name)
        preamble.extend([
            f"#BSUB -oo {self.output_filename}",
            f"#BSUB -eo {self.error_filename}"
        ])
        job_script = "\n".join(preamble) + "\n\n"
        job_script += (
            f"source /dtu/sw/dcc/dcc-sw.bash && module load {version} && "
            f"mpirun -np $LSB_DJOB_NUMPROC mpb-mpi {params} {self.scheme_filename}"
        )
        job_script_filename = f"{self.simulation_name}.sh"
        job_script_path = os.path.join(self.directory, job_script_filename)
        with open(job_script_path, 'w') as job_file:
            job_file.write(job_script)
        time.sleep(0.1)
        result = self._submit_lsf_job(job_script_filename)
        job_id = self._extract_job_id(result.stdout)
        if job_id:
            self.logger.info("Job submitted with ID: %s", job_id)
            self._wait_for_job_completion(job_id, initial_wait, poll_interval)
        else:
            self.logger.warning("Could not determine job ID; proceeding without waiting.")
        self._wait_for_output_files(timeout=output_timeout, interval=5)
        self.logger.debug("Simulation completed")
        if load_epsilon:
            self.load_epsilon_data()
        if extract_frequencies:
            self.extract_frequencies()

    # Extract job ID from the bsub command output.
    def _extract_job_id(self, output: str) -> Optional[str]:
        match = re.search(r"<(\d+)>", output)
        return match.group(1) if match else None

    # Extract frequency data from the simulation output.
    def extract_frequencies(self, remove_line_prefixes: bool = True) -> None:
        prefixes = ["tmfreqs:", "tefreqs:", "zevenfreqs:", "zoddfreqs:", "gaps:", "freqs:"]
        output_path = os.path.join(self.directory, self.output_filename)
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output file {output_path} does not exist.")
        with open(output_path, "r") as f:
            lines = f.readlines()
        modes_data = self._extract_modes_data(lines, prefixes, remove_line_prefixes)
        for mode, data in modes_data.items():
            file_path = os.path.join(self.directory, f"{self.simulation_name}.{mode}.dat")
            self._write_to_file(file_path, "".join(data))
            self.logger.debug("Extracted %d lines of data for mode '%s'", len(data), mode)

    # Helper method to extract mode data from output lines.
    def _extract_modes_data(self, lines: List[str], prefixes: List[str], remove_prefix: bool) -> Dict[str, List[str]]:
        def strip_prefix(line: str) -> str:
            for prefix in prefixes:
                if line.startswith(prefix):
                    return line[len(prefix):].lstrip(" ,")
            return line

        modes = {
            "tm": [],
            "te": [],
            "zeven": [],
            "zodd": [],
            "gaps": [],
            "": [],
        }
        for line in lines:
            for mode in modes.keys():
                prefix = f"{mode}freqs:"
                if line.startswith(prefix):
                    modes[mode].append(strip_prefix(line) if remove_prefix else line)
        # Special handling for modes that may have alternate labels.
        modes["zodd"] = [strip_prefix(line) if remove_prefix else line for line in lines if "zoddfreqs:" in line]
        modes["zeven"] = [strip_prefix(line) if remove_prefix else line for line in lines if "zevenfreqs:" in line]
        modes["gaps"] = [strip_prefix(line) if remove_prefix else line for line in lines if "gaps:" in line]
        return modes

    # Load frequency data from a .dat file and store it in an SQLite database.
    def load_frequency_data(self, mode: str = "te") -> pd.DataFrame:
        filepath = os.path.join(self.directory, f"{self.simulation_name}.{mode}.dat")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found.")
        df = pd.read_csv(filepath, skipinitialspace=True)
        db_path = os.path.join(self.directory, f"{self.simulation_name}_frequencies.db")
        with sqlite3.connect(db_path) as conn:
            df.to_sql("frequencies", conn, if_exists="replace", index=False)
        self.bands_df[mode] = df
        self.logger.debug("Loaded frequency data for mode '%s'", mode)
        return df

    # Load an HDF5 file.
    def load_h5_data(self, filename: str) -> dict:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        with h5py.File(filename, 'r') as f:
            return {key: f[key][...] for key in f.keys()}

    # Run an mpb-data conversion.
    def _run_mpb_data_conversion(self, input_file: str, output_file: str, options: MPBDataOptions) -> None:
        converter = MPBDataConverter(input_file, output_file, options)
        return converter.run_conversion()
        

    # Convert field data using mpb-data.
    def find_field_data(self, 
                        k_idx: int, 
                        b_idx: int, 
                        polarization: str, 
                        field_type: str = "e", 
                        file_comp: Optional[str] = None, 
                        nonbloch: bool = False) -> str:
        field_label = f"{field_type}.v" if nonbloch is True else field_type
        comp_str = f".{file_comp}" if file_comp is not None else ""
        polarization_str = f".{polarization}" if polarization is not "" else ""
        filename = f"{self.simulation_name}-{field_label}.k{k_idx:02d}.b{b_idx:02d}{comp_str}{polarization_str}.h5"
        
        filepath = os.path.join(self.directory, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Field file not found: {filepath}")
            
        self.logger.debug(f"Found field data file: {filename}")
        return filepath
        
    def convert_field_data(self,
                           k_idx: int,
                           b_idx: int,
                           polarization: str,
                           field_type: str,
                           file_comp: Optional[str] = None, 
                           nonbloch: bool = False,
                           options: Optional[MPBDataOptions] = None,
                           overwrite: bool = False) -> str:
        # Find the input field file
        input_filepath = self.find_field_data(k_idx, b_idx, polarization, field_type, file_comp, nonbloch)
        input_filename = os.path.basename(input_filepath)
            
        # Create output filename and path
        output_filename = f"{os.path.splitext(input_filename)[0]}.converted.h5"
        output_filepath = os.path.join(self.directory, output_filename)
        
        # Run the conversion with provided or default options if file doesn't exist
        if overwrite or not os.path.exists(output_filepath):
            self._run_mpb_data_conversion(input_filepath, output_filepath, options or MPBDataOptions())
            self.logger.debug(f"Converted field data from {input_filename} to {output_filename}")
        else:
            self.logger.debug(f"Using cached field data: {output_filepath}")
        return output_filepath
    
    def convert_epsilon_data(self, options: Optional[MPBDataOptions] = None) -> str:
        # Check if input file exists
        input_filename = f"{self.simulation_name}-epsilon.h5"
        input_filepath = os.path.join(self.directory, input_filename)
        
        if not os.path.exists(input_filepath):
            raise FileNotFoundError(f"Epsilon file not found: {input_filepath}")
            
        # Create output path
        output_filename = f"{os.path.splitext(input_filename)[0]}.converted.h5"
        output_filepath = os.path.join(self.directory, output_filename)
        
        # Run conversion with provided or default options
        self._run_mpb_data_conversion(input_filepath, output_filepath, options or MPBDataOptions())
        
        self.logger.debug(f"Converted epsilon data to {output_filename}")
        return output_filepath
    

    def load_epsilon_data(self) -> np.ndarray:
        epsilon_filename = f"{self.simulation_name}-epsilon.h5"
        epsilon_filepath = os.path.join(self.directory, epsilon_filename)
        if not os.path.exists(epsilon_filepath):
            raise FileNotFoundError(f"Epsilon file not found: {epsilon_filepath}")
        with h5py.File(epsilon_filepath, 'r') as f:
            epsilon = f["data"][...]
        self.epsilon = epsilon
        self.logger.debug("Loaded epsilon data from %s", epsilon_filename)
        return epsilon
    
    def load_and_convert_epsilon_data(self, options: Optional[MPBDataOptions] = None) -> str:   
        converted_data_path = self.convert_epsilon_data(options)
        converted_data = self.load_h5_data(converted_data_path)
        if "data" not in converted_data:
            raise KeyError("Epsilon file must contain 'epsilon' dataset.")
        epsilon = converted_data.get("data")
        return epsilon
    
    
    # Load and convert field data, then reconstruct the complex field.
    def load_and_convert_field_data(self,
                                     k_idx: int,
                                     b_idx: int,
                                     component: str,
                                     polarization: str,
                                     field_type: str,
                                     conversion_options: Optional[MPBDataOptions] = None,
                                     file_comp: Optional[str] = None,
                                     nonbloch: bool = False,
                                     overwrite: bool = True) -> np.ndarray:
        converted_data_path = self.convert_field_data(k_idx, b_idx, polarization, field_type, file_comp, nonbloch, options=conversion_options, overwrite=overwrite)
        converted_data = self.load_h5_data(converted_data_path)
        if f"{component}.r" not in converted_data or f"{component}.i" not in converted_data:
            raise KeyError(f"Field file must contain '{component}.r' and '{component}.i'.")
        field_complex = converted_data[f"{component}.r"] + 1j * converted_data[f"{component}.i"]
        return field_complex

    def _find_index_of_closest_k_point(self, df: pd.DataFrame, target_k_point: Tuple[float, ...]) -> int:
        if not {"k1", "k2", "k3"}.issubset(df.columns):
            raise ValueError("The DataFrame must contain 'k1', 'k2', and 'k3' columns for k-point matching.")
        
        k_coords = df[["k1", "k2", "k3"]].values
        target = np.array(target_k_point)
        distances = np.linalg.norm(k_coords - target, axis=1)
        return np.argmin(distances)
    
    def _get_band_frequency(self, df: pd.DataFrame, k_point_index: int, polarization: str, b_idx: int) -> float:
        col_name = f"{polarization} band {b_idx}"
        col_name = col_name.strip()
        if col_name not in df.columns:
            self.logger.warning("Band %s not found in the DataFrame.", b_idx)
            return np.nan
        return df[col_name].iloc[k_point_index]
    
    def get_frequencies_by_band(self, df: pd.DataFrame, polarization: str = "te", 
                                bands: List[int] = [2, 3, 4], 
                                k_point: Tuple[float, float, float] = (0, 0, 0)) -> Dict[int, Any]:
        """
        Returns:
            Dictionary mapping band numbers to their frequencies
        """
        closest_idx = self._find_index_of_closest_k_point(df, k_point)
        
        return {
            band: self._get_band_frequency(df, closest_idx, polarization, band)
            for band in bands
        }

    # Find the closest k-point row in the DataFrame.
    def find_closest_k_point_row(self, df: pd.DataFrame, target_k_point: Tuple[float, ...]) -> pd.Series:
        idx = self._find_index_of_closest_k_point(df, target_k_point)
        return df.iloc[idx]

    # Property to get/set verbosity.
    @property 
    def verbosity(self) -> int:
        return self.logger.level    
    
    @verbosity.setter
    def verbosity(self, level: int) -> None:
        self.logger.setLevel(level)
        print("Log level set to", logging.getLevelName(level))

    def set_verbosity(self, level: int) -> None:
        if level not in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL):
            raise ValueError(f"Invalid verbosity level: {level}")
        self.verbosity = level
    
    # Get the k-point indices from a DataFrame.
    def get_kpoints_indices(self, df: pd.DataFrame) -> pd.Series:
        return df["k index"]
    
    
    def use_nested_temp_directory(func):
        def wrapper(self: Simulation, *args, **kwargs):
            main_directory = self.directory
            with tempfile.TemporaryDirectory() as temp_dir:
                self.directory = temp_dir 
                return func(self, *args, **kwargs)
            self.directory = main_directory
        return wrapper


    def get_freq(self, df, k_idx, b_idx, polarization ):
        row = self._get_row_at_k_idx(df, k_idx)
        return self._get_freq_from_df_row(row, b_idx, polarization)
    
    def _get_freq_from_df_row(self, row, b_idx, polarization):
        col_name = f"{polarization} band {b_idx}"
        return row[col_name.strip()].values[0]

    def get_group_velocity(self, k_idx: int, b_idx: int, polarization: str, direction: str = "x") -> float:
        """
        Extracts the group velocity for a given k-index and band from the MPB output log.
        """
        output_path = os.path.join(self.directory, self.output_filename)
        if not os.path.exists(output_path):
            self.logger.warning(f"Output file {output_path} does not exist.")
            return 0.0
        
        prefix = f"{polarization}velocity:"
        with open(output_path, "r") as f:
            for line in f:
                if line.startswith(prefix):
                    parts = [p.strip() for p in line.split(",")]
                    # parts[1] is the band index integer string
                    if len(parts) > 1 and parts[1] == str(b_idx):
                        # The velocities for each k-point start from parts[2]
                        # So k_idx = 1 maps to parts[2], k_idx = 2 maps to parts[3], etc.
                        part_index = k_idx + 1
                        if part_index < len(parts):
                            vec_str = parts[part_index].strip(" #()")
                            vec_parts = vec_str.split()
                            if direction == "x" and len(vec_parts) > 0:
                                return float(vec_parts[0])
                            elif direction == "y" and len(vec_parts) > 1:
                                return float(vec_parts[1])
                            elif direction == "z" and len(vec_parts) > 2:
                                return float(vec_parts[2])
        self.logger.warning(f"Group velocity not found for k_idx={k_idx}, b_idx={b_idx}. Ensure `display-group-velocities` is in run commands.")
        return 0.0
    

    def get_kmag(self, df, k_idx):
        row = self._get_row_at_k_idx(df, k_idx)
        return self._get_kmag_from_df_row(row)
    
    def _get_kmag_from_df_row(self, row):
        return row["kmag/2pi"].values[0]
    
    def _get_row_at_k_idx(self, df, k_idx):
        row = df[df["k index"] == k_idx]
        if row.empty:
            raise ValueError(f"k index {k_idx} not found in the dataframe")
        return row
    
    def get_kmag_and_freq(self, df, k_idx, b_idx, polarization):
        row = self._get_row_at_k_idx(df, k_idx)
        freq = self._get_freq_from_df_row(row, b_idx, polarization)
        kmag = self._get_kmag_from_df_row(row)
        return kmag, freq
    
    def get_kpoints_idices(self, df):
        return df["k index"]
    

    def get_symmetry_by_parity(self, target_parity):
        """
        Uses re.search to find a specific parity block (e.g., 'te' or 'tm').
        """
        pattern = rf"SYM_DATA_START_{target_parity}\n(.*?)\nSYM_DATA_END_{target_parity}"
        file_path = os.path.join(self.directory, self.output_filename)
        with open(file_path, "r") as f:
            output_text = f.read()
        try:
            match = re.search(pattern, output_text, re.DOTALL)
        except re.error as e:
            print(f"Error: Invalid regex pattern for parity '{target_parity}': {e}")
            return None
        if not match:
            print(f"Error: Block for parity '{target_parity}' not found.")
            return None

        data_lines = match.group(1).strip().split("\n")
        results = {}
        
        for line in data_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3: continue
            
            band_num = int(parts[1])
            chars = {}
            for op_val in parts[2:]:
                op, val = op_val.split("=")
                val_clean = val.replace(" + ", "+").replace(" - ", "-").replace("i", "j")
                chars[op] = complex(val_clean)
                
            results[band_num] = chars
            
        return results


    def compute_projections(self, chars, group="C6v"):
        """
        Computes the projection of a mode onto each irreducible representation.
        Returns 1.0 if the mode purely belongs to that irrep.
        """
        tables = {
            "C6v": {
                "g": 12,
                "weights": {"E": 1, "C6": 2, "C3": 2, "C2": 1, "sv": 3, "sd": 3},
                "irreps": {
                    "A_1": {"E": 1, "C6": 1,  "C3": 1,  "C2": 1,  "sv": 1,  "sd": 1},
                    "A_2": {"E": 1, "C6": 1,  "C3": 1,  "C2": 1,  "sv": -1, "sd": -1},
                    "B_1": {"E": 1, "C6": -1, "C3": 1,  "C2": -1, "sv": 1,  "sd": -1},
                    "B_2": {"E": 1, "C6": -1, "C3": 1,  "C2": -1, "sv": -1, "sd": 1},
                    "E_1": {"E": 2, "C6": 1,  "C3": -1, "C2": -2, "sv": 0,  "sd": 0},
                    "E_2": {"E": 2, "C6": -1, "C3": -1, "C2": 2,  "sv": 0,  "sd": 0},
                }
            },
            "C4v": {
                "g": 8,
                "weights": {"E": 1, "C4": 2, "C2": 1, "sv": 2, "sd": 2},
                "irreps": {
                    "A_1": {"E": 1, "C4": 1,  "C2": 1,  "sv": 1,  "sd": 1},
                    "A_2": {"E": 1, "C4": 1,  "C2": 1,  "sv": -1, "sd": -1},
                    "B_1": {"E": 1, "C4": -1, "C2": 1,  "sv": 1,  "sd": -1},
                    "B_2": {"E": 1, "C4": -1, "C2": 1,  "sv": -1, "sd": 1},
                    "E":  {"E": 2, "C4": 0,  "C2": -2, "sv": 0,  "sd": 0},
                }
            }
        }

        if group not in tables:
            return {}

        group_data = tables[group]
        weights = group_data["weights"]
        g = group_data["g"]

        # Observed Identity E is always 1.0 for a single band
        full_obs = chars.copy()
        full_obs["E"] = 1.0 + 0j

        projections = {}
        for irrep, irrep_chars in group_data["irreps"].items():
            d_i = irrep_chars["E"]
            scalar_product = 0
            for op, w in weights.items():
                if op in full_obs:
                    scalar_product += w * np.conj(irrep_chars[op]) * full_obs[op]
            
            projections[irrep] = (d_i * scalar_product / g).real

        return projections

    def identify_irrep(self, projections):
        """
        Identifies the irrep strictly by finding the max projection value.
        Returns only the string label.
        """
        if not projections:
            return "Unknown"
        return max(projections, key=projections.get)

    def identify_irrep_by_band_indices(self, which_bands: list, which_parity: str, group: str) -> list:
        """
        Identifies the irrep by checking which bands have a projection of 1.0 for the specified parity.
         - which_bands: List of band indices to check (e.g., [2, 3, 4])
         - which_parity: Parity to check (e.g., 'te' or 'tm')
         - group: The point group to use for projection calculations (e.g., "C6v" or "C4v")
            Returns a list of identified irreps for the specified bands and parity.
        """

        symmetry_data = self.get_symmetry_by_parity(which_parity)
        if not symmetry_data:
            return [None] * len(which_bands)
        # Get group from symmetry data keys (e.g., "C6v" or "C4v") and compute projections for each band

        identified_irreps = []
        for band in which_bands:
            chars = symmetry_data.get(band)
            if chars is None:
                identified_irreps.append(None)
                continue
            projections = self.compute_projections(chars, group=group)
            irrep = self.identify_irrep(projections)
            identified_irreps.append(irrep)
        return identified_irreps
        

    def identify_irrep_with_confidence(self, projections):
        """
        Identifies the irrep by finding the max projection value.
        Returns a tuple of (string label, confidence value).
        """
        if not projections:
            return "Unknown", 0.0
        
        best_irrep = max(projections, key=projections.get)
        confidence = projections[best_irrep]
        return best_irrep, confidence

    def identify_irrep_by_band_indices_with_confidence(self, which_bands: list, which_parity: str, group: str) -> list:
        """
        Identifies the irrep by checking which bands have the highest projection for the specified parity.
         - which_bands: List of band indices to check (e.g., [2, 3, 4])
         - which_parity: Parity to check (e.g., 'te' or 'tm')
         - group: The point group to use for projection calculations (e.g., "C6v" or "C4v")
         
        Returns a list of tuples containing (irrep_label, confidence) for the specified bands and parity.
        """
        symmetry_data = self.get_symmetry_by_parity(which_parity)
        if not symmetry_data:
            # Return a list of (None, 0.0) if symmetry data is entirely missing
            return [(None, 0.0)] * len(which_bands)

        identified_irreps_with_conf = []
        for band in which_bands:
            chars = symmetry_data.get(band)
            if chars is None:
                identified_irreps_with_conf.append((None, 0.0))
                continue
                
            projections = self.compute_projections(chars, group=group)
            irrep, confidence = self.identify_irrep_with_confidence(projections)
            identified_irreps_with_conf.append((irrep, confidence))
            
        return identified_irreps_with_conf