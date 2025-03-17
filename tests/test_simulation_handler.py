import os
import sys
import pytest
import tempfile
import h5py
import numpy as np
import pandas as pd
import subprocess
import logging
from unittest.mock import patch, MagicMock, mock_open

from src.simulation_handler import (
    Simulation,
    MPBDataOptions,
    MPBDataConverter,
    LSFJobConfiguration
)


def test_simulation_initialization():
    """Test the basic initialization of a Simulation object"""
    with tempfile.TemporaryDirectory() as temp_dir:
        name = "test_sim"
        script = "(define-param radius 0.3)\n(run)"
        description = "Test simulation"
        
        sim = Simulation(name, script, temp_dir, description)
        
        # Check that files were created
        assert os.path.exists(os.path.join(temp_dir, f"{name}.ctl"))
        assert os.path.exists(os.path.join(temp_dir, f"{name}.txt"))
        
        # Check that content was written correctly
        with open(os.path.join(temp_dir, f"{name}.ctl"), 'r') as f:
            assert f.read() == script
        with open(os.path.join(temp_dir, f"{name}.txt"), 'r') as f:
            assert f.read() == description


def test_read_scheme_script():
    """Test reading the scheme script from file"""
    with tempfile.TemporaryDirectory() as temp_dir:
        name = "test_sim"
        script = "(define-param radius 0.3)\n(run)"
        
        sim = Simulation(name, script, temp_dir)
        read_script = sim.read_scheme_script()
        
        assert read_script == script


def test_check_scheme_script():
    """Test checking if the scheme script exists"""
    with tempfile.TemporaryDirectory() as temp_dir:
        name = "test_sim"
        script = "(define-param radius 0.3)\n(run)"
        
        sim = Simulation(name, script, temp_dir)
        assert sim.check_scheme_script() == True


def test_read_description():
    """Test reading the description from file"""
    with tempfile.TemporaryDirectory() as temp_dir:
        name = "test_sim"
        script = "(define-param radius 0.3)\n(run)"
        description = "Test simulation description"
        
        sim = Simulation(name, script, temp_dir, description)
        read_description = sim.read_description()
        
        assert read_description == description


@patch('subprocess.run')
def test_execute_command(mock_run):
    """Test the _execute_command method"""
    mock_result = MagicMock()
    mock_result.stdout = "Command output"
    mock_result.stderr = "Command error"
    mock_run.return_value = mock_result
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        cmd = "echo test"
        result = sim._execute_command(cmd, shell=True)
        
        # Check that subprocess.run was called with correct arguments
        mock_run.assert_called_once_with(
            cmd, shell=True, capture_output=True, text=True, cwd=temp_dir
        )
        
        # Check that output and error were written to files
        output_path = os.path.join(temp_dir, "test_sim.output.txt")
        error_path = os.path.join(temp_dir, "test_sim.error.txt")
        
        with open(output_path, 'r') as f:
            assert f.read() == "Command output"
        with open(error_path, 'r') as f:
            assert f.read() == "Command error"


@patch('subprocess.run')
@patch('time.sleep')  # Mock sleep to avoid actual delays
def test_make_sure_scheme_script_exists(mock_sleep, mock_run):
    """Test the _make_sure_scheme_script_exists method"""
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        # Remove the script file to force retry
        os.remove(os.path.join(temp_dir, "test_sim.ctl"))
        
        # Mock check_scheme_script to return False first then True
        with patch.object(sim, 'check_scheme_script', side_effect=[False, True]):
            sim._make_sure_scheme_script_exists()
            
            # Should attempt to write the script and wait once
            mock_sleep.assert_called_once()


@patch('subprocess.run')
@patch('time.sleep')  # Mock sleep to avoid actual delays
def test_make_sure_scheme_script_exists_failure(mock_sleep, mock_run):
    """Test the _make_sure_scheme_script_exists method when it fails"""
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        # Remove the script file to force retry
        os.remove(os.path.join(temp_dir, "test_sim.ctl"))
        
        # Mock check_scheme_script to always return False
        with patch.object(sim, 'check_scheme_script', return_value=False):
            with pytest.raises(FileNotFoundError):
                sim._make_sure_scheme_script_exists()
            
            # Should attempt to write 5 times
            assert mock_sleep.call_count == 5


@patch('subprocess.run')
def test_run_hpc(mock_run):
    """Test the run_hpc method"""
    mock_result = MagicMock()
    mock_run.return_value = mock_result
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        # Mock methods that would be called by run_hpc
        with patch.object(sim, 'load_epsilon_data') as mock_load_epsilon_data, \
             patch.object(sim, 'extract_frequencies') as mock_extract_frequencies:
            
            sim.run_hpc({"num-bands": 8})
            
            # Check that subprocess.run was called with correct command
            cmd = "source /dtu/sw/dcc/dcc-sw.bash && module load mpb/1.11.1 && mpb-mpi num-bands=8 test_sim.ctl"
            mock_run.assert_called_once_with(
                cmd, shell=True, capture_output=True, text=True, cwd=temp_dir
            )
            
            # Check that post-processing methods were called
            mock_load_epsilon_data.assert_called_once()
            mock_extract_frequencies.assert_called_once()


@patch('subprocess.run')
def test_submit_lsf_job(mock_run):
    """Test the _submit_lsf_job method"""
    mock_result = MagicMock()
    mock_result.stdout = "Job <123456> is submitted to queue <fotonano>."
    mock_run.return_value = mock_result
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        job_script = os.path.join(temp_dir, "job_script.sh")
        with open(job_script, 'w') as f:
            f.write("#!/bin/bash\necho test")
        
        result = sim._submit_lsf_job(job_script)
        
        # Check that subprocess.run was called with correct command
        expected_cmd = f"bsub -oo test_sim.out -eo test_sim.err < {job_script}"
        mock_run.assert_called_once_with(
            expected_cmd, shell=True, capture_output=True, text=True, cwd=temp_dir
        )


def test_extract_job_id():
    """Test the _extract_job_id method"""
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        # Test valid job ID extraction
        output = "Job <123456> is submitted to queue <fotonano>."
        job_id = sim._extract_job_id(output)
        assert job_id == "123456"
        
        # Test invalid format
        output = "Invalid format"
        job_id = sim._extract_job_id(output)
        assert job_id is None


@patch('subprocess.run')
@patch('time.sleep')  # Mock sleep to avoid actual delays
def test_wait_for_job_completion(mock_sleep, mock_run):
    """Test the _wait_for_job_completion method"""
    # Mock subprocess.run to return job running first, then job completed
    mock_run.side_effect = [
        MagicMock(stdout="JOBID   USER    STAT  QUEUE    FROM_HOST"),
        MagicMock(stdout="123456  user1   RUN   fotonano node001"),
        MagicMock(stdout="No unfinished job found")
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        sim._wait_for_job_completion("123456", 1, 2)
        
        # Should check job status twice and wait once
        assert mock_run.call_count == 3
        assert mock_sleep.call_count == 2


@patch('os.path.exists')
@patch('os.path.getsize')
@patch('time.sleep')  # Mock sleep to avoid actual delays
def test_wait_for_output_files(mock_sleep, mock_getsize, mock_exists):
    """Test the _wait_for_output_files method"""
    # Mock file existence and size checks
    mock_exists.side_effect = [True, False, True, True]  # out exists, err doesn't, then both exist
    mock_getsize.side_effect = [10, 0, 10, 5]  # out has size, err zero, then both have size
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        sim._wait_for_output_files(timeout=10, interval=2)
        
        # Should check files twice
        assert mock_exists.call_count == 4
        assert mock_getsize.call_count == 4
        assert mock_sleep.call_count == 1


@patch('subprocess.run')
@patch('time.sleep')  # Mock sleep to avoid actual delays
def test_run_hpc_lsf(mock_sleep, mock_run):
    """Test the run_hpc_lsf method"""
    # Mock the job submission and status checks
    mock_run.side_effect = [
        MagicMock(stdout="Job <123456> is submitted to queue <fotonano>."),
        MagicMock(stdout="No unfinished job found")
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        # Mock methods that would be called by run_hpc_lsf
        with patch.object(sim, 'load_epsilon_data') as mock_load_epsilon_data, \
             patch.object(sim, '_wait_for_output_files') as mock_wait_files, \
             patch.object(sim, 'extract_frequencies') as mock_extract_frequencies:
            
            options = LSFJobConfiguration(num_processors=4, walltime="1:00")
            sim.run_hpc_lsf(options, initial_wait=1, poll_interval=1, 
                           output_timeout=5, mpb_command_line_params={"num-bands": 8})
            
            # Check that a job script was created
            job_script_path = os.path.join(temp_dir, "test_sim.sh")
            assert os.path.exists(job_script_path)
            
            # Check that post-processing methods were called
            mock_wait_files.assert_called_once()
            mock_load_epsilon_data.assert_called_once()
            mock_extract_frequencies.assert_called_once()


def test_extract_frequencies():
    """Test the extract_frequencies method"""
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        # Create a mock output file with frequency data
        output_content = """
        Some header information
        tefreqs: 0.1, 0.2, 0.3
        tefreqs: 0.4, 0.5, 0.6
        tmfreqs: 0.7, 0.8, 0.9
        gaps: 10%, 20%, between bands 1 and 2
        zevenfreqs: 0.11, 0.22
        zoddfreqs: 0.33, 0.44
        More output lines
        """
        
        output_path = os.path.join(temp_dir, "test_sim.out")
        with open(output_path, 'w') as f:
            f.write(output_content)
        
        sim.extract_frequencies()
        
        # Check that frequency files were created
        for mode in ["te", "tm", "zeven", "zodd", "gaps"]:
            mode_path = os.path.join(temp_dir, f"test_sim.{mode}.dat")
            assert os.path.exists(mode_path)
        
        # Check content of te frequencies file
        with open(os.path.join(temp_dir, "test_sim.te.dat"), 'r') as f:
            content = f.read()
            assert "0.1, 0.2, 0.3" in content
            assert "0.4, 0.5, 0.6" in content


@patch('pandas.read_csv')
@patch('sqlite3.connect')
def test_load_frequency_data(mock_connect, mock_read_csv):
    """Test the load_frequency_data method"""
    # Mock DataFrame and connection
    mock_df = pd.DataFrame({'k1': [0, 1], 'k2': [0, 0], 'te band 1': [0.1, 0.2]})
    mock_read_csv.return_value = mock_df
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        # Create a mock frequency data file
        freq_path = os.path.join(temp_dir, "test_sim.te.dat")
        with open(freq_path, 'w') as f:
            f.write("k1, k2, te band 1\n0, 0, 0.1\n1, 0, 0.2\n")
        
        df = sim.load_frequency_data("te")
        
        # Check that the correct file was read
        mock_read_csv.assert_called_once_with(freq_path, skipinitialspace=True)
        
        # Check that data was stored in the database
        mock_df.to_sql.assert_called_once_with("frequencies", mock_conn, if_exists="replace", index=False)
        
        # Check that data was stored in the object
        assert "te" in sim.bands_df
        assert sim.bands_df["te"] is mock_df


@patch('h5py.File')
def test_load_h5_data(mock_h5py_file):
    """Test the load_h5_data method"""
    # Mock HDF5 file with datasets
    mock_file = MagicMock()
    mock_dataset1 = MagicMock()
    mock_dataset1.__getitem__.return_value = np.array([1, 2, 3])
    mock_dataset2 = MagicMock()
    mock_dataset2.__getitem__.return_value = np.array([[4, 5], [6, 7]])
    
    mock_file.keys.return_value = ['dataset1', 'dataset2']
    mock_file.__getitem__.side_effect = lambda key: mock_dataset1 if key == 'dataset1' else mock_dataset2
    mock_h5py_file.return_value.__enter__.return_value = mock_file
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        # Create a dummy h5 file path
        h5_path = os.path.join(temp_dir, "test_data.h5")
        with open(h5_path, 'w') as f:
            f.write("dummy")
        
        data = sim.load_h5_data(h5_path)
        
        # Check that h5py.File was called with the correct path
        mock_h5py_file.assert_called_once_with(h5_path, 'r')
        
        # Check that the returned data has the expected keys
        assert 'dataset1' in data
        assert 'dataset2' in data


@patch.object(MPBDataConverter, 'run_conversion')
def test_run_mpb_data_conversion(mock_run_conversion):
    """Test the _run_mpb_data_conversion method"""
    mock_run_conversion.return_value = "output.h5"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        input_file = "input.h5"
        output_file = "output.h5"
        options = MPBDataOptions(rectify=True, resolution=32)
        
        result = sim._run_mpb_data_conversion(input_file, output_file, options)
        
        # Check that MPBDataConverter was used correctly
        mock_run_conversion.assert_called_once()
        assert result == "output.h5"


@patch.object(Simulation, 'find_field_data')
@patch.object(Simulation, '_run_mpb_data_conversion')
def test_convert_field_data(mock_run_conversion, mock_find_field_data):
    """Test the convert_field_data method"""
    mock_find_field_data.return_value = "/path/to/input.h5"
    mock_run_conversion.return_value = "/path/to/output.h5"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        result = sim.convert_field_data(
            k_idx=1, 
            b_idx=2, 
            polarization="te", 
            field_type="e", 
            options=MPBDataOptions(resolution=32)
        )
        
        # Check that field data was located and converted
        mock_find_field_data.assert_called_once_with(1, 2, "te", "e", None, True)
        mock_run_conversion.assert_called_once()
        assert result == "/path/to/output.h5"


@patch('os.path.exists')
def test_convert_epsilon_data(mock_exists):
    """Test the convert_epsilon_data method"""
    mock_exists.return_value = True
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        with patch.object(sim, '_run_mpb_data_conversion', return_value="/path/to/output.h5") as mock_convert:
            result = sim.convert_epsilon_data(MPBDataOptions(resolution=32))
            
            # Check that the correct input file was used
            input_path = os.path.join(temp_dir, "test_sim-epsilon.h5")
            output_path = os.path.join(temp_dir, "test_sim-epsilon.converted.h5")
            mock_convert.assert_called_once()
            assert result == "/path/to/output.h5"


@patch.object(Simulation, 'convert_epsilon_data')
@patch.object(Simulation, 'load_h5_data')
def test_load_epsilon_data(mock_load_h5, mock_convert_epsilon):
    """Test the load_epsilon method"""
    mock_convert_epsilon.return_value = "/path/to/epsilon.h5"
    mock_load_h5.return_value = {"epsilon.r": np.array([[1.0, 2.0], [3.0, 4.0]])}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        result = sim.load_epsilon_data(MPBDataOptions(resolution=32))
        
        # Check that epsilon was converted and loaded
        mock_convert_epsilon.assert_called_once_with(MPBDataOptions(resolution=32))
        mock_load_h5.assert_called_once_with("/path/to/epsilon.h5")
        assert "epsilon.r" in result


@patch.object(Simulation, 'convert_field_data')
@patch.object(Simulation, 'load_h5_data')
def test_load_and_convert_field_data(mock_load_h5, mock_convert_field):
    """Test the load_and_convert_field_data method"""
    mock_convert_field.return_value = "/path/to/field.h5"
    mock_load_h5.return_value = {
        "ex.r": np.array([1.0, 2.0]), 
        "ex.i": np.array([0.1, 0.2])
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        field_complex, path = sim.load_and_convert_field_data(
            k_idx=1,
            b_idx=2,
            component="ex",
            polarization="te",
            field_type="e",
            conversion_options=MPBDataOptions(resolution=32),
        )
        
        # Check that field was converted and loaded
        mock_convert_field.assert_called_once()
        mock_load_h5.assert_called_once_with("/path/to/field.h5")
        
        # Check that complex field was reconstructed correctly
        np.testing.assert_array_almost_equal(
            field_complex, np.array([1.0 + 0.1j, 2.0 + 0.2j])
        )


def test_find_index_of_closest_k_point():
    """Test the _find_index_of_closest_k_point method"""
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        df = pd.DataFrame({
            'k1': [0.0, 0.5, 1.0],
            'k2': [0.0, 0.0, 0.0],
            'k3': [0.0, 0.0, 0.0],
            'te band 1': [0.1, 0.2, 0.3]
        })
        
        # Test finding the closest k-point
        idx = sim._find_index_of_closest_k_point(df, (0.48, 0.02, 0.0))
        assert idx == 1  # Second row (0.5, 0.0, 0.0) is closest
        
        # Test with DataFrame missing k columns
        invalid_df = pd.DataFrame({'k1': [0.0], 'te band 1': [0.1]})
        with pytest.raises(ValueError):
            sim._find_index_of_closest_k_point(invalid_df, (0.0, 0.0, 0.0))


def test_get_band_frequency():
    """Test the _get_band_frequency method"""
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        df = pd.DataFrame({
            'k1': [0.0, 0.5, 1.0],
            'k2': [0.0, 0.0, 0.0],
            'k3': [0.0, 0.0, 0.0],
            'te band 1': [0.1, 0.2, 0.3],
            'te band 2': [0.4, 0.5, 0.6]
        })
        
        # Test getting a band frequency at a specific k-point index
        freq = sim._get_band_frequency(df, 1, "te", 2)
        assert freq == 0.5  # Value from 'te band 2' at index 1
        
        # Test with invalid band
        freq = sim._get_band_frequency(df, 1, "te", 3)
        assert np.isnan(freq)


def test_get_frequencies_by_band():
    """Test the get_frequencies_by_band method"""
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        df = pd.DataFrame({
            'k1': [0.0, 0.5, 1.0],
            'k2': [0.0, 0.0, 0.0],
            'k3': [0.0, 0.0, 0.0],
            'te band 1': [0.1, 0.2, 0.3],
            'te band 2': [0.4, 0.5, 0.6],
            'te band 3': [0.7, 0.8, 0.9]
        })
        
        # Store the DataFrame in the simulation object
        sim.bands_df["te"] = df
        
        # Test getting frequencies for multiple bands at a specific k-point
        freqs = sim.get_frequencies_by_band(df, "te", [1, 2, 3], (0.52, 0.0, 0.0))
        assert freqs[1] == 0.2  # Closest to (0.5, 0.0, 0.0)
        assert freqs[2] == 0.5
        assert freqs[3] == 0.8


def test_find_closest_k_point_row():
    """Test the find_closest_k_point_row method"""
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        df = pd.DataFrame({
            'k1': [0.0, 0.5, 1.0],
            'k2': [0.0, 0.0, 0.0],
            'k3': [0.0, 0.0, 0.0],
            'te band 1': [0.1, 0.2, 0.3]
        })
        
        # Test finding the row for the closest k-point
        row = sim.find_closest_k_point_row(df, (0.48, 0.02, 0.0))
        assert row['k1'] == 0.5
        assert row['te band 1'] == 0.2


def test_set_verbosity():
    """Test the set_verbosity method"""
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        # Test setting valid verbosity levels
        sim.set_verbosity(logging.DEBUG)
        assert sim.verbosity == logging.DEBUG
        
        sim.set_verbosity(logging.WARNING)
        assert sim.verbosity == logging.WARNING
        
        # Test with invalid level
        with pytest.raises(ValueError):
            sim.set_verbosity(999)


def test_get_kpoints_indices():
    """Test the get_kpoints_indices method"""
    with tempfile.TemporaryDirectory() as temp_dir:
        sim = Simulation("test_sim", "(run)", temp_dir)
        
        df = pd.DataFrame({
            'k index': [0, 1, 2],
            'k1': [0.0, 0.5, 1.0],
            'te band 1': [0.1, 0.2, 0.3]
        })
        
        indices = sim.get_kpoints_indices(df)
        assert list(indices) == [0, 1, 2]


def test_mpb_data_options():
    """Test the MPBDataOptions class"""
    # Test default initialization
    options = MPBDataOptions()
    assert options.rectify == True
    assert options.axis is None
    assert options.resolution is None
    assert options.periods == (3, 3, 1)
    
    # Test custom initialization
    options = MPBDataOptions(
        rectify=False, 
        axis=2, 
        resolution=32, 
        periods=(2, 2, 0), 
        phase=0.5
    )
    assert options.rectify == False
    assert options.axis == 2
    assert options.resolution == 32
    assert options.periods == (2, 2, 0)
    assert options.phase == 0.5
    
    # Test to_command_args
    cmd_args = options.to_command_args()
    assert "-e" in cmd_args
    assert "2" in cmd_args
    assert "-n" in cmd_args
    assert "32" in cmd_args
    assert "-x" in cmd_args
    assert "2" in cmd_args
    assert "-P" in cmd_args
    assert "0.5" in cmd_args
    assert "-r" not in cmd_args  # rectify is False
    
    # Test with single integer periods
    options = MPBDataOptions(periods=2)
    cmd_args = options.to_command_args()
    assert "-m" in cmd_args
    assert "2" in cmd_args


def test_lsf_job_configuration():
    """Test the LSFJobConfiguration class"""
    # Test default initialization
    config = LSFJobConfiguration()
    assert config.queue == "fotonano"
    assert config.num_processors == 10
    assert config.walltime == "24:00"
    assert config.mem == "4GB"
    assert config.span_option == "hosts"
    assert config.span_value == 1
    
    # Test custom initialization
    config = LSFJobConfiguration(
        queue="hpc", 
        num_processors=4, 
        walltime="1:00", 
        mem="2GB",
        span_option="ptile",
        span_value=2
    )
    assert config.queue == "hpc"
    assert config.num_processors == 4
    assert config.walltime == "1:00"
    assert config.mem == "2GB"
    assert config.span_option == "ptile"
    assert config.span_value == 2
    
    # Test invalid span option falls back to default
    config = LSFJobConfiguration(span_option="invalid")
    assert config.span_option == "hosts"


def test_mpb_data_converter_build_command():
    """Test the MPBDataConverter.build_command method"""
    converter = MPBDataConverter(
        input_file="input.h5",
        output_file="output.h5",
        options=MPBDataOptions(resolution=32, rectify=True)
    )
    
    cmd = converter.build_command()
    
    # Check basic command structure
    assert "source /dtu/sw/dcc/dcc-sw.bash" in cmd
    assert "module load mpb/1.11.1" in cmd
    assert "mpb-data" in cmd
    assert "-r" in cmd  # rectify option
    assert "-n" in cmd  # resolution option
    assert "32" in cmd  # resolution value
    assert "-o" in cmd  # output option
    assert "output.h5"