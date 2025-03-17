import pytest
import meep as mp
from src.mpb_configurator import MPBSchemeConfigurator

from sources.photonic_crystal_maker import (
    Geometry, PhotonicCrystal, Lattice, Material
)

class TestMPBSchemeConfigurator:
    @pytest.fixture
    def basic_phc(self):
        atom = Geometry(mp.Cylinder, {
            "radius": 0.2, 
            "height": 0.5, 
            "center": mp.Vector3(0, 0, 0), 
            "material": Material(epsilon=12)
        })
        lattice = Lattice("SX", (1, 1, Lattice.NO_SIZE))
        return PhotonicCrystal([atom], lattice)

    def test_initialization(self, basic_phc):
        config = MPBSchemeConfigurator(
            basic_phc, 
            simulation_types=["te", "tm"], 
            resolution=32, 
            num_bands=8,
            k_points=[mp.Vector3(0, 0, 0)]
        )
        assert config.simulation_types == ["te", "tm"]
        assert config.resolution == 32
        assert config.num_bands == 8
        assert config.k_points == [mp.Vector3(0, 0, 0)]

    def test_invalid_simulation_type(self, basic_phc):
        with pytest.raises(ValueError):
            MPBSchemeConfigurator(
                basic_phc, 
                simulation_types=["invalid_type"],
                k_points=[mp.Vector3(0, 0, 0)]
            )

    def test_vector3_resolution(self, basic_phc):
        vec_resolution = mp.Vector3(16, 32, 64)
        config = MPBSchemeConfigurator(
            basic_phc, 
            resolution=vec_resolution,
            k_points=[mp.Vector3(0, 0, 0)]
        )
        assert config.resolution == vec_resolution

    def test_invalid_phc_type(self):
        with pytest.raises(ValueError):
            MPBSchemeConfigurator("not_a_phc", k_points=[mp.Vector3(0, 0, 0)])

    def test_build_commands(self, basic_phc):
        config = MPBSchemeConfigurator(
            basic_phc,
            simulation_types=["te"],
            resolution=32,
            num_bands=8,
            k_points=[mp.Vector3(0, 0, 0)]
        )
        commands = config.build_commands()
        assert any("(set! num-bands 8)" in cmd for cmd in commands)
        assert any("(set! resolution 32)" in cmd for cmd in commands)
        assert any("(run-te )" in cmd for cmd in commands)

    def test_generate_scheme_config(self, basic_phc, tmp_path):
        config = MPBSchemeConfigurator(
            basic_phc,
            simulation_types=["tm"],
            resolution=32,
            num_bands=4,
            k_points=[mp.Vector3(0, 0, 0)]
        )
        filename = tmp_path / "test_config.ctl"
        script = config.generate_scheme_config(filename)
        
        assert "(set! num-bands 4)" in script
        assert "(run-tm )" in script
        
        with open(filename, "r") as f:
            content = f.read()
            assert "(set! num-bands 4)" in content

    def test_k_points_interpolation(self, basic_phc):
        k_points = [mp.Vector3(0, 0, 0), mp.Vector3(0.5, 0, 0)]
        config = MPBSchemeConfigurator(
            basic_phc,
            simulation_types=["te"],
            k_points=k_points,
            k_points_interpolation_factor=10
        )
        commands = config.build_commands()
        assert any("(set! k-points (interpolate 10 k-points))" in cmd for cmd in commands)

    def test_extra_runner_command(self, basic_phc):
        config = MPBSchemeConfigurator(
            basic_phc,
            simulation_types=["te"],
            k_points=[mp.Vector3(0, 0, 0)],
            extra_runner_command="output-hfield-z"
        )
        commands = config.build_commands()
        assert any("(run-te output-hfield-z)" in cmd for cmd in commands)

    def test_multiple_simulation_types(self, basic_phc):
        config = MPBSchemeConfigurator(
            basic_phc,
            simulation_types=["te", "tm", "zeven"],
            k_points=[mp.Vector3(0, 0, 0)]
        )
        commands = config.build_commands()
        assert any("(run-te )" in cmd for cmd in commands)
        assert any("(run-tm )" in cmd for cmd in commands)
        assert any("(run-zeven )" in cmd for cmd in commands)

    def test_get_scheme_config(self, basic_phc):
        config = MPBSchemeConfigurator(
            basic_phc,
            simulation_types=["te"],
            k_points=[mp.Vector3(0, 0, 0)]
        )
        joined = config.get_scheme_config(join_newline=True)
        assert isinstance(joined, str)
        assert "(set! num-bands 8)" in joined
        
        space_joined = config.get_scheme_config(join_newline=False)
        assert isinstance(space_joined, str)
        assert "(set! num-bands 8)" in space_joined