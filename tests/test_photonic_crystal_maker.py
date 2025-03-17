import pytest
import meep as mp
import numpy as np
from src.photonic_crystal_maker import (Lattice, ReciprocalLatticeVector,
                                        ScriptParamVector3, ScriptParam, ScriptParams)
import builtins


# Add parent directory to the path
def test_lattice_initialization():
    # Test initialization with valid types
    square_lattice = Lattice(Lattice.SQUARE)
    assert square_lattice.is_square()
    
    triangular_lattice = Lattice(Lattice.TRIANGULAR)
    assert triangular_lattice.is_triangular()
    
    diatomic_square_lattice = Lattice(Lattice.DIATOMIC_SQUARE)
    assert diatomic_square_lattice.is_diatomic_square()
    
    diatomic_triangular_lattice = Lattice(Lattice.DIATOMIC_TRIANGULAR)
    assert diatomic_triangular_lattice.is_diatomic_triangular()

def test_lattice_initialization_with_invalid_type():
    with pytest.raises(ValueError):
        invalid_lattice = Lattice("INVALID_TYPE")

def test_lattice_size():
    # Test with default size
    lattice_default = Lattice(Lattice.SQUARE)
    assert lattice_default._size == (1, 1, 0)
    
    # Test with custom size
    lattice_custom = Lattice(Lattice.TRIANGULAR, (2, 3, 0))
    assert lattice_custom._size == (2, 3, 0)

def test_square_lattice_properties():
    lattice = Lattice(Lattice.SQUARE)
    # Test basis vectors
    assert abs(lattice.basis1.x - 1) < 1e-8 and abs(lattice.basis1.y) < 1e-8 and abs(lattice.basis1.z) < 1e-8
    assert abs(lattice.basis2.x) < 1e-8 and abs(lattice.basis2.y - 1) < 1e-8 and abs(lattice.basis2.z) < 1e-8
    
    # Test center
    center = lattice.get_centers()
    assert center.x == 0 and center.y == 0

def test_triangular_lattice_properties():
    lattice = Lattice(Lattice.TRIANGULAR)
    print(lattice._mp_lattice)
    # Test basis vectors
    assert abs(lattice.basis1.x - 1) < 1e-8 and abs(lattice.basis1.y) < 1e-8 and abs(lattice.basis1.z) < 1e-8
    assert abs(lattice.basis2.x - 0.5) < 1e-8 and abs(lattice.basis2.y - np.sqrt(3)/2) < 1e-8 and abs(lattice.basis2.z) < 1e-8

def test_diatomic_square_centers():
    lattice = Lattice(Lattice.DIATOMIC_SQUARE)
    centers = lattice.get_centers()
    assert len(centers) == 2
    assert centers[0].x == 0 and centers[0].y == 0
    assert centers[1].x == 0.5 and centers[1].y == 0.5

def test_diatomic_triangular_centers():
    lattice = Lattice(Lattice.DIATOMIC_TRIANGULAR)
    centers = lattice.get_centers()
    assert len(centers) == 2
    assert centers[0].x == 1/3 and centers[0].y == 1/3
    assert centers[1].x == 2/3 and centers[1].y == 2/3

def test_high_symmetry_k_points_square_lattice():
    lattice = Lattice(Lattice.SQUARE)
    k_points = lattice.get_high_symmetry_k_points()
    assert "k_points_values" in k_points
    assert "k_points_labels" in k_points
    
    # Check the default path (not centered in gamma)
    labels = k_points["k_points_labels"]
    assert labels[0] == "X"
    assert labels[1] == "Γ"
    assert labels[2] == "M"
    assert labels[3] == "X"
    
    # Check centered in gamma path
    k_points_gamma = lattice.get_high_symmetry_k_points(centered_in_gamma=True)
    labels_gamma = k_points_gamma["k_points_labels"]
    assert labels_gamma[0] == "Γ"
    assert labels_gamma[1] == "X"
    assert labels_gamma[2] == "M"
    assert labels_gamma[3] == "Γ"

def test_high_symmetry_k_points_triangular_lattice():
    lattice = Lattice(Lattice.TRIANGULAR)
    k_points = lattice.get_high_symmetry_k_points()
    assert "k_points_values" in k_points
    assert "k_points_labels" in k_points
    
    # Check the default path (not centered in gamma)
    labels = k_points["k_points_labels"]
    assert labels[0] == "Γ"
    assert labels[1] == "K"
    assert labels[2] == "M"
    assert labels[3] == "Γ"
    
    # Check centered in gamma path
    k_points_gamma = lattice.get_high_symmetry_k_points(centered_in_gamma=True)
    labels_gamma = k_points_gamma["k_points_labels"]
    assert labels_gamma[0] == "Γ"
    assert labels_gamma[1] == "M"
    assert labels_gamma[2] == "K"
    assert labels_gamma[3] == "Γ"

def test_cartesian_to_reciprocal():
    lattice = Lattice(Lattice.SQUARE)
    cartesian_vector = mp.Vector3(1, 0, 0)
    reciprocal_vector = lattice.cartesian_to_reciprocal(cartesian_vector)
    # For a square lattice with unit vectors, the reciprocal vector should be the same
    assert abs(reciprocal_vector.x - 1) < 1e-10
    assert abs(reciprocal_vector.y) < 1e-10
    assert abs(reciprocal_vector.z) < 1e-10

def test_get_k_points_around_gamma():
    lattice = Lattice(Lattice.SQUARE)
    k_points = lattice.get_k_points_around_Gamma(0.1)
    assert "k_points_values" in k_points
    assert "k_points_labels" in k_points
    
    values = k_points["k_points_values"]
    labels = k_points["k_points_labels"]
    
    # Should have 3 points: KY, Gamma, KX
    assert len(values) == 3
    assert len(labels) == 3
    assert labels[0] == "$K_Y$"
    assert labels[1] == "Γ"
    assert labels[2] == "$K_X$"

def test_to_scheme():
    lattice = Lattice(Lattice.SQUARE, (1, 1, 0))
    scheme_str = lattice.to_scheme()
    assert isinstance(scheme_str, str)
    assert "make lattice" in scheme_str
    assert "(size 1 1 no-size)" in scheme_str
    assert "basis1" in scheme_str
    assert "basis2" in scheme_str



def test_script_param_vector3_initialization():
    # Test with numeric values only
    param = ScriptParamVector3(1, 2, 3)
    assert param._scheme_string == "(vector3 1 2 3)"
    assert param._names == []
    assert param._default_values == []
    
    # Test with one string parameter
    param = ScriptParamVector3("x_param", 2, 3, 1.5, 2, 3)
    assert param._scheme_string == "(vector3 x_param 2 3)"
    assert param._names == ["x_param"]
    assert param._default_values == [1.5]
    
    # Test with multiple string parameters
    param = ScriptParamVector3("x_param", "y_param", "z_param", 1, 2, 3)
    assert param._scheme_string == "(vector3 x_param y_param z_param)"
    assert param._names == ["x_param", "y_param", "z_param"]
    assert param._default_values == [1, 2, 3]
    
    # Test with mixed parameters
    param = ScriptParamVector3(1, "y_param", 3, 1, 2.5, 3)
    assert param._scheme_string == "(vector3 1 y_param 3)"
    assert param._names == ["y_param"]
    assert param._default_values == [2.5]

def test_script_param_vector3_to_scheme():
    # Test with one parameter
    param = ScriptParamVector3("radius", 2, 3, 0.5, 2, 3)
    scheme_commands = param.to_scheme()
    assert len(scheme_commands) == 1
    assert scheme_commands[0] == "(define-param radius 0.5)"
    
    # Test with multiple parameters
    param = ScriptParamVector3("x_size", "y_size", "z_size", 1, 2, 3)
    scheme_commands = param.to_scheme()
    assert len(scheme_commands) == 3
    assert "(define-param x_size 1)" in scheme_commands
    assert "(define-param y_size 2)" in scheme_commands
    assert "(define-param z_size 3)" in scheme_commands

def test_script_param_vector3_to_scheme_by_name():
    param = ScriptParamVector3("x_size", "y_size", "z_size", 1, 2, 3)
    
    # Test getting scheme for specific name
    x_scheme = param.to_scheme_by_name("x_size")
    assert x_scheme == ["(define-param x_size 1)"]
    
    y_scheme = param.to_scheme_by_name("y_size")
    assert y_scheme == ["(define-param y_size 2)"]
    
    # Test with invalid name
    with pytest.raises(ValueError):
        param.to_scheme_by_name("invalid_name")

def test_script_param_vector3_str():
    param = ScriptParamVector3(1, "y_param", 3, 1, 2.5, 3)
    assert str(param) == "(vector3 1 y_param 3)"

def test_script_param_vector3_validation():
    # Test with valid parameters
    param = ScriptParamVector3(1, 2, 3)  # All numeric
    param = ScriptParamVector3("x", 2, 3, 1, 2, 3)  # Mixed
    
    # Test with invalid parameter type
    with pytest.raises(ValueError):
        ScriptParamVector3([1, 2], 2, 3)  # List is not a valid type
    
    # Test with invalid default value type
    with pytest.raises(ValueError):
        ScriptParamVector3("x", 2, 3, "invalid", 2, 3)  # String is not a valid default
def test_script_param_initialization():
    # Test initialization with simple values
    param = ScriptParam("radius", 0.5)
    assert param._names == ["radius"]
    assert param._default_values == [0.5]
    assert param._scheme_string == "radius"

def test_script_param_to_scheme():
    # Test generating scheme commands
    param = ScriptParam("height", 1.0)
    scheme_commands = param.to_scheme()
    assert len(scheme_commands) == 1
    assert scheme_commands[0] == "(define-param height 1.0)"
    
    # Test with integer value
    param = ScriptParam("n", 3)
    scheme_commands = param.to_scheme()
    assert len(scheme_commands) == 1
    assert scheme_commands[0] == "(define-param n 3)"

def test_script_param_to_scheme_by_name():
    # Test getting scheme for specific name
    param = ScriptParam("width", 2.5)
    scheme_commands = param.to_scheme_by_name("width")
    assert scheme_commands == ["(define-param width 2.5)"]
    
    # Test with invalid name
    with pytest.raises(ValueError):
        param.to_scheme_by_name("invalid_param")

def test_script_param_str():
    # Test string representation
    param = ScriptParam("depth", 0.75)
    assert str(param) == "depth"

def test_script_params_initialization():
    # Test initialization with multiple params
    param1 = ScriptParam("a", 1.0)
    param2 = ScriptParam("b", 2.0)
    params = ScriptParams(param1, param2)
    assert len(params._script_params) == 2

def test_script_params_addition():
    # Test addition of ScriptParams objects
    param1 = ScriptParam("a", 1.0)
    param2 = ScriptParam("b", 2.0)
    params1 = ScriptParams(param1)
    params2 = ScriptParams(param2)
    
    # Add two ScriptParams objects
    combined = params1 + params2
    assert len(combined._script_params) == 2
    
    # Add a ScriptParam to a ScriptParams
    combined = params1 + param2
    assert len(combined._script_params) == 2
    
    # Test invalid addition
    with pytest.raises(ValueError):
        params1 + "invalid"

def test_script_params_to_scheme():
    # Test generating scheme commands from multiple params
    param1 = ScriptParam("x", 1.0)
    param2 = ScriptParam("y", 2.0)
    params = ScriptParams(param1, param2)
    
    # Mock the print function to avoid output during test
    original_print = builtins.print
    builtins.print = lambda *args, **kwargs: None
    
    commands = params.to_scheme()
    
    # Restore print function
    builtins.print = original_print
    
    assert len(commands) == 2
    assert "(define-param x 1.0)" in commands
    assert "(define-param y 2.0)" in commands

def test_script_params_duplicate_handling():
    # Test that duplicate parameter names are handled correctly
    param1 = ScriptParam("size", 1.0)
    param2 = ScriptParam("size", 2.0)  # Same name as param1
    params = ScriptParams(param1, param2)
    
    # Mock the print function
    original_print = builtins.print
    builtins.print = lambda *args, **kwargs: None
    
    commands = params.to_scheme()
    
    # Restore print function
    builtins.print = original_print
    
    # Should only include one definition for 'size'
    assert len(commands) == 1
    assert commands[0] == "(define-param size 1.0)"


from src.photonic_crystal_maker import Material, Geometry, GeometryGroup    
def test_geometry_initialization():
    # Test initialization with valid parameters
    material = Material(epsilon=12)
    geom = Geometry(mp.Cylinder, {
        "radius": 0.2, 
        "height": 0.3, 
        "center": mp.Vector3(0, 0, 0), 
        "material": material
    })
    assert geom.mp_geom_type == mp.Cylinder
    assert geom.params["radius"] == 0.2
    assert geom.params["height"] == 0.3
    assert geom.params["material"] == material

def test_geometry_with_script_params():
    # Test with ScriptParam objects
    radius_param = ScriptParam("radius", 0.2)
    height_param = ScriptParam("height", 0.3)
    geom = Geometry(mp.Cylinder, {
        "radius": radius_param, 
        "height": height_param, 
        "center": mp.Vector3(0, 0, 0), 
        "material": Material(epsilon=12)
    })
    
    # Check that script parameters are collected
    script_params = geom.get_script_params()
    assert len(script_params._script_params) == 2
    assert "radius" in [param._names[0] for param in script_params._script_params]
    assert "height" in [param._names[0] for param in script_params._script_params]

def test_geometry_with_script_param_vector3():
    # Test with ScriptParamVector3 objects
    size_param = ScriptParamVector3("x_size", "y_size", "z_size", 1, 2, 3)
    geom = Geometry(mp.Block, {
        "size": size_param, 
        "center": mp.Vector3(0, 0, 0), 
        "material": Material(epsilon=12)
    })
    
    # Check that script parameters are collected
    script_params = geom.get_script_params()
    assert len(script_params._script_params) == 1  # One ScriptParamVector3 object with three names
    param = script_params._script_params[0]
    assert "x_size" in param._names
    assert "y_size" in param._names
    assert "z_size" in param._names

def test_geometry_build():
    # Test building the meep geometry object
    material = Material(epsilon=12)
    geom = Geometry(mp.Cylinder, {
        "radius": 0.2, 
        "height": 0.3, 
        "center": mp.Vector3(0, 0, 0), 
        "material": material.to_python()
    })
    
    # Build the object and verify its properties
    mp_obj = geom.build()
    assert isinstance(mp_obj, mp.Cylinder)
    assert mp_obj.radius == 0.2
    assert mp_obj.height == 0.3
    assert mp_obj.center.x == 0 and mp_obj.center.y == 0 and mp_obj.center.z == 0

def test_geometry_to_scheme():
    # Test conversion to scheme string
    material = Material(epsilon=12)
    geom = Geometry(mp.Cylinder, {
        "radius": 0.2, 
        "height": 0.3, 
        "center": mp.Vector3(0, 0, 0), 
        "material": material
    })
    
    scheme_str = geom.to_scheme()
    assert "(make cylinder" in scheme_str
    assert "(radius 0.2)" in scheme_str
    assert "(height 0.3)" in scheme_str
    assert "(center (vector3 0.0 0.0 0.0))" in scheme_str
    assert "(material (make dielectric (epsilon 12)))" in scheme_str

def test_geometry_to_scheme_with_script_params():
    # Test scheme conversion with script parameters
    radius_param = ScriptParam("radius", 0.2)
    geom = Geometry(mp.Cylinder, {
        "radius": radius_param, 
        "height": 0.3, 
        "center": mp.Vector3(0, 0, 0), 
        "material": Material(epsilon=12)
    })
    
    scheme_str = geom.to_scheme()
    assert "(make cylinder" in scheme_str
    assert "(radius radius)" in scheme_str  # The parameter name is used in the scheme

def test_geometry_valid_scheme_geometries():
    # Test that valid scheme geometries are recognized
    for geom_type, scheme_name in Geometry.VALID_SCHEME_GEOMETRIES.items():
        geom = Geometry(geom_type, {"material": Material(epsilon=1)})
        assert geom.to_valid_scheme_geometry_definition() == scheme_name

def test_geometry_invalid_type():
    # Test with invalid geometry type
    invalid_type = "not_a_geometry_type"
    with pytest.raises(ValueError):
        geom = Geometry(invalid_type, {"material": Material(epsilon=1)})
        geom.to_valid_scheme_geometry_definition()

def test_geometry_group_initialization():
    # Test GeometryGroup initialization with multiple geometries
    geom1 = Geometry(mp.Cylinder, {"radius": 0.2, "height": 0.3, "material": Material(epsilon=12)})
    geom2 = Geometry(mp.Block, {"size": mp.Vector3(1, 1, 1), "material": Material(epsilon=8)})
    
    geometry_group = GeometryGroup(geom1, geom2)
    assert len(geometry_group._geometries) == 2

def test_geometry_group_to_scheme():
    # Test conversion of geometry group to scheme
    radius_param = ScriptParam("radius", 0.2)
    geom1 = Geometry(mp.Cylinder, {"radius": radius_param, "height": 0.3, "material": Material(epsilon=12)})
    geom2 = Geometry(mp.Block, {"size": mp.Vector3(1, 1, 1), "material": Material(epsilon=8)})
    
    geometry_group = GeometryGroup(geom1, geom2)
    scheme_str = geometry_group.to_scheme()
    
    assert "(list" in scheme_str
    assert "(make cylinder" in scheme_str
    assert "(make block" in scheme_str
    assert "(radius radius)" in scheme_str
    assert "(size (vector3 1.0 1.0 1.0))" in scheme_str

def test_geometry_group_script_params():
    # Test that script parameters are collected from all geometries
    param1 = ScriptParam("radius", 0.2)
    param2 = ScriptParam("height", 0.3)
    param3 = ScriptParamVector3("x_size", "y_size", "z_size", 1, 2, 3)
    
    geom1 = Geometry(mp.Cylinder, {"radius": param1, "height": param2, "material": Material(epsilon=12)})
    geom2 = Geometry(mp.Block, {"size": param3, "material": Material(epsilon=8)})
    
    geometry_group = GeometryGroup(geom1, geom2)
    script_params = geometry_group.get_script_params()
    
    # Check that all parameters are collected
    param_names = []
    for param in script_params._script_params:
        param_names.extend(param._names)
    
    assert "radius" in param_names
    assert "height" in param_names
    assert "x_size" in param_names
    assert "y_size" in param_names
    assert "z_size" in param_names


from src.photonic_crystal_maker import PhotonicCrystal  
def test_material_initialization():
    # Test basic initialization
    material = Material(epsilon=12)
    assert material.epsilon == 12
    
    # Test initialization with different epsilon
    material = Material(epsilon=2.25)
    assert material.epsilon == 2.25
    
    # Test initialization with integer epsilon
    material = Material(epsilon=1)
    assert material.epsilon == 1

def test_material_to_scheme():
    # Test conversion to scheme string
    material = Material(epsilon=12)
    scheme_str = material.to_scheme()
    assert scheme_str == "(make dielectric (epsilon 12))"
    
    # Test with float epsilon
    material = Material(epsilon=2.25)
    scheme_str = material.to_scheme()
    assert scheme_str == "(make dielectric (epsilon 2.25))"

def test_material_to_python():
    # Test conversion to Python object
    material = Material(epsilon=12)
    mp_medium = material.to_python()
    
    # Verify the Medium object has the correct epsilon
    assert isinstance(mp_medium, mp.Medium)
    assert mp_medium.epsilon_diag.x == 12
    assert mp_medium.epsilon_diag.y == 12
    assert mp_medium.epsilon_diag.z == 12
    
    # Test with float epsilon
    material = Material(epsilon=2.25)
    mp_medium = material.to_python()
    assert mp_medium.epsilon_diag.x == 2.25

def test_photonic_crystal_material_interaction():
    # Test how Material works when used in a PhotonicCrystal
    material = Material(epsilon=12)
    atom = Geometry(mp.Cylinder, {"radius": 0.2, "height": 0.3, "center": mp.Vector3(0, 0, 0), "material": material})
    lattice = Lattice(Lattice.SQUARE)
    
    # Create a photonic crystal with the material
    pc = PhotonicCrystal([atom], lattice, background_material=Material(epsilon=1))
    
    # Check that the material is correctly included in the scheme string
    scheme_str = pc.to_scheme()
    assert "(make dielectric (epsilon 12))" in scheme_str
    assert "(make dielectric (epsilon 1))" not in scheme_str  # Background material isn't directly in the scheme