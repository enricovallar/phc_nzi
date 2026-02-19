"""
Test that the effective parameter extraction produces continuous (non-jumping)
results by checking the relative phase between E_y and H_z across k-points.

A π-jump in angle(E_y / H_z) between consecutive k-points indicates that
fix-hfield-phase and fix-efield-phase are independently choosing sign
conventions, which causes discontinuities in ε_eff, μ_eff.
"""

import numpy as np
import pytest
import os
import sys

sys.path.insert(0, "/zhome/2f/7/202918/phc_nzi/src")
sys.path.insert(0, "/zhome/2f/7/202918/phc_nzi")

from phc_nzi.simulation_handler import Simulation, MPBDataOptions
import phc_nzi.field_analyzer as fa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_relative_phase(simulation, band, polarization, k_indices, 
                        e_component="y", h_component="z"):
    """
    Compute angle(E_component / H_component) at the cell centre for each k-index.
    Returns an array of phases (radians) of length len(k_indices).
    """
    opt = MPBDataOptions(rectify=True, periods=1)
    phases = []
    for k_idx in k_indices:
        E = simulation.load_and_convert_field_data(
            k_idx, band, e_component, polarization, "e",
            conversion_options=opt)
        H = simulation.load_and_convert_field_data(
            k_idx, band, h_component, polarization, "h",
            conversion_options=opt)

        # Take the central pixel (or central slice for 3-D data)
        if E.ndim == 3:
            E = E[:, :, E.shape[2] // 2]
            H = H[:, :, H.shape[2] // 2]

        cy, cx = E.shape[0] // 2, E.shape[1] // 2
        e_val = E[cy, cx]
        h_val = H[cy, cx]

        if np.abs(h_val) < 1e-15:
            # H ≈ 0 at cell centre → use spatial average instead
            h_val = np.mean(H)
        if np.abs(e_val) < 1e-15:
            e_val = np.mean(E)

        phases.append(np.angle(e_val / h_val))

    return np.array(phases)


def _max_phase_jump(phases):
    """
    Return the maximum absolute phase difference between consecutive k-points,
    wrapped to [-π, π].  A value close to π indicates a sign flip.
    """
    diffs = np.diff(phases)
    # Wrap to [-π, π]
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
    return np.max(np.abs(diffs))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPhaseConsistency:
    """Verify that E/H relative phase is smooth across k-points."""

    # Adjust these to match an existing simulation with field data on disk
    DATA_ROOT = "/work3/enrva/phc_nzi_data/MPB_data"
    # Use the peak-linearity effective-parameter simulation (already computed)
    # You may need to update the tag below to match your actual directory name
    SIM_TAG = "C4v_diatomic_holes_1_2D_optimized_eff_r1_0p2849_r2_0p2677"

    @pytest.fixture(autouse=True)
    def setup(self):
        """Locate an existing simulation directory with field output."""
        sim_dir = os.path.join(self.DATA_ROOT, self.SIM_TAG)
        if not os.path.isdir(sim_dir):
            pytest.skip(
                f"Simulation directory not found: {sim_dir}\n"
                "Run the effective-parameter cell in the notebook first, "
                "then update SIM_TAG in this test."
            )
        # We don't need the script to *run* — just to load existing data
        self.sim = Simulation(
            simulation_name=self.SIM_TAG,
            script="",
            directory=sim_dir,
        )
        # Figure out how many k-points are available from frequency data
        df = self.sim.load_frequency_data("te")
        self.n_k = len(df)
        self.k_indices = list(range(1, self.n_k + 1))  # MPB uses 1-based

    # ---- Test 1: no π-jumps in angle(E_y / H_z) for band 4 ----
    def test_no_phase_jump_band4(self):
        phases = _get_relative_phase(
            self.sim, band=4, polarization="te",
            k_indices=self.k_indices,
            e_component="y", h_component="z",
        )
        max_jump = _max_phase_jump(phases)
        assert max_jump < np.pi / 2, (
            f"Band 4: max phase jump = {np.degrees(max_jump):.1f}° "
            f"(threshold 90°). Likely fix-efield-phase sign flip."
        )

    # ---- Test 2: no π-jumps for band 6 ----
    def test_no_phase_jump_band6(self):
        phases = _get_relative_phase(
            self.sim, band=6, polarization="te",
            k_indices=self.k_indices,
            e_component="y", h_component="z",
        )
        max_jump = _max_phase_jump(phases)
        assert max_jump < np.pi / 2, (
            f"Band 6: max phase jump = {np.degrees(max_jump):.1f}° "
            f"(threshold 90°). Likely fix-efield-phase sign flip."
        )

    # ---- Test 3: ε_eff should be continuous (no sign flips) ----
    def test_eps_eff_no_sign_flip(self):
        """
        If ε_eff changes sign between consecutive k-points while frequency
        barely changes, something is wrong with the phase convention.
        """
        analyzer_eff = fa.FieldAnalyzer(self.sim, [4, 6], "te", "x")
        data = analyzer_eff.get_eps_mu_impedance_neff("y", "z", plot=False)

        for band in [4, 6]:
            sub = data[data["band"] == band].sort_values("k_index")
            eps_re = np.real(sub["eps"].values)

            # Check for sign changes in Re(ε_eff)
            sign_changes = np.where(np.diff(np.sign(eps_re)))[0]
            # Near ω_D there should be exactly one zero-crossing;
            # more than 2 suggests spurious flips
            assert len(sign_changes) <= 2, (
                f"Band {band}: Re(ε_eff) has {len(sign_changes)} sign changes — "
                f"expected ≤ 2 (one physical zero-crossing near ω_D). "
                f"This suggests phase-fixing artefacts."
            )

    # ---- Test 4: μ_eff should be continuous (no sign flips) ----
    def test_mu_eff_no_sign_flip(self):
        analyzer_eff = fa.FieldAnalyzer(self.sim, [4, 6], "te", "x")
        data = analyzer_eff.get_eps_mu_impedance_neff("y", "z", plot=False)

        for band in [4, 6]:
            sub = data[data["band"] == band].sort_values("k_index")
            mu_re = np.real(sub["mu"].values)

            sign_changes = np.where(np.diff(np.sign(mu_re)))[0]
            assert len(sign_changes) <= 2, (
                f"Band {band}: Re(μ_eff) has {len(sign_changes)} sign changes — "
                f"expected ≤ 2. This suggests phase-fixing artefacts."
            )

    # ---- Test 5: impedance should vary smoothly ----
    def test_impedance_smoothness(self):
        """
        Check that |Z_eff| doesn't jump by more than 50% between
        consecutive k-points.
        """
        analyzer_eff = fa.FieldAnalyzer(self.sim, [4, 6], "te", "x")
        data = analyzer_eff.get_eps_mu_impedance_neff("y", "z", plot=False)

        for band in [4, 6]:
            sub = data[data["band"] == band].sort_values("k_index")
            z_abs = np.abs(sub["impedance"].values)

            # Relative change between consecutive points
            rel_change = np.abs(np.diff(z_abs)) / (z_abs[:-1] + 1e-20)
            max_rel = np.max(rel_change)
            assert max_rel < 0.5, (
                f"Band {band}: max relative |Z| jump = {max_rel:.2f} "
                f"(threshold 0.5). Suggests discontinuity from phase fixing."
            )


class TestNonblochConsistency:
    """
    Verify that FieldAnalyzer loads nonbloch fields for both E and H,
    not mixing Bloch H with nonbloch E.
    """

    DATA_ROOT = "/work3/enrva/phc_nzi_data/MPB_data"
    SIM_TAG = "C4v_diatomic_holes_1_2D_optimized_eff_r1_0p2849_r2_0p2677"

    @pytest.fixture(autouse=True)
    def setup(self):
        sim_dir = os.path.join(self.DATA_ROOT, self.SIM_TAG)
        if not os.path.isdir(sim_dir):
            pytest.skip(f"Simulation directory not found: {sim_dir}")
        self.sim_dir = sim_dir

    def test_nonbloch_files_exist_for_both_fields(self):
        """
        If the runner command requests output-nonbloch-efield-y and
        output-nonbloch-hfield-z, both file types must exist.
        """
        h5_files = [f for f in os.listdir(self.sim_dir) if f.endswith(".h5")]

        has_nonbloch_e = any("nonbloch" in f and "e" in f for f in h5_files)
        has_nonbloch_h = any("nonbloch" in f and "h" in f for f in h5_files)
        has_bloch_h = any(
            f.startswith("te-h") and "nonbloch" not in f for f in h5_files
        )

        # Both nonbloch must exist
        assert has_nonbloch_e, (
            "No nonbloch E-field files found. "
            "Add output-nonbloch-efield-y to extra_runner_command."
        )
        assert has_nonbloch_h, (
            "No nonbloch H-field files found. "
            "Add output-nonbloch-hfield-z to extra_runner_command."
        )

        # Warn if Bloch H also exists (FieldAnalyzer might load the wrong one)
        if has_bloch_h and has_nonbloch_h:
            import warnings
            warnings.warn(
                "Both Bloch and nonbloch H-field files exist. "
                "Verify FieldAnalyzer loads the nonbloch version. "
                "Consider removing 'output-hfield' from extra_runner_command."
            )