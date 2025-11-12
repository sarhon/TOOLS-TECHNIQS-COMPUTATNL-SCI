"""
Unit tests for Python discrete ordinates transport solver (PyDiscord)
"""

import pytest
import numpy as np
import sys
import os
import json

from pydiscord import Material, Settings, solve_flux


class TestGaussLegendre:
    """Test Gauss-Legendre quadrature generation via solve_flux"""

    def test_s2_quadrature(self):
        """Test S2 (N=2) quadrature"""
        materials = [Material("test", 1.0, 0.0, 1.0, (-5.0, 5.0))]
        settings = Settings(phiL='vac', phiR='vac', num_nodes=5, sn=2)

        x_edges, total_flux, current, angular_flux, mu, w, x_centers = solve_flux(materials, settings)

        # Check symmetry
        assert np.isclose(mu[0], -mu[1])
        assert np.isclose(w[0], w[1])

        # Check weights sum to 2
        assert np.isclose(np.sum(w), 2.0)

        # Check known values for N=2
        expected_mu = 1.0 / np.sqrt(3.0)
        assert np.isclose(abs(mu[0]), expected_mu, rtol=1e-10)
        assert np.isclose(w[0], 1.0)

    def test_s4_quadrature(self):
        """Test S4 (N=4) quadrature"""
        materials = [Material("test", 1.0, 0.0, 1.0, (-5.0, 5.0))]
        settings = Settings(phiL='vac', phiR='vac', num_nodes=5, sn=4)

        x_edges, total_flux, current, angular_flux, mu, w, x_centers = solve_flux(materials, settings)

        # Check symmetry
        for i in range(2):
            assert np.isclose(mu[i], -mu[3-i])
            assert np.isclose(w[i], w[3-i])

        # Check weights sum to 2
        assert np.isclose(np.sum(w), 2.0)

        # Check all mu in [-1, 1]
        assert np.all(mu >= -1.0) and np.all(mu <= 1.0)

    def test_s8_quadrature(self):
        """Test S8 (N=8) quadrature"""
        materials = [Material("test", 1.0, 0.0, 1.0, (-5.0, 5.0))]
        settings = Settings(phiL='vac', phiR='vac', num_nodes=5, sn=8)

        x_edges, total_flux, current, angular_flux, mu, w, x_centers = solve_flux(materials, settings)

        # Check weights sum to 2
        assert np.isclose(np.sum(w), 2.0)

        # Check symmetry
        for i in range(4):
            assert np.isclose(mu[i], -mu[7-i], rtol=1e-10)
            assert np.isclose(w[i], w[7-i], rtol=1e-10)


class TestMaterial:
    """Test Material class"""

    def test_material_creation(self):
        """Test basic material creation"""
        mat = Material("test", 1.0, 0.5, 1.5, (-5.0, 5.0))

        assert mat.name == "test"
        assert mat.total == 1.0
        assert mat.scatter == 0.5
        assert mat.Q == 1.5
        assert mat.bounds[0] == -5.0
        assert mat.bounds[1] == 5.0

    def test_material_absorption(self):
        """Test absorption calculation"""
        mat = Material("test", 1.0, 0.3, 0.0, (0.0, 1.0))
        assert np.isclose(mat.absorption, 0.7)

    def test_material_bounds(self):
        """Test material bounds"""
        mat = Material("test", 1.0, 0.5, 0.0, (-10.0, 10.0))
        assert mat.bounds[0] < mat.bounds[1]


class TestSettings:
    """Test Settings class"""

    def test_settings_vacuum(self):
        """Test vacuum boundary conditions"""
        settings = Settings(phiL='vac', phiR='vac', num_nodes=10, sn=4)

        assert settings.phiL == 'vac'
        assert settings.phiR == 'vac'
        assert settings.num_nodes == 10
        assert settings.sn == 4

    def test_settings_reflective(self):
        """Test reflective boundary conditions"""
        settings = Settings(phiL='ref', phiR='ref', num_nodes=20, sn=8)

        assert settings.phiL == 'ref'
        assert settings.phiR == 'ref'


class TestSingleMaterialVacuum:
    """Test single material with vacuum boundaries"""

    def test_pure_source_no_scatter(self):
        """Test pure source with no scattering"""
        materials = [Material("source", 1.0, 0.0, 1.0, (-5.0, 5.0))]
        settings = Settings(phiL='vac', phiR='vac', num_nodes=10, sn=4)

        x_edges, total_flux, current, angular_flux, mu, w, x_centers = solve_flux(materials, settings)

        # Check that flux is positive
        assert np.all(total_flux >= 0.0)

        # Check that flux is symmetric for symmetric problem
        n = len(total_flux)
        mid = n // 2
        assert np.allclose(total_flux[:mid], total_flux[n-1:mid:-1], rtol=0.01)

        # Check that maximum flux is near center
        max_idx = np.argmax(total_flux)
        assert abs(max_idx - mid) <= 1

    def test_source_with_scatter(self):
        """Test source with scattering"""
        materials = [Material("source", 1.0, 0.5, 1.0, (-5.0, 5.0))]
        settings = Settings(phiL='vac', phiR='vac', num_nodes=10, sn=4)

        x_edges, total_flux, current, angular_flux, mu, w, x_centers = solve_flux(materials, settings)

        # With scattering, flux should be higher
        assert np.max(total_flux) > 1.0

        # Check array shapes
        assert len(x_edges) == settings.num_nodes + 1
        assert len(total_flux) == settings.num_nodes + 1
        assert len(current) == settings.num_nodes + 1
        assert angular_flux.shape == (settings.sn, settings.num_nodes + 1)


class TestSingleMaterialReflective:
    """Test single material with reflective boundaries"""

    def test_reflective_boundaries(self):
        """Test reflective boundary conditions"""
        materials = [Material("source", 1.0, 0.9, 1.0, (-5.0, 5.0))]
        settings = Settings(phiL='ref', phiR='ref', num_nodes=10, sn=4)

        x_edges, total_flux, current, angular_flux, mu, w, x_centers = solve_flux(materials, settings)

        # Reflective boundaries should give higher flux than vacuum
        # Check current at boundaries should be near zero
        assert abs(current[0]) < 0.1
        assert abs(current[-1]) < 0.1

        # Check flux is positive
        assert np.all(total_flux > 0.0)


class TestThreeMaterialProblem:
    """Test three material problem (scatter-source-scatter)"""

    def test_three_material_vacuum(self):
        """Test three material problem with vacuum boundaries"""
        materials = [
            Material("scatter1", 2.0, 1.99, 0.0, (-15.0, -5.0)),
            Material("source", 1.0, 0.0, 1.0, (-5.0, 5.0)),
            Material("scatter2", 2.0, 1.99, 0.0, (5.0, 15.0))
        ]
        settings = Settings(phiL='vac', phiR='vac', num_nodes=27, sn=8)

        x_edges, total_flux, current, angular_flux, mu, w, x_centers = solve_flux(materials, settings)

        # Check flux is symmetric
        n = len(total_flux)
        assert np.allclose(total_flux[:n//2], np.flip(total_flux[-n//2:]), rtol=0.01)

        # Check maximum flux is in source region
        mid = n // 2
        source_flux = total_flux[mid-5:mid+5]
        assert np.max(source_flux) == np.max(total_flux)

        # Known result from previous runs
        assert np.isclose(np.max(total_flux), 0.9997443, rtol=1e-4)
        assert np.isclose(np.min(total_flux), 0.0293688, rtol=1e-4)


class TestMeshRefinement:
    """Test mesh refinement convergence"""

    def test_mesh_convergence(self):
        """Test that finer meshes give more accurate results"""
        materials = [Material("source", 1.0, 0.5, 1.0, (-5.0, 5.0))]

        fluxes = []
        for num_nodes in [5, 10, 20]:
            settings = Settings(phiL='vac', phiR='vac', num_nodes=num_nodes, sn=4)

            x_edges, total_flux, current, _, _, _, _ = solve_flux(materials, settings)

            # Get flux at center
            mid = len(total_flux) // 2
            fluxes.append(total_flux[mid])

        # Finer mesh should converge
        # Check that solutions are getting closer
        diff1 = abs(fluxes[1] - fluxes[0])
        diff2 = abs(fluxes[2] - fluxes[1])
        assert diff2 < diff1  # Convergence


class TestAngularRefinement:
    """Test angular (SN) refinement"""

    def test_sn_convergence(self):
        """Test that higher SN order gives different results"""
        materials = [Material("source", 1.0, 0.8, 1.0, (-5.0, 5.0))]

        fluxes = []
        for sn in [2, 4, 8]:
            settings = Settings(phiL='vac', phiR='vac', num_nodes=10, sn=sn)

            x_edges, total_flux, current, _, _, _, _ = solve_flux(materials, settings)

            mid = len(total_flux) // 2
            fluxes.append(total_flux[mid])

        # Should see convergence with higher SN
        assert len(set(fluxes)) == 3  # All different


class TestConservation:
    """Test conservation properties"""

    def test_particle_balance(self):
        """Test particle balance for pure absorber"""
        materials = [Material("absorber", 1.0, 0.0, 1.0, (-5.0, 5.0))]
        settings = Settings(phiL='vac', phiR='vac', num_nodes=20, sn=8)

        x_edges, total_flux, current, angular_flux, mu, w, x_centers = solve_flux(materials, settings)

        # Calculate total source
        dx = x_edges[1] - x_edges[0]
        total_source = materials[0].Q * (materials[0].bounds[1] - materials[0].bounds[0])

        # Calculate total absorption (using midpoint rule)
        total_absorption = 0.0
        for i in range(len(x_centers)):
            avg_flux = (total_flux[i] + total_flux[i+1]) / 2.0
            total_absorption += materials[0].total * avg_flux * dx

        # Calculate net leakage
        net_leakage = current[-1] - current[0]

        # Balance: source = absorption + leakage
        balance = abs(total_source - total_absorption - net_leakage)
        assert balance / total_source < 0.05  # Within 5%


if __name__ == '__main__':
    pytest.main([__file__, '-v'])