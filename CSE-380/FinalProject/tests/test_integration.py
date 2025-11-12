"""
Integration tests comparing Python and Fortran implementations
"""

import pytest
import numpy as np
import json
import subprocess
import os
import tempfile
import sys

# Add Python directory to path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../Python'))

from pydiscord import Material, Settings, solve_flux


class TestPythonFortranComparison:
    """Compare Python and Fortran solver outputs"""

    @classmethod
    def setup_class(cls):
        """Build Fortran executable if needed"""
        fortran_exe = os.path.join(os.path.dirname(__file__), '../fdiscord/bin/fdiscord')
        if not os.path.exists(fortran_exe):
            print("Building Fortran executable...")
            makefile_dir = os.path.join(os.path.dirname(__file__), '../fdiscord')
            result = subprocess.run(['make', 'clean'], cwd=makefile_dir, capture_output=True)
            result = subprocess.run(['make'], cwd=makefile_dir, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError("Failed to build Fortran code")
        cls.fortran_exe = fortran_exe

    def run_fortran(self, input_file, output_file):
        """Run Fortran solver with input/output files"""
        result = subprocess.run(
            [self.fortran_exe, '-i', input_file, '-o', output_file],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Fortran solver failed: {result.stderr}")

        with open(output_file, 'r') as f:
            return json.load(f)

    def run_python(self, input_file):
        """Run Python solver and return results"""
        # Load input
        with open(input_file, 'r') as f:
            config_data = json.load(f)

        # Create materials
        materials = []
        for mat_data in config_data['materials']:
            total = mat_data['total']
            scatter = mat_data['scatter']
            absorption = total - scatter
            materials.append(Material(
                mat_data['name'],
                total,
                scatter,
                mat_data['Q'],
                tuple(mat_data['bounds'])
            ))

        # Create settings
        settings_data = config_data['settings']
        # Map boundary types to boundary values
        phiL = settings_data['phiL'] if settings_data['phiL_type'] not in ['vac', 'ref'] else settings_data['phiL_type']
        phiR = settings_data['phiR'] if settings_data['phiR_type'] not in ['vac', 'ref'] else settings_data['phiR_type']

        settings = Settings(
            phiL=phiL,
            phiR=phiR,
            num_nodes=settings_data['num_nodes'],
            sn=settings_data['sn']
        )

        # Solve
        x_edges, total_flux, current, angular_flux, mu, w, x_centers = solve_flux(materials, settings)

        return {
            'x_edges': x_edges,
            'total_flux': total_flux,
            'current': current,
            'angular_flux': angular_flux,
            'mu': mu,
            'w': w
        }

    def compare_solutions(self, py_results, fortran_output, rtol=1e-6):
        """Compare Python and Fortran solutions"""
        # Extract Fortran data
        f_flux = np.array(fortran_output['solution']['scalar_flux'])
        f_current = np.array(fortran_output['solution']['current'])
        f_angular = np.array(fortran_output['solution']['angular_flux'])

        # Compare scalar flux
        assert np.allclose(py_results['total_flux'], f_flux, rtol=rtol), \
            f"Scalar flux mismatch: max diff = {np.max(np.abs(py_results['total_flux'] - f_flux))}"

        # Compare current
        assert np.allclose(py_results['current'], f_current, rtol=rtol), \
            f"Current mismatch: max diff = {np.max(np.abs(py_results['current'] - f_current))}"

        # Compare angular flux
        assert np.allclose(py_results['angular_flux'], f_angular, rtol=rtol), \
            f"Angular flux mismatch: max diff = {np.max(np.abs(py_results['angular_flux'] - f_angular))}"

        return True

    def test_single_material_vacuum(self):
        """Test single material with vacuum boundaries"""
        input_file = os.path.join(os.path.dirname(__file__),
                                 'test_cases/single_material_vac.json')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name

        try:
            # Run both solvers
            py_results = self.run_python(input_file)
            fortran_output = self.run_fortran(input_file, output_file)

            # Compare
            self.compare_solutions(py_results, fortran_output, rtol=1e-5)

        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_single_material_reflective(self):
        """Test single material with reflective boundaries"""
        input_file = os.path.join(os.path.dirname(__file__),
                                 'test_cases/single_material_ref.json')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name

        try:
            # Run both solvers
            py_results = self.run_python(input_file)
            fortran_output = self.run_fortran(input_file, output_file)

            # Compare
            self.compare_solutions(py_results, fortran_output, rtol=1e-5)

        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_three_material_vacuum(self):
        """Test three material problem"""
        input_file = os.path.join(os.path.dirname(__file__),
                                 'test_cases/three_material_vac.json')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name

        try:
            # Run both solvers
            py_results = self.run_python(input_file)
            fortran_output = self.run_fortran(input_file, output_file)

            # Compare
            self.compare_solutions(py_results, fortran_output, rtol=1e-5)

            # Check specific values
            py_max = np.max(py_results['total_flux'])
            f_max = fortran_output['summary']['max_flux']
            assert np.isclose(py_max, f_max, rtol=1e-5)

        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_quadrature_agreement(self):
        """Test that Python and Fortran generate same quadrature"""
        input_file = os.path.join(os.path.dirname(__file__),
                                 'test_cases/single_material_vac.json')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name

        try:
            # Run both solvers
            py_results = self.run_python(input_file)
            fortran_output = self.run_fortran(input_file, output_file)

            # Compare quadrature
            f_mu = np.array(fortran_output['quadrature']['mu'])
            f_w = np.array(fortran_output['quadrature']['weights'])

            assert np.allclose(py_results['mu'], f_mu, rtol=1e-10), \
                "Quadrature ordinates don't match"
            assert np.allclose(py_results['w'], f_w, rtol=1e-10), \
                "Quadrature weights don't match"

        finally:
            if os.path.exists(output_file):
                os.remove(output_file)


class TestParameterSpace:
    """Test different parameter combinations"""

    def create_test_input(self, num_nodes, sn, scatter_ratio):
        """Create a test input file"""
        config = {
            "materials": [
                {
                    "name": "test",
                    "total": 1.0,
                    "scatter": scatter_ratio,
                    "Q": 1.0,
                    "bounds": [-5.0, 5.0]
                }
            ],
            "settings": {
                "phiL": 0.0,
                "phiR": 0.0,
                "phiL_type": "vac",
                "phiR_type": "vac",
                "num_nodes": num_nodes,
                "sn": sn
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            return f.name

    @pytest.mark.parametrize("num_nodes", [5, 10, 20])
    def test_mesh_sizes(self, num_nodes):
        """Test different mesh sizes"""
        input_file = self.create_test_input(num_nodes, 4, 0.5)

        try:
            # Load and solve with Python
            with open(input_file, 'r') as f:
                config_data = json.load(f)

            mat_data = config_data['materials'][0]
            materials = [Material(
                mat_data['name'], mat_data['total'], mat_data['scatter'],
                mat_data['Q'], tuple(mat_data['bounds'])
            )]

            settings_data = config_data['settings']
            # Map boundary types to boundary values
            phiL = settings_data['phiL'] if settings_data['phiL_type'] not in ['vac', 'ref'] else settings_data['phiL_type']
            phiR = settings_data['phiR'] if settings_data['phiR_type'] not in ['vac', 'ref'] else settings_data['phiR_type']

            settings = Settings(
                phiL=phiL, phiR=phiR,
                num_nodes=settings_data['num_nodes'], sn=settings_data['sn']
            )

            x_edges, total_flux, current, _, _, _, _ = solve_flux(materials, settings)

            # Basic checks
            assert len(total_flux) == num_nodes + 1
            assert np.all(total_flux >= 0.0)

        finally:
            if os.path.exists(input_file):
                os.remove(input_file)

    @pytest.mark.parametrize("sn", [2, 4, 8])
    def test_sn_orders(self, sn):
        """Test different SN orders"""
        input_file = self.create_test_input(10, sn, 0.5)

        try:
            # Load and solve with Python
            with open(input_file, 'r') as f:
                config_data = json.load(f)

            mat_data = config_data['materials'][0]
            materials = [Material(
                mat_data['name'], mat_data['total'], mat_data['scatter'],
                mat_data['Q'], tuple(mat_data['bounds'])
            )]

            settings_data = config_data['settings']
            # Map boundary types to boundary values
            phiL = settings_data['phiL'] if settings_data['phiL_type'] not in ['vac', 'ref'] else settings_data['phiL_type']
            phiR = settings_data['phiR'] if settings_data['phiR_type'] not in ['vac', 'ref'] else settings_data['phiR_type']

            settings = Settings(
                phiL=phiL, phiR=phiR,
                num_nodes=settings_data['num_nodes'], sn=settings_data['sn']
            )

            x_edges, total_flux, current, angular_flux, mu, w, _ = solve_flux(materials, settings)

            # Check angular flux has correct shape
            assert angular_flux.shape == (sn, settings_data['num_nodes'] + 1)
            assert len(mu) == sn
            assert len(w) == sn

        finally:
            if os.path.exists(input_file):
                os.remove(input_file)

    @pytest.mark.parametrize("scatter_ratio", [0.0, 0.5, 0.9, 0.99])
    def test_scattering_ratios(self, scatter_ratio):
        """Test different scattering ratios"""
        input_file = self.create_test_input(10, 4, scatter_ratio)

        try:
            # Load and solve with Python
            with open(input_file, 'r') as f:
                config_data = json.load(f)

            mat_data = config_data['materials'][0]
            materials = [Material(
                mat_data['name'], mat_data['total'], mat_data['scatter'],
                mat_data['Q'], tuple(mat_data['bounds'])
            )]

            settings_data = config_data['settings']
            # Map boundary types to boundary values
            phiL = settings_data['phiL'] if settings_data['phiL_type'] not in ['vac', 'ref'] else settings_data['phiL_type']
            phiR = settings_data['phiR'] if settings_data['phiR_type'] not in ['vac', 'ref'] else settings_data['phiR_type']

            settings = Settings(
                phiL=phiL, phiR=phiR,
                num_nodes=settings_data['num_nodes'], sn=settings_data['sn']
            )

            x_edges, total_flux, current, _, _, _, _ = solve_flux(materials, settings)

            # Higher scattering should give higher flux
            if scatter_ratio > 0.5:
                assert np.max(total_flux) > 2.0

        finally:
            if os.path.exists(input_file):
                os.remove(input_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])