"""
PINN Data Generation Script

Generate sparse FEMM observations for physics-informed neural network training.
Uses FEMM as physics oracle to create training datasets with:
- Sparse field observations (50, 100, 500 points)
- Collocation points for physics loss (domain + boundary)
- Full FEMM solution for validation

Author: Generated for FEMM-PINN integration
"""

import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from femm_solver import FEMMSolver, MeshInterpolator
import argparse
import sys


class PINNDataGenerator:
    """
    Generate PINN training data from FEMM simulations.

    Creates sparse observation datasets suitable for physics-informed neural
    network training, including:
    - Sparse field measurements at random points
    - Collocation points for enforcing physics constraints
    - Full FEMM solution for validation
    """

    def __init__(self, solver):
        """
        Initialize data generator.

        Args:
            solver: FEMMSolver instance
        """
        self.solver = solver
        self.domain_bounds = (-500, 500, -500, 500)  # xmin, xmax, ymin, ymax [mm]

    def extract_sparse_observations(self, mesh_data, n_obs):
        """
        Sample random points from FEMM solution.

        Args:
            mesh_data: FEMMMeshData from solver
            n_obs: Number of observation points to extract

        Returns:
            Dict with observation coordinates and field values
        """
        # Create interpolator
        interp = MeshInterpolator(mesh_data)

        # Generate random observation points in domain
        xmin, xmax, ymin, ymax = self.domain_bounds
        x_obs = np.random.uniform(xmin, xmax, n_obs)
        y_obs = np.random.uniform(ymin, ymax, n_obs)

        # Create a fine grid for interpolation
        grid = interp.interpolate_to_grid(resolution=256)

        # Get grid coordinates
        xx, yy = grid['grid_coords']
        x_grid = xx[0, :]
        y_grid = yy[:, 0]

        # Interpolate FEMM fields to observation points
        Bx_obs = np.zeros(n_obs)
        By_obs = np.zeros(n_obs)
        Hx_obs = np.zeros(n_obs)
        Hy_obs = np.zeros(n_obs)

        for i in range(n_obs):
            # Find nearest grid indices
            ix = np.argmin(np.abs(x_grid - x_obs[i]))
            iy = np.argmin(np.abs(y_grid - y_obs[i]))

            # Extract values
            Bx_obs[i] = grid['Bx'][iy, ix]
            By_obs[i] = grid['By'][iy, ix]
            Hx_obs[i] = grid['Hx'][iy, ix]
            Hy_obs[i] = grid['Hy'][iy, ix]

        return {
            'coords': np.column_stack([x_obs, y_obs]),
            'Bx': Bx_obs,
            'By': By_obs,
            'Hx': Hx_obs,
            'Hy': Hy_obs
        }

    def generate_collocation_points(self, n_domain=1000, n_boundary=100):
        """
        Generate collocation points for physics loss.

        Args:
            n_domain: Number of interior domain points
            n_boundary: Number of boundary points

        Returns:
            Tuple of (domain_coords, boundary_coords)
        """
        xmin, xmax, ymin, ymax = self.domain_bounds

        # Domain points: uniform random in interior
        domain_x = np.random.uniform(xmin, xmax, n_domain)
        domain_y = np.random.uniform(ymin, ymax, n_domain)
        domain_coords = np.column_stack([domain_x, domain_y])

        # Boundary points: on edges of domain
        # Distribute evenly across 4 edges
        n_per_edge = n_boundary // 4

        # Bottom edge (y = ymin)
        bottom_x = np.linspace(xmin, xmax, n_per_edge)
        bottom_y = np.full(n_per_edge, ymin)

        # Top edge (y = ymax)
        top_x = np.linspace(xmin, xmax, n_per_edge)
        top_y = np.full(n_per_edge, ymax)

        # Left edge (x = xmin)
        left_x = np.full(n_per_edge, xmin)
        left_y = np.linspace(ymin, ymax, n_per_edge)

        # Right edge (x = xmax)
        right_x = np.full(n_per_edge, xmax)
        right_y = np.linspace(ymin, ymax, n_per_edge)

        # Combine all boundary points
        boundary_coords = np.column_stack([
            np.concatenate([bottom_x, top_x, left_x, right_x]),
            np.concatenate([bottom_y, top_y, left_y, right_y])
        ])

        return domain_coords, boundary_coords

    def generate_coil_sample(self):
        """
        Generate single coil problem with random parameters.

        Returns:
            Tuple of (mesh_data, parameters)
        """
        # Random parameters
        current = np.random.uniform(10, 100)  # [A]
        wire_radius = np.random.uniform(0.005, 0.05)  # [m]

        # Run FEMM
        mesh_data = self.solver.analyze_coil(current, wire_radius)

        if not mesh_data.success:
            raise RuntimeError(f"FEMM analysis failed: {mesh_data.error_message}")

        parameters = {
            'current': current,
            'wire_radius': wire_radius
        }

        return mesh_data, parameters

    def generate_transformer_sample(self):
        """
        Generate single transformer problem with random parameters.

        Returns:
            Tuple of (mesh_data, parameters)
        """
        # Random parameters
        primary_turns = np.random.randint(100, 500)
        secondary_turns = np.random.randint(100, 500)
        core_area = np.random.uniform(0.0001, 0.001)  # [m²]
        frequency = np.random.uniform(50, 400)  # [Hz]
        primary_voltage = np.random.uniform(100, 240)  # [V]

        # Run FEMM
        mesh_data = self.solver.analyze_transformer(
            primary_turns, secondary_turns, core_area, frequency, primary_voltage
        )

        if not mesh_data.success:
            raise RuntimeError(f"FEMM analysis failed: {mesh_data.error_message}")

        parameters = {
            'primary_turns': primary_turns,
            'secondary_turns': secondary_turns,
            'core_area': core_area,
            'frequency': frequency,
            'primary_voltage': primary_voltage
        }

        return mesh_data, parameters

    def generate_ipm_motor_sample(self):
        """
        Generate single IPM motor problem with random parameters.

        Returns:
            Tuple of (mesh_data, parameters)
        """
        # Random parameters
        stator_slots = 12
        rotor_poles = 10
        stator_outer_radius = np.random.uniform(0.08, 0.12)  # [m]
        stator_inner_radius = np.random.uniform(0.04, 0.06)  # [m]
        rotor_inner_radius = np.random.uniform(0.01, 0.02)  # [m]
        air_gap = np.random.uniform(0.0005, 0.002)  # [m]
        magnet_strength = np.random.uniform(1.0, 1.4)  # [T]
        rotor_position = np.random.uniform(0, 360)  # [deg]
        current_amplitude = np.random.uniform(10, 100)  # [A]
        magnet_length = np.random.uniform(15, 25)  # [mm]
        magnet_width = np.random.uniform(6, 10)  # [mm]
        slot_depth = np.random.uniform(20, 30)  # [mm]

        # Run FEMM
        mesh_data = self.solver.analyze_ipm_motor(
            stator_slots, rotor_poles, stator_outer_radius, stator_inner_radius,
            rotor_inner_radius, air_gap, magnet_strength, rotor_position,
            current_amplitude, magnet_length, magnet_width, slot_depth
        )

        if not mesh_data.success:
            raise RuntimeError(f"FEMM analysis failed: {mesh_data.error_message}")

        parameters = {
            'stator_slots': stator_slots,
            'rotor_poles': rotor_poles,
            'stator_outer_radius': stator_outer_radius,
            'stator_inner_radius': stator_inner_radius,
            'rotor_inner_radius': rotor_inner_radius,
            'air_gap': air_gap,
            'magnet_strength': magnet_strength,
            'rotor_position': rotor_position,
            'current_amplitude': current_amplitude,
            'magnet_length': magnet_length,
            'magnet_width': magnet_width,
            'slot_depth': slot_depth
        }

        return mesh_data, parameters

    def create_dataset(self, problem_type, n_samples, obs_counts):
        """
        Main dataset generation loop.

        Args:
            problem_type: 'coil', 'transformer', or 'ipm_motor'
            n_samples: Number of samples to generate
            obs_counts: List of observation counts (e.g., [50, 100, 500])
        """
        # Map problem types to generators
        generators = {
            'coil': self.generate_coil_sample,
            'transformer': self.generate_transformer_sample,
            'ipm_motor': self.generate_ipm_motor_sample
        }

        if problem_type not in generators:
            raise ValueError(f"Unknown problem type: {problem_type}")

        generator = generators[problem_type]

        for n_obs in obs_counts:
            print(f"\n{'='*60}")
            print(f"Generating {problem_type} dataset with {n_obs} observations")
            print(f"{'='*60}")

            output_dir = Path(f'dataset/pinn/{problem_type}_n{n_obs}')
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate samples
            success_count = 0
            for i in tqdm(range(n_samples), desc=f"Samples (n_obs={n_obs})"):
                try:
                    # Generate FEMM problem
                    mesh_data, parameters = generator()

                    # Extract sparse observations
                    obs = self.extract_sparse_observations(mesh_data, n_obs)

                    # Generate collocation points
                    domain_pts, boundary_pts = self.generate_collocation_points(1000, 100)

                    # Get FEMM grid for validation and source terms
                    interp = MeshInterpolator(mesh_data)
                    grid = interp.interpolate_to_grid(256)

                    # Create sample dictionary
                    sample = {
                        'sample_id': i,
                        'problem_type': problem_type,
                        'parameters': parameters,
                        'observations': obs,
                        'collocation': {
                            'domain': domain_pts,
                            'boundary': boundary_pts
                        },
                        'femm_sources': {
                            'J_z': grid['source_current_density'],
                            'mu_x': grid['mu_x'],
                            'mu_y': grid['mu_y']
                        },
                        'validation': {
                            'Bx_femm': grid['Bx'],
                            'By_femm': grid['By'],
                            'Hx_femm': grid['Hx'],
                            'Hy_femm': grid['Hy'],
                            'grid_coords': grid['grid_coords']
                        }
                    }

                    # Save sample
                    output_file = output_dir / f'sample_{i:04d}.pkl'
                    with open(output_file, 'wb') as f:
                        pickle.dump(sample, f)
                    success_count += 1

                except Exception as e:
                    print(f"\n[ERROR] Sample {i} failed: {e}")
                    continue

            print(f"\n[OK] Dataset saved to {output_dir}")
            print(f"     Generated {success_count} samples successfully")

            # Show failure count if any
            failed_count = n_samples - success_count
            if failed_count > 0:
                print(f"     Failed: {failed_count} samples")

            # Show total files if there are previous runs
            total_in_dir = len(list(output_dir.glob('*.pkl')))
            if total_in_dir > success_count:
                print(f"     Total in directory: {total_in_dir} (includes previous runs)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate PINN training data from FEMM simulations'
    )
    parser.add_argument(
        '--problem-type',
        choices=['coil', 'transformer', 'ipm_motor'],
        required=True,
        help='Type of electromagnetic problem'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=100,
        help='Number of samples to generate (default: 100)'
    )
    parser.add_argument(
        '--obs-counts',
        nargs='+',
        type=int,
        default=[50, 100, 500],
        help='List of observation counts (default: 50 100 500)'
    )

    args = parser.parse_args()

    print(f"""
{'='*60}
PINN Data Generation
{'='*60}
Problem type: {args.problem_type}
Samples: {args.n_samples}
Observation counts: {args.obs_counts}
{'='*60}
    """)

    # Initialize solver
    solver = FEMMSolver(debug_mode=False)
    solver.open_femm(hidden=True)

    try:
        # Generate dataset
        generator = PINNDataGenerator(solver)
        generator.create_dataset(
            args.problem_type,
            args.n_samples,
            args.obs_counts
        )

        print(f"\n{'='*60}")
        print("Data generation complete!")
        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopping data generation...")
    except Exception as e:
        print(f"\n[ERROR] Data generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        solver.close_femm()


if __name__ == '__main__':
    main()
