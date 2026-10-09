"""
Multi-Problem CNN Training Data Generation

Generates raw FEM mesh data for CNN training on electromagnetic problems.
Supports: coil, transformer, IPM motor, C-core

Usage:
    python generate_cnn_training_data.py --problem-type ipm --num-samples 1000
    python generate_cnn_training_data.py --problem-type coil --num-samples 500 --resume
    python generate_cnn_training_data.py --problem-type transformer --num-samples 800 --output my_tf_data
"""

import argparse
import numpy as np
import pickle
import json
from pathlib import Path
from scipy.stats import qmc
from dataclasses import dataclass
from tqdm import tqdm
import femm
from femm_solver import FEMMSolver


@dataclass
class CoilParams:
    wire_radius: float # m
    current: float     # A


@dataclass
class TransformerParams:
    core_area: float       # m²
    primary_turns: int     # dimensionless
    secondary_turns: int   # dimensionless
    frequency: float       # Hz


@dataclass
class IPMParams:
    magnet_length: float       # mm
    magnet_width: float        # mm
    slot_depth: float          # mm
    current_amplitude: float   # A


@dataclass
class CCoreParams:
    coil_current: float    # A
    armature_gap: float    # mm
    core_width: float      # mm
    coil_turns: int        # dimensionless


# Problem configurations
PROBLEM_CONFIGS = {
    'coil': {
        'n_params': 2,
        'param_class': CoilParams,
        'param_names': ['wire_radius', 'current'],
        'param_ranges': {
            'wire_radius_m': [5e-3, 50e-3],  # 5mm to 50mm (much stronger fields)
            'current_a': [10.0, 100.0]       # 10A to 100A (stronger excitation)
        }
    },
    'transformer': {
        'n_params': 4,
        'param_class': TransformerParams,
        'param_names': ['core_area', 'primary_turns', 'secondary_turns', 'frequency'],
        'param_ranges': {
            'core_area_m2': [0.0001, 0.001],
            'primary_turns': [50, 200],
            'secondary_turns': [10, 100],
            'frequency_hz': [50, 400]
        }
    },
    'ipm': {
        'n_params': 4,
        'param_class': IPMParams,
        'param_names': ['magnet_length', 'magnet_width', 'slot_depth', 'current_amplitude'],
        'param_ranges': {
            'magnet_length_mm': [30.0, 50.0],   # Scaled 2×: 15-25 → 30-50mm
            'magnet_width_mm': [10.0, 24.0],    # Scaled 2×: 5-12 → 10-24mm
            'slot_depth_mm': [40.0, 60.0],      # Scaled 2×: 20-30 → 40-60mm
            'current_amplitude_a': [5.0, 30.0]
        }
    },
    'c_core': {
        'n_params': 4,
        'param_class': CCoreParams,
        'param_names': ['coil_current', 'armature_gap', 'core_width', 'coil_turns'],
        'param_ranges': {
            'coil_current_a': [-50.0, 50.0],
            'armature_gap_mm': [2.5, 25.0],      # Scaled 5×: 0.5-5 → 2.5-25mm
            'core_width_mm': [150.0, 225.0],     # Scaled 5×: 30-45 → 150-225mm
            'coil_turns': [300, 700]
        }
    }
}


def generate_lhs_samples(n_samples, n_dims, seed=42):
    """Generate LHS samples in [0,1]^n_dims"""
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    return sampler.random(n=n_samples)


def lhs_to_params(u, problem_type):
    """Map [0,1]^d to problem-specific parameter ranges"""
    if problem_type == 'coil':
        return CoilParams(
            wire_radius=5e-3 + (50e-3 - 5e-3) * u[0],  # 5mm to 50mm
            current=10 + (100 - 10) * u[1]             # 10A to 100A
        )
    elif problem_type == 'transformer':
        return TransformerParams(
            core_area=0.0001 + (0.001 - 0.0001) * u[0],
            primary_turns=int(50 + (200 - 50) * u[1]),
            secondary_turns=int(10 + (100 - 10) * u[2]),
            frequency=50 + (400 - 50) * u[3]
        )
    elif problem_type == 'ipm':
        return IPMParams(
            magnet_length=30 + (50 - 30) * u[0],   # Scaled 2×: 30-50mm
            magnet_width=10 + (24 - 10) * u[1],    # Scaled 2×: 10-24mm
            slot_depth=40 + (60 - 40) * u[2],      # Scaled 2×: 40-60mm
            current_amplitude=5 + (30 - 5) * u[3]
        )
    elif problem_type == 'c_core':
        return CCoreParams(
            coil_current=-50 + (50 - (-50)) * u[0],
            armature_gap=2.5 + (25.0 - 2.5) * u[1],    # Scaled 5×: 2.5-25mm
            core_width=150 + (225 - 150) * u[2],       # Scaled 5×: 150-225mm
            coil_turns=int(300 + (700 - 300) * u[3])
        )


def load_checkpoint(checkpoint_path):
    """Load existing checkpoint if it exists"""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        return {
            'samples': data['samples'],
            'n_completed': len(data['samples']),
            'failed': data.get('failed', [])
        }
    return {'samples': [], 'n_completed': 0, 'failed': []}


def save_checkpoint(output_path, checkpoint_path, metadata_path, samples, failed_samples, metadata):
    """
    Save checkpoint AND update main dataset files.

    This allows training to start immediately on partial data without waiting for completion.
    Both checkpoint and main dataset have identical format (just list of samples).
    """
    # Save main dataset file (trainable format - just the samples list)
    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)

    # Save checkpoint backup (includes failed samples for resume)
    checkpoint_data = {
        'samples': samples,
        'failed': failed_samples
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)

    # Save/update metadata JSON (always available)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def run_simulation(solver, problem_type, params):
    """Dispatch to appropriate solver method based on problem type"""
    if problem_type == 'coil':
        return solver.analyze_coil(
            current=params.current,
            wire_radius=params.wire_radius
        )
    elif problem_type == 'transformer':
        mesh_data = solver.analyze_transformer(
            primary_turns=params.primary_turns,
            secondary_turns=params.secondary_turns,
            core_area=params.core_area,
            frequency=params.frequency,
            primary_voltage=230.0  # Fixed voltage (matches notebook demo)
        )

        # Calculate primary current (same formula as in solver)
        if params.frequency > 0:
            primary_current = 230.0 / (2 * np.pi * params.frequency *
                                        params.primary_turns * params.core_area)
            if abs(primary_current) < 0.1:
                primary_current = 10.0
        else:
            primary_current = 10.0

        # Store calculated current as additional metadata
        mesh_data.primary_current = primary_current
        return mesh_data
    elif problem_type == 'ipm':
        mesh_data = solver.analyze_ipm_motor(
            stator_slots=12,
            rotor_poles=10,
            stator_outer_radius=0.2,      # Scaled 2×: 0.1 → 0.2m (400mm diameter)
            stator_inner_radius=0.12,     # Scaled 2×: 0.06 → 0.12m
            rotor_inner_radius=0.06,      # Scaled 2×: 0.03 → 0.06m
            air_gap=0.002,                # Scaled 2×: 0.001 → 0.002m
            magnet_strength=1.2,
            rotor_position=0.0,
            current_amplitude=params.current_amplitude,
            magnet_length=params.magnet_length,
            magnet_width=params.magnet_width,
            slot_depth=params.slot_depth
        )

        # Store winding configuration (fixed values)
        mesh_data.turns_per_slot = 50
        mesh_data.active_phase = 'A'
        mesh_data.slots_per_phase = 4
        return mesh_data
    elif problem_type == 'c_core':
        return solver.analyze_c_core_armature(
            coil_current=params.coil_current,
            armature_gap=params.armature_gap,
            core_width=params.core_width,
            coil_turns=params.coil_turns
        )


def extract_excitation_metadata(problem_type, params, mesh_data):
    """Extract excitation information for ampere-turns calculation"""
    if problem_type == 'coil':
        return {
            'current': params.current,
            'turns': 1,  # Straight wire
            'ampere_turns': params.current * 1
        }

    elif problem_type == 'transformer':
        primary_current = getattr(mesh_data, 'primary_current', None)
        return {
            'primary_current': primary_current,
            'primary_turns': params.primary_turns,
            'secondary_turns': params.secondary_turns,
            'primary_ampere_turns': 2 * primary_current * params.primary_turns if primary_current else None,
            'num_primary_coils': 2,
            'num_secondary_coils': 2
        }

    elif problem_type == 'ipm':
        turns_per_slot = getattr(mesh_data, 'turns_per_slot', 50)
        slots_per_phase = getattr(mesh_data, 'slots_per_phase', 4)
        return {
            'current_amplitude': params.current_amplitude,
            'turns_per_slot': turns_per_slot,
            'slots_per_phase': slots_per_phase,
            'phase_A_ampere_turns': params.current_amplitude * turns_per_slot * slots_per_phase,
            'active_phase': 'A'
        }

    elif problem_type == 'c_core':
        return {
            'coil_current': params.coil_current,
            'coil_turns': params.coil_turns,
            'ampere_turns': params.coil_current * params.coil_turns,
            'num_coils': 2,
            'configuration': 'push-pull'
        }


def main():
    parser = argparse.ArgumentParser(
        description='Generate CNN training dataset for electromagnetic problems'
    )
    parser.add_argument('--problem-type', type=str, required=True,
                       choices=['coil', 'transformer', 'ipm', 'c_core'],
                       help='Type of electromagnetic problem to simulate')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--save-interval', type=int, default=50,
                       help='Save checkpoint every N samples')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename (default: {problem_type}_cnn_dataset)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for LHS sampling')
    args = parser.parse_args()

    # Set default output name based on problem type
    if args.output is None:
        args.output = f'{args.problem_type}_cnn_dataset'

    # Create dataset directory if it doesn't exist
    dataset_dir = Path('dataset')
    dataset_dir.mkdir(exist_ok=True)

    # Set output paths in dataset directory
    output_path = dataset_dir / f'{args.output}.pkl'
    checkpoint_path = dataset_dir / f'{args.output}_checkpoint.pkl'
    metadata_path = dataset_dir / f'{args.output}_metadata.json'

    # Get problem configuration
    config = PROBLEM_CONFIGS[args.problem_type]
    n_dims = config['n_params']

    print("=" * 80)
    print(f"CNN TRAINING DATA GENERATION: {args.problem_type.upper()}")
    print("=" * 80)
    print(f"Parameters: {', '.join(config['param_names'])}")
    print(f"Target samples: {args.num_samples}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)

    # Generate or load LHS samples
    all_lhs_samples = generate_lhs_samples(args.num_samples, n_dims, args.seed)

    # Load checkpoint if resuming
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        samples = checkpoint['samples']
        failed_samples = checkpoint['failed']
        start_idx = checkpoint['n_completed']
        print(f"\n[RESUME] Continuing from {start_idx}/{args.num_samples} samples\n")
    else:
        samples = []
        failed_samples = []
        start_idx = 0

    # Save initial metadata (so it's visible immediately)
    initial_metadata = {
        'problem_type': args.problem_type,
        'num_samples': len(samples),
        'num_requested': args.num_samples,
        'num_failed': len(failed_samples),
        'failed_sample_ids': failed_samples,
        'parameter_ranges': config['param_ranges'],
        'feature_names': config['param_names'],
        'num_parameters': config['n_params'],
        'seed': args.seed,
        'timestamp': str(np.datetime64('now')),
        'status': 'in_progress' if start_idx < args.num_samples else 'complete',
        'bounds_mm': [-500, 500, -500, 500]  # Uniform 500mm domain for all problems
    }
    with open(metadata_path, 'w') as f:
        json.dump(initial_metadata, f, indent=2)

    # Initialize FEMM
    femm.openfemm(1)  # Hide window
    solver = FEMMSolver(debug_mode=False)

    # Main loop
    for i in tqdm(range(start_idx, args.num_samples), initial=start_idx, total=args.num_samples, desc="Generating"):
        params = lhs_to_params(all_lhs_samples[i], args.problem_type)

        try:
            # Run FEMM simulation (problem-specific)
            mesh_data = run_simulation(solver, args.problem_type, params)

            if mesh_data.success:
                # Convert params dataclass to dict
                param_dict = {name: getattr(params, name)
                             for name in config['param_names']}

                # Save complete mesh_data object (compatible with MeshInterpolator)
                sample = {
                    'sample_id': i,
                    'problem_type': args.problem_type,
                    'parameters': param_dict,
                    'mesh_data': mesh_data,  # Save entire FEMMMeshData object
                    'excitation': extract_excitation_metadata(args.problem_type, params, mesh_data)  # Ampere-turns info
                }
                samples.append(sample)
            else:
                print(f"\n[FAILED] Sample {i}: {mesh_data.error_message}")
                failed_samples.append(i)

        except Exception as e:
            print(f"\n[ERROR] Sample {i}: {str(e)}")
            failed_samples.append(i)

        # Checkpoint save (updates main dataset + metadata for immediate training)
        if (i + 1) % args.save_interval == 0:
            checkpoint_metadata = {
                'problem_type': args.problem_type,
                'num_samples': len(samples),
                'num_requested': args.num_samples,
                'num_failed': len(failed_samples),
                'failed_sample_ids': failed_samples,
                'parameter_ranges': config['param_ranges'],
                'feature_names': config['param_names'],
                'num_parameters': config['n_params'],
                'seed': args.seed,
                'timestamp': str(np.datetime64('now')),
                'status': 'in_progress',
                'bounds_mm': [-500, 500, -500, 500]  # Uniform 500mm domain for all problems
            }
            save_checkpoint(output_path, checkpoint_path, metadata_path,
                          samples, failed_samples, checkpoint_metadata)
            tqdm.write(f"[CHECKPOINT] Saved {i+1}/{args.num_samples} ({len(failed_samples)} failed)")

    # Final save (mark as complete)
    final_metadata = {
        'problem_type': args.problem_type,
        'num_samples': len(samples),
        'num_requested': args.num_samples,
        'num_failed': len(failed_samples),
        'failed_sample_ids': failed_samples,
        'parameter_ranges': config['param_ranges'],
        'feature_names': config['param_names'],
        'num_parameters': config['n_params'],
        'seed': args.seed,
        'timestamp': str(np.datetime64('now')),
        'status': 'complete',
        'bounds_mm': [-500, 500, -500, 500]  # Uniform 500mm domain for all problems
    }

    # Save final dataset and metadata
    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)
    with open(metadata_path, 'w') as f:
        json.dump(final_metadata, f, indent=2)

    solver.close_femm()

    # Summary
    print(f"\n{'='*80}")
    print(f"DATA GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Problem type:    {args.problem_type}")
    print(f"Output file:     {output_path}")
    print(f"Metadata file:   {metadata_path}")
    print(f"Successful:      {len(samples)}/{args.num_samples} samples")
    print(f"Failed:          {len(failed_samples)} samples")
    if len(samples) > 0:
        sample_nodes = [s['mesh_data'].num_nodes for s in samples]
        print(f"Mesh size range: {min(sample_nodes)}-{max(sample_nodes)} nodes")
        print(f"Parameters:      {', '.join(config['param_names'])}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
