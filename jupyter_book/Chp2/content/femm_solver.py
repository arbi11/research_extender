"""
FEMM Solver Integration Module

This module provides a high-level interface for integrating FEMM (Finite Element Method Magnetics)
into the electromagnetic data generation pipeline. It encapsulates FEMM operations and provides
methods for creating geometries, setting up materials, running analyses, and extracting results.

Author: Generated for electromagnetic data pipeline
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import pickle
from dataclasses import dataclass

# FEMM path - hardcoded as specified
FEMM_PATH = "C:\\femm42"

# FEMM is required - no fallbacks
import femm
FEMM_AVAILABLE = True


@dataclass
class FEMMResult:
    """Container for FEMM analysis results"""
    success: bool
    magnetic_field: Dict[str, np.ndarray]  # Bx, By, B_magnitude at various points
    forces: Dict[str, float]  # Force components
    flux_linkage: float  # Flux linkage for inductance calculations
    energy: float  # Magnetic energy
    torque: float = 0.0  # Torque for rotating machines
    error_message: str = ""
    analysis_time: float = 0.0


class FEMMSolver:
    """
    High-level FEMM solver interface for electromagnetic analysis

    This class manages FEMM operations including:
    - Solver lifecycle management
    - Geometry creation
    - Material property setup
    - Analysis execution
    - Result extraction
    """

    def __init__(self, femm_path: str = FEMM_PATH, debug_mode: bool = False):
        """
        Initialize FEMM solver

        Args:
            femm_path: Path to FEMM installation
            debug_mode: Enable debug mode for verbose logging
        """
        self.femm_path = Path(femm_path)
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(__name__)

        # FEMM state
        self.femm_open = False
        self.current_document = None

        # Material library
        self.materials = {}
        self.circuits = {}

        # Cache for results
        self.result_cache = {}

        if FEMM_AVAILABLE:
            self.logger.info("FEMM solver initialized successfully")
        else:
            self.logger.error("FEMM not available - install pyfemm and ensure FEMM is installed")

    def __enter__(self):
        """Context manager entry"""
        self.open_femm()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_femm()

    def open_femm(self, hidden: bool = True) -> bool:
        """
        Open FEMM instance

        Args:
            hidden: Run FEMM in hidden mode (no GUI)

        Returns:
            True if successful, False otherwise
        """
        if not FEMM_AVAILABLE:
            self.logger.error("FEMM not available")
            return False

        try:
            if self.debug_mode:
                hidden = False

            # Start FEMM with specified mode
            femm.openfemm(int(hidden))
            self.femm_open = True
            self.logger.info(f"FEMM opened (hidden={hidden})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to open FEMM: {e}")
            return False

    def close_femm(self) -> None:
        """Close FEMM instance"""
        if self.femm_open and FEMM_AVAILABLE:
            try:
                femm.closefemm()
                self.femm_open = False
                self.logger.info("FEMM closed")
            except Exception as e:
                self.logger.error(f"Error closing FEMM: {e}")

    def create_magnetics_document(self,
                                frequency: float = 0.0,
                                units: str = 'millimeters',
                                problem_type: str = 'planar',
                                precision: float = 1e-8,
                                depth: float = 1.0) -> bool:
        """
        Create a new magnetics document in FEMM

        Args:
            frequency: Analysis frequency (Hz)
            units: Length units ('millimeters', 'meters', etc.)
            problem_type: 'planar' or 'axisymmetric'
            precision: Solver precision
            depth: Depth for planar problems

        Returns:
            True if successful
        """
        if not self.femm_open:
            self.logger.error("FEMM not open")
            return False

        try:
            # Create new magnetics document
            femm.newdocument(0)  # 0 = magnetics

            # Define problem
            femm.mi_probdef(
                freq=frequency,
                units=units,
                type=problem_type,
                precision=precision,
                depth=depth,
                minangle=30
            )

            self.current_document = 'magnetics'
            self.logger.info(f"Created magnetics document: freq={frequency}Hz, type={problem_type}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create document: {e}")
            return False

    def setup_materials(self) -> None:
        """Setup common electromagnetic materials"""
        if not self.femm_open:
            return

        try:
            # Air
            femm.mi_getmaterial('Air')

            # Copper (windings)
            femm.mi_getmaterial('Copper')

            # Iron (magnetic steel)
            femm.mi_getmaterial('Cold rolled low carbon strip steel')

            # Add custom materials if needed
            # Silicon steel for transformers
            femm.mi_addmaterial('Silicon Steel',
                               3000, 3000,  # permeability
                               0, 0,  # Hc, J
                               2.0e6,  # conductivity
                               0.5,  # lamination thickness
                               0,  # hysteresis angle
                               0.95,  # lamination fill factor
                               1,  # lamination type
                               0, 0)  # hysteresis lag

            # NdFeB permanent magnets
            femm.mi_addmaterial('NdFeB_42',
                               1.05, 1.05,  # relative permeability
                               890000, 0,  # coercivity
                               0,  # conductivity
                               0, 0, 0, 0, 0, 0, 0, 0)  # other params

            self.logger.info("Materials setup complete")

        except Exception as e:
            self.logger.error(f"Error setting up materials: {e}")

    def add_circuit(self, name: str, current: float, circuit_type: int = 1) -> None:
        """
        Add electrical circuit

        Args:
            name: Circuit name
            current: Current in Amperes
            circuit_type: 0=parallel, 1=series
        """
        if not self.femm_open:
            return

        try:
            femm.mi_addcircprop(name, current, circuit_type)
            self.circuits[name] = {'current': current, 'type': circuit_type}
            self.logger.info(f"Added circuit {name}: I={current}A")
        except Exception as e:
            self.logger.error(f"Error adding circuit {name}: {e}")

    def create_coil_geometry(self,
                           center_x: float, center_y: float,
                           radius: float, wire_radius: float,
                           turns: int, circuit_name: str) -> bool:
        """
        Create a circular coil geometry

        Args:
            center_x, center_y: Center position
            radius: Coil radius
            wire_radius: Wire radius
            turns: Number of turns
            circuit_name: Circuit to connect to

        Returns:
            True if successful
        """
        if not self.femm_open:
            return False

        try:
            # Create coil as concentric circles
            for i in range(turns):
                r = radius - i * 2 * wire_radius
                if r <= 0:
                    break

                # Draw coil turn
                theta = np.linspace(0, 2*np.pi, 50)
                points = [(center_x + r * np.cos(t),
                          center_y + r * np.sin(t)) for t in theta]

                # Add line segments to approximate circle
                for j in range(len(points)-1):
                    femm.mi_addsegment(points[j][0], points[j][1],
                                      points[j+1][0], points[j+1][1])

            # Add block label for coil region
            femm.mi_addblocklabel(center_x, center_y)
            femm.mi_setblockprop('Copper', 0, 1.0, circuit_name, 0, turns, 0)

            self.logger.info(f"Created coil: r={radius:.3f}, turns={turns}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating coil: {e}")
            return False

    def create_rectangular_core(self,
                              x: float, y: float,
                              width: float, height: float,
                              material: str = 'Cold rolled low carbon strip steel') -> bool:
        """
        Create rectangular magnetic core

        Args:
            x, y: Bottom-left corner
            width, height: Dimensions
            material: Material name

        Returns:
            True if successful
        """
        if not self.femm_open:
            return False

        try:
            # Draw rectangle
            femm.mi_drawrectangle(x, y, x + width, y + height)

            # Add block label
            femm.mi_addblocklabel(x + width/2, y + height/2)
            femm.mi_setblockprop(material, 0, 1.0, '<None>', 0, 0, 0)

            return True

        except Exception as e:
            self.logger.error(f"Error creating rectangular core: {e}")
            return False

    def create_boundary_box(self,
                          x_min: float, y_min: float,
                          x_max: float, y_max: float,
                          boundary_type: str = 'dirichlet') -> bool:
        """
        Create boundary conditions

        Args:
            x_min, y_min, x_max, y_max: Boundary box coordinates
            boundary_type: 'dirichlet' or 'neumann'

        Returns:
            True if successful
        """
        if not self.femm_open:
            return False

        try:
            # Draw boundary
            femm.mi_drawrectangle(x_min, y_min, x_max, y_max)

            # Add boundary property
            if boundary_type == 'dirichlet':
                femm.mi_addboundprop('ZeroA', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            else:  # neumann
                femm.mi_addboundprop('ZeroH', 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0)

            # Set boundary property
            femm.mi_selectsegment(x_min, y_min)
            femm.mi_selectsegment(x_max, y_min)
            femm.mi_selectsegment(x_max, y_max)
            femm.mi_selectsegment(x_min, y_max)
            femm.mi_setsegmentprop(boundary_type, 1, 0, 0, 0)

            return True

        except Exception as e:
            self.logger.error(f"Error creating boundary: {e}")
            return False

    def analyze_and_extract_results(self,
                                  field_points: List[Tuple[float, float]] = None,
                                  integration_regions: List[int] = None) -> FEMMResult:
        """
        Run FEMM analysis and extract results

        Args:
            field_points: List of (x,y) points to evaluate field
            integration_regions: List of block labels for force/torque integration

        Returns:
            FEMMResult object with analysis results
        """
        if not self.femm_open:
            return FEMMResult(success=False, error_message="FEMM not open")

        result = FEMMResult(success=False)
        start_time = time.time()

        try:
            # Create mesh and analyze
            femm.mi_createmesh()

            # Run analysis
            femm.mi_analyze(1)  # 1 = hidden analysis

            # Load solution
            femm.mi_loadsolution()

            # Extract magnetic field at specified points
            if field_points:
                result.magnetic_field = self._extract_field_at_points(field_points)

            # Extract forces and torques
            if integration_regions:
                result.forces, result.torque = self._extract_forces(integration_regions)

            # Extract energy
            result.energy = femm.mo_blockintegral(2)  # Energy integral

            # Calculate flux linkage if circuits exist
            if self.circuits:
                result.flux_linkage = self._calculate_flux_linkage()

            result.success = True
            result.analysis_time = time.time() - start_time

            self.logger.info(f"Analysis completed in {result.analysis_time:.2f}s")

        except Exception as e:
            result.error_message = str(e)
            self.logger.error(f"Analysis failed: {e}")

        return result

    def _extract_field_at_points(self, points: List[Tuple[float, float]]) -> Dict[str, np.ndarray]:
        """Extract magnetic field at specified points"""
        Bx = []
        By = []
        B_mag = []

        for x, y in points:
            try:
                bx, by = femm.mo_getb(x, y)
                Bx.append(bx)
                By.append(by)
                B_mag.append(np.sqrt(bx**2 + by**2))
            except:
                Bx.append(0)
                By.append(0)
                B_mag.append(0)

        return {
            'Bx': np.array(Bx),
            'By': np.array(By),
            'B_magnitude': np.array(B_mag),
            'points': np.array(points)
        }

    def _extract_forces(self, regions: List[int]) -> Tuple[Dict[str, float], float]:
        """Extract forces and torque from specified regions"""
        forces = {'Fx': 0.0, 'Fy': 0.0, 'Fz': 0.0}
        torque = 0.0

        for region in regions:
            try:
                femm.mo_groupselectblock(region)
                # Force integral (type 19)
                fx, fy = femm.mo_blockintegral(19)
                forces['Fx'] += fx
                forces['Fy'] += fy

                # Torque integral (type 22)
                torque += femm.mo_blockintegral(22)

            except:
                continue

        return forces, torque

    def _calculate_flux_linkage(self) -> float:
        """Calculate flux linkage for circuits"""
        try:
            # Get circuit properties
            if not self.circuits:
                return 0.0

            circuit_name = list(self.circuits.keys())[0]
            flux = femm.mo_getcircuitproperties(circuit_name)[2]  # Flux linkage
            return flux

        except:
            return 0.0

    def save_geometry(self, filename: str) -> bool:
        """Save current geometry to file"""
        if not self.femm_open:
            return False

        try:
            femm.mi_saveas(filename)
            self.logger.info(f"Geometry saved to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving geometry: {e}")
            return False

    def load_geometry(self, filename: str) -> bool:
        """Load geometry from file"""
        if not self.femm_open:
            return False

        try:
            femm.opendocument(filename)
            self.logger.info(f"Geometry loaded from {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading geometry: {e}")
            return False


class CoaxialCoilSolver(FEMMSolver):
    """Specialized solver for coaxial coil problems"""

    def analyze_coil(self,
                    radius: float, turns: int, current: float,
                    wire_radius: float = 0.001,
                    analysis_points: List[Tuple[float, float]] = None) -> FEMMResult:
        """
        Analyze coaxial coil configuration

        Args:
            radius: Coil radius (m)
            turns: Number of turns
            current: Coil current (A)
            wire_radius: Wire radius (m)
            analysis_points: Points to evaluate field

        Returns:
            FEMM analysis results
        """
        # Create new document
        self.create_magnetics_document(frequency=0.0)
        self.setup_materials()

        # Add circuit
        self.add_circuit('coil', current)

        # Create coil geometry
        self.create_coil_geometry(0, 0, radius, wire_radius, turns, 'coil')

        # Create boundary box (2x coil size)
        boundary_size = radius * 4
        self.create_boundary_box(-boundary_size, -boundary_size,
                                boundary_size, boundary_size)

        # Default analysis points if not provided
        if analysis_points is None:
            analysis_points = []
            # Points along axis
            for z in np.linspace(-radius*2, radius*2, 20):
                analysis_points.append((0, z))
            # Points in radial direction
            for r in np.linspace(0, radius*2, 20):
                analysis_points.append((r, 0))

        # Run analysis
        return self.analyze_and_extract_results(field_points=analysis_points)


class TransformerSolver(FEMMSolver):
    """Specialized solver for transformer problems"""

    def analyze_transformer(self,
                          primary_turns: int, secondary_turns: int,
                          core_area: float, frequency: float,
                          primary_voltage: float,
                          window_width: float = None,
                          window_height: float = None) -> FEMMResult:
        """
        Analyze transformer configuration

        Args:
            primary_turns: Number of primary turns
            secondary_turns: Number of secondary turns
            core_area: Core cross-sectional area (m²)
            frequency: Operating frequency (Hz)
            primary_voltage: Primary voltage (V)
            window_width: Transformer window width
            window_height: Transformer window height

        Returns:
            FEMM analysis results
        """
        # Calculate dimensions from core area
        core_side = np.sqrt(core_area)
        if window_width is None:
            window_width = core_side * 0.5
        if window_height is None:
            window_height = core_side * 1.5

        # Create new document
        self.create_magnetics_document(frequency=frequency)
        self.setup_materials()

        # Add circuits
        # Calculate primary current from voltage and impedance (simplified)
        primary_current = primary_voltage / 100  # Simplified impedance
        self.add_circuit('primary', primary_current)
        self.add_circuit('secondary', 0)  # No load condition

        # Create E-I core geometry
        self._create_ei_core(core_side, window_width, window_height)

        # Create windings
        self._create_windings(core_side, window_width, window_height,
                            primary_turns, secondary_turns)

        # Create boundary
        boundary_size = core_side * 3
        self.create_boundary_box(-boundary_size, -boundary_size,
                                boundary_size, boundary_size)

        # Define analysis points (in air gap and core)
        analysis_points = []
        # Points in core
        for x in np.linspace(-core_side, core_side, 10):
            analysis_points.append((x, 0))
        # Points in window
        for x in np.linspace(-window_width/2, window_width/2, 10):
            analysis_points.append((x, 0))

        # Run analysis
        return self.analyze_and_extract_results(field_points=analysis_points)

    def _create_ei_core(self, core_side: float, window_width: float, window_height: float) -> None:
        """Create E-I core geometry"""
        # E-core
        self.create_rectangular_core(-core_side*1.5, -window_height/2,
                                   core_side*0.3, window_height)
        self.create_rectangular_core(-core_side*0.5, -window_height/2,
                                   core_side*0.3, window_height)
        self.create_rectangular_core(core_side*0.2, -window_height/2,
                                   core_side*0.3, window_height)
        self.create_rectangular_core(-core_side*1.5, -window_height/2,
                                   core_side*2.0, core_side*0.2)
        self.create_rectangular_core(-core_side*1.5, window_height/2-core_side*0.2,
                                   core_side*2.0, core_side*0.2)

        # I-core
        self.create_rectangular_core(core_side*0.8, -window_height/2,
                                   core_side*0.2, window_height)

    def _create_windings(self, core_side: float, window_width: float, window_height: float,
                        primary_turns: int, secondary_turns: int) -> None:
        """Create primary and secondary windings"""
        # Primary winding (left window)
        primary_width = window_width * 0.8
        primary_height = window_height * 0.4
        femm.mi_drawrectangle(-core_side*0.4-primary_width/2, -primary_height/2,
                             primary_width, primary_height)
        femm.mi_addblocklabel(-core_side*0.4, 0)
        femm.mi_setblockprop('Copper', 0, 1.0, 'primary', 0, primary_turns, 0)

        # Secondary winding (right window)
        secondary_width = window_width * 0.6
        secondary_height = window_height * 0.3
        femm.mi_drawrectangle(core_side*0.35-secondary_width/2, -secondary_height/2,
                             secondary_width, secondary_height)
        femm.mi_addblocklabel(core_side*0.35, 0)
        femm.mi_setblockprop('Copper', 0, 1.0, 'secondary', 0, secondary_turns, 0)


class IPMMotorSolver(FEMMSolver):
    """Specialized solver for IPM motor problems"""

    def analyze_ipm_motor(self,
                         stator_slots: int, rotor_poles: int,
                         stator_outer_radius: float, stator_inner_radius: float,
                         rotor_inner_radius: float, air_gap: float,
                         magnet_strength: float, rotor_position: float,
                         current_amplitude: float) -> FEMMResult:
        """
        Analyze IPM motor configuration

        Args:
            stator_slots: Number of stator slots
            rotor_poles: Number of rotor poles
            stator_outer_radius: Stator outer radius (m)
            stator_inner_radius: Stator inner radius (m)
            rotor_inner_radius: Rotor inner radius (m)
            air_gap: Air gap length (m)
            magnet_strength: Magnet remanence (T)
            rotor_position: Rotor position (degrees)
            current_amplitude: Stator current amplitude (A)

        Returns:
            FEMM analysis results
        """
        rotor_outer_radius = stator_inner_radius - air_gap
        rotor_pos_rad = np.radians(rotor_position)

        # Create new document
        self.create_magnetics_document(frequency=50.0)  # 50 Hz electrical
        self.setup_materials()

        # Add three-phase circuits
        for phase in ['A', 'B', 'C']:
            # Phase currents shifted by 120 degrees
            phase_angle = 120 * (ord(phase) - ord('A'))
            current = current_amplitude * np.cos(np.radians(phase_angle))
            self.add_circuit(f'phase_{phase}', current)

        # Create motor geometry
        self._create_stator(stator_outer_radius, stator_inner_radius, stator_slots)
        self._create_rotor(rotor_outer_radius, rotor_inner_radius, rotor_poles,
                          magnet_strength, rotor_pos_rad)

        # Create boundary
        boundary_size = stator_outer_radius * 1.5
        self.create_boundary_box(-boundary_size, -boundary_size,
                                boundary_size, boundary_size)

        # Define analysis points
        analysis_points = []
        # Points in air gap
        for theta in np.linspace(0, 2*np.pi, 36):
            r = (stator_inner_radius + rotor_outer_radius) / 2
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            analysis_points.append((x, y))

        # Define integration regions for torque calculation
        integration_regions = [5]  # Rotor region

        # Run analysis
        return self.analyze_and_extract_results(field_points=analysis_points,
                                              integration_regions=integration_regions)

    def _create_stator(self, outer_radius: float, inner_radius: float, slots: int) -> None:
        """Create stator geometry with slots"""
        # Draw stator outer and inner circles
        theta = np.linspace(0, 2*np.pi, 100)

        # Approximate circles with line segments
        for i in range(len(theta)-1):
            x1_outer = outer_radius * np.cos(theta[i])
            y1_outer = outer_radius * np.sin(theta[i])
            x2_outer = outer_radius * np.cos(theta[i+1])
            y2_outer = outer_radius * np.sin(theta[i+1])
            femm.mi_addsegment(x1_outer, y1_outer, x2_outer, y2_outer)

            x1_inner = inner_radius * np.cos(theta[i])
            y1_inner = inner_radius * np.sin(theta[i])
            x2_inner = inner_radius * np.cos(theta[i+1])
            y2_inner = inner_radius * np.sin(theta[i+1])
            femm.mi_addsegment(x1_inner, y1_inner, x2_inner, y2_inner)

        # Create slots
        slot_width = 2 * np.pi / slots * 0.3
        for i in range(slots):
            slot_angle = i * 2 * np.pi / slots

            # Slot corners
            r1 = inner_radius * 0.95
            r2 = outer_radius * 0.95

            corners = [
                (r1 * np.cos(slot_angle - slot_width/2), r1 * np.sin(slot_angle - slot_width/2)),
                (r2 * np.cos(slot_angle - slot_width/2), r2 * np.sin(slot_angle - slot_width/2)),
                (r2 * np.cos(slot_angle + slot_width/2), r2 * np.sin(slot_angle + slot_width/2)),
                (r1 * np.cos(slot_angle + slot_width/2), r1 * np.sin(slot_angle + slot_width/2))
            ]

            # Draw slot
            for j in range(len(corners)):
                next_j = (j + 1) % len(corners)
                femm.mi_addsegment(corners[j][0], corners[j][1],
                                  corners[next_j][0], corners[next_j][1])

        # Add stator material
        femm.mi_addblocklabel(0, 0)
        femm.mi_setblockprop('Silicon Steel', 0, 1.0, '<None>', 0, 0, 0)

    def _create_rotor(self, outer_radius: float, inner_radius: float, poles: int,
                     magnet_strength: float, rotor_position: float) -> None:
        """Create rotor with permanent magnets"""
        theta = np.linspace(0, 2*np.pi, 100)

        # Draw rotor circles (rotated by rotor_position)
        for i in range(len(theta)-1):
            # Apply rotor position rotation
            angle1 = theta[i] + rotor_position
            angle2 = theta[i+1] + rotor_position

            x1_outer = outer_radius * np.cos(angle1)
            y1_outer = outer_radius * np.sin(angle1)
            x2_outer = outer_radius * np.cos(angle2)
            y2_outer = outer_radius * np.sin(angle2)
            femm.mi_addsegment(x1_outer, y1_outer, x2_outer, y2_outer)

            x1_inner = inner_radius * np.cos(angle1)
            y1_inner = inner_radius * np.sin(angle1)
            x2_inner = inner_radius * np.cos(angle2)
            y2_inner = inner_radius * np.sin(angle2)
            femm.mi_addsegment(x1_inner, y1_inner, x2_inner, y2_inner)

        # Create permanent magnets
        magnet_width = 2 * np.pi / poles * 0.6
        for i in range(poles):
            magnet_angle = i * 2 * np.pi / poles + rotor_position

            # V-shaped magnet
            magnet_r1 = inner_radius + (outer_radius - inner_radius) * 0.3
            magnet_r2 = outer_radius * 0.9

            # Magnet corners (V-shape)
            corners = [
                (magnet_r1 * np.cos(magnet_angle - magnet_width/2),
                 magnet_r1 * np.sin(magnet_angle - magnet_width/2)),
                (magnet_r2 * np.cos(magnet_angle - magnet_width/4),
                 magnet_r2 * np.sin(magnet_angle - magnet_width/4)),
                (magnet_r2 * np.cos(magnet_angle + magnet_width/4),
                 magnet_r2 * np.sin(magnet_angle + magnet_width/4)),
                (magnet_r1 * np.cos(magnet_angle + magnet_width/2),
                 magnet_r1 * np.sin(magnet_angle + magnet_width/2))
            ]

            # Draw magnet
            for j in range(len(corners)):
                next_j = (j + 1) % len(corners)
                femm.mi_addsegment(corners[j][0], corners[j][1],
                                  corners[next_j][0], corners[next_j][1])

            # Add magnet material with alternating polarity
            polarity = 1 if i % 2 == 0 else -1
            mag_dir = np.degrees(magnet_angle)
            femm.mi_addblocklabel(
                (magnet_r1 + magnet_r2)/2 * np.cos(magnet_angle),
                (magnet_r1 + magnet_r2)/2 * np.sin(magnet_angle)
            )
            femm.mi_setblockprop('NdFeB_42', 0, 1.0, '<None>', mag_dir * polarity, 3, 0)

        # Add rotor core material
        femm.mi_addblocklabel(0, 0)
        femm.mi_setblockprop('Silicon Steel', 0, 1.0, '<None>', 0, 5, 0)


# Test function to verify FEMM integration
def test_femm_integration():
    """Test basic FEMM integration functionality"""
    print("Testing FEMM integration...")

    if not FEMM_AVAILABLE:
        print("FEMM not available - install pyfemm")
        return False

    try:
        # Test simple coil analysis
        with CoaxialCoilSolver(debug_mode=True) as solver:
            result = solver.analyze_coil(
                radius=0.05,  # 50mm
                turns=50,
                current=5.0,  # 5A
                wire_radius=0.001  # 1mm
            )

            if result.success:
                print(f"✓ Coil analysis successful")
                print(f"  Analysis time: {result.analysis_time:.2f}s")
                print(f"  Max B-field: {np.max(result.magnetic_field['B_magnitude']):.4f} T")
                print(f"  Energy: {result.energy:.6f} J")
                return True
            else:
                print(f"✗ Coil analysis failed: {result.error_message}")
                return False

    except Exception as e:
        print(f"✗ FEMM test failed: {e}")
        return False


if __name__ == "__main__":
    # Run test if executed directly
    test_femm_integration()