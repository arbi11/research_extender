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
from scipy.interpolate import griddata

# Initialize COM for pyFEMM (required for Windows COM automation)
import pythoncom
pythoncom.CoInitialize()

# FEMM path - hardcoded as specified
FEMM_PATH = "C:\\femm42"

# FEMM is required - no fallbacks
import femm
FEMM_AVAILABLE = True


@dataclass
class FEMMResult:
    """Container for FEMM analysis results

    Supports both sparse point sampling (backward compatible) and dense grid sampling (for CNN training).

    Attributes:
        success: Whether the analysis completed successfully
        magnetic_field: Magnetic field components (can be sparse array or 2D grid)
        forces: Force components dict
        flux_linkage: Flux linkage for inductance calculations
        energy: Magnetic energy
        torque: Torque for rotating machines
        error_message: Error description if failure
        analysis_time: Computation time in seconds

        # Grid-based field data (optional, for CNN training):
        grid_coordinates: (xx, yy) meshgrid arrays if grid sampling used
        semantic_mask: 2D array of material IDs at each grid point
        grid_resolution: Grid size (e.g., 800 for 800×800)
        grid_bounds: (xmin, xmax, ymin, ymax) in mm
        material_labels: Mapping from material IDs to names
    """
    success: bool
    magnetic_field: Dict[str, np.ndarray]  # Bx, By, B_magnitude at various points
    forces: Dict[str, float]  # Force components
    flux_linkage: float  # Flux linkage for inductance calculations
    energy: float  # Magnetic energy
    torque: float = 0.0  # Torque for rotating machines
    error_message: str = ""
    analysis_time: float = 0.0

    # NEW: Grid-based field data (optional, for CNN training)
    grid_coordinates: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (xx, yy) meshgrid
    semantic_mask: Optional[np.ndarray] = None  # Material ID at each grid point (0=Air, 1=Iron, 2=Copper, 3=Magnet)
    grid_resolution: Optional[int] = None  # e.g., 800 for 800×800 grid
    grid_bounds: Optional[Tuple[float, float, float, float]] = None  # (xmin, xmax, ymin, ymax) in mm
    material_labels: Optional[Dict[int, str]] = None  # {0: 'Air', 1: 'Iron', 2: 'Copper', 3: 'NdFeB'}


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

    def open_femm(self, hidden: bool = False) -> bool:
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

        # Create new magnetics document
        femm.newdocument(0)  # 0 = magnetics

        # Define problem
        femm.mi_probdef(
            frequency,  # freq
            units,      # units
            problem_type,  # type
            precision,  # precision
            depth,      # depth
            30          # minangle
        )

        self.current_document = 'magnetics'
        self.logger.info(f"Created magnetics document: freq={frequency}Hz, type={problem_type}")
        return True


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

        femm.mi_addcircprop(name, current, circuit_type)
        self.circuits[name] = {'current': current, 'type': circuit_type}
        self.logger.info(f"Added circuit {name}: I={current}A")


    def create_coil_geometry(self,
                           center_x: float, center_y: float,
                           radius: float, wire_radius: float,
                           x: float, y: float,
                           width: float, height: float,
                           turns: int, circuit_name: str) -> bool:
        """
        Create a circular coil geometry
        Create a rectangular coil winding geometry.

        Args:
            center_x, center_y: Center position
            radius: Coil radius
            wire_radius: Wire radius
            x, y: Bottom-left corner of the coil block
            width, height: Dimensions of the coil block
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
            # Draw a rectangle representing the entire coil winding pack
            femm.mi_drawrectangle(x, y, x + width, y + height)

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
            # Add a block label inside the coil region and assign properties
            femm.mi_addblocklabel(x + width / 2, y + height / 2)
            femm.mi_setblockprop('Copper', 0, 1.0, circuit_name, 0, turns, 0)

            self.logger.info(f"Created coil: r={radius:.3f}, turns={turns}")
            self.logger.info(f"Created coil block: turns={turns}")
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
                                  integration_regions: List[int] = None,
                                  grid_resolution: Optional[int] = None,
                                  grid_bounds: Optional[Tuple[float, float, float, float]] = None,
                                  generate_mask: bool = True) -> FEMMResult:
        """
        Run FEMM analysis and extract results

        Supports both sparse point sampling (backward compatible) and dense grid sampling (for CNN training).

        Args:
            field_points: List of (x,y) points to evaluate field (sparse sampling)
            integration_regions: List of block labels for force/torque integration
            grid_resolution: If specified, sample field on uniform grid (e.g., 800 for 800×800)
            grid_bounds: Grid domain bounds (xmin, xmax, ymin, ymax) in mm
            generate_mask: Whether to generate semantic mask (only used if grid_resolution is set)

        Returns:
            FEMMResult object with analysis results

        Example:
            # Sparse sampling (backward compatible):
            result = solver.analyze_and_extract_results(field_points=[(0,0), (10,10)])

            # Grid sampling with semantic mask:
            result = solver.analyze_and_extract_results(
                grid_resolution=200,
                grid_bounds=(-50, 50, -50, 50),
                generate_mask=True
            )
        """
        if not self.femm_open:
            return FEMMResult(success=False, error_message="FEMM not open",
                            magnetic_field={}, forces={}, flux_linkage=0.0, energy=0.0)

        result = FEMMResult(success=False, magnetic_field={}, forces={}, flux_linkage=0.0, energy=0.0)
        start_time = time.time()

        try:
            # Save geometry to temp001.fem for debugging purposes
            femm.mi_saveas("temp001.fem")
            self.logger.info("Geometry saved to temp001.fem for debugging")

            # Create mesh and analyze
            self.logger.info("Creating mesh...")
            femm.mi_createmesh()

            # Run analysis
            self.logger.info("Running analysis...")
            femm.mi_analyze(0)  # 0 = visible analysis for debugging

            # Load solution
            self.logger.info("Loading solution...")
            femm.mi_loadsolution()

            # Extract magnetic field
            if grid_resolution is not None:
                # NEW: Grid-based sampling
                self.logger.info(f"Extracting field on {grid_resolution}×{grid_resolution} grid...")
                field_data = self._extract_field_at_points(
                    grid_resolution=grid_resolution,
                    grid_bounds=grid_bounds
                )

                result.magnetic_field = {
                    'Bx': field_data['Bx'],
                    'By': field_data['By'],
                    'B_magnitude': field_data['B_magnitude']
                }

                # Store grid metadata
                result.grid_coordinates = field_data['grid_coordinates']
                result.grid_resolution = grid_resolution
                result.grid_bounds = grid_bounds

                # Generate semantic mask if requested
                if generate_mask:
                    self.logger.info("Generating semantic mask...")
                    xx, yy = field_data['grid_coordinates']
                    result.semantic_mask, result.material_labels = self._generate_semantic_mask(xx, yy)

            elif field_points:
                # EXISTING: Sparse point sampling (backward compatible)
                self.logger.info(f"Extracting field at {len(field_points)} points...")
                result.magnetic_field = self._extract_field_at_points(points=field_points)

            # Extract forces and torques
            if integration_regions:
                self.logger.info("Extracting forces and torques...")
                result.forces, result.torque = self._extract_forces(integration_regions)
            else:
                result.forces = {}
                result.torque = 0.0

            # Extract energy
            self.logger.info("Extracting energy...")
            result.energy = femm.mo_blockintegral(2)  # Energy integral

            # Calculate flux linkage if circuits exist
            if self.circuits:
                self.logger.info("Calculating flux linkage...")
                result.flux_linkage = self._calculate_flux_linkage()
            else:
                result.flux_linkage = 0.0

            result.success = True
            result.analysis_time = time.time() - start_time

            if grid_resolution:
                self.logger.info(f"Grid analysis completed in {result.analysis_time:.2f}s "
                               f"({grid_resolution}×{grid_resolution} grid)")
            else:
                self.logger.info(f"Analysis completed in {result.analysis_time:.2f}s")

        except Exception as e:
            result.error_message = str(e)
            result.success = False
            self.logger.error(f"Analysis failed: {e}")
            # Save the current state for debugging
            femm.mi_saveas("temp001_ERROR.fem")
            self.logger.error("Problem geometry saved to temp001_ERROR.fem for inspection")

        return result

    def _extract_field_at_points(self,
                                points: Optional[List[Tuple[float, float]]] = None,
                                grid_resolution: Optional[int] = None,
                                grid_bounds: Optional[Tuple[float, float, float, float]] = None
                                ) -> Dict[str, np.ndarray]:
        """
        Extract magnetic field at specified points OR on a uniform grid.

        Supports both sparse point sampling (backward compatible) and dense grid sampling (for CNN training).

        Args:
            points: List of (x,y) coordinates for sparse sampling (backward compat)
            grid_resolution: If specified, sample on uniform grid instead (e.g., 800 for 800×800)
            grid_bounds: Grid domain bounds (xmin, xmax, ymin, ymax) in mm

        Returns:
            Dictionary containing:
                - 'Bx': X-component of flux density (1D array or 2D grid)
                - 'By': Y-component of flux density (1D array or 2D grid)
                - 'B_magnitude': Magnitude of flux density (1D array or 2D grid)
                - 'points': Sample locations (sparse mode only)
                - 'grid_coordinates': (xx, yy) meshgrid (grid mode only)

        Example:
            # Sparse sampling (backward compatible):
            field = solver._extract_field_at_points(points=[(0,0), (10,10)])

            # Grid sampling (new):
            field = solver._extract_field_at_points(grid_resolution=200, grid_bounds=(-50,50,-50,50))
        """
        if grid_resolution is not None:
            # NEW: Grid-based sampling
            self.logger.info(f"Extracting field on {grid_resolution}×{grid_resolution} grid...")

            # Create uniform grid
            if grid_bounds is None:
                grid_bounds = (-100, 100, -100, 100)  # Default ±100mm domain

            xx, yy, grid_points = self._create_field_grid(grid_resolution, grid_bounds)

            # Sample field at all grid points
            Bx = []
            By = []

            for x, y in grid_points:
                bx, by = femm.mo_getb(x, y)
                Bx.append(bx)
                By.append(by)

            # Reshape to 2D grids
            Bx_grid = np.array(Bx).reshape((grid_resolution, grid_resolution))
            By_grid = np.array(By).reshape((grid_resolution, grid_resolution))
            B_mag = np.sqrt(Bx_grid**2 + By_grid**2)

            self.logger.info(f"Field extraction complete. B range: [{B_mag.min():.3f}, {B_mag.max():.3f}] T")

            return {
                'Bx': Bx_grid,
                'By': By_grid,
                'B_magnitude': B_mag,
                'grid_coordinates': (xx, yy)
            }

        else:
            # EXISTING: Sparse point sampling (backward compatible)
            if points is None or len(points) == 0:
                return {
                    'Bx': np.array([]),
                    'By': np.array([]),
                    'B_magnitude': np.array([]),
                    'points': np.array([])
                }

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

    def _create_field_grid(self,
                          resolution: int = 800,
                          bounds: Tuple[float, float, float, float] = (-100, 100, -100, 100)
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create uniform spatial grid for field sampling.

        This method generates a regular grid over the specified domain for dense field sampling,
        which is useful for CNN training that requires 2D spatial field maps.

        Args:
            resolution: Grid size (e.g., 800 for 800×800 grid)
            bounds: (xmin, xmax, ymin, ymax) domain bounds in mm

        Returns:
            xx: 2D X-coordinate meshgrid (resolution × resolution)
            yy: 2D Y-coordinate meshgrid (resolution × resolution)
            points: Flattened (N×2) array of (x,y) coordinates for mo_getb()

        Example:
            xx, yy, points = solver._create_field_grid(resolution=200, bounds=(-50, 50, -50, 50))
            # Returns 200×200 grid over ±50mm domain
        """
        xmin, xmax, ymin, ymax = bounds
        x = np.linspace(xmin, xmax, resolution)
        y = np.linspace(ymin, ymax, resolution)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (resolution²,2)
        return xx, yy, points

    def _generate_semantic_mask(self,
                               xx: np.ndarray,
                               yy: np.ndarray,
                               material_map: Optional[Dict[str, int]] = None
                               ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Generate semantic mask by querying material properties at each grid point.

        Uses FEMM's mo_getmaterial(x,y) to identify material at each spatial location.
        This provides geometry-aware input channels for CNN training.

        Args:
            xx: X-coordinate meshgrid (resolution × resolution)
            yy: Y-coordinate meshgrid (resolution × resolution)
            material_map: Optional custom material-to-ID mapping

        Returns:
            mask: 2D integer array with material IDs (0=Air, 1=Iron, 2=Copper, 3=Magnet, etc.)
            labels: Dictionary mapping IDs to material names

        Performance Note:
            - For 800×800 grid: ~640,000 FEMM queries (~60 seconds)
            - For 200×200 grid: ~40,000 queries (~4 seconds)
            - Consider using coarser grids during development

        Example:
            mask, labels = solver._generate_semantic_mask(xx, yy)
            print(f"Materials found: {labels}")
            # {0: 'Air', 1: 'Iron', 2: 'Copper', 3: 'NdFeB'}
        """
        # Default material ID mapping
        if material_map is None:
            material_map = {
                'Air': 0,
                '<No Mesh>': 0,  # FEMM returns this for unbounded regions
                'Cold rolled low carbon strip steel': 1,
                'Silicon Steel': 1,
                'M-19 Steel': 1,
                'Copper': 2,
                '18 AWG': 2,  # Wire gauge materials
                '20 AWG': 2,
                'NdFeB_42': 3,
                'NdFeB 40 MGOe': 3,
                'NdFeB 52 MGOe': 3
            }

        resolution = xx.shape[0]
        self.logger.info(f"Generating semantic mask for {resolution}×{resolution} grid (optimized)...")

        # --- OPTIMIZED MASK GENERATION ---
        # 1. Get all mesh nodes and their corresponding material labels from FEMM
        num_nodes = femm.mo_numnodes()
        node_coords = np.array([femm.mo_getnode(i+1) for i in range(num_nodes)])
        
        # Get material for each element and map it to its nodes
        num_elements = femm.mo_numelements()
        element_materials = np.zeros(num_nodes, dtype=object)
        for i in range(num_elements):
            element = femm.mo_getelement(i + 1)
            mat_name = element[9] # The 10th item is the block name
            mat_id = material_map.get(mat_name, 0)
            # Assign this material ID to all nodes belonging to this element
            for node_idx in element[0:3]: # An element is a triangle of 3 nodes
                element_materials[node_idx-1] = mat_id

        # 2. Interpolate material IDs onto the target grid
        # This is much faster than querying each point individually.
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
        mask_flat = griddata(node_coords, element_materials, grid_points, method='nearest', fill_value=0)
        mask = mask_flat.reshape((resolution, resolution)).astype(np.uint8)
        # --- END OF OPTIMIZATION ---

        # Create reverse mapping for labels
        labels = {}
        for mat_name, mat_id in material_map.items():
            if mat_id not in labels and mat_id in np.unique(mask):
                labels[mat_id] = mat_name

        self.logger.info(f"Semantic mask generated. Materials found: {labels}")
        return mask, labels

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
                    coil_width: float, coil_height: float, turns: int, current: float,
                    wire_radius: float = 0.001,
                    analysis_points: List[Tuple[float, float]] = None,
                    grid_resolution: Optional[int] = None,
                    grid_bounds: Optional[Tuple[float, float, float, float]] = None,
                    generate_mask: bool = True) -> FEMMResult:
        """
        Analyze coaxial coil configuration

        Supports both sparse point sampling (backward compatible) and dense grid sampling (for CNN training).

        Args:
            radius: Coil radius (m)
            coil_width: Width of the coil's winding pack (m)
            coil_height: Height of the coil's winding pack (m)
            turns: Number of turns
            current: Coil current (A)
            wire_radius: Wire radius (m)
            analysis_points: Points to evaluate field (sparse sampling, backward compatible)
            grid_resolution: If specified, sample field on uniform grid (e.g., 800 for 800×800)
            grid_bounds: Grid domain bounds (xmin, xmax, ymin, ymax) in mm
            generate_mask: Whether to generate semantic mask (only used if grid_resolution is set)

        Returns:
            FEMM analysis results

        Example:
            # Backward compatible sparse sampling:
            result = solver.analyze_coil(radius=0.05, turns=100, current=10.0)
            result = solver.analyze_coil(coil_width=0.02, coil_height=0.05, turns=100, current=10.0)

            # Grid sampling with semantic mask:
            result = solver.analyze_coil(
                radius=0.05, turns=100, current=10.0,
                coil_width=0.02, coil_height=0.05, turns=100, current=10.0,
                grid_resolution=200,
                grid_bounds=(-0.1, 0.1, -0.1, 0.1)
            )
        """
        # Create new document
        self.create_magnetics_document(frequency=0.0)
        self.setup_materials()

        # Add circuit
        self.add_circuit('coil', current)

        # Create coil geometry
        self.create_coil_geometry(0, 0, radius, wire_radius, turns, 'coil')
        # Create coil geometry (as a rectangular block centered at the origin)
        coil_x = -coil_width / 2
        coil_y = -coil_height / 2
        self.create_coil_geometry(coil_x, coil_y, coil_width, coil_height, turns, 'coil')

        # Create boundary box (2x coil size)
        boundary_size = radius * 4
        # Create boundary box
        boundary_size = max(coil_width, coil_height) * 4
        self.create_boundary_box(-boundary_size, -boundary_size,
                                boundary_size, boundary_size)

        # CRITICAL FIX: Add a block label for the Air region
        # A point just outside the coil but inside the boundary
        femm.mi_addblocklabel(radius + 1, 0)
        # CRITICAL FIX: Add a block label for the Air region surrounding the coil
        # A point just outside the coil, but inside the boundary box
        femm.mi_addblocklabel(0, coil_y + coil_height + 0.001)
        femm.mi_setblockprop('Air', 1, 0, '<None>', 0, 0, 0)

        # Default analysis points if not provided (backward compatible)
        if grid_resolution is None and analysis_points is None:
            analysis_points = []
            # Points along axis
            for z in np.linspace(-radius*2, radius*2, 20):
            for z in np.linspace(-coil_height, coil_height, 20):
                analysis_points.append((0, z))
            # Points in radial direction
            for r in np.linspace(0, radius*2, 20):
            for r in np.linspace(0, coil_width, 20):
                analysis_points.append((r, 0))

        # Run analysis
        return self.analyze_and_extract_results(
            field_points=analysis_points,
            grid_resolution=grid_resolution,
            grid_bounds=grid_bounds,
            generate_mask=generate_mask
        )


class TransformerSolver(FEMMSolver):
    """Specialized solver for transformer problems"""

    def analyze_transformer(self,
                          primary_turns: int, secondary_turns: int,
                          core_area: float, frequency: float,
                          primary_voltage: float,
                          window_width: float = None,
                          window_height: float = None,
                          analysis_points: List[Tuple[float, float]] = None,
                          grid_resolution: Optional[int] = None,
                          grid_bounds: Optional[Tuple[float, float, float, float]] = None,
                          generate_mask: bool = True) -> FEMMResult:
        """
        Analyze transformer configuration

        Supports both sparse point sampling (backward compatible) and dense grid sampling (for CNN training).

        Args:
            primary_turns: Number of primary turns
            secondary_turns: Number of secondary turns
            core_area: Core cross-sectional area (m²)
            frequency: Operating frequency (Hz)
            primary_voltage: Primary voltage (V)
            window_width: Transformer window width
            window_height: Transformer window height
            analysis_points: Points to evaluate field (sparse sampling, backward compatible)
            grid_resolution: If specified, sample field on uniform grid (e.g., 800 for 800×800)
            grid_bounds: Grid domain bounds (xmin, xmax, ymin, ymax) in mm
            generate_mask: Whether to generate semantic mask (only used if grid_resolution is set)

        Returns:
            FEMM analysis results

        Example:
            # Backward compatible sparse sampling:
            result = solver.analyze_transformer(primary_turns=100, secondary_turns=50,
                                               core_area=0.001, frequency=60, primary_voltage=120)

            # Grid sampling with semantic mask:
            result = solver.analyze_transformer(
                primary_turns=100, secondary_turns=50,
                core_area=0.001, frequency=60, primary_voltage=120,
                grid_resolution=200,
                grid_bounds=(-50, 50, -50, 50)
            )
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

        # CRITICAL FIX: Add a block label for the Air region
        # A point just outside the core geometry
        femm.mi_addblocklabel(boundary_size - 1, boundary_size - 1)
        femm.mi_setblockprop('Air', 1, 0, '<None>', 0, 0, 0)

        # Default analysis points if not provided (backward compatible)
        if grid_resolution is None and analysis_points is None:
            analysis_points = []
            # Points in core
            for x in np.linspace(-core_side, core_side, 10):
                analysis_points.append((x, 0))
            # Points in window
            for x in np.linspace(-window_width/2, window_width/2, 10):
                analysis_points.append((x, 0))

        # Run analysis
        return self.analyze_and_extract_results(
            field_points=analysis_points,
            grid_resolution=grid_resolution,
            grid_bounds=grid_bounds,
            generate_mask=generate_mask
        )

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
        femm.mi_drawrectangle(-core_side/2 - window_width/2 - primary_width/2, -primary_height/2, -core_side/2 - window_width/2 + primary_width/2, primary_height/2)
        femm.mi_addblocklabel(-core_side/2 - window_width/2, 0)
        femm.mi_setblockprop('Copper', 0, 1.0, 'primary', 0, primary_turns, 0)

        # Secondary winding (right window)
        secondary_width = window_width * 0.8
        secondary_height = window_height * 0.4
        femm.mi_drawrectangle(core_side/2 + window_width/2 - secondary_width/2, -secondary_height/2, core_side/2 + window_width/2 + secondary_width/2, secondary_height/2)
        femm.mi_addblocklabel(core_side/2 + window_width/2, 0)
        femm.mi_setblockprop('Copper', 0, 1.0, 'secondary', 0, secondary_turns, 0)


class IPMMotorSolver(FEMMSolver):
    """Specialized solver for IPM motor problems"""

    def analyze_ipm_motor(self,
                         stator_slots: int, rotor_poles: int,
                         stator_outer_radius: float, stator_inner_radius: float,
                         rotor_inner_radius: float, air_gap: float,
                         magnet_strength: float, rotor_position: float,
                         current_amplitude: float,
                         analysis_points: List[Tuple[float, float]] = None,
                         grid_resolution: Optional[int] = None,
                         grid_bounds: Optional[Tuple[float, float, float, float]] = None,
                         generate_mask: bool = True) -> FEMMResult:
        """
        Analyze IPM motor configuration

        Supports both sparse point sampling (backward compatible) and dense grid sampling (for CNN training).

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
            analysis_points: Points to evaluate field (sparse sampling, backward compatible)
            grid_resolution: If specified, sample field on uniform grid (e.g., 800 for 800×800)
            grid_bounds: Grid domain bounds (xmin, xmax, ymin, ymax) in mm
            generate_mask: Whether to generate semantic mask (only used if grid_resolution is set)

        Returns:
            FEMM analysis results

        Example:
            # Backward compatible sparse sampling:
            result = solver.analyze_ipm_motor(
                stator_slots=24, rotor_poles=4,
                stator_outer_radius=0.1, stator_inner_radius=0.06,
                rotor_inner_radius=0.03, air_gap=0.001,
                magnet_strength=1.2, rotor_position=0, current_amplitude=10
            )

            # Grid sampling with semantic mask:
            result = solver.analyze_ipm_motor(
                stator_slots=24, rotor_poles=4,
                stator_outer_radius=0.1, stator_inner_radius=0.06,
                rotor_inner_radius=0.03, air_gap=0.001,
                magnet_strength=1.2, rotor_position=0, current_amplitude=10,
                grid_resolution=200,
                grid_bounds=(-120, 120, -120, 120)
            )
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

        # CRITICAL FIX: Add a block label for the Air region
        femm.mi_addblocklabel(stator_outer_radius + 1, 0) # A point just outside the stator
        femm.mi_setblockprop('Air', 1, 0, '<None>', 0, 0, 0)

        # Default analysis points if not provided (backward compatible)
        if grid_resolution is None and analysis_points is None:
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
        return self.analyze_and_extract_results(
            field_points=analysis_points,
            integration_regions=integration_regions,
            grid_resolution=grid_resolution,
            grid_bounds=grid_bounds,
            generate_mask=generate_mask
        )

    def _create_stator(self, outer_radius: float, inner_radius: float, slots: int) -> None:
        """Create stator geometry with slots"""
        self.logger.info("Creating stator geometry...")
        # Draw stator yoke using efficient arc commands
        femm.mi_drawarc(0, outer_radius, 0, -outer_radius, 180, 1)
        femm.mi_drawarc(0, -outer_radius, 0, outer_radius, 180, 1)

        # Create slots and teeth
        slot_depth = (outer_radius - inner_radius) * 0.5
        tooth_width_rad = (2 * np.pi / slots) * 0.5

        for i in range(slots):
            angle = i * 2 * np.pi / slots
            # Draw tooth face arc
            femm.mi_drawarc(inner_radius * np.cos(angle - tooth_width_rad/2), inner_radius * np.sin(angle - tooth_width_rad/2),
                          inner_radius * np.cos(angle + tooth_width_rad/2), inner_radius * np.sin(angle + tooth_width_rad/2),
                          np.degrees(tooth_width_rad), 1)

            # Add winding block label in the slot region
            phase_map = ['A', 'B', 'C']
            phase = phase_map[(i % 3)]
            slot_center_r = inner_radius + slot_depth / 2
            slot_center_angle = angle + np.pi / slots
            femm.mi_addblocklabel(slot_center_r * np.cos(slot_center_angle), slot_center_r * np.sin(slot_center_angle))
            femm.mi_setblockprop('Copper', 1, 1.0, f'phase_{phase}', 0, 10, 0) # 10 turns per slot

        # Add stator yoke material label
        stator_yoke_r = (outer_radius + inner_radius + slot_depth) / 2
        femm.mi_addblocklabel(stator_yoke_r, 0)
        femm.mi_setblockprop('Silicon Steel', 1, 1.0, '<None>', 0, 1, 0)

    def _create_rotor(self, outer_radius: float, inner_radius: float, poles: int,
                     magnet_strength: float, rotor_position: float) -> None:
        """Create rotor with permanent magnets"""
        self.logger.info("Creating rotor geometry...")
        # Draw rotor body using efficient arc commands
        femm.mi_drawarc(0, outer_radius, 0, -outer_radius, 180, 1)
        femm.mi_drawarc(0, -outer_radius, 0, outer_radius, 180, 1)
        if inner_radius > 0:
            femm.mi_drawarc(0, inner_radius, 0, -inner_radius, 180, 1)
            femm.mi_drawarc(0, -inner_radius, 0, inner_radius, 180, 1)

        # Create embedded permanent magnets
        magnet_thickness = (outer_radius - inner_radius) * 0.2
        magnet_width = (outer_radius - inner_radius) * 0.6
        magnet_depth = outer_radius * 0.85

        for i in range(poles):
            angle_deg = i * 360 / poles + np.degrees(rotor_position)

            # Create a magnet rectangle at a base position and then rotate it
            femm.mi_clearselected()
            x = magnet_depth
            y = -magnet_width / 2
            femm.mi_drawrectangle(x, y, x + magnet_thickness, y + magnet_width)
            femm.mi_selectgroup(1) # Select all entities in group 1 (the new rectangle)
            femm.mi_moverotate(0, 0, angle_deg)

            # Add magnet material with alternating polarity
            polarity = 1 if i % 2 == 0 else -1
            mag_dir = angle_deg + 90 # Magnetization direction perpendicular to magnet length

            # Place label in the center of the original un-rotated rectangle and then rotate it
            label_x = x + magnet_thickness / 2
            label_y = 0
            rotated_label_x = label_x * np.cos(np.radians(angle_deg))
            rotated_label_y = label_x * np.sin(np.radians(angle_deg))

            femm.mi_addblocklabel(rotated_label_x, rotated_label_y)
            femm.mi_setblockprop('NdFeB_42', 1, 1.0, '<None>', mag_dir * polarity, 3, 0)

        # Add rotor core material
        # Place a label in a region that is clearly rotor iron (between magnets)
        rotor_core_r = (outer_radius + inner_radius) / 2
        label_angle = np.radians(180/poles) + rotor_position
        femm.mi_addblocklabel(rotor_core_r * np.cos(label_angle),
                              rotor_core_r * np.sin(label_angle))
        femm.mi_setblockprop('Silicon Steel', 1, 1.0, '<None>', 0, 5, 0)


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
                coil_width=0.02,  # 20mm
                coil_height=0.05, # 50mm
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