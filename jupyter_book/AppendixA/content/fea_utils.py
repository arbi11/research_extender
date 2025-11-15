"""
FEA Utilities for Magnetostatics Teaching
==========================================

Helper classes and functions for 2D magnetostatic finite element analysis.
Built from scratch using NumPy/SciPy for educational purposes.

Author: Teaching Material
Date: 2025
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Callable, Optional


class TriangularMesh:
    """
    2D triangular mesh for finite element analysis.

    Attributes:
        nodes: (N, 2) array of node coordinates [x, y]
        elements: (E, 3) array of element connectivity [node0, node1, node2]
        n_nodes: Number of nodes
        n_elements: Number of elements
    """

    def __init__(self, nodes: np.ndarray, elements: np.ndarray):
        """
        Initialize mesh.

        Parameters:
            nodes: (N, 2) array of node coordinates
            elements: (E, 3) array of element connectivity (0-indexed)
        """
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        self.n_nodes = len(self.nodes)
        self.n_elements = len(self.elements)

        # Pre-compute element geometry for efficiency
        self._compute_all_geometry()

    def _compute_all_geometry(self):
        """Pre-compute geometry coefficients for all elements."""
        self.areas = np.zeros(self.n_elements)
        self.b_coeffs = np.zeros((self.n_elements, 3))
        self.c_coeffs = np.zeros((self.n_elements, 3))

        for e in range(self.n_elements):
            area, b, c = self.compute_element_geometry(e)
            self.areas[e] = area
            self.b_coeffs[e] = b
            self.c_coeffs[e] = c

    def compute_element_geometry(self, elem_id: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute geometry coefficients for linear triangular element.

        For element with nodes (x0,y0), (x1,y1), (x2,y2):
            b0 = y1 - y2,  c0 = x2 - x1
            b1 = y2 - y0,  c1 = x0 - x2
            b2 = y0 - y1,  c2 = x1 - x0

        Parameters:
            elem_id: Element index

        Returns:
            area: Element area
            b: (3,) array of b coefficients
            c: (3,) array of c coefficients
        """
        nodes_idx = self.elements[elem_id]
        coords = self.nodes[nodes_idx]  # (3, 2)

        x = coords[:, 0]
        y = coords[:, 1]

        # Area using cross product (shoelace formula)
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) -
                         (x[2] - x[0]) * (y[1] - y[0]))

        # Geometry coefficients for shape function gradients
        b = np.array([y[1] - y[2],
                      y[2] - y[0],
                      y[0] - y[1]])

        c = np.array([x[2] - x[1],
                      x[0] - x[2],
                      x[1] - x[0]])

        return area, b, c

    def get_element_centers(self) -> np.ndarray:
        """Get centroid coordinates of all elements."""
        centers = np.zeros((self.n_elements, 2))
        for e in range(self.n_elements):
            nodes_idx = self.elements[e]
            centers[e] = np.mean(self.nodes[nodes_idx], axis=0)
        return centers

    def plot(self, values: Optional[np.ndarray] = None,
             ax: Optional[plt.Axes] = None,
             show_mesh: bool = True,
             show_nodes: bool = False,
             node_labels: bool = False,
             element_labels: bool = False,
             cmap: str = 'viridis',
             levels: int = 20,
             title: str = '',
             colorbar_label: str = '') -> plt.Axes:
        """
        Plot the mesh with optional field values.

        Parameters:
            values: (N,) array of values at nodes (for contour plot)
            ax: Matplotlib axes to plot on
            show_mesh: Show mesh edges
            show_nodes: Show node markers
            node_labels: Label nodes with indices
            element_labels: Label elements with indices
            cmap: Colormap for field values
            levels: Number of contour levels
            title: Plot title
            colorbar_label: Colorbar label

        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        triang = tri.Triangulation(self.nodes[:, 0], self.nodes[:, 1], self.elements)

        # Plot field values as contour
        if values is not None:
            cs = ax.tricontourf(triang, values, levels=levels, cmap=cmap)
            plt.colorbar(cs, ax=ax, label=colorbar_label)

        # Plot mesh edges
        if show_mesh:
            ax.triplot(triang, 'k-', linewidth=0.5, alpha=0.4)

        # Plot nodes
        if show_nodes:
            ax.plot(self.nodes[:, 0], self.nodes[:, 1], 'ko', markersize=4)

        # Label nodes
        if node_labels:
            for i, (x, y) in enumerate(self.nodes):
                ax.text(x, y, f'{i}', fontsize=10, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # Label elements
        if element_labels:
            centers = self.get_element_centers()
            for i, (x, y) in enumerate(centers):
                ax.text(x, y, f'T{i}', fontsize=9, ha='center', va='center',
                       color='red', weight='bold')

        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        return ax


class MagnetostaticSolver:
    """
    2D magnetostatic FEA solver using scalar magnetic potential A_z.

    Solves: ∇×(1/μ ∇×A) = J_z

    Weak form with linear triangular elements gives:
        K A = F
    where
        K_ij^(e) = (1/μ_e) * (1/4A_e) * (b_i*b_j + c_i*c_j)
        F_i^(e) = J_z * A_e / 3  (for uniform J_z)
    """

    def __init__(self, mesh: TriangularMesh,
                 mu_per_element: np.ndarray,
                 J_per_element: np.ndarray,
                 dirichlet_nodes: dict = None):
        """
        Initialize solver.

        Parameters:
            mesh: TriangularMesh object
            mu_per_element: (E,) array of permeability for each element
            J_per_element: (E,) array of current density for each element
            dirichlet_nodes: Dict {node_id: value} for Dirichlet BCs
        """
        self.mesh = mesh
        self.mu = np.asarray(mu_per_element)
        self.J = np.asarray(J_per_element)
        self.dirichlet_nodes = dirichlet_nodes if dirichlet_nodes else {}

        # Solution vector
        self.A = np.zeros(mesh.n_nodes)

        # System matrices (will be assembled)
        self.K = None
        self.F = None
        self.K_free = None  # Reduced system after BCs
        self.F_free = None
        self.free_dofs = None
        self.fixed_dofs = None

    def assemble_system(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        """
        Assemble global stiffness matrix K and load vector F.

        Returns:
            K: (N, N) sparse stiffness matrix
            F: (N,) load vector
        """
        N = self.mesh.n_nodes

        # Use LIL format for efficient assembly
        K = sp.lil_matrix((N, N))
        F = np.zeros(N)

        # Loop over all elements
        for e in range(self.mesh.n_elements):
            # Element data
            nodes = self.mesh.elements[e]
            area = self.mesh.areas[e]
            b = self.mesh.b_coeffs[e]
            c = self.mesh.c_coeffs[e]
            mu_e = self.mu[e]
            J_e = self.J[e]

            # Element stiffness: K^(e)_ij = (1/μ) * (1/4A) * (b_i*b_j + c_i*c_j)
            coeff = (1.0 / mu_e) / (4.0 * area)
            for i in range(3):
                for j in range(3):
                    K_ij = coeff * (b[i] * b[j] + c[i] * c[j])
                    K[nodes[i], nodes[j]] += K_ij

            # Element load: F^(e)_i = J_z * A_e / 3 (uniform source)
            f_e = J_e * area / 3.0
            for i in range(3):
                F[nodes[i]] += f_e

        # Convert to CSR format for efficient solving
        self.K = K.tocsr()
        self.F = F

        return self.K, self.F

    def apply_boundary_conditions(self):
        """
        Apply Dirichlet boundary conditions by eliminating rows/columns.

        Creates reduced system K_free @ A_free = F_free for free DOFs only.
        """
        N = self.mesh.n_nodes

        # Identify free and fixed DOFs
        self.fixed_dofs = np.array(sorted(self.dirichlet_nodes.keys()), dtype=int)
        self.free_dofs = np.array([i for i in range(N) if i not in self.dirichlet_nodes], dtype=int)

        # Set fixed DOF values
        for node, value in self.dirichlet_nodes.items():
            self.A[node] = value

        # Extract sub-matrices for free DOFs
        self.K_free = self.K[self.free_dofs, :][:, self.free_dofs]

        # Adjust RHS for fixed DOFs
        if len(self.fixed_dofs) > 0:
            K_fixed = self.K[self.free_dofs, :][:, self.fixed_dofs]
            A_fixed = self.A[self.fixed_dofs]
            self.F_free = self.F[self.free_dofs] - K_fixed @ A_fixed
        else:
            self.F_free = self.F[self.free_dofs]

    def solve_direct(self):
        """Solve using direct LU factorization (scipy.sparse.linalg.spsolve)."""
        if self.K_free is None:
            raise RuntimeError("Must call apply_boundary_conditions() first")

        A_free = spla.spsolve(self.K_free, self.F_free)
        self.A[self.free_dofs] = A_free

    def solve_jacobi(self, max_iter: int = 1000, tol: float = 1e-6,
                     x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[float]]:
        """
        Solve using Jacobi iteration.

        Updates: x_i^(k+1) = (b_i - Σ_{j≠i} A_ij x_j^(k)) / A_ii

        Returns:
            solution: Final solution vector
            residuals: List of residual norms at each iteration
        """
        if self.K_free is None:
            raise RuntimeError("Must call apply_boundary_conditions() first")

        K = self.K_free.toarray()  # Convert to dense for simplicity
        b = self.F_free
        n = len(b)

        # Initial guess
        x = np.zeros(n) if x0 is None else x0.copy()
        x_new = x.copy()

        D_inv = 1.0 / np.diag(K)  # Inverse of diagonal
        residuals = []

        for iteration in range(max_iter):
            # Jacobi update: x_new = D^(-1) * (b - (K - D) @ x)
            for i in range(n):
                x_new[i] = (b[i] - np.sum(K[i, :] * x) + K[i, i] * x[i]) * D_inv[i]

            # Check convergence
            residual = np.linalg.norm(K @ x_new - b)
            residuals.append(residual)

            if residual < tol:
                break

            x = x_new.copy()

        self.A[self.free_dofs] = x_new
        return x_new, residuals

    def solve_gauss_seidel(self, max_iter: int = 1000, tol: float = 1e-6,
                          x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[float]]:
        """
        Solve using Gauss-Seidel iteration.

        Updates: x_i^(k+1) = (b_i - Σ_{j<i} A_ij x_j^(k+1) - Σ_{j>i} A_ij x_j^(k)) / A_ii

        Returns:
            solution: Final solution vector
            residuals: List of residual norms at each iteration
        """
        if self.K_free is None:
            raise RuntimeError("Must call apply_boundary_conditions() first")

        K = self.K_free.toarray()
        b = self.F_free
        n = len(b)

        x = np.zeros(n) if x0 is None else x0.copy()
        residuals = []

        for iteration in range(max_iter):
            # Gauss-Seidel update: use new values immediately
            for i in range(n):
                x[i] = (b[i] - np.dot(K[i, :], x) + K[i, i] * x[i]) / K[i, i]

            # Check convergence
            residual = np.linalg.norm(K @ x - b)
            residuals.append(residual)

            if residual < tol:
                break

        self.A[self.free_dofs] = x
        return x, residuals

    def solve_conjugate_gradient(self, max_iter: Optional[int] = None,
                                tol: float = 1e-6) -> Tuple[np.ndarray, List[float]]:
        """
        Solve using Conjugate Gradient method.

        Returns:
            solution: Final solution vector
            residuals: List of residual norms at each iteration
        """
        if self.K_free is None:
            raise RuntimeError("Must call apply_boundary_conditions() first")

        residuals = []

        def callback(xk):
            r = self.K_free @ xk - self.F_free
            residuals.append(np.linalg.norm(r))

        if max_iter is None:
            max_iter = len(self.F_free)

        A_free, info = spla.cg(self.K_free, self.F_free,
                               maxiter=max_iter, atol=tol, callback=callback)

        if info != 0:
            print(f"Warning: CG did not converge (info={info})")

        self.A[self.free_dofs] = A_free
        return A_free, residuals

    def solve_cg_scratch(self, max_iter: Optional[int] = None,
                        tol: float = 1e-6,
                        x0: Optional[np.ndarray] = None,
                        preconditioner: str = 'none') -> Tuple[np.ndarray, List[float]]:
        """
        Solve using Conjugate Gradient method (from-scratch implementation).

        The Conjugate Gradient method is an iterative solver for symmetric positive
        definite (SPD) systems K @ x = b. It minimizes the energy functional
        E(x) = (1/2) x^T K x - x^T b by moving along A-conjugate (A-orthogonal)
        search directions.

        Algorithm:
            1. Initialize: r0 = b - K @ x0, p0 = r0 (or preconditioned version)
            2. For each iteration k:
               - Compute step size: α_k = (r_k^T z_k) / (p_k^T K p_k)
               - Update solution: x_{k+1} = x_k + α_k p_k
               - Update residual: r_{k+1} = r_k - α_k K p_k
               - Apply preconditioner: z_{k+1} = M^{-1} r_{k+1}
               - Compute direction update: β_k = (r_{k+1}^T z_{k+1}) / (r_k^T z_k)
               - Update search direction: p_{k+1} = z_{k+1} + β_k p_k
            3. Stop when ||r|| < tol

        Preconditioning:
            Preconditioning transforms the system to M^{-1} K @ x = M^{-1} b where
            M ≈ K but is easy to invert. This improves the condition number and
            accelerates convergence.

            - 'none': No preconditioning (M = I)
            - 'jacobi': Diagonal preconditioning (M = diag(K))

        Parameters:
            max_iter: Maximum iterations (default: system size)
            tol: Convergence tolerance on ||r||_2
            x0: Initial guess (default: zeros)
            preconditioner: 'none' or 'jacobi' (default: 'none')

        Returns:
            solution: Final solution vector
            residuals: List of residual norms at each iteration
        """
        if self.K_free is None:
            raise RuntimeError("Must call apply_boundary_conditions() first")

        # Convert to dense for pedagogical clarity (consistent with Jacobi/GS)
        K = self.K_free.toarray()
        b = self.F_free
        n = len(b)

        # Default parameters
        if max_iter is None:
            max_iter = n

        # Initial guess (typically zeros)
        x = np.zeros(n) if x0 is None else x0.copy()

        # Setup preconditioner
        if preconditioner == 'jacobi':
            # Jacobi preconditioner: M^{-1} = diag(K)^{-1}
            # This is cheap to compute and apply, often gives 2x speedup
            M_inv = 1.0 / np.diag(K)
            use_precond = True
        else:
            M_inv = None
            use_precond = False

        # Initial residual: r = b - K @ x
        r = b - K @ x

        # Apply preconditioner: z = M^{-1} @ r
        # For no preconditioning: z = r (equivalent to M = I)
        if use_precond:
            z = M_inv * r  # Element-wise multiplication for diagonal M_inv
        else:
            z = r.copy()

        # Initial search direction: p = z
        p = z.copy()

        # Initial inner product: rz = r^T @ z
        # This quantity appears in both alpha and beta calculations
        rz = np.dot(r, z)

        residuals = []

        for iteration in range(max_iter):
            # Compute and store residual norm
            r_norm = np.linalg.norm(r)
            residuals.append(r_norm)

            # Check convergence
            if r_norm < tol:
                if iteration > 0:  # Don't print for immediate convergence
                    print(f"CG converged in {iteration} iterations (||r|| = {r_norm:.2e})")
                break

            # Matrix-vector product: Kp = K @ p
            # This is the most expensive operation (O(n^2) for dense)
            Kp = K @ p

            # Compute step size: α = (r^T @ z) / (p^T @ K @ p)
            # α determines how far to move along direction p
            pKp = np.dot(p, Kp)
            alpha = rz / pKp

            # Update solution: x = x + α * p
            x = x + alpha * p

            # Update residual: r = r - α * K @ p
            # Note: We reuse Kp from above, avoiding another matrix-vector product
            r = r - alpha * Kp

            # Apply preconditioner to new residual: z = M^{-1} @ r
            if use_precond:
                z = M_inv * r
            else:
                z = r.copy()

            # Compute new inner product for next iteration
            rz_new = np.dot(r, z)

            # Compute direction update factor: β = (r_{k+1}^T @ z_{k+1}) / (r_k^T @ z_k)
            # β determines how much of the old direction to keep
            beta = rz_new / rz

            # Update search direction: p = z + β * p
            # This creates a new direction that is K-conjugate to all previous directions
            p = z + beta * p

            # Store inner product for next iteration
            rz = rz_new

        # Store final solution in the full solution vector
        self.A[self.free_dofs] = x

        return x, residuals

    def compute_flux_density(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute magnetic flux density B = ∇×A at element centers.

        For 2D with A = A_z ẑ:
            B_x = ∂A_z/∂y
            B_y = -∂A_z/∂x

        Returns:
            B_x: (E,) array of x-component of B
            B_y: (E,) array of y-component of B
            B_mag: (E,) array of magnitude |B|
        """
        B_x = np.zeros(self.mesh.n_elements)
        B_y = np.zeros(self.mesh.n_elements)

        for e in range(self.mesh.n_elements):
            nodes = self.mesh.elements[e]
            A_e = self.A[nodes]
            b = self.mesh.b_coeffs[e]
            c = self.mesh.c_coeffs[e]
            area = self.mesh.areas[e]

            # Gradients: ∇N_i = (b_i, c_i) / (2*area)
            # B_x = ∂A/∂y = Σ A_i * b_i / (2*area)
            # B_y = -∂A/∂x = -Σ A_i * c_i / (2*area)
            B_x[e] = np.dot(A_e, b) / (2 * area)
            B_y[e] = -np.dot(A_e, c) / (2 * area)

        B_mag = np.sqrt(B_x**2 + B_y**2)

        return B_x, B_y, B_mag

    def compute_energy(self) -> float:
        """
        Compute magnetic energy: W = (1/2) ∫ (B²/μ) dV = (1/2) A^T K A

        Returns:
            Magnetic energy
        """
        return 0.5 * self.A @ (self.K @ self.A)


class NonlinearMagnetostaticSolver:
    """
    Nonlinear magnetostatic solver for materials with μ = μ(B).

    Uses Newton-Raphson iteration:
        1. Compute B from current A
        2. Update μ(B) for each element
        3. Rebuild K(A)
        4. Solve K ΔA = F - K A
        5. Update A ← A + ΔA
        6. Repeat until convergence
    """

    def __init__(self, mesh: TriangularMesh,
                 mu_function: Callable[[np.ndarray], np.ndarray],
                 J_per_element: np.ndarray,
                 dirichlet_nodes: dict = None,
                 mu_air: float = 1.0):
        """
        Initialize nonlinear solver.

        Parameters:
            mesh: TriangularMesh object
            mu_function: Function mu(B_mag) that returns permeability
            J_per_element: (E,) array of current density
            dirichlet_nodes: Dict {node_id: value} for BCs
            mu_air: Permeability of air (default 1.0)
        """
        self.mesh = mesh
        self.mu_function = mu_function
        self.J = J_per_element
        self.dirichlet_nodes = dirichlet_nodes
        self.mu_air = mu_air

        # Initialize with linear solution (all μ = μ_air)
        mu_init = np.full(mesh.n_elements, mu_air)
        self.linear_solver = MagnetostaticSolver(mesh, mu_init, J_per_element, dirichlet_nodes)
        self.linear_solver.assemble_system()
        self.linear_solver.apply_boundary_conditions()
        self.linear_solver.solve_direct()

        self.A = self.linear_solver.A.copy()
        self.mu_current = mu_init.copy()

        # History for visualization
        self.iteration_history = []

    def newton_raphson(self, max_iter: int = 20, tol: float = 1e-6,
                      relaxation: float = 1.0) -> Tuple[np.ndarray, List[float]]:
        """
        Solve nonlinear system using Newton-Raphson iteration.

        Parameters:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance on ||ΔA||
            relaxation: Relaxation factor (0 < ω ≤ 1)

        Returns:
            A: Solution vector
            residuals: List of ||ΔA|| at each iteration
        """
        residuals = []

        for iteration in range(max_iter):
            # 1. Compute B from current A
            solver = MagnetostaticSolver(self.mesh, self.mu_current,
                                        self.J, self.dirichlet_nodes)
            solver.A = self.A.copy()
            B_x, B_y, B_mag = solver.compute_flux_density()

            # 2. Update μ(B) for each element
            self.mu_current = self.mu_function(B_mag)

            # 3. Rebuild K with new μ
            solver.mu = self.mu_current
            solver.assemble_system()
            solver.apply_boundary_conditions()

            # 4. Solve for correction: K ΔA = F - K A
            residual_vec = solver.F_free - solver.K_free @ solver.A[solver.free_dofs]
            dA_free = spla.spsolve(solver.K_free, residual_vec)

            # 5. Update solution with relaxation
            dA = np.zeros(self.mesh.n_nodes)
            dA[solver.free_dofs] = relaxation * dA_free
            self.A += dA

            # Store history
            self.iteration_history.append({
                'A': self.A.copy(),
                'mu': self.mu_current.copy(),
                'B_mag': B_mag.copy(),
                'dA_norm': np.linalg.norm(dA)
            })

            # Check convergence
            dA_norm = np.linalg.norm(dA)
            residuals.append(dA_norm)

            print(f"Iteration {iteration+1}: ||ΔA|| = {dA_norm:.3e}, "
                  f"μ_max = {self.mu_current.max():.2f}, μ_min = {self.mu_current.min():.2f}")

            if dA_norm < tol:
                print(f"Converged in {iteration+1} iterations!")
                break

        return self.A, residuals


def create_rectangle_mesh(width: float, height: float,
                         nx: int, ny: int) -> TriangularMesh:
    """
    Create uniform triangular mesh in a rectangle.

    Parameters:
        width: Rectangle width
        height: Rectangle height
        nx: Number of divisions in x
        ny: Number of divisions in y

    Returns:
        TriangularMesh object
    """
    # Create grid of nodes
    x = np.linspace(0, width, nx)
    y = np.linspace(0, height, ny)
    X, Y = np.meshgrid(x, y)
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    # Create triangulation
    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            # Node indices
            n0 = j * nx + i
            n1 = j * nx + i + 1
            n2 = (j + 1) * nx + i
            n3 = (j + 1) * nx + i + 1

            # Two triangles per quad
            elements.append([n0, n1, n2])
            elements.append([n1, n3, n2])

    elements = np.array(elements)

    return TriangularMesh(nodes, elements)


def create_coil_in_air_mesh(domain_size: float = 2.0,
                            coil_center: Tuple[float, float] = (1.0, 1.0),
                            coil_radius: float = 0.3,
                            n_points: int = 100) -> Tuple[TriangularMesh, np.ndarray]:
    """
    Create mesh for coil in air problem using Delaunay triangulation.

    Parameters:
        domain_size: Size of square domain
        coil_center: (x, y) center of coil region
        coil_radius: Radius of coil region
        n_points: Number of random points

    Returns:
        mesh: TriangularMesh object
        is_coil: (E,) boolean array indicating coil elements
    """
    # Generate random points in domain
    np.random.seed(42)
    points = np.random.rand(n_points, 2) * domain_size

    # Add boundary points
    n_boundary = 20
    boundary_x = np.concatenate([
        np.linspace(0, domain_size, n_boundary),
        np.full(n_boundary, domain_size),
        np.linspace(domain_size, 0, n_boundary),
        np.full(n_boundary, 0)
    ])
    boundary_y = np.concatenate([
        np.full(n_boundary, 0),
        np.linspace(0, domain_size, n_boundary),
        np.full(n_boundary, domain_size),
        np.linspace(domain_size, 0, n_boundary)
    ])
    boundary_points = np.column_stack([boundary_x, boundary_y])

    all_points = np.vstack([points, boundary_points])

    # Delaunay triangulation
    delaunay = Delaunay(all_points)

    mesh = TriangularMesh(all_points, delaunay.simplices)

    # Identify coil elements (those with centers in coil region)
    centers = mesh.get_element_centers()
    distances = np.sqrt((centers[:, 0] - coil_center[0])**2 +
                       (centers[:, 1] - coil_center[1])**2)
    is_coil = distances < coil_radius

    return mesh, is_coil
