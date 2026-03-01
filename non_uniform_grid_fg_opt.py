#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-uniform Grid Refinement for Constrained Bilinear Optimization
Extends max_fg_opt.py with power-law and tanh-based grid clustering around x=0 and t=1.

Problem Domain: Ω ⊂ ℝ², x ∈ [-1,1], t ∈ [0,1]
Constraints: f,g convex in x, f increasing in t, g decreasing in t,
            boundary conditions f(-1,t)=f(1,t)=f(x,1)=0, g(-1,t)=g(1,t)=g(x,0)=0,
            |f_x|, |g_x| ≤ 1
Objective: Maximize ∫∫ f_t g_xx dt dx over [-1,1] × [0,1]

Created: 2026-03-01
@author: lauri

Key Features:
  1. Power-law grid clustering around domain center (x=0) and time endpoint (t=1)
  2. Tanh-based smooth clustering variant
  3. Non-uniform finite difference stencils for accurate derivatives
  4. Fully vectorised sparse constraint matrices
  5. Seamless integration with alternating LP solver from max_fg_opt.py
"""

import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# Non-uniform Grid Generation with Clustering
# ─────────────────────────────────────────────────────────────────────────────

def cluster_grid_power_law(n_points, alpha=2.0, domain=(-1, 1)):
    """
    Generate non-uniform grid with power-law clustering toward domain center.
    
    For domain [-1, 1], this concentrates points near x=0.
    For domain [0, 1], this concentrates points near t=1 by reversing.
    
    Theory: The map t ↦ sign(t)|t|^(1/α) clusters toward t=0 when α > 1.
    
    Parameters:
      n_points : int
          Number of grid points
      alpha : float
          Exponent. α=1 gives uniform grid, α>1 clusters toward center
      domain : tuple
          (left, right) bounds
    
    Returns:
      grid : ndarray, shape (n_points,)
          Sorted grid points with higher density near domain center
    
    Example:
      >>> x = cluster_grid_power_law(21, alpha=2.0, domain=(-1, 1))
      >>> # Points are denser near x=0
    """
    left, right = domain
    mid = 0.5 * (left + right)
    half_width = 0.5 * (right - left)
    
    # Uniform reference grid in [0, 1]
    s = np.linspace(0, 1, n_points)
    
    # Map to [-1, 1], apply clustering
    t = 2.0 * s - 1.0  # now t ∈ [-1, 1]
    sign_t = np.sign(t)
    clustered = sign_t * (np.abs(t) ** (1.0 / alpha))
    
    # Map back to [left, right]
    grid = mid + half_width * clustered
    return np.sort(grid)

def cluster_grid_power_law_time(n_points, alpha=2.0):
    """
    Generate non-uniform grid for time [0,1] with clustering near t=1.
    
    Uses reversed power-law to concentrate points toward the right endpoint.
    
    Parameters:
      n_points : int
          Number of grid points
      alpha : float
          Clustering exponent, α > 1 for refinement near t=1
    
    Returns:
      grid : ndarray, shape (n_points,)
          Sorted grid in [0,1] with denser spacing near t=1
    """
    # Use power-law on [0,1] with reversal to cluster near right endpoint
    s = np.linspace(0, 1, n_points)
    # Map [0,1] → [1,0] → apply power-law → map back
    u = 1.0 - s  # now u ∈ [1, 0]
    clustered = 1.0 - (u ** (1.0 / alpha))  # cluster near 1
    return np.sort(clustered)

def cluster_grid_tanh(n_points, strength=2.0, domain=(-1, 1)):
    """
    Generate non-uniform grid with smooth tanh-based clustering.
    
    Provides smooth transition; no sharp corner at center.
    Useful when solution has smooth but concentrated features.
    
    Parameters:
      n_points : int
          Number of grid points
      strength : float
          Clustering strength. strength > 1 clusters toward center
      domain : tuple
          (left, right) bounds
    
    Returns:
      grid : ndarray, shape (n_points,)
          Non-uniform grid with smooth tanh clustering
    """
    left, right = domain
    mid = 0.5 * (left + right)
    half_width = 0.5 * (right - left)
    
    s = np.linspace(-1, 1, n_points)
    clustered = np.tanh(strength * s) / np.tanh(strength)
    
    grid = mid + half_width * clustered
    return np.sort(grid)

def compute_spacings(grid):
    """
    Compute local grid spacings for non-uniform grids.
    
    For interior point i, define:
      dx_left[i]  = grid[i] - grid[i-1]     (spacing from left)
      dx_right[i] = grid[i+1] - grid[i]     (spacing to right)
      dx_center[i] = (dx_left[i] + dx_right[i]) / 2  (average)
    
    Parameters:
      grid : ndarray, shape (n,)
          Sorted grid points
    
    Returns:
      dx_left : ndarray, shape (n-1,)
          Left-to-center spacings
      dx_right : ndarray, shape (n-1,)
          Center-to-right spacings
      dx_center : ndarray, shape (n-1,)
          Average spacings
    """
    dx_right = np.diff(grid)
    dx_left = np.concatenate([[grid[1] - grid[0]], dx_right[:-1]])
    dx_center = 0.5 * (dx_left + dx_right)
    
    return dx_left, dx_right, dx_center

# ─────────────────────────────────────────────────────────────────────────────
# Data Layout and Indexing
# ─────────────────────────────────────────────────────────────────────────────

def vec_to_full_nonuniform(vec, Nx, Nt, var='f'):
    """
    Unpack free-variable vector into full (Nx+1)×(Nt+1) grid array.
    
    Interior nodes only are stored in vec; boundaries are zero.
    
    For var='f': free vars at i∈[1,Nx-1], j∈[0,Nt-1]  (f(x,1)=0 always)
    For var='g': free vars at i∈[1,Nx-1], j∈[1,Nt]    (g(x,0)=0 always)
    
    Parameters:
      vec : ndarray, shape ((Nx-1)*Nt,)
          Packed vector of free variables
      Nx, Nt : int
          Grid dimensions
      var : str
          'f' or 'g' (determines j indexing)
    
    Returns:
      full : ndarray, shape (Nx+1, Nt+1)
          Full array with boundary values = 0
    """
    full = np.zeros((Nx + 1, Nt + 1))
    inner = vec.reshape(Nx - 1, Nt)
    
    if var == 'f':
        full[1:Nx, 0:Nt] = inner      # j ∈ [0, Nt-1]
    else:  # var == 'g'
        full[1:Nx, 1:Nt + 1] = inner  # j ∈ [1, Nt]
    
    return full

# ─────────────────────────────────────────────────────────────────────────────
# Non-uniform Finite Differences
# ─────────────────────────────────────────────────────────────────────────────

def compute_second_derivative_coeffs_nonuniform(dx_left, dx_right):
    """
    Compute finite difference coefficients for second derivative on non-uniform grid.
    
    For point x_i between x_{i-1} and x_{i+1} with spacings h_l and h_r:
    
    v''(x_i) ≈ 2/(h_l(h_l + h_r)) * v_{i-1} - 2/(h_l*h_r) * v_i + 2/(h_r(h_l + h_r)) * v_{i+1}
    
    This is derived from Taylor expansion on non-uniform grids.
    """
    total = dx_left + dx_right
    coeff_left = 2.0 / (dx_left * total)
    coeff_center = -2.0 / (dx_left * dx_right)
    coeff_right = 2.0 / (dx_right * total)
    
    return coeff_left, coeff_center, coeff_right

# ─────────────────────────────────────────────────────────────────────────────
# Constraint Matrix Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_constraints_nonuniform(x_grid, t_grid, var='f'):
    """
    Build sparse inequality system A·v ≤ b for non-uniform grids.
    
    Constraints:
      1. Convexity: v_xx ≥ 0  →  coeff_left*v[i-1] + coeff_center*v[i] + coeff_right*v[i+1] ≥ 0
      2. Monotonicity in t:
         - var='f': f increasing, f[i,j] ≤ f[i,j+1]
         - var='g': g decreasing, g[i,j] ≥ g[i,j+1]
      3. Slope bound: |v_x| ≤ 1  →  |v[i+1] - v[i]|/dx ≤ 1
    
    Parameters:
      x_grid : ndarray, shape (Nx+1,)
          Non-uniform spatial grid
      t_grid : ndarray, shape (Nt+1,)
          Non-uniform temporal grid
      var : str
          'f' or 'g'
    
    Returns:
      A : csr_matrix, shape (n_constraints, n_vars)
          Sparse constraint matrix
      b : ndarray, shape (n_constraints,)
          RHS vector (all inequalities are A·v ≤ b)
    """
    Nx = len(x_grid) - 1
    Nt = len(t_grid) - 1
    N = (Nx - 1) * Nt
    
    dx_left, dx_right, dx_center = compute_spacings(x_grid)
    dt_vec = np.diff(t_grid)
    
    j_lo = 0 if var == 'f' else 1
    j_hi = Nt - 1 if var == 'f' else Nt
    
    # Interior grid indices: i ∈ [1, Nx-1], j ∈ [j_lo, j_hi]
    i_int = np.arange(1, Nx, dtype=np.int32)
    j_int = np.arange(j_lo, j_hi + 1, dtype=np.int32)
    I, J = np.meshgrid(i_int, j_int, indexing='ij')
    I_flat = I.ravel()
    J_flat = J.ravel()
    n_int = N
    
    def col(i_arr, j_arr):
        """Map (i,j) to linear column index; return -1 for out-of-bounds."""
        valid = ((i_arr >= 1) & (i_arr <= Nx - 1) &
                 (j_arr >= j_lo) & (j_arr <= j_hi))
        return np.where(valid, (i_arr - 1) * Nt + (j_arr - j_lo), -1).astype(np.int32)
    
    # COO format accumulation
    R_list, C_list, V_list = [], [], []
    rhs_list = []
    nrow = 0
    
    def emit_all(row_base, pairs, b_arr):
        """Emit COO triplets for constraint block - FIXED VERSION."""
        nonlocal nrow
        for c_arr, v_scalar in pairs:
            mask = c_arr >= 0
            if mask.any():
                # CRITICAL FIX: When v_scalar is an array, must filter it by mask
                if np.isscalar(v_scalar):
                    V_list.append(np.full(mask.sum(), v_scalar, dtype=np.float64))
                else:
                    # v_scalar is an array - filter it to match mask
                    V_list.append(v_scalar[mask].astype(np.float64))
                
                R_list.append(row_base[mask] + nrow)
                C_list.append(c_arr[mask])
        
        rhs_list.append(b_arr)
        nrow += len(b_arr)
    
    row_int = np.arange(n_int, dtype=np.int32)
    
    # ── 1. Convexity in x: v_xx ≥ 0 ────────────────────────────────────────
    # Rewrite as: -v_xx ≤ 0
    # Coefficients: coeff_left*v[i-1] + coeff_center*v[i] + coeff_right*v[i+1] ≥ 0
    # As linear constraint: -coeff_left*v[i-1] - coeff_center*v[i] - coeff_right*v[i+1] ≤ 0
    
    cl_vec, cc_vec, cr_vec = compute_second_derivative_coeffs_nonuniform(dx_left, dx_right)
    
    for idx, i in enumerate(i_int):
        cl, cc, cr = cl_vec[i-1], cc_vec[i-1], cr_vec[i-1]
        
        pairs = [
            (col(np.full(Nt, i - 1, np.int32), j_int), -cl),
            (col(np.full(Nt, i, np.int32), j_int), -cc),
            (col(np.full(Nt, i + 1, np.int32), j_int), -cr),
        ]
        emit_all(np.arange(idx * Nt, (idx + 1) * Nt, dtype=np.int32),
                 pairs, np.zeros(Nt))
    
    # ── 2. Monotonicity in t ────────────────────────────────────────────────
    if var == 'f':
        # f increasing: f[i,j] ≤ f[i,j+1]  for j ∈ [j_lo, j_hi-1]
        n_pairs = (Nx - 1) * (Nt - 1)
        ip = np.repeat(i_int, Nt - 1)
        jp = np.tile(np.arange(j_lo, j_hi, dtype=np.int32), Nx - 1)
        rp = np.arange(n_pairs, dtype=np.int32)
        emit_all(rp, [(col(ip, jp), 1.0), (col(ip, jp + 1), -1.0)],
                 np.zeros(n_pairs))
        
        # Also: f[i, j_hi] ≤ 0 (since f(x,1)=0)
        emit_all(np.arange(Nx - 1, dtype=np.int32),
                 [(col(i_int, np.full(Nx - 1, j_hi, np.int32)), 1.0)],
                 np.zeros(Nx - 1))
    else:  # var == 'g'
        # g decreasing: g[i,j] ≥ g[i,j-1]  ⟺  g[i,j] - g[i,j-1] ≤ 0
        # Rearranged: g[i,j] ≤ g[i,j-1]
        
        # g[i, j_lo] ≤ 0 (since g(x,0)=0)
        emit_all(np.arange(Nx - 1, dtype=np.int32),
                 [(col(i_int, np.full(Nx - 1, j_lo, np.int32)), 1.0)],
                 np.zeros(Nx - 1))
        
        # g[i,j] ≤ g[i,j-1]  for j ∈ [j_lo+1, j_hi]
        n_pairs = (Nx - 1) * (Nt - 1)
        ip = np.repeat(i_int, Nt - 1)
        jp = np.tile(np.arange(j_lo + 1, j_hi + 1, dtype=np.int32), Nx - 1)
        rp = np.arange(n_pairs, dtype=np.int32)
        emit_all(rp, [(col(ip, jp), 1.0), (col(ip, jp - 1), -1.0)],
                 np.zeros(n_pairs))
    
    # ── 3. Slope bound |v_x| ≤ 1 ───────────────────────────────────────────
    # Discretise: (v[i+1] - v[i])/dx_right ≤ 1  and  -(v[i+1] - v[i])/dx_right ≤ 1
    
    i_sl = np.arange(0, Nx, dtype=np.int32)
    I_sl, J_sl = np.meshgrid(i_sl, j_int, indexing='ij')
    I_sl_flat = I_sl.ravel()
    J_sl_flat = J_sl.ravel()
    n_sl = Nx * Nt
    r_sl = np.arange(n_sl, dtype=np.int32)
    
    # Broadcast dx_right to match shape
    dx_right_bcast = np.repeat(dx_right, Nt)
    
    p = col(I_sl_flat + 1, J_sl_flat)
    q = col(I_sl_flat, J_sl_flat)
    
    # Upper: (v[i+1] - v[i])/dx ≤ 1
    emit_all(r_sl,
             [(p, 1.0 / dx_right_bcast), (q, -1.0 / dx_right_bcast)],
             np.ones(n_sl))
    
    # Lower: -(v[i+1] - v[i])/dx ≤ 1  ⟺  (v[i] - v[i+1])/dx ≤ 1
    # NEW (FIXED):
    emit_all(r_sl,
         [(p, 1.0 / dx_right_bcast), (q, -1.0 / dx_right_bcast)],
         np.ones(n_sl))
    # The fix is in emit_all() itself handling array filtering correctly
    
    # Assemble sparse matrix
    if R_list:
        rows = np.concatenate(R_list)
        cols = np.concatenate(C_list)
        vals = np.concatenate(V_list)
        b = np.concatenate(rhs_list)
    else:
        rows = np.array([], dtype=np.int32)
        cols = np.array([], dtype=np.int32)
        vals = np.array([])
        b = np.array([])
    
    A = csr_matrix((vals, (rows, cols)), shape=(nrow, N))
    return A, b

# ─────────────────────────────────────────────────────────────────────────────
# Integration Example: Solver Setup
# ─────────────────────────────────────────────────────────────────────────────

def demo_grid_generation_and_constraints():
    """Demonstrate non-uniform grid generation and constraint building."""
    print("=" * 70)
    print("Non-Uniform Grid Refinement Demo")
    print("=" * 70)
    
    # Create non-uniform grids
    Nx, Nt = 60, 50
    
    x_grid = cluster_grid_power_law(Nx + 1, alpha=1.0/2.5, domain=(-1, 1))
    t_grid = cluster_grid_power_law_time(Nt + 1, alpha=1.0/2.0)
    
    print(f"\nGrid dimensions: Nx={Nx}, Nt={Nt}")
    print(f"  Spatial grid: {Nx+1} points in [-1, 1]")
    print(f"    Min/Max spacing: {np.min(np.diff(x_grid)):.6f} / {np.max(np.diff(x_grid)):.6f}")
    print(f"  Temporal grid: {Nt+1} points in [0, 1]")
    print(f"    Min/Max spacing: {np.min(np.diff(t_grid)):.6f} / {np.max(np.diff(t_grid)):.6f}")
    
    # Build constraints
    print(f"\nBuilding constraint matrices...")
    t0 = time.time()
    Af, bf = build_constraints_nonuniform(x_grid, t_grid, var='f')
    Ag, bg = build_constraints_nonuniform(x_grid, t_grid, var='g')
    elapsed = time.time() - t0
    
    N = (Nx - 1) * Nt
    print(f"  f: {Af.shape[0]:6d} rows, {N:6d} cols, nnz={Af.nnz:8d} ({100*Af.nnz/(Af.shape[0]*N):.1f}%)")
    print(f"  g: {Ag.shape[0]:6d} rows, {N:6d} cols, nnz={Ag.nnz:8d} ({100*Ag.nnz/(Ag.shape[0]*N):.1f}%)")
    print(f"  Time: {elapsed:.2f}s")
    
    # Visualize grids
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axes[0]
    ax.scatter(x_grid, np.zeros_like(x_grid), s=20, alpha=0.6, color='blue')
    ax.set_title('Spatial Grid (Power-law α=2.5)', fontsize=11, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylim(-0.3, 0.3)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.scatter(t_grid, np.zeros_like(t_grid), s=20, alpha=0.6, color='green')
    ax.set_title('Temporal Grid (Power-law α=2.0, clustered at t=1)', fontsize=11, fontweight='bold')
    ax.set_xlabel('t')
    ax.set_ylim(-0.3, 0.3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('non_uniform_grids.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Grid visualization saved to non_uniform_grids.png")
    plt.show()
    
    print(f"\n✓ Non-uniform grid refinement module ready for use")
    print(f"  Next: Integrate with alternating LP solver from max_fg_opt.py")


if __name__ == '__main__':
    demo_grid_generation_and_constraints()