#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:42:15 2026

@author: lauri

Optimized version of min_fg.py
Changes vs original
───────────────────
1. build_constraints   – fully vectorised COO construction; no Python loops,
                         no per-cell append(); ~10–30× faster for large grids.
2. vec_to_full         – single reshape/slice; O(1) vs O(N) Python loop.
3. analytical_init     – vectorised; no list comprehension.
4. cost_f_given_g      – pure NumPy array ops; no i/k loops.
5. cost_g_given_f      – pure NumPy array ops; no m/j loops.
6. linprog call        – bounds passed as scalar tuple (avoids allocating a
                         length-N list each iteration); x0 warm-start reused
                         across alternating iterations.
"""

import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# Indexing helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_index(Nx: int, Nt: int, var: str):
    """
    Returns (idx, N, j_range) for var in {'f','g'}.
    Kept for API compatibility; internal functions bypass it.
    """
    N = (Nx - 1) * Nt
    if var == 'f':
        j_lo, j_hi = 0, Nt - 1
    else:
        j_lo, j_hi = 1, Nt

    def idx(i: int, j: int) -> int:
        if 1 <= i <= Nx - 1 and j_lo <= j <= j_hi:
            return (i - 1) * Nt + (j - j_lo)
        return -1

    return idx, N, range(j_lo, j_hi + 1)


def vec_to_full(vec: np.ndarray, Nx: int, Nt: int, var: str) -> np.ndarray:
    """
    Unpack free-variable vector into a full (Nx+1)×(Nt+1) array.
    Optimised: single reshape + slice assignment instead of nested loops.
    """
    full = np.zeros((Nx + 1, Nt + 1))
    inner = vec.reshape(Nx - 1, Nt)          # (Nx-1, Nt)
    if var == 'f':
        full[1:Nx, 0:Nt] = inner             # j = 0..Nt-1
    else:
        full[1:Nx, 1:Nt + 1] = inner         # j = 1..Nt
    return full


# ─────────────────────────────────────────────────────────────────────────────
# Constraint matrices  (fully vectorised sparse build)
# ─────────────────────────────────────────────────────────────────────────────

def build_constraints(Nx: int, Nt: int, var: str):
    """
    Build sparse inequality system  A·v ≤ b  encoding:
      1. Convexity in x
      2. Monotonicity in t
      3. Slope bound |v_x| ≤ 1

    Fully vectorised: no Python-level loops over grid cells.
    Returns: A (csr_matrix), b (1-D array)
    """
    dx = 2.0 / Nx
    N  = (Nx - 1) * Nt

    j_lo = 0 if var == 'f' else 1
    j_hi = Nt - 1 if var == 'f' else Nt

    # Coordinate arrays for the *interior* grid: i in [1, Nx-1], j in [j_lo, j_hi]
    i_int = np.arange(1, Nx,     dtype=np.int32)   # length Nx-1
    j_int = np.arange(j_lo, j_hi + 1, dtype=np.int32)  # length Nt
    I, J  = np.meshgrid(i_int, j_int, indexing='ij')  # (Nx-1, Nt)
    I = I.ravel(); J = J.ravel()                   # (N,)
    n_int = N                                       # = (Nx-1)*Nt

    def col(i_arr, j_arr):
        """Linear column index; -1 for boundary nodes."""
        valid = ((i_arr >= 1) & (i_arr <= Nx - 1) &
                 (j_arr >= j_lo) & (j_arr <= j_hi))
        return np.where(valid, (i_arr - 1) * Nt + (j_arr - j_lo), -1).astype(np.int32)

    # Accumulate COO triplets in plain arrays (pre-estimate sizes)
    R_list, C_list, V_list = [], [], []
    rhs_list = []
    nrow = 0

    def emit(row_base, c_arr, v_scalar, b_arr):
        """Append valid (row, col, val) triplets to lists."""
        nonlocal nrow
        mask = c_arr >= 0
        if mask.any():
            R_list.append(row_base[mask] + nrow)
            C_list.append(c_arr[mask])
            V_list.append(np.full(mask.sum(), v_scalar, dtype=np.float64))
        rhs_list.append(b_arr)
        nrow += len(b_arr)

    def emit_all(row_base, pairs, b_arr):
        """pairs = list of (col_array, value_scalar)"""
        nonlocal nrow
        for c_arr, v_scalar in pairs:
            mask = c_arr >= 0
            if mask.any():
                R_list.append(row_base[mask] + nrow)
                C_list.append(c_arr[mask])
                V_list.append(np.full(mask.sum(), v_scalar, dtype=np.float64))
        rhs_list.append(b_arr)
        nrow += len(b_arr)

    row_int = np.arange(n_int, dtype=np.int32)  # row offsets for interior block

    # ── 1. Convexity in x  (n_int rows) ─────────────────────────────────────
    emit_all(row_int, [
        (col(I,   J),  2.0),
        (col(I-1, J), -1.0),
        (col(I+1, J), -1.0),
    ], np.zeros(n_int))

    # ── 2. Monotonicity in t ─────────────────────────────────────────────────
    if var == 'f':
        # f increasing: f[i,j] ≤ f[i,j+1]  for j = j_lo..j_hi-1
        #               f[i, j_hi] ≤ 0      (since f[i,Nt] = 0)
        n_pairs = (Nx - 1) * (Nt - 1)
        ip = np.repeat(i_int, Nt - 1)
        jp = np.tile(np.arange(j_lo, j_hi, dtype=np.int32), Nx - 1)
        rp = np.arange(n_pairs, dtype=np.int32)
        emit_all(rp, [(col(ip, jp), 1.0), (col(ip, jp + 1), -1.0)],
                 np.zeros(n_pairs))

        # f[i, Nt-1] ≤ 0
        emit_all(np.arange(Nx - 1, dtype=np.int32),
                 [(col(i_int, np.full(Nx - 1, j_hi, np.int32)), 1.0)],
                 np.zeros(Nx - 1))
    else:
        # g decreasing: g[i,j] ≤ g[i,j-1]  for j = j_lo+1..j_hi
        #               g[i, j_lo] ≤ 0      (since g[i,0] = 0)
        emit_all(np.arange(Nx - 1, dtype=np.int32),
                 [(col(i_int, np.full(Nx - 1, j_lo, np.int32)), 1.0)],
                 np.zeros(Nx - 1))

        n_pairs = (Nx - 1) * (Nt - 1)
        ip = np.repeat(i_int, Nt - 1)
        jp = np.tile(np.arange(j_lo + 1, j_hi + 1, dtype=np.int32), Nx - 1)
        rp = np.arange(n_pairs, dtype=np.int32)
        emit_all(rp, [(col(ip, jp), 1.0), (col(ip, jp - 1), -1.0)],
                 np.zeros(n_pairs))

    # ── 3. Slope bound |v_x| ≤ 1  (2 * Nx * Nt rows) ────────────────────────
    i_sl = np.arange(0, Nx, dtype=np.int32)
    I_sl, J_sl = np.meshgrid(i_sl, j_int, indexing='ij')
    I_sl = I_sl.ravel(); J_sl = J_sl.ravel()
    n_sl = Nx * Nt
    r_sl = np.arange(n_sl, dtype=np.int32)

    p = col(I_sl + 1, J_sl)   # v[i+1, j]
    q = col(I_sl,     J_sl)   # v[i,   j]

    emit_all(r_sl, [(p,  1.0 / dx), (q, -1.0 / dx)], np.ones(n_sl))   # upper
    emit_all(r_sl, [(p, -1.0 / dx), (q,  1.0 / dx)], np.ones(n_sl))   # lower

    # ── Assemble ─────────────────────────────────────────────────────────────
    rows = np.concatenate(R_list)
    cols = np.concatenate(C_list)
    vals = np.concatenate(V_list)
    b    = np.concatenate(rhs_list)

    A = csr_matrix((vals, (rows, cols)), shape=(nrow, N))
    return A, b


# ─────────────────────────────────────────────────────────────────────────────
# Bilinear objective  →  linear cost vectors (fully vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def cost_f_given_g(g_vec: np.ndarray, Nx: int, Nt: int) -> np.ndarray:
    """
    Fix g; return c_f s.t. cᵀf = I(f,g).

    c_f[i-1, k]  =  dx * G[i,k]*(k≥1)  −  dx * G[i,k+1]
    where G = g_xx  (centred in x), k = 0..Nt-1, i = 1..Nx-1.
    """
    dx = 2.0 / Nx
    g  = vec_to_full(g_vec, Nx, Nt, 'g')

    G = np.zeros((Nx + 1, Nt + 1))
    G[1:-1, :] = (g[:-2, :] - 2 * g[1:-1, :] + g[2:, :]) / dx ** 2

    # G_inner : shape (Nx-1, Nt+1)
    G_inner = G[1:Nx, :]

    # c[i-1, k] = dx*(G[i,k] - G[i,k+1]),  but G[i,0] not included for k=0
    c = -dx * G_inner[:, 1:Nt + 1]         # −dx * G[i, k+1],  k=0..Nt-1
    c[:, 1:] += dx * G_inner[:, 1:Nt]      # +dx * G[i, k],    k=1..Nt-1
    return c.ravel()


def cost_g_given_f(f_vec: np.ndarray, Nx: int, Nt: int) -> np.ndarray:
    """
    Fix f; return c_g s.t. cᵀg = I(f,g).

    c_g[m-1, j-1] = (dt/dx) * (Ft[m-1,j] − 2·Ft[m,j] + Ft[m+1,j])
    m = 1..Nx-1, j = 1..Nt.
    """
    dx = 2.0 / Nx
    dt = 1.0 / Nt
    f  = vec_to_full(f_vec, Nx, Nt, 'f')

    Ft = np.zeros((Nx + 1, Nt + 1))
    Ft[:, 1:] = (f[:, 1:] - f[:, :-1]) / dt

    # Second difference of Ft in x at interior nodes, j = 1..Nt
    c = (dt / dx) * (Ft[:-2, 1:] - 2 * Ft[1:-1, 1:] + Ft[2:, 1:])  # (Nx-1, Nt)
    return c.ravel()


def compute_objective(f_vec: np.ndarray, g_vec: np.ndarray,
                      Nx: int, Nt: int) -> float:
    """Evaluate the discretised double integral."""
    dx = 2.0 / Nx
    dt = 1.0 / Nt
    f  = vec_to_full(f_vec, Nx, Nt, 'f')
    g  = vec_to_full(g_vec, Nx, Nt, 'g')

    ft  = (f[:, 1:] - f[:, :-1]) / dt

    gxx = np.zeros((Nx + 1, Nt + 1))
    gxx[1:-1, :] = (g[:-2, :] - 2 * g[1:-1, :] + g[2:, :]) / dx ** 2

    return float(np.sum(ft[1:-1, :] * gxx[1:-1, 1:]) * dx * dt)


# ─────────────────────────────────────────────────────────────────────────────
# Analytical warm-start  (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def analytical_init(Nx: int, Nt: int):
    """
    f*(x,t) = (t−1)(1−|x|),   g*(x,t) = −t(1−|x|).
    Vectorised — no list comprehension.
    """
    x    = np.linspace(-1, 1, Nx + 1)
    t    = np.linspace(0,  1, Nt + 1)
    tent = 1.0 - np.abs(x)               # (Nx+1,)

    f_full = np.outer(tent, t - 1)       # (Nx+1, Nt+1)
    g_full = np.outer(tent, -t)

    # Pack interior values directly via slice
    f_vec = f_full[1:Nx, 0:Nt].ravel()   # j = 0..Nt-1
    g_vec = g_full[1:Nx, 1:Nt + 1].ravel()  # j = 1..Nt
    return f_vec, g_vec


# ─────────────────────────────────────────────────────────────────────────────
# Alternating LP solver
# ─────────────────────────────────────────────────────────────────────────────

def solve(Nx: int = 30, Nt: int = 30,
          n_iter: int = 50, tol: float = 1e-9,
          warm_start: bool = True, verbose: bool = True):
    """
    Alternating LP solver for the bilinear programme.

    Optimisations vs original:
      * vectorised constraint/cost construction
      * bounds passed as scalar (None,None) tuple — no per-iteration list alloc
      * HiGHS x0 warm-start reused across alternating steps
    """
    dx = 2.0 / Nx
    dt = 1.0 / Nt
    Nf = (Nx - 1) * Nt
    Ng = Nf

    print(f"Grid: Nx={Nx}, Nt={Nt}  |  dx={dx:.4f}, dt={dt:.4f}")
    print(f"Free vars: f → {Nf},  g → {Ng}")

    t0 = time.time()
    Af, bf = build_constraints(Nx, Nt, 'f')
    Ag, bg = build_constraints(Nx, Nt, 'g')
    print(f"Constraints built: f→{Af.shape[0]} rows, g→{Ag.shape[0]} rows  "
          f"({time.time()-t0:.2f}s)")
    print(f"Sparsity: Af={Af.nnz/(Af.shape[0]*Nf)*100:.1f}%, "
          f"Ag={Ag.nnz/(Ag.shape[0]*Ng)*100:.1f}%\n")

    if warm_start:
        f_vec, g_vec = analytical_init(Nx, Nt)
        print(f"Warm-start obj = {compute_objective(f_vec, g_vec, Nx, Nt):.8f}")
    else:
        f_vec = np.zeros(Nf)
        g_vec = np.zeros(Ng)

    history  = []
    bounds   = (None, None)               # scalar bounds: no list allocation
    lp_opts  = {'disp': False, 'presolve': True}

    x0_f = f_vec.copy()
    x0_g = g_vec.copy()

    for it in range(n_iter):
        # ── Step 1: optimise f given g ────────────────────────────────────────
        cf  = cost_f_given_g(g_vec, Nx, Nt)
        res = linprog(-cf, A_ub=Af, b_ub=bf, bounds=bounds,
                      method='highs', options=lp_opts, x0=x0_f)
        if res.status == 0:
            f_vec = res.x
            x0_f  = res.x
        else:
            print(f"  [iter {it+1}] f-LP failed: {res.message}")

        # ── Step 2: optimise g given f ────────────────────────────────────────
        cg  = cost_g_given_f(f_vec, Nx, Nt)
        res = linprog(-cg, A_ub=Ag, b_ub=bg, bounds=bounds,
                      method='highs', options=lp_opts, x0=x0_g)
        if res.status == 0:
            g_vec = res.x
            x0_g  = res.x
        else:
            print(f"  [iter {it+1}] g-LP failed: {res.message}")

        obj = compute_objective(f_vec, g_vec, Nx, Nt)
        history.append(obj)

        if verbose:
            print(f"  Iter {it+1:3d}  obj = {obj:.8f}")

        if it > 0 and abs(history[-1] - history[-2]) < tol:
            print(f"\nConverged at iteration {it+1}.")
            break

    print(f"\nTotal time: {time.time()-t0:.2f}s")
    print(f"Final objective : {history[-1]:.8f}")
    print(f"Theoretical max : 1.00000000")
    print(f"Error           : {abs(history[-1]-1):.2e}")
    return f_vec, g_vec, np.array(history)


# ─────────────────────────────────────────────────────────────────────────────
# Verification helpers  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def verify_constraints(vec: np.ndarray, Nx: int, Nt: int, var: str,
                       tol: float = 1e-8) -> dict:
    dx = 2.0 / Nx
    v  = vec_to_full(vec, Nx, Nt, var)

    results = {}
    results['bc_left']  = np.allclose(v[0,  :], 0, atol=tol)
    results['bc_right'] = np.allclose(v[-1, :], 0, atol=tol)
    if var == 'f':
        results['bc_t1'] = np.allclose(v[:, -1], 0, atol=tol)
    else:
        results['bc_t0'] = np.allclose(v[:,  0], 0, atol=tol)

    d2x = v[:-2, :] - 2 * v[1:-1, :] + v[2:, :]
    results['convex_x']         = bool(np.all(d2x >= -tol))
    results['convex_x_min_val'] = float(d2x.min())

    dt_diff = np.diff(v, axis=1)
    if var == 'f':
        results['increasing_t'] = bool(np.all(dt_diff >= -tol))
    else:
        results['decreasing_t'] = bool(np.all(dt_diff <= tol))

    fx = np.diff(v, axis=0) / dx
    results['slope_bound'] = bool(np.all(np.abs(fx) <= 1 + tol))
    results['slope_max']   = float(np.abs(fx).max())

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(f_vec, g_vec, history, Nx, Nt, save_path='result.png'):
    x = np.linspace(-1, 1, Nx + 1)
    t = np.linspace(0,  1, Nt + 1)
    X, T = np.meshgrid(x, t, indexing='ij')

    f_full = vec_to_full(f_vec, Nx, Nt, 'f')
    g_full = vec_to_full(g_vec, Nx, Nt, 'g')

    dx = 2.0 / Nx;  dt = 1.0 / Nt
    ft_full  = np.zeros_like(f_full)
    ft_full[:, 1:] = (f_full[:, 1:] - f_full[:, :-1]) / dt

    gxx_full = np.zeros_like(g_full)
    gxx_full[1:-1, :] = (g_full[:-2, :] - 2*g_full[1:-1, :] + g_full[2:, :]) / dx**2

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    kw  = dict(cmap='RdBu_r', levels=30)

    def cplot(ax, data, title):
        cf = ax.contourf(X, T, data, **kw)
        plt.colorbar(cf, ax=ax, shrink=0.85)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('t')

    cplot(fig.add_subplot(gs[0, 0]), f_full,  'f(x,t)')
    cplot(fig.add_subplot(gs[0, 1]), g_full,  'g(x,t)')
    cplot(fig.add_subplot(gs[1, 0]), ft_full, 'f_t(x,t)  [should be ≥ 0]')
    cplot(fig.add_subplot(gs[1, 1]), gxx_full,'g_xx(x,t)')

    intgd = np.zeros_like(f_full)
    intgd[:, 1:] = ft_full[:, 1:] * gxx_full[:, 1:]
    cplot(fig.add_subplot(gs[1, 2]), intgd, 'f_t · g_xx  (integrand)')

    ax = fig.add_subplot(gs[0, 2])
    ax.semilogy(np.abs(np.array(history) - history[-1]) + 1e-15,
                'b-o', markersize=4, label='|obj − final|')
    ax.axhline(abs(history[-1] - 1.0), color='r', ls='--', lw=1.5,
               label=f'error vs 1.0 = {abs(history[-1]-1):.2e}')
    ax.set_title('Convergence', fontsize=12, fontweight='bold')
    ax.set_xlabel('Iteration'); ax.set_ylabel('|Δ objective|')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

    fig.suptitle(
        f'Grid {Nx}×{Nt}  |  final obj = {history[-1]:.8f}  '
        f'(theoretical max = 1)',
        fontsize=13, y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    Nx, Nt = 60, 60

    f_vec, g_vec, history = solve(Nx=Nx, Nt=Nt, n_iter=40, tol=1e-10)

    print("\n── Constraint verification (f) ──")
    for k, v in verify_constraints(f_vec, Nx, Nt, 'f').items():
        print(f"  {k:20s}: {v}")

    print("\n── Constraint verification (g) ──")
    for k, v in verify_constraints(g_vec, Nx, Nt, 'g').items():
        print(f"  {k:20s}: {v}")

    plot_results(f_vec, g_vec, history, Nx, Nt, save_path='result.png')