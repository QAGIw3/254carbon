"""
QP solver for curve smoothing using CVXPY and OSQP.
"""
import logging
from typing import Optional

import numpy as np
import cvxpy as cp

logger = logging.getLogger(__name__)


def smooth_curve(
    mu: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    lam: float = 50.0,
) -> Optional[np.ndarray]:
    """
    Smooth forward curve using quadratic programming.
    
    Minimizes:
        Σ_t (p_t - μ_t)^2 + λ Σ_t (Δ² p_t)^2
    
    Subject to:
        lb_t ≤ p_t ≤ ub_t
        p_t ≥ 0
    
    Args:
        mu: Target prices from fundamentals (T,)
        lb: Lower bounds (T,)
        ub: Upper bounds (T,)
        lam: Smoothness penalty weight
    
    Returns:
        Optimized prices (T,) or None if infeasible
    """
    T = len(mu)
    
    # Decision variable
    p = cp.Variable(T)
    
    # Second-order difference matrix for smoothness
    D = np.diff(np.eye(T), n=2, axis=0)
    
    # Objective: fit + smoothness
    fit_term = cp.sum_squares(p - mu)
    smooth_term = lam * cp.sum_squares(D @ p)
    objective = cp.Minimize(fit_term + smooth_term)
    
    # Constraints
    constraints = [
        p >= lb,
        p <= ub,
        p >= 0,  # No negative prices
    ]
    
    # Solve
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(
            solver=cp.OSQP,
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=10000,
            verbose=False,
        )
        
        if problem.status == cp.OPTIMAL:
            logger.info(f"QP solved optimally, objective: {problem.value:.2f}")
            return np.array(p.value, dtype=float)
        else:
            logger.warning(f"QP solver status: {problem.status}")
            return None
            
    except Exception as e:
        logger.error(f"QP solver error: {e}")
        return None


def reconcile_tenors(monthly_prices: np.ndarray) -> np.ndarray:
    """
    Reconcile monthly prices into consistent quarterly and annual tenors.
    
    Ensures hierarchical add-up: monthly -> quarterly -> annual.
    
    Args:
        monthly_prices: Monthly forward prices
    
    Returns:
        Reconciled monthly prices
    """
    # Simplified reconciliation - in production would enforce
    # weighted average consistency across tenor hierarchies
    
    n_months = len(monthly_prices)
    reconciled = monthly_prices.copy()
    
    # Smooth quarterly averages back into months
    for q in range(0, n_months - 2, 3):
        if q + 2 < n_months:
            q_avg = np.mean(monthly_prices[q : q + 3])
            # Soft adjustment toward quarterly consistency
            reconciled[q : q + 3] = (
                0.7 * reconciled[q : q + 3] + 0.3 * q_avg
            )
    
    return reconciled

