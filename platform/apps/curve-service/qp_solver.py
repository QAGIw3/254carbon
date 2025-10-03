"""
Enhanced QP solver for curve smoothing with tenor reconciliation and basis modeling.
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
    tenor_weights: Optional[np.ndarray] = None,
    basis_constraints: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Enhanced forward curve smoothing with tenor reconciliation and basis constraints.

    Minimizes:
        Σ_t w_t (p_t - μ_t)^2 + λ Σ_t (Δ² p_t)^2 + β Σ_t |p_t - b_t|²

    Subject to:
        lb_t ≤ p_t ≤ ub_t
        p_t ≥ 0
        Tenor reconciliation: monthly → quarterly → annual consistency

    Args:
        mu: Target prices from fundamentals (T,)
        lb: Lower bounds (T,)
        ub: Upper bounds (T,)
        lam: Smoothness penalty weight
        tenor_weights: Weights for different tenors (T,)
        basis_constraints: Basis surface constraints (T,)

    Returns:
        Optimized prices (T,) or None if infeasible
    """
    T = len(mu)

    # Decision variable
    p = cp.Variable(T)

    # Weights for different objectives
    fit_weights = tenor_weights if tenor_weights is not None else np.ones(T)
    basis_penalty = 10.0  # Penalty for basis constraint violations

    # Second-order difference matrix for smoothness
    D = np.diff(np.eye(T), n=2, axis=0)

    # Enhanced objective: fit + smoothness + basis consistency
    fit_term = cp.sum(fit_weights * cp.square(p - mu))
    smooth_term = lam * cp.sum_squares(D @ p)

    # Basis constraint penalty (soft constraint)
    basis_term = 0
    if basis_constraints is not None:
        basis_term = basis_penalty * cp.sum_squares(p - basis_constraints)

    objective = cp.Minimize(fit_term + smooth_term + basis_term)
    
    # Enhanced constraints with tenor reconciliation
    constraints = [
        p >= lb,
        p <= ub,
        p >= 0,  # No negative prices
    ]

    # Add tenor reconciliation constraints (monthly → quarterly → annual)
    # This ensures consistency across different contract tenors
    if T >= 12:  # Need at least 12 months for quarterly constraints
        # Quarterly averages should be consistent with monthly prices
        quarterly_indices = np.arange(2, T, 3)  # Every 3rd month starting from month 3
        for i in quarterly_indices:
            if i + 2 < T:  # Ensure we have 3 months for the quarter
                # Average of months i-2, i-1, i should equal month i
                constraints.append(p[i] == (p[i-2] + p[i-1] + p[i]) / 3)

    # Annual constraints (if we have enough data)
    if T >= 12:
        annual_indices = np.arange(11, T, 12)  # Every 12th month
        for i in annual_indices:
            if i >= 11:  # Need at least 12 months for annual constraint
                # Average of previous 12 months should be consistent
                constraints.append(p[i] == cp.sum(p[i-11:i+1]) / 12)
    
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


def calculate_basis_constraints(
    hub_prices: np.ndarray,
    node_prices: np.ndarray,
    correlation_threshold: float = 0.7,
) -> np.ndarray:
    """
    Calculate basis surface constraints for curve optimization.

    Uses historical hub-node price relationships to constrain
    the optimization and ensure realistic basis relationships.

    Args:
        hub_prices: Historical hub prices (T,)
        node_prices: Historical node prices (T,)
        correlation_threshold: Minimum correlation for basis constraint

    Returns:
        Basis constraint array (T,)
    """
    if len(hub_prices) != len(node_prices):
        raise ValueError("Hub and node prices must have same length")

    # Calculate historical basis
    historical_basis = node_prices - hub_prices

    # Calculate correlation
    correlation = np.corrcoef(hub_prices, node_prices)[0, 1]

    if abs(correlation) < correlation_threshold:
        # Low correlation, use simple average basis
        basis_constraints = hub_prices + np.mean(historical_basis)
    else:
        # High correlation, use regression-based basis
        # Simple linear regression: basis = slope * hub_price + intercept
        from scipy import stats
        slope, intercept, _, _, _ = stats.linregress(hub_prices, historical_basis)
        basis_constraints = hub_prices + (slope * hub_prices + intercept)

    return basis_constraints


def optimize_curve_with_tenor_reconciliation(
    mu: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    tenors: list = ['monthly', 'quarterly', 'annual'],
    **kwargs
) -> dict:
    """
    Optimize curve with tenor reconciliation across different contract periods.

    Args:
        mu: Target prices (T,)
        lb: Lower bounds (T,)
        ub: Upper bounds (T,)
        tenors: List of tenor types to reconcile
        **kwargs: Additional arguments for smooth_curve

    Returns:
        Dict with optimized prices for each tenor and reconciliation info
    """
    results = {}

    # Base optimization (monthly)
    monthly_prices = smooth_curve(mu, lb, ub, **kwargs)
    results['monthly'] = monthly_prices

    if monthly_prices is None:
        return results

    # Quarterly reconciliation
    if 'quarterly' in tenors and len(mu) >= 12:
        # Aggregate monthly to quarterly
        quarterly_mu = []
        quarterly_lb = []
        quarterly_ub = []

        for i in range(0, len(mu), 3):
            if i + 2 < len(mu):
                quarterly_mu.append(np.mean(mu[i:i+3]))
                quarterly_lb.append(np.mean(lb[i:i+3]))
                quarterly_ub.append(np.mean(ub[i:i+3]))

        quarterly_mu = np.array(quarterly_mu)
        quarterly_lb = np.array(quarterly_lb)
        quarterly_ub = np.array(quarterly_ub)

        quarterly_prices = smooth_curve(quarterly_mu, quarterly_lb, quarterly_ub, **kwargs)
        results['quarterly'] = quarterly_prices

    # Annual reconciliation
    if 'annual' in tenors and len(mu) >= 12:
        # Aggregate to annual
        annual_mu = np.mean(mu[:12]) if len(mu) >= 12 else np.mean(mu)
        annual_lb = np.mean(lb[:12]) if len(mu) >= 12 else np.mean(lb)
        annual_ub = np.mean(ub[:12]) if len(mu) >= 12 else np.mean(ub)

        annual_prices = smooth_curve(
            np.array([annual_mu]),
            np.array([annual_lb]),
            np.array([annual_ub]),
            **kwargs
        )
        results['annual'] = annual_prices

    return results

