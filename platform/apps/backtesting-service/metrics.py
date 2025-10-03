"""
Accuracy metrics for forecast evaluation.
"""
import numpy as np
from typing import Union


def calculate_mape(
    actual: Union[np.ndarray, list],
    forecast: Union[np.ndarray, list],
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    MAPE = (1/n) * Σ |actual - forecast| / |actual| * 100
    
    Args:
        actual: Realized values
        forecast: Forecasted values
    
    Returns:
        MAPE as percentage
    """
    actual = np.array(actual)
    forecast = np.array(forecast)
    
    # Avoid division by zero
    mask = actual != 0
    if not mask.any():
        return 0.0
    
    mape = np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100
    return float(mape)


def calculate_wape(
    actual: Union[np.ndarray, list],
    forecast: Union[np.ndarray, list],
) -> float:
    """
    Calculate Weighted Absolute Percentage Error (WAPE).
    
    WAPE = Σ |actual - forecast| / Σ |actual| * 100
    
    More robust than MAPE for low values.
    
    Args:
        actual: Realized values
        forecast: Forecasted values
    
    Returns:
        WAPE as percentage
    """
    actual = np.array(actual)
    forecast = np.array(forecast)
    
    sum_actual = np.sum(np.abs(actual))
    if sum_actual == 0:
        return 0.0
    
    wape = np.sum(np.abs(actual - forecast)) / sum_actual * 100
    return float(wape)


def calculate_rmse(
    actual: Union[np.ndarray, list],
    forecast: Union[np.ndarray, list],
) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    RMSE = sqrt((1/n) * Σ (actual - forecast)²)
    
    Args:
        actual: Realized values
        forecast: Forecasted values
    
    Returns:
        RMSE in same units as input
    """
    actual = np.array(actual)
    forecast = np.array(forecast)
    
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    return float(rmse)


def calculate_mae(
    actual: Union[np.ndarray, list],
    forecast: Union[np.ndarray, list],
) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    MAE = (1/n) * Σ |actual - forecast|
    
    Args:
        actual: Realized values
        forecast: Forecasted values
    
    Returns:
        MAE in same units as input
    """
    actual = np.array(actual)
    forecast = np.array(forecast)
    
    mae = np.mean(np.abs(actual - forecast))
    return float(mae)


def calculate_bias(
    actual: Union[np.ndarray, list],
    forecast: Union[np.ndarray, list],
) -> float:
    """
    Calculate forecast bias (mean error).
    
    Bias = (1/n) * Σ (forecast - actual)
    
    Positive bias = over-forecasting
    Negative bias = under-forecasting
    
    Args:
        actual: Realized values
        forecast: Forecasted values
    
    Returns:
        Bias in same units as input
    """
    actual = np.array(actual)
    forecast = np.array(forecast)
    
    bias = np.mean(forecast - actual)
    return float(bias)

