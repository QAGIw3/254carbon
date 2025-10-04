"""
Test script for transformer model implementation.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Test the transformer model
def test_transformer_model():
    """Test basic transformer model functionality."""
    print("Testing transformer model...")

    try:
        from models import TransformerPriceForecastModel

        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        input_size = 10

        # Generate sample features (time series data)
        X = np.random.randn(n_samples, input_size).astype(np.float32)

        # Generate sample targets (price-like data)
        # Add some autocorrelation to make it more realistic
        y = np.zeros(n_samples, dtype=np.float32)
        for i in range(1, n_samples):
            y[i] = 0.7 * y[i-1] + 0.3 * np.random.randn()

        # Add trend and seasonality
        trend = np.linspace(50, 60, n_samples)
        seasonal = 5 * np.sin(np.linspace(0, 4*np.pi, n_samples))
        y = y + trend + seasonal

        print(f"Sample data shape: X={X.shape}, y={y.shape}")
        print(f"Data range: y min={y.min():.2f}, max={y.max():.2f}")

        # Initialize and train model
        model = TransformerPriceForecastModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
            learning_rate=1e-3,
            max_epochs=10,  # Quick test
        )

        print("Training model...")
        model.fit(X, y)

        print("Generating predictions...")
        # Test prediction
        test_X = X[:10]  # First 10 samples
        predictions = model.predict(test_X)

        print(f"Predictions shape: {predictions.shape}")
        print(f"Sample predictions: {predictions[:5]}")
        print(f"Actual values: {y[:5]}")

        # Test prediction intervals
        print("Testing prediction intervals...")
        ci_lower, ci_upper = model.predict_interval(test_X)

        print(f"Confidence interval shape: {ci_lower.shape}")
        print(
            f"Sample CI: pred={predictions[0]:.2f}, "
            f"lower={ci_lower[0]:.2f}, upper={ci_upper[0]:.2f}"
        )

        print("✅ Transformer model test passed!")
        return True

    except Exception as e:
        print(f"❌ Transformer model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_transformer_model()
    if success:
        print("\n🎉 All transformer model tests passed!")
    else:
        print("\n💥 Some transformer model tests failed!")
        exit(1)
