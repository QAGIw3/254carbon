"""
Deep Learning Models for Price Forecasting

Implements transformer-based models with attention mechanisms
for multi-horizon forecasting.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    
    Injects information about the relative position of tokens.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Allows the model to jointly attend to information from different
    representation subspaces.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
    
    def forward(
        self,
        query,
        key,
        value,
        mask: Optional[torch.Tensor] = None
    ):
        """Apply multi-head attention."""
        batch_size = query.size(0)
        
        # Linear projections in batch
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(query.device)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        x = torch.matmul(attention, V)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        return self.W_o(x)


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        """Apply transformer block."""
        # Multi-head attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class PriceForecaster(nn.Module):
    """
    Transformer-based price forecasting model.
    
    Multi-horizon forecasting with uncertainty quantification.
    """
    
    def __init__(
        self,
        input_features: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 168,  # 1 week of hourly data
        forecast_horizons: List[int] = [1, 6, 24, 168],  # 1h, 6h, 1d, 1w
    ):
        super().__init__()
        
        self.input_features = input_features
        self.d_model = d_model
        self.forecast_horizons = forecast_horizons
        
        # Input embedding
        self.input_projection = nn.Linear(input_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Multi-horizon output heads
        self.output_heads = nn.ModuleDict({
            f"horizon_{h}": nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, 2),  # Mean and std for uncertainty
            )
            for h in forecast_horizons
        })
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, features)
            mask: Optional attention mask
        
        Returns:
            Dictionary of forecasts by horizon with mean and std
        """
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Take the last timestep's representation
        x = x[:, -1, :]
        
        # Generate multi-horizon forecasts
        forecasts = {}
        for horizon in self.forecast_horizons:
            output = self.output_heads[f"horizon_{horizon}"](x)
            forecasts[horizon] = {
                "mean": output[:, 0],
                "std": torch.exp(output[:, 1]),  # Ensure positive std
            }
        
        return forecasts


class PriceDataset(Dataset):
    """Dataset for price forecasting."""
    
    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        seq_len: int = 168,
        forecast_horizons: List[int] = [1, 6, 24, 168],
    ):
        self.prices = prices
        self.features = features
        self.seq_len = seq_len
        self.forecast_horizons = forecast_horizons
        
        # Maximum horizon to ensure we have targets
        self.max_horizon = max(forecast_horizons)
    
    def __len__(self):
        return len(self.prices) - self.seq_len - self.max_horizon
    
    def __getitem__(self, idx):
        # Input sequence
        x = self.features[idx:idx + self.seq_len]
        
        # Targets for each horizon
        targets = {}
        for h in self.forecast_horizons:
            targets[h] = self.prices[idx + self.seq_len + h - 1]
        
        return torch.FloatTensor(x), targets


def train_transformer_model(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> PriceForecaster:
    """
    Train transformer forecasting model.
    
    Args:
        train_data: (prices, features) for training
        val_data: (prices, features) for validation
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    
    Returns:
        Trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    
    # Create datasets
    train_prices, train_features = train_data
    val_prices, val_features = val_data
    
    train_dataset = PriceDataset(train_prices, train_features)
    val_dataset = PriceDataset(val_prices, val_features)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Create model
    model = PriceForecaster(
        input_features=train_features.shape[1],
        d_model=128,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_targets in train_loader:
            batch_x = batch_x.to(device)
            
            # Forward pass
            forecasts = model(batch_x)
            
            # Calculate loss (negative log-likelihood assuming Gaussian)
            loss = 0.0
            for horizon in model.forecast_horizons:
                target = batch_targets[horizon].to(device)
                mean = forecasts[horizon]["mean"]
                std = forecasts[horizon]["std"]
                
                # Negative log-likelihood
                nll = 0.5 * torch.log(2 * np.pi * std**2) + \
                      0.5 * ((target - mean) ** 2) / (std ** 2)
                loss += nll.mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_targets in val_loader:
                batch_x = batch_x.to(device)
                
                forecasts = model(batch_x)
                
                loss = 0.0
                for horizon in model.forecast_horizons:
                    target = batch_targets[horizon].to(device)
                    mean = forecasts[horizon]["mean"]
                    std = forecasts[horizon]["std"]
                    
                    nll = 0.5 * torch.log(2 * np.pi * std**2) + \
                          0.5 * ((target - mean) ** 2) / (std ** 2)
                    loss += nll.mean()
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Logging
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_transformer_model.pth")
    
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load("best_transformer_model.pth"))
    
    return model


if __name__ == "__main__":
    # Test transformer model
    logger.info("Testing Transformer Price Forecaster")
    
    # Generate synthetic data
    np.random.seed(42)
    seq_len = 1000
    
    # Prices with trend and seasonality
    t = np.arange(seq_len)
    prices = 50 + 0.01 * t + 10 * np.sin(2 * np.pi * t / 24) + np.random.randn(seq_len) * 5
    
    # Features: hour, day_of_week, lagged prices, etc.
    features = np.column_stack([
        t % 24,  # Hour of day
        (t // 24) % 7,  # Day of week
        np.roll(prices, 1),  # Lag-1 price
        np.roll(prices, 24),  # Lag-24 price
        np.roll(prices, 168),  # Lag-168 price (1 week)
    ])
    
    # Split data
    split = int(0.8 * seq_len)
    train_data = (prices[:split], features[:split])
    val_data = (prices[split:], features[split:])
    
    # Train model
    model = train_transformer_model(
        train_data,
        val_data,
        epochs=10,
        batch_size=16,
    )
    
    logger.info("Transformer model training complete!")

