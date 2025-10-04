"""
Unit tests for Multi-Source Redundancy Router core engine.

Tests trust scoring, routing decisions, outlier removal, and circuit breaker.
"""
import pytest
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (
    TrustScoringEngine,
    RoutingPolicyEngine,
    CircuitBreaker,
    SourceValue,
    RoutingStrategy
)


class TestTrustScoringEngine:
    """Test trust score computation."""
    
    def test_perfect_score(self):
        """Test perfect trust score with optimal metrics."""
        weights = {
            'freshness': 0.30,
            'error_rate': 0.20,
            'deviation': 0.15,
            'consistency': 0.15,
            'uptime': 0.20
        }
        engine = TrustScoringEngine(weights)
        
        trust_score, components = engine.compute_trust_score(
            freshness_lag_sec=0,
            sla_freshness_sec=300,
            error_rate_win=0.0,
            deviation_from_blend=0.0,
            consistency_ratio=1.0,
            uptime_roll=1.0
        )
        
        assert trust_score == 1.0
        assert components['freshness'] == 1.0
        assert components['error_rate'] == 1.0
        assert components['deviation'] == 1.0
        assert components['consistency'] == 1.0
        assert components['uptime'] == 1.0
    
    def test_stale_data_penalty(self):
        """Test trust score penalty for stale data."""
        weights = {
            'freshness': 0.30,
            'error_rate': 0.20,
            'deviation': 0.15,
            'consistency': 0.15,
            'uptime': 0.20
        }
        engine = TrustScoringEngine(weights)
        
        # Data that exceeds SLA
        trust_score, components = engine.compute_trust_score(
            freshness_lag_sec=600,  # 10 minutes old
            sla_freshness_sec=300,  # 5 minute SLA
            error_rate_win=0.0,
            deviation_from_blend=0.0,
            consistency_ratio=1.0,
            uptime_roll=1.0
        )
        
        # Freshness should be penalized
        assert components['freshness'] < 0.5
        assert trust_score < 1.0
    
    def test_high_error_rate_penalty(self):
        """Test trust score penalty for high error rates."""
        weights = {
            'freshness': 0.30,
            'error_rate': 0.20,
            'deviation': 0.15,
            'consistency': 0.15,
            'uptime': 0.20
        }
        engine = TrustScoringEngine(weights)
        
        trust_score, components = engine.compute_trust_score(
            freshness_lag_sec=0,
            sla_freshness_sec=300,
            error_rate_win=0.5,  # 50% error rate
            deviation_from_blend=0.0,
            consistency_ratio=1.0,
            uptime_roll=1.0
        )
        
        # Error rate should be heavily penalized
        assert components['error_rate'] == 0.0
        assert trust_score < 0.9
    
    def test_deviation_penalty(self):
        """Test trust score penalty for deviation from consensus."""
        weights = {
            'freshness': 0.30,
            'error_rate': 0.20,
            'deviation': 0.15,
            'consistency': 0.15,
            'uptime': 0.20
        }
        engine = TrustScoringEngine(weights)
        
        trust_score, components = engine.compute_trust_score(
            freshness_lag_sec=0,
            sla_freshness_sec=300,
            error_rate_win=0.0,
            deviation_from_blend=0.3,  # 30% deviation
            consistency_ratio=1.0,
            uptime_roll=1.0,
            dev_ceiling=0.2
        )
        
        # Deviation should be penalized
        assert components['deviation'] == 0.0
        assert trust_score < 1.0


class TestRoutingPolicyEngine:
    """Test routing decision logic."""
    
    def get_test_policy_config(self):
        """Get test policy configuration."""
        return {
            'policy_version': 'test_v1',
            'min_trust': 0.55,
            'max_fresh_lag_sec': 180,
            'stable_dispersion': 0.012,
            'switch_margin': 0.07,
            'mad_k_factor': 3.0,
            'weights_json': {
                'freshness': 0.30,
                'error_rate': 0.20,
                'deviation': 0.15,
                'consistency': 0.15,
                'uptime': 0.20
            }
        }
    
    def test_single_source_selection(self):
        """Test single source selection when only one candidate."""
        engine = RoutingPolicyEngine(self.get_test_policy_config())
        
        candidate = SourceValue(
            source_id='source_a',
            value=100.0,
            trust_score=0.9,
            freshness_lag_sec=60,
            response_latency_ms=50
        )
        
        decision = engine.route(
            metric_key='test_metric',
            candidate_values=[candidate]
        )
        
        assert decision.strategy == RoutingStrategy.SINGLE
        assert decision.value == 100.0
        assert decision.confidence == 0.9
        assert len(decision.sources) == 1
    
    def test_blend_low_dispersion(self):
        """Test blending when dispersion is low."""
        engine = RoutingPolicyEngine(self.get_test_policy_config())
        
        candidates = [
            SourceValue('source_a', 100.0, 0.9, 60, 50),
            SourceValue('source_b', 101.0, 0.8, 70, 60),
            SourceValue('source_c', 100.5, 0.85, 65, 55)
        ]
        
        decision = engine.route(
            metric_key='test_metric',
            candidate_values=candidates
        )
        
        assert decision.strategy == RoutingStrategy.BLEND
        # Weighted average should be close to inputs
        assert 99.0 <= decision.value <= 102.0
        assert len(decision.sources) == 3
    
    def test_outlier_removal(self):
        """Test outlier removal with high dispersion."""
        engine = RoutingPolicyEngine(self.get_test_policy_config())
        
        candidates = [
            SourceValue('source_a', 100.0, 0.9, 60, 50),
            SourceValue('source_b', 101.0, 0.8, 70, 60),
            SourceValue('source_c', 200.0, 0.85, 65, 55)  # Outlier
        ]
        
        decision = engine.route(
            metric_key='test_metric',
            candidate_values=candidates
        )
        
        # Should blend after removing outlier
        assert decision.strategy == RoutingStrategy.BLEND
        # Value should not be heavily influenced by outlier
        assert 99.0 <= decision.value <= 102.0
    
    def test_synthetic_fallback_no_good_sources(self):
        """Test synthetic fallback when no sources meet criteria."""
        engine = RoutingPolicyEngine(self.get_test_policy_config())
        
        # All sources have low trust scores
        candidates = [
            SourceValue('source_a', 100.0, 0.3, 60, 50),
            SourceValue('source_b', 101.0, 0.2, 70, 60)
        ]
        
        decision = engine.route(
            metric_key='test_metric',
            candidate_values=candidates
        )
        
        assert decision.strategy == RoutingStrategy.SYNTHETIC
        assert decision.is_synthetic
        # Should use average of available sources
        assert 99.0 <= decision.value <= 102.0
    
    def test_synthetic_fallback_with_previous(self):
        """Test synthetic fallback uses previous good value."""
        engine = RoutingPolicyEngine(self.get_test_policy_config())
        
        # Previous good decision
        from engine import RoutingDecision
        previous = RoutingDecision(
            decision_id='prev_id',
            metric_key='test_metric',
            value=95.0,
            strategy=RoutingStrategy.SINGLE,
            confidence=0.9,
            sources=[],
            rationale_hash='hash',
            policy_version='test_v1',
            is_synthetic=False
        )
        
        # No good sources now
        candidates = [
            SourceValue('source_a', 100.0, 0.3, 60, 50)
        ]
        
        decision = engine.route(
            metric_key='test_metric',
            candidate_values=candidates,
            previous_decision=previous
        )
        
        assert decision.strategy == RoutingStrategy.SYNTHETIC
        assert decision.value == 95.0  # Uses previous value
        assert decision.confidence < 0.9  # Decayed confidence
    
    def test_filter_stale_sources(self):
        """Test filtering of stale sources."""
        engine = RoutingPolicyEngine(self.get_test_policy_config())
        
        candidates = [
            SourceValue('source_a', 100.0, 0.9, 600, 50),  # Too stale
            SourceValue('source_b', 101.0, 0.8, 70, 60)    # Fresh
        ]
        
        decision = engine.route(
            metric_key='test_metric',
            candidate_values=candidates
        )
        
        # Should only use fresh source
        assert decision.strategy == RoutingStrategy.SINGLE
        assert len(decision.sources) == 1
        assert decision.sources[0]['source_id'] == 'source_b'


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_initial_state_closed(self):
        """Test circuit breaker starts in closed state."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_sec=10)
        
        assert cb.is_available('source_a')
    
    def test_open_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_sec=10)
        
        cb.record_failure('source_a')
        assert cb.is_available('source_a')
        
        cb.record_failure('source_a')
        assert cb.is_available('source_a')
        
        cb.record_failure('source_a')
        assert not cb.is_available('source_a')  # Now open
    
    def test_reset_on_success(self):
        """Test circuit breaker resets failure count on success."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_sec=10)
        
        cb.record_failure('source_a')
        cb.record_failure('source_a')
        cb.record_success('source_a')  # Reset
        
        # Should still be available after reset
        assert cb.is_available('source_a')
        
        # Need 3 more failures to open
        cb.record_failure('source_a')
        assert cb.is_available('source_a')
    
    def test_half_open_after_cooldown(self):
        """Test circuit breaker transitions to half-open after cooldown."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_sec=1)
        
        cb.record_failure('source_a')
        cb.record_failure('source_a')
        
        # Should be open
        assert not cb.is_available('source_a')
        
        # Wait for cooldown
        import time
        time.sleep(1.5)
        
        # Should transition to half-open
        assert cb.is_available('source_a')


class TestMADOutlierRemoval:
    """Test Median Absolute Deviation outlier removal."""
    
    def test_no_outliers(self):
        """Test MAD with no outliers."""
        engine = RoutingPolicyEngine({
            'policy_version': 'test_v1',
            'min_trust': 0.55,
            'max_fresh_lag_sec': 180,
            'stable_dispersion': 0.012,
            'switch_margin': 0.07,
            'mad_k_factor': 3.0,
            'weights_json': {'freshness': 0.3, 'error_rate': 0.2, 
                           'deviation': 0.15, 'consistency': 0.15, 'uptime': 0.2}
        })
        
        sources = [
            SourceValue('a', 100.0, 0.9, 60, 50),
            SourceValue('b', 101.0, 0.9, 60, 50),
            SourceValue('c', 100.5, 0.9, 60, 50)
        ]
        
        filtered = engine._remove_outliers(sources)
        
        # All should remain
        assert len(filtered) == 3
    
    def test_remove_outlier(self):
        """Test MAD removes clear outlier."""
        engine = RoutingPolicyEngine({
            'policy_version': 'test_v1',
            'min_trust': 0.55,
            'max_fresh_lag_sec': 180,
            'stable_dispersion': 0.012,
            'switch_margin': 0.07,
            'mad_k_factor': 3.0,
            'weights_json': {'freshness': 0.3, 'error_rate': 0.2,
                           'deviation': 0.15, 'consistency': 0.15, 'uptime': 0.2}
        })
        
        sources = [
            SourceValue('a', 100.0, 0.9, 60, 50),
            SourceValue('b', 101.0, 0.9, 60, 50),
            SourceValue('c', 500.0, 0.9, 60, 50)  # Clear outlier
        ]
        
        filtered = engine._remove_outliers(sources)
        
        # Outlier should be removed
        assert len(filtered) == 2
        assert all(s.value < 200 for s in filtered)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
