"""
Multi-Source Redundancy Router - Core Engine

Continuously choose, blend, or fail over between multiple upstream data sources
to maximize availability, correctness, continuity, and auditability.
"""
import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from uuid import uuid4
import statistics

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """Routing strategy types."""
    SINGLE = "single"
    BLEND = "blend"
    FALLBACK = "fallback"
    SYNTHETIC = "synthetic"


class SourceRole(str, Enum):
    """Source fallback roles."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SYNTHETIC = "synthetic"


@dataclass
class SourceValue:
    """Single source value with metadata."""
    source_id: str
    value: float
    trust_score: float
    freshness_lag_sec: int
    response_latency_ms: int
    variance_est: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class RoutingDecision:
    """Routing decision result."""
    decision_id: str
    metric_key: str
    value: float
    strategy: RoutingStrategy
    confidence: float
    sources: List[Dict[str, Any]]
    rationale_hash: str
    policy_version: str
    is_synthetic: bool = False
    previous_decision_id: Optional[str] = None


class TrustScoringEngine:
    """
    Computes trust scores for data sources based on health metrics.
    
    Trust score is a composite of:
    - Freshness: How recent is the data relative to SLA
    - Error rate: Reliability of responses
    - Deviation: Consistency with peer consensus
    - Consistency: Low variance over time
    - Uptime: Historical availability
    """
    
    def __init__(self, weights: Dict[str, float]):
        """
        Initialize trust scoring engine.
        
        Args:
            weights: Weight coefficients for each component
                     (freshness, error_rate, deviation, consistency, uptime)
        """
        self.weights = weights
        
    def compute_trust_score(
        self,
        freshness_lag_sec: int,
        sla_freshness_sec: int,
        error_rate_win: float,
        deviation_from_blend: float,
        consistency_ratio: float,
        uptime_roll: float,
        err_ceiling: float = 0.1,
        dev_ceiling: float = 0.2
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute trust score and component breakdown.
        
        Args:
            freshness_lag_sec: Seconds since last update
            sla_freshness_sec: Maximum acceptable freshness lag
            error_rate_win: Error rate in recent window (0-1)
            deviation_from_blend: Absolute deviation from consensus
            consistency_ratio: Consistency metric (0-1, higher is better)
            uptime_roll: Rolling uptime ratio (0-1)
            err_ceiling: Max error rate for normalization
            dev_ceiling: Max deviation for normalization
            
        Returns:
            Tuple of (trust_score, components_dict)
        """
        # Freshness component: penalize stale data
        freshness_component = max(0.0, 1.0 - (freshness_lag_sec / sla_freshness_sec))
        
        # Error component: penalize high error rates
        error_rate_norm = min(1.0, error_rate_win / err_ceiling)
        error_component = 1.0 - error_rate_norm
        
        # Deviation component: penalize outliers
        deviation_norm = min(1.0, abs(deviation_from_blend) / dev_ceiling)
        deviation_component = 1.0 - deviation_norm
        
        # Consistency component (already normalized 0-1)
        consistency_component = consistency_ratio
        
        # Uptime component (already normalized 0-1)
        uptime_component = uptime_roll
        
        # Weighted composite
        trust_score = (
            self.weights.get('freshness', 0.30) * freshness_component +
            self.weights.get('error_rate', 0.20) * error_component +
            self.weights.get('deviation', 0.15) * deviation_component +
            self.weights.get('consistency', 0.15) * consistency_component +
            self.weights.get('uptime', 0.20) * uptime_component
        )
        
        # Clamp to [0, 1]
        trust_score = max(0.0, min(1.0, trust_score))
        
        components = {
            'freshness': freshness_component,
            'error_rate': error_component,
            'deviation': deviation_component,
            'consistency': consistency_component,
            'uptime': uptime_component
        }
        
        return trust_score, components


class RoutingPolicyEngine:
    """
    Core routing decision engine.
    
    Implements the routing algorithm:
    1. Filter sources by trust and freshness
    2. Single source if only one candidate
    3. Blend if dispersion is low
    4. Remove outliers and retry if dispersion is high
    5. Synthetic fallback if no good candidates
    """
    
    def __init__(self, policy_config: Dict[str, Any]):
        """
        Initialize routing policy engine.
        
        Args:
            policy_config: Policy configuration including thresholds and weights
        """
        self.min_trust = policy_config.get('min_trust', 0.55)
        self.max_fresh_lag_sec = policy_config.get('max_fresh_lag_sec', 180)
        self.stable_dispersion = policy_config.get('stable_dispersion', 0.012)
        self.switch_margin = policy_config.get('switch_margin', 0.07)
        self.mad_k_factor = policy_config.get('mad_k_factor', 3.0)
        self.policy_version = policy_config.get('policy_version', 'v1')
        
        weights = policy_config.get('weights_json', {})
        if isinstance(weights, str):
            weights = json.loads(weights)
        self.scoring_engine = TrustScoringEngine(weights)
        
    def route(
        self,
        metric_key: str,
        candidate_values: List[SourceValue],
        previous_decision: Optional[RoutingDecision] = None
    ) -> RoutingDecision:
        """
        Make routing decision for given metric and candidate sources.
        
        Args:
            metric_key: Metric identifier
            candidate_values: List of candidate source values
            previous_decision: Previous decision for hysteresis
            
        Returns:
            RoutingDecision with selected value and metadata
        """
        # Filter active candidates
        active = [
            c for c in candidate_values
            if c.trust_score >= self.min_trust
            and c.freshness_lag_sec <= self.max_fresh_lag_sec
        ]
        
        if not active:
            return self._synthetic_fallback(metric_key, candidate_values, previous_decision)
        
        if len(active) == 1:
            return self._single_source(metric_key, active[0], previous_decision)
        
        # Check dispersion
        values = [c.value for c in active]
        median_val = statistics.median(values)
        if median_val == 0:
            median_val = 1e-9  # Avoid division by zero
        
        dispersion = (max(values) - min(values)) / abs(median_val)
        
        if dispersion <= self.stable_dispersion:
            return self._blend_sources(metric_key, active, previous_decision)
        
        # High dispersion: remove outliers
        filtered = self._remove_outliers(active)
        
        if not filtered:
            return self._synthetic_fallback(metric_key, active, previous_decision)
        
        if len(filtered) == 1:
            # Single source after outlier removal, reduce confidence
            decision = self._single_source(metric_key, filtered[0], previous_decision)
            decision.confidence *= 0.9
            return decision
        
        # Blend remaining after outlier removal
        return self._blend_sources(metric_key, filtered, previous_decision, adjusted=True)
    
    def _single_source(
        self,
        metric_key: str,
        source: SourceValue,
        previous_decision: Optional[RoutingDecision]
    ) -> RoutingDecision:
        """Select single source."""
        decision_id = str(uuid4())
        rationale = {
            'policy_version': self.policy_version,
            'strategy': 'single',
            'source_id': source.source_id,
            'trust_score': source.trust_score
        }
        
        return RoutingDecision(
            decision_id=decision_id,
            metric_key=metric_key,
            value=source.value,
            strategy=RoutingStrategy.SINGLE,
            confidence=source.trust_score,
            sources=[{
                'source_id': source.source_id,
                'value': source.value,
                'trust_score': source.trust_score,
                'weight': 1.0,
                'freshness_lag_sec': source.freshness_lag_sec
            }],
            rationale_hash=self._hash_rationale(rationale),
            policy_version=self.policy_version,
            previous_decision_id=previous_decision.decision_id if previous_decision else None
        )
    
    def _blend_sources(
        self,
        metric_key: str,
        sources: List[SourceValue],
        previous_decision: Optional[RoutingDecision],
        adjusted: bool = False
    ) -> RoutingDecision:
        """Blend multiple sources with weighted average."""
        # Compute weights (normalized trust scores)
        total_trust = sum(s.trust_score for s in sources)
        weights = [s.trust_score / total_trust for s in sources]
        
        # Weighted average
        blended_value = sum(s.value * w for s, w in zip(sources, weights))
        
        # Confidence based on dispersion
        values = [s.value for s in sources]
        value_range = max(values) - min(values)
        median_val = statistics.median(values)
        dispersion = value_range / abs(median_val) if median_val != 0 else 0
        
        # Lower confidence for higher dispersion
        confidence = 1.0 - min(dispersion, 0.5)
        if adjusted:
            confidence *= 0.95  # Slightly lower confidence if we removed outliers
        
        decision_id = str(uuid4())
        rationale = {
            'policy_version': self.policy_version,
            'strategy': 'blend',
            'num_sources': len(sources),
            'dispersion': dispersion,
            'adjusted': adjusted
        }
        
        return RoutingDecision(
            decision_id=decision_id,
            metric_key=metric_key,
            value=blended_value,
            strategy=RoutingStrategy.BLEND,
            confidence=confidence,
            sources=[{
                'source_id': s.source_id,
                'value': s.value,
                'trust_score': s.trust_score,
                'weight': w,
                'freshness_lag_sec': s.freshness_lag_sec
            } for s, w in zip(sources, weights)],
            rationale_hash=self._hash_rationale(rationale),
            policy_version=self.policy_version,
            previous_decision_id=previous_decision.decision_id if previous_decision else None
        )
    
    def _remove_outliers(self, sources: List[SourceValue]) -> List[SourceValue]:
        """Remove statistical outliers using Median Absolute Deviation (MAD)."""
        if len(sources) < 3:
            return sources
        
        values = [s.value for s in sources]
        value_median = statistics.median(values)
        
        # Compute MAD
        absolute_deviations = [abs(v - value_median) for v in values]
        mad = statistics.median(absolute_deviations)
        
        if mad == 0:
            return sources  # No outliers if no deviation
        
        # MAD threshold (1.4826 is consistency constant for normal distribution)
        threshold = self.mad_k_factor * 1.4826 * mad
        
        # Filter outliers
        filtered = [
            s for s in sources
            if abs(s.value - value_median) <= threshold
        ]
        
        return filtered if filtered else sources
    
    def _synthetic_fallback(
        self,
        metric_key: str,
        candidate_values: List[SourceValue],
        previous_decision: Optional[RoutingDecision]
    ) -> RoutingDecision:
        """Generate synthetic fallback when no good sources available."""
        # Simple fallback: use last known good value with decay
        if previous_decision and not previous_decision.is_synthetic:
            synthetic_value = previous_decision.value
            # Apply decay to confidence
            confidence = previous_decision.confidence * 0.85
        else:
            # No previous decision, use average of available sources or zero
            if candidate_values:
                synthetic_value = statistics.mean([c.value for c in candidate_values])
                confidence = 0.3
            else:
                synthetic_value = 0.0
                confidence = 0.1
        
        decision_id = str(uuid4())
        rationale = {
            'policy_version': self.policy_version,
            'strategy': 'synthetic',
            'reason': 'no_good_sources',
            'num_candidates': len(candidate_values)
        }
        
        return RoutingDecision(
            decision_id=decision_id,
            metric_key=metric_key,
            value=synthetic_value,
            strategy=RoutingStrategy.SYNTHETIC,
            confidence=confidence,
            sources=[{
                'source_id': 'synthetic',
                'value': synthetic_value,
                'trust_score': confidence,
                'weight': 1.0,
                'freshness_lag_sec': 0
            }],
            rationale_hash=self._hash_rationale(rationale),
            policy_version=self.policy_version,
            is_synthetic=True,
            previous_decision_id=previous_decision.decision_id if previous_decision else None
        )
    
    def _hash_rationale(self, rationale: Dict[str, Any]) -> str:
        """Generate hash of rationale for reproducibility."""
        rationale_str = json.dumps(rationale, sort_keys=True)
        return hashlib.sha256(rationale_str.encode()).hexdigest()[:16]


class CircuitBreaker:
    """
    Circuit breaker pattern for source availability management.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Source disabled due to failures
    - HALF_OPEN: Testing if source has recovered
    """
    
    def __init__(self, failure_threshold: int = 5, cooldown_sec: int = 300):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of consecutive failures before opening
            cooldown_sec: Cooldown period before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec
        self.state: Dict[str, Dict[str, Any]] = {}
    
    def record_success(self, source_id: str):
        """Record successful source query."""
        if source_id in self.state:
            self.state[source_id]['failure_count'] = 0
            self.state[source_id]['state'] = 'closed'
    
    def record_failure(self, source_id: str):
        """Record failed source query."""
        if source_id not in self.state:
            self.state[source_id] = {
                'failure_count': 0,
                'state': 'closed',
                'last_failure': None,
                'cooldown_until': None
            }
        
        self.state[source_id]['failure_count'] += 1
        self.state[source_id]['last_failure'] = datetime.utcnow()
        
        if self.state[source_id]['failure_count'] >= self.failure_threshold:
            self.state[source_id]['state'] = 'open'
            cooldown_until = datetime.utcnow() + timedelta(seconds=self.cooldown_sec)
            self.state[source_id]['cooldown_until'] = cooldown_until
            logger.warning(
                f"Circuit breaker OPEN for source {source_id}. "
                f"Cooldown until {cooldown_until}"
            )
    
    def is_available(self, source_id: str) -> bool:
        """Check if source is available (circuit not open)."""
        if source_id not in self.state:
            return True
        
        state = self.state[source_id]
        if state['state'] == 'closed':
            return True
        
        if state['state'] == 'open':
            # Check if cooldown expired
            if state['cooldown_until'] and datetime.utcnow() > state['cooldown_until']:
                state['state'] = 'half_open'
                logger.info(f"Circuit breaker HALF_OPEN for source {source_id}")
                return True
            return False
        
        # Half-open: allow attempt
        return True
