"""
Metrics collection using Prometheus.

This module provides utilities for collecting and exposing metrics
for monitoring and observability.
"""

from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
from functools import wraps
import time
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Create a custom registry (can be replaced with default registry)
registry = CollectorRegistry()


# Node execution metrics
node_execution_count = Counter(
    'node_executions_total',
    'Total number of node executions',
    ['node_name', 'status'],
    registry=registry
)

node_execution_duration = Histogram(
    'node_execution_duration_seconds',
    'Time spent executing nodes',
    ['node_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=registry
)


# Planning metrics
plan_generation_time = Histogram(
    'plan_generation_duration_seconds',
    'Time to generate plans',
    buckets=[1.0, 5.0, 10.0, 20.0, 30.0, 60.0],
    registry=registry
)

plan_confidence = Histogram(
    'plan_confidence_score',
    'Confidence scores of generated plans',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=registry
)

plan_node_count = Histogram(
    'plan_node_count',
    'Number of nodes in plans',
    buckets=[1, 2, 3, 5, 10, 15, 20, 30, 50],
    registry=registry
)


# LLM metrics
llm_call_count = Counter(
    'llm_calls_total',
    'Total number of LLM API calls',
    ['status'],
    registry=registry
)

llm_call_duration = Histogram(
    'llm_call_duration_seconds',
    'Duration of LLM API calls',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    registry=registry
)

llm_token_usage = Counter(
    'llm_tokens_total',
    'Total tokens used in LLM calls',
    ['type'],  # 'input' or 'output'
    registry=registry
)


# Validation metrics
validation_errors = Counter(
    'validation_errors_total',
    'Total validation errors',
    ['validation_type'],
    registry=registry
)

plan_validation_duration = Histogram(
    'plan_validation_duration_seconds',
    'Time spent validating plans',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
    registry=registry
)


# System metrics
active_sessions = Gauge(
    'active_sessions',
    'Number of active planning sessions',
    registry=registry
)

error_count = Counter(
    'errors_total',
    'Total errors by type',
    ['error_type'],
    registry=registry
)


class MetricsCollector:
    """
    Central metrics collector with convenience methods.
    
    Example:
        >>> metrics = MetricsCollector()
        >>> metrics.record_node_execution("plan_tot", duration=2.5, status="success")
    """
    
    def __init__(self, custom_registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics collector.
        
        Args:
            custom_registry: Optional custom registry (uses default if None)
        """
        self.registry = custom_registry or registry
    
    def record_node_execution(self, node_name: str, duration: float, status: str = "success"):
        """
        Record node execution metrics.
        
        Args:
            node_name: Name of the node
            duration: Execution duration in seconds
            status: Execution status ('success' or 'error')
        """
        node_execution_count.labels(node_name=node_name, status=status).inc()
        node_execution_duration.labels(node_name=node_name).observe(duration)
        
        logger.debug("metrics_recorded",
                    metric="node_execution",
                    node=node_name,
                    duration=duration,
                    status=status)
    
    def record_plan_generation(self, duration: float, confidence: float, node_count: int):
        """
        Record plan generation metrics.
        
        Args:
            duration: Generation duration in seconds
            confidence: Plan confidence score
            node_count: Number of nodes in plan
        """
        plan_generation_time.observe(duration)
        plan_confidence.observe(confidence)
        plan_node_count.observe(node_count)
        
        logger.debug("metrics_recorded",
                    metric="plan_generation",
                    duration=duration,
                    confidence=confidence,
                    nodes=node_count)
    
    def record_llm_call(self, duration: float, status: str = "success", 
                       input_tokens: int = 0, output_tokens: int = 0):
        """
        Record LLM call metrics.
        
        Args:
            duration: Call duration in seconds
            status: Call status
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        llm_call_count.labels(status=status).inc()
        llm_call_duration.observe(duration)
        
        if input_tokens > 0:
            llm_token_usage.labels(type="input").inc(input_tokens)
        if output_tokens > 0:
            llm_token_usage.labels(type="output").inc(output_tokens)
        
        logger.debug("metrics_recorded",
                    metric="llm_call",
                    duration=duration,
                    status=status,
                    tokens=input_tokens + output_tokens)
    
    def record_validation_error(self, validation_type: str):
        """
        Record validation error.
        
        Args:
            validation_type: Type of validation that failed
        """
        validation_errors.labels(validation_type=validation_type).inc()
    
    def record_error(self, error_type: str):
        """
        Record error occurrence.
        
        Args:
            error_type: Type of error
        """
        error_count.labels(error_type=error_type).inc()
    
    def increment_active_sessions(self, delta: int = 1):
        """
        Increment active sessions counter.
        
        Args:
            delta: Amount to increment (can be negative)
        """
        if delta > 0:
            active_sessions.inc(delta)
        elif delta < 0:
            active_sessions.dec(-delta)
    
    def get_metrics(self) -> bytes:
        """
        Get metrics in Prometheus format.
        
        Returns:
            Metrics as bytes
        """
        return generate_latest(self.registry)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance.
    
    Returns:
        Global MetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def track_execution_time(node_name: str):
    """
    Decorator to track node execution time.
    
    Args:
        node_name: Name of the node
        
    Returns:
        Decorator function
        
    Example:
        >>> @track_execution_time("plan_tot")
        >>> def plan_tot(state):
        >>>     return {"plan": {...}}
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                metrics = get_metrics_collector()
                metrics.record_node_execution(node_name, duration, status)
        
        return wrapper
    return decorator


def track_llm_call(func):
    """
    Decorator to track LLM API calls.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
        
    Example:
        >>> @track_llm_call
        >>> def call_llm(messages):
        >>>     return llm.invoke(messages)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        status = "success"
        
        try:
            result = func(*args, **kwargs)
            
            # Try to extract token usage if available
            input_tokens = 0
            output_tokens = 0
            
            if hasattr(result, 'response_metadata'):
                metadata = result.response_metadata
                if 'token_usage' in metadata:
                    token_usage = metadata['token_usage']
                    input_tokens = token_usage.get('prompt_tokens', 0)
                    output_tokens = token_usage.get('completion_tokens', 0)
            
            duration = time.time() - start_time
            metrics = get_metrics_collector()
            metrics.record_llm_call(duration, status, input_tokens, output_tokens)
            
            return result
            
        except Exception as e:
            status = "error"
            duration = time.time() - start_time
            metrics = get_metrics_collector()
            metrics.record_llm_call(duration, status)
            raise
    
    return wrapper
