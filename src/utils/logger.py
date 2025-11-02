"""
Structured logging configuration using structlog.

This module provides a centralized logging configuration with structured
logging capabilities for better observability and debugging.
"""

import logging
import sys
from typing import Any, Dict, Optional
import structlog
from structlog.types import EventDict, Processor


def add_trace_id(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add trace_id to log events if available in context.
    
    Args:
        logger: Logger instance
        method_name: Method name
        event_dict: Event dictionary
        
    Returns:
        Modified event dictionary
    """
    # Trace ID will be passed in event_dict if available
    return event_dict


def add_timestamp(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add timestamp to log events.
    
    Args:
        logger: Logger instance
        method_name: Method name
        event_dict: Event dictionary
        
    Returns:
        Modified event dictionary with timestamp
    """
    from datetime import datetime
    event_dict["timestamp"] = datetime.utcnow().isoformat()
    return event_dict


def configure_logging(
    log_level: str = "INFO",
    json_logs: bool = False,
    include_trace_id: bool = True
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Whether to output logs in JSON format
        include_trace_id: Whether to include trace IDs in logs
        
    Example:
        >>> configure_logging(log_level="DEBUG", json_logs=True)
    """
    # Configure processors
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_timestamp,
    ]
    
    if include_trace_id:
        processors.append(add_trace_id)
    
    # Add stack info for exceptions
    processors.append(structlog.processors.StackInfoRenderer())
    processors.append(structlog.dev.set_exc_info)
    
    # Choose renderer based on format
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Structured logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("operation_started", user_id=123, operation="plan")
    """
    return structlog.get_logger(name)


class LoggerContext:
    """
    Context manager for adding contextual information to logs.
    
    Example:
        >>> with LoggerContext(trace_id="abc123", user_id=456):
        >>>     logger.info("processing_request")
        # Logs will include trace_id and user_id
    """
    
    def __init__(self, **kwargs):
        """
        Initialize context with key-value pairs.
        
        Args:
            **kwargs: Context variables to add to logs
        """
        self.context = kwargs
    
    def __enter__(self):
        """Enter context and bind variables."""
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and clear variables."""
        structlog.contextvars.clear_contextvars()


def log_function_call(logger: structlog.BoundLogger):
    """
    Decorator to automatically log function entry and exit.
    
    Args:
        logger: Structured logger instance
        
    Returns:
        Decorator function
        
    Example:
        >>> logger = get_logger(__name__)
        >>> @log_function_call(logger)
        >>> def process_data(data):
        >>>     return data
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(
                "function_entry",
                function=func_name,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys())
            )
            
            try:
                result = func(*args, **kwargs)
                logger.debug("function_exit", function=func_name, success=True)
                return result
            except Exception as e:
                logger.error(
                    "function_error",
                    function=func_name,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
        
        return wrapper
    return decorator


def log_node_execution(node_name: str):
    """
    Decorator specifically for logging agent node executions.
    
    Args:
        node_name: Name of the node
        
    Returns:
        Decorator function
        
    Example:
        >>> @log_node_execution("plan_tot")
        >>> def plan_tot(state):
        >>>     return {"plan": {...}}
    """
    def decorator(func):
        def wrapper(state: Dict[str, Any], *args, **kwargs):
            logger = get_logger(__name__)
            
            # Extract trace_id if available
            trace_id = state.get("trace_id", "unknown")
            
            with LoggerContext(trace_id=trace_id, node=node_name):
                logger.info(
                    "node_execution_start",
                    has_plan="plan" in state,
                    has_split_tasks="split_tasks" in state,
                    message_count=len(state.get("messages", []))
                )
                
                try:
                    result = func(state, *args, **kwargs)
                    
                    logger.info(
                        "node_execution_complete",
                        result_keys=list(result.keys()) if isinstance(result, dict) else None
                    )
                    
                    return result
                    
                except Exception as e:
                    logger.error(
                        "node_execution_error",
                        error=str(e),
                        error_type=type(e).__name__,
                        exc_info=True
                    )
                    raise
        
        return wrapper
    return decorator


# Performance tracking utilities
class PerformanceLogger:
    """
    Context manager for tracking and logging performance metrics.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> with PerformanceLogger(logger, "llm_call"):
        >>>     response = llm.invoke(messages)
    """
    
    def __init__(self, logger: structlog.BoundLogger, operation: str):
        """
        Initialize performance logger.
        
        Args:
            logger: Structured logger instance
            operation: Name of the operation being tracked
        """
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.time()
        self.logger.debug("operation_start", operation=self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log duration."""
        import time
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(
                "operation_complete",
                operation=self.operation,
                duration_seconds=round(duration, 3),
                success=True
            )
        else:
            self.logger.error(
                "operation_failed",
                operation=self.operation,
                duration_seconds=round(duration, 3),
                error=str(exc_val),
                error_type=exc_type.__name__
            )


# Initialize logging with default configuration
configure_logging()
