"""
Timeout handling utilities for long-running operations.

This module provides context managers and decorators for adding timeouts
to operations that might hang or take too long.
"""

import signal
import time
from contextlib import contextmanager
from typing import Callable, Optional
from functools import wraps
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass


@contextmanager
def timeout(seconds: int):
    """
    Context manager for adding timeout to code blocks (Unix only).
    
    Args:
        seconds: Timeout in seconds
        
    Yields:
        None
        
    Raises:
        TimeoutError: If operation exceeds timeout
        
    Example:
        >>> with timeout(10):
        >>>     long_running_operation()
    
    Note:
        This uses SIGALRM which only works on Unix systems.
        For cross-platform support, use timeout_decorator instead.
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def timeout_decorator(seconds: int, fallback_result=None):
    """
    Decorator for adding timeout to functions (Unix only).
    
    Args:
        seconds: Timeout in seconds
        fallback_result: Value to return on timeout
        
    Returns:
        Decorated function
        
    Example:
        >>> @timeout_decorator(30, fallback_result={})
        >>> def slow_function():
        >>>     return expensive_computation()
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with timeout(seconds):
                    return func(*args, **kwargs)
            except TimeoutError as e:
                logger.error("function_timeout", 
                           function=func.__name__, 
                           timeout=seconds,
                           error=str(e))
                
                if fallback_result is not None:
                    logger.info("returning_fallback_result", function=func.__name__)
                    return fallback_result
                raise
        
        return wrapper
    return decorator


class ThreadTimeout:
    """
    Cross-platform timeout implementation using threading.
    
    This works on all platforms including Windows.
    
    Example:
        >>> def long_task():
        >>>     time.sleep(100)
        >>> 
        >>> result = ThreadTimeout.run(long_task, timeout=5, fallback="timeout")
    """
    
    @staticmethod
    def run(func: Callable, timeout_seconds: int, fallback_result=None, args=(), kwargs=None):
        """
        Run a function with timeout using threading.
        
        Args:
            func: Function to run
            timeout_seconds: Timeout in seconds
            fallback_result: Value to return on timeout
            args: Positional arguments for func
            kwargs: Keyword arguments for func
            
        Returns:
            Function result or fallback_result on timeout
            
        Raises:
            TimeoutError: If timeout occurs and no fallback provided
        """
        import threading
        
        if kwargs is None:
            kwargs = {}
        
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            logger.error("thread_timeout", 
                        function=func.__name__, 
                        timeout=timeout_seconds)
            
            if fallback_result is not None:
                return fallback_result
            raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds}s")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]


@contextmanager
def time_limit(seconds: int, operation_name: str = "operation"):
    """
    Cross-platform context manager for timing operations.
    
    Logs a warning if operation takes longer than expected,
    but doesn't interrupt execution.
    
    Args:
        seconds: Expected maximum duration
        operation_name: Name for logging
        
    Yields:
        None
        
    Example:
        >>> with time_limit(5, "database_query"):
        >>>     result = db.query(...)
    """
    start_time = time.time()
    yield
    duration = time.time() - start_time
    
    if duration > seconds:
        logger.warning("operation_exceeded_time_limit",
                      operation=operation_name,
                      expected_seconds=seconds,
                      actual_seconds=round(duration, 2))
    else:
        logger.debug("operation_within_time_limit",
                    operation=operation_name,
                    duration_seconds=round(duration, 2))


def with_timeout(seconds: int, fallback_result=None):
    """
    Cross-platform timeout decorator using threading.
    
    Args:
        seconds: Timeout in seconds
        fallback_result: Value to return on timeout
        
    Returns:
        Decorated function
        
    Example:
        >>> @with_timeout(10, fallback_result={"error": "timeout"})
        >>> def api_call():
        >>>     return requests.get(url)
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return ThreadTimeout.run(
                func, 
                timeout_seconds=seconds,
                fallback_result=fallback_result,
                args=args,
                kwargs=kwargs
            )
        return wrapper
    return decorator


class ProgressTimeout:
    """
    Timeout that resets on progress updates.
    
    Useful for operations that show progress but might stall.
    
    Example:
        >>> pt = ProgressTimeout(30)  # 30 second timeout
        >>> for item in items:
        >>>     process(item)
        >>>     pt.reset()  # Reset timeout on each item
        >>>     pt.check()  # Raises TimeoutError if no progress for 30s
    """
    
    def __init__(self, timeout_seconds: int):
        """
        Initialize progress timeout.
        
        Args:
            timeout_seconds: Timeout duration in seconds
        """
        self.timeout_seconds = timeout_seconds
        self.last_progress = time.time()
    
    def reset(self):
        """Reset the timeout counter."""
        self.last_progress = time.time()
    
    def check(self):
        """
        Check if timeout has been exceeded.
        
        Raises:
            TimeoutError: If no progress for timeout_seconds
        """
        elapsed = time.time() - self.last_progress
        if elapsed > self.timeout_seconds:
            raise TimeoutError(
                f"No progress for {self.timeout_seconds} seconds (elapsed: {elapsed:.1f}s)"
            )
    
    def remaining(self) -> float:
        """
        Get remaining time before timeout.
        
        Returns:
            Remaining seconds (0 if expired)
        """
        elapsed = time.time() - self.last_progress
        remaining = self.timeout_seconds - elapsed
        return max(0, remaining)
