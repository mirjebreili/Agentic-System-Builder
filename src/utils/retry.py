"""
Retry utilities with exponential backoff for LLM calls.

This module provides decorators and utilities for retrying operations
with configurable backoff strategies, particularly useful for LLM API calls.
"""

from typing import Callable, Optional, Type, Tuple
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
import logging

logger = logging.getLogger(__name__)


# Common exceptions to retry
RETRIABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
)


def create_llm_retry_decorator(
    max_attempts: int = 3,
    min_wait: int = 2,
    max_wait: int = 10,
    multiplier: int = 2,
    exception_types: Optional[Tuple[Type[Exception], ...]] = None,
):
    """
    Create a retry decorator with exponential backoff for LLM calls.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds
        multiplier: Multiplier for exponential backoff
        exception_types: Tuple of exception types to retry on
        
    Returns:
        Retry decorator
        
    Example:
        >>> @create_llm_retry_decorator(max_attempts=3)
        >>> def call_llm(prompt):
        >>>     return llm.invoke(prompt)
    """
    if exception_types is None:
        exception_types = RETRIABLE_EXCEPTIONS
    
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exception_types),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.DEBUG),
        reraise=True
    )


# Default retry decorator for LLM calls
retry_llm_call = create_llm_retry_decorator(
    max_attempts=3,
    min_wait=2,
    max_wait=10,
    multiplier=2
)


def invoke_llm_with_retry(llm, messages, **kwargs):
    """
    Invoke an LLM with automatic retry on failure.
    
    Args:
        llm: LLM instance to invoke
        messages: Messages to send to LLM
        **kwargs: Additional arguments to pass to invoke
        
    Returns:
        LLM response
        
    Example:
        >>> response = invoke_llm_with_retry(llm, [HumanMessage(content="Hello")])
    """
    @retry_llm_call
    def _invoke():
        return llm.invoke(messages, **kwargs)
    
    try:
        return _invoke()
    except Exception as e:
        logger.error(f"LLM invocation failed after retries: {e}")
        raise


def invoke_chain_with_retry(chain, input_data, **kwargs):
    """
    Invoke a LangChain chain with automatic retry on failure.
    
    Args:
        chain: Chain instance to invoke
        input_data: Input data for the chain
        **kwargs: Additional arguments to pass to invoke
        
    Returns:
        Chain response
        
    Example:
        >>> response = invoke_chain_with_retry(chain, {"input": "Hello"})
    """
    @retry_llm_call
    def _invoke():
        return chain.invoke(input_data, **kwargs)
    
    try:
        return _invoke()
    except Exception as e:
        logger.error(f"Chain invocation failed after retries: {e}")
        raise


class RetryableLLM:
    """
    Wrapper for LLM that automatically retries on failure.
    
    Example:
        >>> llm = RetryableLLM(base_llm)
        >>> response = llm.invoke([HumanMessage(content="Hello")])
    """
    
    def __init__(
        self,
        llm,
        max_attempts: int = 3,
        min_wait: int = 2,
        max_wait: int = 10,
        multiplier: int = 2
    ):
        """
        Initialize RetryableLLM.
        
        Args:
            llm: Base LLM instance
            max_attempts: Maximum retry attempts
            min_wait: Minimum wait time between retries
            max_wait: Maximum wait time between retries
            multiplier: Backoff multiplier
        """
        self.llm = llm
        self.retry_decorator = create_llm_retry_decorator(
            max_attempts=max_attempts,
            min_wait=min_wait,
            max_wait=max_wait,
            multiplier=multiplier
        )
    
    def invoke(self, messages, **kwargs):
        """
        Invoke LLM with retry logic.
        
        Args:
            messages: Messages to send
            **kwargs: Additional invoke arguments
            
        Returns:
            LLM response
        """
        @self.retry_decorator
        def _invoke():
            return self.llm.invoke(messages, **kwargs)
        
        return _invoke()
    
    def __getattr__(self, name):
        """Delegate attribute access to base LLM."""
        return getattr(self.llm, name)


# Retry decorator for general operations (not just LLM)
def retry_operation(
    max_attempts: int = 3,
    min_wait: int = 1,
    max_wait: int = 5,
    exception_types: Optional[Tuple[Type[Exception], ...]] = None
):
    """
    Decorator for retrying general operations.
    
    Args:
        max_attempts: Maximum retry attempts
        min_wait: Minimum wait time
        max_wait: Maximum wait time
        exception_types: Exceptions to retry on
        
    Returns:
        Retry decorator
        
    Example:
        >>> @retry_operation(max_attempts=3)
        >>> def fetch_data():
        >>>     return requests.get(url)
    """
    if exception_types is None:
        exception_types = (Exception,)
    
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exception_types),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
