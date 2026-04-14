import time
import json
import functools
from typing import Callable, Any, Optional
from app.core.logger import logger
from app.core.token_tracker import token_tracker

def retry_and_log_llm_usage(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """
    Decorator for LLM function calls that:
    - Retries on non-200 status codes up to max_retries
    - Logs token usage for the function call
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = retry_delay
            
            for attempt in range(max_retries + 1):
                try:
                    # Call the original function
                    result = func(*args, **kwargs)
                    
                    # If function returns a tuple (raw_response, processed_result)
                    if isinstance(result, tuple) and len(result) == 2:
                        raw_response, processed_result = result
                        token_data = extract_token_usage(raw_response)
                        log_token_usage(func.__name__, token_data, attempt)
                        # Track token usage for current request
                        token_tracker.record_llm_call(func.__name__, token_data, attempt)
                        return processed_result
                    else:
                        # For backward compatibility - try to extract from function attribute
                        if hasattr(func, '_last_raw_response'):
                            raw_response = getattr(func, '_last_raw_response')
                            token_data = extract_token_usage(raw_response)
                            log_token_usage(func.__name__, token_data, attempt)
                            # Track token usage for current request
                            token_tracker.record_llm_call(func.__name__, token_data, attempt)
                            # Clean up
                            delattr(func, '_last_raw_response')
                        return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if it's a status code error
                    status_code = getattr(e, 'status_code', None)
                    if status_code and status_code == 200:
                        raise e
                    
                    logger.warning(
                        f"LLM call failed in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}): {str(e)}"
                    )
                    
                    if attempt >= max_retries:
                        break
                    
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
            logger.error(
                f"LLM call failed after {max_retries + 1} attempts in {func.__name__}: {str(last_exception)}"
            )
            raise last_exception
        
        return wrapper
    return decorator

def extract_token_usage(response: Any) -> dict:
    """
    Extract token usage data from Groq/LLaMA response format
    """
    token_data = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "model": "unknown",
        "queue_time": 0,
        "prompt_time": 0,
        "completion_time": 0,
        "total_time": 0
    }
    
    try:
        # Handle Groq/LLaMA response format (from your JSON example)
        if hasattr(response, 'usage'):
            usage = response.usage
            # Extract basic token counts
            if hasattr(usage, 'prompt_tokens'):
                token_data["prompt_tokens"] = getattr(usage, 'prompt_tokens', 0)
            if hasattr(usage, 'completion_tokens'):
                token_data["completion_tokens"] = getattr(usage, 'completion_tokens', 0)
            if hasattr(usage, 'total_tokens'):
                token_data["total_tokens"] = getattr(usage, 'total_tokens', 0)
            
            # Extract timing information from Groq
            if hasattr(usage, 'queue_time'):
                token_data["queue_time"] = getattr(usage, 'queue_time', 0)
            if hasattr(usage, 'prompt_time'):
                token_data["prompt_time"] = getattr(usage, 'prompt_time', 0)
            if hasattr(usage, 'completion_time'):
                token_data["completion_time"] = getattr(usage, 'completion_time', 0)
            if hasattr(usage, 'total_time'):
                token_data["total_time"] = getattr(usage, 'total_time', 0)
        
        # Handle dictionary response format
        elif isinstance(response, dict):
            usage = response.get('usage', {})
            token_data["prompt_tokens"] = usage.get('prompt_tokens', 0)
            token_data["completion_tokens"] = usage.get('completion_tokens', 0)
            token_data["total_tokens"] = usage.get('total_tokens', 0)
            token_data["queue_time"] = usage.get('queue_time', 0)
            token_data["prompt_time"] = usage.get('prompt_time', 0)
            token_data["completion_time"] = usage.get('completion_time', 0)
            token_data["total_time"] = usage.get('total_time', 0)
        
        # Extract model information
        if hasattr(response, 'model'):
            token_data["model"] = response.model
        elif isinstance(response, dict):
            token_data["model"] = response.get('model', 'unknown')
        else:
            token_data["model"] = getattr(response, 'model', 'unknown')
            
    except Exception as e:
        logger.debug(f"Could not extract token usage: {str(e)}")
    
    return token_data

def log_token_usage(func_name: str, token_data: dict, attempt: int = 0):
    """
    Log token usage in JSON format with Groq-specific metrics
    """
    log_entry = {
        "function": func_name,
        "timestamp": time.time(),
        "attempt": attempt + 1,
        "token_usage": token_data,
        "summary": f"{func_name} - Tokens: {token_data['total_tokens']} (P:{token_data['prompt_tokens']} C:{token_data['completion_tokens']}) Model: {token_data['model']}"
    }
    
    logger.info(f"LLM Token Usage - {json.dumps(log_entry, indent=None, separators=(',', ':'))}")

# Store the last raw response for functions that don't return tuples
def store_raw_response(func: Callable) -> Callable:
    """Decorator to store the raw LLM response"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # This is a simplified version - you'd need to modify your actual function
        # to call this decorator and store the response
        result = func(*args, **kwargs)
        return result
    return wrapper