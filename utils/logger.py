"""
Logger Configuration for PALMPlot
Sets up logging with file and console handlers
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(level: str = "INFO", 
                log_file: Optional[str] = None,
                console_output: bool = True) -> logging.Logger:
    """
    Set up logging configuration for PALMPlot
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('palmplot')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    # Add file handler if requested
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    # Add initial log message
    logger.info("=" * 60)
    logger.info(f"PALMPlot logging initialized at {datetime.now()}")
    logger.info(f"Log level: {level}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)
    
    return logger


class LoggerContext:
    """Context manager for temporary logging configuration changes"""
    
    def __init__(self, logger_name: str = 'palmplot', 
                 level: Optional[str] = None):
        """
        Initialize logger context
        
        Args:
            logger_name: Name of logger to modify
            level: Temporary logging level
        """
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
        self.new_level = getattr(logging, level.upper()) if level else None
        
    def __enter__(self):
        """Enter context - change logging level"""
        if self.new_level:
            self.logger.setLevel(self.new_level)
        return self.logger
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore original logging level"""
        self.logger.setLevel(self.original_level)
        

def log_execution_time(func):
    """Decorator to log function execution time"""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('palmplot')
        start_time = time.time()
        
        logger.debug(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {elapsed_time:.2f} seconds")
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed_time:.2f} seconds: {str(e)}")
            raise
            
    return wrapper
    

def log_memory_usage(func):
    """Decorator to log memory usage of function"""
    import functools
    import psutil
    import os
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('palmplot')
        process = psutil.Process(os.getpid())
        
        # Get memory before execution
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        # Get memory after execution
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_diff = mem_after - mem_before
        
        logger.debug(f"{func.__name__} memory usage: {mem_diff:.2f} MB "
                    f"(before: {mem_before:.2f} MB, after: {mem_after:.2f} MB)")
        
        return result
        
    return wrapper