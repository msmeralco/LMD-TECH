"""Example usage of the logger module."""

from machine_learning.utils.logger import (
    get_logger,
    log_execution_time,
    LogContext,
    configure_root_logger
)
import time

# Configure root logger
configure_root_logger(log_level='INFO')

# Get module logger
logger = get_logger('example_usage')

# Basic logging
logger.info('Starting ML pipeline')

# Logging with context
with LogContext(model='IsolationForest', contamination=0.1):
    logger.info('Training model')
    
    # Log execution time
    with log_execution_time(logger, 'model_training'):
        time.sleep(0.05)  # Simulate training
    
    logger.info('Model trained successfully')

# Multiple context levels
with LogContext(phase='evaluation'):
    logger.info('Starting evaluation')
    
    with LogContext(metric='accuracy'):
        logger.info('Calculating metrics')

logger.info('Pipeline complete')

print("\n✓ Example execution complete!")
print("Check logs/ml_pipeline.log for output")
