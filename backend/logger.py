"""
Logging configuration for CalcBERT v2 Backend.
Provides structured logging with consistent tags.
"""

import logging
import sys
from datetime import datetime

# Create logger
logger = logging.getLogger("calcbert")
logger.setLevel(logging.INFO)

# Create console handler with formatting
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

# Create formatter with timestamp and tags
formatter = logging.Formatter(
    '%(asctime)s - [%(name)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(handler)

# Logging helper functions with tags
def log_tfidf(message: str, **kwargs):
    """Log TF-IDF related messages."""
    logger.info(f"[TFIDF] {message}", extra=kwargs)

def log_bert(message: str, **kwargs):
    """Log DistilBERT related messages."""
    logger.info(f"[BERT] {message}", extra=kwargs)

def log_fusion(message: str, **kwargs):
    """Log fusion related messages."""
    logger.info(f"[FUSION] {message}", extra=kwargs)

def log_policy(message: str, **kwargs):
    """Log policy engine related messages."""
    logger.info(f"[POLICY] {message}", extra=kwargs)

def log_prepay(message: str, **kwargs):
    """Log prepay API related messages."""
    logger.info(f"[PREPAY] {message}", extra=kwargs)

def log_workflow(message: str, **kwargs):
    """Log workflow orchestration messages."""
    logger.info(f"[WORKFLOW] {message}", extra=kwargs)

def log_vector(message: str, **kwargs):
    """Log vector store related messages."""
    logger.info(f"[VECTOR] {message}", extra=kwargs)

def log_error(message: str, exc_info=None, **kwargs):
    """Log error messages."""
    logger.error(f"[ERROR] {message}", exc_info=exc_info, extra=kwargs)

def log_warning(message: str, **kwargs):
    """Log warning messages."""
    logger.warning(f"[WARNING] {message}", extra=kwargs)
