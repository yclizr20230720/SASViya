import logging
import sys
from functools import wraps
import swat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('viya_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def handle_viya_errors(func):
    """Decorator for handling Viya-specific errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except swat.SWATError as e:
            logger.error(f"SWAT Error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    return wrapper

class ViyaErrorHandler:
    """Centralized error handling for Viya operations"""
    
    @staticmethod
    @handle_viya_errors
    def safe_cas_operation(cas_session, operation, *args, **kwargs):
        """Safely execute CAS operations with error handling"""
        try:
            result = getattr(cas_session, operation)(*args, **kwargs)
            logger.info(f"Successfully executed {operation}")
            return result
        except Exception as e:
            logger.error(f"Failed to execute {operation}: {e}")
            raise