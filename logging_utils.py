import logging

def setup_logging(level=logging.INFO):
    """
    Set up logging configuration for the causal analysis pipeline.
    """
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

def get_logger(name):
    """
    Get a logger by name.
    """
    return logging.getLogger(name) 