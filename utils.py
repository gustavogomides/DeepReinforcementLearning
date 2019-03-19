import logging

import numpy as np


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(filename)s %(lineno)d %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def get_init_positions(positions):
    return np.array([0] * positions, dtype=np.int)
