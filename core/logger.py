import logging
import pathlib
import sys


def get_logger(name, results_dir=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if results_dir:
        results_dir = pathlib.Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(results_dir / "experiment.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
