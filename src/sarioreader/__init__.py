import logging
import os
from importlib.metadata import PackageNotFoundError, version
from logging.handlers import RotatingFileHandler

from sarioreader.ocr import srOCR
from sarioreader.sario import SarioDetector

# Create and configure custom logger
logger = logging.getLogger("sarioreader")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevents log messages from being passed

# Add handlers to custom logger
file_handler = RotatingFileHandler(
    "sarioreader.log", maxBytes=1e6, backupCount=3
)  # 1 MB file size limit, keep last 3 logs
stream_handler = logging.StreamHandler()

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

ocr = srOCR()
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "models/sario-best.pt")

detector = SarioDetector(model_path)

try:
    dist_name = "sarioreader"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
