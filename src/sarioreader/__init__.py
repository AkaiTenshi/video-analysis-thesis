import logging
import os
import sys

from sarioreader.ocr import srOCR
from sarioreader.sario import SarioDetector

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

# Create and configure your custom logger
from logging.handlers import RotatingFileHandler

# Create and configure your custom logger
logger = logging.getLogger("sarioreader")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevents log messages from being passed to the root logger

# Add handlers to your custom logger
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
    # Change here if project is renamed and does not equal the package name
    dist_name = "sarioreader"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
