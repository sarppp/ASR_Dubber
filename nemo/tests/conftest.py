"""nemo tests never start or reach a real warm aligner service."""
import os

os.environ["ALIGN_SERVICE"] = "off"
