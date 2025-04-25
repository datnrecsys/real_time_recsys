import papermill as pm
from loguru import logger
import os

loss ="bce"

logger.info(f"Running with bce")
output_path = f"./outputs/014-sequence-modelling-{loss}.ipynb"
pm.execute_notebook(
    os.path.abspath(f"./014-sequence-modelling-bce.ipynb"),
    os.path.abspath(output_path),
    parameters=dict(loss=loss),
)