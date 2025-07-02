from typing import List, Union

import numpy as np


def sigmoid(z: Union[float, int, np.ndarray, List]) -> Union[np.float64, np.ndarray]:
    return 1 / (1 + np.exp(-z))