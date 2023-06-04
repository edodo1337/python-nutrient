from dataclasses import dataclass
import numpy as np


@dataclass
class MealItem:
    name: str
    nutrients: np.ndarray
