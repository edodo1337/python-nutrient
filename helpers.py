import numpy as np
import yaml
from consts import BATCH_SIZE

from dto import MealItem


def load_meals_data(filename: str) -> list[MealItem]:
    with open(filename) as file:
        nutrients_data = yaml.load(file, Loader=yaml.SafeLoader).get("meals", [])
        all_meals = [
            MealItem(name=kwargs["name"], nutrients=np.array(kwargs["nutrients"]))
            for kwargs in nutrients_data
        ]

    return all_meals


def batched(iterable: list, size=BATCH_SIZE):
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]
