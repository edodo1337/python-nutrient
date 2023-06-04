from consts import MAX_MEALS, MAX_WORKERS
from dto import MealItem
import numpy as np
import itertools as iter
import functools as func

# from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

from helpers import batched


def calc_cos_similarity(
    missing_nutrients_vector: np.ndarray,
    meals: tuple[MealItem, ...],
):
    nutrients_vector = np.sum([meal.nutrients for meal in meals], axis=0)

    cosine_similarity = (
        np.dot(missing_nutrients_vector, nutrients_vector)
        / np.linalg.norm(missing_nutrients_vector)
        / np.linalg.norm(nutrients_vector)
    )

    return cosine_similarity


def naive_brute_force_approach(
    *,
    all_meals: list[MealItem],
    missing_nutrients: np.ndarray,
    max_meals: int = MAX_MEALS,
) -> tuple[tuple[MealItem, ...], float]:
    best_similarity = 0
    best_meals: tuple[MealItem, ...] = ()

    for meals_count in range(1, max_meals + 1):
        for meals in iter.combinations(all_meals, meals_count):
            cos_sim = calc_cos_similarity(missing_nutrients, meals)
            if (similarity := abs(cos_sim)) > best_similarity:
                print("----New best", similarity)
                best_similarity = similarity
                best_meals = meals

    return best_meals, best_similarity


def naive_brute_force_mp_approach(
    *,
    all_meals: list[MealItem],
    missing_nutrients: np.ndarray,
    max_meals: int = MAX_MEALS,
) -> tuple[tuple[MealItem, ...], float]:
    best_similarity = 0
    best_meals: tuple[MealItem, ...] = ()

    for meals_count in range(1, max_meals + 1):
        with Pool(processes=MAX_WORKERS) as pool:
            worker = func.partial(
                _worker,
                missing_nutrients=missing_nutrients,
                best_similarity=best_similarity,
            )
            results = pool.map(
                worker, batched(list(iter.combinations(all_meals, meals_count)))
            )

            for _meals, _similarity in results:
                if _similarity > best_similarity:
                    print("----New best", _similarity)
                    best_similarity = _similarity
                    best_meals = _meals

    return best_meals, best_similarity


def genetic(
    *,
    all_meals: list[MealItem],
    missing_nutrients: np.ndarray,
    max_meals: int = MAX_MEALS,
) -> tuple[tuple[MealItem, ...], float]:
    ...


def _worker(
    meal_batches: list[tuple[MealItem, ...]],
    missing_nutrients: np.ndarray,
    best_similarity: int,
):
    _best_meals: tuple[MealItem, ...] = ()
    _best_similarity = best_similarity

    for meals in meal_batches:
        cos_sim = calc_cos_similarity(missing_nutrients, meals)
        if (similarity := abs(cos_sim)) > _best_similarity:
            _best_similarity = similarity
            _best_meals = meals

    return _best_meals, _best_similarity
