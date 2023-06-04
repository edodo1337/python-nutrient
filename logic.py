from consts import (
    EPOCH_THREESOLD,
    MAX_MEALS,
    MAX_WORKERS,
    EPOCHS,
    INIT_POPULATION_BATCH_SIZE,
    MAX_CHILDREN,
    ADD_CHANCE,
    REMOVE_CHANCE,
)
from dto import MealItem
import numpy as np
import itertools as iter
import functools as func
import heapq
import random

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
    def _select_best_population(
        missing_nutrients: np.ndarray,
        population: list[tuple[MealItem, ...]],
        threesold: int = EPOCH_THREESOLD,
    ):
        return heapq.nlargest(
            EPOCH_THREESOLD,
            population,
            lambda x: calc_cos_similarity(missing_nutrients, x),
        )

    def _make_crossing(
        population: list[tuple[MealItem, ...]],
        all_meals: list[MealItem],
    ):
        random.shuffle(population)
        mothers = population[: len(population) // 2]
        fathers = population[len(population) // 2 :]

        new_population = []

        for _mother, _father in zip(mothers, fathers):
            mother = list(_mother)
            father = list(_father)

            for _ in range(1, MAX_CHILDREN):
                child_len = (
                    len(mothers) + len(fathers)
                ) // 2  # some medial len of mother and father

                # some randomization
                random.shuffle(mother)
                random.shuffle(father)

                # half from mother and half from father
                child = [*mother[: child_len // 2], *mother[: child_len // 2]]

                # some chances to add new item
                if random.choice(range(ADD_CHANCE)) == 0:
                    child.append(random.choice(all_meals))

                # some chances to remove item
                if random.choice(range(REMOVE_CHANCE)) == 0:
                    del_ind = random.randint(0, len(child) - 1)
                    child.pop(del_ind)

                new_population.append(child)
        return new_population

    initial_population = [
        *iter.combinations(all_meals, 1),
    ]

    for meals_count in range(2, max_meals + 1):
        initial_population.extend(
            list(
                iter.islice(
                    iter.combinations(all_meals, meals_count),
                    INIT_POPULATION_BATCH_SIZE,
                )
            )
        )

    prev_population = initial_population

    for epoch in range(EPOCHS):
        print("Epoch:", epoch)
        best_population = _select_best_population(missing_nutrients, prev_population)
        new_population = _make_crossing(best_population, all_meals)
        prev_population = new_population

    r = _select_best_population(missing_nutrients, prev_population, 1)

    return r


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
