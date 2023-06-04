from cli import Approach, build_parser
import numpy as np
from helpers import load_meals_data

from logic import naive_brute_force_approach, naive_brute_force_mp_approach, genetic


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    logic_map = {
        Approach.naive: naive_brute_force_approach,
        Approach.naive_mp: naive_brute_force_mp_approach,
        Approach.genetic: genetic,
    }

    target_nutrients = np.array(args.target_nutrients)
    current_nutrients = np.array(args.current_nutrients)
    all_meals = load_meals_data(args.meals_file[0])

    logic_func = logic_map[args.approach]
    missing_nutrients = target_nutrients - current_nutrients
    best_meals, best_similarity = logic_func(
        all_meals=all_meals,
        missing_nutrients=missing_nutrients,
    )

    print("Best meals:", *best_meals, sep="\n")
    print("Best similarity:", best_similarity)
