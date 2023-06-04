import argparse
from enum import Enum

from consts import MAX_MEALS


class Approach(Enum):
    naive = "naive"
    naive_mp = "naive_mp"
    genetic = "genetic"

    def __str__(self):
        return self.value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize diet nutrients")
    parser.add_argument(
        "-m",
        "--meals_file",
        type=str,
        nargs=1,
        metavar="meals_file",
        default=None,
        help="""
            File (.yaml) which contains list of meals with nutrients described.
            E.g.: nutrient_vectors.yaml
            Required: True
            """,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--current_nutrients",
        type=int,
        nargs="+",
        metavar="current_nutrients",
        default=None,
        help="""
            Current nutrients consumption array[int].
            E.g.: 1 2 3 4 5
            Required: True
            """,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--target_nutrients",
        type=int,
        nargs="+",
        metavar="target_nutrients",
        default=None,
        help="""
            Target nutrients consumption array[int].
            E.g.: 1 2 3 4 5
            Required: True
            """,
        required=True,
    )
    parser.add_argument(
        "-mc",
        "--max_meals_count",
        type=int,
        nargs=1,
        metavar="max_meals_count",
        default=MAX_MEALS,
        help="""
            Max product count to achive target diet. 
            E.g.: 10. 
            Required: False
            """,
        required=False,
    )
    parser.add_argument(
        "-a",
        "--approach",
        type=Approach,
        metavar="approach",
        default=None,
        help="""
            Solution approach: naive, naive-mp, genetic
            Required: True
            """,
        choices=list(Approach),
        required=True,
    )

    return parser
