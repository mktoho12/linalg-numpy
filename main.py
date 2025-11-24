import argparse

from linalg_lib.decompositions import (demonstrate_lu_decomposition,
                                       demonstrate_qr_decomposition)
from linalg_lib.matrix_operations import demonstrate_matrix_operations
from linalg_lib.vector_operations import demonstrate_vector_operations


def main() -> None:
    """
    Entry point to demonstrate linear algebra operations.
    Parses command-line arguments to run specific demonstrations.
    """
    parser = argparse.ArgumentParser(
        description="A CLI tool to demonstrate linear algebra operations with NumPy.",
        epilog="Example: python main.py qr"
    )
    parser.add_argument(
        "topic",
        nargs="?",  # Make the argument optional
        choices=["vector", "matrix", "lu", "qr", "all"],
        default=None,  # Default to None if no argument is provided
        help="Specify the topic to demonstrate. 'all' runs all topics."
    )

    args = parser.parse_args()

    if args.topic is None:
        parser.print_help()
        return

    if args.topic in ["vector", "all"]:
        demonstrate_vector_operations()

    if args.topic in ["matrix", "all"]:
        demonstrate_matrix_operations()

    if args.topic in ["lu", "all"]:
        # We need to make sure scipy is installed
        try:
            demonstrate_lu_decomposition()
        except ImportError:
            print("\nError: 'scipy' is not installed. Please run the command inside the dev container:")
            print("uv pip sync pyproject.toml")

    if args.topic in ["qr", "all"]:
        demonstrate_qr_decomposition()


if __name__ == "__main__":
    main()