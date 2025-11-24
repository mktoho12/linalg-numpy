import argparse
from linalg_lib.vector_operations import demonstrate_vector_operations
from linalg_lib.matrix_operations import demonstrate_matrix_operations


def main() -> None:
    """
    Entry point to demonstrate linear algebra operations.
    Parses command-line arguments to run specific demonstrations.
    """
    parser = argparse.ArgumentParser(
        description="A CLI tool to demonstrate linear algebra operations with NumPy.",
        epilog="Example: python main.py vector"
    )
    parser.add_argument(
        "topic",
        nargs="?",  # Make the argument optional
        choices=["vector", "matrix", "all"],
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


if __name__ == "__main__":
    main()