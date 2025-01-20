#!/usr/bin/env python
# Padding

import argparse
from utils.metadata_tools import get_image_file_shape

def get_max_axis_value(files):

    shapes = [get_image_file_shape(file, format=".nd2") for file in files]
    max_shape = tuple(map(max, zip(*shapes)))

    return max_shape


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        required=True,
        help="A string of nd2 files.",
    )

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    files = args.input.split()
    max_shape = get_max_axis_value(files)

    with open("pad.txt", "w") as file:
        file.write(str(max_shape))


if __name__ == "__main__":
    main()
