# output.py
#
# Course: ABM (2022/2023)
#
# Description: Function to get the personal path to
# the save_results folder for the user.

import os
from pathlib import Path


def get_output_path():
    """Generates output path line to save_results folder
     for local user for saving files.

    Returns: local user path
    """
    path = os.path.abspath(__file__)
    output_path = Path(path).parent
    # move to the parent if still in the code directory
    if os.path.split(output_path)[1] == "code":
        output_path = output_path.parent

    return output_path