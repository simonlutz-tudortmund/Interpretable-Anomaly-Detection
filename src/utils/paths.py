import logging
import os
from pathlib import Path


def remove_files_from_folder(absolute_path: Path):
    parent_folder = os.path.normpath(absolute_path)

    if os.path.isdir(parent_folder):
        for filename in os.listdir(parent_folder):
            logging.warning(filename)
            absPath = os.path.join(parent_folder, filename)
            logging.warning(f"ABSPATH {absPath}")
            os.remove(absPath)
    else:
        logging.warning("There is no directory {}".format(parent_folder))


def get_source_path() -> Path:
    return Path(os.path.join(os.getcwd(), "src"))


def get_experiments_path() -> Path:
    return Path(os.path.join(os.getcwd(), "experiments"))
