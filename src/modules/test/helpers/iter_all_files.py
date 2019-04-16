import os
from typing import Callable
from pdb import set_trace


def apply_all_mocks(func) -> Callable[[str], any]:
    """
    This should be used as decorator. ex: @apply_all_mocks
    """

    def wrapper(file_or_dir_name):
        dir_name = format_as_dir(file_or_dir_name)
        files = os.listdir(dir_name)
        for file in files:
            func(file)

    return wrapper


def format_as_dir(file_or_dir_name: str) -> str:
    if os.path.isfile(file_or_dir_name):
        return os.path.dirname(file_or_dir_name)

    if os.path.isdir(file_or_dir_name):
        return file_or_dir_name

    raise Exception('{} does not exist'.format(file_or_dir_name))
