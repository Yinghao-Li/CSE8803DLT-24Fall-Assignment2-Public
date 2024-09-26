"""
# Author: Yinghao Li
# Modified: September 13th, 2023
# ---------------------------------------
# Description: IO functions
"""


import os
import os.path as op
import regex
import json
import shutil
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["set_log_path", "set_logging", "logging_args", "init_dir", "save_json"]


def set_logging(log_path: Optional[str] = None):
    """
    setup logging format

    Parameters
    ----------
    log_path: where to save logging file. Leave None to save no log files

    Returns
    -------
    None
    """
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    if log_path and log_path != "disabled":
        log_path = op.abspath(log_path)
        if not op.isdir(op.split(log_path)[0]):
            os.makedirs(op.abspath(op.normpath(op.split(log_path)[0])))
        if op.isfile(log_path):
            os.remove(log_path)

        file_handler = logging.FileHandler(filename=log_path)
        file_handler.setLevel(logging.INFO)

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=0,
            handlers=[stream_handler, file_handler],
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[stream_handler],
        )

    return None


def remove_dir(directory: str):
    """
    Remove a directory and its subtree folders/files
    """
    dirpath = Path(directory)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    return None


def init_dir(directory: str, clear_original_content: Optional[bool] = True):
    """
    Create the target directory. If the directory exists, remove all subtree folders/files in it.
    """

    if clear_original_content:
        remove_dir(directory)
    os.makedirs(op.normpath(directory), exist_ok=True)
    return None


def save_json(obj, path: str, collapse_level: Optional[int] = None):
    """
    Save objective to a json file.
    Create this function so that we don't need to worry about creating parent folders every time

    Parameters
    ----------
    obj: the objective to save
    path: the path to save
    collapse_level: set to any collapse value to prettify output json accordingly

    Returns
    -------
    None
    """
    file_dir = op.dirname(op.normpath(path))
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    json_obj = json.dumps(obj, indent=2, ensure_ascii=False)
    if collapse_level:
        json_obj = prettify_json(json_obj, collapse_level=collapse_level)

    with open(path, "w", encoding="utf-8") as f:
        f.write(json_obj)

    return None


def prettify_json(text, indent=2, collapse_level=4):
    """
    Make json file more readable by collapsing indent levels higher than `collapse_level`.

    Parameters
    ----------
    text: input json text obj
    indent: the indent value of your json text. Notice that this value needs to be larger than 0
    collapse_level: the level from which the program stops adding new lines

    Usage
    -----
    ```
    my_instance = list()  # user-defined serializable data structure
    json_obj = json.dumps(my_instance, indent=2, ensure_ascii=False)
    json_obj = prettify_json(json_text, indent=2, collapse_level=4)
    with open(path_to_file, 'w', encoding='utf=8') as f:
        f.write(json_text)
    ```
    """
    pattern = r"[\r\n]+ {%d,}" % (indent * collapse_level)
    text = regex.sub(pattern, " ", text)
    text = regex.sub(r"([\[({])+ +", r"\g<1>", text)
    text = regex.sub(
        r"[\r\n]+ {%d}([])}])" % (indent * (collapse_level - 1)),
        r"\g<1>",
        text,
    )
    text = regex.sub(r"(\S) +([])}])", r"\g<1>\g<2>", text)
    return text
