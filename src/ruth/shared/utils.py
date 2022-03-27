import json
from pathlib import Path
from typing import Any, Text, Union
import yaml
from ruth.shared.constants import DEFAULT_ENCODING


def write_text_file(
        content: Text,
        file_path: Union[Text, Path],
        encoding: Text = DEFAULT_ENCODING,
        append: bool = False,
) -> None:
    mode = "a" if append else "w"
    with open(file_path, mode, encoding=encoding) as file:
        file.write(content)


def json_pickle(file_name: Union[Text, Path], obj: Any) -> None:
    import jsonpickle.ext.numpy as jsonpickle_numpy
    import jsonpickle

    jsonpickle_numpy.register_handlers()

    write_text_file(jsonpickle.dumps(obj), file_name)


def read_data(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_training_data(path: Path):
    with open(path, "r") as f:
        return json.load(f)
