import json
import re
from pathlib import Path
from typing import Any, Dict, List, Text

import yaml
from pydantic import BaseModel
from ruth.nlu.model import ElementBuilder, Interpreter


def get_config(pipeline_path: Path) -> Dict[Text, Any]:
    with open(pipeline_path, "r") as f:
        return yaml.safe_load(f)


def load_json_data(path: Path) -> Dict[Text, Any]:
    with open(path, "r") as f:
        return json.load(f)


def build_pipeline_from_metadata(
    metadata: Dict[Text, Any],
    model_dir: Path,
    element_builder: ElementBuilder = None,
):
    pipeline_element = []
    if not element_builder:
        element_builder = ElementBuilder()

    pipeline: List[Dict[Text, Any]] = metadata["pipeline"]
    for element in pipeline:
        pipeline_element.append(
            element_builder.load_element(element["name"], element, model_dir=model_dir)
        )
    return pipeline_element


def get_metadata_from_model(model_path: Path) -> Dict[Text, Any]:
    metadata_file_path = model_path / "metadata.json"
    metadata = load_json_data(metadata_file_path)
    return metadata


def get_interpreter_from_model_path(model_path: str) -> Interpreter:
    model_path = check_model_path(model_path)
    metadata = get_metadata_from_model(model_path.absolute())
    pipeline = build_pipeline_from_metadata(metadata=metadata, model_dir=model_path)
    return Interpreter(pipeline)


def check_model_path(model_path: str) -> Path:
    if model_path:
        if Path.exists(Path(model_path)):
            model_file = model_path
        else:
            raise FileNotFoundError(
                "Model does not exist in the given path.\nTo train: ruth train"
            )
    else:
        model_folder = "models"
        if not Path(model_folder).exists():
            raise FileNotFoundError(
                "No models found.\nTrain new models using: ruth train"
            )
        models = [
            directory
            for directory in Path(model_folder).iterdir()
            if directory.is_dir() and re.search("ruth", str(directory))
        ]
        if models:
            models.sort()
            model_file = models[-1]
        else:
            raise FileNotFoundError(
                "No models found.\nTrain new models using: ruth train"
            )

    return Path(model_file)


def local_example_path(output_path: Text) -> Path:
    if output_path:
        return Path(output_path) / "data" / "example.yml"
    else:
        data_path = Path().absolute() / "data"
        data_path.mkdir(exist_ok=True)
        data_path = data_path / "example.yml"
        return data_path


def local_pipeline_path(output_path: Text) -> Path:
    if output_path:
        return Path(output_path)
    else:
        pipeline_path = Path().absolute() / "pipeline.yml"
        return pipeline_path


class Item(BaseModel):
    text: str
