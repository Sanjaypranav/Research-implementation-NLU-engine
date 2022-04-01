import json
from pathlib import Path
from typing import Any, Dict, List, Text

import yaml

from ruth.nlu.model import ElementBuilder


def get_config(pipeline_path: Path) -> Dict[Text, Any]:
    with open(pipeline_path, "r") as f:
        return yaml.safe_load(f)


def load_json_data(path: Path) -> Dict[Text, Any]:
    with open(path, "r") as f:
        return json.load(f)


def build_pipeline_from_metadata(
    metadata: Dict[Text, Any], model_dir: Path, element_builder: ElementBuilder = None,
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
