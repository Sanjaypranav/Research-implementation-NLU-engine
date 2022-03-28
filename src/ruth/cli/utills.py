from pathlib import Path
from typing import Any, Dict, Text

import yaml
from ruth.shared.nlu.ruth_elements import Element
from ruth.nlu.registry import registered_classes


def get_config(pipeline_path: Path) -> Dict[Text, Any]:
    with open(pipeline_path, "r") as f:
        return yaml.safe_load(f)
