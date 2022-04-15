import copy
from typing import Any, Dict, Optional, Text


def override_defaults(
    defaults: Optional[Dict[Text, Any]], custom: Optional[Dict[Text, Any]]
) -> Dict[Text, Any]:
    if defaults:
        config = copy.deepcopy(defaults)
    else:
        config = {}

    if custom:
        for key in custom.keys():
            if isinstance(config.get(key), dict):
                config[key].update(custom[key])
            else:
                config[key] = custom[key]

    return config
