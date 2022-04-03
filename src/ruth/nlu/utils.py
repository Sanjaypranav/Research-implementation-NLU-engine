from typing import Text

from ruth.nlu.elements import Element


def module_path_from_object(o: Element) -> Text:
    return o.__class__.__module__ + "." + o.__class__.__name__
