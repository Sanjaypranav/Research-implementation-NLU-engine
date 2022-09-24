from typing import List, Text

from ruth.nlu.elements import Element


def module_path_from_object(o: Element) -> Text:
    return o.__class__.__module__ + "." + o.__class__.__name__


def check_required_elements(required_element: Element, pipeline: List[Element]) -> bool:
    for element in pipeline:
        if isinstance(element, required_element):
            return True
    return False
