from pathlib import Path
from typing import Any, Dict, List, Text

import click
from rich.console import Console
from ruth import VERSION
from ruth.cli.utills import create_component, get_config
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_config import RuthConfig

console = Console()


class RichGroup(click.Group):
    def format_help(self, ctx, formatter):
        # TODO: Want to write the help description whenever the user call the ruth --help
        ...


@click.group(cls=RichGroup)
@click.version_option(VERSION)
def entrypoint():
    pass


@entrypoint.command(name="train")
@click.option(
    "-d",
    "--data",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Data for training as json",
)
@click.option(
    "-p",
    "--pipeline",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="pipeline for training as yaml",
)
def train(data: Path, pipeline: Path):
    config = get_config(pipeline)
    pipeline: List[Dict[Text, Any]] = pipeline["pipeline"]
    training_data = TrainData.build(data)
    pipeline_classes = []
    for element in pipeline:
        pipeline_classes.append(create_component(element.get("name")))

    config = RuthConfig(config)
    for element_class, element_config in zip(pipeline_classes, pipeline):
        e_class = element_class.build(config=config, element_config=element_config)
        e_class.train(training_data)
