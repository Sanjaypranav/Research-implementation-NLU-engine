from pathlib import Path
from typing import Any, Dict, List, Text

import click
from rich.console import Console
from ruth.cli.utills import get_config
from ruth.nlu.train import train_pipeline
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_config import RuthConfig

console = Console()


class RichGroup(click.Group):
    def format_help(self, ctx, formatter):
        # TODO: Want to write the help description whenever the user call the ruth --help
        ...


# @click.group(cls=RichGroup)
# @click.version_option(VERSION)
# def entrypoint():
#     pass
#
#
# @entrypoint.command(name="train")
# @click.option(
#     "-d",
#     "--data",
#     type=click.Path(exists=True, dir_okay=False),
#     required=True,
#     help="Data for training as json",
# )
# @click.option(
#     "-p",
#     "--pipeline",
#     type=click.Path(exists=True, dir_okay=False),
#     required=True,
#     help="pipeline for training as yaml",
# )
def train(data: Path, pipeline: Path):
    config = get_config(pipeline)
    pipeline: List[Dict[Text, Any]] = config["pipeline"]
    training_data = TrainData.build(data)

    config = RuthConfig(config)
    train_pipeline(config, training_data)
