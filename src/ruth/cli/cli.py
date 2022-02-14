from pathlib import Path

import click
from rich.console import Console
from ruth import VERSION
from ruth.shared.nlu.training_data.collections import TrainData

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
    training_data = TrainData.build(data)
    print(training_data)
    print(pipeline)
