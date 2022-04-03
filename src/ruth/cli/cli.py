import re
from pathlib import Path
from typing import Text

import click
from rich.console import Console
from ruth import VERSION
from ruth.cli.utills import (
    build_pipeline_from_metadata,
    get_config,
    get_metadata_from_model,
)
from ruth.constants import INTENT
from ruth.nlu.model import Interpreter
from ruth.nlu.train import train_pipeline
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
    training_data = TrainData.build(data)

    config = RuthConfig(config)
    model_absolute_dir = train_pipeline(config, training_data)
    console.print(f"Training is completed and model is stored at [yellow]{model_absolute_dir}[/yellow]")


@entrypoint.command(name="parse")
@click.option(
    "-t",
    "--text",
    type=click.STRING,
    required=True,
    help="Data that need to be get parsed",
)
@click.option(
    "-p",
    "--path",
    type=click.STRING,
    default="models",
    help="Directory where the model is stored",
)
def parse(text: Text, path: Text):
    models = [
        directory
        for directory in Path(path).iterdir()
        if directory.is_dir() and re.search("ruth", str(directory))
    ]
    models.sort()

    latest_model = models[-1]

    console.print(f"Latest Model found {latest_model}")
    metadata = get_metadata_from_model(latest_model.absolute())
    pipeline = build_pipeline_from_metadata(metadata=metadata, model_dir=latest_model)
    interpreter = Interpreter(pipeline)
    output = interpreter.parse(text)
    console.print(f"Predicted intent is {output.get(INTENT)}")
