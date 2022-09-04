import json
import os
import re
from pathlib import Path
from typing import Text

import click
import matplotlib.pyplot as plt
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
from sklearn.metrics import confusion_matrix

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
    console.print(
        f"Training is completed and model is stored at [yellow]{model_absolute_dir}[/yellow]"
    )


@entrypoint.command(name="parse")
@click.option(
    "-t",
    "--text",
    type=click.STRING,
    required=True,
    help="Data that need to be get parsed",
)
@click.option(
    "-m",
    "--model_path",
    type=click.STRING,
    default="models",
    help="Directory where the model is stored",
)
def parse(text: Text, model_path: Text):
    models = [
        directory
        for directory in Path(model_path).iterdir()
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


@entrypoint.command(name="evaluate")
@click.option(
    "-d",
    "--data",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Data for testing as json",
)
@click.option(
    "-m",
    "--model_path",
    type=click.Path(exists=True),
    default=Path("models"),
    help="Directory where the model is stored",
)
@click.option(
    "-o",
    "--output_folder",
    type=click.Path(),
    default=Path("results"),
    help="Directory where the results is stored",
)
def evaluate(data: Path, model_path: Text, output_folder: Text):
    models = [
        directory
        for directory in Path(model_path).iterdir()
        if directory.is_dir() and re.search("ruth", str(directory))
    ]
    models.sort()

    latest_model = models[-1]

    console.print(f"Latest Model found {latest_model}")
    metadata = get_metadata_from_model(latest_model.absolute())
    pipeline = build_pipeline_from_metadata(metadata=metadata, model_dir=latest_model)
    interpreter = Interpreter(pipeline)
    with open(data, "r") as f:
        examples = json.load(f)

    correct_predictions = 0
    y_pred = []
    y_actual = []
    for example in examples:
        output = interpreter.parse(example["text"])

        y_pred.append(output.get(INTENT).get("name"))
        y_actual.append(example["intent"])

        if output.get(INTENT).get("name") == example["intent"]:
            correct_predictions += 1

    accuracy = correct_predictions / len(examples)
    conf_matrix = confusion_matrix(y_true=y_actual, y_pred=y_pred)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(
                x=j, y=i, s=conf_matrix[i, j], va="center", ha="center", size="xx-large"
            )

    plt.xlabel("Predictions", fontsize=18)
    plt.ylabel("Actual", fontsize=18)
    plt.title("Confusion Matrix", fontsize=18)

    if output_folder:
        result_path = Path(output_folder).absolute()
    else:
        result_path = Path().absolute() / output_folder
    result_path.mkdir(exist_ok=True)
    directories = os.listdir(str(result_path))
    indexes = []
    model_name = str(latest_model).split("/")[-1]
    for result in directories:
        if model_name in result:
            indexes.append(int(result.split("@")[-1]))
    if indexes:
        index = max(indexes) + 1
    else:
        index = 0

    folder_for_the_result = result_path / f"{model_name}@{index}"
    folder_for_the_result.mkdir(exist_ok=True)
    final_file_path = folder_for_the_result / "confusion_matrix.png"
    plt.savefig(final_file_path)

    print("accuracy: ", accuracy)
    print("confusion matrix is created.")
    print("results are stored here: ", folder_for_the_result)
