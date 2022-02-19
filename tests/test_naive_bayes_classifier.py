# import json
# from pathlib import Path
#
# import pytest
# from ruth.nlu.classifiers.naive_bayes_classifier import NaiveBayesClassifier
# from ruth.shared.nlu.training_data.collections import TrainData
#
#
# @pytest.fixture
# def classifier_data(example_classifier_data: Path) -> TrainData:
#     with open(example_classifier_data, "r") as f:
#         examples = json.load(f)
#
#     training_data = TrainData()
#     for value in examples:
#         training_data.add_example(value)
#
#     return training_data
#
#
# def test_naive_bayes_classifier(classifier_data: TrainData):
#
#     classifier = NaiveBayesClassifier()
#     classifier.train(training_data=classifier_data)
