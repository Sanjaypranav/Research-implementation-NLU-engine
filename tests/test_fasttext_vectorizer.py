from pathlib import Path

from ruth.constants import TEXT
from ruth.nlu.featurizers.dense_featurizers.fast_text import FastTextFeaturizer
from ruth.nlu.tokenizer.whitespace_tokenizer import WhiteSpaceTokenizer
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData


def test_fasttext_featurizer(example_data_path: Path):
    training_data = TrainData.build(example_data_path)
    white_space_tokenizer = WhiteSpaceTokenizer({})
    white_space_tokenizer.train(training_data=training_data)
    fasttext_vectorizer = FastTextFeaturizer({})
    fasttext_vectorizer.train(training_data=training_data)
    message = RuthData(data={TEXT: "hello i am from coimbatore"})
    white_space_tokenizer.parse(message)
    print(fasttext_vectorizer.parse(message).shape)
