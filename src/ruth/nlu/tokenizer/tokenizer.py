from typing import Text, Dict, Any, List

from ruth.constants import TEXT
from ruth.nlu.elements import Element
from ruth.shared.nlu.training_data.collections import TrainData


class Token:
    def __init__(self, text: Text, start: int, end: int, data: Dict[Text, Any] = None):
        self.text = text
        self.start = start
        self.end = end
        self.data = data or {}

    def __eq__(self, other):
        return (self.text, self.start, self.end) == (other.text, other.start, other.end)


class Tokenizer(Element):
    def train(self, training_data: TrainData):
        for data in training_data.training_examples:
            text = data.get(TEXT)
            tokens = self.tokenize(text)


    @staticmethod
    def _convert_words_to_tokens(words: List[Text], text: Text) -> List[Token]:
        running_offset = 0
        tokens = []

        for word in words:
            word_offset = text.index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            tokens.append(Token(word, start=word_offset, end=running_offset))

        return tokens

    def tokenize(self, text: Text):
        raise NotImplementedError(f"failed to implement the tokenize function in {self.name}")