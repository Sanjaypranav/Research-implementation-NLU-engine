import torch


class Tokens:
    def __init__(self, tokens: torch.Tensor):  # TODO add origin
        self.tokens = tokens

    def is_token(self):
        return isinstance(self.tokens, torch.Tensor)
