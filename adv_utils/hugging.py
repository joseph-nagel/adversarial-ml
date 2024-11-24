'''Hugging Face model wrappers.'''

from typing import Self, Any

import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification


class HFClassifier(nn.Module):
    '''Hugging Face classifier.'''

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @property
    def class_names(self) -> dict[int, str]:
        return self.model.config.id2label

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: Any) -> Self:
        model = AutoModelForImageClassification.from_pretrained(
            model_name, **kwargs
        )
        return cls(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        return outputs['logits']

