from copy import deepcopy
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from dataclasses import dataclass
from typing import List, Callable
from torch import Tensor, torch
from enum import Enum


class ActivationMode(Enum):
    ORIGINAL = 1
    CAPTURE = 2
    STEER = 3


@dataclass
class SteerElement:
    text: str
    tensor: Tensor
    coeff: float
    try_keep_nr: int
    exclude_bos_token: bool = False
    steering_method: Callable = None


@dataclass
class SteerData:
    orig_forward_fn: torch.nn.Module.forward
    layer_idx: int
    steer_vectors: List[SteerElement]


class Steer:
    steers = {}

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        copyModel: bool = False,
    ):
        self.model = deepcopy(model) if copyModel else model
        self.tokenizer = tokenizer
        self.device = torch.device(next(model.parameters()).device)

    def _set_forward_fn(self, option: ActivationMode, layer_idx: int):
        if option == ActivationMode.ORIGINAL:
            steer = self.steers.pop(layer_idx, None)
            if steer is not None:
                self.model._modules["model"].layers[
                    layer_idx
                ].forward = steer.orig_forward_fn
        elif option == ActivationMode.CAPTURE:
            self.steers.setdefault(
                layer_idx,
                SteerData(
                    orig_forward_fn=self.model._modules["model"]
                    .layers[layer_idx]
                    .forward,
                    layer_idx=layer_idx,
                    steer_vectors=[],
                ),
            )
            self.model._modules["model"].layers[
                layer_idx
            ].forward = self._store_activations_forward(layer_idx)
        elif option == ActivationMode.STEER:
            self.steers.setdefault(
                layer_idx,
                SteerData(
                    orig_forward_fn=self.model._modules["model"]
                    .layers[layer_idx]
                    .forward,
                    layer_idx=layer_idx,
                    steer_vectors=[],
                ),
            )
            self.model._modules["model"].layers[
                layer_idx
            ].forward = self._steer_vector_forward(layer_idx)

    def _store_activations_forward(self, layer_idx: int):
        def _store_activations_forward_inner(*args, **kwargds):
            self.captured_tensor = (
                kwargds["hidden_states"] if "hidden_states" in kwargds else args[0]
            )
            return self.steers[layer_idx].orig_forward_fn(*args, **kwargds)

        return _store_activations_forward_inner

    def _steer_vector_forward(self, layer_idx: int):
        def _steer_vector_forward_inner(*args, **kwargds):
            for elem in self.steers[layer_idx].steer_vectors:
                if elem.tensor.size()[1] <= elem.try_keep_nr:
                    extraText = ""
                    if elem.tensor.size()[1] == elem.try_keep_nr:
                        extraText = """ In case you're using exclude_bos_token=True, 
                        you could also consider setting it to False and retrying."""
                    raise Exception(
                        f"""Invalid try_keep_nr value. Current value is {elem.try_keep_nr}, 
                    but it has to be less than the number of text tokens (in this case {elem.tensor.size()[1]}) on layer index {layer_idx}. 
                    You could set a lower value for try_keep_nr or provide longer text for steering. {extraText}"""
                    )

                if elem.steering_method is not None:
                    delta = elem.steering_method(elem.tensor, elem.coeff, elem.try_keep_nr)
                else:
                    delta = torch.mean(
                        elem.coeff * elem.tensor[:, elem.try_keep_nr :, :],
                        dim=1,
                        keepdim=True,
                    )

                if "hidden_states" in kwargds:
                    if kwargds["hidden_states"].size()[1] == 1:
                        kwargds["hidden_states"][:, -1:, :] += delta
                    else:
                        kwargds["hidden_states"][:, elem.try_keep_nr :, :] += delta
                elif isinstance(args[0], Tensor):
                    if args[0].size()[1] == 1:
                        args[0][:, -1:, :] += delta
                    else:
                        args[0][:, elem.try_keep_nr :, :] += delta
                else:
                    raise Exception(
                        "The model is not currently supported. Please open an issue in the official GitHub repository."
                    )

            return self.steers[layer_idx].orig_forward_fn(*args, **kwargds)

        return _steer_vector_forward_inner

    def get_all(self):
        """
        Get all the steering vectors data that are applied on the model.
        Can be used for replicating in the future the state.
        """
        return [{'layer_idx': val.layer_idx, 'text': x.text, 'coeff': x.coeff, 'try_keep_nr': x.try_keep_nr, 'exclude_bos_token': x.exclude_bos_token} for val in self.steers.values() for x in val.steer_vectors]

    def reset(self, layer_idx: int):
        """
        Remove the steering vectors on a particular layer.      
        Args:
            layer_idx (int): The layer index that will have the steering vectors removed.
        """
        self._set_forward_fn(ActivationMode.ORIGINAL, layer_idx)

    def reset_all(self):
        """
        Remove all steering vectors that were applied on the model.
        Gets the model to initial state, before wrapping it in the Steer class and using add(). 
        """
        [self.reset(idx) for idx in range(len(self.model._modules["model"].layers))]

    def add(
        self,
        layer_idx: int,
        coeff: float,
        text: str,
        try_keep_nr: int = None,
        exclude_bos_token: bool = False,
        steering_method: Callable = None,
    ):
        """
        Add a steering vector.
        Args:
            layer_idx (int): The layer index to apply the steering vector on. Usually is toward the end.
            coeff: The steerging vectors coefficient. Usually is below 1. Can also be negative.
            text: The steering vector text.
            try_keep_nr: This is used in advanced usage and determines the number of rows of the initial
                matrix to be kept. The param is used for expetimenting. Leave to default value for best usage.
            exclude_bos_token: This is used in advanced usage and determines if the beginning of a sentence
                (bos) token should be removed. By default, the code ensures the tokens used for generating
                start with the bos token. The param is used for experimenting. Leave to default value for best usage.
            steering_method: A function that can be used to determine the steering method/formula. For more details, see https://github.com/Mihaiii/llm_steer/pull/2
        """
        assert layer_idx >= 0 and layer_idx < len(
            self.model._modules["model"].layers
        ), f"""Current model has {len(self.model._modules['model'].layers)} layers, 
        but the provided layer_idx is not within 
        [0, {len(self.model._modules['model'].layers) - 1}] interval."""
        
        text_tokens = self.tokenizer.encode(text)

        #inject bos_token
        #This can be reverted with exclude_bos_token=True
        if self.tokenizer.bos_token is not None and text_tokens[0] != self.tokenizer.encode(self.tokenizer.bos_token)[-1]:
            text_tokens.insert(0, self.tokenizer.encode(self.tokenizer.bos_token)[-1])
        
        if (
            exclude_bos_token
            and self.tokenizer.bos_token is not None
        ):
            text_tokens = text_tokens[1:]
            
        print(f"text tokens: {text_tokens}")
        
        layer_tensor = self._capture_tensor(
            layer_idx, torch.tensor(text_tokens).to(self.device).unsqueeze(0)
        )

        if try_keep_nr is None:
            try_keep_nr = 0 if self.tokenizer.bos_token is None else 1

        self._add_steer_vector(
            layer_idx,
            SteerElement(
                text=text,
                tensor=layer_tensor,
                coeff=coeff,
                try_keep_nr=try_keep_nr,
                exclude_bos_token=exclude_bos_token,
                steering_method=steering_method,
            ),
        )

    def _add_steer_vector(self, layer_idx: int, steerElem: SteerElement):
        steer = self.steers.setdefault(
            layer_idx,
            SteerData(
                orig_forward_fn=self.model._modules["model"].layers[layer_idx].forward,
                layer_idx=layer_idx,
                steer_vectors=[],
            ),
        )
        steer.steer_vectors.append(steerElem)
        self._set_forward_fn(ActivationMode.STEER, layer_idx)

    def _capture_tensor(self, layer_idx: int, tokens: Tensor):
        self._set_forward_fn(ActivationMode.CAPTURE, layer_idx)
        with torch.inference_mode():
            self.model(tokens)
        result = self.captured_tensor
        print(f"captured tensor: {result}")
        return result
