from copy import deepcopy
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from dataclasses import dataclass
from typing import List, Callable, Optional
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
    steering_method: Callable = None
    coeff_schedule: Optional[Callable] = None


@dataclass
class SteerData:
    orig_forward_fn: torch.nn.Module.forward
    layer_idx: int
    steer_vectors: List[SteerElement]
    step: int = 0


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
        self._layer_norm_eps = self._resolve_layer_norm_eps()

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
            self.model._modules["model"].layers[layer_idx].forward = (
                self._store_activations_forward(layer_idx)
            )
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
            self.model._modules["model"].layers[layer_idx].forward = (
                self._steer_vector_forward(layer_idx)
            )

    @staticmethod
    def _rms(x: Tensor, eps: float = 1e-6) -> Tensor:
        # Use float32 for stability regardless of model dtype.
        return torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)

    @staticmethod
    def _layer_norm(x: Tensor, eps: float) -> Tensor:
        x_float = x.float()
        mean = x_float.mean(dim=-1, keepdim=True)
        var = x_float.var(dim=-1, unbiased=False, keepdim=True)
        return (x_float - mean) / torch.sqrt(var + eps)

    def _resolve_layer_norm_eps(self) -> Optional[float]:
        for module in self.model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                return float(module.eps)
        return None

    def _normalize_direction(self, direction: Tensor) -> Tensor:
        if self._layer_norm_eps is not None:
            return self._layer_norm(direction, eps=self._layer_norm_eps)
        direction_rms = self._rms(direction)
        return direction / (direction_rms + 1e-6)

    def _store_activations_forward(self, layer_idx: int):
        def _store_activations_forward_inner(*args, **kwargds):
            self.captured_tensor = (
                kwargds["hidden_states"] if "hidden_states" in kwargds else args[0]
            )
            return self.steers[layer_idx].orig_forward_fn(*args, **kwargds)

        return _store_activations_forward_inner

    def _steer_vector_forward(self, layer_idx: int):
        def _steer_vector_forward_inner(*args, **kwargds):
            steer_state = self.steers[layer_idx]
            step = steer_state.step
            for elem in self.steers[layer_idx].steer_vectors:
                if elem.steering_method is not None:
                    delta = elem.steering_method(elem, step, layer_idx)
                else:
                    direction = torch.mean(
                        elem.tensor,
                        dim=1,
                        keepdim=True,
                    )
                    direction = self._normalize_direction(direction)
                    delta = elem.coeff * direction
                    if elem.coeff_schedule is not None:
                        schedule_multiplier = elem.coeff_schedule(step)
                        delta = delta * float(schedule_multiplier)
                if "hidden_states" in kwargds:
                    delta = delta * self._rms(kwargds["hidden_states"][:, -1:, :])
                    kwargds["hidden_states"][:, :, :] += delta.to(
                        kwargds["hidden_states"].dtype
                    )
                elif isinstance(args[0], Tensor):
                    delta = delta * self._rms(args[0][:, -1:, :])
                    args[0][:, :, :] += delta.to(args[0].dtype)
                else:
                    raise Exception(
                        "The model is not currently supported. Please open an issue in the official GitHub repository."
                    )
            steer_state.step = step + 1
            return self.steers[layer_idx].orig_forward_fn(*args, **kwargds)

        return _steer_vector_forward_inner

    def get_all(self):
        """
        Get all the steering vectors data that are applied on the model.
        Can be used for replicating in the future the state.
        """
        return [
            {
                "layer_idx": val.layer_idx,
                "text": x.text,
                "coeff": x.coeff,
                "coeff_schedule": (
                    x.coeff_schedule.to_dict()
                    if hasattr(x.coeff_schedule, "to_dict")
                    else None
                ),
            }
            for val in self.steers.values()
            for x in val.steer_vectors
        ]

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
        steering_method: Callable = None,
        coeff_schedule: Optional[Callable] = None,
    ):
        """
        Add a steering vector.
        Args:
            layer_idx (int): The layer index to apply the steering vector on. Usually is toward the end.
            coeff: The steerging vectors coefficient. Usually is below 1. Can also be negative.
            text: The steering vector text.
            steering_method: A function that can be used to determine the steering method/formula. For more details, see https://github.com/Mihaiii/llm_steer/pull/2
            coeff_schedule: Optional callable that returns a multiplier per step;
                effective coefficient is `coeff * coeff_schedule(step)` (step increments each time this layer runs).
        """
        assert layer_idx >= 0 and layer_idx < len(
            self.model._modules["model"].layers
        ), f"""Current model has {len(self.model._modules['model'].layers)} layers, 
        but the provided layer_idx is not within 
        [0, {len(self.model._modules['model'].layers) - 1}] interval."""

        text_tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # print(f"text tokens: {text_tokens}")

        layer_tensor = self._capture_tensor(
            layer_idx, torch.tensor(text_tokens).to(self.device).unsqueeze(0)
        )

        self._add_steer_vector(
            layer_idx,
            SteerElement(
                text=text,
                tensor=layer_tensor,
                coeff=coeff,
                steering_method=steering_method,
                coeff_schedule=coeff_schedule,
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
        # print(f"captured tensor: {result}")
        return result
