from __future__ import annotations

from typing import Any

from torch import nn

from flightrl.puffer4_edge_contract import (
    EDGE_HEIGHT,
    EDGE_OBSERVATION_DIM,
    EDGE_WIDTH,
)
from flightrl.puffer4_edge_policy import EdgeNavigationActor
from flightrl.puffer4_edge_wire import EDGE_INPUT_PACKET_BYTES


def edge_actor_budget(actor: EdgeNavigationActor) -> dict[str, Any]:
    """Return static graph costs; an actual GAP8 ELF map remains mandatory."""
    convolutions = [
        module for module in actor.visual.modules() if isinstance(module, nn.Conv2d)
    ]
    if len(convolutions) != 2:
        raise ValueError("edge budget supports the approved two-convolution graph")
    weights = sum(
        parameter.numel()
        for name, parameter in actor.named_parameters()
        if name.endswith("weight")
    )
    biases = sum(
        parameter.numel()
        for name, parameter in actor.named_parameters()
        if name.endswith("bias")
    )
    parameter_count = weights + biases
    conv_macs, largest_internal_activation = _convolution_costs(convolutions)
    linear_macs = sum(
        module.in_features * module.out_features
        for module in actor.modules()
        if isinstance(module, nn.Linear)
    )
    quantized_bytes = weights + 4 * biases
    return {
        "parameter_count": parameter_count,
        "int8_weight_bytes": weights,
        "int32_bias_bytes": 4 * biases,
        "quantized_parameter_bytes": quantized_bytes,
        "packed_input_bytes": EDGE_INPUT_PACKET_BYTES,
        "macs_per_step": conv_macs + linear_macs,
        "model_input_elements": EDGE_OBSERVATION_DIM,
        "largest_internal_activation_elements": largest_internal_activation,
        "largest_single_tensor_elements": max(
            EDGE_OBSERVATION_DIM,
            largest_internal_activation,
        ),
        "within_contract": parameter_count <= 50_000
        and quantized_bytes <= 64 * 1024,
        "measurement_boundary": "static_graph_estimate_not_gap8_elf_or_latency",
    }


def _convolution_costs(
    convolutions: list[nn.Conv2d],
) -> tuple[int, int]:
    height, width = EDGE_HEIGHT, EDGE_WIDTH
    macs = 0
    largest_internal_activation = 0
    for convolution in convolutions:
        kernel_h, kernel_w = convolution.kernel_size
        stride_h, stride_w = convolution.stride
        padding_h, padding_w = convolution.padding
        height = (height + 2 * padding_h - kernel_h) // stride_h + 1
        width = (width + 2 * padding_w - kernel_w) // stride_w + 1
        output_elements = convolution.out_channels * height * width
        kernel_macs = convolution.in_channels * kernel_h * kernel_w
        macs += output_elements * kernel_macs
        largest_internal_activation = max(
            largest_internal_activation,
            output_elements,
        )
    return macs, largest_internal_activation
