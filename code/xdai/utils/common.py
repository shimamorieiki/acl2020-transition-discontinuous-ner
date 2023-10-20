import json
import logging
import os
import random
import shutil
from typing import Any

import numpy as np
import spacy
import torch
from xdai.utils.args import SimpleArgumentParser

logger = logging.getLogger(__name__)


"""Update date: 2019-Nov-3"""


def create_output_dir(args: SimpleArgumentParser) -> SimpleArgumentParser:
    """_summary_
    出力結果を保存するフォルダを作成する
    Args:
        args (_type_): _description_

    Raises:
        ValueError: _description_
    """
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
    ):
        if args.overwrite_output_dir:
            shutil.rmtree(args.output_dir)
        else:
            raise ValueError("Output directory (%s) already exists." % args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    return args


"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/common/util.py#dump_metrics
Update date: 2019-Nov-3"""


def dump_metrics(file_path: str, metrics: dict[str, Any], log: bool = False) -> None:
    metrics_json = json.dumps(metrics, indent=2)
    with open(file_path, "w") as metrics_file:
        metrics_file.write(metrics_json)
    if log:
        logger.info("Metrics: %s", metrics_json)


"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
Update date: 2019-April-26"""


def has_tensor(obj) -> bool:
    if isinstance(obj, torch.Tensor):
        return True
    if isinstance(obj, dict):
        return any(has_tensor(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(has_tensor(i) for i in obj)
    return False


LOADED_SPACY_MODELS = {}
"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/common/util.py#get_spacy_model
Update date: 2019-Nov-25"""


def load_spacy_model(spacy_model_name, parse=False):
    options = (spacy_model_name, parse)
    if options not in LOADED_SPACY_MODELS:
        disable = ["vectors", "textcat", "tagger", "ner"]
        if not parse:
            disable.append("parser")
        spacy_model = spacy.load(spacy_model_name, disable=disable)
        LOADED_SPACY_MODELS[options] = spacy_model
    return LOADED_SPACY_MODELS[options]


"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/metrics/metric.py
Update date: 2019-03-01"""


def move_to_cpu(*tensors):
    return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)


"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py (move_to_device)
Update date: 2019-April-26"""


def move_to_gpu(
    obj: dict[
        str,
        torch.Tensor | dict[str, torch.Tensor] | list[str] | tuple[torch.Tensor, ...],
    ],
    cuda_device: int = 0,
) -> dict[
    str,
    torch.Tensor | dict[str, torch.Tensor] | list[str] | tuple[torch.Tensor, ...],
]:
    """_summary_
    dictの要素のうち、torch.Tensorのものはcudaに乗せる。それ以外はそのまま返す
    Args:
        obj (dict[ str, torch.Tensor  |  dict[str, torch.Tensor]  |  list[torch.Tensor]  |  tuple[torch.Tensor, ...], ]): _description_
        cuda_device (int, optional): _description_. Defaults to 0.

    Returns:
        dict[ str, torch.Tensor | dict[str, torch.Tensor] | list[torch.Tensor] | tuple[torch.Tensor, ...], ]: _description_
    """
    # print(type(obj))
    for k, v in obj.items():
        if isinstance(v, torch.Tensor):
            obj[k] = move_tensor_to_gpu(tensor=v, cuda_device=cuda_device)

        elif isinstance(v, list):
            # list[str]はcudaに乗せるものは存在しない
            pass

        elif isinstance(v, dict):
            obj[k] = {
                ik: move_tensor_to_gpu(tensor=tensor, cuda_device=cuda_device)
                for ik, tensor in v.items()
            }
        elif isinstance(v, tuple):
            # 今回の引数にはtupleは関係なさそうなので何もしない
            pass
    return obj


def move_tensor_to_gpu(
    tensor: torch.Tensor,
    cuda_device: int = 0,
):
    """_summary_
    1つのtorchをgpuに乗せる
    Args:
        tensor (torch.Tensor): _description_
        cuda_device (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    # print(type(tensor))
    # print(tensor)
    return tensor.cuda(cuda_device)


"""Update date: 2019-Nov-3"""


def pad_sequence_to_length(
    sequence: list, desired_length: int, default_value=lambda: 0
) -> list:
    padded_sequence = sequence[:desired_length]
    for _ in range(desired_length - len(padded_sequence)):
        padded_sequence.append(default_value())
    return padded_sequence


"""Update date: 2019-Nov-4"""


def set_cuda(args: SimpleArgumentParser) -> None:
    """_summary_
    cudaを準備する
    Args:
        args (_type_): _description_
    """
    cuda_device: list[int] = args.cuda_device
    args.cuda_device = [i for i in cuda_device if i >= 0]
    args.n_gpu = len(args.cuda_device)
    logger.info("Device: %s, n_gpu: %s" % (args.cuda_device, args.n_gpu))


"""Update date: 2019-Nov-3"""


def set_random_seed(args: SimpleArgumentParser) -> None:
    """_summary_
    乱数のシードを設定する。
    すでに設定されている場合は何もしない
    Args:
        args (_type_): _description_
    """
    if args.seed <= 0:
        logger.info("Does not set the random seed, since the value is %s" % args.seed)
        return
    random.seed(args.seed)
    np.random.seed(int(args.seed / 2))
    torch.manual_seed(int(args.seed / 4))
    torch.cuda.manual_seed_all(int(args.seed / 8))


"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py#sort_batch_by_length
Update date: 2019-Nov-5"""


def sort_batch_by_length(tensor, sequence_lengths):
    """restoration_indices: sorted_tensor.index_select(0, restoration_indices) == original_tensor"""
    assert isinstance(tensor, torch.Tensor) and isinstance(
        sequence_lengths, torch.Tensor
    )

    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(
        0, descending=True
    )
    sorted_tensor = tensor.index_select(0, permutation_index)

    index_range = torch.arange(0, len(sequence_lengths), device=sequence_lengths.device)
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)

    return (
        sorted_tensor,
        sorted_sequence_lengths,
        restoration_indices,
        permutation_index,
    )
