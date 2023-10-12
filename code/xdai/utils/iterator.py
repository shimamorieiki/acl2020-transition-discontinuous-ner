from __future__ import annotations
import itertools
import math
import random
from collections import defaultdict
from typing import Iterable, cast, Generator, Any
from xdai.utils.vocab import Vocabulary
from xdai.utils.instance import Instance
import torch

"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset.py
Update date: 2019-Nov-18"""


class Batch(Iterable):
    def __init__(self, instances: list[Instance]):
        super().__init__()
        self.instances: list[Instance] = instances

    """return: 
        {
            'tokens': {
                'tokens_length': 45,
                'token_characters_length': 45,
                'elmo_characters_length': 45, 
                'num_token_characters': 15
            }, 
            'tags': {'num_tokens': 45}
        }
    """

    def get_padding_lengths(self) -> defaultdict[str, dict[str, int]]:
        """_summary_
        paddingの長さを取得するらしいが、命名のわりに戻り値の型が複雑なので多分まだ検討することがある
        Returns:
            defaultdict[str, dict[str, int]]: _description_
        """
        padding_lengths: dict[str, dict[str, int]] = defaultdict(dict)

        all_instance_lengths: list[dict[str, dict[str, int]]] = [
            i.get_padding_lengths() for i in self.instances
        ]

        all_field_lengths: dict[str, list[dict[str, int]]] = defaultdict(list)
        for instance_lengths in all_instance_lengths:
            for field_name, instance_field_lengths in instance_lengths.items():
                all_field_lengths[field_name].append(instance_field_lengths)

        for field_name, field_lengths in all_field_lengths.items():
            for padding_key in field_lengths[0].keys():
                max_value = max(x.get(padding_key, 0) for x in field_lengths)
                padding_lengths[field_name][padding_key] = max_value

        return padding_lengths

    def as_tensor_dict(
        self, padding_lengths: dict[str, dict[str, int]] = {}
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor | list[torch.Tensor]]:
        if padding_lengths is None:
            padding_lengths = defaultdict(dict)
        instance_padding_lengths: dict[str, dict[str, int]] = self.get_padding_lengths()
        lengths_to_use: dict[str, dict[str, int]] = defaultdict(dict)
        for field_name, instance_field_lengths in instance_padding_lengths.items():
            for padding_key in instance_field_lengths.keys():
                if padding_key in padding_lengths[field_name]:
                    lengths_to_use[field_name][padding_key] = padding_lengths[
                        field_name
                    ][padding_key]
                else:
                    lengths_to_use[field_name][padding_key] = instance_field_lengths[
                        padding_key
                    ]

        field_tensors: dict[str, list] = defaultdict(list)
        for instance in self.instances:
            for field, tensors in instance.as_tensor_dict(lengths_to_use).items():
                field_tensors[field].append(tensors)

        field_classes = self.instances[0].fields
        final_fields: dict[
            str, dict[str, torch.Tensor] | torch.Tensor | list[torch.Tensor]
        ] = {}
        for field_name, field_tensor_list in field_tensors.items():
            final_fields[field_name] = field_classes[field_name].batch_tensors(
                field_tensor_list
            )
        return final_fields

    def __iter__(self):
        return iter(self.instances)

    def index_instances(self, vocab: Vocabulary):
        for instance in self.instances:
            instance.index_fields(vocab)


"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/iterators/data_iterator.py
Update date: 2019-Nov-18"""


class _Iterator:
    def __init__(
        self,
        batch_size: int = 64,
        max_instances_in_memory: int | None = None,
        cache_instances: bool = False,
    ):
        self.vocab: Vocabulary | None = None
        self._batch_size: int = batch_size
        self._max_instances_in_memory: int | None = max_instances_in_memory
        self._cache_instances: bool = cache_instances
        self._cache: dict[
            int,
            list[
                dict[str, dict[str, torch.Tensor] | torch.Tensor | list[torch.Tensor]]
            ],
        ] = defaultdict(list)

    def __call__(
        self, instances: list[Instance], shuffle: bool = True
    ) -> Generator[
        dict[str, dict[str, torch.Tensor] | torch.Tensor | list[torch.Tensor]],
        dict[str, dict[str, torch.Tensor] | torch.Tensor | list[torch.Tensor]],
        None,
    ]:
        """_summary_

        Args:
            instances (_type_): _description_
            shuffle (bool, optional): _description_. Defaults to True.

        Yields:
            _type_: _description_
        """
        key: int = id(instances)

        if self._cache_instances and key in self._cache:
            tensor_dicts: list = self._cache[key]
            if shuffle:
                random.shuffle(tensor_dicts)
            for tensor_dict in tensor_dicts:
                yield tensor_dict
        else:
            # TODO create_batchesの型を明確にする
            batches = self._create_batches(instances, shuffle)
            add_to_cache: bool = self._cache_instances and key not in self._cache
            for batch in batches:
                if self.vocab is not None:
                    batch.index_instances(self.vocab)
                padding_lengths: dict[str, dict[str, int]] = batch.get_padding_lengths()
                tensor_dict: dict[
                    str, dict[str, torch.Tensor] | torch.Tensor | list[torch.Tensor]
                ] = batch.as_tensor_dict(padding_lengths)
                if add_to_cache:
                    self._cache[key].append(tensor_dict)
                yield tensor_dict

    def _take_instances(self, instances: list[Instance]):
        yield from iter(instances)

    def _memory_sized_lists(self, instances: list[Instance]):
        lazy = not isinstance(instances, list)
        iterator = self._take_instances(instances)
        if lazy and self._max_instances_in_memory is None:
            yield from _Iterator.lazy_group_of(iterator, self._batch_size)
        elif self._max_instances_in_memory is not None:
            yield from _Iterator.lazy_group_of(iterator, self._max_instances_in_memory)
        else:
            yield instances

    def _ensure_batch_is_sufficiently_small(self, batch_instances):
        return [list(batch_instances)]

    def get_num_batches(self, instances: list[Instance]) -> int:
        return math.ceil(len(instances) / self._batch_size)

    def index_with(self, vocab: Vocabulary) -> None:
        """_summary_
        vocabの値をインスタンス変数に追加する
        Args:
            vocab (Vocabulary): _description_
        """
        self.vocab = vocab

    def _create_batches(
        self, instances: list[Instance], shuffle: bool = True
    ) -> Generator[Batch, Any, None]:
        raise NotImplementedError

    @staticmethod
    def lazy_group_of(iterator, group_size: int | None):
        return iter(lambda: list(itertools.islice(iterator, 0, group_size)), [])


"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/iterators/basic_iterator.py
Update date: 2019-Nov-19"""


class BasicIterator(_Iterator):
    def _create_batches(
        self, instances: list[Instance], shuffle: bool = True
    ) -> Generator[Batch, Any, None]:
        for instance_list in self._memory_sized_lists(instances):
            if shuffle:
                random.shuffle(instance_list)
            iterator = iter(instance_list)
            for batch_instances in _Iterator.lazy_group_of(iterator, self._batch_size):
                for (
                    possibly_smaller_batches
                ) in self._ensure_batch_is_sufficiently_small(batch_instances):
                    yield Batch(possibly_smaller_batches)


"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/common/util.py#add_noise_to_dict_values
Update date: 2019-Nov-19"""


def _add_noise_to_dict_values(dictionary, noise_param):
    dict_with_noise = {}
    for key, value in dictionary.items():
        noise_value = value * noise_param
        noise = random.uniform(-noise_value, noise_value)
        dict_with_noise[key] = value + noise
    return dict_with_noise


"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/iterators/bucket_iterator.py#sort_by_padding
Update date: 2019-Nov-19"""


def _sort_by_padding(
    instances, sorting_keys: list[tuple[str, str]], vocab, padding_noise=0.1
):
    instances_with_lengths = []
    for instance in instances:
        instance.index_fields(vocab)
        padding_lengths = cast(
            dict[str, dict[str, float]], instance.get_padding_lengths()
        )

        if padding_noise > 0.0:
            noisy_lengths = {}
            for field_name, field_lengths in padding_lengths.items():
                noisy_lengths[field_name] = _add_noise_to_dict_values(
                    field_lengths, padding_noise
                )
            padding_lengths = noisy_lengths

        instance_with_lengths = (
            [
                padding_lengths[field_name][padding_key]
                for (field_name, padding_key) in sorting_keys
            ],
            instance,
        )
        instances_with_lengths.append(instance_with_lengths)
    instances_with_lengths.sort(key=lambda x: x[0])
    sorted_instances = [
        instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths
    ]
    return sorted_instances


"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/iterators/bucket_iterator.py
Update date: 2019-Nov-19"""


class BucketIterator(_Iterator):
    def __init__(
        self,
        sorting_keys: list[tuple[str, str]],
        padding_noise: float = 0.1,
        batch_size: int = 64,
        biggest_batch_first: bool = False,
        max_instances_in_memory: int | None = None,
        cache_instances=False,
    ):
        super(BucketIterator, self).__init__(
            batch_size=batch_size,
            max_instances_in_memory=max_instances_in_memory,
            cache_instances=cache_instances,
        )
        self._sorting_keys = sorting_keys
        self._padding_noise = padding_noise
        self._biggest_batch_first = biggest_batch_first

    def _create_batches(self, instances, shuffle: bool = True):
        """_summary_
        batchを作るということなんだと思う
        Args:
            instances (_type_): _description_
            shuffle (bool, optional): _description_. Defaults to True.

        Yields:
            _type_: _description_
        """
        for instance_list in self._memory_sized_lists(instances):
            instance_list = _sort_by_padding(
                instance_list, self._sorting_keys, self.vocab, self._padding_noise
            )
            batches = []
            for batch_instances in _Iterator.lazy_group_of(
                iter(instance_list), self._batch_size
            ):
                for (
                    possibly_smaller_batches
                ) in self._ensure_batch_is_sufficiently_small(batch_instances):
                    batches.append(Batch(possibly_smaller_batches))

            move_to_front = self._biggest_batch_first and len(batches) > 1

            # move_to_front and shuffle の場合は末尾2つ以外をシャッフルし、これらを先頭に持っていく
            # move_to_front and !shuffle の場合は末尾2つを先頭に持っていく
            # !move_to_front and shuffle の場合は全体をシャッフルする
            # !move_to_front and !shuffle の場合は何もしない
            if move_to_front:
                last_batch = batches.pop()
                penultimate_batch = batches.pop()
                if shuffle:
                    random.shuffle(batches)
                batches.insert(0, penultimate_batch)
                batches.insert(0, last_batch)
            else:
                if shuffle:
                    random.shuffle(batches)
            yield from batches
