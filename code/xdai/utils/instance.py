from collections import defaultdict
from typing import Any

import torch
from xdai.utils.common import pad_sequence_to_length
from xdai.utils.token import Token
from xdai.utils.instance import ActionField, MetadataField, TextField
from xdai.utils.token_indexer import (
    ELMoIndexer,
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
)
from xdai.utils.vocab import Vocabulary

"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/fields/field.py
Update date: 2019-Nov-5"""


class _Field:
    def count_vocab_items(self, counter: dict[str, dict[str, int]]):
        raise NotImplementedError

    def index(self, vocab):
        raise NotImplementedError

    def get_padding_lengths(self):
        raise NotImplementedError

    def as_tensor(self, padding_lengths: dict[str, int]):
        raise NotImplementedError

    def batch_tensors(self, tensor_list):
        return torch.stack(tensor_list)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented


class ActionField:
    def __init__(self, actions: list[str], inputs: TextField):
        self._key = "actions"
        self.actions: list[str] = actions
        self._indexed_actions: list[int] | None = None
        self.inputs: TextField = inputs

        # TODO actionsがlist[int]ならって言っているけどそんなわけなくない？
        # if all([isinstance(a, int) for a in actions]):
        #     self._indexed_actions = actions

    def count_vocab_items(self, counter: dict[str, dict[str, int]]):
        if self._indexed_actions is None:
            for action in self.actions:
                counter[self._key][action] += 1

    def index(self, vocab: Vocabulary):
        if self._indexed_actions is None:
            self._indexed_actions = [
                vocab.get_item_index(action, self._key) for action in self.actions
            ]

    def get_padding_lengths(self) -> dict[str, int]:
        """_summary_
        paddingの長さを取得する
        Returns:
            _type_: _description_
        """
        return {"num_tokens": self.inputs.sequence_length() * 2}

    def as_tensor(self, padding_lengths: dict[str, int]):
        if self._indexed_actions is None:
            print("self._indexed_actionsがNoneです")
            return
        desired_num_actions = padding_lengths["num_tokens"]
        padded_actions = pad_sequence_to_length(
            self._indexed_actions, desired_num_actions
        )
        return torch.LongTensor(padded_actions)

    def batch_tensors(self, tensor_list):
        return torch.stack(tensor_list)


"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/fields/metadata_field.py
Update date: 2019-Nov-5"""


class MetadataField(_Field):
    def __init__(self, metadata):
        self.metadata = metadata

    def __getitem__(self, key):
        try:
            return self.metadata[key]
        except TypeError:
            raise TypeError("Metadata is not a dict.")

    def __iter__(self):
        try:
            return iter(self.metadata)
        except TypeError:
            raise TypeError("Metadata is not iterable.")

    def __len__(self):
        try:
            return len(self.metadata)
        except TypeError:
            raise TypeError("Metadata has no length.")

    def get_padding_lengths(self) -> dict[str, int]:
        return {}

    def as_tensor(self, padding_lengths):
        return self.metadata

    def empty_field(self):
        return MetadataField(None)

    @classmethod
    def batch_tensors(cls, tensor_list):
        return tensor_list


"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py#batch_tensor_dicts
Update date: 2019-Nov-5"""


def _batch_tensor_dicts(tensor_dicts):
    """takes a list of tensor dictionaries, returns a single dictionary with all tensors with the same key batched"""
    key_to_tensors = defaultdict(list)
    for tensor_dict in tensor_dicts:
        for key, tensor in tensor_dict.items():
            key_to_tensors[key].append(tensor)

    batched_tensors = {}
    for key, tensor_list in key_to_tensors.items():
        batched_tensor = torch.stack(tensor_list)
        batched_tensors[key] = batched_tensor
    return batched_tensors


"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/fields/text_field.py
Update date: 2019-Nov-5"""


class TextField(_Field):
    def __init__(
        self,
        tokens: list[Token],
        token_indexers: dict[
            str, SingleIdTokenIndexer | TokenCharactersIndexer | ELMoIndexer
        ],
    ):
        self.tokens: list[Token] = tokens
        self._token_indexers: dict[
            str, SingleIdTokenIndexer | TokenCharactersIndexer | ELMoIndexer
        ] = token_indexers
        # 以下の3つは初期値が None だが、必ずしもその必要はないはずなので辞書型だと仮定する
        self._indexed_tokens: dict[str, list[list[int]]] = {}
        self._indexer_name_to_indexed_token: dict[str, list[str]] = {}
        self._token_index_to_indexer_name: dict[str, str] = {}

    def __iter__(self):
        return iter(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]

    def __len__(self):
        return len(self.tokens)

    def count_vocab_items(self, counter: dict[str, dict[str, int]]):
        """_summary_
        self.tokensに含まれる各token.textについて
        引数に与えたcounterが扱う単位でその(語や文字単位での)出現回数を加算する
        Args:
            counter (_type_): _description_
        """
        for indexer in self._token_indexers.values():
            for token in self.tokens:
                indexer.count_vocab_items(token, counter)

    def index(self, vocab: Vocabulary) -> None:
        token_arrays: dict[str, list[list[int]]] = {}
        indexer_name_to_indexed_token: dict[str, list[str]] = {}
        token_index_to_indexer_name: dict[str, str] = {}

        for indexer_name, indexer in self._token_indexers.items():
            token_indices: dict[str, list[list[int]]] = indexer.tokens_to_indices(
                self.tokens, vocab, indexer_name
            )
            token_arrays.update(token_indices)
            indexer_name_to_indexed_token[indexer_name] = list(token_indices.keys())
            for token_index in token_indices:
                token_index_to_indexer_name[token_index] = indexer_name
        self._indexed_tokens: dict[str, list[list[int]]] = token_arrays
        self._indexer_name_to_indexed_token: dict[
            str, list[str]
        ] = indexer_name_to_indexed_token
        self._token_index_to_indexer_name: dict[str, str] = token_index_to_indexer_name

    def get_padding_lengths(self) -> dict[str, int]:
        """_summary_
        (多分だけど)各tokenについてpaddingの長さを求めている(?)
        Returns:
            dict[str, int]: _description_
        """
        lengths: list[dict[str, int]] = []
        assert (
            self._indexed_tokens is not None
        ), "Call .index(vocabulary) before determining padding lengths."

        for indexer_name, indexer in self._token_indexers.items():
            indexer_lengths: dict[str, int] = {}
            for indexed_tokens_key in self._indexer_name_to_indexed_token[indexer_name]:
                token_lengths: list[dict[str, int]] = [
                    indexer.get_padding_lengths(token)
                    for token in self._indexed_tokens[indexed_tokens_key]
                ]
                if not token_lengths:
                    token_lengths = [indexer.get_padding_lengths([])]
                for key in token_lengths[0]:
                    indexer_lengths[key] = max(
                        x[key] if key in x else 0 for x in token_lengths
                    )
            lengths.append(indexer_lengths)

        padding_lengths: dict[str, int] = {}
        num_tokens: set[int] = set()
        for token_index, token_list in self._indexed_tokens.items():
            indexer_name = self._token_index_to_indexer_name[token_index]
            indexer = self._token_indexers[indexer_name]
            padding_lengths[f"{token_index}_length"] = max(
                len(token_list), indexer.get_token_min_padding_length()
            )
            num_tokens.add(len(token_list))
        padding_lengths["num_tokens"] = max(num_tokens)

        padding_keys: set[str] = {key for d in lengths for key in d.keys()}
        for padding_key in padding_keys:
            padding_lengths[padding_key] = max(
                x[padding_key] if padding_key in x else 0 for x in lengths
            )
        return padding_lengths

    def sequence_length(self) -> int:
        return len(self.tokens)

    def as_tensor(self, padding_lengths: dict[str, int]):
        tensors = {}
        for indexer_name, indexer in self._token_indexers.items():
            desired_num_tokens: dict[str, int] = {
                indexed_tokens_key: padding_lengths[f"{indexed_tokens_key}_length"]
                for indexed_tokens_key in self._indexer_name_to_indexed_token[
                    indexer_name
                ]
            }
            indices_to_pad: dict[str, list[list[int]]] = {
                indexed_tokens_key: self._indexed_tokens[indexed_tokens_key]
                for indexed_tokens_key in self._indexer_name_to_indexed_token[
                    indexer_name
                ]
            }
            padded_array = indexer.pad_token_sequence(
                indices_to_pad, desired_num_tokens, padding_lengths
            )
            indexer_tensors = {
                key: torch.LongTensor(array) for key, array in padded_array.items()
            }
            tensors.update(indexer_tensors)
        return tensors

    def empty_field(self) -> TextField:
        text_field = TextField([], self._token_indexers)
        text_field._indexed_tokens = {}
        text_field._indexer_name_to_indexed_token = {}
        for indexer_name, indexer in self._token_indexers.items():
            array_keys = indexer.get_keys(indexer_name)
            for key in array_keys:
                text_field._indexed_tokens[key] = []
            text_field._indexer_name_to_indexed_token[indexer_name] = array_keys
        return text_field

    def batch_tensors(self, tensor_dicts):
        return _batch_tensor_dicts(tensor_dicts)

    """Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py#get_text_field_mask"""

    @classmethod
    def get_text_field_mask(cls, text_field_tensors: dict[str, torch.Tensor]):
        if "mask" in text_field_tensors:
            return text_field_tensors["mask"]

        tensor_dims = [(tensor.dim(), tensor) for tensor in text_field_tensors.values()]
        tensor_dims.sort(key=lambda x: x[0])

        assert tensor_dims[0][0] == 2

        token_tensor = tensor_dims[0][1]
        return (token_tensor != 0).long()


"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/instance.py
An instance is a collection of Field objects, specifying the inputs and outputs to the model.
Update date: 2019-Nov-5"""


class Instance:
    """_summary_
    fields: dict[str, TextField | ActionField | MetadataField]
    self.indexed: bool
    """

    def __init__(self, fields: dict[str, TextField | ActionField | MetadataField]):
        self.fields: dict[str, TextField | ActionField | MetadataField] = fields
        self.indexed: bool = False

    def __getitem__(self, key: str):
        return self.fields[key]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self):
        return len(self.fields)

    def add_field(self, field_name: str, field, vocab):
        """_summary_
        フィールドを追加する
        keyがstr型なのはそうとして、valueの型が分からない
        Args:
            field_name (str): _description_
            field (_type_): _description_
            vocab (_type_): _description_
        """
        self.fields[field_name] = field
        if self.indexed:
            field.index(vocab)

    def count_vocab_items(self, counter: dict[str, dict[str, int]]):
        """_summary_
        vocabのitemの出現回数を加算する
        Args:
            counter (_type_): _description_
        """
        for field in self.fields.values():
            field.count_vocab_items(counter)

    def index_fields(self, vocab):
        """_summary_
        TODO これは何？
        Args:
            vocab (_type_): _description_
        """
        if not self.indexed:
            self.indexed = True
            for field in self.fields.values():
                field.index(vocab)

    def get_padding_lengths(self) -> dict[str, dict[str, int]]:
        """_summary_
        TODO padding の長さと言っているが、それにしては戻り値が複雑すぎるので何をしているかが分からない
        Returns:
            _type_: _description_
        """
        lengths: dict[str, dict[str, int]] = {}
        for field_name, field in self.fields.items():
            lengths[field_name] = field.get_padding_lengths()
        return lengths

    def as_tensor_dict(self, padding_lengths):
        """_summary_
        # TODO なにこれ
        Args:
            padding_lengths (_type_): _description_

        Returns:
            _type_: _description_
        """
        padding_lengths = padding_lengths or self.get_padding_lengths()
        tensors = {}
        for field_name, field in self.fields.items():
            tensors[field_name] = field.as_tensor(padding_lengths[field_name])
        return tensors
