"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/vocabulary.py
Update date: 2019-Nov-5"""
from __future__ import annotations

import codecs
import logging
import os
from collections import defaultdict

# from xdai.utils.instance import Instance
from tqdm import tqdm

logger = logging.getLogger(__name__)


DEFAULT_NON_PADDED_NAMESPACES = ["tags", "labels"]


class _NamespaceDependentDefaultDict(defaultdict):
    def __init__(self, padded_function, non_padded_function):
        # we do not take non_padded_namespaces as a parameter,
        # because we consider any namespace whose key name ends with labels or tags as non padded namespace,
        # and use padded namespace otherwise

        # lambda関数
        self._padded_function = padded_function
        # lambda関数
        self._non_padded_function = non_padded_function

        super(_NamespaceDependentDefaultDict, self).__init__()

    def __missing__(self, key: str) -> dict:
        """_summary_
        keyに対応するものが見つからなかった場合
        Args:
            key (str): _description_

        Returns:
            _type_: _description_
        """
        if any(key.endswith(pattern) for pattern in DEFAULT_NON_PADDED_NAMESPACES):
            value = self._non_padded_function()
        else:
            value = self._padded_function()
        dict.__setitem__(self, key, value)
        return value


class _ItemToIndexDefaultDict(_NamespaceDependentDefaultDict):
    """_summary_
    dict[str,dict[str,int]]
    Args:
        _NamespaceDependentDefaultDict (_type_): _description_
    """

    def __init__(self, padding_item: str, oov_item: str):
        super(_ItemToIndexDefaultDict, self).__init__(
            padded_function=lambda: {padding_item: 0, oov_item: 1},
            non_padded_function=lambda: {},
        )


class _IndexToItemDefaultDict(_NamespaceDependentDefaultDict):
    """_summary_
    dict[str,dict[int,str]]
    Args:
        _NamespaceDependentDefaultDict (_type_): _description_
    """

    def __init__(self, padding_item: str, oov_item: str):
        super(_IndexToItemDefaultDict, self).__init__(
            padded_function=lambda: {0: padding_item, 1: oov_item},
            non_padded_function=lambda: {},
        )


class Vocabulary:
    def __init__(
        self,
        counter: dict[str, dict[str, int]] | None = None,
        min_count: dict[str, int] | None = None,
        max_vocab_size: dict[str, int] | None = None,
    ):
        self._padding_item = "@@PADDING@@"
        self._oov_item = "@@UNKNOWN@@"
        self._item_to_index = _ItemToIndexDefaultDict(
            self._padding_item, self._oov_item
        )
        self._index_to_item = _IndexToItemDefaultDict(
            self._padding_item, self._oov_item
        )
        self._extend(
            counter=counter, min_count=min_count, max_vocab_size=max_vocab_size
        )

    """Update date: 2019-Nov-9"""

    def save_to_files(self, directory: str):
        """_summary_
        引数に指定した directory に語彙の情報を保存する
        Args:
            directory (str): _description_
        """
        os.makedirs(directory, exist_ok=True)
        for namespace, mapping in self._index_to_item.items():
            with codecs.open(
                os.path.join(directory, "%s.txt" % namespace), "w", "utf-8"
            ) as f:
                for i in range(len(mapping)):
                    f.write("%s\n" % (mapping[i].replace("\n", "@@NEWLINE@@").strip()))

    """Update date: 2019-Nov-9"""

    @classmethod
    def from_files(cls, directory: str):
        logger.info("Loading item dictionaries from %s.", directory)
        vocab = cls()
        for namespace in os.listdir(directory):
            if not namespace.endswith(".txt"):
                continue
            with codecs.open(os.path.join(directory, namespace), "r", "utf-8") as f:
                namespace = namespace.replace(".txt", "")
                for i, line in enumerate(f):
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    item = line.replace("@@NEWLINE@@", "\n")
                    vocab._item_to_index[namespace][item] = i
                    vocab._index_to_item[namespace][i] = item
        return vocab

    @staticmethod
    def from_instances(
        instances: list,  # list[Instance]なんだけど循環importのエラーが出るので仕方なく
        min_count: dict[str, int] | None = None,
        max_vocab_size: dict[str, int] | None = None,
    ) -> Vocabulary:
        """_summary_
        instanceから
        Args:
            instances (_type_): _description_
            min_count (_type_, optional): _description_. Defaults to None.
            max_vocab_size (_type_, optional): _description_. Defaults to None.

        Returns:
            Vocabulary: _description_
        """
        counter: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for instance in tqdm(instances):
            # 関数の引数にdictを与えてごにょごにょすることで、
            # 副作用的にcounterの値を更新している
            instance.count_vocab_items(counter)
        return Vocabulary(
            counter=counter, min_count=min_count, max_vocab_size=max_vocab_size
        )

    def _extend(
        self,
        counter,
        min_count: dict[str, int] | None = None,
        max_vocab_size: dict[str, int] | None = None,
    ):
        """_summary_

        Args:
            counter (_type_): _description_
            min_count (dict[str, int] | None, optional): _description_. Defaults to None.
            max_vocab_size (dict[str, int] | None, optional): _description_. Defaults to None.
        """
        counter = counter or {}
        min_count = min_count or {}
        max_vocab_size = max_vocab_size or {}
        for namespace in counter:
            item_counts = list(counter[namespace].items())
            item_counts.sort(key=lambda x: x[1], reverse=True)

            if namespace in max_vocab_size and max_vocab_size[namespace] > 0:
                item_counts = item_counts[: max_vocab_size[namespace]]
            for item, count in item_counts:
                if count >= min_count.get(namespace, 1):
                    self._add_item_to_namespace(item, namespace)

    def _add_item_to_namespace(self, item, namespace="tokens"):
        """_summary_

        Args:
            item (_type_): _description_
            namespace (str, optional): _description_. Defaults to "tokens".
        """
        if item not in self._item_to_index[namespace]:
            idx = len(self._item_to_index[namespace])
            self._item_to_index[namespace][item] = idx
            self._index_to_item[namespace][idx] = item

    def get_index_to_item_vocabulary(self, namespace="tokens"):
        """_summary_

        Args:
            namespace (str, optional): _description_. Defaults to "tokens".

        Returns:
            _type_: _description_
        """
        return self._index_to_item[namespace]

    def get_item_to_index_vocabulary(self, namespace: str = "tokens") -> dict[str, int]:
        """_summary_
        Args:
            namespace (str, optional): _description_. Defaults to "tokens".

        Returns:
            _type_: _description_
        """
        return self._item_to_index[namespace]

    def get_item_index(self, item: str, namespace: str = "tokens") -> int:
        """_summary_
        namespaceのitemのインデックスを取得する
        Args:
            item (_type_): _description_
            namespace (str, optional): _description_. Defaults to "tokens".

        Returns:
            _type_: _description_
        """
        if item in self._item_to_index[namespace]:
            return self._item_to_index[namespace][item]
        else:
            return self._item_to_index[namespace][self._oov_item]

    def get_item_from_index(self, idx: int, namespace: str = "tokens") -> str:
        """_summary_
        namespaceのidxのitemを取得する
        Args:
            idx (int): _description_
            namespace (str, optional): _description_. Defaults to "tokens".

        Returns:
            _type_: _description_
        """
        return self._index_to_item[namespace][idx]

    def get_vocab_size(self, namespace: str = "tokens") -> int:
        """_summary_
        語彙のサイズを取得する
        Args:
            namespace (str, optional): _description_. Defaults to "tokens".

        Returns:
            _type_: _description_
        """
        return len(self._item_to_index[namespace])
