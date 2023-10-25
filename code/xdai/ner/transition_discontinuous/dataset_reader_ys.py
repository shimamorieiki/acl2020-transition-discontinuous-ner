from xdai.utils.token_indexer import (
    ELMoIndexer,
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
)
from xdai.utils.instance import ActionField, Instance, MetadataField, TextField
from xdai.ner.transition_discontinuous.parsing import Parser
from xdai.utils.token import Token
import logging
import json

logger = logging.getLogger(__name__)


class DatasetReaderYS:
    """_summary_
    YSのデータセットを読み込むクラス
    """

    def __init__(self, model_type: str):
        """_summary_
        初期化する
        Args:
            args (_type_): _description_
        """
        # parserを用意する
        self.parse: Parser = Parser()
        self._token_indexers: dict[
            str, SingleIdTokenIndexer | TokenCharactersIndexer | ELMoIndexer
        ] = {
            "tokens": SingleIdTokenIndexer(),
            "token_characters": TokenCharactersIndexer(),
        }

        if model_type == "elmo":
            self._token_indexers["elmo_characters"] = ELMoIndexer()

    def read(self, filepath: str, training: bool = False) -> list[Instance]:
        """_summary_
        ファイルを読みこみ、インスタンスのリストを返す
        Args:
            filepath (str): _description_
            training (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        # count = 0
        instances: list[Instance] = []

        with open(filepath, mode="r", encoding="utf-8") as f:
            jsonl_data = [json.loads(l) for l in f.readlines()]
            for jsonl in jsonl_data:
                wakati = jsonl["wakati"]
                labels = jsonl["label"]
                tokens: list[Token] = [Token(t) for t in wakati]
                actions: list[str] = jsonl["actions"]

                # ラベル間の不均衡を緩和するため、
                # OUT のラベルしか含まれていないデータを利用しないようにする
                if all(action == "OUT" for action in actions):
                    continue

                # 元の文に復元したもの
                sentence = "".join(wakati)

                # 本当は1つの要素が複数spanに分かれているものも対応したいが
                # 現在のアノテーションではそれは不可能なので
                # spanに分かれているものは扱わない
                annotations: str = "|".join(
                    [f"{label[0]},{label[1]} {label[2]}" for label in labels]
                )

                # 本当はここで、パースの結果得られたものが正しく復元できるかを試したいが
                # 今回は、あらかじめパースに成功したもののみをデータに利用しているので
                # 特にその分の処理は記述しないものとする。
                # 必要が生じたときにその分の処理をここに追加する

                instances.append(
                    self._to_instance(
                        sentence_str=sentence,
                        annotations_str=annotations,
                        tokens=tokens,
                        actions=actions,
                    )
                )
                # count += 1
                # if count >= 300:
                #     break

        return instances

    def _to_instance(
        self,
        sentence_str: str,
        annotations_str: str,
        tokens: list[Token],
        actions: list[str],
    ) -> Instance:
        """_summary_
        Instanceクラスのオブジェクトを生成する
        Args:
            sentence_str (str): _description_
            annotations_str (str): _description_
            tokens (list[Token]): _description_
            actions (list[str]): _description_

        Returns:
            _type_: _description_
        """
        text_fields: TextField = TextField(
            tokens=tokens, token_indexers=self._token_indexers
        )
        action_fields: ActionField = ActionField(actions=actions, inputs=text_fields)
        sentence: MetadataField = MetadataField(metadata=sentence_str.strip())
        annotations: MetadataField = MetadataField(metadata=annotations_str.strip())

        return Instance(
            {
                "sentence": sentence,
                "annotations": annotations,
                "tokens": text_fields,
                "actions": action_fields,
            }
        )
