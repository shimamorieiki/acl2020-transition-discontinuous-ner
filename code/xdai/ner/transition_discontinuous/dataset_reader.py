from xdai.utils.token_indexer import (
    ELMoIndexer,
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
)
from xdai.utils.instance import ActionField, Instance, MetadataField, TextField
from xdai.ner.transition_discontinuous.parsing import Parser
from xdai.utils.token import Token
import logging

logger = logging.getLogger(__name__)


class DatasetReader:
    """_summary_
    データセットをロードするクラス
    """

    def __init__(self, args):
        """_summary_
        初期化する
        Args:
            args (_type_): _description_
        """
        self.args = args
        self.parse: Parser = Parser()
        self._token_indexers: dict[
            str, SingleIdTokenIndexer | TokenCharactersIndexer | ELMoIndexer
        ] = {
            "tokens": SingleIdTokenIndexer(),
            "token_characters": TokenCharactersIndexer(),
        }

        if args.model_type == "elmo":
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
        instances: list[Instance] = []

        with open(file=filepath, mode="r", encoding="utf-8") as f:
            # for sentenec in f は危なくない？
            # と思ったけどnextが使われているのでイテレータとして回している可能性がある
            # sentences = f.readlines()
            for sentence in f:
                # strip split で取れるのは対象が英語だから
                tokens: list[Token] = [Token(t) for t in sentence.strip().split()]
                # sentenceの次の行にannotationsがあるタイプのデータセット？
                annotations = next(f).strip()

                # tokenとaccnotationsからactionsを生成している
                actions: list[str] = self.parse.mention2actions(
                    mentions_str=annotations, sentence_length=len(tokens)
                )
                oracle_mentions = [
                    str(s)
                    for s in self.parse.parse(actions=actions, seq_length=len(tokens))
                ]
                # annotationsからmentionを作る
                gold_mentions: list[str] = (
                    annotations.split("|") if len(annotations) > 0 else []
                )

                # 「annotationsから作成したactions」を基に作成したmentionsと
                # annotationsから作成したmentionsは一致するはず
                # もし一致しなければそれは作成が間違っているということで学習時には捨てる
                # 学習時じゃなければ評価の対象とするため利用する
                if len(oracle_mentions) != len(gold_mentions) or len(
                    oracle_mentions
                ) != len(set(oracle_mentions) & set(gold_mentions)):
                    logger.debug(
                        "Discard this instance whose oracle mention is: %s, while its gold mention is: %s"
                        % ("|".join(oracle_mentions), annotations)
                    )
                    if not training:
                        instances.append(
                            self._to_instance(
                                sentence_str=sentence,
                                annotations_str=annotations,
                                tokens=tokens,
                                actions=actions,
                            )
                        )
                else:
                    instances.append(
                        self._to_instance(
                            sentence_str=sentence,
                            annotations_str=annotations,
                            tokens=tokens,
                            actions=actions,
                        )
                    )

                assert len(next(f).strip()) == 0
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
        text_fields: TextField = TextField(tokens, self._token_indexers)
        action_fields: ActionField = ActionField(actions, text_fields)
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
