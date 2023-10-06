# cadecの形式のファイルを読み取り、
# sentenceからlist[instance]に変更しようとしている箇所だと思うが
# 実際に何をやっているかが少し怪しくどのような状態が正解かわからない
# & 日本語で同じことをやろうとすると形態素解析などを利用することになるし、
# 入出力の形式が異なる可能性があるため一度別に切り出して何が起こっているか、
# どうなってほしいかを検討する
# from code.xdai.utils.token_indexer import (
#     ELMoIndexer,
#     SingleIdTokenIndexer,
#     TokenCharactersIndexer,
# )
# from code.xdai.utils.instance import ActionField, Instance, MetadataField, TextField
# from code.xdai.ner.transition_discontinuous.parsing import Parser
from code.xdai.utils.token import Token
from code.xdai.ner.mention import Mention, merge_consecutive_indices

# import logging

# from code.xdai.utils.args import SimpleArgumentParser

# logger = logging.getLogger(__name__)


# def read(self, filepath: str, training: bool = False) -> list[Instance]:
#     """_summary_
#     ファイルを読みこみ、インスタンスのリストを返す
#     Args:
#         filepath (str): _description_
#         training (bool, optional): _description_. Defaults to False.

#     Returns:
#         _type_: _description_
#     """
#     instances: list[Instance] = []

#     with open(file=filepath, mode="r", encoding="utf-8") as f:
#         # for sentenec in f は危なくない？
#         # と思ったけどnextが使われているのでイテレータとして回している可能性がある
#         for sentence in f:
#             # strip split で取れるのは対象が英語だから
#             tokens: list[Token] = [Token(t, 0) for t in sentence.strip().split()]
#             # sentenceの次の行にannotationsがあるタイプのデータセット？
#             annotations = next(f).strip()

#             # tokenとaccnotationsからactionsを生成している
#             actions: list[str] = self.parse.mention2actions(
#                 mentions_str=annotations, sentence_length=len(tokens)
#             )
#             oracle_mentions = [
#                 str(s)
#                 for s in self.parse.parse(actions=actions, seq_length=len(tokens))
#             ]
#             # annotationsからmentionを作る
#             gold_mentions: list[str] = (
#                 annotations.split("|") if len(annotations) > 0 else []
#             )

#             # 「annotationsから作成したactions」を基に作成したmentionsと
#             # annotationsから作成したmentionsは一致するはず
#             # もし一致しなければそれは作成が間違っているということで学習時には捨てる
#             # 学習時じゃなければ評価の対象とするため利用する
#             if len(oracle_mentions) != len(gold_mentions) or len(
#                 oracle_mentions
#             ) != len(set(oracle_mentions) & set(gold_mentions)):
#                 logger.debug(
#                     "Discard this instance whose oracle mention is: %s, while its gold mention is: %s"
#                     % ("|".join(oracle_mentions), annotations)
#                 )
#                 if not training:
#                     instances.append(
#                         self._to_instance(
#                             sentence_str=sentence,
#                             annotations_str=annotations,
#                             tokens=tokens,
#                             actions=actions,
#                         )
#                     )
#             else:
#                 instances.append(
#                     self._to_instance(
#                         sentence_str=sentence,
#                         annotations_str=annotations,
#                         tokens=tokens,
#                         actions=actions,
#                     )
#                 )

#             assert len(next(f).strip()) == 0
#     return instances


class _NodeInStack(Mention):
    """_summary_
    stackの中のノード
    Args:
        Mention (_type_): _description_

    Returns:
        _type_: _description_
    """

    @classmethod
    def single_token_node(cls, idx: int) -> Mention:
        return Mention.create_mention(indices=[idx, idx], label="")

    @classmethod
    def merge_nodes(cls, m1: Mention, m2: Mention) -> Mention:
        # TODO: m1 and m2 cannot completely contain each other
        # TODO m1とm2のスパンを統合して、新しいスパンを作るっぽい
        indices: list[int] = sorted(m1.indices + m2.indices)
        # (多分)indicesの両端を取得して新しいスパンを作成する
        indices = merge_consecutive_indices(indices)
        return Mention.create_mention(indices=indices, label="")


def mention2actions(mentions_str: str, sentence_length: int) -> list[str]:
    """_summary_
    mentionをactionに変換するらしい
    Args:
        mentions (str): _description_
        sentence_length (int): _description_
    """

    def _detect_overlapping_mentions(mentions_str: str) -> list[Mention]:
        """_summary_
        # TODO 何やってるんだここ
        Args:
            mentions_str (str): _description_

        Returns:
            _type_: _description_
        """
        mentions: list[Mention] = Mention.create_mentions(mentions_str)

        for i in range(len(mentions)):
            if mentions[i]._overlapping:
                continue
            for j in range(len(mentions)):
                if i == j:
                    continue
                if Mention.overlap_spans(mentions[i], mentions[j]):
                    assert mentions[i].label == mentions[j].label
                    mentions[i]._overlapping = True
                    mentions[j]._overlapping = True
        return mentions

    def _involve_mention(mentions: list[Mention], token_id: int) -> bool:
        """_summary_
        token_idがどこかのmentionのどこかのスパンに含まれているか
        Args:
            mentions (list[Mention]): _description_
            token_id (int): _description_

        Returns:
            bool: _description_
        """
        for _, mention in enumerate(mentions):
            for span in mention.spans:
                if span.start <= token_id and token_id <= span.end:
                    return True
        return False

    def _find_relevant_mentions(
        mentions: list[Mention], node: Mention
    ) -> tuple[list[int], list[int]]:
        """_summary_
        parents = node が含まれている mention の index のリスト
        equals = node と等しい mention の index のリスト
        Args:
            mentions (list[Mention]): _description_
            node (Mention): _description_

        Returns:
            tuple[list[int], list[int]]: _description_
        """
        parents: list[int] = []
        equals: list[int] = []

        for i in range(len(mentions)):
            if Mention.contains(mentions[i], node):
                parents.append(i)
                if Mention.equal_spans(mentions[i], node):
                    equals.append(i)
        return parents, equals

    # ここから関数のメインの中身
    mentions: list[Mention] = _detect_overlapping_mentions(mentions_str=mentions_str)
    actions: list[str] = []
    stack: list[Mention] = []
    buffer: list[int] = [i for i in range(sentence_length)]

    while len(buffer) > 0:
        if not _involve_mention(mentions=mentions, token_id=buffer[0]):
            actions.append("OUT")
            buffer.pop(0)
        else:
            actions.append("SHIFT")
            stack.append(_NodeInStack.single_token_node(idx=buffer[0]))
            buffer.pop(0)

            stack_changed = True

            # COMPLETE, REDUCE, LEFT-REDUCE, RIGHT-REDUCE
            # if the last item of the stack is a mention, and does not involve with other mentions, then COMPLETE
            while stack_changed:
                stack_changed = False

                if len(stack) >= 1:
                    parents, equals = _find_relevant_mentions(mentions, stack[-1])
                    if len(equals) == 1 and len(parents) == 1:
                        actions.append("COMPLETE-%s" % mentions[equals[0]].label)
                        stack.pop(-1)
                        mentions.pop(equals[0])
                        stack_changed = True

                    # three REDUCE actions
                    if len(stack) >= 2:
                        if not Mention.overlap_spans(stack[-2], stack[-1]):
                            last_two_nodes = _NodeInStack.merge_nodes(
                                stack[-2], stack[-1]
                            )
                            parents_of_two, _ = _find_relevant_mentions(
                                mentions, last_two_nodes
                            )
                            if len(parents_of_two) > 0:
                                parent_of_left, _ = _find_relevant_mentions(
                                    mentions, stack[-2]
                                )
                                parent_of_right, _ = _find_relevant_mentions(
                                    mentions, stack[-1]
                                )
                                if len(parents_of_two) != len(parent_of_left):
                                    actions.append("LEFT-REDUCE")
                                    stack.pop(-1)
                                    stack.append(last_two_nodes)
                                    stack_changed = True
                                elif len(parents_of_two) != len(parent_of_right):
                                    actions.append("RIGHT-REDUCE")
                                    stack.pop(-2)
                                    stack.append(last_two_nodes)
                                    stack_changed = True
                                else:
                                    actions.append("REDUCE")
                                    stack.pop(-1)
                                    stack.pop(-1)
                                    stack.append(last_two_nodes)
                                    stack_changed = True
    return actions


if __name__ == "__main__":
    sentence = "Aches and pains , breathing difficulties , panic attacks , stength and stamina destroyed , palpatations , strong tingling in one hand , some tinititis and occassional ear aches ."
    # read(sentences=sentences)
    # sentence = "ab cd"
    tokens = [Token(t, 0) for t in sentence.strip().split()]
    print(tokens)

    annotations = (
        "0,2 ADR|4,5 ADR|7,8 ADR|10,13 ADR|15,15 ADR|17,21 ADR|24,24 ADR|27,28 ADR"
    )

    actions: list[str] = mention2actions(
        mentions_str=annotations, sentence_length=len(tokens)
    )

    print(actions)
