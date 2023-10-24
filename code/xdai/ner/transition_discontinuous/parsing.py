import logging

from xdai.ner.mention import Mention, merge_consecutive_indices

logger = logging.getLogger(__name__)


"""Update date: 2019-Nov-5"""


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


"""Update date: 2019-Nov-5"""


class Parser(object):
    """_summary_
    パーサーのクラス
    Args:
        object (_type_): _description_

    Returns:
        _type_: _description_
    """

    # 元の論文では 引数 actions の要素は
    # [SHIFT, OUT, COMPLETE-Y, REDUCE, LEFT-REDUCE, RIGHT-REDUCE] のいずれか
    # 今回新しく定義したものでは引数actionsの要素は
    # [SHIFT, OUT, COMPLETE, COMPLETE-RIGHT, REDUCE, REDUCE-LR] のいずれか
    def parse(
        self, actions: list[str], seq_length: int | None = None
    ) -> list[tuple[Mention, Mention]]:
        """_summary_
        actionのリストを受け取り、メンション範囲を確定させる
        Args:
            actions (_type_): _description_
            seq_length (_type_, optional): _description_. Defaults to None.

        Returns:
            List[Mention]: _description_
        """
        mentions: list[tuple[Mention, Mention]] = []
        stack: list[Mention] = []

        # もしseq_lengthが設定されていなかったらactionの2倍を長さとして設定する
        if seq_length is None:
            seq_length = len(actions) * 2

        buffer: list[int] = [i for i in range(seq_length)]

        for action in actions:
            # print("---------------------------------------------------")
            # print(f"action: ", action)
            # print(f"mentions: ", mentions)
            # print("stack: ")
            # for s in stack:
            #     print("spans: ", [str(span) for span in s.spans])
            #     print("label: ", s.label)
            # print(f"buffer: ", buffer)
            # print("---------------------------------------------------")
            # SHIFTのとき
            if action == "SHIFT":
                if len(buffer) < 1:
                    logger.info("Invalid SHIFT action: the buffer is empty.")
                else:
                    stack.append(_NodeInStack.single_token_node(idx=buffer[0]))
                    buffer.pop(0)
            # OUTのとき
            elif action == "OUT":
                if len(buffer) < 1:
                    logger.info("Invalid OUT action: the buffer is empty.")
                else:
                    buffer.pop(0)
            # COMPLETEから始まるとき(COMPLETE-RIGHT)
            elif action.startswith("COMPLETE"):
                if len(stack) < 2:
                    logger.info(
                        "Invalid COMPLETE action: the stack have lesser than 2 items"
                    )
                    continue

                aft_mention: Mention = stack.pop(-1)
                bef_mention: Mention = stack.pop(-1)
                # 今回はcompleteをした際のラベルには興味がないので空文字でよい
                bef_mention.label = ""
                aft_mention.label = ""
                mentions.append((bef_mention, aft_mention))
                
                # COMPLETE-RIGHT の時はstackの先頭を残す
                if "RIGHT" in action:
                    stack.append(aft_mention)

            # REDUCEが含まれるとき(REDUCE, REDUCE-LR)
            elif action.startswith("REDUCE"):
                # stack の 要素が 1 つ以下の場合には reduce 操作ができない
                # ただ、候補の選択肢を選ぶ段階でこのようなものが存在しないような設定にはなっている
                if len(stack) < 2:
                    logger.info(
                        "Invalid REDUCE action: %s, the number of elements in the stack is %d."
                        % (action, len(stack))
                    )
                    continue

                # stack に 要素が 2 つ以上ある場合には reduce 操作ができる
                # stackの先頭の要素
                right_node = stack.pop(-1)
                # stackの先頭から2番目の要素
                left_node = stack.pop(-1)

                if Mention.contains(left_node, right_node) or Mention.contains(
                    mention1=right_node, mention2=left_node
                ):
                    logger.info(
                        "Invalid REDUCE action: the last two elements in the stack contain each other"
                    )
                    continue

                merged: Mention = _NodeInStack.merge_nodes(left_node, right_node)

                # 連結(REUDCE)したノードも追加する
                stack.append(merged)

                # REDUCE-LRの場合
                if "LR" in action:
                    stack.append(left_node)
            else:
                raise ValueError(f"action: {action} は定義されていません")

        return mentions

    # def mention2actions(self, mentions_str: str, sentence_length: int) -> list[str]:
    #     """_summary_
    #     mentionをactionに変換するらしい
    #     Args:
    #         mentions (str): _description_
    #         sentence_length (int): _description_
    #     """

    #     def _detect_overlapping_mentions(mentions_str: str) -> list[Mention]:
    #         """_summary_
    #         # TODO 何やってるんだここ
    #         Args:
    #             mentions_str (str): _description_

    #         Returns:
    #             _type_: _description_
    #         """
    #         mentions: list[Mention] = Mention.create_mentions(mentions_str)

    #         for i in range(len(mentions)):
    #             if mentions[i]._overlapping:
    #                 continue
    #             for j in range(len(mentions)):
    #                 if i == j:
    #                     continue
    #                 if Mention.overlap_spans(mentions[i], mentions[j]):
    #                     assert mentions[i].label == mentions[j].label
    #                     mentions[i]._overlapping = True
    #                     mentions[j]._overlapping = True
    #         return mentions

    #     def _involve_mention(mentions: list[Mention], token_id: int) -> bool:
    #         """_summary_
    #         token_idがどこかのmentionのどこかのスパンに含まれているか
    #         Args:
    #             mentions (list[Mention]): _description_
    #             token_id (int): _description_

    #         Returns:
    #             bool: _description_
    #         """
    #         for _, mention in enumerate(mentions):
    #             for span in mention.spans:
    #                 if span.start <= token_id and token_id <= span.end:
    #                     return True
    #         return False

    #     def _find_relevant_mentions(
    #         mentions: list[Mention], node: Mention
    #     ) -> tuple[list[int], list[int]]:
    #         """_summary_
    #         parents = node が含まれている mention の index のリスト
    #         equals = node と等しい mention の index のリスト
    #         Args:
    #             mentions (list[Mention]): _description_
    #             node (Mention): _description_

    #         Returns:
    #             tuple[list[int], list[int]]: _description_
    #         """
    #         parents: list[int] = []
    #         equals: list[int] = []

    #         for i in range(len(mentions)):
    #             if Mention.contains(mentions[i], node):
    #                 parents.append(i)
    #                 if Mention.equal_spans(mentions[i], node):
    #                     equals.append(i)
    #         return parents, equals

    #     # ここから関数のメインの中身
    #     mentions: list[Mention] = _detect_overlapping_mentions(
    #         mentions_str=mentions_str
    #     )
    #     actions: list[str] = []
    #     stack: list[Mention] = []
    #     buffer: list[int] = [i for i in range(sentence_length)]

    #     while len(buffer) > 0:
    #         if not _involve_mention(mentions=mentions, token_id=buffer[0]):
    #             actions.append("OUT")
    #             buffer.pop(0)
    #         else:
    #             actions.append("SHIFT")
    #             stack.append(_NodeInStack.single_token_node(idx=buffer[0]))
    #             buffer.pop(0)

    #             stack_changed = True

    #             # COMPLETE, REDUCE, LEFT-REDUCE, RIGHT-REDUCE
    #             # if the last item of the stack is a mention, and does not involve with other mentions, then COMPLETE
    #             while stack_changed:
    #                 stack_changed = False

    #                 if len(stack) >= 1:
    #                     parents, equals = _find_relevant_mentions(mentions, stack[-1])
    #                     if len(equals) == 1 and len(parents) == 1:
    #                         actions.append("COMPLETE-%s" % mentions[equals[0]].label)
    #                         stack.pop(-1)
    #                         mentions.pop(equals[0])
    #                         stack_changed = True

    #                     # three REDUCE actions
    #                     if len(stack) >= 2:
    #                         if not Mention.overlap_spans(stack[-2], stack[-1]):
    #                             last_two_nodes = _NodeInStack.merge_nodes(
    #                                 stack[-2], stack[-1]
    #                             )
    #                             parents_of_two, _ = _find_relevant_mentions(
    #                                 mentions, last_two_nodes
    #                             )
    #                             if len(parents_of_two) > 0:
    #                                 parent_of_left, _ = _find_relevant_mentions(
    #                                     mentions, stack[-2]
    #                                 )
    #                                 parent_of_right, _ = _find_relevant_mentions(
    #                                     mentions, stack[-1]
    #                                 )
    #                                 if len(parents_of_two) != len(parent_of_left):
    #                                     actions.append("LEFT-REDUCE")
    #                                     stack.pop(-1)
    #                                     stack.append(last_two_nodes)
    #                                     stack_changed = True
    #                                 elif len(parents_of_two) != len(parent_of_right):
    #                                     actions.append("RIGHT-REDUCE")
    #                                     stack.pop(-2)
    #                                     stack.append(last_two_nodes)
    #                                     stack_changed = True
    #                                 else:
    #                                     actions.append("REDUCE")
    #                                     stack.pop(-1)
    #                                     stack.pop(-1)
    #                                     stack.append(last_two_nodes)
    #                                     stack_changed = True
    #     return actions
