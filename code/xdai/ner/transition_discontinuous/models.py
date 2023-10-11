import torch
import torch.nn.functional as F
from xdai.ner.transition_discontinuous.parsing import Parser
from xdai.utils.attention import BilinearAttention
from xdai.utils.instance import TextField
from xdai.utils.seq2seq import LstmEncoder
from xdai.utils.token_embedder import Embedding, TextFieldEmbedder
from xdai.utils.vocab import Vocabulary
from xdai.utils.args import SimpleArgumentParser


# 本当はここら辺のクラスも全部個別のファイルに分けたい気持ちがある
class _Buffer:
    """_summary_
    バッファ
    """

    def __init__(self, sentence, empty_state: torch.nn.Parameter):
        """_summary_
        初期化
        Args:
            sentence (_type_): _description_
            empty_state (_type_): _description_
        """
        # 多分sentenceはTensor
        (
            sentence_length,
            _,
        ) = sentence.size()  # sentence length, contextual word embedding size
        self.sentence_length: int = sentence_length
        self.sentence = sentence
        self.pointer: int = 0
        # 空の状態を表すParameter
        self.empty_state: torch.nn.Parameter = empty_state

    def top(self) -> torch.Tensor:
        """_summary_
        bufferの先頭を返す
        もし、先頭が存在しない場合は「空の状態」を表す表現を返す。
        Returns:
            _type_: _description_
        """
        if self.pointer < self.sentence_length:
            return self.sentence[self.pointer]
        else:
            return self.empty_state

    def pop(self) -> torch.Tensor:
        """_summary_
        bufferの先頭を捨てる
        引数の戻り値としてはその捨てた値を返す
        Returns:
            _type_: _description_
        """
        assert self.pointer < self.sentence_length

        self.pointer += 1
        return self.sentence[self.pointer - 1]

    def __len__(self):
        """_summary_
        bufferの長さを返す。
        具体的にはsentenceの長さから、現在の位置を引いた残りの長さを返す
        Returns:
            _type_: _description_
        """
        return self.sentence_length - self.pointer


class _StackLSTM:
    """_summary_
    stackLSTM
    """

    def __init__(
        self, lstm_cell: torch.nn.LSTMCell, initial_state: list[torch.nn.Parameter]
    ):
        self.lstm_cell: torch.nn.LSTMCell = lstm_cell
        self.state: list[list[torch.nn.Parameter]] = [initial_state]

    def push(self, input: torch.Tensor) -> None:
        """_summary_
        stackにinputの要素を追加する
        Args:
            input (torch.Tensor): 入力
        """
        self.state.append(self.lstm_cell(input.unsqueeze(0), self.state[-1]))

    def pop(self) -> torch.Tensor:
        """_summary_
        stackの先頭から要素を捨てる
        捨てた要素を関数の戻り値とする
        Returns:
            _type_: _description_
        """
        assert len(self.state) > 1
        return self.state.pop()[0].squeeze(0)

    def top(self):
        """_summary_
        stackの先頭要素を取得する
        Returns:
            _type_: _description_
        """
        assert len(self.state) > 0
        return self.state[-1][0].squeeze(0)

    def top3(self):
        """_summary_
        stackの先頭から3つを取得する
        3つない場合は同じものを2つ以上取得していそう
        Returns:
            _type_: _description_
        """
        if len(self) >= 3:
            return (
                self.state[-3][0].squeeze(0),
                self.state[-2][0].squeeze(0),
                self.state[-1][0].squeeze(0),
            )
        if len(self) >= 2:
            return (
                self.state[-2][0].squeeze(0),
                self.state[-2][0].squeeze(0),
                self.state[-1][0].squeeze(0),
            )

        return (
            self.state[-1][0].squeeze(0),
            self.state[-1][0].squeeze(0),
            self.state[-1][0].squeeze(0),
        )

    def __len__(self):
        """_summary_
        stackの長さを取得する
        Returns:
            _type_: _description_
        """
        return len(self.state) - 1


class _LeafModule(torch.nn.Module):
    """_summary_
    LeafModule?
    Args:
        torch (_type_): _description_
    """

    def __init__(self, input_linear: torch.nn.Linear, output_linear: torch.nn.Linear):
        super(_LeafModule, self).__init__()
        self.input_linear: torch.nn.Linear = input_linear
        self.output_linear: torch.nn.Linear = output_linear

    def forward(self, inputs: torch.Tensor):
        """_summary_
        順伝播層
        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        # TODO RNNの一般的な流れらしいが、あんまりピンと来ていないので調べる
        cell_state: torch.Tensor = self.input_linear(inputs)
        hidden_state = torch.sigmoid(self.output_linear(inputs)) * torch.tanh(
            cell_state
        )
        return hidden_state, cell_state


class _ReduceModule(torch.nn.Module):
    """_summary_
    ReduceModule
    Args:
        torch (_type_): _description_
    """

    def __init__(self, reduce_linears):
        super(_ReduceModule, self).__init__()
        self.reduce_linears = reduce_linears

    def forward(self, left_cell, left_hidden, right_cell, right_hidden):
        concate_hidden = torch.cat([left_hidden, right_hidden], 0)
        input_gate = torch.sigmoid(self.reduce_linears[0](concate_hidden))
        left_gate = torch.sigmoid(self.reduce_linears[1](concate_hidden))
        right_gate = torch.sigmoid(self.reduce_linears[2](concate_hidden))
        candidate_cell_state = torch.tanh(self.reduce_linears[3](concate_hidden))
        cell_state = (
            input_gate * candidate_cell_state
            + left_gate * left_cell
            + right_gate * right_cell
        )
        hidden_state = torch.tanh(cell_state)
        return hidden_state, cell_state


class _Stack(object):
    """_summary_
    Stack
    Args:
        object (_type_): _description_
    """

    def __init__(
        self,
        lstm_cell: torch.nn.LSTMCell,
        initial_state: list[torch.nn.Parameter],
        input_linear: torch.nn.Linear,
        output_linear: torch.nn.Linear,
        reduce_linears: torch.nn.ModuleList,
    ):
        # stack lstmのモデル(？側と言うべき？)
        self.stack_lstm: _StackLSTM = _StackLSTM(lstm_cell, initial_state)
        # leaf モジュール(結局これはなに？)
        self.leaf_module: _LeafModule = _LeafModule(input_linear, output_linear)
        # reduce操作を行うモジュール
        self.reduce_module: _ReduceModule = _ReduceModule(reduce_linears)
        # 何かの状態
        # TODO listの中の型
        self._states: list[tuple[torch.Tensor, torch.Tensor]] = []

    def shift(self, input: torch.Tensor):
        """_summary_
        shift操作
        Args:
            input (_type_): _description_
        """
        # 分割代入には型アノテーションを付けられないので事前に定義しておく
        hidden: torch.Tensor
        cell: torch.Tensor
        hidden, cell = self.leaf_module(input)
        self._states.append((hidden, cell))
        self.stack_lstm.push(hidden)

    def reduce(self, keep: str | None = None) -> None:
        """_summary_
        reduce操作
        Args:
            keep (_type_, optional): _description_. Defaults to None.
        """
        assert len(self._states) > 1
        right_hidden: torch.Tensor
        right_cell: torch.Tensor
        left_hidden: torch.Tensor
        left_cell: torch.Tensor
        right_hidden, right_cell = self._states.pop()
        left_hidden, left_cell = self._states.pop()

        hidden: torch.Tensor
        cell: torch.Tensor
        hidden, cell = self.reduce_module(
            left_cell, left_hidden, right_cell, right_hidden
        )

        # reduce後に残す場合がある(right_reduce,left_reduce)
        if keep == "RIGHT":
            self._states.append((right_hidden, right_cell))
            self.stack_lstm.state.pop(-2)
        elif keep == "LEFT":
            self._states.append((left_hidden, left_cell))
            self.stack_lstm.state.pop(-1)
        else:
            self.stack_lstm.state.pop()
            self.stack_lstm.state.pop()

        self._states.append((hidden, cell))
        self.stack_lstm.push(hidden)

    def top(self) -> torch.Tensor:
        """_summary_
        stackの先頭を取得する
        Returns:
            torch.Tensor: _description_
        """
        return self.stack_lstm.top()

    def top3(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """_summary_
        stackの先頭から3つを取得する
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        return self.stack_lstm.top3()

    def pop(self) -> None:
        """_summary_
        stackの中身を捨てる
        """
        self._states.pop()
        self.stack_lstm.pop()

    def __len__(self) -> int:
        return len(self._states)


def _xavier_initialization(*size) -> torch.nn.Parameter:
    """_summary_
    ザビエル初期化
    Returns:
        _type_: _description_
    """
    p = torch.nn.init.xavier_normal_(torch.FloatTensor(*size)).cuda()
    return torch.nn.Parameter(p)


"""Update at 2019-Nov-7"""


class TransitionModel(torch.nn.Module):
    """_summary_
    TransitionModel
    Args:
        torch (_type_): _description_
    """

    def __init__(self, args: SimpleArgumentParser, vocab: Vocabulary):
        super(TransitionModel, self).__init__()
        self.idx2action = vocab.get_index_to_item_vocabulary("actions")
        self.action2idx = vocab.get_item_to_index_vocabulary("actions")

        self.text_filed_embedder = TextFieldEmbedder.create_embedder(args, vocab)
        self.action_embedding = Embedding(
            vocab.get_vocab_size("actions"), args.action_embedding_size
        )
        self.encoder = LstmEncoder(
            input_size=self.text_filed_embedder.get_output_dim(),
            hidden_size=args.lstm_cell_size,
            num_layers=args.lstm_layers,
            dropout=args.dropout,
            bidirectional=True,
        )
        self.dropout = torch.nn.Dropout(args.dropout)
        self.token_empty: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn(args.lstm_cell_size * 2)
        )

        self.stack_lstm = torch.nn.LSTMCell(
            args.lstm_cell_size * 2, args.lstm_cell_size * 2
        )
        self.stack_lstm_initial = [
            _xavier_initialization(1, args.lstm_cell_size * 2),
            _xavier_initialization(1, args.lstm_cell_size * 2),
        ]

        self.action_lstm = torch.nn.LSTMCell(
            args.action_embedding_size, args.lstm_cell_size * 2
        )
        self.action_lstm_initial = [
            _xavier_initialization(1, args.lstm_cell_size * 2),
            _xavier_initialization(1, args.lstm_cell_size * 2),
        ]

        self.leaf_input_linear = torch.nn.Linear(
            args.lstm_cell_size * 2, args.lstm_cell_size * 2
        )
        self.leaf_output_linear = torch.nn.Linear(
            args.lstm_cell_size * 2, args.lstm_cell_size * 2
        )
        self.reduce_module = torch.nn.ModuleList(
            [
                torch.nn.Linear(2 * args.lstm_cell_size * 2, args.lstm_cell_size * 2)
                for _ in range(4)
            ]
        )

        self.hidden2feature = torch.nn.Linear(
            args.lstm_cell_size * 2 * 8, args.lstm_cell_size
        )
        self.feature2action = torch.nn.Linear(
            args.lstm_cell_size, vocab.get_vocab_size("actions")
        )

        self.stack1_attention = BilinearAttention(
            args.lstm_cell_size * 2, args.lstm_cell_size * 2
        )
        self.stack2_attention = BilinearAttention(
            args.lstm_cell_size * 2, args.lstm_cell_size * 2
        )
        self.stack3_attention = BilinearAttention(
            args.lstm_cell_size * 2, args.lstm_cell_size * 2
        )

        self.parser = Parser()

        self._metric = {
            "correct_actions": 0,
            "total_actions": 0,
            "correct_mentions": 0,
            "total_gold_mentions": 0,
            "total_pred_mentions": 0,
            "correct_disc_mentions": 0,
            "total_gold_disc_mentions": 0,
            "total_pred_disc_mentions": 0,
        }

    def _get_possible_actions(self, stack, buffer, previous_action_name=""):
        valid_actions = []

        if len(buffer) > 0:
            valid_actions.append(self.action2idx["SHIFT"])
            valid_actions.append(self.action2idx["OUT"])
        if len(stack) >= 1:
            valid_actions += [
                i for i, a in self.idx2action.items() if a.startswith("COMPLETE")
            ]
            if len(stack) >= 2 and not previous_action_name in [
                "LEFT-REDUCE",
                "RIGHT-REDUCE",
            ]:
                valid_actions += [
                    i for i, a in self.idx2action.items() if a.find("REDUCE") >= 0
                ]
        valid_actions = sorted(valid_actions)
        return valid_actions

    def get_metrics(self, reset=False):
        _metrics = {}
        _metrics["accuracy"] = (
            self._metric["correct_actions"] / self._metric["total_actions"]
            if self._metric["total_actions"] > 0
            else 0.0
        )
        _metrics["precision-overall"] = (
            self._metric["correct_mentions"] / self._metric["total_pred_mentions"]
            if self._metric["total_pred_mentions"] > 0
            else 0.0
        )
        _metrics["recall-overall"] = (
            self._metric["correct_mentions"] / self._metric["total_gold_mentions"]
            if self._metric["total_gold_mentions"] > 0
            else 0.0
        )
        _metrics["f1-overall"] = (
            2
            * _metrics["precision-overall"]
            * _metrics["recall-overall"]
            / (_metrics["precision-overall"] + _metrics["recall-overall"])
            if _metrics["precision-overall"] + _metrics["recall-overall"] > 0
            else 0.0
        )

        _metrics["discontinuous-precision-overall"] = (
            self._metric["correct_disc_mentions"]
            / self._metric["total_pred_disc_mentions"]
            if self._metric["total_pred_disc_mentions"] > 0
            else 0.0
        )
        _metrics["discontinuous-recall-overall"] = (
            self._metric["correct_disc_mentions"]
            / self._metric["total_gold_disc_mentions"]
            if self._metric["total_gold_disc_mentions"] > 0
            else 0.0
        )
        _metrics["discontinuous-f1-overall"] = (
            2
            * _metrics["discontinuous-precision-overall"]
            * _metrics["discontinuous-recall-overall"]
            / (
                _metrics["discontinuous-precision-overall"]
                + _metrics["discontinuous-recall-overall"]
            )
            if _metrics["discontinuous-precision-overall"]
            + _metrics["discontinuous-recall-overall"]
            > 0
            else 0.0
        )

        if reset:
            self._metric = {
                "correct_actions": 0,
                "total_actions": 0,
                "correct_mentions": 0,
                "total_gold_mentions": 0,
                "total_pred_mentions": 0,
                "correct_disc_mentions": 0,
                "total_gold_disc_mentions": 0,
                "total_pred_disc_mentions": 0,
            }

        return _metrics

    def _build_state_representation(self, buffer, stack, action_history):
        top3_stack, top2_stack, top1_stack = stack.top3()
        buffer_outputs = torch.unsqueeze(buffer.sentence, 0)
        top1_attn_weights = self.stack1_attention(
            torch.unsqueeze(top1_stack, 0), buffer_outputs
        )
        top1_attn_applied = torch.bmm(
            top1_attn_weights.unsqueeze(0), buffer_outputs
        ).squeeze()
        top2_attn_weights = self.stack2_attention(
            torch.unsqueeze(top2_stack, 0), buffer_outputs
        )
        top2_attn_applied = torch.bmm(
            top2_attn_weights.unsqueeze(0), buffer_outputs
        ).squeeze()
        top3_attn_weights = self.stack3_attention(
            torch.unsqueeze(top3_stack, 0), buffer_outputs
        )
        top3_attn_applied = torch.bmm(
            top3_attn_weights.unsqueeze(0), buffer_outputs
        ).squeeze()
        features = torch.cat(
            [
                top1_stack,
                top1_attn_applied,
                top2_stack,
                top2_attn_applied,
                top3_stack,
                top3_attn_applied,
                buffer.top(),
                action_history.top(),
            ],
            0,
        )
        return features

    def _apply_action(self, stack, buffer, action_name):
        if action_name == "SHIFT":
            stack.shift(buffer.pop().cuda())
        elif action_name.startswith("OUT"):
            buffer.pop()
        elif action_name.startswith("COMPLETE"):
            stack.pop()
        elif action_name.find("REDUCE") >= 0:
            if action_name.startswith("LEFT"):
                stack.reduce(keep="LEFT")
            elif action_name.startswith("RIGHT"):
                stack.reduce(keep="RIGHT")
            else:
                stack.reduce()
        else:
            raise ValueError
        return stack, buffer

    def forward(self, tokens, actions, annotations, **kwargs):
        embedded_tokens = self.dropout(self.text_filed_embedder(tokens))
        mask = TextField.get_text_field_mask(tokens)
        sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
        encoded_tokens = self.dropout(self.encoder(embedded_tokens, mask))

        batch_size, _, _ = encoded_tokens.size()

        total_loss = 0
        preds = []
        for i in range(batch_size):
            gold_actions, pred_actions = [], []

            stack = _Stack(
                self.stack_lstm,
                self.stack_lstm_initial,
                self.leaf_input_linear,
                self.leaf_output_linear,
                self.reduce_module,
            )
            action_history = _StackLSTM(self.action_lstm, self.action_lstm_initial)
            buffer = _Buffer(
                encoded_tokens[i][0 : sequence_lengths[i]], self.token_empty
            )
            previous_action_name = ""

            if self.training:
                for j in range(len(actions[i])):
                    gold_action = actions[i][j]
                    if type(gold_action) != int:
                        gold_action = gold_action.cpu().data.numpy().item()
                    if gold_action == 0:
                        break

                    valid_actions = self._get_possible_actions(
                        stack, buffer, previous_action_name
                    )
                    # TODO 多分actionsのラベルが一致していないので
                    # ここでエラーが出るのはそりゃそうって感じだと思う
                    assert gold_action in valid_actions
                    gold_actions.append(gold_action)
                    gold_action_name = self.idx2action[gold_action]
                    previous_action_name = gold_action_name

                    if len(valid_actions) == 1:
                        pred_action = valid_actions[0]
                        assert pred_action == gold_action
                    else:
                        features = self._build_state_representation(
                            buffer, stack, action_history
                        )
                        features = F.relu(self.hidden2feature(self.dropout(features)))
                        logits = self.feature2action(features)[
                            torch.LongTensor(valid_actions).cuda()
                        ]
                        log_probs = torch.nn.functional.log_softmax(logits, 0)
                        pred_action = valid_actions[
                            torch.max(logits.cpu(), 0)[1].data.numpy().item()
                        ]

                        valid_actions = {a: i for i, a in enumerate(valid_actions)}
                        assert len(log_probs) == len(valid_actions)
                        total_loss += log_probs[valid_actions[gold_action]]

                    pred_actions.append(pred_action)
                    action_history.push(
                        self.dropout(
                            self.action_embedding(
                                torch.LongTensor([gold_action]).cuda()
                            )
                        ).squeeze(0)
                    )

                    stack, buffer = self._apply_action(stack, buffer, gold_action_name)

                    assert len(gold_actions) == len(pred_actions)
                    for g, p in zip(gold_actions, pred_actions):
                        if g == p:
                            self._metric["correct_actions"] += 1
                        self._metric["total_actions"] += 1
            else:
                while True:
                    valid_actions = self._get_possible_actions(
                        stack, buffer, previous_action_name
                    )
                    if len(valid_actions) < 1:
                        break
                    elif len(valid_actions) == 1:
                        pred_action = valid_actions[0]
                    else:
                        features = self._build_state_representation(
                            buffer, stack, action_history
                        )
                        features = F.relu(self.hidden2feature(self.dropout(features)))

                        logits = self.feature2action(features)[
                            torch.LongTensor(valid_actions).cuda()
                        ]
                        pred_action = valid_actions[
                            torch.max(logits.cpu(), 0)[1].data.numpy().item()
                        ]

                    pred_actions.append(pred_action)
                    pred_action_name = self.idx2action[pred_action]
                    previous_action_name = pred_action_name

                    action_history.push(
                        self.dropout(
                            self.action_embedding(
                                torch.LongTensor([pred_action]).cuda()
                            )
                        ).squeeze(0)
                    )

                    stack, buffer = self._apply_action(stack, buffer, pred_action_name)

                gold_mentions = (
                    annotations[i].split("|") if len(annotations[i].strip()) > 0 else []
                )
                discontinuous = [
                    1 if len(m.split(" ")[0].split(",")) > 2 else 0
                    for m in gold_mentions
                ]
                discontinuous = sum(discontinuous) > 0
                pred_actions = [self.idx2action[p] for p in pred_actions]
                pred_mentions = self.parser.parse(pred_actions)
                pred_mentions = [str(p) for p in pred_mentions]
                for p in pred_mentions:
                    if p in gold_mentions:
                        self._metric["correct_mentions"] += 1
                        if discontinuous:
                            self._metric["correct_disc_mentions"] += 1
                self._metric["total_gold_mentions"] += len(gold_mentions)
                self._metric["total_pred_mentions"] += len(pred_mentions)
                if discontinuous:
                    self._metric["total_gold_disc_mentions"] += len(gold_mentions)
                    self._metric["total_pred_disc_mentions"] += len(pred_mentions)
                preds.append("|".join(pred_mentions))

        return {"loss": -1.0 * total_loss, "preds": preds}
