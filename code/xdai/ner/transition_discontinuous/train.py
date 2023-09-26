import json, logging, os, sys, torch

sys.path.insert(0, os.path.abspath("../../.."))

from xdai.utils.args import parse_parameters
from xdai.utils.common import (
    create_output_dir,
    set_cuda,
    set_random_seed,
)
from xdai.utils.instance import Instance, MetadataField, TextField, ActionField
from xdai.utils.iterator import BasicIterator, BucketIterator
from xdai.utils.token import Token
from xdai.utils.token_indexer import (
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
    ELMoIndexer,
)
from xdai.utils.train import train_op, eval_op
from xdai.utils.vocab import Vocabulary
from xdai.ner.transition_discontinuous.models import TransitionModel
from xdai.ner.mention import Mention
from xdai.ner.transition_discontinuous.parsing import Parser


logger = logging.getLogger(__name__)


"""Update at April-22-2019"""


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


if __name__ == "__main__":
    # TODO argsの型
    args = parse_parameters()
    # 出力先のファイルを作成する
    create_output_dir(args)

    # loggerを作成する
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=args.log_filepath,
    )

    # configを追加(上書き)する
    addition_args = json.load(open("config.json"))
    for k, v in addition_args.items():
        setattr(args, k, v)

    # argsの中身を出力する
    logger.info(
        "Parameters: %s"
        % json.dumps(
            {k: v for k, v in vars(args).items() if v is not None},
            indent=2,
            sort_keys=True,
        )
    )

    # cudaの準備
    set_cuda(args)

    # 乱数のシードを設定する
    set_random_seed(args)

    # データセットを読み込む箇所が始まったっぽい
    dataset_reader: DatasetReader = DatasetReader(args)

    # 学習データを読み込む
    train_data: list[Instance] = dataset_reader.read(
        filepath=args.train_filepath, training=True
    )
    # 開発データを用意する
    # 開発用のファイルがあればそこから読み込む
    # なければ学習データの1/10を開発データに用いる
    if args.dev_filepath is None:
        num_dev_instances = int(len(train_data) / 10)
        dev_data = train_data[0:num_dev_instances]
        train_data = train_data[num_dev_instances:]
    else:
        dev_data = dataset_reader.read(args.dev_filepath)

    # 一通り割り振った後に、引数が存在していたら後から絞り込む
    # args.num_train_instances, args.num_dev_instances はともに int 型
    if args.num_train_instances is not None:
        train_data = train_data[0 : args.num_train_instances]
    if args.num_dev_instances is not None:
        dev_data = dev_data[0 : args.num_dev_instances]

    logger.info("Load %d instances from train set." % (len(train_data)))
    logger.info("Load %d instances from dev set." % (len(dev_data)))

    # テストデータを用意する
    test_data = dataset_reader.read(args.test_filepath)
    logger.info("Load %d instances from test set." % (len(test_data)))

    datasets = {"train": train_data, "validation": dev_data, "test": test_data}
    # すべてのdataset内のinstanceからvocabを作成している
    vocab: Vocabulary = Vocabulary.from_instances(
        instances=[instance for dataset in datasets.values() for instance in dataset]
    )
    # vocabをファイルに保存する
    vocab.save_to_files(os.path.join(args.output_dir, "vocabulary"))

    # TODO これは何？
    # trainに対してなんやかんややっている場所
    train_iterator = BucketIterator(
        sorting_keys=[("tokens", "tokens_length")],
        batch_size=args.train_batch_size_per_gpu,
    )
    train_iterator.index_with(vocab)

    # devに対してなんやかんややっている場所
    dev_iterator = BasicIterator(batch_size=args.eval_batch_size_per_gpu)
    dev_iterator.index_with(vocab)

    # transitionModelを動かしている(ようやく)
    model: TransitionModel = TransitionModel(args, vocab).cuda(args.cuda_device[0])
    # どういうこと？
    parameters = [p for _, p in model.named_parameters() if p.requires_grad]

    # ちょっと見たことあるoptimizer
    # Adamを使っているのか
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    # これが学習を行っている箇所
    metrics = train_op(
        args, model, optimizer, train_data, train_iterator, dev_data, dev_iterator
    )
    logger.info(metrics)

    # 一番良かったデータに対してテストを行う
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best.th")))
    test_metrics, test_preds = eval_op(args, model, test_data, dev_iterator)
    logger.info(test_metrics)
    with open(os.path.join(args.output_dir, "test.pred"), "w") as f:
        for i in test_preds:
            f.write("%s\n%s\n\n" % (i[0], i[1]))

    # devが存在するときにはdevをやっている？
    if args.dev_filepath is not None:
        dev_metrics, dev_preds = eval_op(args, model, dev_data, dev_iterator)
        logger.info(dev_metrics)
        with open(os.path.join(args.output_dir, "dev.pred"), "w") as f:
            for i in dev_preds:
                f.write("%s\n%s\n\n" % (i[0], i[1]))
