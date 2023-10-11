import json
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.abspath("../../.."))

from xdai.ner.transition_discontinuous.models import TransitionModel
from xdai.utils.args import parse_parameters
from xdai.utils.common import create_output_dir, set_cuda, set_random_seed
from xdai.utils.instance import Instance
from xdai.utils.iterator import BasicIterator, BucketIterator
from xdai.utils.train import eval_op, train_op
from xdai.utils.vocab import Vocabulary
from xdai.ner.transition_discontinuous.dataset_reader import DatasetReader
from xdai.ner.transition_discontinuous.dataset_reader_ys import DatasetReaderYS
from xdai.utils.args import SimpleArgumentParser

logger = logging.getLogger(__name__)


def save_vocabulary(datasets: dict[str, list[Instance]], output_dir: str) -> Vocabulary:
    """_summary_
    すべてのdataset内のinstanceからvocabを作成している
    Args:
        datasets (_type_): _description_
        output_dir (str): _description_
    """
    vocab: Vocabulary = Vocabulary.from_instances(
        instances=[instance for dataset in datasets.values() for instance in dataset]
    )
    # vocabをファイルに保存する
    vocab.save_to_files(os.path.join(output_dir, "vocabulary"))
    print("vocab saved")
    return vocab


def get_datasets(
    model_type: str,
    train_filepath: str | None,
    dev_filepath: str | None,
    test_filepath: str | None,
    num_train_instances: int | None,
    num_dev_instances: int | None,
) -> dict[str, list[Instance]]:
    """_summary_
    データセットを読み込む箇所
    """
    # データセットを読み込む箇所が始まったっぽい
    # DatasetReader interface を継承するデータセット独自のインフラ層を作りたい
    # 少なくとも日本語を対象としたものに対してはパーサを作成したのでそれを利用する
    dataset_reader: DatasetReader = DatasetReader(model_type=model_type)

    # 学習データを読み込む
    # このreadもファイルパスで読み込むファイルを分岐するのではなく、
    # DatasetReaderの読み込み時に決定したい
    if train_filepath is None:
        raise ValueError("train_filepath is not found")
    train_data: list[Instance] = dataset_reader.read(
        filepath=train_filepath, training=True
    )

    # 開発データを用意する
    # 開発用のファイルがあればそこから読み込む
    # なければ学習データの1/10を開発データに用いる
    if dev_filepath is None:
        num_dev_instances = int(len(train_data) / 10)
        dev_data = train_data[0:num_dev_instances]
        train_data = train_data[num_dev_instances:]
    else:
        dev_data = dataset_reader.read(dev_filepath)

    # 一通り割り振った後に、引数が存在していたら後から絞り込む
    # args.num_train_instances, args.num_dev_instances はともに int 型
    if num_train_instances is not None:
        train_data = train_data[0:num_train_instances]
    if num_dev_instances is not None:
        dev_data = dev_data[0:num_dev_instances]

    logger.info("Load %d instances from train set." % (len(train_data)))
    logger.info("Load %d instances from dev set." % (len(dev_data)))

    # テストデータを用意する
    if test_filepath is None:
        raise ValueError("test_filepath is not found")
    test_data = dataset_reader.read(test_filepath)
    logger.info("Load %d instances from test set." % (len(test_data)))

    return {"train": train_data, "validation": dev_data, "test": test_data}


def get_datasets_ys() -> dict[str, list[Instance]]:
    """_summary_
    ysデータセットを読み込む箇所
    Returns:
        dict[str, list[Instance]]: _description_
    """

    # 現時点ではすべてのデータが1つのファイルに存在している
    data_filepath = "../../../../data/ys/actions.jsonl"
    data_reader_ys = DatasetReaderYS("")
    instances = data_reader_ys.read(filepath=data_filepath)

    # データ全体をtrain:dev:test = 6:1:3で分ける
    num_dev_instances = int(len(instances) / 10)
    num_test_instances = int(3 * len(instances) / 10)
    dev_data = instances[0:num_dev_instances]
    test_data = instances[num_dev_instances : num_dev_instances + num_test_instances]
    train_data = instances[num_dev_instances + num_test_instances :]

    logger.info("Load %d instances from train set." % (len(train_data)))
    logger.info("Load %d instances from dev set." % (len(dev_data)))
    logger.info("Load %d instances from test set." % (len(test_data)))

    return {"train": train_data, "validation": dev_data, "test": test_data}


def main():
    args: SimpleArgumentParser = parse_parameters()
    # 出力先のファイルを作成する
    create_output_dir(args)

    # loggerを作成する
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=args.log_filepath,
    )

    # # configを追加(上書き)する
    # argでデフォルトとして明示した
    # addition_args = json.load(open("config.json"))
    # for k, v in addition_args.items():
    #     setattr(args, k, v)

    # TODO 取りあえず一旦無視する
    # argsの中身を出力する
    # logger.info(
    #     "Parameters: %s"
    #     % json.dumps(
    #         {k: v for k, v in vars(args).items() if v is not None},
    #         indent=2,
    #         sort_keys=True,
    #     )
    # )

    # cudaの準備
    set_cuda(args)

    # 乱数のシードを設定する
    set_random_seed(args)

    datasets = get_datasets_ys()
    # すべてのdataset内のinstanceからvocabを作成している
    # TODO 具体的な詳細についてはあとからちゃんと見る
    vocab: Vocabulary = save_vocabulary(datasets=datasets, output_dir=args.output_dir)

    # train_iteratorにvocabの値を追加している
    # だからなんなのか
    train_iterator = BucketIterator(
        sorting_keys=[("tokens", "tokens_length")],
        batch_size=args.train_batch_size_per_gpu,
    )
    train_iterator.index_with(vocab)

    # dev_iteratorにvocabの値を追加している
    # だからなんなのか
    dev_iterator = BasicIterator(batch_size=args.eval_batch_size_per_gpu)
    dev_iterator.index_with(vocab)

    # transitionModelを動かしている(ようやく)
    model: TransitionModel = TransitionModel(args, vocab).cuda(args.cuda_device[0])
    # どういうこと？
    # TODO 調べる
    parameters = [p for _, p in model.named_parameters() if p.requires_grad]

    # ちょっと見たことあるoptimizer
    # Adamを使っているのか
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    # これが学習を行っている箇所
    # この中も後から見るが今は別に関係ない
    metrics = train_op(
        args=args,
        model=model,
        optimizer=optimizer,
        train_data=datasets["train"],
        train_iterator=train_iterator,
        dev_data=datasets["validation"],
        dev_iterator=dev_iterator,
    )
    logger.info(metrics)

    # 一番良かったデータに対してテストを行う
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best.th")))
    test_metrics, test_preds = eval_op(args, model, datasets["test"], dev_iterator)
    logger.info(test_metrics)
    with open(os.path.join(args.output_dir, "test.pred"), "w") as f:
        for i in test_preds:
            f.write("%s\n%s\n\n" % (i[0], i[1]))

    # devが存在するときにはdevをやっている？
    if args.dev_filepath is not None:
        dev_metrics, dev_preds = eval_op(
            args, model, datasets["validation"], dev_iterator
        )
        logger.info(dev_metrics)
        with open(os.path.join(args.output_dir, "dev.pred"), "w") as f:
            for i in dev_preds:
                f.write("%s\n%s\n\n" % (i[0], i[1]))


if __name__ == "__main__":
    main()
