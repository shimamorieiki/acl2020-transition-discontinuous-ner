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


logger = logging.getLogger(__name__)


"""Update at April-22-2019"""


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
