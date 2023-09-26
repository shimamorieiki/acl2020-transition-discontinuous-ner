import logging
from tap import Tap

logger = logging.getLogger(__name__)


class SimpleArgumentParser(Tap):
    train_filepath: str | None = None
    num_train_instances: int | None = None
    dev_filepath: str | None = None
    num_dev_instances: int | None = None
    test_filepath: str | None = None
    cache_dir: str | None = None
    overwrite_cache: bool = False
    output_dir: str = "./output"
    overwrite_output_dir: bool = False
    log_filepath: str | None = None
    summary_json: str | None = None
    encoding: str = "utf-8-sig"
    do_train: bool = False
    learning_rate: float = 5e-5
    train_batch_size_per_gpu: int = 8
    num_train_epochs: int = 3
    max_steps: int = 0  # If > 0, override num_train_epochs.
    warmup_steps: int = 0
    logging_steps: int = 50
    save_steps: int = 0
    max_save_checkpoints: int = 2
    patience: int = 0
    max_grad_norm: float | None = None
    grad_clipping: float = 5
    adam_epsilon: float = 1e-8
    weight_decay: float | None = None
    gradient_accumulation_steps: int = 1  # Number of update steps to accumulate before performing a backward/update pass
    seed: int = 52
    cuda_device: list[int] = [0]  # a list cuda devices, splitted by
    do_eval: bool = False
    eval_batch_size_per_gpu: int = 8
    eval_metric: str | None = None
    eval_all_checkpoints: bool = False  # Evaluate all checkpoints.
    eval_during_training: bool = False  # Evaluate during training at each save step.

    model_type: str | None = None
    pretrained_model_dir: str | None = None
    max_seq_length: int = 128
    labels: str = "0,1"
    label_filepath: str | None = None
    tag_schema: str = "B,I"
    do_lower_case: bool = False

    # 実行時引数には与えられないが後々追加されるやつ
    n_gpu: int = 0

    # config.jsonから追加されるやつ
    # そういう意味では、標準入力のargsをずっと引き回すことに問題があるといわれればそれはとてもそう
    # 本当は標準入力から受け取るものとArgクラスと、こういうのを追加して一般に利用するArgクラスを別で持つべき
    pretrained_word_embeddings: str = "/data/dai031/Corpora/GloVe/glove.6B.100d.txt"
    word_embedding_size: int = 100
    char_embedding_size: int = 16
    action_embedding_size: int = 20
    lstm_cell_size: int = 200
    lstm_layers: int = 2
    dropout: float = 0.5


def parse_parameters() -> SimpleArgumentParser:
    return SimpleArgumentParser().parse_args(known_only=True)
