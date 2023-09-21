import torch


"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
Update date: 2019-Nov-5"""


def add_sentence_boundary_token_ids(
    tensor: torch.FloatTensor,
    mask: torch.long,
    sentence_begin_token: torch.Tensor,
    sentence_end_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.LongTensor]:
    """_summary_
        文頭と文末に、それぞれ特殊なtokenを追加する
    Args:
        tensor (torch.Tensor): _description_
        mask (torch.long): _description_
        sentence_begin_token (torch.Tensor): _description_
        sentence_end_token (torch.Tensor): _description_

    Returns:
        tuple[torch.Tensor, torch.Tensor[torch.long]]: _description_
    """
    # TODO CPUを使うことになっているけど本当？
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()

    # TODO list[_int]って型らしいんだけどそれはlist[int]とは別？
    input_shape = list(tensor.data.shape)

    # 何でそんなことをする必要があるんですか？
    # -> 文頭と文末のtokenを入れるために要素を2つ分追加する
    output_shape = list(input_shape)
    output_shape[1] = input_shape[1] + 2

    # 要素を0で埋めたTensorを返す
    tensor_with_boundary_tokens: torch.Tensor = tensor.new_zeros(*output_shape)
    assert len(input_shape) == 3

    # TODO こういうのをさらっとやられても意味が分からない
    # 多分sentenceの中身に相当するtokenのtensorを移している
    tensor_with_boundary_tokens[:, 1:-1, :] = tensor

    # 先頭と末尾にtokenを追加する
    for i, j in enumerate(sequence_lengths):
        tensor_with_boundary_tokens[i, 0, :] = sentence_begin_token
        tensor_with_boundary_tokens[i, j + 1, :] = sentence_end_token

    # tensor_with_boundary_tokens は、文の始まりと終わりのトークンを追加した新しいテンソルです。
    # このテンソルは、文の始まりと終わりのトークンを含むため、通常のデータと特殊なトークンが混在しています。
    # mask_with_boundary_tokens は、新しいテンソル tensor_with_boundary_tokens の
    # 各要素が実際のデータを表すかどうかを示すバイナリマスクです。
    # データが存在する場合、対応する要素は1に設定され、
    # データが存在しない場合（パディングや特別なトークンの場合）、対応する要素は0に設定されます。
    # このバイナリマスクは、テンソル内の各トークンがモデルによって処理されるかどうかを制御するために使用されます。
    # 例えば、自然言語処理のモデルでは、パディングトークンは無視され、
    # 文の始まりと終わりのトークンは特別な処理を受けることがあります。
    # このバイナリマスクは、モデルにデータの有効な部分を正確に指示する役割を果たします。
    mask_with_boundary_tokens: torch.LongTensor = (
        (tensor_with_boundary_tokens > 0).long().sum(dim=-1) > 0
    ).long()

    return tensor_with_boundary_tokens, mask_with_boundary_tokens


"""Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
Update date: 2019-Nov-5"""


def remove_sentence_boundaries(tensor, mask):
    """_summary_
     文頭と文末の特殊なtokenをそれぞれ削除する
    Args:
        tensor (_type_): _description_
        mask (_type_): _description_

    Returns:
        _type_: _description_
    """
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    input_shape = list(tensor.data.shape)
    output_shape = list(input_shape)
    output_shape[1] = input_shape[1] - 2
    tensor_without_boundary_tokens = tensor.new_zeros(*output_shape)
    output_mask = tensor.new_zeros((output_shape[0], output_shape[1]), dtype=torch.long)
    for i, j in enumerate(sequence_lengths):
        if j > 2:
            tensor_without_boundary_tokens[i, : (j - 2), :] = tensor[i, 1 : (j - 1), :]
            output_mask[i, : (j - 2)] = 1
    return tensor_without_boundary_tokens, output_mask
