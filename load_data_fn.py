from pathlib import Path

import pandas as pd
from build_dataset import PROMPT_TEMPLATE, prompt_transfer
from ngram import get_cn_char_unigram
from toolkit.enums import Split
from toolkit.nlp.data import ClassificationLabel, FinelyControlledText, PairedText, RegressionLabel
from transformers import PreTrainedTokenizer


def load_data_fn(data_file_path: Path | str, model_type: str, tokenizer: PreTrainedTokenizer, split: Split, **kwargs):
    special_tokens_map = tokenizer.special_tokens_map
    BOS = special_tokens_map["bos_token"] if "bos_token" in special_tokens_map.keys() else None
    EOS = special_tokens_map["eos_token"] if "eos_token" in special_tokens_map.keys() else None
    SEP = special_tokens_map["sep_token"] if "sep_token" in special_tokens_map.keys() else None
    MASK = special_tokens_map["mask_token"] if "mask_token" in special_tokens_map.keys() else None
    CLS = special_tokens_map["cls_token"] if "cls_token" in special_tokens_map.keys() else None
    sep_num = 1

    df = pd.read_json(data_file_path, lines=True)
    if "debug" in kwargs and kwargs["debug"]:
        df = df[:kwargs["part"]]
    inputs = []
    labels = []
    for idx, row in df.iterrows():
        # Single
        a_sample = PairedText(prompt_transfer(row["input"], row["output"]))

        # label
        a_label = row["output"]

        inputs.append(a_sample)
        labels.append(a_label)

    return inputs, labels
