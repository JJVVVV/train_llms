# from dis import Instruction
import os
from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Sequence

import datasets
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from toolkit.enums import Split
from torch.utils.data import Dataset

from ngram import get_cn_char_unigram

IGNORE_INDEX = -100


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: "
)


def prompt_transfer(instruction, input, output):
    target_length = len(get_cn_char_unigram(output))
    tmp = round(target_length / 50)
    target_length = tmp * 50
    if not instruction:
        instruction = "给定文本：\n" + input.strip() + "\n生成" + str(target_length) + "字左右视频文案。"
    prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})
    return prompt


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))


class TrainingDataset(Dataset):
    def __init__(
        self,
        split,
        config,
        data_path: List[str] | str,
        tokenizer: transformers.PreTrainedTokenizer,
        max_seq_length: int,
        data_cache_dir=None,
        preprocessing_num_workers=None,
        use_cache=False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        def tokenization(examples):
            sources = []
            targets = []
            prompt = PROMPT_TEMPLATE
            for instruction, input, output in zip(examples["instruction"], examples["input"], examples["output"]):
                # if input is not None and input !="":
                #     instruction = instruction+'\n'+input
                # source = prompt.format_map({'instruction':instruction})
                source = prompt_transfer(instruction, input, output)
                target = f"{output}{tokenizer.eos_token}"

                sources.append(source)
                targets.append(target)

            tokenized_sources = tokenizer(sources, return_attention_mask=False)
            tokenized_targets = tokenizer(targets, return_attention_mask=False, add_special_tokens=False)

            all_input_ids = []
            all_labels = []
            for s, t in zip(tokenized_sources["input_ids"], tokenized_targets["input_ids"]):
                input_ids = torch.LongTensor(s + t)[:max_seq_length]
                labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
                assert len(input_ids) == len(labels)
                all_input_ids.append(input_ids)
                all_labels.append(labels)

            results = {"input_ids": all_input_ids, "labels": all_labels}
            return results

        all_datasets = []

        if not isinstance(data_path, (list, tuple)):
            data_path = [data_path]
        for file in data_path:
            if data_cache_dir is None:
                data_cache_dir = str(os.path.dirname(file))
            cache_path = os.path.join(data_cache_dir, os.path.basename(file).split(".")[0])
            try:
                # todo bug: do not load from cache, it will cause `CUDA error: device-side assert triggered`
                if use_cache:
                    processed_dataset = datasets.load_from_disk(cache_path)
                    print(f"training datasets-{file} has been loaded from disk")
                else:
                    raise Exception("")
            except Exception:
                raw_dataset = load_dataset("json", data_files=file)
                tokenization_func = tokenization
                tokenized_dataset = raw_dataset.map(
                    tokenization_func,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    remove_columns=["instruction", "input", "output"],
                    keep_in_memory=False,
                    desc="preprocessing on dataset",
                )
                processed_dataset = tokenized_dataset
                if use_cache:
                    os.makedirs(cache_path, exist_ok=True)
                    processed_dataset.save_to_disk(cache_path)
            processed_dataset.set_format("torch")
            all_datasets.append(processed_dataset["train"])
        self.all_datasets = concatenate_datasets(all_datasets)

    def __getitem__(self, index) -> dict:
        return self.all_datasets[index]

    def __len__(self):
        return len(self.all_datasets)

    @property
    def collate_fn(self):
        return DataCollatorForSupervisedDataset(self.tokenizer)
