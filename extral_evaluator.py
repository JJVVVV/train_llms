import json
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from toolkit.config import TrainConfig
from toolkit.enums import Split
from toolkit.metric import MetricDict, bleu, rouge, self_bleu
from toolkit.nlp import TextDataset
from toolkit.training import Evaluator, get_dataloader
from toolkit.training.evaluator import logger
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def extra_calculate_metric_callback(all_labels, all_logits, config: TrainConfig, dataset: TextDataset):
    df = pd.DataFrame.from_dict(
        dict(inputs=[a_sample.to_list() for a_sample in dataset.texts_input[: len(all_labels)]], preds=all_logits, labels=all_labels)
    )
    generate_result_path = (
        config.save_dir
        / "evaluators"
        / "extal_evaluator"
        / f"epoch={config.training_runtime['cur_epoch']:03d}_step={config.training_runtime['cur_step']}.json"
    )
    generate_result_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(generate_result_path, force_ascii=False, indent=2, orient="records")
    metric = (rouge([logits_regen[0] for logits_regen in all_logits], all_labels, "zh", ("rougeL")) * 100).round(2)
    metric.update(((1 - self_bleu(all_logits, "zh", ("bleu4"), smoothing_level=1)) * 100).round(2))
    # metric.update((1-self_bleu(all_logits, "zh", ("bleu1", "bleu2", "bleu3", "bleu4"), smoothing_level=1)).round()*100)
    return metric


class Extral_Evaluator(Evaluator):
    def __init__(
        self,
        task_type: str,
        split: Split,
        config: TrainConfig,
        model,
        dataset: Dataset,
        calculate_metric_callback: Callable[..., MetricDict],
        extral_args_evaluation: dict | None = None,
        tokenizer=None,
    ) -> None:
        super().__init__(task_type, split, config, model, dataset, calculate_metric_callback, extral_args_evaluation, tokenizer)
        with open(config.generate_config_file, "r", encoding="utf8") as f:
            self.generate_config = json.load(f)

    def eval(self, cuda_id=None) -> MetricDict:
        """
        if specify the `cuda_id`, the model will run in it, ohterwise, default
        """
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.dataloader = (
            get_dataloader(self.dataset, self.config, self.split, collate_fn=self.dataset.collate_fn)
            if not hasattr(self, "dataloader")
            else self.dataloader
        )

        if cuda_id is not None:
            self.config.gpu = True
            torch.cuda.set_device(cuda_id)
            self.model.cuda()
        self.model.eval()

        if "sample_context" in self.config.model_name:
            all_labels = []
            all_logits = []
            for _batch in tqdm(self.dataloader, desc=self.split.name.capitalize(), colour="BLUE", unit="batch", smoothing=0.9):
                labels = _batch.pop("labels")
                # batch: (batch_size, re_gen_num, seq_len)
                texts_regen = []
                for i in range(self.config.re_gen_num):
                    batch = {key: value[:, i] for key, value in _batch.items()}
                    with torch.no_grad():
                        custom_inputs = batch.pop("custom_inputs", dict())
                        if self.config.gpu:
                            batch = {key: value.cuda() for key, value in batch.items()}
                        outputs = self.model.generate(**batch, **custom_inputs, **self.extral_args_evaluation, **self.generate_config)
                        if self.config.cut_input_from_output:
                            texts = []
                            for idx, output in enumerate(outputs):
                                texts.append(self.tokenizer.decode(output[batch["input_ids"][idx].size(0) :], skip_special_tokens=True))
                        else:
                            texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    texts_regen.append(texts)
                texts_regen = np.array(texts_regen).transpose((1, 0))  # (batch_size, re_gen_num)
                all_labels.extend(labels)
                all_logits.append(texts_regen)
            all_logits = np.concatenate(all_logits, axis=0).tolist()

        else:
            all_regenerate = []
            for _ in range(self.config.re_gen_num):
                all_labels = []
                all_logits = []
                for batch in tqdm(self.dataloader, desc=self.split.name, colour="BLUE", unit="batch", smoothing=0.9):
                    with torch.no_grad():
                        labels = batch.pop("labels")
                        custom_inputs = batch.pop("custom_inputs", dict())
                        if self.config.gpu:
                            batch = {key: value.cuda() for key, value in batch.items()}
                        outputs = self.model.generate(**batch, **custom_inputs, **self.extral_args_evaluation, **self.generate_config)
                        if self.config.cut_input_from_output:
                            texts = []
                            for idx, output in enumerate(outputs):
                                texts.append(self.tokenizer.decode(output[batch["input_ids"][idx].size(0) :], skip_special_tokens=True))
                        else:
                            texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        all_labels.extend(labels)
                        all_logits.extend(texts)
                all_regenerate.append(all_logits)
            all_regenerate = np.array(all_regenerate)
            all_logits = all_regenerate.transpose((1, 0)).tolist()

        self.model.train()

        if world_size > 1:
            logger.debug(
                f"local rank {local_rank}: num_labels: {len(all_labels)}, num_logits: {len(all_logits)}, num_batches: {len(self.dataloader)}"
            )

            labels_gather_list = [None for _ in range(world_size)]
            logits_gather_list = [None for _ in range(world_size)]

            dist.all_gather_object(labels_gather_list, all_labels)
            dist.all_gather_object(logits_gather_list, all_logits)

            all_labels = sum(labels_gather_list, [])
            all_logits = sum(logits_gather_list, [])

        return self.calculate_metric_callback(all_labels, all_logits, config=self.config, dataset=self.dataset)
