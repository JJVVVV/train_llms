import time
from pathlib import Path

import deepspeed
import hjson
import numpy as np
import toolkit
import torch
import torch.distributed as dist
from fire import Fire
from toolkit import getLogger
from toolkit.metric import MetricDict, calculate_rouge
from build_dataset import TrainingDataset
from toolkit.training import Trainer, initialize
from toolkit.nlp import TextDataset, NLPTrainingConfig
import os

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, CONFIG_MAPPING

from transformers.integrations import HfDeepSpeedConfig
from toolkit.enums import Split
from load_data_fn import load_data_fn

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_PAD_TOKEN = "<unk>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

# todo
# next line instructs transformers to partition the model directly over multiple gpus using
# deepspeed.zero.Init when model's `from_pretrained` method is called.
#
# **it has to be run before loading the model AutoModelForSeq2SeqLM.from_pretrained(model_name)**
#
# otherwise the model will first be loaded normally and only partitioned at forward time which is
# less efficient and when there is little CPU RAM may fail
# dschf = HfDeepSpeedConfig(ds_config)


def eval_callback(all_labels, all_logits, mean_loss):
    return calculate_rouge(all_logits, all_labels, ("rouge1", "rouge2", "rougeL"), "zh")


def load_tokenizer() -> PreTrainedTokenizer:
    # * Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": config.cache_dir,
        "use_fast": config.use_fast_tokenizer,
        "revision": config.model_revision,
        "use_auth_token": True if config.use_auth_token else None,
    }
    if config.model_dir:
        tokenizer = AutoTokenizer.from_pretrained(config.model_dir, **tokenizer_kwargs, trust_remote_code=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # * resize embedding
    if tokenizer.pad_token is None:
        logger.debug(f"Adding pad token {DEFAULT_PAD_TOKEN}")
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    logger.debug(f"len(tokenizer):{len(tokenizer)}")
    if dist.is_initialized():
        dist.barrier()
    return tokenizer


def load_dataset(tokenizer: PreTrainedTokenizer) -> tuple:
    # * Load training data, development data and test data
    path = Path(config.train_file_path)
    files = [os.path.join(path, file.name) for file in path.glob("*.json")]
    logger.debug(str(files))
    train_dataset = TrainingDataset(
        Split.TRAINING, config, files, tokenizer, config.max_seq_length, preprocessing_num_workers=config.preprocessing_num_workers
    )

    val_dataset = TextDataset.from_file(config.val_file_path, tokenizer, split=Split.VALIDATION, configs=config, load_data_fn=load_data_fn)

    test_dataset = TextDataset.from_file(config.test_file_path, tokenizer, split=Split.TEST, configs=config, load_data_fn=load_data_fn)
    if dist.is_initialized():
        dist.barrier()
    return train_dataset, val_dataset, test_dataset


def load_model(tokenizer):
    # global dschf
    start = time.time()

    # * define model class
    model_class = AutoModelForCausalLM

    # * define from_pretrained kwargs
    from_pretrained_kwargs = None

    # * Load model config
    model_kwargs = {"cache_dir": config.cache_dir, "revision": config.model_revision, "use_auth_token": True if config.use_auth_token else None}
    if config.model_dir:
        model_config = AutoConfig.from_pretrained(config.model_dir, **model_kwargs, trust_remote_code=True)
    else:
        model_config = CONFIG_MAPPING[config.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if config.config_overrides is not None:
            logger.info(f"Overriding config: {config.config_overrides}")
            model_config.update_from_string(config.config_overrides)
            logger.info(f"New config: {config}")

    # * Load model
    if model_config.vocab_size != len(tokenizer):
        logger.debug(f"local_rank {local_rank}: Loading model ...")
        if config.model_dir:
            torch_dtype = config.torch_dtype if config.torch_dtype in ["auto", None] else getattr(torch, config.torch_dtype)
            model = AutoModelForCausalLM.from_pretrained(
                config.model_dir,
                from_tf=bool(".ckpt" in config.model_dir),
                config=model_config,
                cache_dir=config.cache_dir,
                revision=config.model_revision,
                use_auth_token=True if config.use_auth_token else None,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_config(config)
            n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
            logger.debug(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) != embedding_size:
            logger.debug("resize the embedding size by the size of the tokenizer")
            model.resize_token_embeddings(len(tokenizer))
    else:
        logger.debug(f"local_rank {local_rank}: Construct `from_pretrained` kwargs ...")
        model = None
        torch_dtype = config.torch_dtype if config.torch_dtype in ["auto", None] else getattr(torch, config.torch_dtype)
        from_pretrained_kwargs = dict(
            from_tf=bool(".ckpt" in config.model_dir),
            cache_dir=config.cache_dir,
            revision=config.model_revision,
            use_auth_token=True if config.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        )

    end = time.time()

    logger.debug(f"local_rank {local_rank}: Loading model takes {end - start:.2f} sec.")
    # * sync
    if dist.is_initialized():
        dist.barrier()
    return model, model_config, model_class, from_pretrained_kwargs


def main() -> None:
    # * Loading tokenizer
    tokenizer = load_tokenizer()

    # * load dataset
    train_dataset, val_dataset, test_dataset = load_dataset(tokenizer)

    # *load model
    model, model_config, model_class, from_pretrained_kwargs = load_model(tokenizer)

    # * Train
    trainer = Trainer(
        task_type="generate",
        evaluate_only=False,
        config=config,
        model=model,
        model_config=model_config,
        model_class=model_class,
        from_pretrained_kwargs=from_pretrained_kwargs,
        dataset_train=train_dataset,
        dataset_val=val_dataset,
        dataset_test=test_dataset,
        calculate_metric_callback=eval_callback,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    # * Get args
    config: NLPTrainingConfig = Fire(NLPTrainingConfig)

    # * Create checkpoints,tensorboard outputs directory
    _dir = Path(
        config.dataset_name,
        config.model_type,
        config.model_name,
        str(config.epochs),
        str(config.train_batch_size),
        str(config.opt_lr),
        str(config.seed),
    )
    if config.save_dir is None:
        config.save_dir = Path("outputs", _dir)
    config.save(config.save_dir, silence=False)

    # * Create logger
    output_path_logger = config.save_dir / "report.log"
    logger = getLogger(__name__, output_path_logger)
    toolkit.set_file_logger(output_path_logger)

    # * Initalize parallel and seed
    local_rank, world_size = initialize(config)

    main()
