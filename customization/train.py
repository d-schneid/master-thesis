import torch
from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig

from structure_aware_config import StructureAwareConfig
from structure_aware_data import StructureAwareDataModule
from structure_aware_model import StructureAwareModel


if __name__ == "__main__":
    seq_length = 2048
    global_batch_size = 16

    data = StructureAwareDataModule()

    model = StructureAwareModel(config=StructureAwareConfig())

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
    )

    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=6e-4,
        bf16=True,
    )
    opt = nl.MegatronOptimizerModule(config=opt_config)

    trainer = nl.Trainer(
        devices=1,
        max_steps=50,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )

    nemo_logger = nl.NeMoLogger(
        log_dir="../scripts/test_logdir", # logs and checkpoints will be written here
    )

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer='data',
        optim=opt,
    )
