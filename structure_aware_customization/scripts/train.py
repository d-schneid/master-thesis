import os
from typing import Mapping, Optional

from data_preprocessing.tasks.code_text import CodeText
from data_preprocessing.datasets.cornstack import CornStack
from structure_aware_customization.model.structure_aware_starcoder2_config import StructureAwareStarcoder2Config
from structure_aware_customization.dataset.structure_aware_data_module import StructureAwareDataModule
from structure_aware_customization.model.structure_aware_starcoder2_model import StructureAwareStarcoder2Model
from structure_aware_customization.dataset.structure_aware_ct_dataset import StructureAwareCTDataset

from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.pytorch.strategies.utils import RestoreConfig

from lightning.pytorch.loggers import MLFlowLogger


class SafeMLFlowLogger(MLFlowLogger):

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        if step is not None:
            step = int(step)
        super().log_metrics(metrics, step)


if __name__ == "__main__":
    user_path = "/shared/home/xxx"

    task = CodeText()
    train_datasets = [CornStack(task=task, split="train")]
    validation_datasets = [CornStack(task=task, split="validation")]
    test_datasets = [CornStack(task=task, split="test")]
    train_ds = StructureAwareCTDataset(datasets=train_datasets)
    validation_ds = StructureAwareCTDataset(datasets=validation_datasets)
    test_ds = StructureAwareCTDataset(datasets=test_datasets)

    global_batch_size = 128
    num_epochs = 3

    data = StructureAwareDataModule(
        train_dataset=train_ds,
        validation_dataset=validation_ds,
        test_dataset=test_ds,
        micro_batch_size=16,
        global_batch_size=global_batch_size,
        seq_length=task.max_seq_len,
        num_train_samples=train_ds.num_samples,
        num_val_samples=validation_ds.num_samples,
        num_test_samples=test_ds.num_samples
    )

    model = StructureAwareStarcoder2Model(
        config=StructureAwareStarcoder2Config(),
    )

    trainer = nl.Trainer(
        num_nodes=1,
	    devices=8,
        max_epochs=num_epochs,
        max_steps=(train_ds.num_samples * num_epochs) // global_batch_size,
        accelerator="gpu",
        strategy=nl.MegatronStrategy(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=1,
            sequence_parallel=False,
            expert_model_parallel_size=1
        ),
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed"
        ),
	    log_every_n_steps=50,
        val_check_interval=500,
        limit_val_batches=0.6,
	    accumulate_grad_batches=1,
    )

    log = nl.NeMoLogger(
        name="structure_aware_starcoder2_finetuning_cc",
        log_dir=os.path.join(user_path, "finetuning/code_completion/log_dir"),
        ckpt=nl.ModelCheckpoint(
            save_last=True,
            monitor="val_loss",
            save_top_k=2,
            mode="min",
            save_optim_on_train_end=True,
            always_save_context=True,
            save_context_on_train_end=True,
            filename="structure_aware_starcoder2-{val_loss:.2f}-{step}-{consumed_samples}",
            every_n_epochs=1
        ),
        extra_loggers=[
            SafeMLFlowLogger(
                experiment_name="structure_aware_starcoder2_finetuning_cc",
                run_name="finetuning_cc",
                tracking_uri="http://ec2-18-208-185-48.compute-1.amazonaws.com:5000",
            )
        ],
    )

    resume_from_pretraining = nl.AutoResume(
        restore_config=RestoreConfig(
            path=os.path.join(user_path, 'pretraining_path'),
            load_model_state=True,
            load_optim_state=False,
            load_artifacts=False,
        ),
    )

    resume_from_ckpt = nl.AutoResume(
        resume_from_path=os.path.join(user_path, 'ckpt_path'),
        resume_if_exists=True,
        resume_past_end=False,
        resume_ignore_no_checkpoint=False,
    )

    optim = nl.MegatronOptimizerModule(
        config=OptimizerConfig(
            optimizer='adam',
            lr=4.3e-2,
            use_distributed_optimizer=True,
        ),
        lr_scheduler=nl.lr_scheduler.NoamAnnealingScheduler(
            d_model=model.config.hidden_size,
            max_steps=(train_ds.num_samples * num_epochs) // global_batch_size,
            warmup_steps=1500,
            min_lr=2e-6,
        )
    )

    tokenizer = get_nmt_tokenizer(library='huggingface', model_name='bigcode/starcoder2-3b', use_fast=True)

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=log,
        resume=resume_from_pretraining,
        optim=optim,
        tokenizer=tokenizer
    )
