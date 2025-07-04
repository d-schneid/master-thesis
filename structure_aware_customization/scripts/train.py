import os
from typing import Mapping, Optional

from data_preprocessing.tasks.pretraining import Pretraining
from data_preprocessing.datasets.code_search_net import CodeSearchNet
from data_preprocessing.datasets.cornstack import CornStack
from data_preprocessing.datasets.stack import Stack
from structure_aware_customization.model.structure_aware_starcoder2_config import StructureAwareStarcoder2Config
from structure_aware_customization.dataset.structure_aware_data_module import StructureAwareDataModule
from structure_aware_customization.model.structure_aware_starcoder2_model import StructureAwareStarcoder2Model
from structure_aware_customization.dataset.structure_aware_pretraining_dataset import StructureAwarePretrainingDataset

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
    user_path = "/shared/home/i741961"

    task = Pretraining()
    train_datasets = [CornStack(task=task, split="train"), CodeSearchNet(task=task, split="train"), Stack(task=task, split="train")]
    validation_datasets = [CornStack(task=task, split="validation"), CodeSearchNet(task=task, split="validation"), Stack(task=task, split="validation")]
    test_datasets = [CornStack(task=task, split="test"), CodeSearchNet(task=task, split="test"), Stack(task=task, split="test")]
    train_ds = StructureAwarePretrainingDataset(datasets=train_datasets)
    validation_ds = StructureAwarePretrainingDataset(datasets=validation_datasets)
    test_ds = StructureAwarePretrainingDataset(datasets=test_datasets)

    data = StructureAwareDataModule(
        train_dataset=train_ds,
        validation_dataset=validation_ds,
        test_dataset=test_ds,
        micro_batch_size=16,
        global_batch_size=64,
        seq_length=task.max_seq_len,
        num_train_samples=train_ds.num_samples,
        num_val_samples=validation_ds.num_samples,
        num_test_samples=test_ds.num_samples
    )

    model = StructureAwareStarcoder2Model(
        config=StructureAwareStarcoder2Config()
    )

    trainer = nl.Trainer(
        num_nodes=1,
	    devices=4,
        max_epochs=3,
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
	    log_every_n_steps=10,
        val_check_interval=50,
        limit_val_batches=1.0,
	    accumulate_grad_batches=1,
    )

    log = nl.NeMoLogger(
        name="structure_aware_starcoder2",
        log_dir=os.path.join(user_path, "log_dir"),
        ckpt=nl.ModelCheckpoint(
            save_last=True,
            monitor="val_loss",
            save_top_k=2,
            every_n_train_steps=50,
            mode="min",
            always_save_context=True,
            filename="structure_aware_starcoder2-{val_loss:.2f}-{step}-{consumed_samples}",
        ),
        extra_loggers=[
            SafeMLFlowLogger(
                experiment_name="structure_aware_starcoder2",
                run_name="structure_aware_starcoder2_run",
                tracking_uri="http://ec2-18-208-185-48.compute-1.amazonaws.com:5000",
            )
        ],
    )

    resume = nl.AutoResume(
        restore_config=RestoreConfig(
            path=os.path.join(user_path, 'local_checkpoint_path'),
            load_model_state=True,
            load_optim_state=False,
            load_artifacts=False,
        ),
    )

    optim = nl.MegatronOptimizerModule(
        config=OptimizerConfig(
            optimizer='adam',
            lr=1e-3,
            use_distributed_optimizer=True,
        ),
        lr_scheduler=nl.lr_scheduler.CosineAnnealingScheduler()
    )

    tokenizer = get_nmt_tokenizer(library='huggingface', model_name='bigcode/starcoder2-3b', use_fast=True)

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=log,
        resume=resume,
        optim=optim,
        tokenizer=tokenizer
    )
