import os

from data_preprocessing.tasks.code_completion import CodeCompletion
from data_preprocessing.datasets.code_search_net import CodeSearchNet
from data_preprocessing.datasets.cornstack import CornStack
from data_preprocessing.datasets.stack import Stack
from structure_aware_customization.model.structure_aware_starcoder2_model import StructureAwareStarcoder2Model
from structure_aware_customization.model.structure_aware_starcoder2_config import StructureAwareStarcoder2Config
from structure_aware_customization.dataset.eval.structure_aware_eval_cc_dataset import StructureAwareEvalCCDataset
from structure_aware_customization.dataset.structure_aware_data_module import StructureAwareDataModule

from megatron.core.optimizer import OptimizerConfig

from nemo.collections import llm
from nemo import lightning as nl
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.pytorch.strategies.utils import RestoreConfig


if __name__ == "__main__":
    user_path = "/shared/home/xxx"

    task = CodeCompletion()
    train_datasets = [Stack(task=task, split="train"), CodeSearchNet(task=task, split="train"), CornStack(task=task, split="train")]
    validation_datasets = [Stack(task=task, split="validation"), CodeSearchNet(task=task, split="validation"), CornStack(task=task, split="validation")]
    test_datasets = [Stack(task=task, split="test"), CodeSearchNet(task=task, split="test"), CornStack(task=task, split="test")]
    train_ds = StructureAwareEvalCCDataset(datasets=train_datasets)
    validation_ds = StructureAwareEvalCCDataset(datasets=validation_datasets)
    test_ds = StructureAwareEvalCCDataset(datasets=test_datasets)

    global_batch_size = 1
    num_epochs = 3

    data = StructureAwareDataModule(
        train_dataset=train_ds,
        validation_dataset=validation_ds,
        test_dataset=test_ds,
        micro_batch_size=1,
        global_batch_size=global_batch_size,
        seq_length=task.max_seq_len,
        num_train_samples=train_ds.num_samples,
        num_val_samples=validation_ds.num_samples,
        num_test_samples=test_ds.num_samples
    )

    model = StructureAwareStarcoder2Model(
        config=StructureAwareStarcoder2Config(),
        task=task,
    )

    trainer = nl.Trainer(
        num_nodes=1,
	    devices=1,
        max_epochs=num_epochs,
        max_steps=(train_ds.num_samples * num_epochs) // global_batch_size,
        accelerator="gpu",
        strategy=nl.MegatronStrategy(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=1,
            sequence_parallel=False,
            expert_model_parallel_size=1,
        ),
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed"
        ),
	    log_every_n_steps=50,
        val_check_interval=500,
        limit_val_batches=0.5,
	    accumulate_grad_batches=1,
    )

    log = nl.NeMoLogger(
        name="structure_aware_starcoder2",
        log_dir=os.path.join(user_path, os.path.join(user_path, "testing/log_dir")),
        ckpt=nl.ModelCheckpoint(
            save_last=True,
            monitor="val_loss",
            save_top_k=2,
            mode="min",
            save_optim_on_train_end=True,
            always_save_context=True,
            save_context_on_train_end=True,
            filename="structure_aware_starcoder2-{val_loss:.2f}-{step}-{consumed_samples}",
        ),
    )

    resume = nl.AutoResume(
        restore_config=RestoreConfig(
            path=os.path.join(user_path, 'path_finetuned_model'),
            load_model_state=True,
            load_optim_state=False,
            load_artifacts=False,
        ),
    )

    optim = nl.MegatronOptimizerModule(
        config=OptimizerConfig(
            optimizer='adam',
            lr=3e-5,
            use_distributed_optimizer=True,
        ),
        lr_scheduler=nl.lr_scheduler.CosineAnnealingScheduler(
            max_steps=(train_ds.num_samples * num_epochs) // global_batch_size,
            warmup_steps=1500,
            constant_steps=5000,
            min_lr=3e-6,
        )
    )

    tokenizer = get_nmt_tokenizer(library='huggingface', model_name='bigcode/starcoder2-3b', use_fast=True)

    llm.validate(
        model=model,
        data=data,
        trainer=trainer,
        log=log,
        resume=resume,
        optim=optim,
        tokenizer=tokenizer
    )
