from data_preprocessing.tasks.pretraining import Pretraining
from data_preprocessing.datasets.code_search_net import CodeSearchNet
from structure_aware_customization.model.structure_aware_starcoder2_config import StructureAwareStarcoder2Config
from structure_aware_customization.dataset.structure_aware_data_module import StructureAwareDataModule
from structure_aware_customization.model.structure_aware_starcoder2_model import StructureAwareStarcoder2Model
from structure_aware_customization.dataset.structure_aware_pretraining_dataset import StructureAwarePretrainingDataset
from structure_aware_customization.dataset.structure_aware_ct_dataset import StructureAwareCTDataset

from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.pytorch.strategies.utils import RestoreConfig


if __name__ == "__main__":
    task = Pretraining()
    train_ds = StructureAwarePretrainingDataset(dataset=CodeSearchNet(task=task, split="train"))
    validation_ds = StructureAwarePretrainingDataset(dataset=CodeSearchNet(task=task, split="validation"))
    test_ds = StructureAwarePretrainingDataset(dataset=CodeSearchNet(task=task, split="test"))

    data = StructureAwareDataModule(train_dataset=train_ds,
                                    validation_dataset=validation_ds,
                                    test_dataset=test_ds,
                                    micro_batch_size=4,
                                    global_batch_size=16,
                                    seq_length=task.max_seq_len,
                                    num_train_samples=train_ds.num_samples,
                                    num_val_samples=validation_ds.num_samples,
                                    num_test_samples=test_ds.num_samples)

    model = StructureAwareStarcoder2Model(config=StructureAwareStarcoder2Config())

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
	    virtual_pipeline_model_parallel_size=None,
	    context_parallel_size=1,
	    sequence_parallel=False,
        expert_model_parallel_size=1
    )

    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=0.001,
	    use_distributed_optimizer=True
    )
    opt = nl.MegatronOptimizerModule(config=opt_config, lr_scheduler=nl.lr_scheduler.CosineAnnealingScheduler())

    trainer = nl.Trainer(
        num_nodes=1,
	    devices=4,
        max_epochs=2,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
	    log_every_n_steps=1,
	    accumulate_grad_batches=1,
    )

    nemo_logger = nl.NeMoLogger(
        log_dir="logdir", # logs and checkpoints will be written here
    )

    tokenizer = get_nmt_tokenizer(library='huggingface', model_name='bigcode/starcoder2-3b', use_fast=True)

    restore_config = RestoreConfig(
        path='local_checkpoint_path',
        load_model_state=True,
        load_optim_state=False,
        load_artifacts=False,
    )

    resume = nl.AutoResume(
        restore_config=restore_config,
    )

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        resume=resume,
        tokenizer=tokenizer,
        optim=opt,
    )
