from transformers import TrainingArguments, Trainer

from snow.config import SnowConfig, DataConfig, TrainConfig
from snow.data.data_pipeline import DataPipeline
from snow.model.snow_model import SnowModel


def run_train(model_config: SnowConfig, data_config: DataConfig, train_config: TrainConfig):
    # Step 1 Create datasets, processor, collator
    data_pipeline = DataPipeline(data_config)

    # Step 2. load pre-trained model
    model = SnowModel.from_pretrained(model_config.model_path)

    # Step 3. Configuration for training
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        max_steps=train_config.max_steps,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=1e-5,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        tf32=True,
        gradient_checkpointing=False,
        optim="adamw_torch_fused",
        dataloader_num_workers=train_config.dataloader_num_workers,
        report_to="tensorboard",
        seed=data_config.seed,
        deepspeed=train_config.deepspeed_config,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        eval_strategy="no",
        eval_steps=500,
        batch_eval_metrics=True,
        remove_unused_columns=False,
        ignore_data_skip=True,
        accelerator_config={"split_batches": True},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_pipeline.collator,
        train_dataset=data_pipeline.dataset,
    )

    # Start train and save model
    trainer.train()
    trainer.save_model()