import os
import torch
import wandb
from models import *
from dataset import *
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from jsonargparse import lazy_instance
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.trainer import Trainer
from datetime import timedelta


def cli_main():
    # remove slurm env vars due to this issue:
    if 'SLURM_NTASKS' in os.environ:
        del os.environ["SLURM_NTASKS"]
    if 'SLURM_JOB_NAME' in os.environ:
        del os.environ["SLURM_JOB_NAME"]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('medium')

    wandb_id = os.environ.get('WANDB_RUN_ID', wandb.util.generate_id())
    exp_dir = os.path.join('logs', wandb_id)
    os.makedirs(exp_dir, exist_ok=True)
    wandb_logger = lazy_instance(
        WandbLogger,
        project='SE360',
        id=wandb_id,
        save_dir=exp_dir
        )

    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,
        # every_n_train_steps=1000, 
        train_time_interval=timedelta(minutes=10),  
        )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    class MyLightningCLI(LightningCLI):
        def before_instantiate_classes(self):
            # Handle --ckpt parameter
            subcommand = self.config.subcommand
            if hasattr(self.config[subcommand], 'ckpt'):
                print(f"DEBUG: Found ckpt in config: {self.config[subcommand].ckpt}")
                # Override ckpt_path with the command-line --ckpt value (could be None or a path)
                self.config[subcommand].model.init_args.ckpt_path = self.config[subcommand].ckpt
                print(f"DEBUG: Set model.init_args.ckpt_path to: {self.config[subcommand].model.init_args.ckpt_path}")
            else:
                print("DEBUG: ckpt not found in config")
            
            # Handle --result_dir parameter
            if hasattr(self.config[subcommand], 'result_dir') and self.config[subcommand].result_dir is not None:
                self.config[subcommand].data.init_args.result_dir = self.config[subcommand].result_dir
            
            # set result_dir, data and pano_height for evaluation
            if self.config.get('test', {}).get('model', {}).get('class_path') == 'models.EvalPanoGen':
                if self.config.test.data.init_args.result_dir is None:
                    result_dir = os.path.join(exp_dir, 'test')
                    self.config.test.data.init_args.result_dir = result_dir
                self.config.test.model.init_args.data = self.config.test.data.class_path.split('.')[-1]
                self.config.test.model.init_args.pano_height = self.config.test.data.init_args.pano_height
                self.config.test.data.init_args.val_batch_size = 1

        def add_arguments_to_parser(self, parser):
            parser.link_arguments("model.init_args.cam_sampler", "data.init_args.cam_sampler")
            # Add --ckpt shorthand for checkpoint path
            parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint file")
            # Add --result_dir shorthand for result directory
            parser.add_argument("--result_dir", type=str, default=None, help="Path to result directory")

    cli = MyLightningCLI(
        trainer_class=Trainer,
        save_config_kwargs={'overwrite': True},
        parser_kwargs={'parser_mode': 'omegaconf', 'default_env': True},
        seed_everything_default=os.environ.get("LOCAL_RANK", 0),
        trainer_defaults={
            'strategy': 'ddp',
            'devices': [0],
            'log_every_n_steps': 10,
            'num_sanity_val_steps': 0,
            'limit_val_batches': 4,
            'benchmark': True,
            'max_steps': 10000,  # Set maximum iteration count, replacing max_epochs
            # 'max_epochs': 5,   # Comment out epoch-based settings
            'accumulate_grad_batches': 1,  # Gradient accumulation: accumulate gradients every 1 batch before updating parameters
            'precision': 'bf16-mixed',
            'callbacks': [lr_monitor, checkpoint_callback],#
            'logger': wandb_logger,
            # 'gradient_clip_val': 1.0,  # New: set gradient clipping value
            # 'gradient_clip_algorithm': 'norm'  # New: set gradient clipping algorithm
        })


if __name__ == '__main__':
    cli_main()

