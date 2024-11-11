import argparse
import os
import torch
from datetime import datetime
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDatasetWithParams, transformation_params_to_tensor_batch, params_collate_fn
from models.resnet_simclr import ResNetSimCLR
from models.vits import KQConModel, ViT
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR with PyTorch Lightning')
parser.add_argument('-train_data', metavar='DIR', default='../../pytorch-data/imagenet2/ILSVRC/Data/CLS-LOC/train',
                    help='path to train dataset')
parser.add_argument('-val_data', metavar='DIR', default='../../pytorch-data/imagenet2/ILSVRC/Data/CLS-LOC/val2',
                    help='path to val dataset')
parser.add_argument('-dataset-name', default='imagenet',
                    help='dataset name', choices=['stl10', 'cifar10', 'imagenet'])
parser.add_argument('-models', '--arch', metavar='ARCH', default='base_vit',
                    choices=['resnet', 'base_vit'],
                    help='model architecture: resnet | base_vit (default: base_vit)')
parser.add_argument('-j', '--workers', default=23, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=48, type=int,
                    metavar='N',
                    help='mini-batch size (default: 48)')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='GPU index.')
parser.add_argument('--use-fourier', action='store_true', help='Use Fourier encoding')

MODELS = {
    'resnet': ResNetSimCLR,
    'base_vit': ViT
}

def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    pl.seed_everything(args.seed)

    # Data preparation
    dataset = ContrastiveLearningDatasetWithParams(args.train_data)
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=params_collate_fn)

    
    val_dataset = ContrastiveLearningDatasetWithParams(
        args.val_data
    ).get_dataset(args.dataset_name, args.n_views)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=params_collate_fn
    )

    
    model = MODELS[args.arch]()
    lightning_model = KQConModel(model=model, args=args)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = pl.loggers.TensorBoardLogger(save_dir=f"/datadrive/ellington/logs/{args.arch}_fe={args.use_fourier}_{current_time}")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='train_loss',
        mode='min',
        filename='{epoch}-{train_loss:.2f}',
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=torch.cuda.device_count(),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else None,
        logger=logger,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=1)],
        precision=16 if args.fp16_precision else 32,
        enable_progress_bar=True,
    )


    trainer.fit(lightning_model, train_loader, val_loader)

if __name__ == '__main__':
    main()