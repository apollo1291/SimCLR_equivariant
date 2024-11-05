import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDatasetWithParams,  transformation_params_to_tensor_batch,  params_collate_fn
from models.resnet_simclr import ResNetSimCLR
from models.vits import KQConModel, ViT
from simclr import SimCLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='../../pytorch-data/imagenet2/ILSVRC/Data/CLS-LOC/train',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='imagenet',
                    help='dataset name', choices=['stl10', 'cifar10', 'imagenet'])
parser.add_argument('-models', '--arch', metavar='ARCH', default='base_vit',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=48, type=int,
                    metavar='N',
                    help='mini-batch size (default: 48), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
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
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--use-fourier', default=False, type=bool, help='Whether to use fourier encoding')


# class SimCLRConfig:
#     def __init__(self):
#         self.data = './datasets'
#         self.dataset_name = 'cifar10'
#         self.arch = 'resnet18'
#         self.workers = 8
#         self.epochs = 200
#         self.batch_size = 128
#         self.lr = 0.0003
#         self.weight_decay = 1e-4
#         self.seed = None
#         self.disable_cuda = False
#         self.fp16_precision = False
#         self.out_dim = 128
#         self.log_every_n_steps = 100
#         self.temperature = 0.07
#         self.n_views = 2
#         self.gpu_index = 0
#         self.model = 'base_vit'
#         self.use_fourier = False

MODELS = {
    'resnet': ResNetSimCLR,
    'base_vit': ViT
}

NUM_TRANSFORMATION_PARAMS = 12


def main(rank, world_size, args):
    setup(rank, world_size)
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."

    dataset = ContrastiveLearningDatasetWithParams(args.data)

    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
    
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=params_collate_fn, sampler=sampler)

    model = MODELS[args.arch]().to(rank) # ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    KQCon = KQConModel(model,  args=args, optimizer=optimizer, scheduler=scheduler)
    KQCon.train(train_loader, use_fourier=args.use_fourier)
    cleanup()

if __name__ == "__main__":
    args = parser.parse_args()
    world_size = torch.cuda.device_count()  # Set the number of GPUs available
    torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
            
