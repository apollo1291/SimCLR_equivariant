import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDatasetWithParams,  transformation_params_to_tensor_batch,  params_collate_fn
from models.resnet_simclr import ResNetSimCLR
from models.vits import KQConModel, ViT
from simclr import SimCLR
import modal

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
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

vol = modal.Volume.from_name("simclr", create_if_missing=True)
app = modal.App()

train_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "absl-py==1.4.0",
    "cachetools==5.3.1",
    "certifi==2023.5.7",
    "chardet==5.1.0",
    "google-auth==2.19.1",
    "google-auth-oauthlib==1.0.0",
    "grpcio==1.54.2",
    "idna==3.4",
    "markdown==3.4.3",
    "numpy==1.24.3",
    "oauthlib==3.2.2",
    "pillow==9.5.0",
    "protobuf==4.23.2",
    "pyasn1==0.5.0",
    "pyasn1-modules==0.3.0",
    "torch==2.0.1",
    "pyyaml==6.0",
    "requests==2.31.0",
    "requests-oauthlib==1.3.1",
    "rsa==4.9",
    "six==1.16.0",
    "tensorboard==2.13.0",
    "torchvision==0.15.2",
    "urllib3==1.26.3",
    "werkzeug==2.3.4",
    "aiohappyeyeballs==2.4.3",
    "aiohttp==3.10.10",
    "aiosignal==1.3.1",
    "aiostream==0.5.2",
    "annotated-types==0.7.0",
    "anyio==4.6.2.post1",
    "async-timeout==4.0.3",
    "attrs==24.2.0",
    "charset-normalizer==3.4.0",
    "click==8.1.7",
    "exceptiongroup==1.2.2",
    "fastapi==0.115.2",
    "filelock==3.16.1",
    "frozenlist==1.4.1",
    "fsspec==2024.9.0",
    "grpclib==0.4.7",
    "h2==4.1.0",
    "hpack==4.0.0",
    "huggingface-hub==0.25.2",
    "hyperframe==6.0.1",
    "info-nce-pytorch==0.1.4",
    "Jinja2==3.1.4",
    "markdown-it-py==3.0.0",
    "MarkupSafe==3.0.1",
    "mdurl==0.1.2",
    "modal==0.64.199",
    "mpmath==1.3.0",
    "multidict==6.1.0",
    "networkx==3.4.1",
    "packaging==24.1",
    "propcache==0.2.0",
    "pydantic==2.9.2",
    "pydantic_core==2.23.4",
    "Pygments==2.18.0",
    "rich==13.9.2",
    "safetensors==0.4.5",
    "shellingham==1.5.4",
    "sigtools==4.0.1",
    "sniffio==1.3.1",
    "starlette==0.40.0",
    "sympy==1.13.3",
    "synchronicity==0.8.2",
    "tensorboard-data-server==0.7.1",
    "timm==1.0.9",
    "toml==0.10.2",
    "tqdm==4.66.5",
    "typer==0.12.5",
    "types-certifi==2021.10.8.3",
    "types-toml==0.10.8.20240310",
    "typing_extensions==4.12.2",
    "watchfiles==0.24.0",
    "yarl==1.15.5"
)


class SimCLRConfig:
    def __init__(self):
        self.data = './datasets'
        self.dataset_name = 'cifar10'
        self.arch = 'resnet18'
        self.workers = 8
        self.epochs = 200
        self.batch_size = 128
        self.lr = 0.0003
        self.weight_decay = 1e-4
        self.seed = None
        self.disable_cuda = False
        self.fp16_precision = False
        self.out_dim = 128
        self.log_every_n_steps = 100
        self.temperature = 0.07
        self.n_views = 2
        self.gpu_index = 0
        self.model = 'base_vit'
        self.use_fourier = False

models = {
    'resnet': ResNetSimCLR,
    'base_vit': ViT
}

@app.function(gpu="a100", image=train_image, volumes={"/runs/simclr": vol}, timeout=84600)
def main():
    #args = parser.parse_args()
    args = SimCLRConfig()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ContrastiveLearningDatasetWithParams(args.data)

    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=params_collate_fn)

    #vit = ViT()
    #KQConModel(vit)
    for images, params_list in train_loader:
    # images is a list of tensors [img1, img2]
    # params_list is a list of dicts [{params_img1}, {params_img2}]
        #print(len(images), images[0].shape, images[1].shape)
       #print(len(params_list[0]["color_jitter_applied"]))
        x1 = images[0]
        x2 = images[1]

        t1 = params_list[0]
        t2 = params_list[1]
        #print(t1)

        # Convert transformation parameters dict to tensor
        t1_tensor = transformation_params_to_tensor_batch(t1)
        t2_tensor = transformation_params_to_tensor_batch(t2)
        #print(t1_tensor.shape)
    
    model = models[args.model]() # ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    # #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        # simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        # simclr.train(train_loader, vol)
        KQCon = KQConModel(model, args=args, optimizer=optimizer, scheduler=scheduler)
        KQCon.train(train_loader, vol, use_fourier=args.use_fourier)
    import subprocess
    import os
    token = "as"  # Generate a secure token
    logdir = "/runs/simclr"  # Replace with the actual log directory for TensorBoard
    
    # Expose TensorBoard through a tunnel
    with modal.forward(6006) as tunnel:
        url = tunnel.url + "/?token=" + token
        print(f"Starting TensorBoard at {url}")
        
        # Run TensorBoard with the specified options
        subprocess.run(
            [
                "tensorboard",
                "--logdir", logdir,
                "--port", "6006",
                "--bind_all",
                "--reload_multifile=true",  # Useful if logs are spread across multiple files
            ],
            env={**os.environ, "TENSORBOARD_AUTH_TOKEN": token, "SHELL": "/bin/bash"},
            stderr=subprocess.DEVNULL,
        )
    try:
        while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("Terminating app")
    

# @app.function(
#     image=train_image,
#     volumes={"/runs/simclr": vol}
# )
# @modal.wsgi_app()
# def tensorboard_app():
#     import tensorboard

#     board = tensorboard.program.TensorBoard()
#     board.configure(logdir="/runs/simclr")
#     (data_provider, deprecated_multiplexer) = board._make_data_provider()
#     wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
#         board.flags,
#         board.plugin_loaders,
#         data_provider,
#         board.assets_zip_provider,
#         deprecated_multiplexer,
#     )
#     return wsgi_app

# @app.function(image=train_image, volumes={"/runs/simclr": vol})
# def run_tensorboard():

if __name__ == "__main__":
    
    with modal.enable_output():
        with app.run():
            main.remote()
            print("done")
            run_tensorboard.remote()

            # Run TensorBoard as a background process
            # subprocess.Popen([
            #     "tensorboard",
            #     "--logdir", "/runs/simclr",
            #     "--host", "0.0.0.0",
            #     "--port", "6006"
            # ])
            import time
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Terminating app")
            