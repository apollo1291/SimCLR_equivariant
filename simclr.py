import logging
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']

        # Wrap the model with DataParallel for multi-GPU support
        self.model = nn.DataParallel(kwargs['model']).to("cuda:0")
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(log_dir="../simclr_base_logs")
        print(self.writer.log_dir)

        logging.basicConfig(
            filename=os.path.join(self.writer.log_dir, 'simclr_training.log'), 
            level=logging.DEBUG
        )

        # Criterion should also be on the GPU
        self.criterion = torch.nn.CrossEntropyLoss().to("cuda:0")

    def info_nce_loss(self, features):
        # Labels and normalization remain the same
        labels = torch.cat([torch.arange(self.args.batch_size) for _ in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to("cuda:0")

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # Remove diagonal (self-similarity) and reshape
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to("cuda:0")
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # Extract positives and negatives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to("cuda:0")
        logits = logits / self.args.temperature

        return logits, labels

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # Save configuration
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, _, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0).to("cuda:0")

                with autocast(enabled=self.args.fp16_precision):
                    # Pass through the model (DataParallel handles multi-GPU splitting)
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
                   
                    self.writer.flush()

                n_iter += 1

            # Warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # Save model checkpoint
        checkpoint_name = f'checkpoint_{self.args.epochs:04d}.pth.tar'
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")


        classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1000)
        ).to("cuda:0")

        for epoch_counter in range(self.args.epochs):
            for images, _, labels in tqdm(train_loader):
                images = images[0]#torch.cat(images, dim=0)

                images = images.to("cuda:0")
                labels = torch.tensor(labels).to("cuda:0")

                with autocast(enabled=self.args.fp16_precision):
                    with torch.no_grad():
                        features = self.model(images)
                        features = torch.clone(features.detach())
                    logits = classifier(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('probe_loss', loss, global_step=n_iter)
                    self.writer.add_scalar('linear_class_acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('linear_class_acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('probe_learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
                    self.writer.flush()

                    #vol.commit()

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'classifier_checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': classifier.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Classifier model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        
