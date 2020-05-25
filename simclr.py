import torch
import time
from models.resnet_simclr import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
from loss.margin_triplet import MarginTripletLoss
from loss.nt_logistic import NTLogisticLoss
from LARS import LARS
import os
import shutil
import sys

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

import numpy as np

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class SimCLR(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.batch_size = config['batch_size']
        self.model = ResNetSimCLR(**self.config["model"]).to(self.device)
        self.loss_func = self._choose_loss()

    def _choose_loss(self):
        if self.config['loss_select'] == 'NT_Xent':
            print("using NT_Xent as loss func")
            return NTXentLoss(self.device, self.config['batch_size'], **self.config['loss'])
        elif self.config['loss_select'] == 'NT_Logistic':
            print("using NT_Logistic as loss func")
            return NTLogisticLoss(self.device, self.config['batch_size'], **self.config['loss'])
        elif self.config['loss_select'] == 'MarginTriplet':
            print("using MarginTriplet as loss func")
            return MarginTripletLoss(self.device, self.config['batch_size'], self.config['semi_hard'],
                                     **self.config['loss'])
        else:
            print('not a valid loss, use NT_Xent as default')
            return NTXentLoss(self.device, self.config['batch_size'], **self.config['loss'])

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs, n_iter):

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.loss_func(zis, zjs)
        return loss

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = self.model
        model = self._load_pre_trained_weights(model)

        if self.config['optimizer'] == 'LARS':
            print('using LARS as optimizer.')
            optimizer = LARS(model.parameters(), lr=0.3 * self.batch_size / 256, eta=1e-3,
                             weight_decay=eval(self.config['weight_decay']))
        elif self.config['optimizer'] == 'SGD':
            print('using SGD as optimizer. In order to obtain less space')
            optimizer = torch.optim.SGD(model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))
        else:
            print('using Adam as optimizer.')
            optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            start_time = time.time()
            print("now in epoch {0}".format(epoch_counter))
            for (xis, xjs), _ in train_loader:
                optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            with torch.no_grad():
                if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                    valid_loss = self._validate(model, valid_loader)
                    if valid_loss < best_valid_loss:
                        # save the model weights
                        best_valid_loss = valid_loss
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                    self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                    valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)
            end_time = time.time()
            print('In epoch {0}, time cost:{1}'.format(epoch_counter, end_time - start_time))

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs), _ in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
