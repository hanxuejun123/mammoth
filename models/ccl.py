# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from copy import deepcopy
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Information Propagation.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--w', type=float, required=True,
                        help='Re-representation weight.')
    parser.add_argument('--eta', type=float, required=True,
                        help='Hyperparameter of the attention score.')
    parser.add_argument('--tau', type=float, required=True,
                        help='Temperature of the contrastive loss function.')
    parser.add_argument('--alpha', type=float, required=True,
                        help='Trade-off parameter of contrastive loss.')
    return parser



class CCL(ContinualModel):
    NAME = 'ccl'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(CCL, self).__init__(backbone, loss, args, transform)
        self.old_net = None
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.buffer = Buffer(self.args.buffer_size, self.device)


    def end_task(self, dataset) -> None:
        self.old_net = deepcopy(self.net.eval())
        self.net.train()


    def observe(self, inputs, labels, not_aug_inputs, logits=None):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()

        if self.buffer.is_empty():
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels) 

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

            if not self.old_net is None:
                feats = self.net.features(inputs)
                old_feats = self.old_net.features(inputs)
                sim = -torch.cdist(feats, old_feats)
                A = self.soft(self.args.eta * sim) 
                weighted_old_feats = torch.matmul(A.unsqueeze(1), old_feats).squeeze(1)
                m_feats = (1 - self.args.w) * feats + self.args.w * weighted_old_feats

                outputs = self.net.classifier(m_feats)
                loss = self.loss(outputs, labels)

                loss += - self.args.alpha * torch.mean(torch.diag(self.logsoft(-torch.cdist(feats, old_feats) * self.args.tau)))

            else:
                outputs = self.net(inputs)
                loss = self.loss(outputs, labels)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

       
        return loss.item()
