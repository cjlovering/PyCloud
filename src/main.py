from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import os
from argparse import ArgumentParser

from train import train_mnist, evaluate_mnist, train_snli

from timeit import default_timer as timer

from models import MnistNet
from models import SNLINet

def get_args():
    parser = ArgumentParser(description='')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_embed', type=int, default=100)
    parser.add_argument('--d_proj', type=int, default=300)
    parser.add_argument('--d_hidden', type=int, default=300)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--no-projection', action='store_false', dest='projection')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.6B.100d')
    parser.add_argument('--resume_snapshot', type=str, default='')


    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--capture-results', type=bool, default=False,
                        help='times each process (default: True)')
    parser.add_argument('--restarts', type=int, default=1)

    args = parser.parse_args()
    return args

class Evaluate(object):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def train(self, args, model_class: nn.Module, train) -> nn.Module:
        """Trains a model locally.

        Args:
            model_class (nn.Module): The class of a PyTorch model.
            train: A function that trains the model instance locally.
                It must also encapsulate the dataloading.
        """

        torch.manual_seed(args.seed)

        start = timer()

        model = train(args=args, model_class=model_class)

        end = timer()
        t = end - start

        return t

if __name__ == '__main__':
    # Training settings
    args = get_args()

    eval = Evaluate()
    tasks = [
        ('MNIST', MnistNet, train_mnist),
        ('SNLI', SNLINet, train_snli)
    ]

    for _ in range(args.restarts):
        for task in tasks:
            task_title, model_class, train = task
            result = eval.train(args, model_class, train)
            with open("results/results.csv", "a") as f:
                f.write("{}, {}\n".format(task_title, result))