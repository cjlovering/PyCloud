from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from train import train_mnist, evaluate_mnist

from timeit import default_timer as timer

from models import MnistNet
from models import SNLINet

class Evaluate(object):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def train(self, args, model_class: nn.Module, train, evaluate) -> nn.Module:
        """Trains a model locally.

        Args:
            model_class (nn.Module): The class of a PyTorch model.
            train: A function that trains the model instance locally.
                It must also encapsulate the dataloading.
        """

        torch.manual_seed(args.seed)

        start = timer()

        model = model_class()
        train(args=args, model=model)
        acc, loss = evaluate(args, model)

        end = timer()
        t = end - start

        return '{} {} {}'.format(acc, loss, t)

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                        help='how many training processes to use (default: 2)')
    parser.add_argument('--capture-results', type=bool, default=False,
                        help='times each process (default: True)')
    
    args = parser.parse_args()

    eval = Evaluate()
    print(args)
    iterations = 3
    tasks = [('MNIST', MnistNet, train_mnist, evaluate_mnist)]

    for task in tasks:
        task_title, model, train, evaluate = task
        result = eval.train(args, model, train, evaluate)
        print("{}, {}".format(task_title, result))  # TODO: log or store results, etc
