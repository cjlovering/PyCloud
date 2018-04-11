from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from train import train, evaluate

from timeit import default_timer as timer

from .. import MnistNet


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

class Hogwild(object):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def train(self, args, model_class: nn.Module, train) -> nn.Module:
        """Trains a model locally using Hogwild!

        Args:
            model_class (nn.Module): The class of a PyTorch model.
            train: A function that trains the model instance locally.
                It must also encapsulate the dataloading.
        """

        torch.manual_seed(args.seed)

        model = model_class()
        model.share_memory() # gradients are allocated lazily, so they are not shared here

        processes = []

        if args.capture_results:
            start = timer()

        for rank in range(args.num_processes):
            p = mp.Process(target=train, args=(rank, args, model))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        acc, loss = evaluate(args, model)

        if args.capture_results:
            end = timer()
            acc, loss = evaluate(args, model)
            t = end - start
            print('{} {} {}'.format(acc, loss, t))
    
        return model


if __name__ == '__main__':
    args = parser.parse_args()
    local_distributed =  Hogwild()
    local_distributed.train(args, MnistNet, train)
