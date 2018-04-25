import os
import time
import glob

from argparse import ArgumentParser

import torch
import torch.optim as O
import torch.nn as nn

from torchtext import data
from torchtext import datasets

def download_snli(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    inputs = data.Field(lower=args.lower)
    answers = data.Field(sequential=False)

    train, dev, test = datasets.SNLI.splits(inputs, answers)

    inputs.build_vocab(train, dev, test)
    if args.word_vectors:
        if os.path.isfile(args.vector_cache):
            inputs.vocab.vectors = torch.load(args.vector_cache)
        else:
            inputs.vocab.load_vectors(args.word_vectors)
            torch.save(inputs.vocab.vectors, args.vector_cache)
    answers.build_vocab(train)

    if torch.cuda.is_available():
        train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=args.batch_size, device=args.gpu)
    else:
        train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=args.batch_size, device=-1)
    config = args
    config.n_embed = len(inputs.vocab)
    config.d_out = len(answers.vocab)
    config.n_cells = config.n_layers

    # double the number of cells for bidirectional networks
    if config.birnn:
        config.n_cells *= 2

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

if __name__ == "__main__":
    download_snli(get_args())