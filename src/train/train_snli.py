import os
import time
import glob

from argparse import ArgumentParser

import torch
import torch.optim as O
import torch.nn as nn

from torchtext import data
from torchtext import datasets

def train_snli(args, model_class):
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

    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=args.batch_size, device=args.gpu)

    config = args
    config.n_embed = len(inputs.vocab)
    config.d_out = len(answers.vocab)
    config.n_cells = config.n_layers

    # double the number of cells for bidirectional networks
    if config.birnn:
        config.n_cells *= 2

    model = model_class(config=config)
    if args.word_vectors:
        model.embed.weight.data = inputs.vocab.vectors
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    opt = O.Adam(model.parameters(), lr=args.lr)

    iterations = 0
    start = time.time()
    best_dev_acc = -1
    train_iter.repeat = False
    header = 'Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
    dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
    log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
    if not args.capture_results:
        print(header)

    for epoch in range(args.epochs):
        train_iter.init_epoch()
        n_correct, n_total = 0, 0
        for batch_idx, batch in enumerate(train_iter):

            # switch model to training mode, clear gradient accumulators
            model.train(); opt.zero_grad()
            iterations += 1

            # forward pass
            answer = model(batch)

            # calculate accuracy of predictions in the current batch
            n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
            n_total += batch.batch_size
            train_acc = 100. * n_correct/n_total

            # calculate loss of the network output with respect to training labels
            loss = criterion(answer, batch.label)
            # backpropagate and update optimizer learning rate
            loss.backward(); opt.step()

            if not args.capture_results and iterations % args.log_every == 0:
                print(log_template.format(time.time()-start,
                    epoch, iterations, 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx) / len(train_iter), loss.data[0], ' '*8, n_correct/n_total*100, ' '*12))

    return model