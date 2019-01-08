import argparse

import torch
import torch.optim as optim

from tqdm import tqdm
import numpy as np

import datasets
import networks
import plots
import utils

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train network!')
parser.add_argument('net', metavar='network', type=str)
parser.add_argument('dataset', metavar='dataset', type=str)
parser.add_argument('--lr', metavar='0.0001', type=float, help='learning rate', default=0.0001)
parser.add_argument('--epochs', metavar='10', type=int, help='no. epochs', default=1)
parser.add_argument('--batch-size', metavar='32', type=int, help='batch size', default=32)
parser.add_argument('--output', metavar='./tmp', type=str, help='output directory', default='./tmp')
parser.add_argument('--log-interval', metavar='50', type=int, help='logging interval', default=50)

args = parser.parse_args()

add_prefix = lambda x: '%s/%s-%s-%s' % (args.output, args.net, args.dataset, x)

# Prepare dataset
stats_collector = dict(
    train_loss=[],
    train_acc=[],
    val_loss=[],
    val_acc=[]
)

net = networks.net_provider(args.net)

optimizer = optim.Adam(net.parameters(), lr=args.lr)

data = datasets.get_dataset(net.loss, args.dataset, batch_size=args.batch_size)
train_loader, test_loader, original_train_loader, original_test_loader = data

total_loop = args.epochs * len(list(train_loader))
no_test_samples = len(list(test_loader))

with tqdm(total=total_loop) as pbar:
    status = dict(
        epoch=0,
        train_loss=0.0,
        val_loss=-1,
    )
    pbar.set_postfix(status)

    for epoch in range(args.epochs):
        status['epoch'] = epoch
        sample_seen = 0.0
        running_loss = 0.0
        running_acc = 0.0

        # train
        net.train()
        for i, data in enumerate(train_loader):
            samples_in_batch = data[0].shape[0]

            optimizer.zero_grad()

            data = map(lambda x: x.to(utils.device), data)
            res = net(*data)
            loss = net.loss(*res)
            loss.backward()
            optimizer.step()

            loss_np = loss.cpu().detach().numpy()

            if np.isnan(loss_np):
                print('found nana')
                raise SystemExit('found nana')

            running_loss += loss_np * samples_in_batch
            sample_seen += samples_in_batch

            # print statistics
            pbar.update(1)

            if i % args.log_interval == (args.log_interval - 1):
                status['train_loss'] = running_loss / sample_seen
                pbar.set_postfix(status)

        status['train_loss'] = running_loss / sample_seen

        net.eval()
        test_loss = 0.0
        total_test_samples = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                samples_in_batch = data[0].shape[0]

                data = map(lambda x: x.to(utils.device), data)
                res = net(*data)
                loss = net.loss(*res)

                test_loss += (loss.cpu().detach().numpy()) * samples_in_batch
                total_test_samples += samples_in_batch

        status['val_loss'] = test_loss / total_test_samples

        pbar.set_postfix(status)

        # collect stats
        stats_collector['train_loss'].append(status['train_loss'])
        stats_collector['val_loss'].append(status['val_loss'])

# Produce artifacts
print('Saving artifacts to %s' % add_prefix('*'))
plots.plot_stats(stats_collector, filename=add_prefix('stats.png'))

plots.plot_embedding(*utils.get_embedding(net, original_train_loader),
                     filename=add_prefix('training-set-embedding.png'),
                     title='Training Set Embedding')
plots.plot_embedding(*utils.get_embedding(net, original_test_loader),
                     filename=add_prefix('testing-set-embedding.png'),
                     title='Test Set Embedding')

