import torch
import torch.nn as nn
import torch.nn.functional as F
import losses


def net_provider(name, latent_dims=2, **kwargs):
    embedding_net = Embedding(latent_dims=latent_dims)

    if name == 'embedding-classification':
        return Classifier(embedding_net, **kwargs)
    elif name == 'siamese-constrastive':
        return SiameseContrastive(embedding_net, **kwargs)
    elif name == 'siamese-binary-cross-entropy':
        return SiameseBinaryCrossEntropy(embedding_net, **kwargs)
    elif name == 'tripet-loss-net':
        return TripetLossModel(embedding_net, **kwargs)

    raise SystemExit('No %s net available' % name)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Classifier(nn.Module):
    def __init__(self, embedding_net, no_classes=10):
        super(Classifier, self).__init__()
        self.embedding = embedding_net
        self.classifier = nn.Sequential(
            nn.Linear(2, no_classes),
            nn.LogSoftmax(dim=1)
        )

        self.loss = F.nll_loss

    def forward(self, x, y):
        return self.classifier(self.get_embedding(x)), y

    def get_embedding(self, x):
        return self.embedding(x)


class Embedding(nn.Module):
    def __init__(self, latent_dims=2):
        super(Embedding, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.PReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, latent_dims),
        )

    def forward(self, x):
        return self.encoder(x)


class SiameseContrastive(nn.Module):
    def __init__(self, embedding_net, margin=1.0):
        super(SiameseContrastive, self).__init__()
        self.embedding = embedding_net
        self.loss = losses.contastive_loss
        self.margin = margin

    def forward(self, x, x_sampling, is_the_sample_class):
        z = self.get_embedding(x)
        zs = self.get_embedding(x_sampling)
        return z, zs, is_the_sample_class

    def get_embedding(self, x):
        return self.embedding(x)


class SiameseBinaryCrossEntropy(nn.Module):
    def __init__(self, embedding_net, latent_dims=2, margin=1.0):
        super(SiameseBinaryCrossEntropy, self).__init__()
        self.embedding = embedding_net

        self.output = nn.Sequential(
            nn.Linear(latent_dims, 1),
            nn.Sigmoid()
        )

        self.scaling_factor = nn.Parameter(torch.randn(2))

        self.margin = margin

        self.loss = losses.binary_cross_entropy

    def forward(self, x, x_sampling, is_the_sample_class):
        z = self.get_embedding(x)
        zs = self.get_embedding(x_sampling)

        # todo add scaling factor
        l1 = (z-zs).abs().mul(self.scaling_factor)
        p_same_class = self.output(l1)

        return p_same_class.view(-1), is_the_sample_class.view(-1)

    def get_embedding(self, x):
        return self.embedding(x)


class TripetLossModel(nn.Module):
    def __init__(self, embedding_net, margin=1.0):
        super(TripetLossModel, self).__init__()
        self.embedding = embedding_net
        self.loss = losses.tripet_loss
        self.margin = margin

    def forward(self, x, pos_x, neg_x):
        z = self.get_embedding(x)
        pz = self.get_embedding(pos_x)
        nz = self.get_embedding(neg_x)
        return z, pz, nz

    def get_embedding(self, x):
        return self.embedding(x)

