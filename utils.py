import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_embedding(model, loader):
    embedding = []
    labels = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            x, y = data
            x = x.to(device)
            b_embedding = model.get_embedding(x).cpu().detach().numpy()

            labels.append(y.numpy())
            embedding.append(b_embedding)
    return np.vstack(embedding), np.concatenate(labels)
