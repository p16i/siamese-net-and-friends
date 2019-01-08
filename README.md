# Siamese networks and Friends (in progress)


<div align="center">
    <img src="https://via.placeholder.com/500x500"/>
</div>

## Introduction

## Architectures

## Command


## Results

| Network      | MNIST           | FashionMNIST  |
|:-------------:|:-------------:| :-----:|
| EmbeddingNet <br> (CrossEntropyLoss) | ![emb_mnist] | ![emb_fmnist] |
| SiameseNet   <br> (ContrastiveLoss)  | ![sc_mnist]     | TBD |
| SiameseNet   <br> (BinaryCrossEntropyLoss)   | ![img_placeholder]      | TBD |
| SiameseNet   <br> (TripetLoss)   | ![tp_mnist]      | TBD |


## Acknowledgements
- @adambielski's [siamese-tripet](https://github.com/adambielski/siamese-triplet): experiments here are mainly based on his experiments.



[emb_mnist]: https://raw.githubusercontent.com/heytitle/siamese-net-and-friends/master/output/embedding-classification-MNIST-testing-set-embedding.png
[emb_fmnist]: https://raw.githubusercontent.com/heytitle/siamese-net-and-friends/master/output/embedding-classification-FashionMNIST-testing-set-embedding.png
[sc_mnist]: https://raw.githubusercontent.com/heytitle/siamese-net-and-friends/master/output/siamese-constrastive-MNIST-testing-set-embedding.png
[tp_mnist]: https://raw.githubusercontent.com/heytitle/siamese-net-and-friends/master/output/tripet-loss-net-MNIST-testing-set-embedding.png

[img_placeholder]: https://via.placeholder.com/500x500

[siamese-paper]: http://something
[constrastive-loss]: http://something
