# Siamese networks and Friends (in progress)


<div align="center">
<img src="https://github.com/heytitle/siamese-net-and-friends/blob/master/output/siamese-binary-cross-entropy-MNIST-latent-space-development.gif?raw=true"/> <br>
<b>Fig. 1: Latent space development throughout training epochs.
These dots are MNIST test samples. <br>
They are highlighted with the same color if they come from the same class.
</b>
</div>

## Introduction

## Architectures
<div align="center">
<img src="https://raw.githubusercontent.com/heytitle/siamese-net-and-friends/master/diagrams/architectures.png?raw=1"/>
<br>
<b> Fig. 2: xxx
</b>
</div>

## Command

## Results

| Network      | MNIST           | FashionMNIST  |
|:-------------:|:-------------:| :-----:|
| EmbeddingNet <br> (CrossEntropyLoss) | ![emb_mnist] | ![emb_fmnist] |
| SiameseNet   <br> (ContrastiveLoss)  | ![sc_mnist]     | ![sc_fmnist] |
| SiameseNet   <br> (BinaryCrossEntropyLoss)   | ![scb_mnist]      | ![scb_fmnist] |
| SiameseNet   <br> (TripetLoss)   | ![tp_mnist]      | ![tp_fmnist] |
| VAE (TBD) | | |


## Acknowledgements
- @adambielski's [siamese-tripet](https://github.com/adambielski/siamese-triplet): experiments here are mainly based on his experiments.

## References
1. Koch, G.R. (2015). [Siamese Neural Networks for One-Shot Image Recognition.][siamnet]
2. Hadsell, R., Chopra, S., & LeCun, Y. (2006). [Dimensionality reduction by learning an invariant mapping.][contrastive-loss] In Proceedings of the IEEE conference on CVPR (pp. 1735-1742).
3. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). [Facenet: A unified embedding for face recognition and clustering.][tripet-loss] In Proceedings of the IEEE conference on CVPR (pp. 815-823).


[architecures]: https://raw.githubusercontent.com/heytitle/siamese-net-and-friends/master/diagrams/architectures.png?raw=1
[emb_mnist]: https://raw.githubusercontent.com/heytitle/siamese-net-and-friends/master/output/embedding-classification-MNIST-testing-set-embedding.png
[emb_fmnist]: https://raw.githubusercontent.com/heytitle/siamese-net-and-friends/master/output/embedding-classification-FashionMNIST-testing-set-embedding.png
[sc_mnist]: https://raw.githubusercontent.com/heytitle/siamese-net-and-friends/master/output/siamese-constrastive-MNIST-testing-set-embedding.png
[sc_fmnist]: https://raw.githubusercontent.com/heytitle/siamese-net-and-friends/master/output/siamese-constrastive-FashionMNIST-testing-set-embedding.png
[scb_mnist]: https://raw.githubusercontent.com/heytitle/siamese-net-and-friends/master/output/siamese-binary-cross-entropy-MNIST-testing-set-embedding.png
[scb_fmnist]: https://raw.githubusercontent.com/heytitle/siamese-net-and-friends/master/output/siamese-binary-cross-entropy-FashionMNIST-testing-set-embedding.png
[tp_mnist]: https://raw.githubusercontent.com/heytitle/siamese-net-and-friends/master/output/tripet-loss-net-MNIST-testing-set-embedding.png
[tp_fmnist]: https://raw.githubusercontent.com/heytitle/siamese-net-and-friends/master/output/tripet-loss-net-FashionMNIST-testing-set-embedding.png


[siamese-paper]: https://www.semanticscholar.org/paper/Siamese-Neural-Networks-for-One-Shot-Image-Koch/e66955e4a24b611c54f9e7f6b178e7cbaddd0fbb
[contrastive-loss]: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
[tripet-loss]: https://arxiv.org/pdf/1503.03832.pdf

[img_placeholder]: https://via.placeholder.com/500x500
