# Creative Adversarial Networks
![collage](assets/collage.png)

*128x128 pixel Samples from CAN train on WikiART.*

A WIP implementation of [CAN: Creative Adversarial Networks, Generating "Art" 
by Learning About Styles and Deviating from Style Norms](https://arxiv.org/abs/1706.07068). 
Repo bases DCGAN implementation on [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) 
with modifications to reduce checkerboard artifacts according to [this 
distill article](https://distill.pub/2016/deconv-checkerboard/)

The paper authors basically modified the GAN objective to encourage the network to deviate away from art norms.

## Getting the Dataset
We used this compiled [wikiart](https://www.wikiart.org/) dataset 
[available here](https://github.com/cs-chan/ICIP2016-PC/tree/f5d6f6b58a6d8a4bd05aaaedd9688d08c02df8f2/WikiArt%20Dataset). 
Using the dataset 
is subject to wikiart's [terms of use](https://www.wikiart.org/en/terms-of-use)

Extract the dataset, then set the path in `train.sh`

## Training a DCGAN model
Edit the parameters of train.sh then
```
bash train.sh
```
## Citation
If you use this implementation in your own work please cite the following
```
@misc{2017cans,
  author = {Phillip Kravtsov and Phillip Kuznetsov},
  title = {Creative Adversarial Networks},
  year = {2017},
  howpublished = {\url{https://github.com/mlberkeley/Creative-Adversarial-Networks}},
  note = {commit xxxxxxx}
}
```
## Authors 
[Phillip Kravtsov](https://github.com/phillip-kravtsov)

[Phillip Kuznetsov](https://github.com/philkuz)





