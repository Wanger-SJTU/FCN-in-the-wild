# FCNs in the Wild Pixel-level Adversarial and Constraint-based Adaptation

**To be finished later**

Pytorch implemention of this [arxiv paper](https://arxiv.org/abs/1612.02649)

The FCN model used is papre [Multi-scale context aggregation by dilated convolutions](https://arxiv.org/abs/1511.07122)
**note** not finished



### dataset

- [GTA5 dataset](https://download.visinf.tu-darmstadt.de/data/from_games/)

### requirements 

- tqdm
- pytorch
- numpy
- scipy
- Pillow
- visdom

### training

### examples


## note 
In the [GTA5 dataset](https://download.visinf.tu-darmstadt.de/data/from_games/), the label file is png format which uses palette, so to train the model should record the palette infomation to recover the output  with color