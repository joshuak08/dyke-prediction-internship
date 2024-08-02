# Dyke parameter predictions using deep learning vision models

This repository contains all relevant code and scripts to train and test models relating to the prediction of dyke parameters given wrapped InSAR images. Vision models from the following papers were used.

[Liu, Z. et al. (2022) ‘Swin Transformer V2: Scaling Up Capacity and Resolution’, in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12009–12019](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.html)

[Liu, Z. et al. (2022) ‘A ConvNet for the 2020s’, in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11976–11986](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html)

[Touvron, H. et al. (2021) ‘Going Deeper With Image Transformers’, in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 32–42](https://openaccess.thecvf.com/content/ICCV2021/html/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.html)

[Touvron, H. et al. (2021) ‘Training data-efficient image transformers & distillation through attention’, in Meila, M. and Zhang, T. (eds) Proceedings of the 38th International Conference on Machine Learning. PMLR (Proceedings of Machine Learning Research), pp. 10347–10357](https://proceedings.mlr.press/v139/touvron21a)

## Installation

To install the required Python packages, please run the following command.

```bash
conda env create -f environment.yml
```

The default conda environment name is ``strike``. You can change it by either addding ``-n env_name`` to the command above or changing the name parameter in ``environment.yml``

## Dataset Setup/Preparation

Input all the training and testing synthetic images into the following directory structure.

```bash
dataset
├── D
│   ├── img1.jpeg
│   ├── img2.jpeg
│   └── ...
│   
└── DST_Coherence
    ├── img4.jpeg
    ├── img5.jpeg
    └── ...
```

Then, run the following command.

```bash
python split_dataset.py --dir "./dataset" --imageType "DST_Coherence"
```

This will produce a `file_split.json` that will split images into a training and testing partitions. The training partitions will be for both training and validation phase. Change ``--imageType`` to the relevant subdirectory as fit.

```bash
dataset
├── D
│   ├── img1.jpeg
│   ├── img2.jpeg
│   └── ...
│   
└── DST_Coherence
    ├── file_split.json
    ├── img4.jpeg
    ├── img5.jpeg
    └── ...
```

## Single Parameter

### Training

To train models, go to repository root directory and run:

```bash
python train.py 
--network "swin" # change this for different architectures
--imageSize 512 # reduce in power of 2 in case of insufficient cuda memory 
--learningParam "strike" # change to wanted learning parameter
--lossFunction "L1" # change to wanted loss function
--imageType "DST_Coherence" # change to relevant image type in dataset
--outputDir "/output"
--inputDir "/dataset"
--batchSize 8
--maxepoch 200
--lr 3e-5 # learning rate
--edgeDetection False # could experiment for strike predictions
```

### Testing

To test on synthetic InSAR images, go to repository root directory and run:

```bash
python testing.py 
--network "swin" # change this for different architectures
--imageSize 512 # reduce in power of 2 in case of insufficient cuda memory 
--learningParam "strike" # change to wanted learning parameter
--lossFunction "L1" # change to wanted loss function
--imageType "DST_Coherence" # change to relevant image type in dataset
--weights "/output"
--inputDir "/dataset" # change to directory containing synthetic dataset
```

This will produce a spreadsheet to view the differences for each saved model state and individual images.

To test on real InSAR images, go to repository root directory and run:

```bash
python real_testing.py
--network "swin" # change this for different architectures
--imageSize 512 # reduce in power of 2 in case of insufficient cuda memory 
--learningParam "strike" # change to wanted learning parameter
--lossFunction "L1" # change to wanted loss function
--imageType "DST_Coherence" # change to relevant image type in dataset
--weights "/output"
--inputDir "/real_dataset" # change to directory containing synthetic dataset
```

Note: the outputDir argument should remain the same for both synthetic and real image testing.

## Multi-parameter

### Training

Before training the models on multiple parameters, please ensure the loss function is adjusted as needed. The default loss function in this repository is specifically for strike and opening only, as shown below.

```python
class MultiLoss(nn.Module):
    def __init__(self, strike_weight, opening_weight):
        super(MultiLoss, self).__init__()
        self.strike_weight = strike_weight
        self.opening_weight = opening_weight


    def forward(self, predicted, target):
        strike_predicted = predicted[:, 0]
        opening_predicted = predicted[:, 1]

        strike_target = target['strike']
        opening_target = target['opening']

        strike_loss = L1AngularLoss()(strike_predicted, strike_target)
        opening_loss = OpeningLoss()(opening_predicted, opening_target)

        total_loss = self.strike_weight*strike_loss + self.opening_weight*opening_loss

        return torch.mean(total_loss)
```

To run the training on multiple models, run the foll0wing command:

```bash
python multi_param_training.py
--network "swin"
--imageSize 512
--imageType "DST_Coherence" # change to relevant image type in dataset
--outputDir "/output"
--inputDir "/dataset" # change to directory containing synthetic dataset
--strikeWeight 1.0 # change the weights for loss function as required
--openingWeight 1.0 
--learningParam strike opening
```

Note: the order of the ``--learningParam`` parameters matter, default is strike first then opening. If more parameters are to be added, please adjust the ``MultiLoss`` class loss function as required to support more parameters, ensuring the predicted tensors are the right column corresponding to the right label parameter.

### Testing

To test the multi-parameter model on the synthetic images, please run the following command.

```bash
python multi_param_testing.py 
--network "swin" # change this for different architectures
--imageSize 512
--learningParam "strike" "opening" 
--imageType "DST_Coherence" # change to relevant image type in dataset
--weights "/output"
--inputDir "/dataset" # change to directory containing synthetic dataset
```

To test the model with real images, run the following command:

```bash
python multi_param_real_testing.py
--network "swin" # change this for different architectures
--imageSize 512 # reduce in power of 2 in case of insufficient cuda memory 
--learningParam "strike" "opening" # change to wanted learning parameter
--imageType "DST_Coherence" # change to relevant image type in dataset
--weights "/output"
--inputDir "/real_dataset" # change to directory containing synthetic dataset
```

## BlueCrystal4 Usage

Please see [BlueCrystal4 Usage](BC4.md) for detailed instructions to run the training and testing using BlueCrystal4.
