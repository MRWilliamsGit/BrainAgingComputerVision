<p align="center">
  <img src="https://github.com/MRWilliamsGit/BrainAgingComputerVision/blob/main/title.png" />
</p>

# Assessing Brain Age with Computer Vision

This project uses transfer learning to train a ResNet18 model to identify the age of a brain from its MRI scan. This kind of identification could potentially be used to flag various congnitive disorders such as Alzhiemers Disease if the "brain age" is identified as more advanced than the patient's chronological age.

## Data

We used 1267 congnitively normal files from the ADNI MRI dataset. These were black and white 3D images, stored in 3D numpy arrays that had already been cleaned and pre-processed. 

A sub-sample of ten has been included in the data directory of this repository.

## Modeling

The base model used was the pre-trained resnet18 model. Its weights were retained, but its first layer was adapted to handle our inputs and its last layer was trained on our data to produce regression outputs.

## Conclusions

## Demo

Please open main.ipynb in Google Colab to interact with the model.

