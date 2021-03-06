<p align="center">
  <img src="https://github.com/MRWilliamsGit/BrainAgingComputerVision/blob/89fe88432401e0f8bccd7f958580734e5440c353/images/title.png" 
       width= 443
       height= 175 />
</p>

# Assessing Brain Age with Computer Vision

This project uses transfer learning to train a ResNet18 model to identify the age of a brain from its MRI scan. This kind of identification could potentially be used to flag various congnitive disorders such as Alzhiemers Disease if the "brain age" is identified as more advanced than the patient's chronological age.

## Data

We used 1267 congnitively normal files from the ADNI MRI dataset. These were black and white 3D images, stored in 3D numpy arrays that had already been cleaned and pre-processed. 

A sub-sample of ten has been included in the sample_data directory of this repository.

## Modeling

The base model used was the pre-trained resnet18 model. Its weights were retained, but its first layer was adapted to handle our inputs and its last layer was trained on our data to produce regression outputs.

The results were promising: our model reached a MAE of about 5 years.

## Demo

Please open main.ipynb in Google Colab to interact with the model.

