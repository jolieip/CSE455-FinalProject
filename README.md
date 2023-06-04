# CSE455-FinalProject

# Deepfake Detection 

This repository contains a deepfake detection model built using the ResNet50 pretrained model. The motivation behind developing this model was the increasing prevalence of deepfake technology, which poses a significant threat to various aspects of society, including misinformation, privacy violations, and potential harm to individuals.

## Motivation

The decision to choose the ResNet50 pretrained model was based on extensive research and analysis. After reading the paper titled "Improved Deep Learning Model for Deepfake Detection" (https://arxiv.org/pdf/2210.00361.pdf), it became evident that ResNet50 has demonstrated exceptional performance in deepfake detection tasks. This pretrained model provides a solid foundation for our deepfake detection model and significantly reduces the burden of training a model from scratch.

## Challenges and Struggles

During the development process, several challenges were encountered. One significant struggle was the time-consuming process of loading data into Google Colab. Mounting the data each time the notebook was reloaded made the workflow infeasible. To overcome this challenge, we decided to build our model using Kaggle notebooks. We leveraged the "140k Real and Fake Faces" dataset from Kaggle (https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) to train and evaluate the model. This dataset provided a diverse and comprehensive set of images to improve the model's accuracy.

Initially, we attempted to train the model using a custom CNN architecture. However, this approach resulted in low training accuracy, reaching only 50%. Recognizing the need for a more powerful model, we turned to pretrained models. By using the ResNet50 pretrained model, we observed a significant improvement in training accuracy. However, the testing accuracy was relatively low, indicating the possibility of overfitting. Further optimization and fine-tuning of the model are required to address this issue.

## Techniques and Data Preprocessing

To preprocess the images and prepare them for the CNN, we utilized TensorFlow's ImageDataGenerator. This powerful tool automates various image preprocessing techniques, including rescaling, data augmentation, and normalization. By utilizing this approach, we were able to streamline the data preparation process and make the dataset ready for training the deepfake detection model. This significantly reduced the manual effort required for cleaning and preprocessing the data.

## Repository Structure

- `data/`: Placeholder directory for storing the dataset (not included in this repository).
- `notebooks/`: Contains the Jupyter notebook used for model development.
- `models/`: Saved model checkpoints and trained weights.
- `src/`: Source code files for data preprocessing, model architecture, and evaluation.

## Getting Started

To get started with this deepfake detection model, follow these steps:

1. Download the "140k Real and Fake Faces" dataset from Kaggle (link provided above).
2. Place the dataset files in the `data/` directory.
3. Open the Jupyter notebook `deepfake_detection.ipynb` located in the `notebooks/` directory.
4. Follow the instructions in the notebook to preprocess the data, train the model, and evaluate its performance.
5. Experiment with different hyperparameters, regularization techniques, or model architectures to enhance the model's accuracy and generalization.

## Future Work

This project serves as a starting point for deepfake detection. There are several avenues for future improvement, including:

- Investigating other pretrained models such as VGG and XCeption to compare their performance against ResNet50.
- Implementing regularization techniques, such as dropout or batch normalization, to reduce overfitting and improve testing accuracy.
- Exploring advanced deep learning architectures, such as attention mechanisms or adversarial training, to enhance the model's ability to detect sophisticated deepfakes.

