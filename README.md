# CSE455-FinalProject

**Author:** Jolie Ip

# Deepfake Detection 

## About

This repository contains a deepfake detection model built using the ResNet50 pretrained model. The motivation behind developing this model was the increasing prevalence of deepfake technology, which poses a significant threat to various aspects of society, including misinformation, privacy violations, and potential harm to individuals. Our aim was to build a model that specifically discerns between authentic and manipulated images of individuals.

## Motivation

The decision to choose the ResNet50 pretrained model was based on extensive research and analysis. After reading the paper titled "Improved Deep Learning Model for Deepfake Detection" (https://arxiv.org/pdf/2210.00361.pdf), it became evident that ResNet50 has demonstrated exceptional performance in deepfake detection tasks. This pretrained model provides a solid foundation for our deepfake detection model and significantly reduces the burden of training a model from scratch.

## Challenges and Struggles

During the development process, several challenges were encountered. One significant struggle was the time-consuming process of loading data into Google Colab. Mounting the data each time the notebook was reloaded made the workflow infeasible. To overcome this challenge, I decided to build the model using Kaggle notebooks. We leveraged the "140k Real and Fake Faces" dataset from Kaggle (https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) to train and evaluate the model. This dataset provided a diverse and comprehensive set of images to improve the model's accuracy.

Initially, I attempted to train the model using a custom CNN architecture. However, this approach resulted in low training accuracy, reaching only 50%. Recognizing the need for a more powerful model, I turned to pretrained models. By using the ResNet50 pretrained model, I observed a significant improvement in training accuracy. However, the testing accuracy was relatively low, indicating the possibility of overfitting. Further optimization and fine-tuning of the model are required to address this issue.

## Techniques and Data Preprocessing

To preprocess the images and prepare them for the CNN, I utilized TensorFlow's ImageDataGenerator. This powerful tool automates various image preprocessing techniques, including rescaling, data augmentation, and normalization. By utilizing this approach, I was able to streamline the data preparation process and make the dataset ready for training the deepfake detection model. This significantly reduced the manual effort required for cleaning and preprocessing the data.

## Code Attribution

In developing this deepfake detection model, I built upon various resources and code snippets. The code for loading and preprocessing the dataset using ImageDataGenerator was borrowed from the official TensorFlow documentation and adapted to suit the specific requirements of this project. Additionally, the implementation of the ResNet50 architecture was sourced from the Keras library, which provided a reliable and efficient implementation of the model.

I would like to acknowledge the authors of the following resources:

- TensorFlow documentation: https://www.tensorflow.org/
- Keras library: https://keras.io/

The remaining code, including model training, evaluation, and additional custom modifications, was developed by me for this project.

## Future Work

This project serves as a starting point for deepfake detection. There are several avenues for future improvement, including:

- Investigating other pretrained models such as VGG and XCeption to compare their performance against ResNet50.
- Implementing regularization techniques, such as dropout or batch normalization, to reduce overfitting and improve testing accuracy.
- Exploring advanced deep learning architectures, such as attention mechanisms or adversarial training, to enhance the model's ability to detect sophisticated deepfakes.

## Video Demo

https://github.com/jolieip/CSE455-FinalProject/assets/61493372/4aec9409-e634-4a07-925a-6a48705d2cd8 




