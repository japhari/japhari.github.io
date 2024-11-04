---
title: "Image Classification with Convolutional Neural Networks on the CIFAR-10 Dataset"
excerpt: "This code demonstrates building a Convolutional Neural Network (CNN) using various types of layers to classify images in the CIFAR-10 dataset. Each type of layer serves a specific purpose in enhancing the model's ability to capture and process complex features effectively, enabling it to classify CIFAR-10 images with improved performance"
collection: teaching
# colab_url: "https://colab.research.google.com/drive/1-T78BMZQg3w9m4iSG2zF154xGAZfcuah"
# github_url: "https://github.com/japhari/natural-language-processing/blob/main/hugging_face_transformer_for_nlp_task.ipynb"
thumbnail: "/images/topics/convolution.png"
type: "Convolutional Neural Network"
permalink: /teaching/conv-layers-cnn
venue: "PTIT , Department of Computer Science"
date: 2024-10-24
location: "Hanoi, Vietnam"
categories:
  - teaching
tags:
  - cnn

---
<img src="/images/topics/convolution.png" alt="CNN" width="640" height="360">


## Convolutional Neural Network Layers


Convolutional neural networks are distinguished from other neural networks by their superior performance with image, speech, or audio signal inputs. In Convolutional Neural Networks (CNNs), the main layers are designed to capture spatial features of images efficiently. Here’s an overview of the key layers in a ConvNet and their roles ;

### 1. **Convolutional Layer (CONV)**
The convolutional layer is the core building block of a CNN, and it is where the majority of computation occurs. It requires a few components, which are input data, a filter, and a feature map.

- **Purpose** Detect local features in the image, such as edges, textures, or more complex patterns.
- **How it works** 
  - The convolutional layer applies **filters** (or kernels), which are small matrices (e.g., 3x3 or 5x5) that slide across the input image (or feature map).
  - Each filter produces a **feature map** by computing a dot product between the filter values and the input values at each spatial location.
  - The output of this layer retains the spatial structure, meaning the height and width of the feature maps depend on the filter size and stride (step size of the filter movement).
- **Example** In a 32x32 RGB image, using 12 filters of size 3x3 results in an output of 12 feature maps, each of shape 32x32. This layer captures patterns within the input region.

### 2. **ReLU Layer (Rectified Linear Unit)**

- **Purpose** Introduce **non-linearity** into the network, allowing it to learn more complex functions and better capture relationships in data.
- **How it works** 
  - The ReLU function is applied element-wise to each activation in the feature map, changing negative values to zero (using `f(x) = max(0, x)`).
  - This layer does not change the size of the input—it only modifies the values.
- **Example** If a convolutional layer output has negative values due to the dot product operation, ReLU converts them to zero, while positive values remain unchanged, helping to reduce vanishing gradients and speed up training.

### 3. **Pooling Layer (POOL)**

- **Purpose** **Downsample** the spatial dimensions (height and width) of feature maps, reducing the number of parameters and computational cost, as well as providing translation invariance (the network becomes less sensitive to small shifts in the image).
- **How it works** 
  - The pooling layer applies a small window (e.g., 2x2 or 3x3) over each feature map, taking either the **maximum** value (Max Pooling) or the **average** value (Average Pooling) within that window.
  - Pooling is typically applied with a stride equal to the window size (e.g., a 2x2 window with a stride of 2), effectively halving the size of the feature maps.
- **Example** Applying a 2x2 Max Pooling on a 32x32 feature map reduces it to a 16x16 feature map, retaining only the most prominent features within each 2x2 window.

### 4. **Fully Connected Layer (FC)**

- **Purpose** Perform the final classification by connecting all features learned by previous layers to each class in the dataset.
- **How it works** 
  - The fully connected layer (often the last layer) **flattens** the 3D output of the previous layer into a 1D vector and then **connects** it to each output neuron, one for each class (e.g., 10 neurons for 10 CIFAR-10 classes).
  - Each neuron in the FC layer is connected to all activations in the previous layer, allowing it to synthesize all learned features and output a score for each class.
- **Example** If the final pooled feature map has a shape of [8, 8, 24], flattening it results in a 1D vector of size 1536. The FC layer then connects this vector to 10 output neurons for CIFAR-10, outputting 10 class scores.

### 5. **Softmax Layer (for Classification Tasks)**

- **Purpose** Convert the raw scores from the last fully connected layer into probabilities for each class.
- **How it works**
  - The **Softmax** function normalizes the output scores so that they sum to 1, making them interpretable as probabilities.
 
- **Example** If the FC layer outputs raw scores for 10 classes, the Softmax layer converts these scores into probabilities, making it easier to identify the most likely class.

