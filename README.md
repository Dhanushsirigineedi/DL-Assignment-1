# DL-Assignment-1

## Overview

This repository contains the implementation of a **feedforward neural network** for image classification on **MNIST** and **Fashion MNIST** datasets. The network is trained using different optimization algorithms and logs results using **Weights & Biases (wandb)**.

---

## **Code Structure**

### **1. Import Libraries**

- Essential libraries like `numpy`, `matplotlib`, `seaborn`, `wandb`, `argparse`, `sklearn.metrics`, and `train_test_split` are imported.
- These libraries support **data preprocessing, visualization, training, evaluation, and logging**.

### **2. Data Loading and Preprocessing**

- The dataset (either **MNIST** or **Fashion MNIST**) is loaded using `keras.datasets`.
- The dataset is split into **training (90%)** and **validation (10%)** sets using `train_test_split`.
- Training images are **flattened** (converted from a 2D matrix to a 1D vector) and **normalized** by dividing pixel values by 255.
- The function `plot_sample_images()` is available for **visualizing sample images** from the dataset.

### **3. Neural Network Functionalities**

The code implements a **feedforward neural network** with the following functionalities:

#### **Parameter Initialization (**``**)**

- Initializes weights and biases for all layers.
- Supports **random initialization** and **Xavier initialization**.

#### **Forward Propagation (**``**)**

- Computes activations for each layer using **ReLU, Sigmoid, Tanh, or Identity** functions.
- Uses **Softmax activation** for the output layer.

#### **Backpropagation (**``**)**

- Computes **gradients of the loss function** (cross-entropy or MSE) for weights and biases.
- Updates weights and biases based on the chosen **optimizer**.

---

### **4. Optimization Algorithms**

Several optimization algorithms are implemented to update the network parameters:

- **Stochastic Gradient Descent (SGD)**: Standard SGD updates after each batch.
- **Momentum-based Gradient Descent**: Adds momentum to accelerate convergence.
- **Nesterov Accelerated Gradient Descent (NAG)**: Looks ahead to improve updates.
- **RMSprop**: Uses an **adaptive learning rate** to stabilize training.
- **Adam & Nadam**: Combines momentum and adaptive learning rates for better convergence.

Each optimizer iterates over **epochs**, applies **mini-batch updates**, and logs metrics using `wandb.log()`.

---

### **5. Training the Network (**``**)**

- Calls the **appropriate optimizer** based on user input (`sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`).
- Trains the network, updating **weights and biases** through **backpropagation**.
- Evaluates **training and validation accuracy** after each epoch.
- Logs accuracy and loss metrics to **Weights & Biases (wandb)**.

---

## **How to Run the Code**

### **1. Run **``

- Open a terminal and run:

```bash
python dhanush_train.py [arguments]
```

- Replace `[arguments]` with desired values (see available arguments below).

### **2. Command-line Arguments**

```
-wp  or --wandb_project   : WandB project name.
-we  or --wandb_entity    : WandB entity name.
-d   or --dataset         : Dataset choice (mnist/fashion_mnist).
-e   or --epochs          : Number of training epochs.
-b   or --batch_size      : Batch size.
-l   or --loss            : Loss function (mean_squared_error/cross_entropy).
-o   or --optimizer       : Optimizer (sgd, momentum, nag, rmsprop, adam, nadam).
-lr  or --learning_rate   : Learning rate.
-m   or --momentum        : Momentum for certain optimizers.
-beta or --beta           : Beta parameter for RMSprop.
-beta1 or --beta1         : Beta1 parameter for Adam/Nadam.
-beta2 or --beta2         : Beta2 parameter for Adam/Nadam.
-eps  or --epsilon        : Small constant for numerical stability.
-w_d  or --weight_decay   : Weight decay for regularization.
-w_i  or --weight_init    : Weight initialization method (random/Xavier).
-nhl  or --num_layers     : Number of hidden layers.
-sz   or --hidden_size    : Number of neurons per hidden layer.
-a    or --activation     : Activation function (identity/sigmoid/tanh/relu).
```

### **3. Example Usage**

```bash
python dhanush_train.py -wp myproject -we myname -d fashion_mnist -e 10 -b 32 -l cross_entropy -o adam -lr 0.0001 -m 0.9 -beta 0.5 -beta1 0.9 -beta2 0.999 -eps 0.000001 -w_d 0 -w_i xavier -nhl 3 -sz 128 -a relu
```

This command trains a neural network on **Fashion MNIST** for **10 epochs** using the **Adam optimizer** and logs the results to **Weights & Biases**.

---

## **Logging with Weights & Biases**

- Training and validation metrics are **automatically logged** to **WandB**.
- You can visualize training progress on the **WandB dashboard**.

---

## **Conclusion**

This project provides a **flexible and customizable neural network training framework**, supporting multiple optimization methods and logging features for better experiment tracking. 🚀

