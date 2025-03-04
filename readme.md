# GenMind Blog Code Repository

This repository contains the source code for a couple of [my blog posts](https://genmind.ch) about data visualization in machine learning:

- **[Why Visualizing Data is So Important](https://genmind.ch/why-visualizing-data-is-so-important/)**  
  This article dives into the importance of data visualization in understanding model performance and insights. You’ll find the corresponding source code here, which reinforces the concepts discussed in the blog.

- **[Data Visualization with Weights & Biases: A Powerful Tool for Tracking and Organizing ML Experiments](https://genmind.ch/data-visualization-with-weights-biases-a-powerful-tool-for-tracking-and-organizing-ml-experiments/)**  
  In this post, I explain how to leverage Weights & Biases for tracking machine learning experiments. The code in this repository, tensorboard_mnist.py, shows you how powerful data visualization is.

- **[Adding TensorBoard to Your Keras Workflow](https://genmind.ch/adding-tensorboard-to-your-keras-workflow/)**  
  In this post, I explain how add TensorBoard to a Keras model. File in this repo train_mnist_tensorboard.py.




## Getting Started

1. **Clone the repository:**
```bash
   git clone https://github.com/gsantopaolo/dataviz.git
   cd dataviz
```

2. **Using Weights and Biases :**
To use the weights and biases sample you would need to be registered top W&B and ou need your API Key. Follow the instructions 
[here](https://docs.wandb.ai/support/find_api_key/)
for where to find it. Once you have it add to your .env file. 
In the repo you will find a .env.example to help you.
