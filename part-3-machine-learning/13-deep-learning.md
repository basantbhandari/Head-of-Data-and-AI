# Chapter 13. Deep Learning

_Part III — Machine Learning_

Backpropagation, activations, normalization, residual networks, sequence models, generative models and the practicalities of training deep networks.

This chapter contains **84 questions**.

## Contents

- **13.1** [What are neural networks?](#131-what-are-neural-networks)
- **13.2** [Explain the Feedforward Neural Network.](#132-explain-the-feedforward-neural-network)
- **13.3** [What are forward propagation and backward propagation?](#133-what-are-forward-propagation-and-backward-propagation)
- **13.4** [What is backpropagation?](#134-what-is-backpropagation)
- **13.5** [Can you name and explain a few hyperparameters used for training a neural network?](#135-can-you-name-and-explain-a-few-hyperparameters-used-for-training-a-neural-network)
- **13.6** [What is the advantage of deep learning over traditional machine learning?](#136-what-is-the-advantage-of-deep-learning-over-traditional-machine-learning)
- **13.7** [What are activation functions, and why are they used?](#137-what-are-activation-functions-and-why-are-they-used)
- **13.8** [Explain Sigmoid, Tanh, ReLU, LeakyReLU, and Softmax activation functions with their pros and cons.](#138-explain-sigmoid-tanh-relu-leakyrelu-and-softmax-activation-functions-with-their-pros-and-cons)
- **13.9** [Why are Sigmoid and Tanh not preferred in the hidden layers of a neural network?](#139-why-are-sigmoid-and-tanh-not-preferred-in-the-hidden-layers-of-a-neural-network)
- **13.10** [What is dropout, and why is it effective?](#1310-what-is-dropout-and-why-is-it-effective)
- **13.11** [What is the effect of dropout on training and inference speed?](#1311-what-is-the-effect-of-dropout-on-training-and-inference-speed)
- **13.12** [What is L1/L2 regularization, and how does it affect a neural network?](#1312-what-is-l1l2-regularization-and-how-does-it-affect-a-neural-network)
- **13.13** [What is batch normalization, and why is it used?](#1313-what-is-batch-normalization-and-why-is-it-used)
- **13.14** [What are the hyperparameters for batch normalization that can be optimized?](#1314-what-are-the-hyperparameters-for-batch-normalization-that-can-be-optimized)
- **13.15** [What is parameter sharing in deep learning?](#1315-what-is-parameter-sharing-in-deep-learning)
- **13.16** [What is representation learning, and why is it useful?](#1316-what-is-representation-learning-and-why-is-it-useful)
- **13.17** [What is a generative model, and how does it differ from a discriminative model?](#1317-what-is-a-generative-model-and-how-does-it-differ-from-a-discriminative-model)
- **13.18** [Can you explain how a generative model works?](#1318-can-you-explain-how-a-generative-model-works)
- **13.19** [Explain Encoder-Decoder Architecture.](#1319-explain-encoder-decoder-architecture)
- **13.20** [What is the difference between encoder-only, decoder-only, and encoder-decoder Transformer architectures?](#1320-what-is-the-difference-between-encoder-only-decoder-only-and-encoder-decoder-transformer-architectures)
- **13.21** [What is Latent Space?](#1321-what-is-latent-space)
- **13.22** [What are autoencoders? Explain their layers and practical uses.](#1322-what-are-autoencoders-explain-their-layers-and-practical-uses)
- **13.23** [What is a Variational Autoencoder (VAE), and how is it different from a traditional autoencoder?](#1323-what-is-a-variational-autoencoder-vae-and-how-is-it-different-from-a-traditional-autoencoder)
- **13.24** [How does VAE impose a probabilistic structure on the latent space, and why is that important?](#1324-how-does-vae-impose-a-probabilistic-structure-on-the-latent-space-and-why-is-that-important)
- **13.25** [What is the architecture of a Generative Adversarial Network (GAN)?](#1325-what-is-the-architecture-of-a-generative-adversarial-network-gan)
- **13.26** [What are the roles of the generator and discriminator in a GAN?](#1326-what-are-the-roles-of-the-generator-and-discriminator-in-a-gan)
- **13.27** [What is mode collapse in GANs, and how can it be mitigated?](#1327-what-is-mode-collapse-in-gans-and-how-can-it-be-mitigated)
- **13.28** [How are GANs used in image synthesis or image-to-image translation tasks?](#1328-how-are-gans-used-in-image-synthesis-or-image-to-image-translation-tasks)
- **13.29** [Explain Convolutional Neural Networks (CNNs).](#1329-explain-convolutional-neural-networks-cnns)
- **13.30** [Explain filters in CNN.](#1330-explain-filters-in-cnn)
- **13.31** [Explain the stride in CNN.](#1331-explain-the-stride-in-cnn)
- **13.32** [Explain padding in CNN.](#1332-explain-padding-in-cnn)
- **13.33** [Explain pooling in CNN.](#1333-explain-pooling-in-cnn)
- **13.34** [Explain fully connected layers in CNN.](#1334-explain-fully-connected-layers-in-cnn)
- **13.35** [What is a Recurrent Neural Network (RNN)?](#1335-what-is-a-recurrent-neural-network-rnn)
- **13.36** [What are the limitations of RNNs, and how are they solved?](#1336-what-are-the-limitations-of-rnns-and-how-are-they-solved)
- **13.37** [What are LSTM and GRU? How do they solve long-term dependency issues?](#1337-what-are-lstm-and-gru-how-do-they-solve-long-term-dependency-issues)
- **13.38** [What are the main gates in LSTM and their roles?](#1338-what-are-the-main-gates-in-lstm-and-their-roles)
- **13.39** [How to identify exploding gradient issues in your model?](#1339-how-to-identify-exploding-gradient-issues-in-your-model)
- **13.40** [What is a Transformer architecture, and what makes it different from CNNs and RNNs?](#1340-what-is-a-transformer-architecture-and-what-makes-it-different-from-cnns-and-rnns)
- **13.41** [What is the basic difference between LSTM and Transformers?](#1341-what-is-the-basic-difference-between-lstm-and-transformers)
- **13.42** [What are diffusion models?](#1342-what-are-diffusion-models)
- **13.43** [Why does Diffusion work better than autoregression?](#1343-why-does-diffusion-work-better-than-autoregression)
- **13.44** [Explain transfer learning and when to use it.](#1344-explain-transfer-learning-and-when-to-use-it)
- **13.45** [What are Multimodal AI models, and how do they process different types of data?](#1345-what-are-multimodal-ai-models-and-how-do-they-process-different-types-of-data)
- **13.46** [How do World Models work?](#1346-how-do-world-models-work)
- **13.47** [How do Diffusion Language Models (DLMs) work?](#1347-how-do-diffusion-language-models-dlms-work)
- **13.48** [Explain deep reinforcement learning from human preferences.](#1348-explain-deep-reinforcement-learning-from-human-preferences)
- **13.49** [Normalization in machine learning?](#1349-normalization-in-machine-learning)
- **13.50** [What is the difference between normalization and standardization?](#1350-what-is-the-difference-between-normalization-and-standardization)
- **13.51** [What is a neural network?](#1351-what-is-a-neural-network)
- **13.52** [What do you understand about the exploding gradient technique problem in the back propagation technique?](#1352-what-do-you-understand-about-the-exploding-gradient-technique-problem-in-the-back-propagation-technique)
- **13.53** [What differences exist between Softmax and Sigmoid functions?](#1353-what-differences-exist-between-softmax-and-sigmoid-functions)
- **13.54** [How do you decide when to go for deep learning for a project?](#1354-how-do-you-decide-when-to-go-for-deep-learning-for-a-project)
- **13.55** [Why do binary\_crossentropy and categorical\_crossentropy give different performances for the same problem?](#1355-why-do-binary_crossentropy-and-categorical_crossentropy-give-different-performances-for-the-same-problem)
- **13.56** [Compare RNN, LSTM, GRU and Transformers.](#1356-compare-rnn-lstm-gru-and-transformers)
- **13.57** [Compare world models and JEPA.](#1357-compare-world-models-and-jepa)
- **13.58** [What is circuit/graph(operation like addition, multiplication, activation function are represented by nodes of the graph are called gates) search( process of systematically exploring nodes and edges of graph to determine the valid path from starting point to end point ) ? How does backpropagation solve circuit search?](#1358-what-is-circuitgraphoperation-like-addition-multiplication-activation-function-are-represented-by-nodes-of-the-graph-are-called-gates-search-process-of-systematically-exploring-nodes-and-edges-of-graph-to-determine-the-valid-path-from-starting-point-to-end-point---how-does-backpropagation-solve-circuit-search)
- **13.59** [What is a function approximator?](#1359-what-is-a-function-approximator)
- **13.60** [Compare Hopfield networks and Boltzmann machines.](#1360-compare-hopfield-networks-and-boltzmann-machines)
- **13.61** [Compare content-addressable memory and parametric memory.](#1361-compare-content-addressable-memory-and-parametric-memory)
- **13.62** [Why do Transformers use layer normalization instead of batch normalization, and what is Pre-LN vs Post-LN?](#1362-why-do-transformers-use-layer-normalization-instead-of-batch-normalization-and-what-is-pre-ln-vs-post-ln)
- **13.63** [How would you implement a regex matcher supporting `.` and `*`, and how do you debug a PyTorch training loss that is NaN or not decreasing?](#1363-how-would-you-implement-a-regex-matcher-supporting--and--and-how-do-you-debug-a-pytorch-training-loss-that-is-nan-or-not-decreasing)
- **13.64** [Explain backpropagation from first principles.](#1364-explain-backpropagation-from-first-principles)
- **13.65** [Why do neural networks need nonlinear activation functions?](#1365-why-do-neural-networks-need-nonlinear-activation-functions)
- **13.66** [Compare ReLU, Leaky ReLU, GELU, SiLU, and sigmoid.](#1366-compare-relu-leaky-relu-gelu-silu-and-sigmoid)
- **13.67** [Why can sigmoid cause vanishing gradients?](#1367-why-can-sigmoid-cause-vanishing-gradients)
- **13.68** [What is Batch Normalization, and why does it help?](#1368-what-is-batch-normalization-and-why-does-it-help)
- **13.69** [Compare BatchNorm, LayerNorm, GroupNorm, and RMSNorm.](#1369-compare-batchnorm-layernorm-groupnorm-and-rmsnorm)
- **13.70** [Why is LayerNorm preferred over BatchNorm in Transformers?](#1370-why-is-layernorm-preferred-over-batchnorm-in-transformers)
- **13.71** [What is residual learning, and why are skip connections so effective?](#1371-what-is-residual-learning-and-why-are-skip-connections-so-effective)
- **13.72** [Why do deeper networks sometimes perform worse even when optimization should theoretically be easier?](#1372-why-do-deeper-networks-sometimes-perform-worse-even-when-optimization-should-theoretically-be-easier)
- **13.73** [Explain dropout mathematically.](#1373-explain-dropout-mathematically)
- **13.74** [Is dropout still useful in modern large Transformers?](#1374-is-dropout-still-useful-in-modern-large-transformers)
- **13.75** [What is knowledge distillation?](#1375-what-is-knowledge-distillation)
- **13.76** [Explain teacher-student training.](#1376-explain-teacher-student-training)
- **13.77** [What is early stopping actually doing from an optimization perspective?](#1377-what-is-early-stopping-actually-doing-from-an-optimization-perspective)
- **13.78** [How would you debug a neural network whose loss suddenly becomes NaN?](#1378-how-would-you-debug-a-neural-network-whose-loss-suddenly-becomes-nan)
- **13.79** [What are Graph Neural Networks (GNNs)?](#1379-what-are-graph-neural-networks-gnns)
- **13.80** [What is geometric deep learning, and how do symmetry, invariance and equivariance unify CNNs, GNNs and Transformers?](#1380-what-is-geometric-deep-learning-and-how-do-symmetry-invariance-and-equivariance-unify-cnns-gnns-and-transformers)
- **13.81** [What are equivariant neural networks, and where are they used in practice?](#1381-what-are-equivariant-neural-networks-and-where-are-they-used-in-practice)
- **13.82** [How do Graph Attention Networks work, and how do they differ from GCNs?](#1382-how-do-graph-attention-networks-work-and-how-do-they-differ-from-gcns)
- **13.83** [What are modern Hopfield networks, and how are they related to Transformer attention?](#1383-what-are-modern-hopfield-networks-and-how-are-they-related-to-transformer-attention)
- **13.84** [How is a Restricted Boltzmann Machine trained, and what is contrastive divergence?](#1384-how-is-a-restricted-boltzmann-machine-trained-and-what-is-contrastive-divergence)

---

### 13.1 What are neural networks?

**Neural Networks**

- Computational models inspired by biological neurons, composed of interconnected layers of nodes ("neurons")
- **Structure**: Input layer → one or more hidden layers → output layer
- Each neuron: weighted sum of inputs + bias, passed through activation function

**How they work**

- **Forward propagation**: data flows through layers, each layer transforms input using weights, biases, activation functions
- **Loss calculation**: compare predicted output to actual (using loss function)
- **Backpropagation**: computes gradients of loss w.r.t. weights via chain rule
- **Weight update**: gradient descent (or variants like Adam) adjusts weights to minimize loss

**Key components**

- **Weights & biases**: learnable parameters
- **Activation functions**: introduce non-linearity (ReLU, sigmoid, tanh, softmax)
- **Layers**: Dense/fully-connected, Convolutional (CNN), Recurrent (RNN), Transformer

**Types**

- **Feedforward (MLP)**: basic architecture
- **CNN**: image/spatial data
- **RNN/LSTM/GRU**: sequential/time-series data
- **Transformers**: NLP, attention-based, current state-of-the-art
- Capable of learning complex non-linear patterns, foundation of deep learning

### 13.2 Explain the Feedforward Neural Network.

**Feedforward Neural Network (FNN)**

- Simplest type of artificial neural network — data flows in one direction: input → hidden layers → output (no loops/cycles)
- Also called Multi-Layer Perceptron (MLP) when it has hidden layers

**Structure**

- **Input layer**: receives raw features
- **Hidden layer(s)**: apply weighted transformations + activation functions
- **Output layer**: produces final prediction (class probabilities or continuous value)

**How it works**

1. Each neuron computes: $z = w \cdot x + b$, then applies activation function $f(z)$
2. Output of one layer becomes input to next layer
3. Process repeats until output layer produces final prediction
4. **Training**: backpropagation + gradient descent to update weights, minimizing loss

**Key characteristics**

- No memory of previous inputs (unlike RNNs) — each input processed independently
- Fully connected layers (each neuron connects to all neurons in next layer) — Dense/Linear layers
- Uses non-linear activations (ReLU, sigmoid, tanh) to model complex patterns

**Use cases**

- Tabular data classification/regression
- Basic pattern recognition
- Building block for more complex architectures (CNNs, Transformers often include feedforward layers)

### 13.3 What are forward propagation and backward propagation?

**Forward Propagation & Backpropagation**

**Forward Propagation**

- Process of passing input data through the network layer by layer to compute output/prediction
- Each layer: $z = w \cdot x + b$ → activation $f(z)$ → passed to next layer
- Ends with output layer producing prediction
- Loss function then compares prediction to actual value

**Backpropagation**

- Process of computing gradients of loss function w.r.t. each weight, by propagating error backward from output to input layer
- Uses **chain rule** of calculus to compute how much each weight contributed to the error
- Gradients then used to update weights via gradient descent: $\theta = \theta - \alpha \nabla J(\theta)$

**Together**

1. Forward pass: compute prediction and loss
2. Backward pass: compute gradients (backpropagation)
3. Update weights (optimizer step)
4. Repeat for multiple epochs until convergence

### 13.4 What is backpropagation?

**Backpropagation**

- Algorithm to compute gradients of the loss function with respect to each weight in a neural network
- Propagates error backward from output layer to input layer using the **chain rule** of calculus
- Gradients indicate how much each weight contributed to the prediction error
- These gradients are then used by an optimizer (e.g., gradient descent, Adam) to update weights and minimize loss
- Core mechanism enabling neural networks to learn from data

### 13.5 Can you name and explain a few hyperparameters used for training a neural network?

**Neural Network Hyperparameters**

- **Learning rate**: step size for weight updates during gradient descent — too high diverges, too low is slow
- **Batch size**: number of samples processed before weight update — affects training speed/stability (small = noisy, large = stable but memory-heavy)
- **Number of epochs**: how many full passes through training data — too few underfits, too many overfits
- **Number of hidden layers/neurons**: controls model capacity/depth — more = higher capacity but risk of overfitting
- **Activation function**: ReLU, sigmoid, tanh — introduces non-linearity, affects gradient flow
- **Optimizer**: SGD, Adam, RMSprop — determines how weights are updated
- **Dropout rate**: fraction of neurons randomly dropped during training — prevents overfitting
- **Regularization (L1/L2)**: penalty added to loss to prevent overfitting
- **Weight initialization**: method for initializing weights (Xavier, He) — affects convergence
- **Momentum**: accelerates gradient descent, dampens oscillations

### 13.6 What is the advantage of deep learning over traditional machine learning?

**Deep Learning vs Traditional ML**

- **Automatic feature extraction**: deep learning learns features directly from raw data (pixels, text); traditional ML requires manual feature engineering
- **Handles unstructured data better**: images, audio, text, video — deep learning excels; traditional ML struggles without heavy preprocessing
- **Scales with data**: performance improves significantly with more data; traditional ML often plateaus
- **Captures complex, non-linear patterns**: deep architectures model highly complex relationships better than shallow models
- **Hierarchical representation learning**: lower layers learn simple features (edges), deeper layers learn complex/abstract features (objects, concepts)
- **End-to-end learning**: single model can go from raw input to final output without separate pipeline stages

**Trade-offs (traditional ML still wins here)**

- Requires more data and compute
- Less interpretable ("black box")
- Longer training time
- Traditional ML (e.g., XGBoost) often outperforms on small/tabular datasets

### 13.7 What are activation functions, and why are they used?

**Activation Functions**

- Mathematical functions applied to neuron outputs to introduce **non-linearity** into neural networks
- Without them, network would just be a linear transformation (stacking linear layers = still linear), unable to learn complex patterns

**Why used**

- Enable network to learn non-linear, complex relationships in data
- Determine whether a neuron should "fire" (activate) based on its input
- Affect gradient flow during backpropagation

**Common activation functions**

- **Sigmoid**: $\frac{1}{1+e^{-x}}$, output 0-1, used in binary classification output — suffers vanishing gradient
- **Tanh**: output -1 to 1, zero-centered, still has vanishing gradient issue
- **ReLU**: $\max(0, x)$, most popular — fast, avoids vanishing gradient for positive values, but "dying ReLU" problem
- **Leaky ReLU**: allows small negative slope, fixes dying ReLU
- **Softmax**: converts logits to probability distribution, used in multi-class output layer
- **GELU/Swish**: smoother variants used in modern architectures (Transformers)

### 13.8 Explain Sigmoid, Tanh, ReLU, LeakyReLU, and Softmax activation functions with their pros and cons.

**Activation Functions: Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax**

**Sigmoid**

- Formula: $\frac{1}{1+e^{-x}}$, output range: 0 to 1
- Pros: Smooth, good for binary classification output (probability)
- Cons: Vanishing gradient (saturates at extremes), not zero-centered, computationally expensive (exp)

**Tanh**

- Formula: $\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$, output range: −1 to 1
- Pros: Zero-centered (better than sigmoid for hidden layers)
- Cons: Still suffers vanishing gradient at extremes

**ReLU (Rectified Linear Unit)**

- Formula: $\max(0, x)$, output range: 0 to $\infty$
- Pros: Computationally efficient, avoids vanishing gradient for $x > 0$, sparse activation (efficiency)
- Cons: **Dying ReLU** problem — neurons stuck outputting 0 if input always negative (gradient = 0)

**Leaky ReLU**

- Formula: $x$ if $x > 0$, else $\alpha x$ (small $\alpha$ like 0.01)
- Pros: Fixes dying ReLU — allows small gradient for negative inputs
- Cons: Results can be inconsistent; $\alpha$ is a hyperparameter to tune

**Softmax**

- Formula: $\frac{e^{z_i}}{\sum_j e^{z_j}}$, output: probability distribution summing to 1
- Pros: Ideal for multi-class classification output layer, interpretable as probabilities
- Cons: Sensitive to outliers/large logits (can saturate), not used in hidden layers

### 13.9 Why are Sigmoid and Tanh not preferred in the hidden layers of a neural network?

**Why Sigmoid and Tanh Aren't Preferred in Hidden Layers**

- **Vanishing gradient problem**: both saturate at extremes (very high/low inputs) where derivative ≈ 0
  - Sigmoid derivative max = 0.25; Tanh derivative max = 1, but still saturates
  - In deep networks, gradients multiply across layers (chain rule) → shrink exponentially toward zero
  - Early layers get near-zero updates → learn very slowly or stop learning
- **Computationally expensive**: involve exponential calculations ($e^x$), slower than ReLU's simple $\max(0, x)$
- **Sigmoid not zero-centered**: outputs always positive (0 to 1) → gradients during backprop tend to all be same sign → inefficient zig-zag weight updates
- **Tanh is zero-centered** (−1 to 1) — better than sigmoid, but still suffers vanishing gradient at saturation

**Why ReLU preferred instead**

- No saturation for positive inputs → gradient stays constant (1) → no vanishing gradient issue in that region
- Computationally cheap (simple thresholding)
- Sparse activation (some neurons output exactly 0) → more efficient computation

**Where Sigmoid/Tanh still used**

- Sigmoid: output layer for binary classification, gates in LSTM/GRU
- Tanh: hidden states in RNN/LSTM (bounded output helps stability)

### 13.10 What is dropout, and why is it effective?

**Dropout**

- Regularization technique that randomly deactivates (sets to zero) a fraction of neurons during each training iteration
- Dropout rate ($p$) typically 0.2–0.5 — probability a neuron is dropped

**Why effective**

- **Prevents co-adaptation**: neurons can't rely too heavily on specific other neurons, forces more robust, independent feature learning
- **Acts like ensemble learning**: each training pass uses a different "thinned" sub-network — effectively trains many different network architectures that share weights, final model approximates averaging their predictions
- **Reduces overfitting**: network can't memorize training data through complex co-dependencies between neurons
- **Improves generalization**: forces redundant, distributed representations rather than relying on a few neurons

**Implementation notes**

- Only applied during training; all neurons active during inference (weights scaled by $(1-p)$ to compensate, or "inverted dropout" scales during training instead)
- Typically applied to fully connected layers, less common in convolutional layers

### 13.11 What is the effect of dropout on training and inference speed?

**Effect of Dropout on Training and Inference Speed**

**Training**

- Slightly increases computation per iteration (extra masking operation)
- Requires **more epochs to converge** — since network is effectively training many different sub-networks, learning signal per neuron is noisier/sparser
- Overall training time increases (more epochs × similar per-epoch cost)

**Inference**

- Dropout is **disabled** at inference time — all neurons active
- No dropout-related computation overhead at inference (masking removed)
- Weights scaled appropriately during training (inverted dropout) so no scaling needed at inference either
- **No slowdown at inference** — inference speed is same as a network without dropout, using full capacity

**Summary**

- Training: slower to converge, more epochs needed → longer total training time
- Inference: no impact — full network used, same speed as any standard forward pass

### 13.12 What is L1/L2 regularization, and how does it affect a neural network?

**L1/L2 Regularization in Neural Networks**

- Technique to prevent overfitting by adding penalty term to loss function based on weight magnitudes
- $\text{Total Loss} = \text{Original Loss} + \lambda \times \text{Penalty}$

**L1 Regularization (Lasso)**

- $\text{Penalty} = \lambda \sum_i \lvert w_i \rvert$
- Drives some weights to exactly **zero** → sparse network (effectively removes some connections)
- Acts like built-in feature/neuron selection

**L2 Regularization (Ridge / Weight Decay)**

- $\text{Penalty} = \lambda \sum_i w_i^2$
- Shrinks weights toward zero but rarely exactly zero → smaller, more distributed weights
- Most commonly used in neural networks (often called "weight decay")

**Effect on neural network**

- Prevents weights from growing too large → reduces model complexity/capacity
- Smaller weights → smoother decision boundaries → less overfitting, better generalization
- L2 more common in deep learning (smoother optimization, doesn't zero out neurons abruptly)
- $\lambda$ (regularization strength) is a hyperparameter — too high causes underfitting, too low has little effect

### 13.13 What is batch normalization, and why is it used?

**Batch Normalization**

- Technique that normalizes layer inputs (activations) within each mini-batch during training
- Normalizes to mean=0, std=1, then applies learnable scale ($\gamma$) and shift ($\beta$) parameters
- Formula: $\hat{x} = \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}$, then $y = \gamma \hat{x} + \beta$

**Why used**

- **Reduces internal covariate shift**: stabilizes distribution of layer inputs as weights update during training
- **Faster convergence**: allows higher learning rates without instability
- **Reduces vanishing/exploding gradients**: keeps activations in stable range through network
- **Acts as slight regularizer**: batch statistics add noise, reducing need for dropout somewhat
- **Less sensitive to weight initialization**: normalizes regardless of initial scale

**Implementation notes**

- Applied typically after linear/conv layer, before activation function
- Uses batch statistics during training; uses running mean/variance (computed during training) at inference
- Variants: Layer Normalization (used in Transformers, normalizes across features not batch), Group Norm, Instance Norm

### 13.14 What are the hyperparameters for batch normalization that can be optimized?

**Batch Normalization — Tunable Hyperparameters**

- **Momentum**: controls how running mean/variance (used at inference) are updated via exponential moving average — typically 0.9–0.99
- **Epsilon ($\epsilon$)**: small constant added to variance for numerical stability (avoid division by zero) — typically 1e-5
- **$\gamma$ (scale) and $\beta$ (shift)**: learnable parameters per feature, initialized ($\gamma = 1$, $\beta = 0$) but learned during training — not manually tuned but initialization can matter
- **Batch size**: indirectly affects BN — larger batches give more stable/accurate batch statistics; very small batches make BN noisy/unreliable
- **Placement**: before or after activation function (commonly before) — architectural choice affecting performance
- **Axis/dimension** to normalize over: standard BN normalizes over batch dimension per channel/feature; choice matters for conv layers (per-channel) vs dense layers
- **Whether to use running statistics or batch statistics at inference**: standard is running stats, but affects behavior with small/varying batch sizes
- In practice, momentum and epsilon are the main hyperparameters tuned; $\gamma$/$\beta$ are learned automatically

### 13.15 What is parameter sharing in deep learning?

**Parameter Sharing**

- Technique where the same set of weights/parameters is reused across different parts of the input or network, instead of learning separate parameters for each
- Reduces total number of parameters → less memory, less overfitting, more efficient training

**Examples**

- **CNNs (Convolutional Neural Networks)**: same filter/kernel weights applied across all spatial locations of an image
  - Assumes feature (e.g., edge detector) useful anywhere in image → learn once, apply everywhere
  - Drastically reduces parameters vs fully-connected layer
- **RNNs (Recurrent Neural Networks)**: same weight matrix applied at every time step across the sequence
  - Same transformation logic reused regardless of sequence position/length
- **Transformers**: shared weights across positions in self-attention (position-independent weight matrices), and often shared embeddings between input/output layers

**Benefits**

- Fewer parameters → reduces overfitting, especially with limited data
- Enables **translation invariance** (CNNs) — detects pattern regardless of position in image
- Allows handling variable-length inputs (RNNs) — same weights work for any sequence length
- More memory/compute efficient

### 13.16 What is representation learning, and why is it useful?

**Representation Learning**

- Process where a model automatically learns useful features/representations from raw data, instead of relying on manual feature engineering
- Data transformed into a form (often dense vectors/embeddings) that captures underlying structure, patterns, and relationships

**How it works**

- Achieved through deep learning layers — each layer learns increasingly abstract representations (e.g., edges → shapes → objects in CNNs)
- Can be **supervised** (learned via labeled task) or **self-supervised/unsupervised** (learned via pretext tasks, e.g., masked language modeling, contrastive learning)

**Why useful**

- Removes need for manual, domain-specific feature engineering
- Learned representations often **transferable** across tasks (transfer learning, pretrained models)
- Captures complex, non-linear relationships humans might miss
- Enables better performance on unstructured data (text, images, audio)
- Foundation for embeddings, pretrained models (BERT, ResNet, CLIP)

**Examples**

- Word embeddings (Word2Vec, BERT) — learn semantic representations of words
- CNN feature maps — learn visual representations (edges, textures, objects)
- Autoencoders — learn compressed representations by reconstructing input

### 13.17 What is a generative model, and how does it differ from a discriminative model?

**Generative vs Discriminative Models**

**Generative Model**

- Learns the joint probability distribution $P(X, Y)$ — models how data is generated
- Can generate new samples resembling training data
- Learns underlying distribution of each class
- Examples: Naive Bayes, GANs, VAEs, Diffusion Models, GPT (language generation)

**Discriminative Model**

- Learns conditional probability $P(Y \mid X)$ — directly models decision boundary between classes
- Focuses on distinguishing between classes, not generating data
- Examples: Logistic Regression, SVM, Random Forest, standard CNNs for classification

| Aspect | Generative | Discriminative |
| --- | --- | --- |
| Models | $P(X, Y)$ — joint distribution | $P(Y \mid X)$ — conditional distribution |
| Goal | Learn how data is generated | Learn decision boundary |
| Can generate new data | Yes | No |
| Typically | More flexible, can handle missing data | Better performance on classification tasks (with enough data) |
| Examples | Naive Bayes, GANs, VAEs, GPT | Logistic Regression, SVM, standard classifiers |

- Generative models can be used for classification too (via Bayes' rule: $P(Y \mid X) \propto P(X \mid Y)\,P(Y)$) but typically less accurate than discriminative for pure classification when data is abundant

### 13.18 Can you explain how a generative model works?

**How Generative Models Work**

- Goal: learn the underlying probability distribution of training data $P(X)$ (or joint $P(X, Y)$), then use it to generate new samples

**General approach**

1. Model learns patterns/structure in training data
2. Learns to approximate the true data distribution
3. Sample from learned distribution to generate new, realistic data points

**Common architectures & how they work**

**GANs (Generative Adversarial Networks)**

- Two networks compete: **Generator** creates fake samples from random noise, **Discriminator** tries to distinguish real vs fake
- Trained adversarially — generator improves by fooling discriminator, discriminator improves at detecting fakes
- Converges when generator produces realistic samples discriminator can't distinguish

**VAEs (Variational Autoencoders)**

- **Encoder** compresses input into latent distribution (mean, variance)
- Sample from latent space, **Decoder** reconstructs/generates output
- Trained to reconstruct input while regularizing latent space to follow a known distribution (e.g., Gaussian)

**Diffusion Models**

- Gradually add noise to data (forward process), then train model to reverse process (denoise step-by-step)
- Generate new samples by starting from pure noise and iteratively denoising

**Autoregressive models (GPT, PixelRNN)**

- Generate output sequentially, one element at a time, conditioned on previous elements
- $P(x) = \prod_i P(x_i \mid x_1, \ldots, x_{i-1})$
- All approaches aim to capture data distribution well enough to produce novel, realistic samples

### 13.19 Explain Encoder-Decoder Architecture.

**Encoder-Decoder Architecture**

- Neural network architecture with two components: an **encoder** that compresses input into a representation, and a **decoder** that generates output from that representation
- Used for tasks where input and output are sequences (possibly different lengths/domains)

**Encoder**

- Processes input sequence, compresses it into a fixed-size (or set of) context vector(s)/latent representation
- Captures the meaning/features of the input

**Decoder**

- Takes encoder's representation, generates output sequence step-by-step
- Often conditioned on previous outputs (autoregressive generation)

**Common implementations**

- **RNN/LSTM-based**: encoder RNN processes input, final hidden state passed to decoder RNN — used in early seq2seq (machine translation)
- **Attention-based Seq2Seq**: decoder attends to all encoder hidden states (not just final one) — solves bottleneck of fixed-size context vector
- **Transformer Encoder-Decoder**: uses self-attention + cross-attention — encoder processes full input in parallel, decoder generates output attending to encoder outputs (e.g., original Transformer, T5, BART)

**Use cases**

- Machine translation
- Text summarization
- Image captioning (CNN encoder + RNN/Transformer decoder)
- Speech recognition

**Limitation of basic version**

- Fixed-size context vector (in RNN-based) creates information bottleneck for long sequences → addressed via **attention mechanism**

### 13.20 What is the difference between encoder-only, decoder-only, and encoder-decoder Transformer architectures?

> **See also:** [Question 17.31 — What is the difference between encoder-only, decoder-only, and encoder-decoder Transformer architectures?](../part-4-generative-ai/17-large-language-model-foundations.md#1731-what-is-the-difference-between-encoder-only-decoder-only-and-encoder-decoder-transformer-architectures), which covers the same question from another angle.

**Encoder-Only vs Decoder-Only vs Encoder-Decoder Transformers**

| **Aspect** | **Encoder-Only** | **Decoder-Only** | **Encoder-Decoder** |
| --- | --- | --- | --- |
| Attention | Bidirectional (sees full context) | Causal/masked (only sees past tokens) | Encoder: bidirectional; Decoder: causal + cross-attention to encoder |
| Best for | Understanding tasks | Generation tasks | Sequence-to-sequence tasks |
| Examples | BERT, RoBERTa | GPT, LLaMA, Claude | T5, BART, original Transformer |
| Use cases | Classification, NER, embeddings, sentence similarity | Text generation, chatbots, code generation | Translation, summarization, question answering |
| Training objective | Masked Language Modeling (MLM) | Next-token prediction (autoregressive) | Denoising/seq2seq objectives |
| Output | Contextual representations/embeddings | Generated text sequence | Generated output sequence conditioned on input |

**Key distinctions**

- **Encoder-only**: processes entire input at once, each token attends to all others (past + future) — good for understanding/representation tasks, not generation
- **Decoder-only**: generates tokens one at a time, each token only attends to previous tokens (causal masking) — dominant architecture for modern LLMs (GPT-family, Claude)
- **Encoder-decoder**: separate encoder processes input fully, decoder generates output while cross-attending to encoder's representations — good when input and output are distinct sequences (e.g., translation)

### 13.21 What is Latent Space?

**Latent Space**

- A lower-dimensional, compressed representation space where data is encoded to capture its essential underlying features/structure
- Points in latent space represent abstract, learned features rather than raw input values

**Key characteristics**

- Similar inputs map to nearby points in latent space (captures semantic/structural similarity)
- Dimensions often don't have direct human-interpretable meaning, but can capture meaningful factors of variation (e.g., in face generation: smile, pose, lighting)
- Enables interpolation — moving between points can generate smooth transitions between data samples

**Where it's used**

- **Autoencoders/VAEs**: encoder maps input to latent space, decoder reconstructs/generates from it
- **GANs**: generator maps random noise vector (latent space) to realistic output
- **Embeddings**: word/sentence embeddings are a form of latent space representation
- **Diffusion models**: some operate in latent space (e.g., Stable Diffusion) instead of pixel space for efficiency

**Why useful**

- Compresses high-dimensional data into meaningful, dense representation
- Enables generation of new samples by sampling/manipulating latent space
- Facilitates tasks like interpolation, clustering, anomaly detection (outliers = far from normal latent region)

### 13.22 What are autoencoders? Explain their layers and practical uses.

**Autoencoders**

- Unsupervised neural network that learns to compress data into a lower-dimensional representation, then reconstruct it back to original
- Trained to minimize reconstruction error (output ≈ input)

**Layers/Architecture**

1. **Encoder**: series of layers that compress input into smaller latent representation (bottleneck)
   - Input → hidden layers (decreasing size) → latent vector ($z$)
2. **Bottleneck/Latent space**: compressed, lower-dimensional representation of input — forces network to learn only most important features
3. **Decoder**: series of layers that reconstruct original input from latent representation
   - Latent vector ($z$) → hidden layers (increasing size) → output (reconstructed input)

- Trained end-to-end using reconstruction loss (MSE for continuous data, cross-entropy for binary/categorical)

**Practical uses**

- **Dimensionality reduction**: non-linear alternative to PCA
- **Anomaly detection**: high reconstruction error indicates anomalous/out-of-distribution input
- **Denoising**: train on noisy input, clean output (Denoising Autoencoders) — learns to remove noise
- **Image compression**: compact latent representation of images
- **Feature learning/pretraining**: learned latent features used for downstream tasks
- **Recommendation systems**: collaborative filtering via reconstructing user-item matrices

**Variants**

- **Variational Autoencoder (VAE)**: latent space follows probability distribution — enables generation of new samples
- **Denoising Autoencoder**: robust to noisy inputs
- **Sparse Autoencoder**: enforces sparsity in latent representation

### 13.23 What is a Variational Autoencoder (VAE), and how is it different from a traditional autoencoder?

**Variational Autoencoder (VAE) vs Traditional Autoencoder**

**VAE**

- Generative model that learns a **probabilistic** latent space instead of a fixed point
- Encoder outputs **parameters of a distribution** (mean $\mu$, variance $\sigma^2$) for each input, not a single latent vector
- Sample $z$ from this distribution (using reparameterization trick), decoder reconstructs from sampled $z$
- **Loss function** = Reconstruction Loss + KL Divergence (regularizes latent space to follow standard normal distribution)

**Traditional Autoencoder**

- Encoder maps input to a **single fixed point** in latent space (deterministic)
- No constraint on latent space structure/distribution
- Only trained to minimize reconstruction loss

**Key Differences**

| Aspect | Traditional Autoencoder | VAE |
| --- | --- | --- |
| Latent representation | Fixed point (deterministic) | Probability distribution ($\mu$, $\sigma$) |
| Latent space structure | No constraint, can have gaps | Continuous, smooth (regularized via KL divergence) |
| Generative capability | Poor — latent space not smooth/continuous | Good — can sample new points, generate new realistic data |
| Loss function | Reconstruction loss only | Reconstruction loss + KL divergence |
| Interpolation | Latent space may have "holes" (unrealistic outputs) | Smooth interpolation between points produces meaningful outputs |

**Why VAE is better for generation**

- KL divergence term forces latent space to be continuous and well-structured (close to standard normal)
- Enables meaningful sampling/interpolation — nearby points in latent space produce similar, realistic outputs
- Traditional autoencoders can have discontinuous latent space, leading to unrealistic outputs when sampling

### 13.24 How does VAE impose a probabilistic structure on the latent space, and why is that important?

**How VAE Imposes Probabilistic Structure on Latent Space**

**Mechanism**

1. **Encoder outputs distribution parameters**, not a point: for each input $x$, encoder outputs mean $\mu(x)$ and variance $\sigma^2(x)$ — parameterizing a Gaussian distribution $q(z \mid x)$
2. **Reparameterization trick**: sample $z = \mu + \sigma \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, 1)$
   - Allows sampling while keeping the process differentiable (backprop-able) — direct sampling isn't differentiable; this trick separates randomness ($\epsilon$) from learnable parameters ($\mu$, $\sigma$)
3. **KL Divergence regularization**: loss includes $\operatorname{KL}(q(z \mid x) \parallel p(z))$, where $p(z)$ = standard normal $\mathcal{N}(0, 1)$
   - Forces the learned distribution $q(z \mid x)$ for each input to stay close to a standard normal prior
   - Penalizes latent distributions that drift too far from $\mathcal{N}(0, 1)$ — prevents encoder from "cheating" by spreading points arbitrarily far apart
4. **Total loss**: $\text{Reconstruction Loss} + \beta \cdot \text{KL Divergence}$ ($\beta$ controls regularization strength — $\beta$-VAE)

**Why this is important**

- **Continuity**: Forces nearby points in latent space to decode to similar outputs — no discontinuous "holes"
- **Completeness**: Every point sampled from the prior $\mathcal{N}(0, 1)$ should decode to a realistic output (since all data's distributions were pulled toward this same prior)
- **Enables generation**: Because latent space is well-structured and covers a known distribution, you can sample random $z \sim \mathcal{N}(0, 1)$ directly from the prior and decode it into a realistic new sample — impossible reliably with vanilla autoencoders
- **Smooth interpolation**: moving between two points in latent space produces gradual, meaningful transitions in output (e.g., morphing between two generated faces)
- **Prevents overfitting/memorization**: regularization stops the model from just memorizing exact latent points for each training example

**Trade-off**

- Too much KL weight (high $\beta$) → latent space very regularized but reconstruction quality drops (posterior collapse risk)
- Too little KL weight → latent space less structured, generation quality suffers, behaves more like vanilla autoencoder

### 13.25 What is the architecture of a Generative Adversarial Network (GAN)?

**GAN (Generative Adversarial Network) Architecture**

- Two neural networks trained simultaneously in an adversarial (competing) setup

**Generator (G)**

- Takes random noise vector $z$ (from latent space, e.g., $\mathcal{N}(0, 1)$) as input
- Generates fake data samples: $G(z)$ → fake sample
- Goal: produce samples realistic enough to fool the discriminator

**Discriminator (D)**

- Binary classifier — takes a sample (real or fake) as input
- Outputs probability that input is real: $D(x) \in [0, 1]$
- Goal: correctly distinguish real samples (from training data) vs fake samples (from generator)

**Training process (adversarial)**

1. Generator creates fake sample from noise: $G(z)$
2. Discriminator evaluates both real sample ($x$) and fake sample ($G(z)$)
3. Discriminator updated to better classify real vs fake (maximize correct classification)
4. Generator updated to better fool discriminator (maximize D's error on fake samples)
5. Repeat — both networks improve together in a minimax game

**Objective function**

- $\min_G \max_D \left[\mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]\right]$
- Discriminator maximizes this; Generator minimizes it

**Convergence**

- Ideally reaches Nash equilibrium — generator produces samples indistinguishable from real data, discriminator outputs \~0.5 (can't tell real from fake)

**Challenges**

- Training instability, mode collapse (generator produces limited variety), vanishing gradients
- Solutions: Wasserstein GAN (WGAN), spectral normalization, careful learning rate balancing

### 13.26 What are the roles of the generator and discriminator in a GAN?

**Roles of Generator and Discriminator in GAN**

**Generator (G)**

- Takes random noise ($z$) as input, transforms it into synthetic/fake data samples
- Goal: learn to produce samples realistic enough to fool the discriminator
- Improves over training by receiving feedback (gradients) based on how well it deceived the discriminator
- Never sees real data directly — only learns indirectly through discriminator's feedback

**Discriminator (D)**

- Binary classifier that takes a sample (real or generated) as input
- Goal: correctly distinguish real data (from training set) from fake data (from generator)
- Outputs probability of input being real (close to 1) vs fake (close to 0)
- Provides the training signal that helps the generator improve

**Adversarial relationship**

- Generator tries to **minimize** discriminator's ability to detect fakes
- Discriminator tries to **maximize** its accuracy at detecting fakes
- Both compete in a minimax game — as one improves, it pushes the other to improve too
- Training succeeds when generator produces samples so realistic that discriminator can no longer reliably distinguish them (ideally \~50% accuracy, random guessing)

**Analogy**

- Generator = counterfeiter trying to make fake currency
- Discriminator = police trying to detect fake currency
- Both get better over time through this competition

### 13.27 What is mode collapse in GANs, and how can it be mitigated?

**Mode Collapse in GANs**

- Failure mode where the generator produces limited variety of outputs — collapses to generating only a few types of samples (or even a single sample) regardless of input noise
- Generator finds a small set of outputs that reliably fool the discriminator, and keeps generating those instead of covering the full diversity of the real data distribution
- Discriminator overfits to these specific fakes, but generator doesn't get pushed toward diversity

**Why it happens**

- Generator optimizes to exploit discriminator's weaknesses rather than learning the true data distribution
- Minimax game imbalance — discriminator too strong/weak relative to generator
- No explicit mechanism forcing generator to cover all modes of data distribution

**Mitigation strategies**

- **Wasserstein GAN (WGAN)**: uses Wasserstein distance instead of JS divergence — provides smoother gradients, more stable training, reduces mode collapse
- **Minibatch discrimination**: discriminator looks at batch of samples together, can detect lack of diversity (near-identical samples flagged as fake)
- **Unrolled GANs**: generator optimized against several future discriminator updates, not just current state
- **Feature matching**: generator loss based on matching intermediate feature statistics (from discriminator) rather than just fooling final output
- **Adding noise/diversity terms**: penalize generator for producing similar outputs across different noise inputs
- **Spectral normalization**: stabilizes discriminator training, indirectly helps prevent collapse
- **Multiple generators/discriminators**: ensemble approaches to encourage diverse outputs (e.g., MAD-GAN)
- **Better architecture/hyperparameter tuning**: balanced learning rates for G and D, proper initialization

### 13.28 How are GANs used in image synthesis or image-to-image translation tasks?

**GANs for Image Synthesis & Image-to-Image Translation**

**Image Synthesis (generating new images from noise)**

- Generator takes random noise vector → produces realistic image
- Discriminator distinguishes real vs generated images
- **Examples**:
  - **DCGAN**: convolutional architecture for stable image generation
  - **StyleGAN**: generates highly realistic, high-resolution images (faces), allows style/feature control at different layers
  - **BigGAN**: large-scale, high-fidelity image generation

**Image-to-Image Translation (transforming one image domain to another)**

- **Pix2Pix**:
  - Uses **conditional GAN (cGAN)** — generator takes input image (not just noise) and translates it to target domain
  - Requires **paired training data** (input-output image pairs, e.g., sketch → photo)
  - Generator: typically U-Net (encoder-decoder with skip connections)
  - Discriminator: PatchGAN — classifies local patches as real/fake instead of whole image (captures fine texture details)
- **CycleGAN**:
  - For **unpaired** image translation (no matching input-output pairs needed, e.g., horses ↔ zebras, photo ↔ painting style)
  - Uses **two generators** ($G: A \to B$, $F: B \to A$) and **two discriminators**
  - **Cycle consistency loss**: $F(G(A)) \approx A$ — translating to the target domain and back should reconstruct the original image
  - Enables style transfer, domain adaptation without paired examples

**Common use cases**

- Photo-realistic image generation (faces, art, synthetic data)
- Style transfer (photo → painting style)
- Super-resolution (low-res → high-res, e.g., SRGAN)
- Colorization (grayscale → color)
- Domain adaptation (synthetic → real-world images, e.g., for training autonomous vehicles)
- Data augmentation (generating synthetic training data)

### 13.29 Explain Convolutional Neural Networks (CNNs).

**Convolutional Neural Networks (CNN)**

- Neural network architecture designed for grid-like data, primarily images — exploits spatial structure
- Uses **convolution operations** instead of full connectivity to reduce parameters and capture local patterns

**Key layers**

**Convolutional Layer**

- Applies learnable filters/kernels that slide across input, computing dot products → feature maps
- Detects local patterns (edges, textures, shapes)
- Parameters: filter size, stride, padding, number of filters
- Uses **parameter sharing** — same filter applied across entire image → fewer parameters than fully-connected

**Activation (ReLU)**

- Applied after convolution to introduce non-linearity

**Pooling Layer**

- Downsamples feature maps, reduces spatial dimensions (Max Pooling most common)
- Provides translation invariance, reduces computation, controls overfitting

**Fully Connected Layer**

- Final layers, flatten feature maps and connect to output (classification/regression)

**Key concepts**

- **Local receptive field**: each neuron connects only to a small region of input (not entire image)
- **Parameter sharing**: same filter weights reused across spatial locations → efficient, translation-invariant feature detection
- **Hierarchical feature learning**: early layers learn simple features (edges), deeper layers learn complex features (shapes, objects)

**Use cases**

- Image classification, object detection, segmentation
- Also applied to text (1D convolutions), audio, video

### 13.30 Explain filters in CNN.

**Filters in CNN**

- Small learnable matrices (kernels) that slide across input to detect specific patterns/features
- Typically small spatial size (e.g., 3×3, 5×5) but extend through full depth of input (e.g., 3 channels for RGB)

**How they work**

- Filter slides (convolves) across input using a defined **stride**
- At each position, computes element-wise multiplication between filter and input patch, sums result → single output value
- Produces a **feature map** (activation map) — one feature map per filter

**What filters detect**

- **Early layers**: simple, low-level features — edges, corners, color gradients
- **Deeper layers**: complex, high-level features — shapes, textures, object parts, eventually whole objects
- Each filter specializes in detecting a specific pattern

**Key parameters**

- **Number of filters**: determines depth of output feature map (more filters = more features detected, but more parameters/computation)
- **Filter size (kernel size)**: spatial dimensions (e.g., 3×3) — smaller filters capture fine details, larger capture broader patterns
- **Stride**: step size the filter moves each time — larger stride = more downsampling
- **Padding**: adds border pixels (zero-padding) to control output size — "same" padding preserves input size, "valid" padding shrinks it

**Why effective**

- **Parameter sharing**: same filter weights used across entire image → far fewer parameters than a fully-connected layer, and detects pattern regardless of location (translation invariance)
- Filters are **learned automatically** during training via backpropagation — no manual feature engineering needed

### 13.31 Explain the stride in CNN.

**Stride in CNN**

- Number of pixels the filter/kernel moves (shifts) at each step while sliding across the input
- Controls how much overlap occurs between successive filter applications

**Effect of stride**

- **Stride = 1**: filter moves one pixel at a time — maximum overlap, produces larger output feature map (finer detail retained)
- **Stride = 2 (or more)**: filter skips pixels — less overlap, produces smaller output feature map (downsampling effect)

**Output size formula**

- $\text{Output size} = \left\lfloor \frac{\text{Input size} - \text{Filter size} + 2 \times \text{Padding}}{\text{Stride}} \right\rfloor + 1$

**Why stride matters**

- **Larger stride**:
  - Reduces spatial dimensions of output → less computation, less memory
  - Acts similar to pooling — provides downsampling
  - Risk of missing fine-grained details (larger jumps between filter applications)
- **Smaller stride (stride=1)**:
  - Preserves more spatial detail/resolution
  - More computation (more filter applications), larger output feature maps
  - Common for capturing fine-grained features

**Practical use**

- Stride=1 common in convolutional layers wanting to preserve detail
- Stride=2 sometimes used instead of pooling layers for downsampling (e.g., in some modern architectures) — learns downsampling instead of fixed pooling operation

### 13.32 Explain padding in CNN.

**Padding in CNN**

- Adding extra pixels (typically zeros) around the border of input before applying convolution
- Controls spatial size of output feature map and preserves border information

**Why needed**

- Without padding, output shrinks after every convolution (filter can't center on edge pixels)
- Edge/corner pixels get used in fewer convolutions than center pixels → border information underrepresented
- Padding helps retain spatial dimensions and edge features

**Types**

- **Valid padding (no padding)**:
  - No extra pixels added
  - Output size shrinks: $\text{Output} = \text{Input} - \text{Filter} + 1$
  - Loses some border information
- **Same padding**:
  - Adds enough zero-padding so output size equals input size (with stride=1)
  - Preserves spatial dimensions across layers — useful for deep networks

**Output size formula**

- $\text{Output size} = \left\lfloor \frac{\text{Input size} - \text{Filter size} + 2 \times \text{Padding}}{\text{Stride}} \right\rfloor + 1$

**Benefits**

- Preserves spatial resolution across layers (allows deeper networks without shrinking too fast)
- Ensures border/edge pixels contribute equally to feature extraction
- Gives more control over output dimensions for architecture design

### 13.33 Explain pooling in CNN.

**Pooling in CNN**

- Downsampling operation that reduces spatial dimensions (height, width) of feature maps while retaining important information
- Applied after convolution + activation, operates independently per channel

**Types**

**Max Pooling**

- Takes maximum value from each patch/window (e.g., 2×2)
- Most commonly used — retains strongest/most prominent features
- Provides some translation invariance (small shifts in input don't change max value much)

**Average Pooling**

- Takes average value from each patch/window
- Smoother downsampling, less commonly used than max pooling (except in specific cases like Global Average Pooling)

**Global Average Pooling (GAP)**

- Averages entire feature map into a single value per channel
- Often used before final classification layer instead of flattening + fully connected layers (fewer parameters)

**Key parameters**

- **Pool size**: window size (e.g., 2×2)
- **Stride**: typically equals pool size (non-overlapping windows)

**Why used**

- **Reduces spatial dimensions**: less computation, memory for subsequent layers
- **Controls overfitting**: fewer parameters downstream
- **Translation invariance**: small shifts/distortions in input have less effect on output
- **Reduces sensitivity to exact feature location**: focuses on presence of feature, not precise position

**Trade-off**

- Loses some spatial/precise location information (can matter for tasks like segmentation, where exact pixel location matters)

### 13.34 Explain fully connected layers in CNN.

Fully connected (FC) layers—also called dense layers—are the layers near the end of a CNN that take the high-level features extracted by earlier convolutional/pooling layers and use them to make a final prediction (like classifying an image).

#### Where they sit in a CNN

A typical CNN pipeline looks like:

**Input → \[Conv → Activation → Pooling\] × N → Flatten → Fully Connected layers → Output**

The convolutional layers act as feature extractors (edges, textures, shapes, object parts), while the fully connected layers act as the "reasoning" stage that combines those features into a decision.

- **Flatten step:** the final feature map (e.g., 7×7×512) is flattened into a 1-D vector (25,088 values) so it can feed a dense layer. Each FC neuron computes $y = f(W \cdot x + b)$ and connects to *every* input value, so FC layers learn global combinations of features, unlike the local, weight-shared filters of conv layers.
- **Typical head:** one or two hidden FC layers (often with ReLU + dropout), then an output FC layer sized to the number of classes, followed by softmax (multi-class) or sigmoid (binary/multi-label).
- **Parameter cost:** FC layers usually hold most of a classic CNN's parameters. In VGG-16, the first FC layer alone (25,088 × 4,096 ≈ 103M weights) is most of the model's ~138M parameters, which makes these layers the main source of overfitting and memory use.
- **Modern practice:** many architectures (ResNet, EfficientNet) swap the big flatten + FC stack for **Global Average Pooling** followed by a single FC layer. This cuts parameters sharply, reduces overfitting and lets the network accept variable input sizes.
- **Production angle:** in transfer learning you usually freeze the convolutional backbone and replace or retrain only the FC head for the new label set. That is cheap and works well with little labelled data.

### 13.35 What is a Recurrent Neural Network (RNN)?

#### Recurrent Neural Network (RNN)

- An RNN is a class of neural networks designed specifically for processing **sequential data**, such as text, speech, time series, video frames, or any data where the order of elements carries meaning.
- Unlike feedforward neural networks, which assume all inputs are independent of each other, RNNs introduce the concept of **memory** through a hidden state that is passed from one time step to the next.
- **Core mechanism:**
  - At each time step $t$, the RNN takes two inputs: the current input vector `x_t` and the previous hidden state `h_(t-1)`.
  - It computes a new hidden state: $h_t = f(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$, where $f$ is typically a non-linear activation like tanh or ReLU.
  - It can also produce an output at that step: $y_t = g(W_{hy} h_t + b_y)$.
  - Crucially, the **same weight matrices** ($W_{xh}$, $W_{hh}$, $W_{hy}$) are shared/reused across all time steps — this is called **parameter sharing** and is what allows RNNs to generalize across sequences of varying length.
- **Unrolling:** Conceptually, an RNN can be "unrolled" through time into a chain of repeated units, one per time step, making it easier to visualize how information flows forward through the sequence and how gradients flow backward during training.
- **Training:** RNNs are trained using **Backpropagation Through Time (BPTT)**, a variant of standard backpropagation where gradients are propagated backward across all time steps of the unrolled network.
- **Key limitation — Vanishing/Exploding Gradient Problem:**
  - Because the same weight matrix is multiplied repeatedly across many time steps during backpropagation, gradients can shrink toward zero (vanishing) or grow uncontrollably (exploding) as sequence length increases.
  - This makes it very difficult for vanilla RNNs to learn **long-range dependencies** — i.e., relationships between elements that are far apart in the sequence.
- **Variants that address this limitation:**
  - **LSTM (Long Short-Term Memory):** Introduces a cell state and three gates (input, forget, output) that regulate what information is added, retained, or discarded, allowing better retention of long-term dependencies.
  - **GRU (Gated Recurrent Unit):** A simplified alternative to LSTM with two gates (update and reset), fewer parameters, and faster training, while achieving comparable performance in many tasks.
  - **Bidirectional RNN (BiRNN):** Processes the sequence in both forward and backward directions, allowing the model to use both past and future context at each time step.
- **Types of RNN architectures based on input/output structure:**
  - **One-to-one:** Standard neural network (not really a recurrent use case).
  - **One-to-many:** Single input, sequence output (e.g., image captioning).
  - **Many-to-one:** Sequence input, single output (e.g., sentiment classification).
  - **Many-to-many:** Sequence input, sequence output (e.g., machine translation, video frame labeling).
- **Common applications:**
  - Language modeling and text generation
  - Machine translation
  - Speech recognition
  - Time-series forecasting (stock prices, sensor data)
  - Sentiment analysis
  - Named entity recognition
- **Current relevance:**
  - RNNs (and LSTMs/GRUs) were the dominant architecture for sequence modeling before **Transformers** emerged.
  - Transformers, using **self-attention** instead of recurrence, largely replaced RNNs in large-scale NLP tasks because they parallelize better across sequence positions and capture long-range dependencies more effectively.
  - RNNs are still relevant in resource-constrained environments, streaming/online inference scenarios, and as a foundational concept for understanding sequence modeling.

### 13.36 What are the limitations of RNNs, and how are they solved?

#### Limitations of RNNs and Their Solutions

- **Vanishing Gradient Problem**
  - During Backpropagation Through Time (BPTT), gradients are computed by repeatedly multiplying the same weight matrix and derivative terms across many time steps.
  - When these values are less than 1, repeated multiplication causes gradients to shrink exponentially as they propagate backward, effectively becoming close to zero.
  - This means the network fails to update weights based on information from early time steps, making it very difficult to learn **long-range dependencies**.
  - **Solution:** LSTM and GRU architectures introduce gating mechanisms (forget, input, output gates in LSTM; update, reset gates in GRU) that regulate information flow and preserve gradients over long sequences via additive cell-state updates rather than repeated multiplicative updates.
- **Exploding Gradient Problem**
  - The opposite issue — when weight values are greater than 1, repeated multiplication causes gradients to grow exponentially, leading to numerical instability and huge, erratic weight updates.
  - **Solution:**
    - **Gradient clipping** — capping the gradient value at a threshold during backpropagation so it doesn't exceed a set limit.
    - Proper **weight initialization** (e.g., orthogonal initialization) to keep gradient scales stable.
- **Short-Term Memory / Difficulty Capturing Long-Range Dependencies**
  - Vanilla RNNs struggle to retain information from far back in a sequence, since the hidden state is repeatedly overwritten at each step.
  - **Solution:**
    - **LSTM** — uses a separate cell state that runs through the sequence with only minor linear interactions, allowing information to persist over long ranges.
    - **GRU** — simplifies LSTM's gating while still enabling better long-term retention than vanilla RNNs.
    - **Attention mechanisms** — allow the model to directly access any previous time step's information rather than relying solely on a compressed hidden state.
- **Sequential Computation — No Parallelization**
  - RNNs process one time step at a time, and each step depends on the previous one's output, making it impossible to parallelize computation across the sequence during training.
  - This leads to slow training, especially on long sequences and large datasets.
  - **Solution:** **Transformer architectures** eliminate recurrence, using **self-attention** to process all positions in a sequence simultaneously, enabling full parallelization on GPUs/TPUs.
- **Fixed-Size Context Bottleneck (in Encoder-Decoder RNNs)**
  - In sequence-to-sequence RNN models, the entire input sequence must be compressed into a single fixed-length context vector passed to the decoder, which becomes a bottleneck for long sequences.
  - **Solution:** **Attention mechanisms** (e.g., Bahdanau/Luong attention) allow the decoder to look back at all encoder hidden states at each decoding step, rather than relying on one fixed vector.
- **Difficulty Handling Very Long Sequences Overall**
  - Even LSTMs/GRUs degrade in performance as sequence length grows very large (hundreds/thousands of steps).
  - **Solution:** **Transformer-based models** (e.g., BERT, GPT) use self-attention to model relationships between all positions directly, regardless of distance, effectively solving the long-sequence dependency issue without recurrence.
- **Difficulty with Bidirectional Context (in unidirectional RNNs)**
  - A standard RNN only has access to past context, not future context, at each time step.
  - **Solution:** **Bidirectional RNNs (BiRNNs)** process the sequence in both forward and backward directions, and the two hidden states are combined to capture context from both sides.
- **Training Instability and Slow Convergence**
  - Sequential dependency and gradient issues make RNNs harder and slower to train, especially on deep or long-sequence architectures.
  - **Solution:**
    - **Layer normalization** and **batch normalization variants** adapted for recurrent networks.
    - **Residual/skip connections** in some RNN variants.
    - Careful **learning rate scheduling** and **gradient clipping**.

### 13.37 What are LSTM and GRU? How do they solve long-term dependency issues?

#### LSTM and GRU

##### What is LSTM (Long Short-Term Memory)?

- LSTM is a special type of RNN architecture designed to overcome the vanishing gradient problem and effectively learn **long-term dependencies** in sequential data.
- It introduces a separate **cell state ($C_t$)** in addition to the hidden state ($h_t$). The cell state acts like a "conveyor belt" that carries information across time steps with only minor linear modifications, allowing information to flow largely unchanged unless explicitly modified.
- LSTM controls the flow of information using **three gates**, each implemented as a sigmoid-activated layer that outputs values between 0 (block) and 1 (allow):
  - **Forget Gate ($f_t$):** Decides what information to discard from the previous cell state.
    - $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
  - **Input Gate ($i_t$):** Decides what new information to add to the cell state.
    - $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
    - A candidate value $\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$ is also computed.
  - **Cell State Update:** Combines the forget and input gates to update the cell state.
    - $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
  - **Output Gate ($o_t$):** Decides what part of the cell state to output as the hidden state.
    - $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
    - $h_t = o_t \odot \tanh(C_t)$
- **How this solves long-term dependency issues:**
  - The cell state update is **additive** ($f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$) rather than purely multiplicative like in vanilla RNNs, which prevents gradients from vanishing exponentially over many time steps.
  - The forget gate allows the network to **selectively retain or discard** information, so relevant information from far back in the sequence can be preserved unchanged for long durations if needed.
  - Gradients can flow through the cell state pathway with minimal decay, since there's no repeated matrix multiplication with the same weight matrix at every step along that specific pathway — this is often called the **constant error carousel**.

---

##### What is GRU (Gated Recurrent Unit)?

- GRU is a simplified variant of LSTM, introduced to achieve similar performance with **fewer parameters** and **less computational complexity** by combining the cell state and hidden state into a single hidden state.
- GRU uses **two gates** instead of three:
  - **Update Gate ($z_t$):** Controls how much of the past hidden state should be carried forward versus how much of the new candidate state should be used. It essentially combines the roles of LSTM's forget and input gates.
    - $z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$
  - **Reset Gate ($r_t$):** Controls how much of the past hidden state to forget when computing the new candidate state.
    - $r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$
  - **Candidate Hidden State:**
    - $\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$
  - **Final Hidden State Update:**
    - $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$
- **How this solves long-term dependency issues:**
  - Similar to LSTM, the final hidden state update is a **linear interpolation** between the previous hidden state and the new candidate state, controlled by the update gate — this additive structure helps preserve gradients over long sequences.
  - When $z_t$ is close to 0, the hidden state from the past is carried forward almost unchanged, allowing information to persist over many time steps without significant decay.
  - The reset gate allows the model to drop irrelevant past information when computing new candidates, giving it flexibility similar to LSTM's forget gate.

---

##### Key Differences Between LSTM and GRU

- **Number of gates:** LSTM has 3 gates (forget, input, output); GRU has 2 gates (update, reset).
- **State representation:** LSTM maintains two separate states (cell state and hidden state); GRU merges them into a single hidden state.
- **Parameters:** GRU has fewer parameters than LSTM, making it computationally cheaper and faster to train.
- **Performance:** Both perform similarly on many tasks; LSTM sometimes performs better on tasks requiring finer control over memory (very long sequences), while GRU often trains faster and works well on smaller datasets.
- **Use case selection:** GRU is often preferred when computational efficiency matters or the dataset is smaller; LSTM is preferred when the task requires more precise memory control over very long sequences.

### 13.38 What are the main gates in LSTM and their roles?

#### Main Gates in LSTM and Their Roles

- **LSTM has three main gates**, each implemented as a sigmoid neural network layer that outputs values between 0 and 1, acting as a filter to control how much information passes through.

---

##### 1. Forget Gate ($f_t$)

- **Role:** Decides what information from the **previous cell state** ($C_{t-1}$) should be discarded or retained.
- **Formula:**
  - $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
- **How it works:**
  - Takes the previous hidden state $h_{t-1}$ and the current input $x_t$, concatenates them, and passes them through a sigmoid function.
  - An output close to **0** means "completely forget this information."
  - An output close to **1** means "completely retain this information."
  - This output is multiplied element-wise with the previous cell state $C_{t-1}$.
- **Example use case:** In language modeling, when a new subject appears in a sentence, the forget gate may decide to discard the gender/number information of the old subject that's no longer relevant.

---

##### 2. Input Gate ($i_t$)

- **Role:** Decides what **new information** from the current input should be added to the cell state.
- **Formula:**
  - $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
  - Along with a candidate value: $\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$
- **How it works:**
  - The sigmoid part ($i_t$) decides **which values** to update (how much of the new information matters).
  - The tanh part ($\tilde{C}_t$) creates a vector of **new candidate values** that could be added to the cell state.
  - These are combined ($i_t \odot \tilde{C}_t$) and added to the cell state during the cell state update step.
- **Example use case:** When a new subject is introduced in a sentence, the input gate adds this new subject's relevant information (e.g., gender, number) into the cell state.

---

##### 3. Output Gate ($o_t$)

- **Role:** Decides what part of the **current cell state** should be exposed as the hidden state (output) at this time step.
- **Formula:**
  - $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
  - $h_t = o_t \odot \tanh(C_t)$
- **How it works:**
  - The sigmoid part decides which parts of the cell state are relevant to output right now.
  - The cell state is passed through a $\tanh$ (to normalize values between -1 and 1), then multiplied by the output gate's decision.
  - The result becomes the new hidden state $h_t$, which is used both as the output for this time step and passed to the next time step.
- **Example use case:** If the cell state contains information about a subject being singular or plural, the output gate might expose this information only when it's relevant to predicting the next word (e.g., a verb that needs to agree in number).

---

##### Summary Table

| Gate | Formula | Purpose | Acts On |
| --- | --- | --- | --- |
| Forget Gate | $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ | Decide what to discard from memory | Previous cell state $C_{t-1}$ |
| Input Gate | $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ | Decide what new info to add to memory | Candidate values $\tilde{C}_t$ |
| Output Gate | $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ | Decide what to output from memory | Current cell state $C_t$ |

- **Combined effect:** Together, these three gates give LSTM fine-grained control over what to remember, what to forget, and what to expose at each time step — this is precisely what allows it to maintain relevant information over long sequences while discarding irrelevant details, solving the long-term dependency problem faced by vanilla RNNs.

### 13.39 How to identify exploding gradient issues in your model?

- **Loss becomes NaN or Inf during training**
  - One of the most obvious signs — the loss value suddenly turns into `NaN` (Not a Number) or `Inf` (Infinity) after a few iterations or epochs.
  - This happens because extremely large gradient updates push weights to values that cause numerical overflow in subsequent computations.
- **Loss fluctuates wildly or increases sharply instead of decreasing**
  - Instead of a smooth, gradually decreasing loss curve, you observe sudden spikes or erratic jumps in the loss value from one iteration to the next.
  - The model may seem to be learning, then suddenly diverge without any clear pattern.
- **Model weights grow to very large values**
  - By monitoring the magnitude of weight values during training (e.g., printing `min`, `max`, `mean`, `norm` of weight matrices), you may notice they grow exponentially over time rather than stabilizing.
  - This can be tracked using tools like **TensorBoard**, `weights & biases (W&B)`, or simple logging of `torch.norm(weight)` at intervals.
- **Gradient norms become very large**
  - Explicitly compute and log the **L2 norm of gradients** for each layer during backpropagation.
  - If gradient norms are consistently increasing across iterations, or spike to unusually high values (e.g., in the thousands or millions when they should be in single digits), this is a strong indicator of exploding gradients.
  - Frameworks like PyTorch allow easy inspection via `parameter.grad.norm()` for each parameter.
- **Model outputs become extreme or nonsensical**
  - Predictions may saturate to extreme values (e.g., all outputs becoming 0, 1, or some constant), or output values that are clearly out of the expected range for the task.
- **Training becomes unstable across different runs**
  - The model behaves inconsistently across different random seeds or initializations — sometimes converging, sometimes diverging — which can hint at gradient instability rather than a fundamental data/architecture issue.
- **Visualizing gradient flow**
  - Plotting a **histogram of gradients** across layers (common in tools like TensorBoard) can reveal if certain layers have disproportionately large gradient magnitudes compared to others.
  - A "healthy" gradient flow typically shows gradients of a similar, moderate scale across layers; exploding gradients often show a few layers with dramatically larger magnitudes.
- **Sudden accuracy drop after a period of normal training**
  - If accuracy or other evaluation metrics suddenly crash after training seemed to be progressing normally, it's often linked to an exploding gradient event that corrupted the model weights.
- **Common tools/techniques used to monitor this in practice:**
  - **Gradient norm logging:** Track `torch.nn.utils.clip_grad_norm_()` return value (which gives the total norm before clipping) at every step.
  - **TensorBoard / Weights & Biases:** Visualize gradient norms, weight norms, and loss curves over time.
  - **Assertions/checks:** Add explicit checks like `torch.isnan(loss)` or `torch.isinf(gradient)` during training loops to catch issues early and halt training with a clear error message.
- **Once identified, common fixes include:**
  - **Gradient clipping** (clip by value or by norm) to cap gradient magnitudes.
  - **Lower learning rate** to reduce the size of weight updates.
  - **Better weight initialization** (e.g., Xavier/Glorot or He initialization) to keep gradient scales reasonable from the start.
  - **Batch normalization/layer normalization** to stabilize activations and gradients throughout training.

### 13.40 What is a Transformer architecture, and what makes it different from CNNs and RNNs?

#### Transformer Architecture

##### What is a Transformer?

- A Transformer is a neural network architecture introduced in the paper *"Attention Is All You Need"*, designed primarily for sequence modeling tasks like machine translation, text generation, and language understanding.
- Unlike RNNs, it does **not process data sequentially, step by step**. Instead, it processes an entire sequence **in parallel**, using a mechanism called **self-attention** to model relationships between all elements of the sequence simultaneously, regardless of their distance from each other.
- The original architecture follows an **Encoder-Decoder** structure:
  - **Encoder:** Maps an input sequence into a set of continuous representations, capturing contextual relationships between all tokens.
  - **Decoder:** Generates the output sequence one token at a time, attending to both the encoder's representations and previously generated tokens.

---

##### Core Components of a Transformer

- **Self-Attention Mechanism**
  - For each token in the sequence, self-attention computes a weighted representation by attending to **all other tokens** in the sequence, determining how relevant each other token is to the current one.
  - Computed using **Query (Q)**, **Key (K)**, and **Value (V)** vectors derived from the input embeddings:
    - $\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$
  - This allows the model to directly capture dependencies between distant tokens without the information having to pass through many intermediate steps (unlike RNNs).
- **Multi-Head Attention**
  - Instead of computing a single attention function, the model runs **multiple attention "heads"** in parallel, each learning to focus on different types of relationships (e.g., syntactic vs. semantic patterns).
  - The outputs of all heads are concatenated and linearly transformed to produce the final output.
- **Positional Encoding**
  - Since Transformers process tokens in parallel (no inherent sense of order like RNNs), **positional encodings** are added to the input embeddings to inject information about the position of each token in the sequence.
  - Typically implemented using sine and cosine functions of different frequencies, or learned positional embeddings.
- **Feedforward Neural Network (FFN)**
  - Each position's representation, after attention, is passed through a position-wise fully connected feedforward network (applied identically and independently to each position).
- **Residual Connections and Layer Normalization**
  - Each sub-layer (attention, FFN) has a residual/skip connection followed by layer normalization, helping stabilize training and allowing deeper architectures.
- **Stacked Layers**
  - Both encoder and decoder consist of multiple identical layers stacked on top of each other (e.g., 6, 12, 24, or more layers depending on model size), each refining the representation further.
- **Masked Self-Attention (Decoder only)**
  - In the decoder, self-attention is **masked** so that a token can only attend to previous tokens (and itself), preventing it from "seeing" future tokens during generation — preserving the autoregressive property.

---

##### What Makes Transformers Different from RNNs?

- **Parallelization vs. Sequential Processing**
  - RNNs process tokens one at a time, where each step depends on the previous hidden state — this is inherently sequential and cannot be parallelized during training.
  - Transformers process the entire sequence **simultaneously**, since self-attention computes relationships between all tokens at once — enabling much faster training on modern hardware (GPUs/TPUs).
- **Long-Range Dependency Handling**
  - RNNs struggle with long-range dependencies due to vanishing gradients across many time steps, even with LSTM/GRU improvements.
  - Transformers can directly connect any two tokens in a sequence in a **single step** via self-attention, regardless of their distance, making long-range dependency modeling far more effective.
- **Memory Mechanism**
  - RNNs rely on a compressed hidden state that is updated sequentially, which can lose information over long sequences.
  - Transformers don't rely on a single compressed state — every token has direct access to every other token's representation through attention weights.
- **Scalability**
  - RNNs become increasingly slow and harder to train as sequence length grows due to their sequential nature.
  - Transformers scale better with sequence length (though attention has quadratic complexity in sequence length, $O(n^2)$, which is its own limitation) and scale very well with more data and parameters (leading to large language models like GPT, BERT).

---

##### What Makes Transformers Different from CNNs?

- **Receptive Field / Context Range**
  - CNNs use local receptive fields (kernels/filters) that capture patterns within a limited window; capturing long-range dependencies requires stacking many convolutional layers or using dilated convolutions to expand the receptive field.
  - Transformers, via self-attention, have a **global receptive field from the very first layer** — every token attends to every other token directly, without needing many stacked layers to reach distant tokens.
- **Inductive Bias**
  - CNNs have strong **spatial locality** and **translation invariance** biases, which are especially useful for grid-like data such as images, where nearby pixels are highly correlated.
  - Transformers have **minimal inductive bias** about structure — they learn relationships purely from data via attention weights, which makes them very flexible but often requires more data to train effectively (though this is mitigated by pretraining and positional encodings).
- **Positional Awareness**
  - CNNs inherently capture some positional/spatial information through the fixed arrangement of the convolutional kernel over the input grid.
  - Transformers have no inherent sense of order/position (since attention treats input as a set), so positional encodings must be explicitly added to inject sequence order information.
- **Parameter Sharing Pattern**
  - CNNs share parameters (filters) across spatial locations, exploiting local patterns efficiently with fewer parameters for image-like data.
  - Transformers share parameters across positions differently — through the same attention mechanism and feedforward layers applied at every position, but relationships between positions are dynamically computed rather than fixed by a local kernel.
- **Typical Use Cases**
  - CNNs are traditionally dominant in **computer vision** tasks (image classification, object detection, segmentation) due to their strong spatial locality bias.
  - Transformers were originally designed for **NLP/sequence tasks**, but have since been adapted successfully to vision as well (e.g., **Vision Transformers - ViT**), by treating image patches as a sequence of tokens.

---

##### Summary Table

| Aspect | RNN | CNN | Transformer |
| --- | --- | --- | --- |
| Processing | Sequential | Local, parallel over spatial windows | Fully parallel over sequence |
| Long-range dependencies | Weak (vanishing gradients) | Requires deep stacking / dilation | Strong (direct attention to all tokens) |
| Positional info | Implicit (via order of processing) | Implicit (via kernel position) | Explicit (positional encodings needed) |
| Inductive bias | Sequential/temporal bias | Spatial locality, translation invariance | Minimal, learned via data |
| Parallelizability | Low | High (within layers) | Very high |
| Typical domain | Sequential data (text, time series) | Grid data (images) | Sequence data (NLP), adapted to vision (ViT) |

### 13.41 What is the basic difference between LSTM and Transformers?

- **Processing style:** LSTM processes sequences step-by-step (sequential); Transformer processes the whole sequence in parallel using self-attention.
- **Dependency handling:** LSTM passes information through a chain of hidden/cell states, so long-range dependencies can weaken over distance; Transformer connects any two tokens directly via attention, regardless of distance.
- **Speed/training:** LSTM is slower to train (no parallelization across time steps); Transformer trains much faster on GPUs due to parallel computation.
- **Positional info:** LSTM has order built-in naturally (sequential processing); Transformer needs explicit positional encodings since attention has no inherent sense of order.
- **Scalability:** Transformers scale better with data/model size, which is why they power modern large language models; LSTMs don't scale as effectively.

### 13.42 What are diffusion models?

- **Core idea:** Generative models that learn to create data (e.g., images) by reversing a gradual noising process.
- **Forward process:** Data is progressively corrupted by adding small amounts of Gaussian noise over many steps ($T$ steps), until it becomes pure random noise.
- **Reverse process:** A neural network (typically a U-Net) is trained to predict and remove the noise added at each step, gradually reconstructing data from pure noise back to a realistic sample.
- **Training objective:** The model learns to predict the noise component at each timestep (or equivalently the denoised sample), usually trained with a simple mean-squared-error loss between predicted and actual noise.
- **Sampling/Generation:** Starting from pure random noise, the model iteratively applies the learned reverse process step-by-step to produce a final, coherent sample (image, audio, etc.).
- **Key advantage:** Produces very high-quality, diverse outputs, often surpassing GANs in image fidelity and training stability (no adversarial training, no mode collapse).
- **Key drawback:** Slow sampling — generating a sample requires many sequential denoising steps (can be hundreds to thousands), though techniques like DDIM, distillation, and latent diffusion have significantly sped this up.
- **Popular examples:** DALL-E 2, Stable Diffusion, Imagen — all use diffusion models (often in a compressed "latent space" rather than raw pixel space, as in **Latent Diffusion Models**) for image generation.

### 13.43 Why does Diffusion work better than autoregression?

- **Parallel generation vs. sequential:** Autoregressive (AR) models generate outputs one token/pixel at a time, conditioned on all previous ones — this is slow, and errors can compound (exposure bias). Diffusion models refine the *entire* output simultaneously at each denoising step, so errors aren't strictly compounding token-by-token.
- **Global consistency:** AR models build outputs left-to-right, so early mistakes propagate forward and are hard to fix. Diffusion iteratively refines the whole sample together, allowing global structure to be corrected across many steps rather than committed early and locked in.
- **No fixed generation order:** AR models impose an artificial sequential order (e.g., raster-scan order for images), which isn't natural for data like images where relationships are spatial, not linear. Diffusion has no such ordering constraint — it denoises all parts together.
- **Better sample diversity/quality trade-off:** Diffusion models tend to avoid the repetitive, low-diversity outputs that AR models can produce (especially with greedy/beam search), and avoid the blurriness common in older non-autoregressive one-shot generators.
- **Training stability:** Diffusion training reduces to a simple denoising/regression objective (predict noise), which is stable and doesn't suffer teacher-forcing/exposure bias problems the way AR sequence models can.
- **Trade-off to note:** Diffusion isn't strictly "better" in all respects — it's typically slower at *inference* (many denoising steps) compared to a single forward pass in some AR setups, though AR generation is also inherently sequential and slow per-token. For images/audio specifically, diffusion has empirically produced state-of-the-art fidelity, which is why it's preferred there; AR still dominates in domains like text (LLMs) where sequential token dependencies are natural and well-suited to the AR objective.

### 13.44 Explain transfer learning and when to use it.

- **Core idea:** Reuse a model already trained on one task/dataset (usually large-scale) as the starting point for a different but related task, instead of training from scratch.
- **How it works:** Take a pretrained model's learned weights (which capture general features — e.g., edges/textures in vision, syntax/semantics in NLP), then adapt it to a new task via:
  - **Feature extraction:** Freeze the pretrained layers; only train a new task-specific head (e.g., classifier) on top.
  - **Fine-tuning:** Unfreeze some/all pretrained layers and continue training on the new dataset, usually with a smaller learning rate.
- **Why it works:** Lower layers of deep networks tend to learn generic, transferable features (e.g., edges, shapes in CNNs; grammar, word relationships in language models), while higher layers learn task-specific features — so the generic knowledge transfers well.
- **When to use it:**
  - **Limited labeled data** for the target task — training from scratch would overfit or underperform.
  - **Related domains** — the source task's features are relevant to the target task (e.g., ImageNet-pretrained CNN for a custom image classifier).
  - **Limited compute/time budget** — avoids costly training from scratch.
  - **Faster convergence needed** — pretrained weights give a better starting point than random initialization.
- **When NOT to use it / caution:**
  - Source and target domains are very different (e.g., medical images vs. natural images) — transfer may give little benefit or even hurt performance ("negative transfer").
  - If you have abundant labeled data and compute, training from scratch may outperform a poorly-matched pretrained model.
- **Common examples:** Using BERT/GPT pretrained on large text corpora and fine-tuning for sentiment analysis or Q&A; using ResNet/VGG pretrained on ImageNet and fine-tuning for a custom image classification task.

### 13.45 What are Multimodal AI models, and how do they process different types of data?

> **See also:** [Question 23.1 — What are Multimodal AI models, and how do they process different types of data?](../part-4-generative-ai/23-multimodal-ai.md#231-what-are-multimodal-ai-models-and-how-do-they-process-different-types-of-data), which covers the same question from another angle.

#### Multimodal AI Models

- **Core idea:** Models that can process and relate information across multiple data types (modalities) — e.g., text, images, audio, video — rather than being restricted to a single modality.
- **Why they matter:** Real-world understanding often requires combining modalities (e.g., understanding a video needs both visual frames and audio/speech), and multimodal models can perform tasks single-modality models can't, like image captioning, visual question answering, or text-to-image generation.

---

##### How They Process Different Data Types

- **Modality-specific encoders:** Each input type is first processed by an encoder suited to its structure:
  - Text → Transformer-based encoder (e.g., BERT-style)
  - Images → CNN or Vision Transformer (ViT)
  - Audio → Spectrogram-based CNN or audio transformer
  - Video → Combination of spatial (CNN/ViT) + temporal (RNN/Transformer) processing
- **Embedding into a shared representation space:** Each modality's encoder output is projected into a **common embedding space**, so different types of data become comparable (e.g., an image embedding and its text caption embedding end up close together in this space).
- **Cross-modal alignment/fusion:** Once in a shared space, the model learns relationships between modalities using techniques like:
  - **Contrastive learning** (e.g., CLIP) — trains the model so matching image-text pairs have similar embeddings, and non-matching pairs are pushed apart.
  - **Cross-attention mechanisms** — one modality's representation attends to another's (e.g., text tokens attending to image regions), allowing fine-grained interaction between modalities.
  - **Early fusion vs. late fusion:** Early fusion combines raw/low-level features from different modalities early in the network; late fusion processes each modality separately and combines high-level representations near the output.
- **Unified decoder/output generation:** Depending on the task, a decoder generates the final output — which could be text (captioning), an image (text-to-image diffusion models), or a classification label (e.g., matching score).

---

##### Examples of Multimodal Models

- **CLIP (Contrastive Language-Image Pretraining):** Learns joint text-image embeddings via contrastive learning; used for zero-shot image classification and retrieval.
- **DALL-E / Stable Diffusion:** Text-to-image generation — text embeddings condition a diffusion/generative image model.
- **GPT-4V / Gemini:** Large language models extended to accept and reason over images alongside text.
- **Whisper:** Primarily audio-to-text, but represents multimodal handling of speech and language.

---

##### Key Challenges

- **Modality alignment:** Ensuring representations from different modalities are meaningfully comparable in the shared space.
- **Data scarcity:** Large-scale paired multimodal datasets (e.g., image-text pairs) are harder to curate than single-modality data.
- **Computational cost:** Processing multiple modalities simultaneously is more resource-intensive than single-modality models.

### 13.46 How do World Models work?

#### World Models

- **Core idea:** Models that learn an internal representation of how an environment works — i.e., they learn to predict how the world (or a specific environment/task) evolves in response to actions, rather than just mapping inputs to outputs directly.
- **Key components:**
  - **Perception/Encoder:** Compresses raw high-dimensional observations (e.g., images, sensor data) into a compact latent representation of the current state.
  - **Dynamics/Transition Model:** Learns to predict the **next latent state** given the current state and an action taken — essentially learning the "physics" or rules of the environment.
  - **Reward/Outcome Predictor (optional):** In RL contexts, predicts the reward or outcome associated with a state-action pair.
  - **Decoder (optional):** Reconstructs the predicted latent state back into observable form (e.g., an image) for visualization or auxiliary training signals.
- **How they're trained:** Typically trained on sequences of (state, action, next state) data collected from an agent interacting with an environment, using self-supervised objectives like reconstruction loss or next-state prediction loss.
- **Why they're useful:**
  - **Planning/imagination:** Once a world model is learned, an agent can "imagine" or simulate future trajectories entirely within the latent space, without needing to interact with the real environment — enabling faster, cheaper planning.
  - **Sample efficiency:** Reduces the need for costly real-world interactions (e.g., in robotics) since the agent can practice/plan using its internal simulation.
  - **Model-based reinforcement learning:** The agent can use the world model to simulate rollouts and choose actions that maximize predicted future reward, rather than relying purely on trial-and-error (model-free RL).
- **Example architectures:**
  - **World Models (Ha & Schmidhuber, 2018):** Uses a VAE for perception, an RNN (MDN-RNN) for dynamics, and a simple controller trained on the compressed latent representations.
  - **Dreamer (and DreamerV2/V3):** Learns a latent world model and trains a policy purely by imagining trajectories inside the learned latent space, achieving strong performance in RL benchmarks with high sample efficiency.
  - **PlaNet:** A model-based RL approach that plans directly in latent space using a learned dynamics model.
- **Connection to broader AI trends:** Video generation/prediction models (e.g., predicting future video frames) and some large-scale generative models (e.g., Genie, Sora) can be viewed as forms of world models, learning to simulate plausible future states of an environment from observed data.

### 13.47 How do Diffusion Language Models (DLMs) work?

> **See also:** [Question 17.69 — How do Diffusion Language Models (DLMs) work?](../part-4-generative-ai/17-large-language-model-foundations.md#1769-how-do-diffusion-language-models-dlms-work), which covers the same question from another angle.

- **Core idea:** Apply the diffusion paradigm (originally used for images) to text generation — instead of generating tokens left-to-right like autoregressive LLMs, DLMs generate/refine an entire sequence in parallel through iterative denoising.
- **Forward process:** Text tokens are progressively corrupted, typically by:
  - **Discrete noising:** Randomly masking or replacing tokens with a special `[MASK]` token or random tokens over multiple steps, until the sequence becomes fully masked/random.
  - (Some approaches instead operate on continuous token embeddings and add Gaussian noise, similar to image diffusion.)
- **Reverse process:** A neural network (often a Transformer) is trained to predict the original clean tokens from a corrupted/masked sequence, gradually denoising it step-by-step until a coherent sequence emerges.
- **Generation/sampling:** Starting from a fully masked or random sequence, the model iteratively fills in/refines tokens across multiple steps, progressively converging on a fluent, coherent piece of text — rather than committing to one token at a time irrevocably.
- **Key differences from autoregressive LLMs:**
  - **Bidirectional refinement:** All positions can be updated at each denoising step, allowing global corrections (e.g., fixing an earlier word based on later context) — something AR models can't do once a token is generated.
  - **Parallel generation:** Multiple tokens can be generated/refined simultaneously per step, potentially reducing the number of sequential steps needed compared to token-by-token AR decoding.
  - **No fixed left-to-right order:** Generation order isn't constrained to be sequential, which can better suit tasks needing global consistency (e.g., infilling, editing).
- **Challenges:**
  - Text is **discrete**, unlike continuous pixel values in images, so applying continuous Gaussian noise directly is less natural — most DLMs use discrete corruption strategies (masking-based diffusion) instead.
  - Still generally less mature and less widely adopted than autoregressive LLMs (GPT-style), which dominate in quality and tooling for most NLP tasks currently.
  - Sampling can still require many iterative steps, so speed advantages over AR aren't always guaranteed in practice.
- **Examples/research directions:** Diffusion-LM, SEDD (Score Entropy Discrete Diffusion), and Mercury/other recent "diffusion-based" LLM research exploring parallel, non-autoregressive text generation with denoising objectives.

### 13.48 Explain deep reinforcement learning from human preferences.

- **Core idea:** Train an RL agent using **human preferences** instead of manually designed reward functions.
- **Human feedback:** Humans compare two short behavior segments and choose the better one.
- **Reward model:** A neural network learns to predict human preferences and acts as a **learned reward function**.
- **RL optimization:** The agent optimizes the learned reward using reinforcement learning.
- **Iterative process:** Agent behavior → human comparisons → reward model → policy improvement.
- **Importance:** Reduces the need for **hand-crafted reward functions**.
- **Foundation:** An important early foundation for **RLHF, reward modeling, and AI alignment**.
- **Limitation:** Human feedback can be **expensive, noisy, and inconsistent**, and agents may exploit weaknesses in the learned reward model.

### 13.49 Normalization in machine learning?

> **See also:** [Question 17.102 — What is normalization?](../part-4-generative-ai/17-large-language-model-foundations.md#17102-what-is-normalization), which covers the same question from another angle.

Normalization refers to techniques that rescale or reshape data (or model activations) so that different features or values are on a comparable scale. It matters because many ML algorithms are sensitive to the scale of inputs — without normalization, features with larger numeric ranges can dominate training, and optimization (like gradient descent) can converge slowly or unstably.

#### Why It Matters

- **Faster, more stable training** — gradient-based optimizers converge more smoothly when inputs are on similar scales
- **Fair feature contribution** — prevents features with large ranges (e.g., income in thousands) from overpowering smaller-range features (e.g., age in years)
- **Required by certain algorithms** — distance-based methods (KNN, SVM, K-means) and gradient descent-based methods (neural networks, logistic regression) are especially sensitive to scale
- **Numerical stability** — helps avoid overflow/underflow issues, especially in deep networks

#### Common Techniques for Input Data

**Min-Max Scaling (Normalization)**

Rescales values to a fixed range, usually \[0, 1\]:

$$
x' = \frac{x - \min(x)}{\max(x) - \min(x)}
$$

Sensitive to outliers since it depends on the extremes.

**Other common techniques for input data**

- **Standardization (Z-score):** $z = \frac{x - \mu}{\sigma}$, which gives mean 0 and standard deviation 1. It is the default for linear models, SVMs, PCA and neural networks, and is less affected by outliers than Min-Max.
- **Robust scaling:** $x' = \frac{x - \text{median}}{\text{IQR}}$, for data with heavy outliers.
- **Max-abs scaling:** divide by the maximum absolute value. It keeps sparsity, so it suits sparse matrices such as TF-IDF.
- **Unit-vector (L2) normalization:** scale each *sample* (row) to unit length. It is common for text vectors and embeddings before cosine similarity.
- **Log / power transforms (Box-Cox, Yeo-Johnson):** reshape skewed distributions; often combined with one of the scalers above.

**Normalization of activations inside neural networks**

- **Batch Normalization:** normalizes each feature over the mini-batch, then applies a learnable scale and shift ($\gamma$, $\beta$). It uses running statistics at inference and is the standard choice in CNNs.
- **Layer Normalization:** normalizes over the features of each individual sample, so it does not depend on batch size. It is standard in Transformers and RNNs; RMSNorm is a cheaper variant used in many LLMs.
- **Group / Instance Normalization:** normalize over groups of channels, or per channel per sample. They are used when batches are small (detection, segmentation) or for style transfer.

**Production gotcha:** fit scaler statistics on the training set only. Persist them with the model (for example, a scikit-learn `Pipeline`) and apply the same transform at inference. Recomputing them on live data causes training/serving skew.

### 13.50 What is the difference between normalization and standardization?

Both are **feature scaling techniques** used to put numerical features on comparable scales.

| Aspect | Normalization | Standardization |
| --- | --- | --- |
| **Definition** | Scales values to a fixed range | Rescales values based on mean and standard deviation |
| **Common range** | 0 to 1 | No fixed range |
| **Formula** | $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | $z = \frac{x - \mu}{\sigma}$ |
| **Mean** | Not necessarily 0 | 0 |
| **Standard deviation** | Not necessarily 1 | 1 |
| **Uses** | Minimum and maximum | Mean and standard deviation |
| **Outlier sensitivity** | Highly sensitive | Less sensitive than Min-Max |
| **Common applications** | KNN, K-Means, neural networks | Logistic Regression, SVM, PCA, linear models |
| changes | the range of the data | distribution's scale and centers it around zero. |

### 13.51 What is a neural network?

- A **Neural Network** is a machine learning model inspired by the way the human brain processes information.
- It consists of interconnected **neurons (nodes)** organized into layers:
  - **Input layer** → receives the input data.
  - **Hidden layers** → learn patterns and relationships from the data.
  - **Output layer** → produces the final prediction or result.
- During training, the network **adjusts its weights** to reduce prediction errors.
- Neural networks are widely used in **image recognition, natural language processing, speech recognition, recommendation systems, and generative AI**.

**In simple terms:** A neural network learns patterns from data and uses those patterns to make predictions.

### 13.52 What do you understand about the exploding gradient technique problem in the back propagation technique?

#### Exploding Gradient Problem

- **Exploding gradients** occur during **backpropagation** when the gradients become **extremely large** as they are propagated backward through the network.
- This is common in **very deep neural networks** and **Recurrent Neural Networks (RNNs)**.
- Large gradients cause the model's **weights to change by very large amounts** during training.
- This can make training **unstable**, causing the loss to become extremely large or even result in **NaN** values.
- Common causes include:
  - Very large weights
  - Deep networks
  - Repeated multiplication of large gradients, especially in RNNs
- Common solutions include:
  - **Gradient clipping:** Limits gradients to a maximum value.
  - **Proper weight initialization**
  - **Lower learning rate**
  - **Normalization techniques**
  - Using architectures such as **LSTM or GRU** for sequence problems.

### 13.53 What differences exist between Softmax and Sigmoid functions?

#### Softmax vs Sigmoid

- **Sigmoid:**
  - Produces an output between **0 and 1**.
  - Mainly used for **binary classification**.
  - Can also be used for **multi-label classification**, where multiple classes can be positive at the same time.
  - Each output is treated independently.
  - Example: **Cat = 0.8, Dog = 0.7** — both can be positive.
- **Softmax:**
  - Produces a probability for **each class**, with all probabilities adding up to **1**.
  - Mainly used for **multi-class classification**.
  - Classes are treated as mutually exclusive.
  - Example: **Cat = 0.7, Dog = 0.2, Horse = 0.1**.

| Feature | Sigmoid | Softmax |
| --- | --- | --- |
| Output | 0 to 1 | 0 to 1 |
| Sum of outputs | Not necessarily 1 | Always 1 |
| Typical use | Binary / Multi-label | Multi-class |
| Classes | Independent | Mutually exclusive |
| Example | Spam / Not Spam | Cat / Dog / Horse |

**Simple terms:**

**Sigmoid → independent probabilities.**

**Softmax → probabilities competing among multiple classes.**

### 13.54 How do you decide when to go for deep learning for a project?

Deep learning is worth it when the **problem, data and constraints** line up; otherwise gradient-boosted trees or linear models usually win on cost and maintainability.

**Go for deep learning when**
- **Unstructured data:** images, audio, text, video, graphs — where representation learning beats hand-made features.
- **Lots of (labeled or self-supervised) data**, or a pretrained foundation model you can fine-tune or prompt (transfer learning lowers the data requirement drastically).
- **Complex, non-linear patterns** with high-dimensional inputs (sequences, spatial structure).
- **Multimodal or generative tasks** (captioning, chat, summarization, image generation).
- You have **GPU budget** for training and serving, and latency requirements that allow it.

**Prefer classical ML when**
- Data is **tabular** and modest in size — XGBoost/LightGBM typically match or beat neural nets.
- **Interpretability** or regulatory explainability is a must (credit, insurance).
- Data is small, the deadline is tight, or the team lacks DL/MLOps experience.
- A simple baseline already meets the business target.

**Process I follow**
1. Define the metric and business target; build a quick **baseline** (heuristic, logistic regression, GBM).
2. Check whether a **pretrained model / API** solves it (embeddings + classifier, LLM prompting).
3. Move to custom DL only if the gap to the target justifies the extra training, serving and monitoring cost.

**Example:** churn prediction on CRM tables → GBM. Defect detection on factory images → fine-tune a pretrained CNN/ViT or YOLO.

### 13.55 Why do binary\_crossentropy and categorical\_crossentropy give different performances for the same problem?

Usually because they are being used for **different problem formulations**, or because of how Keras computes accuracy.

**1. The metric, not the model, differs (the classic Keras gotcha).** With `metrics=['accuracy']`, Keras picks the accuracy function from the loss. With `binary_crossentropy` on a multi-class softmax output, it uses **binary accuracy**, which thresholds each of the $K$ outputs independently at 0.5 and averages them. For 10 classes, predicting all zeros already scores ~90% "accuracy". The model isn't better; the reported number is misleading. Fix: specify `CategoricalAccuracy` explicitly.

**2. They optimize different things.**

- `categorical_crossentropy` + **softmax**: one label per sample; classes compete, probabilities sum to 1. $\text{Loss} = -\sum_k y_k \log p_k$.
- `binary_crossentropy` + **sigmoid**: each output is an independent yes/no problem (multi-label). Loss sums $K$ separate Bernoulli losses, also penalizing the probability mass on negative classes.

Using the wrong pairing gives different gradients and a different optimum, so training dynamics and final performance differ.

**Rule of thumb:**

| Task | Output | Loss |
| --- | --- | --- |
| Binary | 1 sigmoid | binary_crossentropy |
| Multi-class (one label) | K softmax | categorical / sparse_categorical_crossentropy |
| Multi-label | K sigmoid | binary_crossentropy |

For two classes, 1 sigmoid + BCE and 2 softmax + CCE are mathematically equivalent.

### 13.56 Compare RNN, LSTM, GRU and Transformers.

RNN: Processes sequences step by step, passing a hidden state forward. Simple, but suffers from vanishing/exploding gradients, so it forgets long-range context.

LSTM: An RNN with a separate cell state and three gates (input, forget, output) that control what to keep or discard. Handles long dependencies much better, but is heavier (more parameters) and still sequential.

GRU: A slimmed-down LSTM with two gates (update, reset) and no separate cell state. Fewer parameters, trains faster, and often performs about as well as LSTM on smaller datasets.

Transformer: Drops recurrence entirely. Self-attention lets every token look at every other token directly, and positional encodings supply order. Fully parallelizable, captures long-range dependencies easily, and scales extremely well, but attention cost grows quadratically with sequence length.

### 13.57 Compare world models and JEPA.

**World model:** an internal model that predicts how an environment evolves, $p(s_{t+1} \mid s_t, a_t)$, so an agent can *imagine* outcomes and plan before acting. Examples include Ha & Schmidhuber's World Models, Dreamer (RL in a learned latent space), MuZero (planning with a learned dynamics model), and video-generation "world simulators" such as Genie and Sora-style models. Many of these are **generative**: they reconstruct or predict pixels or observations.

**JEPA (Joint-Embedding Predictive Architecture, Yann LeCun):** a self-supervised architecture that predicts the **representation** of a missing or future part of the input from the representation of the visible part, *in embedding space*, never in pixel space. Implementations include I-JEPA (images) and V-JEPA / V-JEPA 2 (video, used for robot planning).

```text
context x --enc--> s_x --predictor(z)--> ŝ_y   ≈   s_y <--enc(EMA)-- target y
loss = || ŝ_y - s_y ||   (no pixel reconstruction)
```

**How they relate:** JEPA is a *way to build* a world model. LeCun's argument is that a world model should predict abstract states, not every pixel, because much of the future is unpredictable detail (the exact texture of leaves). Generative world models spend capacity on that detail. JEPA discards it and keeps what is predictable and useful for planning.

| | Generative world model | JEPA |
| --- | --- | --- |
| Predicts | Observations (pixels/tokens) or a reconstructable latent | Latent embeddings only |
| Strength | Can render and simulate; easy to inspect visually | Efficient, semantic features; less wasted capacity |
| Risk | Blurry or hallucinated details; expensive | Representation collapse (handled with EMA target encoders and masking); cannot render outputs directly |

**Interview angle:** both target the same goal, agents that understand and plan in the physical world. JEPA is a non-generative, representation-level approach to building that world model.

### 13.58 What is circuit/graph(operation like addition, multiplication, activation function are represented by nodes of the graph are called gates) search( process of systematically exploring nodes and edges of graph to determine the valid path from starting point to end point ) ? How does backpropagation solve circuit search?

backpropagation solves a computational circuit by reversing the flow of data to instantly calculate exactly how to fix the circuit's errors

### 13.59 What is a function approximator?

Neural networks

### 13.60 Compare Hopfield networks and Boltzmann machines.

All three are **energy-based** neural networks from the 1980s with symmetric weights. They differ in whether they are deterministic or stochastic, and in what they are used for.

**Hopfield network**

- A fully connected, recurrent network of binary ($\pm 1$) units with symmetric weights and no self-connections.
- Energy: $E = -\tfrac12 \sum_{i,j} w_{ij} s_i s_j - \sum_i b_i s_i$. Updates are **deterministic** and always lower the energy, so the network settles into a local minimum.
- Weights are set by a Hebbian rule ($w_{ij} \propto \sum_\mu \xi_i^\mu \xi_j^\mu$), so the stored patterns become attractors. Use: **associative / content-addressable memory**. Give it a noisy or partial pattern and it recalls the closest stored one.
- Limits: classical capacity is about $0.14N$ patterns, and it suffers from spurious attractors. *Modern Hopfield networks* have exponential capacity, and their update rule is mathematically equivalent to Transformer attention.

**Boltzmann machine (BM)**

- Same energy function, but units are **stochastic**: $p(s_i=1) = \sigma(\Delta E_i / T)$. It also has **hidden units** in addition to visible ones.
- It is a **generative model**: it learns a probability distribution $p(v) \propto \sum_h e^{-E(v,h)}$ rather than storing specific patterns.
- Learning needs expectations under the model distribution (sampled via MCMC/Gibbs), which is extremely slow for a fully connected graph. As a result it is mostly theoretical.

**Restricted Boltzmann machine (RBM)**

- A BM with a **bipartite** graph: there are visible↔hidden connections but none within a layer.
- That makes hidden units conditionally independent given the visible units (and vice versa), so block Gibbs sampling is cheap. Training uses **Contrastive Divergence (CD-k)** (Hinton).
- Uses: feature learning, collaborative filtering (the Netflix Prize), and stacked as **Deep Belief Networks** for layer-wise pretraining (around 2006), which helped spark the deep learning revival.

| | Hopfield | Boltzmann | RBM |
| --- | --- | --- | --- |
| Units | Deterministic | Stochastic | Stochastic |
| Hidden units | No | Yes | Yes (bipartite) |
| Purpose | Memory / pattern completion | Generative modeling | Practical generative modeling / features |
| Training | Hebbian, one-shot | Very slow MCMC | Contrastive Divergence |

**Today:** they are largely historical. VAEs, GANs and diffusion models replaced them for generative modeling, but the energy-based view and the Hopfield ↔ attention link still come up in interviews.

### 13.61 Compare content-addressable memory and parametric memory.

These are two ways a system can "remember" information.

**Content-addressable memory (CAM)**: you retrieve an item by its *content*, not by an address. A partial or noisy cue returns the best-matching stored item.

- Classic examples: Hopfield networks (pattern completion) and hardware CAM/TCAM in network routers.
- Modern examples: attention (query·key similarity selects values), vector databases, kNN-LM, memory networks, and **RAG**. You look up stored chunks by embedding similarity.
- Properties: knowledge is **explicit and inspectable**, easy to **update or delete** (add or remove a document), and attributable (you can cite the source). Costs: lookup latency, storage, and dependence on retrieval quality.

**Parametric memory**: knowledge is stored *implicitly in the model's weights* by training.

- Example: an LLM "knowing" that Paris is the capital of France.
- Properties: fast at inference (no lookup), generalizes and composes knowledge, and is compressed. But it is **opaque**, **hard to update** (needs fine-tuning or model editing, and risks catastrophic forgetting), frozen at the training cutoff, and prone to **hallucination** because there is no explicit source.

| | Content-addressable (non-parametric) | Parametric |
| --- | --- | --- |
| Where stored | External store / keys | Model weights |
| Update cost | Cheap (insert/delete) | Expensive (retrain/fine-tune) |
| Traceability | High (citations) | Low |
| Generalization | Limited to what is stored | Strong |
| Failure mode | Retrieval misses | Hallucination, staleness |

**Production takeaway:** modern LLM systems are **hybrid**. Parametric memory provides language skill and reasoning, and content-addressable memory (RAG, tool lookups, agent memory stores) provides fresh, private, auditable facts. That is why "prompting vs. RAG vs. fine-tuning" is mostly a question of which memory type should hold a given piece of knowledge.

### 13.62 Why do Transformers use layer normalization instead of batch normalization, and what is Pre-LN vs Post-LN?

**Why LayerNorm instead of BatchNorm**

- **BatchNorm** normalizes each feature across the *batch* (and spatial dims). In NLP, sequences have variable length and padding, so batch statistics get mixed with padding and are noisy. Statistics also depend on batch composition and size, which hurts at small per-device batches and with gradient accumulation.
- At **inference**, BN relies on running averages from training. With autoregressive decoding (one token at a time, batch size 1, changing distributions) those statistics mismatch badly.
- **LayerNorm** normalizes across the *feature dimension of each token independently*: $\operatorname{LN}(x) = \gamma \frac{x-\mu}{\sigma} + \beta$, with $\mu, \sigma$ computed over the hidden size. That makes it batch-size independent, identical at train and test time, and naturally handles variable lengths. Many modern LLMs use **RMSNorm**, which drops mean-centering and is cheaper with similar quality (LLaMA, Mistral).

**Post-LN vs Pre-LN**

```text
Post-LN (original Transformer):  x = LN(x + Sublayer(x))
Pre-LN  (GPT-2 onwards):         x = x + Sublayer(LN(x))
```

- **Post-LN:** the normalization sits on the residual path, so gradients near the output are large and those near the input shrink with depth. It needs **careful learning-rate warmup**, is unstable for deep stacks, but when it trains successfully it can reach slightly better final quality.
- **Pre-LN:** the residual stream is an identity path, so gradients flow cleanly to every layer. It is much more stable, trains deep models with little or no warmup, and is the default in almost all LLMs. One downside: the residual stream's magnitude grows with depth and later layers can contribute less, so a final LayerNorm is added before the LM head. Variants such as DeepNorm and sandwich-LN try to get the best of both.

**One-liner:** LN because NLP batches are variable and inference is per-token; Pre-LN because stability at depth matters more than the small quality edge of Post-LN.

### 13.63 How would you implement a regex matcher supporting `.` and `*`, and how do you debug a PyTorch training loss that is NaN or not decreasing?

This title merges two separate questions, so both are answered: (a) a regex matcher supporting `.` and `*`, and (b) debugging a PyTorch loss that is NaN or not decreasing.

**(a) Regex matcher with `.` (any char) and `*` (zero or more of the preceding element), full match**

Use DP: `dp[i][j]` means `s[i:]` matches `p[j:]`. Time and space are $O(|s|\cdot|p|)$.

```python
from functools import lru_cache

def is_match(s: str, p: str) -> bool:
    @lru_cache(maxsize=None)
    def dp(i: int, j: int) -> bool:
        if j == len(p):
            return i == len(s)
        first = i < len(s) and p[j] in (s[i], '.')
        if j + 1 < len(p) and p[j + 1] == '*':
            # skip "x*" entirely, or consume one char and stay on "x*"
            return dp(i, j + 2) or (first and dp(i + 1, j))
        return first and dp(i + 1, j + 1)
    return dp(0, 0)

assert is_match("aab", "c*a*b") and is_match("ab", ".*") and not is_match("mississippi", "mis*is*p*.")
```

Discuss: the recursive version without memoization is exponential, a bottom-up table avoids recursion limits, and real regex engines use NFA/Thompson construction for linear-time guarantees, which avoids catastrophic backtracking (ReDoS).

**(b) Loss is NaN or not decreasing: a debugging checklist**

1. **Overfit one small batch first.** If the loss can't go to about 0 on 1–10 examples, there is a bug (wrong labels, detached graph, wrong loss), not a tuning problem.
2. **Data:** check inputs and labels for NaN/Inf (`torch.isfinite(x).all()`), label ranges (class ids < num_classes), normalization, and that shuffling did not misalign x and y.
3. **Loss setup:** passing softmax outputs to `CrossEntropyLoss` (it expects logits); `log(0)` in a custom loss (add eps or use `log_softmax`/`BCEWithLogitsLoss`); division by zero in normalization.
4. **Training loop:** forgetting `optimizer.zero_grad()`, calling `model.eval()` by mistake, frozen params (`requires_grad=False`), params missing from the optimizer, or `.detach()`/`.item()` breaking the graph.
5. **Optimization:** a learning rate that is too high (NaN/divergence) or too low (flat loss). Run an LR range test, add warmup, use gradient clipping (`clip_grad_norm_(…, 1.0)`), and log the gradient norm per layer to spot explosion or vanishing.
6. **Numerics:** fp16 overflow, so use bf16 or `GradScaler`. Turn on `torch.autograd.set_detect_anomaly(True)` to find the op that first produces NaN, and check BatchNorm with batch size 1.
7. **Initialization and architecture:** bad init, missing normalization or residuals in deep nets, dead ReLUs.

Log loss, gradient norm, LR and activation stats (W&B/TensorBoard) so the failure step is visible, and fix seeds to reproduce it.

### 13.64 Explain backpropagation from first principles.

**Backpropagation is an efficient way to calculate how much each neural-network parameter contributed to the final error, so we can update those parameters to reduce the error.**

The core idea is simply the **chain rule of calculus**.

#### 1. Forward pass

Suppose a simple network is:

$$
x \rightarrow z \rightarrow \hat{y} \rightarrow L
$$

For one neuron:

$$
z = wx+b
$$

$$
\hat{y}=f(z)
$$

$$
L=L(\hat{y},y)
$$

The forward pass calculates the prediction and then the loss.

---

#### 2. What do we actually need?

To update $w$, gradient descent needs:

$$
\frac{\partial L}{\partial w}
$$

This answers:

> "If I slightly change $w$, how much does the loss change?"

But $w$ doesn't directly produce the loss. It affects $z$, which affects $\hat y$, which affects $L$.

So we use the chain rule:

$$
\boxed{ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat y} \frac{\partial \hat y}{\partial z} \frac{\partial z}{\partial w} }
$$

This is the essence of backpropagation.

---

#### 3. Example

Consider:

$$
z=wx+b
$$

$$
\hat y=z
$$

and squared-error loss:

$$
L=\frac{1}{2}(\hat y-y)^2
$$

We calculate derivatives backward.

First:

$$
\frac{\partial L}{\partial \hat y}=\hat y-y
$$

Then:

$$
\frac{\partial \hat y}{\partial z}=1
$$

And:

$$
\frac{\partial z}{\partial w}=x
$$

Therefore:

$$
\frac{\partial L}{\partial w} = (\hat y-y)(1)(x)
$$

$$
\boxed{ \frac{\partial L}{\partial w}=(\hat y-y)x }
$$

Similarly:

$$
\boxed{ \frac{\partial L}{\partial b}=\hat y-y }
$$

---

#### 4. Then gradient descent updates the parameters

Once backpropagation gives us the gradients:

$$
w \leftarrow w-\eta\frac{\partial L}{\partial w}
$$

$$
b \leftarrow b-\eta\frac{\partial L}{\partial b}
$$

where $\eta$ is the learning rate.

---

#### 5. Why "back" propagation?

The network computes values **forward**:

$$
x \rightarrow z_1 \rightarrow z_2 \rightarrow \hat y \rightarrow L
$$

But gradients are computed **backward**:

$$
L \rightarrow \hat y \rightarrow z_2 \rightarrow z_1 \rightarrow w
$$

Each layer receives a gradient from the layer after it and uses the chain rule to calculate its own gradient.

For a layer:

$$
z = Wx+b
$$

$$
a=f(z)
$$

the backward pass essentially determines:

$$
\frac{\partial L}{\partial W}, \quad \frac{\partial L}{\partial b}, \quad \frac{\partial L}{\partial x}
$$

The last one is important because it allows the gradient to continue propagating toward earlier layers.

---

#### 6. The key intuition

Think of the network as a sequence of functions:

$$
L=f_3(f_2(f_1(x)))
$$

Backpropagation asks:

> **How sensitive is the final loss to each intermediate value and parameter?**

The chain rule decomposes this sensitivity:

$$
\frac{dL}{dx} = \frac{dL}{df_2} \frac{df_2}{df_1} \frac{df_1}{dx}
$$

So **backpropagation is essentially repeated application of the chain rule, performed efficiently from the output layer back toward the input.**

### 13.65 Why do neural networks need nonlinear activation functions?

Neural networks need **nonlinear activation functions** because without them, stacking multiple layers would still produce only a **linear transformation**.

#### Without nonlinear activation

Suppose:

$$
h = W_1x+b_1
$$

$$
y = W_2h+b_2
$$

Substitute $h$:

$$
y=W_2(W_1x+b_1)+b_2
$$

$$
y=(W_2W_1)x+(W_2b_1+b_2)
$$

This is still just:

$$
y=Wx+b
$$

So even with 100 layers, a network containing only linear operations can be mathematically reduced to **one linear layer**.

#### With nonlinear activation

Add an activation such as ReLU:

$$
h=\operatorname{ReLU}(W_1x+b_1)
$$

$$
y=W_2h+b_2
$$

Now the network can represent **nonlinear functions and complex decision boundaries**.

For example, nonlinear activations allow networks to learn:

- XOR-like relationships
- Curved decision boundaries
- Image features
- Speech patterns
- Complex interactions between features

#### Intuition

Think of each layer as transforming the data.

**Linear only:**

$$
x \rightarrow \text{line} \rightarrow \text{line} \rightarrow \text{line}
$$

Still fundamentally a linear mapping.

**With nonlinearities:**

$$
x \rightarrow \text{transform} \rightarrow \text{bend} \rightarrow \text{transform} \rightarrow \text{bend}
$$

The network can progressively build complex functions.

#### Common nonlinear activations

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

$$
\operatorname{sigmoid}(x)=\frac{1}{1+e^{-x}}
$$

$$
\tanh(x)
$$

Modern deep networks commonly use **ReLU and its variants** such as GELU, SiLU, and Leaky ReLU.

### 13.66 Compare ReLU, Leaky ReLU, GELU, SiLU, and sigmoid.

#### ReLU vs Leaky ReLU vs GELU vs SiLU vs Sigmoid

| Activation | Formula | Output Range | Main Advantage | Main Limitation | Common Use |
| --- | --- | --- | --- | --- | --- |
| **ReLU** | $f(x)=\max(0,x)$ | $[0,\infty)$ | Simple and fast | **Dying ReLU** for negative inputs | CNNs, MLPs |
| **Leaky ReLU** | $f(x)=\max(\alpha x,x)$ | $(-\infty,\infty)$ | Keeps a small gradient for $x<0$ | Requires choosing $\alpha$ | CNNs, MLPs |
| **GELU** | $f(x)=x\Phi(x)$ | $(-\infty,\infty)$ | Smooth, probabilistic gating | More computationally expensive than ReLU | **Transformers, BERT** |
| **SiLU / Swish** | $f(x)=x\sigma(x)$ | $(-\infty,\infty)$ | Smooth and non-monotonic | More expensive than ReLU | **Modern deep learning, LLMs** |
| **Sigmoid** | $f(x)=\frac{1}{1+e^{-x}}$ | $(0,1)$ | Interpretable as probability | Vanishing gradients when saturated | Binary classification output |

#### Key differences

**1. ReLU**

- Negative values → 0.
- Positive values → unchanged.
- Very cheap computationally.
- Can cause **dying neurons** because the gradient is 0 for $x<0$.

**2. Leaky ReLU**

- Similar to ReLU, but allows a small negative output.
- Usually $\alpha$ is small, such as $0.01$.
- Reduces the dying-ReLU problem.

**3. GELU**

- Smoothly gates inputs instead of using a hard cutoff.
- Widely used in **Transformer architectures**, including BERT-style models.
- Approximation often used: $\operatorname{GELU}(x)\approx \frac{x}{2}\left(1+\tanh\left[ \sqrt{\frac{2}{\pi}} \left(x+0.044715x^3\right) \right]\right)$

**4. SiLU / Swish**

- Defined as: $\operatorname{SiLU}(x)=x\sigma(x)$
- Smooth and non-monotonic.
- Used extensively in modern neural networks.

**5. Sigmoid**

- Squashes values into $(0,1)$.
- Useful when the output represents a **binary probability**.
- Usually avoided as the main hidden-layer activation in deep networks because of **vanishing gradients**.

#### Interview takeaway

> **ReLU:** fast and simple
>
> **Leaky ReLU:** ReLU with a small negative gradient
>
> **GELU:** smooth gating, common in Transformers
>
> **SiLU:** smooth, non-monotonic activation used in modern architectures
>
> **Sigmoid:** mainly for binary-probability outputs

### 13.67 Why can sigmoid cause vanishing gradients?

#### Why can Sigmoid cause vanishing gradients?

The sigmoid function is:

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

Its derivative is:

$$
\sigma'(x)=\sigma(x)(1-\sigma(x))
$$

The **maximum derivative is only $0.25$**, occurring at $x=0$.

For very large positive or negative values, sigmoid **saturates**:

- $x \gg 0 \rightarrow \sigma(x)\approx 1 \rightarrow \sigma'(x)\approx 0$
- $x \ll 0 \rightarrow \sigma(x)\approx 0 \rightarrow \sigma'(x)\approx 0$

During backpropagation, gradients are multiplied through many layers:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \prod_i \frac{\partial y_i}{\partial x_i}
$$

Since sigmoid derivatives are often **much smaller than 1**, repeatedly multiplying them can make the gradient become extremely small.

**Example:**

If each layer contributes a gradient of $0.1$:

$$
0.1^{10}=10^{-10}
$$

So early layers receive almost no gradient and learn very slowly.

### 13.68 What is Batch Normalization, and why does it help?

#### What is Batch Normalization?

**Batch Normalization (BatchNorm)** normalizes a neural network's activations using the **mean and variance of the current mini-batch**, then applies learnable scale and shift parameters.

For a 2D activation matrix, imagine:

- **Rows** → training examples in the mini-batch
- **Columns** → features

BatchNorm typically normalizes **each feature across the mini-batch**.

##### How Does BatchNorm Work?

For each feature:

**1. Calculate batch statistics**

Compute the mean and variance across the mini-batch:

$$
\mu_B=\frac{1}{m}\sum_{i=1}^{m}x_i
$$

$$
\sigma_B^2=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2
$$

**2. Normalize**

Standardize the activation:

$$
\hat{x}_i = \frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

This gives the activation approximately **zero mean and unit variance** within the batch.

**3. Scale and shift**

BatchNorm then applies learnable parameters:

$$
y_i=\gamma\hat{x}_i+\beta
$$

where:

- $\gamma$ → learnable **scale**
- $\beta$ → learnable **shift**
- $\epsilon$ → small constant for numerical stability

The learnable $\gamma$ and $\beta$ allow the network to choose an appropriate distribution rather than being forced to use exactly zero mean and unit variance.

---

#### Why Does BatchNorm Help?

- **Stabilizes activations:** Keeps intermediate activations at a more controlled scale.
- **Improves gradient flow:** Can make optimization more stable, particularly in deep networks.
- **Enables higher learning rates:** Training can often use larger learning rates without becoming unstable.
- **Speeds up convergence:** Optimization often reaches a good solution faster.
- **Reduces sensitivity to initialization:** Training becomes less dependent on the exact initial scale of the weights.
- **Provides mild regularization:** Batch-to-batch variation in the estimated statistics introduces noise that can have a regularizing effect.

> **Important:** BatchNorm does not simply "prevent activations from exploding or shrinking to zero." Its main effect is to normalize activations and improve the conditioning/stability of optimization.

---

#### Training vs. Inference

During **training**, BatchNorm uses statistics calculated from the current mini-batch.

During **inference**, it normally uses **running estimates of the mean and variance** accumulated during training.

This allows inference to work even when processing a single example.

---

#### Why Do Transformers Usually Use LayerNorm Instead?

Transformers generally use **Layer Normalization (LayerNorm)** rather than BatchNorm because LayerNorm is independent of the mini-batch.

##### Key reasons:

- **Independent of batch size:** LayerNorm works even with a batch size of 1.
- **Works naturally with variable-length sequences:** Normalization does not depend on statistics across different examples in the batch.
- **Suitable for autoregressive generation:** Each token's representation can be normalized without requiring statistics from other sequences or examples.
- **Well suited to sequence models:** It normalizes across the features of each individual token representation rather than across the batch.

##### BatchNorm vs. LayerNorm

|  | **BatchNorm** | **LayerNorm** |
| --- | --- | --- |
| Normalization basis | Across mini-batch | Across features |
| Depends on batch size | Yes | No |
| Training statistics | Current batch | Individual sample |
| Inference | Running statistics | Same normalization principle |
| Commonly used in | CNNs, vision | Transformers, NLP |
| Works well with batch size 1 | Not ideal | Yes |

### 13.69 Compare BatchNorm, LayerNorm, GroupNorm, and RMSNorm.

##### BatchNorm vs LayerNorm vs GroupNorm vs RMSNorm

All four normalization methods help control the scale of neural-network activations, but they differ in **which dimensions they normalize over**.

| Feature | **BatchNorm** | **LayerNorm** | **GroupNorm** | **RMSNorm** |
| --- | --- | --- | --- | --- |
| Normalizes over | Batch dimension | Feature dimension | Groups of channels | Feature dimension |
| Depends on batch size | **Yes** | No | No | No |
| Uses mean? | Yes | Yes | Yes | **No** |
| Uses variance? | Yes | Yes | Yes | RMS only |
| Learnable scale | Yes | Yes | Yes | Yes |
| Learnable shift | Yes | Yes | Yes | Usually **No** |
| Common use | CNNs | Transformers | CNNs / vision | Modern LLMs |
| Main advantage | Efficient training | Batch-independent | Good for small batches | Simpler/faster than LayerNorm |

---

#### 1. Batch Normalization

For each feature/channel, BatchNorm computes statistics across the **mini-batch**:

$$
\hat{x}= \frac{x-\mu_B} {\sqrt{\sigma_B^2+\epsilon}}
$$

Then:

$$
y=\gamma\hat{x}+\beta
$$

**Key point:** BatchNorm depends on batch statistics.

**Best known for:** CNNs and computer vision.

**Limitation:** Performance can be affected by very small batch sizes.

---

#### 2. Layer Normalization

LayerNorm normalizes the features **within each individual sample**.

$$
\mu=\frac{1}{d}\sum_{i=1}^{d}x_i
$$

$$
\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$

$$
\operatorname{LayerNorm}(x) = \gamma \frac{x-\mu} {\sqrt{\sigma^2+\epsilon}} +\beta
$$

**Key point:** It does **not depend on other samples in the batch**.

**Best known for:** Transformers and sequence models.

---

#### 3. Group Normalization

GroupNorm divides channels into **groups** and computes normalization statistics within each group.

For example, with 32 channels, you might divide them into 8 groups of 4 channels.

Conceptually:

$$
\text{Channels} \rightarrow \text{Groups} \rightarrow \text{Normalize each group}
$$

**Key point:** It provides a middle ground between BatchNorm and LayerNorm.

It does not depend on the batch size, making it useful when CNNs are trained with **small batches**.

**Best known for:** Computer vision, especially object detection and segmentation.

---

#### 4. RMSNorm

RMSNorm simplifies LayerNorm by removing the **mean-centering** step.

Instead of calculating:

$$
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

RMSNorm uses the root mean square:

$$
\operatorname{RMS}(x) = \sqrt{ \frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon }
$$

Then:

$$
\operatorname{RMSNorm}(x) = \gamma \frac{x} {\operatorname{RMS}(x)}
$$

So RMSNorm **rescales** the activations but does not subtract their mean.

**Key point:** It is computationally simpler than LayerNorm.

**Best known for:** Modern Transformer/LLM architectures.

---

#### The easiest way to remember them

> **BatchNorm:** Normalize across the **batch**.
>
> **LayerNorm:** Normalize across the **features** of each sample.
>
> **GroupNorm:** Normalize across **groups of channels**.
>
> **RMSNorm:** Normalize by **RMS without subtracting the mean**.

### 13.70 Why is LayerNorm preferred over BatchNorm in Transformers?

**LayerNorm is preferred because it normalizes each token independently across its feature dimensions, while BatchNorm depends on statistics computed across the mini-batch.** This makes LayerNorm much better suited to Transformer architectures.

#### Key reasons

1. **Independent of batch size**
   - BatchNorm depends on the other samples in the batch.
   - LayerNorm works even with **batch size = 1**.
   - This is useful for training and inference with varying batch sizes.
2. **Works naturally with variable-length sequences**
   - Transformer inputs can have different sequence lengths.
   - BatchNorm's batch-level statistics can be awkward when sequences are padded or masked.
   - LayerNorm normalizes each token's representation independently.
3. **Suitable for autoregressive generation**
   - During LLM inference, tokens may be generated **one at a time**.
   - LayerNorm does not require statistics from other examples or tokens in the batch.
4. **Consistent between training and inference**
   - BatchNorm uses batch statistics during training but running statistics during inference.
   - LayerNorm uses the same normalization principle during both training and inference.
5. **Works well with Transformer representations**
   - A token is represented as a vector of hidden features: $x \in \mathbb{R}^{d_{\text{model}}}$
   - LayerNorm normalizes those $d_{\text{model}}$ features for each token independently.

#### BatchNorm vs LayerNorm

|  | **BatchNorm** | **LayerNorm** |
| --- | --- | --- |
| Normalizes | Across batch | Across features |
| Depends on batch | **Yes** | **No** |
| Batch size = 1 | Problematic | Works |
| Variable sequences | Less natural | Well suited |
| Autoregressive LLM inference | Less suitable | Well suited |
| Training/inference behavior | Different statistics | Same principle |
| Typical use | CNNs | Transformers |

### 13.71 What is residual learning, and why are skip connections so effective?

#### What is Residual Learning?

**Residual learning** is the idea of having a neural network learn the **difference (residual)** between the desired output and the input, rather than learning the complete transformation directly.

Instead of learning:

$$
H(x)
$$

a residual block learns:

$$
F(x)=H(x)-x
$$

and produces:

$$
y=F(x)+x
$$

This is the core idea behind **ResNet**.

#### What is a Skip Connection?

A **skip connection** provides a shortcut that carries the input directly to a later layer:

$$
y=x+F(x)
$$

The input $x$ effectively **skips** one or more transformations and is added to their output.

#### Why Are Skip Connections So Effective?

**1. Improve gradient flow**

During backpropagation:

$$
\frac{\partial y}{\partial x} = 1+\frac{\partial F(x)}{\partial x}
$$

The **$1$** provides a direct gradient path, helping gradients propagate through very deep networks.

**2. Reduce the vanishing-gradient problem**

Without a shortcut, gradients must pass through many transformations. With a residual connection, part of the gradient can flow directly through the identity path.

**3. Make deep networks easier to optimize**

Learning:

$$
H(x)=x+F(x)
$$

can be easier than learning $H(x)$ directly, especially when the desired transformation is close to the identity.

If the optimal transformation is approximately:

$$
H(x)\approx x
$$

the network only needs to learn:

$$
F(x)\approx0
$$

rather than reconstructing the identity transformation from scratch.

**4. Preserve information**

The original representation can bypass transformations that might otherwise distort or lose useful information.

**5. Enable very deep architectures**

Skip connections are a fundamental component of architectures such as **ResNet** and **Transformers**.

#### In Transformers

A simplified Transformer sublayer looks like:

$$
x' = x + F(x)
$$

where $F(x)$ could be:

- Multi-Head Self-Attention
- Feed-Forward Network (FFN)

Modern Transformers commonly combine residual connections with normalization, for example:

$$
x' = x + \operatorname{Attention}(\operatorname{LayerNorm}(x))
$$

### 13.72 Why do deeper networks sometimes perform worse even when optimization should theoretically be easier?

#### Why can deeper networks perform worse?

A deeper network has **greater representational capacity**, so in theory it should be able to represent everything a shallower network can. However, **representational capacity is not the same as optimization being easy**.

This is known as the **degradation problem**.

#### 1. Optimization becomes harder

Adding layers introduces more parameters and nonlinear transformations. Even if a deeper network could theoretically reproduce the shallower network, gradient-based optimization may have difficulty finding those parameters.

For example, suppose the added layers should simply perform an identity mapping:

$$
H(x)=x
$$

A plain deep network may struggle to learn this identity transformation.

#### 2. Vanishing or exploding gradients

During backpropagation, gradients are repeatedly multiplied through layers:

$$
\frac{\partial L}{\partial x} = \prod_i \frac{\partial h_i}{\partial h_{i-1}} \frac{\partial L}{\partial h_n}
$$

If these derivatives are consistently smaller than 1, gradients can **vanish**.

If they are consistently larger than 1, gradients can **explode**.

Normalization, careful initialization, and appropriate activations help, but very deep plain networks can still be difficult to optimize.

#### 3. Error accumulation

Each additional layer applies another transformation. Small optimization errors can accumulate through many layers, making it harder for the network to preserve useful representations.

#### 4. Overfitting can also occur

A deeper network has more parameters and therefore can have higher capacity. With insufficient data or inadequate regularization, it may fit training data too closely and perform worse on unseen data.

However, **overfitting is not the same as the degradation problem**.

The degradation problem can occur when the **training error itself becomes worse** as depth increases. That indicates an optimization issue rather than simply poor generalization.

#### Why do residual connections help?

A residual block learns:

$$
y=x+F(x)
$$

If additional layers are not useful, they can learn:

$$
F(x)\approx0
$$

so that:

$$
y\approx x
$$

This makes it much easier for a deeper network to preserve the behavior of a shallower network.

The gradient also gets a direct path:

$$
\frac{\partial y}{\partial x} = 1+\frac{\partial F(x)}{\partial x}
$$

The identity term provides a strong route for gradient propagation.

### 13.73 Explain dropout mathematically.

#### Dropout — Mathematical Explanation

**Dropout** is a regularization technique that randomly deactivates a fraction of neurons during training to reduce overfitting.

Suppose a layer produces:

$$
h = f(Wx+b)
$$

During training, we generate a random **Bernoulli mask**:

$$
m_i \sim \text{Bernoulli}(1-p)
$$

where:

- $p$ = dropout probability
- $1-p$ = probability that a neuron is kept
- $m_i \in \{0,1\}$

The mask is applied to the activations.

#### Inverted Dropout

Modern implementations typically use **inverted dropout**:

$$
\tilde{h}_i = \frac{m_i}{1-p}h_i
$$

or in vector form:

$$
\boxed{ \tilde{h} = \frac{m\odot h}{1-p} }
$$

where $\odot$ represents element-wise multiplication.

The scaling factor $\frac{1}{1-p}$ keeps the **expected activation approximately unchanged**:

$$
\mathbb{E}[\tilde{h}_i] = \mathbb{E}\left[\frac{m_i h_i}{1-p}\right]
$$

Since:

$$
\mathbb{E}[m_i]=1-p
$$

we get:

$$
\mathbb{E}[\tilde{h}_i] = \frac{(1-p)h_i}{1-p} = h_i
$$

#### Example

If:

$$
p=0.5
$$

then each neuron has a 50% probability of being dropped.

Suppose:

$$
h=[2,4,6,8]
$$

and the sampled mask is:

$$
m=[1,0,1,0]
$$

Then:

$$
\tilde{h} = \frac{[1,0,1,0]\odot[2,4,6,8]}{0.5}
$$

$$
\tilde{h}=[4,0,12,0]
$$

So some neurons are temporarily deactivated while the remaining activations are scaled up.

#### Training vs. Inference

**During training:**

$$
\tilde{h}=\frac{m\odot h}{1-p}
$$

A new random mask is typically sampled for each training pass.

**During inference:**

$$
\tilde{h}=h
$$

No neurons are randomly dropped because the scaling was already handled during training.

#### Why Does Dropout Reduce Overfitting?

Dropout prevents the network from relying too heavily on particular neurons or specific combinations of features.

You can think of training as approximately optimizing many different **thinned subnetworks** that share parameters.

This encourages the network to learn more robust representations.

### 13.74 Is dropout still useful in modern large Transformers?

Yes, but **its role has changed in modern large Transformers**.

#### Dropout in modern Transformers

Dropout is still a valid regularization technique, but many **large-scale LLMs use little or no dropout during pretraining**.

| Situation | Dropout usefulness |
| --- | --- |
| Small Transformer | **Often useful** |
| Limited training data | **Often useful** |
| Fine-tuning | **Can be useful** |
| Large-scale LLM pretraining | **Often low or zero** |
| Overfitting concern | **Can help** |
| Very large datasets | Often less necessary |

#### Why is dropout often reduced or removed?

**1. Massive datasets already provide regularization**

Large language models are trained on enormous datasets. The diversity of training examples can reduce overfitting, so aggressive dropout may not be necessary.

**2. It introduces noise during training**

Dropout randomly removes activations:

$$
\tilde{h}=\frac{m\odot h}{1-p}
$$

For very large models, this additional noise can sometimes make optimization less efficient.

**3. Other regularization techniques are important**

Modern LLM training commonly relies on techniques such as:

- Large and diverse datasets
- Weight decay
- Careful initialization
- Learning-rate schedules
- Data filtering
- Early stopping or training-duration control
- Parameter-efficient fine-tuning methods

#### What about fine-tuning?

Dropout can become more useful when fine-tuning a model on a **small dataset**, because overfitting becomes a bigger concern.

However, whether to use it depends on the model, dataset size, training objective, and existing regularization.

#### Important distinction

**Dropout is not what makes Transformers generalize well by itself.**

For large LLMs, the combination of **scale + data + optimization + regularization choices** matters much more than simply adding dropout.

### 13.75 What is knowledge distillation?

#### What is Knowledge Distillation?

**Knowledge distillation (KD)** is a model-compression technique where a smaller **student model** learns to reproduce the behavior of a larger, more capable **teacher model**.

Instead of learning only from the hard labels, the student learns from the teacher's **soft predictions**, which contain information about how the teacher distributes probability across different classes.

#### Basic idea

$$
\text{Teacher model} \rightarrow \text{Soft targets} \rightarrow \text{Student model}
$$

For example, suppose the correct class is **cat**.

A hard label might be:

$$
[0,1,0]
$$

The teacher might produce:

$$
[0.05,0.90,0.05]
$$

The student learns from this richer probability distribution.

#### Temperature

Knowledge distillation commonly uses a **temperature $T$** to soften the probability distribution:

$$
p_i^{(T)} = \frac{\exp(z_i/T)} {\sum_j\exp(z_j/T)}
$$

where:

- $z_i$ = model logits
- $T$ = temperature
- Higher $T$ → softer probability distribution

At $T=1$, this is the normal softmax.

#### Distillation Loss

A common objective combines the normal supervised loss with the teacher's soft predictions:

$$
\mathcal{L} = \alpha\mathcal{L}_{\text{CE}} + (1-\alpha)\mathcal{L}_{\text{KD}}
$$

where:

$$
\mathcal{L}_{\text{KD}} = T^2 D_{\text{KL}} \left( p_{\text{teacher}}^{(T)} \parallel p_{\text{student}}^{(T)} \right)
$$

So the student learns from both:

1. **Ground-truth labels**
2. **Teacher's behavior**

#### Why is it useful?

- **Model compression:** Smaller models can approximate larger models.
- **Lower inference cost:** Fewer parameters and FLOPs.
- **Lower latency:** Useful for edge/mobile deployment.
- **Knowledge transfer:** The student can learn useful patterns from the teacher's outputs.
- **Deploying large-model capabilities:** A smaller model can inherit some behavior of a larger model.

#### In LLMs

For language models, distillation can involve more than simply matching class probabilities. A student may learn from:

- Teacher-generated responses
- Token-level probability distributions/logits
- Intermediate representations
- Reasoning traces or demonstrations
- Preference or instruction-following behavior

For example:

$$
\text{Large LLM} \rightarrow \text{generated training data} \rightarrow \text{smaller LLM}
$$

This is one approach used to transfer capabilities from a larger model to a smaller one.

### 13.76 Explain teacher-student training.

#### Teacher–Student Training

**Teacher–student training** is a framework where a **teacher model provides guidance or supervision to a student model**.

The teacher is usually larger, more accurate, or already trained, while the student is smaller or designed to be more efficient.

$$
\text{Teacher} \rightarrow \text{Knowledge/Targets} \rightarrow \text{Student}
$$

#### How it works

**1. Train or obtain the teacher**

The teacher is first trained or selected as a strong model.

$$
f_T(x)\rightarrow z_T
$$

where $z_T$ represents the teacher's output.

**2. Generate teacher targets**

The teacher processes training examples and produces predictions:

$$
p_T=\operatorname{softmax}\left(\frac{z_T}{T}\right)
$$

The temperature $T$ can soften the distribution and expose more information about the teacher's relative preferences.

**3. Train the student**

The student produces:

$$
p_S=\operatorname{softmax}\left(\frac{z_S}{T}\right)
$$

The student is trained to match the teacher:

$$
\mathcal{L}_{\text{KD}} = T^2 D_{\text{KL}}(p_T\parallel p_S)
$$

Often, ground-truth labels are also included:

$$
\mathcal{L} = \alpha\mathcal{L}_{\text{CE}} + (1-\alpha)\mathcal{L}_{\text{KD}}
$$

#### Why does this work?

The teacher provides **soft information** that ordinary labels don't contain.

For example, instead of only:

$$
\text{label}=\text{cat}
$$

the teacher might indicate:

$$
P(\text{cat})=0.90,\quad P(\text{dog})=0.08,\quad P(\text{car})=0.02
$$

The student learns not only the correct answer but also the teacher's **similarity structure and decision behavior**.

#### Teacher–Student in LLMs

For LLMs, teacher–student training can work at several levels:

| Knowledge transferred | Example |
| --- | --- |
| **Logits** | Student matches teacher token probabilities |
| **Generated text** | Student learns from teacher-generated responses |
| **Hidden representations** | Student matches intermediate representations |
| **Reasoning behavior** | Student learns from teacher-generated reasoning examples |
| **Instruction following** | Student trains on teacher-generated instruction/response pairs |

A common LLM workflow is:

$$
\text{Large Teacher LLM} \rightarrow \text{Generate high-quality examples} \rightarrow \text{Train Student LLM}
$$

#### Teacher–Student vs Knowledge Distillation

**Teacher–student training** is the broader framework.

**Knowledge distillation** is a specific technique within that framework that explicitly transfers knowledge from teacher to student, commonly through **soft targets, logits, or representations**.

### 13.77 What is early stopping actually doing from an optimization perspective?

**Early stopping** stops gradient-based training before the model reaches the minimum training loss, usually when validation performance stops improving.

From an optimization perspective, it acts as a form of **implicit regularization**.

#### 1. Without early stopping

Suppose we optimize:

$$
\theta_{t+1} = \theta_t-\eta\nabla_\theta L_{\text{train}}(\theta_t)
$$

As training continues:

$$
L_{\text{train}}\downarrow
$$

But after some point, validation performance may stop improving:

$$
L_{\text{val}}\uparrow
$$

The model starts fitting increasingly specific patterns in the training data.

#### 2. Early stopping

We monitor validation performance and stop when it no longer improves:

$$
t^{\ast} = \arg\min_t L_{\text{val}}(\theta_t)
$$

Then we use: $\theta_{t^{\ast}}$

instead of the parameters obtained after fully optimizing the training loss.

#### Why is this regularization?

The model has not had unlimited optimization time to fit the training data.

Early stopping effectively **restricts the set of parameter values the optimizer can reach from the initialization**.

So instead of: $\theta_0 \rightarrow \theta_1 \rightarrow \theta_2 \rightarrow \cdots \rightarrow \theta_{\text{train optimum}}$

we select an earlier point: $\theta_0 \rightarrow \cdots \rightarrow \boxed{\theta_{t^{\ast}}}$

This can prevent the model from fitting increasingly complex or noisy patterns.

#### Connection to Gradient Descent

For some simple models, the connection can be made mathematically explicit.

Consider least-squares regression: $L(\theta) = \frac{1}{2}\lVert X\theta-y\rVert^2$

Gradient descent gives:

$$
\theta_{t+1} = \theta_t-\eta X^T(X\theta_t-y)
$$

Starting from $\theta_0=0$, the solution after a finite number of iterations is biased toward certain lower-complexity solutions.

Thus, **finite optimization time itself can act like regularization**.

This is sometimes called **implicit regularization** or **iterative regularization**.

#### Early Stopping vs Explicit Regularization

| Technique | Mechanism |
| --- | --- |
| **Early stopping** | Stop optimization early |
| **Weight decay / L2** | Penalize large weights |
| **Dropout** | Add stochastic noise / remove activations |
| **Data augmentation** | Increase effective training diversity |

Early stopping does **not** explicitly add a term such as:

$$
L_{\text{total}} = L_{\text{train}} + \lambda\lVert\theta\rVert^2
$$

Instead, the **optimization trajectory and stopping time** provide the regularizing effect.

#### Important distinction

Early stopping is **not simply "stopping when training loss gets bad."**

Usually:

- Training loss → continues decreasing
- Validation loss → decreases initially
- Validation loss → eventually stops improving or increases
- Training stops → parameters from the best validation point are retained

### 13.78 How would you debug a neural network whose loss suddenly becomes NaN?

Treat it as a bisection problem: find the **first step** and the **first tensor** that goes non-finite.

1. **Reproduce deterministically** — fix seeds, save the batch and checkpoint just before the blow-up, and replay that step.
2. **Turn on anomaly detection** — `torch.autograd.set_detect_anomaly(True)` points at the backward op that produced NaN; add forward hooks that assert `torch.isfinite` on activations.
3. **Check the data** — NaN/Inf in inputs or labels, division by zero in preprocessing, labels outside the class range, empty batches.
4. **Check numerically unsafe ops** — `log(0)`, `sqrt` of negative, `exp` overflow, softmax on huge logits, division by a tiny std in a custom norm. Use fused stable versions (`log_softmax`, `BCEWithLogitsLoss`, `cross_entropy`) and add `eps`.
5. **Check optimisation** — learning rate too high (loss spikes before NaN), missing warmup, exploding gradients. Log the global gradient norm; add gradient clipping and lower the LR.
6. **Mixed precision** — FP16 overflow is a classic cause. Use a dynamic `GradScaler`, keep loss/softmax/norms in FP32, or switch to BF16.
7. **Optimizer state** — Adam with a tiny `eps` in FP16, or a corrupted checkpoint, can produce Inf in the second moment.

```python
for name, p in model.named_parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all():
        print("non-finite grad in", name)
```

**Production habit:** log loss, grad-norm and LR every step, and have the training loop skip and alert on non-finite losses rather than silently corrupting weights.

### 13.79 What are Graph Neural Networks (GNNs)?

A **Graph Neural Network (GNN)** is a neural network designed to learn from graph-structured data.

A graph consists of:

```txt
Nodes + Edges + Features
```

Examples:

- Social networks
- Recommendation systems
- Fraud networks
- Knowledge graphs
- Molecular structures
- Transportation networks

#### Core idea

GNNs use **message passing**.

Each node aggregates information from its neighboring nodes:

```txt
Neighbor features
       ↓
Aggregation
       ↓
Node representation
       ↓
Updated node embedding
```

A simplified formulation is:

$$
h_v^{(l+1)} = \operatorname{UPDATE} \left( h_v^{(l)}, \operatorname{AGGREGATE}\left(\{h_u^{(l)}:u\in \mathcal{N}(v)\}\right) \right)
$$

#### Common GNN architectures

- **GCN** — Graph Convolutional Network
- **GAT** — Graph Attention Network
- **GraphSAGE**
- **GIN** — Graph Isomorphism Network

#### Common tasks

**Node classification:** Predict a label for a node.

**Link prediction:** Predict whether an edge should exist.

**Graph classification:** Classify an entire graph.

GNNs are especially useful when **relationships between entities contain predictive information**, which conventional tabular models may fail to capture.

### 13.80 What is geometric deep learning, and how do symmetry, invariance and equivariance unify CNNs, GNNs and Transformers?

**Geometric deep learning (GDL)** is the framework (Bronstein, Bruna, Cohen and Veličković, 2021) that explains most successful architectures as one recipe: pick the **domain** your data lives on, identify its **symmetry group**, and build layers that respect that symmetry. CNNs, GNNs and Transformers are the same blueprint applied to different domains.

**Key definitions**

For a transformation $g$ from a group $G$ acting on the input:

- **Invariance:** the output does not change, $f(g \cdot x) = f(x)$. A cat classifier should say "cat" wherever the cat is.
- **Equivariance:** the output changes in the same way as the input, $f(g \cdot x) = g \cdot f(x)$. Shift the image and the feature map shifts by the same amount; permute the nodes and the node embeddings are permuted the same way.

Equivariance keeps information in intermediate layers. Invariance is applied once, at the end, when you need a single answer.

**The blueprint**

1. Stack **local equivariant layers** (convolution, message passing, attention), each followed by a pointwise nonlinearity.
2. Optionally add **local pooling / coarsening** to build a multi-scale hierarchy (scale separation).
3. Finish with a **global invariant pooling** (global average, sum or mean readout) when the task needs an invariant output.

The other two principles are **geometric stability** (small deformations of the input should cause small changes in the output) and **scale separation** (learn local interactions, then compose them). Together they explain why these models beat the curse of dimensionality: the symmetry prior cuts the hypothesis space massively.

**The "5 Gs" and the architectures they produce**

| Domain | Symmetry group | Architecture |
| --- | --- | --- |
| Grids (images, audio) | Translation | CNN |
| Groups / homogeneous spaces (sphere, 3D) | Rotations, roto-translations | Group-equivariant CNNs, spherical CNNs |
| Graphs and sets | Permutation of nodes | GNNs, DeepSets |
| Geodesics / manifolds (meshes) | Isometries | Geodesic and mesh CNNs |
| Gauges (local frames on manifolds) | Gauge transformations | Gauge-equivariant CNNs |

**Where Transformers fit**

Self-attention without positional encodings is **permutation-equivariant**: it is a GNN over a **fully connected graph** in which the attention weights are learned edge weights. Positional encodings deliberately break that symmetry to inject sequence order. In this view a Transformer is a GNN with a complete graph plus a position prior; a GAT is attention restricted to the given edges.

**Why it matters in practice**

- **Sample efficiency:** baking in the right symmetry means the model does not have to learn it from data (no need to augment every rotation of a molecule).
- **Architecture choice becomes a modelling question:** "What transformations should not change my answer?" For tabular rows, the answer may be none, which is why gradient boosting often wins there.
- **Failure mode:** the wrong prior hurts. Full rotation invariance makes "6" and "9" indistinguishable; translation equivariance is wrong where absolute position matters (e.g. medical images with a fixed anatomical frame). Large data can sometimes substitute for priors, which is why ViTs with weak priors match CNNs only at scale.

**Interview one-liner:** GDL says architectures are defined by the symmetries they respect: CNNs are translation-equivariant, GNNs and Transformers are permutation-equivariant, and the final readout makes the prediction invariant.

### 13.81 What are equivariant neural networks, and where are they used in practice?

An **equivariant neural network** is built so that transforming the input by a symmetry (rotation, translation, reflection, permutation) transforms the output in a predictable, matching way, guaranteed by construction rather than learned from augmentation. In applied ML the term usually means networks equivariant to **3D Euclidean symmetries**: the groups $E(3)$ (rotations, translations, reflections) or $SE(3)$ (no reflections).

**Why not just augment the data?**

- Augmentation only gives *approximate* symmetry and only for the rotations you sampled; equivariance is exact for all of them.
- It wastes capacity and data learning something you already know.
- For physical quantities it is essential: if you rotate a molecule, predicted **forces** must rotate with it (equivariant) while predicted **energy** must not change (invariant). A model that violates this produces unphysical simulations.

**Main design families**

- **Invariant-feature models** (SchNet, DimeNet): feed only distances and angles, which are already invariant. Simple, but can lose directional information.
- **Scalar-vector models** like **E(n)-equivariant GNNs (EGNN, Satorras et al. 2021)**: each node carries features $h_i$ and coordinates $x_i$. Messages use only the invariant squared distance, and coordinates are updated along relative position vectors:

$$
m_{ij} = \phi_e\left(h_i, h_j, \lVert x_i - x_j \rVert^2, a_{ij}\right)
$$

$$
x_i' = x_i + C \sum_{j \neq i} (x_i - x_j)\, \phi_x(m_{ij}), \qquad h_i' = \phi_h\left(h_i, \sum_{j} m_{ij}\right)
$$

  Because $x_i - x_j$ rotates with the input and the weights $\phi_x(m_{ij})$ are invariant, the update is equivariant without any spherical harmonics. It is cheap and works in any dimension $n$.
- **Tensor-field / irreducible-representation models** (Tensor Field Networks, SE(3)-Transformer, NequIP, MACE, the e3nn library): features are typed by their rotation order (scalars, vectors, higher-order tensors) and combined with Clebsch–Gordan tensor products. More expressive and very data-efficient, but heavier compute.
- **Group-equivariant CNNs** for images (rotation/reflection-equivariant filters) in microscopy, satellite and medical imaging.

**Where they are used**

- **Molecular property prediction and machine-learned interatomic potentials:** NequIP and MACE reach DFT-level accuracy with orders of magnitude less training data, enabling molecular dynamics at scale.
- **Protein structure and design:** AlphaFold 2's structure module uses an SE(3)-equivariant invariant point attention; diffusion models such as RFdiffusion generate backbones equivariantly.
- **Drug discovery:** docking, conformer generation, equivariant diffusion for 3D molecule generation.
- **Physics and simulation:** particle systems, fluid and N-body dynamics, crystal/materials discovery.
- **Point clouds and robotics:** LiDAR perception, grasping and pose estimation where the object's orientation is arbitrary.

**Trade-offs**

| Gain | Cost |
| --- | --- |
| Exact symmetry, physically consistent outputs | More complex layers, harder to implement and debug |
| Big data efficiency, better extrapolation to unseen poses | Higher-order tensor products are slow and memory-heavy |
| Fewer parameters spent on learning symmetry | Wrong or overly strict symmetry hurts (chirality needs $SE(3)$, not $E(3)$) |

**Decision rule:** use equivariance when the symmetry is genuinely exact (physics, geometry) and data is scarce. With huge datasets and approximate symmetries, a plain Transformer plus augmentation can be competitive; AlphaFold 3, for instance, dropped strict equivariance in parts of its architecture.

### 13.82 How do Graph Attention Networks work, and how do they differ from GCNs?

A **Graph Attention Network (GAT, Veličković et al. 2018)** is a message-passing GNN in which each node computes **learned, data-dependent weights** over its neighbours instead of using fixed weights derived from the graph structure.

**GCN aggregation (fixed weights)**

$$
h_i' = \sigma\left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{d_i d_j}}\, W h_j \right)
$$

The weight of neighbour $j$ depends only on node degrees, so every neighbour of equal degree contributes equally, whatever its features.

**GAT attention coefficients**

For each edge, score the pair, then normalise with a softmax over the neighbourhood:

$$
e_{ij} = \operatorname{LeakyReLU}\left( a^\top \left[ W h_i \,\Vert\, W h_j \right] \right)
$$

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}, \qquad h_i' = \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j \right)
$$

Here $\Vert$ is concatenation and $a$ is a learned vector. Attention is **masked** to existing edges, so cost is $O(\lvert E \rvert)$, not $O(N^2)$ as in a Transformer.

**Multi-head attention**

$K$ independent heads stabilise training. Hidden layers **concatenate** the heads; the output layer **averages** them:

$$
h_i' = \Big\Vert_{k=1}^{K} \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{k} W^{k} h_j \right)
$$

**GATv2: fixing static attention**

Brody et al. showed that because $a^\top[W h_i \Vert W h_j]$ splits into $a_1^\top W h_i + a_2^\top W h_j$ and LeakyReLU is monotonic, the **ranking of neighbours is the same for every query node**. That is *static* attention: one globally "best" neighbour type wins everywhere. GATv2 moves the nonlinearity inside:

$$
e_{ij} = a^\top \operatorname{LeakyReLU}\left( W \left[ h_i \,\Vert\, h_j \right] \right)
$$

Now the score is a genuine function of the pair, giving **dynamic** attention, at the same parameter cost. GATv2 is the safer default and is available in PyG and DGL as `GATv2Conv`.

**GAT vs GCN**

| | GCN | GAT / GATv2 |
| --- | --- | --- |
| Neighbour weights | Fixed (degree-normalised) | Learned per edge from features |
| Inductive to new graphs | Originally used transductively; works inductively | Yes, naturally |
| Heterophily / noisy edges | Averages in bad neighbours | Can down-weight irrelevant neighbours |
| Cost and memory | Lowest (sparse matmul) | Higher: per-edge scores, $K$ heads |
| Interpretability | None | Attention weights (treat with caution) |

**Over-smoothing**

Both suffer from it: repeated neighbourhood averaging drives node embeddings towards the same vector, so accuracy usually peaks at 2–4 layers. Attention slows it but does not remove it, since $\alpha_{ij}$ is still a convex combination. Mitigations: residual/skip connections, jumping knowledge, PairNorm, DropEdge, and shallow-but-wide models. A related issue is **over-squashing**: information from exponentially many distant nodes is compressed into a fixed-size vector, which graph rewiring or graph Transformers address.

**When to use which**

- **GCN or GraphSAGE:** large, homophilous graphs, tight latency/memory budgets, strong baseline.
- **GATv2:** neighbours differ in relevance (fraud graphs with noisy links, citation graphs with mixed topics, knowledge graphs), or you need edge-level importance.
- **Caveat:** attention weights are not faithful explanations; validate with ablations before showing them to a fraud analyst as "the reason".

### 13.83 What are modern Hopfield networks, and how are they related to Transformer attention?

**Modern Hopfield networks** (dense associative memories) replace the quadratic energy of the classical Hopfield network with a **sharply peaked, higher-order or exponential interaction**, which raises storage capacity from linear to exponential in the dimension. The continuous version of Ramsauer et al. (2020) has an update rule that is *exactly* softmax attention.

**From classical to dense**

- Classical energy is quadratic in the overlaps between state $\xi$ and stored patterns $x_\mu$, so patterns interfere and capacity is about $0.14N$.
- **Krotov and Hopfield (2016)** used $E = -\sum_\mu F(x_\mu^\top \xi)$ with a polynomial $F(z)=z^n$, giving capacity growing like $N^{n-1}$. **Demircigil et al. (2017)** used $F(z)=e^z$ for exponential capacity in binary networks.
- A sharper $F$ makes each memory's energy well narrow, so wells interfere far less.

**The continuous modern Hopfield network**

Store $N$ patterns as columns of $X \in \mathbb{R}^{d \times N}$, with state $\xi \in \mathbb{R}^d$ and inverse temperature $\beta$:

$$
E(\xi) = -\frac{1}{\beta} \log \sum_{\mu=1}^{N} \exp\left(\beta\, x_\mu^\top \xi\right) + \frac{1}{2} \xi^\top \xi + \text{const}
$$

Minimising it with the concave-convex procedure gives the update:

$$
\xi^{\text{new}} = X \operatorname{softmax}\left( \beta X^\top \xi \right)
$$

Properties proved in the paper:

- It converges to stationary points of the energy, and retrieval usually happens in **one update**.
- Capacity is **exponential** in the dimension $d$ for well-separated patterns (under a relaxed, $\epsilon$-close notion of retrieval).
- $\beta$ controls the regime: large $\beta$ retrieves a single stored pattern; small $\beta$ gives a **metastable state** that averages similar patterns; very small $\beta$ gives a global average.

**The link to attention**

Treat the state patterns as queries and the stored patterns as keys and values, with learned projections:

$$
Q = R W_Q, \quad K = Y W_K, \quad V = K W_V
$$

$$
Z = \operatorname{softmax}\left( \beta\, Q K^\top \right) V
$$

With $\beta = 1/\sqrt{d_k}$ this is exactly Transformer self-/cross-attention. So **one attention head performs one step of associative memory retrieval**: the query is a noisy cue and the output is the recalled (or blended) memory.

**Insights this view gives**

- Ramsauer et al. found that lower Transformer layers often sit in the *global averaging* regime and higher layers in the *metastable / single-pattern* regime, which motivates pruning or simplifying some heads.
- It explains why **temperature** and attention entropy matter: too flat and recall blurs; too sharp and gradients vanish.
- It connects attention to **content-addressable memory**, which is why retrieval-augmented and memory-augmented models fit naturally on top of it.

**Uses in practice**

- **Hopfield layers** (the `hflayers` PyTorch package): `Hopfield`, `HopfieldPooling` (a learned query attends over a set, e.g. pooling a bag of instances) and `HopfieldLayer` (a learned, fixed memory as a lookup table).
- **Multiple-instance learning** on immune repertoires (DeepRC), where one sample has hundreds of thousands of sequences.
- **Few-shot and tabular learning**, and memory modules in reinforcement learning.
- Later work (sparse Hopfield networks with sparsemax/entmax) addresses softmax's inability to retrieve exactly and its metastable blending.

**Interview one-liner:** attention is one gradient-free update of an energy-based associative memory with exponential capacity; $\beta$ decides whether it recalls one memory or averages many.

### 13.84 How is a Restricted Boltzmann Machine trained, and what is contrastive divergence?

An RBM is trained by **maximum likelihood**, but the exact gradient needs an expectation under the model distribution, which is intractable. **Contrastive divergence (CD, Hinton 2002)** approximates that expectation with a Gibbs chain started at the data and run for only $k$ steps.

**Energy, joint and free energy** (binary visible $v$, binary hidden $h$)

$$
E(v, h) = -a^\top v - b^\top h - v^\top W h, \qquad p(v, h) = \frac{e^{-E(v, h)}}{Z}
$$

Summing out the hidden units analytically gives the **free energy**:

$$
F(v) = -a^\top v - \sum_{j} \log\left(1 + e^{\,b_j + v^\top W_{:,j}}\right), \qquad p(v) = \frac{e^{-F(v)}}{Z}
$$

The partition function $Z$ sums over all $2^{n_v + n_h}$ configurations, so likelihood cannot be computed directly.

**Factorised conditionals** (the payoff of the bipartite structure)

$$
p(h_j = 1 \mid v) = \sigma\left(b_j + \sum_i v_i W_{ij}\right), \qquad p(v_i = 1 \mid h) = \sigma\left(a_i + \sum_j W_{ij} h_j\right)
$$

All hidden units can be sampled in parallel given $v$, and vice versa: **block Gibbs sampling**.

**The exact gradient**

$$
\frac{\partial \log p(v)}{\partial W_{ij}} = \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}
$$

The **positive phase** (data term) is cheap. The **negative phase** needs samples from the model, which would require running Gibbs sampling to equilibrium for every update.

**CD-k update**

Start at a training example $v^{(0)}$, alternate $h \sim p(h \mid v)$ and $v \sim p(v \mid h)$ for $k$ steps, and use the end of the chain as the negative sample:

$$
\Delta W_{ij} = \eta \left( \langle v_i h_j \rangle_{0} - \langle v_i h_j \rangle_{k} \right), \quad \Delta a_i = \eta \left( v_i^{(0)} - v_i^{(k)} \right), \quad \Delta b_j = \eta \left( p_j^{(0)} - p_j^{(k)} \right)
$$

Intuitively: lower the free energy of the data and raise it for the model's "fantasies" near the data. CD-1 often works well for feature learning. It is **biased**: it does not follow the true likelihood gradient and leaves spurious modes far from the data untouched.

```python
def cd_k(v0, W, a, b, k=1, lr=1e-2):
    ph0 = torch.sigmoid(v0 @ W + b)          # positive phase
    vk = v0
    for _ in range(k):                         # Gibbs chain from the data
        hk = torch.bernoulli(torch.sigmoid(vk @ W + b))
        vk = torch.bernoulli(torch.sigmoid(hk @ W.T + a))
    phk = torch.sigmoid(vk @ W + b)
    W += lr * (v0.T @ ph0 - vk.T @ phk) / v0.shape[0]
    a += lr * (v0 - vk).mean(0)
    b += lr * (ph0 - phk).mean(0)
```

**Persistent CD (PCD, Tieleman 2008)**

Keep a set of "fantasy particles" whose chains are **not reset to the data** between updates; each update advances them a few steps. Because weights change slowly, the chains stay close to the model distribution, giving a much better likelihood gradient and better generative samples. Variants: fast-weights PCD, parallel tempering.

**Practical tips (Hinton's practical guide)**

- Use hidden **probabilities** rather than samples for the final statistics to reduce variance; sample hidden states when driving the chain.
- Small random initial weights, hidden bias near 0, visible bias at $\log(p_i/(1-p_i))$; momentum and weight decay help.
- **Reconstruction error is a misleading metric.** Monitor the free-energy gap between training and validation data for overfitting, or estimate likelihood with annealed importance sampling.
- Real-valued inputs need Gaussian–Bernoulli RBMs with a much smaller learning rate.

**Deep Belief Networks**

Hinton, Osindero and Teh (2006) trained a stack of RBMs **greedily layer by layer**: each RBM models the hidden activations of the one below. The stack was then fine-tuned with backpropagation (or wake-sleep for the generative model). This unsupervised pretraining made deep networks trainable at the time and helped launch the deep learning revival.

**Why they faded**

- ReLU, good initialisation (Xavier/He), batch normalisation, dropout and GPUs made deep nets trainable **without** pretraining.
- Sampling-based training is slow, hard to tune and hard to evaluate (no tractable likelihood).
- VAEs, GANs, autoregressive models and diffusion models gave better samples with simple gradient training.
- The legacy remains: energy-based modelling, the positive/negative phase idea in contrastive learning, score matching and diffusion, and RBMs in quantum many-body physics.

---

← [Chapter 12. Model Evaluation and Experimentation](12-model-evaluation-and-experimentation.md) · [Table of contents](../README.md) · [Chapter 14. Natural Language Processing](14-natural-language-processing.md) →
