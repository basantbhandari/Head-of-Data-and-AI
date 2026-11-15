<!-- notion-id: 3d667418285480b49b0ef1c9f36220a3 -->
# Deep Learning

Deep learning interview questions: neural network basics, activations, regularization, generative models (autoencoders, VAEs, GANs, diffusion), CNNs, RNNs/LSTMs, Transformers, transfer learning, multimodal and world models.

### Q1. What are neural networks?

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

### Q2. Explain the Feedforward Neural Network.

**Feedforward Neural Network (FNN)**

- Simplest type of artificial neural network — data flows in one direction: input → hidden layers → output (no loops/cycles)
- Also called Multi-Layer Perceptron (MLP) when it has hidden layers

**Structure**

- **Input layer**: receives raw features
- **Hidden layer(s)**: apply weighted transformations + activation functions
- **Output layer**: produces final prediction (class probabilities or continuous value)

**How it works**

1. Each neuron computes: z = w·x + b, then applies activation function f(z)
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

### Q3. What are forward propagation and backward propagation?

**Forward Propagation & Backpropagation**

**Forward Propagation**

- Process of passing input data through the network layer by layer to compute output/prediction
- Each layer: z = w·x + b → activation f(z) → passed to next layer
- Ends with output layer producing prediction
- Loss function then compares prediction to actual value

**Backpropagation**

- Process of computing gradients of loss function w.r.t. each weight, by propagating error backward from output to input layer
- Uses **chain rule** of calculus to compute how much each weight contributed to the error
- Gradients then used to update weights via gradient descent: θ = θ − α∇J(θ)

**Together**

1. Forward pass: compute prediction and loss
2. Backward pass: compute gradients (backpropagation)
3. Update weights (optimizer step)
4. Repeat for multiple epochs until convergence

### Q4. What is backpropagation?

**Backpropagation**

- Algorithm to compute gradients of the loss function with respect to each weight in a neural network
- Propagates error backward from output layer to input layer using the **chain rule** of calculus
- Gradients indicate how much each weight contributed to the prediction error
- These gradients are then used by an optimizer (e.g., gradient descent, Adam) to update weights and minimize loss
- Core mechanism enabling neural networks to learn from data

### Q5. Can you name and explain a few hyperparameters used for training a neural network?

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

### Q6. What is the advantage of deep learning over traditional machine learning?

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

### Q7. What are activation functions, and why are they used?

**Activation Functions**

- Mathematical functions applied to neuron outputs to introduce **non-linearity** into neural networks
- Without them, network would just be a linear transformation (stacking linear layers = still linear), unable to learn complex patterns

**Why used**

- Enable network to learn non-linear, complex relationships in data
- Determine whether a neuron should "fire" (activate) based on its input
- Affect gradient flow during backpropagation

**Common activation functions**

- **Sigmoid**: 1/(1+e⁻ˣ), output 0-1, used in binary classification output — suffers vanishing gradient
- **Tanh**: output -1 to 1, zero-centered, still has vanishing gradient issue
- **ReLU**: max(0, x), most popular — fast, avoids vanishing gradient for positive values, but "dying ReLU" problem
- **Leaky ReLU**: allows small negative slope, fixes dying ReLU
- **Softmax**: converts logits to probability distribution, used in multi-class output layer
- **GELU/Swish**: smoother variants used in modern architectures (Transformers)

### Q8. Explain Sigmoid, Tanh, ReLU, LeakyReLU, and Softmax activation functions with their pros and cons.

**Activation Functions: Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax**

**Sigmoid**

- Formula: 1/(1+e⁻ˣ), output range: 0 to 1
- Pros: Smooth, good for binary classification output (probability)
- Cons: Vanishing gradient (saturates at extremes), not zero-centered, computationally expensive (exp)

**Tanh**

- Formula: (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ), output range: −1 to 1
- Pros: Zero-centered (better than sigmoid for hidden layers)
- Cons: Still suffers vanishing gradient at extremes

**ReLU (Rectified Linear Unit)**

- Formula: max(0, x), output range: 0 to ∞
- Pros: Computationally efficient, avoids vanishing gradient for x>0, sparse activation (efficiency)
- Cons: **Dying ReLU** problem — neurons stuck outputting 0 if input always negative (gradient = 0)

**Leaky ReLU**

- Formula: x if x>0, else αx (small α like 0.01)
- Pros: Fixes dying ReLU — allows small gradient for negative inputs
- Cons: Results can be inconsistent; α is a hyperparameter to tune

**Softmax**

- Formula: e\^zᵢ / Σe\^zⱼ, output: probability distribution summing to 1
- Pros: Ideal for multi-class classification output layer, interpretable as probabilities
- Cons: Sensitive to outliers/large logits (can saturate), not used in hidden layers

### Q9. Why are Sigmoid and Tanh not preferred in the hidden layers of a neural network?

**Why Sigmoid and Tanh Aren't Preferred in Hidden Layers**

- **Vanishing gradient problem**: both saturate at extremes (very high/low inputs) where derivative ≈ 0
  - Sigmoid derivative max = 0.25; Tanh derivative max = 1, but still saturates
  - In deep networks, gradients multiply across layers (chain rule) → shrink exponentially toward zero
  - Early layers get near-zero updates → learn very slowly or stop learning
- **Computationally expensive**: involve exponential calculations (e\^x), slower than ReLU's simple max(0,x)
- **Sigmoid not zero-centered**: outputs always positive (0 to 1) → gradients during backprop tend to all be same sign → inefficient zig-zag weight updates
- **Tanh is zero-centered** (−1 to 1) — better than sigmoid, but still suffers vanishing gradient at saturation

**Why ReLU preferred instead**

- No saturation for positive inputs → gradient stays constant (1) → no vanishing gradient issue in that region
- Computationally cheap (simple thresholding)
- Sparse activation (some neurons output exactly 0) → more efficient computation

**Where Sigmoid/Tanh still used**

- Sigmoid: output layer for binary classification, gates in LSTM/GRU
- Tanh: hidden states in RNN/LSTM (bounded output helps stability)

### Q10. What is dropout, and why is it effective?

**Dropout**

- Regularization technique that randomly deactivates (sets to zero) a fraction of neurons during each training iteration
- Dropout rate (p) typically 0.2–0.5 — probability a neuron is dropped

**Why effective**

- **Prevents co-adaptation**: neurons can't rely too heavily on specific other neurons, forces more robust, independent feature learning
- **Acts like ensemble learning**: each training pass uses a different "thinned" sub-network — effectively trains many different network architectures that share weights, final model approximates averaging their predictions
- **Reduces overfitting**: network can't memorize training data through complex co-dependencies between neurons
- **Improves generalization**: forces redundant, distributed representations rather than relying on a few neurons

**Implementation notes**

- Only applied during training; all neurons active during inference (weights scaled by (1−p) to compensate, or "inverted dropout" scales during training instead)
- Typically applied to fully connected layers, less common in convolutional layers

### Q11. What is the effect of dropout on training and inference speed?

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

### Q12. What is L1/L2 regularization, and how does it affect a neural network?

**L1/L2 Regularization in Neural Networks**

- Technique to prevent overfitting by adding penalty term to loss function based on weight magnitudes
- Total Loss = Original Loss + λ × Penalty

**L1 Regularization (Lasso)**

- Penalty = λ Σ\|wᵢ\|
- Drives some weights to exactly **zero** → sparse network (effectively removes some connections)
- Acts like built-in feature/neuron selection

**L2 Regularization (Ridge / Weight Decay)**

- Penalty = λ Σwᵢ²
- Shrinks weights toward zero but rarely exactly zero → smaller, more distributed weights
- Most commonly used in neural networks (often called "weight decay")

**Effect on neural network**

- Prevents weights from growing too large → reduces model complexity/capacity
- Smaller weights → smoother decision boundaries → less overfitting, better generalization
- L2 more common in deep learning (smoother optimization, doesn't zero out neurons abruptly)
- λ (regularization strength) is a hyperparameter — too high causes underfitting, too low has little effect

### Q13. What is batch normalization, and why is it used?

**Batch Normalization**

- Technique that normalizes layer inputs (activations) within each mini-batch during training
- Normalizes to mean=0, std=1, then applies learnable scale (γ) and shift (β) parameters
- Formula: x̂ = (x − μ\_batch) / √(σ²\_batch + ε), then y = γx̂ + β

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

### Q14. What are the hyperparameters for batch normalization that can be optimized?

**Batch Normalization — Tunable Hyperparameters**

- **Momentum**: controls how running mean/variance (used at inference) are updated via exponential moving average — typically 0.9–0.99
- **Epsilon (ε)**: small constant added to variance for numerical stability (avoid division by zero) — typically 1e-5
- **γ (scale) and β (shift)**: learnable parameters per feature, initialized (γ=1, β=0) but learned during training — not manually tuned but initialization can matter
- **Batch size**: indirectly affects BN — larger batches give more stable/accurate batch statistics; very small batches make BN noisy/unreliable
- **Placement**: before or after activation function (commonly before) — architectural choice affecting performance
- **Axis/dimension** to normalize over: standard BN normalizes over batch dimension per channel/feature; choice matters for conv layers (per-channel) vs dense layers
- **Whether to use running statistics or batch statistics at inference**: standard is running stats, but affects behavior with small/varying batch sizes
- In practice, momentum and epsilon are the main hyperparameters tuned; γ/β are learned automatically

### Q15. What is parameter sharing in deep learning?

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

### Q16. What is representation learning, and why is it useful?

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

### Q17. What is a generative model, and how does it differ from a discriminative model?

**Generative vs Discriminative Models**

**Generative Model**

- Learns the joint probability distribution P(X, Y) — models how data is generated
- Can generate new samples resembling training data
- Learns underlying distribution of each class
- Examples: Naive Bayes, GANs, VAEs, Diffusion Models, GPT (language generation)

**Discriminative Model**

- Learns conditional probability P(Y\|X) — directly models decision boundary between classes
- Focuses on distinguishing between classes, not generating data
- Examples: Logistic Regression, SVM, Random Forest, standard CNNs for classification

| Aspect | Generative | Discriminative |
| --- | --- | --- |
| Models | P(X, Y) — joint distribution | P(Y\|X) — conditional distribution |
| Goal | Learn how data is generated | Learn decision boundary |
| Can generate new data | Yes | No |
| Typically | More flexible, can handle missing data | Better performance on classification tasks (with enough data) |
| Examples | Naive Bayes, GANs, VAEs, GPT | Logistic Regression, SVM, standard classifiers |

- Generative models can be used for classification too (via Bayes' rule: P(Y\|X) ∝ P(X\|Y)P(Y)) but typically less accurate than discriminative for pure classification when data is abundant

### Q18. Can you explain how a generative model works?

**How Generative Models Work**

- Goal: learn the underlying probability distribution of training data P(X) (or joint P(X,Y)), then use it to generate new samples

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
- P(x) = Π P(xᵢ \| x₁...xᵢ₋₁)
- All approaches aim to capture data distribution well enough to produce novel, realistic samples

### Q19. Explain Encoder-Decoder Architecture.

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

### Q20. What is the difference between encoder-only, decoder-only, and encoder-decoder Transformer architectures?

> 🔁 _See also [01-llm-fundamentals.md → Q31](../ai-engineering/01-llm-fundamentals.md#q31-what-is-the-difference-between-encoder-only-decoder-only-and-encoder-decoder-transformer-architectures) (separate Notion notes on the same question)._

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

### Q21. What is Latent Space?

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

### Q22. What are autoencoders? Explain their layers and practical uses.

**Autoencoders**

- Unsupervised neural network that learns to compress data into a lower-dimensional representation, then reconstruct it back to original
- Trained to minimize reconstruction error (output ≈ input)

**Layers/Architecture**

1. **Encoder**: series of layers that compress input into smaller latent representation (bottleneck)
   - Input → hidden layers (decreasing size) → latent vector (z)
2. **Bottleneck/Latent space**: compressed, lower-dimensional representation of input — forces network to learn only most important features
3. **Decoder**: series of layers that reconstruct original input from latent representation
   - Latent vector (z) → hidden layers (increasing size) → output (reconstructed input)

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

### Q23. What is a Variational Autoencoder (VAE), and how is it different from a traditional autoencoder?

**Variational Autoencoder (VAE) vs Traditional Autoencoder**

**VAE**

- Generative model that learns a **probabilistic** latent space instead of a fixed point
- Encoder outputs **parameters of a distribution** (mean μ, variance σ²) for each input, not a single latent vector
- Sample z from this distribution (using reparameterization trick), decoder reconstructs from sampled z
- **Loss function** = Reconstruction Loss + KL Divergence (regularizes latent space to follow standard normal distribution)

**Traditional Autoencoder**

- Encoder maps input to a **single fixed point** in latent space (deterministic)
- No constraint on latent space structure/distribution
- Only trained to minimize reconstruction loss

**Key Differences**

| Aspect | Traditional Autoencoder | VAE |
| --- | --- | --- |
| Latent representation | Fixed point (deterministic) | Probability distribution (μ, σ) |
| Latent space structure | No constraint, can have gaps | Continuous, smooth (regularized via KL divergence) |
| Generative capability | Poor — latent space not smooth/continuous | Good — can sample new points, generate new realistic data |
| Loss function | Reconstruction loss only | Reconstruction loss + KL divergence |
| Interpolation | Latent space may have "holes" (unrealistic outputs) | Smooth interpolation between points produces meaningful outputs |

**Why VAE is better for generation**

- KL divergence term forces latent space to be continuous and well-structured (close to standard normal)
- Enables meaningful sampling/interpolation — nearby points in latent space produce similar, realistic outputs
- Traditional autoencoders can have discontinuous latent space, leading to unrealistic outputs when sampling

### Q24. How does VAE impose a probabilistic structure on the latent space, and why is that important?

**How VAE Imposes Probabilistic Structure on Latent Space**

**Mechanism**

1. **Encoder outputs distribution parameters**, not a point: for each input x, encoder outputs mean μ(x) and variance σ²(x) — parameterizing a Gaussian distribution q(z\|x)
2. **Reparameterization trick**: sample z = μ + σ⊙ε, where ε \~ N(0,1)
   - Allows sampling while keeping the process differentiable (backprop-able) — direct sampling isn't differentiable; this trick separates randomness (ε) from learnable parameters (μ, σ)
3. **KL Divergence regularization**: loss includes KL(q(z\|x) \|\| p(z)), where p(z) = standard normal N(0,1)
   - Forces the learned distribution q(z\|x) for each input to stay close to a standard normal prior
   - Penalizes latent distributions that drift too far from N(0,1) — prevents encoder from "cheating" by spreading points arbitrarily far apart
4. **Total loss** = Reconstruction Loss + β·KL Divergence (β controls regularization strength — β-VAE)

**Why this is important**

- **Continuity**: Forces nearby points in latent space to decode to similar outputs — no discontinuous "holes"
- **Completeness**: Every point sampled from the prior N(0,1) should decode to a realistic output (since all data's distributions were pulled toward this same prior)
- **Enables generation**: Because latent space is well-structured and covers a known distribution, you can sample random z \~ N(0,1) directly from the prior and decode it into a realistic new sample — impossible reliably with vanilla autoencoders
- **Smooth interpolation**: moving between two points in latent space produces gradual, meaningful transitions in output (e.g., morphing between two generated faces)
- **Prevents overfitting/memorization**: regularization stops the model from just memorizing exact latent points for each training example

**Trade-off**

- Too much KL weight (high β) → latent space very regularized but reconstruction quality drops (posterior collapse risk)
- Too little KL weight → latent space less structured, generation quality suffers, behaves more like vanilla autoencoder

### Q25. What is the architecture of a Generative Adversarial Network (GAN)?

**GAN (Generative Adversarial Network) Architecture**

- Two neural networks trained simultaneously in an adversarial (competing) setup

**Generator (G)**

- Takes random noise vector z (from latent space, e.g., N(0,1)) as input
- Generates fake data samples: G(z) → fake sample
- Goal: produce samples realistic enough to fool the discriminator

**Discriminator (D)**

- Binary classifier — takes a sample (real or fake) as input
- Outputs probability that input is real: D(x) → \[0,1\]
- Goal: correctly distinguish real samples (from training data) vs fake samples (from generator)

**Training process (adversarial)**

1. Generator creates fake sample from noise: G(z)
2. Discriminator evaluates both real sample (x) and fake sample (G(z))
3. Discriminator updated to better classify real vs fake (maximize correct classification)
4. Generator updated to better fool discriminator (maximize D's error on fake samples)
5. Repeat — both networks improve together in a minimax game

**Objective function**

- min\_G max\_D \[E(log D(x)) + E(log(1−D(G(z))))\]
- Discriminator maximizes this; Generator minimizes it

**Convergence**

- Ideally reaches Nash equilibrium — generator produces samples indistinguishable from real data, discriminator outputs \~0.5 (can't tell real from fake)

**Challenges**

- Training instability, mode collapse (generator produces limited variety), vanishing gradients
- Solutions: Wasserstein GAN (WGAN), spectral normalization, careful learning rate balancing

### Q26. What are the roles of the generator and discriminator in a GAN?

**Roles of Generator and Discriminator in GAN**

**Generator (G)**

- Takes random noise (z) as input, transforms it into synthetic/fake data samples
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

### Q27. What is mode collapse in GANs, and how can it be mitigated?

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

### Q28. How are GANs used in image synthesis or image-to-image translation tasks?

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
  - Uses **two generators** (G: A→B, F: B→A) and **two discriminators**
  - **Cycle consistency loss**: F(G(A)) ≈ A — translating to the target domain and back should reconstruct the original image
  - Enables style transfer, domain adaptation without paired examples

**Common use cases**

- Photo-realistic image generation (faces, art, synthetic data)
- Style transfer (photo → painting style)
- Super-resolution (low-res → high-res, e.g., SRGAN)
- Colorization (grayscale → color)
- Domain adaptation (synthetic → real-world images, e.g., for training autonomous vehicles)
- Data augmentation (generating synthetic training data)

### Q29. Explain Convolutional Neural Networks (CNNs).

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

### Q30. Explain filters in CNN.

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

### Q31. Explain the stride in CNN.

**Stride in CNN**

- Number of pixels the filter/kernel moves (shifts) at each step while sliding across the input
- Controls how much overlap occurs between successive filter applications

**Effect of stride**

- **Stride = 1**: filter moves one pixel at a time — maximum overlap, produces larger output feature map (finer detail retained)
- **Stride = 2 (or more)**: filter skips pixels — less overlap, produces smaller output feature map (downsampling effect)

**Output size formula**

- Output size = ⌊(Input size − Filter size + 2×Padding) / Stride⌋ + 1

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

### Q32. Explain padding in CNN.

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
  - Output size shrinks: Output = Input − Filter + 1
  - Loses some border information
- **Same padding**:
  - Adds enough zero-padding so output size equals input size (with stride=1)
  - Preserves spatial dimensions across layers — useful for deep networks

**Output size formula**

- Output size = ⌊(Input size − Filter size + 2×Padding) / Stride⌋ + 1

**Benefits**

- Preserves spatial resolution across layers (allows deeper networks without shrinking too fast)
- Ensures border/edge pixels contribute equally to feature extraction
- Gives more control over output dimensions for architecture design

### Q33. Explain pooling in CNN.

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

### Q34. Explain fully connected layers in CNN.

Fully connected (FC) layers—also called dense layers—are the layers near the end of a CNN that take the high-level features extracted by earlier convolutional/pooling layers and use them to make a final prediction (like classifying an image).

#### Where they sit in a CNN

A typical CNN pipeline looks like:

**Input → \[Conv → Activation → Pooling\] × N → Flatten → Fully Connected layers → Output**

The convolutional layers act as feature extractors (edges, textures, shapes, object parts), while the fully connected layers act as the "reasoning" stage that combines those features into a decision.

> ✍️ _Addition drafted during sync._

- **Flatten step:** the final feature map (e.g., 7×7×512) is flattened into a 1-D vector (25,088 values) so it can feed a dense layer. Each FC neuron computes `y = f(W·x + b)` and connects to *every* input value, so FC layers learn global combinations of features, unlike the local, weight-shared filters of conv layers.
- **Typical head:** one or two hidden FC layers (often with ReLU + dropout), then an output FC layer sized to the number of classes, followed by softmax (multi-class) or sigmoid (binary/multi-label).
- **Parameter cost:** FC layers usually hold most of a classic CNN's parameters. In VGG-16, the first FC layer alone (25,088 × 4,096 ≈ 103M weights) is most of the model's ~138M parameters, which makes these layers the main source of overfitting and memory use.
- **Modern practice:** many architectures (ResNet, EfficientNet) swap the big flatten + FC stack for **Global Average Pooling** followed by a single FC layer. This cuts parameters sharply, reduces overfitting and lets the network accept variable input sizes.
- **Production angle:** in transfer learning you usually freeze the convolutional backbone and replace or retrain only the FC head for the new label set. That is cheap and works well with little labelled data.

### Q35. What is a Recurrent Neural Network (RNN)?

#### Recurrent Neural Network (RNN)

- An RNN is a class of neural networks designed specifically for processing **sequential data**, such as text, speech, time series, video frames, or any data where the order of elements carries meaning.
- Unlike feedforward neural networks, which assume all inputs are independent of each other, RNNs introduce the concept of **memory** through a hidden state that is passed from one time step to the next.
- **Core mechanism:**
  - At each time step *t*, the RNN takes two inputs: the current input vector `x_t` and the previous hidden state `h_(t-1)`.
  - It computes a new hidden state: `h_t = f(W_xh * x_t + W_hh * h_(t-1) + b_h)`, where `f` is typically a non-linear activation like tanh or ReLU.
  - It can also produce an output at that step: `y_t = g(W_hy * h_t + b_y)`.
  - Crucially, the **same weight matrices** (`W_xh`, `W_hh`, `W_hy`) are shared/reused across all time steps — this is called **parameter sharing** and is what allows RNNs to generalize across sequences of varying length.
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

### Q36. What are the limitations of RNNs, and how are they solved?

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

### Q37. What are LSTM and GRU? How do they solve long-term dependency issues?

#### LSTM and GRU

##### What is LSTM (Long Short-Term Memory)?

- LSTM is a special type of RNN architecture designed to overcome the vanishing gradient problem and effectively learn **long-term dependencies** in sequential data.
- It introduces a separate **cell state (`C_t`)** in addition to the hidden state (`h_t`). The cell state acts like a "conveyor belt" that carries information across time steps with only minor linear modifications, allowing information to flow largely unchanged unless explicitly modified.
- LSTM controls the flow of information using **three gates**, each implemented as a sigmoid-activated layer that outputs values between 0 (block) and 1 (allow):
  - **Forget Gate (`f_t`):** Decides what information to discard from the previous cell state.
    - `f_t = σ(W_f · [h_(t-1), x_t] + b_f)`
  - **Input Gate (`i_t`):** Decides what new information to add to the cell state.
    - `i_t = σ(W_i · [h_(t-1), x_t] + b_i)`
    - A candidate value `C̃_t = tanh(W_c · [h_(t-1), x_t] + b_c)` is also computed.
  - **Cell State Update:** Combines the forget and input gates to update the cell state.
    - `C_t = f_t * C_(t-1) + i_t * C̃_t`
  - **Output Gate (`o_t`):** Decides what part of the cell state to output as the hidden state.
    - `o_t = σ(W_o · [h_(t-1), x_t] + b_o)`
    - `h_t = o_t * tanh(C_t)`
- **How this solves long-term dependency issues:**
  - The cell state update is **additive** (`f_t * C_(t-1) + i_t * C̃_t`) rather than purely multiplicative like in vanilla RNNs, which prevents gradients from vanishing exponentially over many time steps.
  - The forget gate allows the network to **selectively retain or discard** information, so relevant information from far back in the sequence can be preserved unchanged for long durations if needed.
  - Gradients can flow through the cell state pathway with minimal decay, since there's no repeated matrix multiplication with the same weight matrix at every step along that specific pathway — this is often called the **constant error carousel**.

---

##### What is GRU (Gated Recurrent Unit)?

- GRU is a simplified variant of LSTM, introduced to achieve similar performance with **fewer parameters** and **less computational complexity** by combining the cell state and hidden state into a single hidden state.
- GRU uses **two gates** instead of three:
  - **Update Gate (`z_t`):** Controls how much of the past hidden state should be carried forward versus how much of the new candidate state should be used. It essentially combines the roles of LSTM's forget and input gates.
    - `z_t = σ(W_z · [h_(t-1), x_t] + b_z)`
  - **Reset Gate (`r_t`):** Controls how much of the past hidden state to forget when computing the new candidate state.
    - `r_t = σ(W_r · [h_(t-1), x_t] + b_r)`
  - **Candidate Hidden State:**
    - `h̃_t = tanh(W_h · [r_t * h_(t-1), x_t] + b_h)`
  - **Final Hidden State Update:**
    - `h_t = (1 - z_t) * h_(t-1) + z_t * h̃_t`
- **How this solves long-term dependency issues:**
  - Similar to LSTM, the final hidden state update is a **linear interpolation** between the previous hidden state and the new candidate state, controlled by the update gate — this additive structure helps preserve gradients over long sequences.
  - When `z_t` is close to 0, the hidden state from the past is carried forward almost unchanged, allowing information to persist over many time steps without significant decay.
  - The reset gate allows the model to drop irrelevant past information when computing new candidates, giving it flexibility similar to LSTM's forget gate.

---

##### Key Differences Between LSTM and GRU

- **Number of gates:** LSTM has 3 gates (forget, input, output); GRU has 2 gates (update, reset).
- **State representation:** LSTM maintains two separate states (cell state and hidden state); GRU merges them into a single hidden state.
- **Parameters:** GRU has fewer parameters than LSTM, making it computationally cheaper and faster to train.
- **Performance:** Both perform similarly on many tasks; LSTM sometimes performs better on tasks requiring finer control over memory (very long sequences), while GRU often trains faster and works well on smaller datasets.
- **Use case selection:** GRU is often preferred when computational efficiency matters or the dataset is smaller; LSTM is preferred when the task requires more precise memory control over very long sequences.

### Q38. What are the main gates in LSTM and their roles?

#### Main Gates in LSTM and Their Roles

- **LSTM has three main gates**, each implemented as a sigmoid neural network layer that outputs values between 0 and 1, acting as a filter to control how much information passes through.

---

##### 1. Forget Gate (`f_t`)

- **Role:** Decides what information from the **previous cell state** (`C_(t-1)`) should be discarded or retained.
- **Formula:**
  - `f_t = σ(W_f · [h_(t-1), x_t] + b_f)`
- **How it works:**
  - Takes the previous hidden state `h_(t-1)` and the current input `x_t`, concatenates them, and passes them through a sigmoid function.
  - An output close to **0** means "completely forget this information."
  - An output close to **1** means "completely retain this information."
  - This output is multiplied element-wise with the previous cell state `C_(t-1)`.
- **Example use case:** In language modeling, when a new subject appears in a sentence, the forget gate may decide to discard the gender/number information of the old subject that's no longer relevant.

---

##### 2. Input Gate (`i_t`)

- **Role:** Decides what **new information** from the current input should be added to the cell state.
- **Formula:**
  - `i_t = σ(W_i · [h_(t-1), x_t] + b_i)`
  - Along with a candidate value: `C̃_t = tanh(W_c · [h_(t-1), x_t] + b_c)`
- **How it works:**
  - The sigmoid part (`i_t`) decides **which values** to update (how much of the new information matters).
  - The tanh part (`C̃_t`) creates a vector of **new candidate values** that could be added to the cell state.
  - These are combined (`i_t * C̃_t`) and added to the cell state during the cell state update step.
- **Example use case:** When a new subject is introduced in a sentence, the input gate adds this new subject's relevant information (e.g., gender, number) into the cell state.

---

##### 3. Output Gate (`o_t`)

- **Role:** Decides what part of the **current cell state** should be exposed as the hidden state (output) at this time step.
- **Formula:**
  - `o_t = σ(W_o · [h_(t-1), x_t] + b_o)`
  - `h_t = o_t * tanh(C_t)`
- **How it works:**
  - The sigmoid part decides which parts of the cell state are relevant to output right now.
  - The cell state is passed through a `tanh` (to normalize values between -1 and 1), then multiplied by the output gate's decision.
  - The result becomes the new hidden state `h_t`, which is used both as the output for this time step and passed to the next time step.
- **Example use case:** If the cell state contains information about a subject being singular or plural, the output gate might expose this information only when it's relevant to predicting the next word (e.g., a verb that needs to agree in number).

---

##### Summary Table

| Gate | Formula | Purpose | Acts On |
| --- | --- | --- | --- |
| Forget Gate | `f_t = σ(W_f · [h_(t-1), x_t] + b_f)` | Decide what to discard from memory | Previous cell state `C_(t-1)` |
| Input Gate | `i_t = σ(W_i · [h_(t-1), x_t] + b_i)` | Decide what new info to add to memory | Candidate values `C̃_t` |
| Output Gate | `o_t = σ(W_o · [h_(t-1), x_t] + b_o)` | Decide what to output from memory | Current cell state `C_t` |

- **Combined effect:** Together, these three gates give LSTM fine-grained control over what to remember, what to forget, and what to expose at each time step — this is precisely what allows it to maintain relevant information over long sequences while discarding irrelevant details, solving the long-term dependency problem faced by vanilla RNNs.

### Q39. How to identify exploding gradient issues in your model?

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

### Q40. What is a Transformer architecture, and what makes it different from CNNs and RNNs?

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
    - `Attention(Q, K, V) = softmax(QK^T / √d_k) * V`
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
  - Transformers scale better with sequence length (though attention has quadratic complexity in sequence length, `O(n²)`, which is its own limitation) and scale very well with more data and parameters (leading to large language models like GPT, BERT).

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

### Q41. What is the basic difference between LSTM and Transformers?

- **Processing style:** LSTM processes sequences step-by-step (sequential); Transformer processes the whole sequence in parallel using self-attention.
- **Dependency handling:** LSTM passes information through a chain of hidden/cell states, so long-range dependencies can weaken over distance; Transformer connects any two tokens directly via attention, regardless of distance.
- **Speed/training:** LSTM is slower to train (no parallelization across time steps); Transformer trains much faster on GPUs due to parallel computation.
- **Positional info:** LSTM has order built-in naturally (sequential processing); Transformer needs explicit positional encodings since attention has no inherent sense of order.
- **Scalability:** Transformers scale better with data/model size, which is why they power modern large language models; LSTMs don't scale as effectively.

### Q42. Diffusion Models

- **Core idea:** Generative models that learn to create data (e.g., images) by reversing a gradual noising process.
- **Forward process:** Data is progressively corrupted by adding small amounts of Gaussian noise over many steps (`T` steps), until it becomes pure random noise.
- **Reverse process:** A neural network (typically a U-Net) is trained to predict and remove the noise added at each step, gradually reconstructing data from pure noise back to a realistic sample.
- **Training objective:** The model learns to predict the noise component at each timestep (or equivalently the denoised sample), usually trained with a simple mean-squared-error loss between predicted and actual noise.
- **Sampling/Generation:** Starting from pure random noise, the model iteratively applies the learned reverse process step-by-step to produce a final, coherent sample (image, audio, etc.).
- **Key advantage:** Produces very high-quality, diverse outputs, often surpassing GANs in image fidelity and training stability (no adversarial training, no mode collapse).
- **Key drawback:** Slow sampling — generating a sample requires many sequential denoising steps (can be hundreds to thousands), though techniques like DDIM, distillation, and latent diffusion have significantly sped this up.
- **Popular examples:** DALL-E 2, Stable Diffusion, Imagen — all use diffusion models (often in a compressed "latent space" rather than raw pixel space, as in **Latent Diffusion Models**) for image generation.

### Q43. Why does Diffusion work better than autoregression?

- **Parallel generation vs. sequential:** Autoregressive (AR) models generate outputs one token/pixel at a time, conditioned on all previous ones — this is slow, and errors can compound (exposure bias). Diffusion models refine the *entire* output simultaneously at each denoising step, so errors aren't strictly compounding token-by-token.
- **Global consistency:** AR models build outputs left-to-right, so early mistakes propagate forward and are hard to fix. Diffusion iteratively refines the whole sample together, allowing global structure to be corrected across many steps rather than committed early and locked in.
- **No fixed generation order:** AR models impose an artificial sequential order (e.g., raster-scan order for images), which isn't natural for data like images where relationships are spatial, not linear. Diffusion has no such ordering constraint — it denoises all parts together.
- **Better sample diversity/quality trade-off:** Diffusion models tend to avoid the repetitive, low-diversity outputs that AR models can produce (especially with greedy/beam search), and avoid the blurriness common in older non-autoregressive one-shot generators.
- **Training stability:** Diffusion training reduces to a simple denoising/regression objective (predict noise), which is stable and doesn't suffer teacher-forcing/exposure bias problems the way AR sequence models can.
- **Trade-off to note:** Diffusion isn't strictly "better" in all respects — it's typically slower at *inference* (many denoising steps) compared to a single forward pass in some AR setups, though AR generation is also inherently sequential and slow per-token. For images/audio specifically, diffusion has empirically produced state-of-the-art fidelity, which is why it's preferred there; AR still dominates in domains like text (LLMs) where sequential token dependencies are natural and well-suited to the AR objective.

### Q44. Explain transfer learning and when to use it.

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

### Q45. What are Multimodal AI models, and how do they process different types of data?

> 🔁 _See also [10-multimodal-ai.md → Q1](../ai-engineering/10-multimodal-ai.md#q1-what-are-multimodal-ai-models-and-how-do-they-process-different-types-of-data) (separate Notion notes on the same question)._

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

### Q46. How do World Models work?

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

### Q47. How do Diffusion Language Models (DLMs) work?

> 🔁 _See also [01-llm-fundamentals.md → Q69](../ai-engineering/01-llm-fundamentals.md#q69-how-do-diffusion-language-models-dlms-work) (separate Notion notes on the same question)._

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

### Q48. Decoding Deep RL from Human Preferences

- **Core idea:** Train an RL agent using **human preferences** instead of manually designed reward functions.
- **Human feedback:** Humans compare two short behavior segments and choose the better one.
- **Reward model:** A neural network learns to predict human preferences and acts as a **learned reward function**.
- **RL optimization:** The agent optimizes the learned reward using reinforcement learning.
- **Iterative process:** Agent behavior → human comparisons → reward model → policy improvement.
- **Importance:** Reduces the need for **hand-crafted reward functions**.
- **Foundation:** An important early foundation for **RLHF, reward modeling, and AI alignment**.
- **Limitation:** Human feedback can be **expensive, noisy, and inconsistent**, and agents may exploit weaknesses in the learned reward model.

### Q49. Normalization in machine learning?

> 🔁 _See also [09-large-language-model.md → Q32](09-large-language-model.md#q32-what-is-normalization) (separate Notion notes on the same question)._

Normalization refers to techniques that rescale or reshape data (or model activations) so that different features or values are on a comparable scale. It matters because many ML algorithms are sensitive to the scale of inputs — without normalization, features with larger numeric ranges can dominate training, and optimization (like gradient descent) can converge slowly or unstably.

#### Why It Matters

- **Faster, more stable training** — gradient-based optimizers converge more smoothly when inputs are on similar scales
- **Fair feature contribution** — prevents features with large ranges (e.g., income in thousands) from overpowering smaller-range features (e.g., age in years)
- **Required by certain algorithms** — distance-based methods (KNN, SVM, K-means) and gradient descent-based methods (neural networks, logistic regression) are especially sensitive to scale
- **Numerical stability** — helps avoid overflow/underflow issues, especially in deep networks

#### Common Techniques for Input Data

**Min-Max Scaling (Normalization)**

Rescales values to a fixed range, usually \[0, 1\]:

```txt
x' = (x - min) / (max - min)
```

Sensitive to outliers since it depends on the extremes.

> ✍️ _Addition drafted during sync._

**Other common techniques for input data**

- **Standardization (Z-score):** `z = (x − μ) / σ`, which gives mean 0 and standard deviation 1. It is the default for linear models, SVMs, PCA and neural networks, and is less affected by outliers than Min-Max.
- **Robust scaling:** `x' = (x − median) / IQR`, for data with heavy outliers.
- **Max-abs scaling:** divide by the maximum absolute value. It keeps sparsity, so it suits sparse matrices such as TF-IDF.
- **Unit-vector (L2) normalization:** scale each *sample* (row) to unit length. It is common for text vectors and embeddings before cosine similarity.
- **Log / power transforms (Box-Cox, Yeo-Johnson):** reshape skewed distributions; often combined with one of the scalers above.

**Normalization of activations inside neural networks**

- **Batch Normalization:** normalizes each feature over the mini-batch, then applies a learnable scale and shift (γ, β). It uses running statistics at inference and is the standard choice in CNNs.
- **Layer Normalization:** normalizes over the features of each individual sample, so it does not depend on batch size. It is standard in Transformers and RNNs; RMSNorm is a cheaper variant used in many LLMs.
- **Group / Instance Normalization:** normalize over groups of channels, or per channel per sample. They are used when batches are small (detection, segmentation) or for style transfer.

**Production gotcha:** fit scaler statistics on the training set only. Persist them with the model (for example, a scikit-learn `Pipeline`) and apply the same transform at inference. Recomputing them on live data causes training/serving skew.

### Q50. Normalization vs Standardization

Both are **feature scaling techniques** used to put numerical features on comparable scales.

| Aspect | Normalization | Standardization |
| --- | --- | --- |
| **Definition** | Scales values to a fixed range | Rescales values based on mean and standard deviation |
| **Common range** | 0 to 1 | No fixed range |
| **Formula** | x' = (x − x\_min) / (x\_max − x\_min) | z = (x − μ) / σ |
| **Mean** | Not necessarily 0 | 0 |
| **Standard deviation** | Not necessarily 1 | 1 |
| **Uses** | Minimum and maximum | Mean and standard deviation |
| **Outlier sensitivity** | Highly sensitive | Less sensitive than Min-Max |
| **Common applications** | KNN, K-Means, neural networks | Logistic Regression, SVM, PCA, linear models |
| changes | the range of the data | distribution's scale and centers it around zero. |
