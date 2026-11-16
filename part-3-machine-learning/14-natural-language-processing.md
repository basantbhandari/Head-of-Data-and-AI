# Chapter 14. Natural Language Processing

_Part III — Machine Learning_

Text representation, classical NLP, BERT-family models and applied NLP systems.

This chapter contains **43 questions**.

## Contents

- **14.1** [What are the advantages of Transformers over traditional sequence-to-sequence models?](#141-what-are-the-advantages-of-transformers-over-traditional-sequence-to-sequence-models)
- **14.2** [What are the limitations of Transformers, and how can they be addressed?](#142-what-are-the-limitations-of-transformers-and-how-can-they-be-addressed)
- **14.3** [What is BERT, and how does it improve language understanding?](#143-what-is-bert-and-how-does-it-improve-language-understanding)
- **14.4** [How are Transformers trained (pre-training and fine-tuning)?](#144-how-are-transformers-trained-pre-training-and-fine-tuning)
- **14.5** [What is the difference between pre-training, fine-tuning and post-training?](#145-what-is-the-difference-between-pre-training-fine-tuning-and-post-training)
- **14.6** [Explain transfer learning in the context of Transformers.](#146-explain-transfer-learning-in-the-context-of-transformers)
- **14.7** [Describe the process of text generation using Transformer-based language models.](#147-describe-the-process-of-text-generation-using-transformer-based-language-models)
- **14.8** [What are Seq2Seq models?](#148-what-are-seq2seq-models)
- **14.9** [Compare N-gram models and deep learning models (trade-offs).](#149-compare-n-gram-models-and-deep-learning-models-trade-offs)
- **14.10** [What is the n-gram model?](#1410-what-is-the-n-gram-model)
- **14.11** [What is TF-IDF, and how does it differ from word embeddings?](#1411-what-is-tf-idf-and-how-does-it-differ-from-word-embeddings)
- **14.12** [What is Bag-of-Words?](#1412-what-is-bag-of-words)
- **14.13** [What is perplexity used for in NLP?](#1413-what-is-perplexity-used-for-in-nlp)
- **14.14** [What is stemming vs. lemmatization?](#1414-what-is-stemming-vs-lemmatization)
- **14.15** [What is Latent Semantic Indexing (LSI)?](#1415-what-is-latent-semantic-indexing-lsi)
- **14.16** [What is dependency parsing?](#1416-what-is-dependency-parsing)
- **14.17** [What are some approaches for text summarization?](#1417-what-are-some-approaches-for-text-summarization)
- **14.18** [What are word embeddings?](#1418-what-are-word-embeddings)
- **14.19** [What is Word2Vec?](#1419-what-is-word2vec)
- **14.20** [What is t-SNE, and how is it used for NLP?](#1420-what-is-t-sne-and-how-is-it-used-for-nlp)
- **14.21** [Explain ColBERT](#1421-explain-colbert)
- **14.22** [Text encoding in Natural Language Processing?](#1422-text-encoding-in-natural-language-processing)
- **14.23** [Why is an encoder-decoder model used for NLP?](#1423-why-is-an-encoder-decoder-model-used-for-nlp)
- **14.24** [Explain the BERT architecture.](#1424-explain-the-bert-architecture)
- **14.25** [What is next-sentence prediction in BERT?](#1425-what-is-next-sentence-prediction-in-bert)
- **14.26** [How does BERT work, and what are its pre-training objectives (MLM, NSP)?](#1426-how-does-bert-work-and-what-are-its-pre-training-objectives-mlm-nsp)
- **14.27** [How does RoBERTa differ from BERT?](#1427-how-does-roberta-differ-from-bert)
- **14.28** [How do you approach text summarization?](#1428-how-do-you-approach-text-summarization)
- **14.29** [How are knowledge graphs used with NLP?](#1429-how-are-knowledge-graphs-used-with-nlp)
- **14.30** [How would you solve entity resolution?](#1430-how-would-you-solve-entity-resolution)
- **14.31** [How does named entity recognition work, from rule-based systems to Transformer token classification?](#1431-how-does-named-entity-recognition-work-from-rule-based-systems-to-transformer-token-classification)
- **14.32** [How do Hidden Markov Models and Conditional Random Fields work for sequence labelling?](#1432-how-do-hidden-markov-models-and-conditional-random-fields-work-for-sequence-labelling)
- **14.33** [How do you choose an approach for a text classification problem, from TF-IDF baselines to fine-tuned encoders and LLMs?](#1433-how-do-you-choose-an-approach-for-a-text-classification-problem-from-tf-idf-baselines-to-fine-tuned-encoders-and-llms)
- **14.34** [What is aspect-based sentiment analysis, and why are negation, sarcasm and domain shift hard for sentiment models?](#1434-what-is-aspect-based-sentiment-analysis-and-why-are-negation-sarcasm-and-domain-shift-hard-for-sentiment-models)
- **14.35** [How do GloVe and fastText differ from Word2Vec, and how did contextual embeddings change NLP?](#1435-how-do-glove-and-fasttext-differ-from-word2vec-and-how-did-contextual-embeddings-change-nlp)
- **14.36** [How did Bahdanau and Luong attention improve sequence-to-sequence models?](#1436-how-did-bahdanau-and-luong-attention-improve-sequence-to-sequence-models)
- **14.37** [How does beam search work, and when should you use it instead of sampling?](#1437-how-does-beam-search-work-and-when-should-you-use-it-instead-of-sampling)
- **14.38** [How do topic models such as LDA and BERTopic work, and when would you use each?](#1438-how-do-topic-models-such-as-lda-and-bertopic-work-and-when-would-you-use-each)
- **14.39** [How do you build an information extraction system for entities and relations, and how have LLMs changed it?](#1439-how-do-you-build-an-information-extraction-system-for-entities-and-relations-and-how-have-llms-changed-it)
- **14.40** [What is coreference resolution, and why does it matter in NLP pipelines?](#1440-what-is-coreference-resolution-and-why-does-it-matter-in-nlp-pipelines)
- **14.41** [How do extractive, abstractive and open-domain question answering systems differ?](#1441-how-do-extractive-abstractive-and-open-domain-question-answering-systems-differ)
- **14.42** [How do weak supervision and LLM-assisted labelling help when labelled text data is scarce?](#1442-how-do-weak-supervision-and-llm-assisted-labelling-help-when-labelled-text-data-is-scarce)
- **14.43** [How do you detect near-duplicate documents at scale using MinHash and locality-sensitive hashing?](#1443-how-do-you-detect-near-duplicate-documents-at-scale-using-minhash-and-locality-sensitive-hashing)

---

### 14.1 What are the advantages of Transformers over traditional sequence-to-sequence models?

| Aspect | Transformers | Traditional Seq2Seq (RNN/LSTM/GRU) | Advantage of Transformers |
| --- | --- | --- | --- |
| **Parallelization** | Processes the sequence in parallel using self-attention during training | Processes tokens sequentially across time steps | **Much faster training and better GPU/TPU utilization** |
| **Long-range dependencies** | Directly connects any two tokens through self-attention | Information passes through a sequential hidden-state chain and can decay over long sequences | **Captures long-range dependencies more effectively** |
| **Context bottleneck** | Decoder can attend directly to relevant encoder representations | Classic encoder-decoder compresses input into a fixed-length context vector; RNN attention improves this but remains less flexible | **Avoids the traditional fixed-length bottleneck** |
| **Gradient flow** | Residual connections and layer normalization provide more stable optimization | Deep recurrent chains can suffer from vanishing/exploding gradients | **More stable training of deep models** |
| **Contextual representations** | Multi-head attention captures multiple relationships between tokens simultaneously | Primarily relies on sequential hidden-state representations | **Richer semantic, syntactic, and contextual representations** |
| **Scalability** | Scales effectively with large datasets, model sizes, and compute | Sequential computation makes large-scale scaling more difficult | **Better suited to large-scale models such as GPT, BERT, and T5** |
| **Training efficiency** | Highly parallelizable and optimized for modern accelerators | Limited by sequential dependencies | **Higher training throughput** |
| **Flexibility** | Attention architecture works across NLP, vision, audio, and multimodal tasks | More naturally suited to sequential/temporal data | **More flexible across modalities and tasks** |
| **Transfer learning** | Excellent for large-scale pretraining followed by fine-tuning or PEFT | Traditionally less effective for modern large-scale pretrained models | **Strong reuse of pretrained models across tasks** |
| **Overall performance** | Generally provides stronger performance for modern large-scale sequence tasks | Effective for many sequence tasks but less competitive at scale | **Better performance and scalability for modern AI systems** |

### 14.2 What are the limitations of Transformers, and how can they be addressed?

| Limitation | Explanation | Solutions |
| --- | --- | --- |
| **Quadratic complexity with sequence length** | Self-attention computes pairwise interactions between all tokens, resulting in **$O(n^2)$** time and memory complexity. This becomes expensive for long sequences. | Sparse attention, linear attention, FlashAttention, hierarchical/chunked processing |
| **High memory and compute requirements** | Large Transformer models require significant GPU/TPU resources for training and inference, making them expensive at scale. | Quantization, pruning, knowledge distillation, smaller models, Mixture-of-Experts |
| **Lack of inherent positional awareness** | Self-attention does not inherently understand token order, so positional information must be added. | RoPE, ALiBi, and improved positional encoding techniques |
| **Data hunger** | Transformers have relatively weak inductive biases and often require large datasets when trained from scratch. | Large-scale pretraining, transfer learning, fine-tuning, data augmentation, hybrid architectures |
| **Limited context length** | Practical context windows can limit tasks requiring extremely long context, such as whole-book reasoning. | Long-context architectures, memory mechanisms, chunking, RAG |
| **Sequential autoregressive decoding** | Although Transformer training is highly parallelizable, autoregressive generation produces one token at a time, increasing inference latency. | KV caching, speculative decoding, parallel/non-autoregressive decoding |
| **Interpretability challenges** | Transformers are complex, and attention weights alone do not always provide reliable explanations of model decisions. | Probing, attribution methods, Integrated Gradients, interpretability techniques |
| **Hallucination** | Generative Transformers can produce fluent but factually incorrect or unsupported information. | RAG, tool calling, grounding, verification, output evaluation |
| **Stale knowledge** | A pretrained model's knowledge is limited by its training data and may become outdated. | RAG, external search/tools, databases, periodic updates |
| **Bias and safety issues** | Models can learn biases from training data and may produce undesirable outputs. | Data filtering, safety training, guardrails, evaluation, human feedback |
| **High inference cost and latency** | Large models can be expensive and slow when serving many users or high request volumes. | Quantization, batching, KV caching, model compression, optimized inference engines |

### 14.3 What is BERT, and how does it improve language understanding?

##### BERT

**BERT (Bidirectional Encoder Representations from Transformers)** is a Transformer-based language model developed by Google. It uses the **Transformer encoder** to understand the meaning of words based on the words appearing **before and after them**.

| **Aspect** | **BERT** |
| --- | --- |
| **Full form** | Bidirectional Encoder Representations from Transformers |
| **Architecture** | Transformer **Encoder** |
| **Main purpose** | Understand the meaning and context of text |
| **Key feature** | **Bidirectional** context understanding |
| **Pre-training** | Mainly trained using **Masked Language Modeling (MLM)** and originally **Next Sentence Prediction (NSP)** |
| **Context understanding** | Looks at both left and right context |
| **Example** | `"I went to the bank to deposit money"` → understands **bank** as a financial institution |
| **Representation** | Produces contextual representations of tokens |
| **Fine-tuning** | Can be adapted to specific NLP tasks |
| **Applications** | Sentiment analysis, text classification, NER, question answering, search |

##### How BERT Improves Language Understanding

| **Feature** | **How it helps** |
| --- | --- |
| **Bidirectional attention** | Understands a word using context from both directions |
| **Self-attention** | Captures relationships between words, even when they are far apart |
| **Masked Language Modeling** | Learns to infer missing words from surrounding context |
| **Contextual representations** | The same word can have different representations in different sentences |
| **Transfer learning** | A pre-trained BERT can be fine-tuned for many NLP tasks |

##### Example

Consider:

> **"The bank approved my loan."**

and

> **"The boat reached the bank of the river."**

BERT uses the surrounding words to understand that **"bank" has different meanings** in these two sentences.

### 14.4 How are Transformers trained (pre-training and fine-tuning)?

#### How Transformers Are Trained — Pre-training

**Pre-training is the first major stage where a Transformer learns general language patterns, knowledge, and relationships from a very large dataset.**

| **Step** | **What happens** | **Example** |
| --- | --- | --- |
| **1. Collect data** | Gather a very large text dataset | Books, websites, articles, code |
| **2. Clean data** | Remove duplicates, corrupted text, and low-quality content | Data filtering and deduplication |
| **3. Tokenization** | Convert text into tokens | `"I love AI"` → tokens |
| **4. Create training examples** | Prepare input and target tokens according to the training objective | GPT: predict the next token |
| **5. Embeddings** | Convert tokens into numerical vectors | Token IDs → embedding vectors |
| **6. Transformer processing** | Tokens pass through attention and feed-forward layers | Self-attention learns relationships between tokens |
| **7. Predict tokens** | Model predicts the target/next token | `"The capital of Nepal is"` → `"Kathmandu"` |
| **8. Calculate loss** | Compare predicted tokens with correct tokens | Cross-entropy loss |
| **9. Backpropagation** | Calculate gradients showing how weights should change | Gradient computation |
| **10. Update weights** | Optimizer adjusts model parameters | Adam/AdamW |
| **11. Repeat** | Repeat across huge amounts of training data for many iterations | Model gradually improves |
| **12. Final pre-trained model** | The model learns general language representations and knowledge | A base GPT-style or BERT-style model |

#### Fine-tuning a Pre-trained Transformer

**Fine-tuning means taking an already pre-trained Transformer and continuing its training on a smaller, task-specific dataset.**

| **Step** | **What happens** | **Example** |
| --- | --- | --- |
| **1. Start with pre-trained model** | Load a model that already learned general language patterns | BERT, T5, GPT |
| **2. Prepare task-specific data** | Collect and label data for the target task | Customer reviews → Positive/Negative |
| **3. Tokenize data** | Convert the text into tokens using the model's tokenizer | `"This product is good"` → tokens |
| **4. Add task-specific output** | Add/adapt a suitable output layer or task head when needed | Classification head for sentiment |
| **5. Forward pass** | Input passes through the Transformer | Attention + feed-forward layers |
| **6. Make prediction** | Model produces an output for the task | Positive = 0.92 |
| **7. Calculate loss** | Compare prediction with the correct answer | Predicted: Positive; Actual: Positive |
| **8. Backpropagation** | Calculate gradients from the loss | Determine which parameters need adjustment |
| **9. Update parameters** | Optimizer updates model weights | AdamW updates the parameters |
| **10. Repeat** | Train for several epochs/batches | Continue until performance is good |
| **11. Evaluate** | Test on unseen validation/test data | Accuracy, F1, BLEU, etc. |
| **12. Final model** | Obtain a model specialized for the target task | Fine-tuned sentiment classifier |

### 14.5 What is the difference between pre-training, fine-tuning and post-training?

| **Aspect** | **Pre-training** | **Fine-tuning** | **Post-training** |
| --- | --- | --- | --- |
| **Main goal** | Learn general language/world knowledge | Adapt the model to a specific task or domain | Improve the model's behavior, instruction following, reasoning, safety, and usefulness |
| **When it happens** | First major training stage | After pre-training; can be part of post-training | After pre-training, usually before deployment |
| **Data** | Huge, diverse datasets | Smaller, task/domain-specific datasets | Curated instruction, preference, reasoning, safety, and feedback data |
| **What model learns** | Language patterns, facts, syntax, general representations | Specific task/domain behavior | How to respond to users and follow desired behaviors |
| **Typical objective** | Next-token prediction or masked-token prediction | Task-specific loss or supervised next-token prediction | Instruction tuning, preference optimization, RL-based objectives, etc. |
| **Human feedback** | Usually minimal/directly absent | Usually not required | Often important |
| **Model weights** | Trained extensively, often from random initialization | Existing pre-trained weights are adapted | Existing model is further optimized |
| **Dataset size** | Very large | Relatively small | Usually smaller and highly curated |
| **Compute cost** | **Very high** | **Lower** | **Moderate to high**, depending on method |
| **Example** | Train a GPT-style model on large-scale text | Fine-tune a model for medical text classification | Train an LLM to follow instructions and prefer helpful, safe answers |
| **Common methods** | Self-supervised learning | Supervised fine-tuning (SFT), domain fine-tuning, LoRA/QLoRA | SFT, RLHF, DPO, RLAIF, preference optimization, reasoning/post-training |
| **Simple meaning** | **Teach the model language and knowledge** | **Teach it a specific task** | **Teach it how to behave and perform better for users** |

### 14.6 Explain transfer learning in the context of Transformers.

#### Transfer Learning in Transformers

**Transfer learning** means taking a Transformer model that has already been **pre-trained on a large dataset** and adapting it to a specific task instead of training a model from scratch.

| **Step** | **Process** | **Example** |
| --- | --- | --- |
| **1. Pre-training** | Train a Transformer on a large text dataset to learn general language patterns | BERT learns from large amounts of text |
| **2. Learned knowledge** | The model learns grammar, vocabulary, relationships, and language patterns | Understands relationships between words |
| **3. Select pre-trained model** | Choose a model suitable for the task | BERT, T5, GPT, etc. |
| **4. Fine-tuning** | Train the pre-trained model on a smaller task-specific dataset | Fine-tune BERT for sentiment analysis |
| **5. Task-specific output** | Add or adapt an output layer for the target task | Positive/Negative classification |
| **6. Inference** | Use the adapted model on new data | Classify new customer reviews |

#### Example

Instead of training a sentiment-analysis Transformer from scratch:

**Large general dataset → Pre-trained BERT → Fine-tune on review data → Sentiment classifier**

#### Why Transfer Learning Is Useful

| **Benefit** | **Explanation** |
| --- | --- |
| **Less training data** | Requires much less task-specific data |
| **Lower training cost** | Avoids training the entire model from scratch |
| **Faster development** | Start with an already-trained model |
| **Better performance** | Leverages knowledge learned from large datasets |
| **Adaptability** | One pre-trained model can be adapted to many different tasks |

### 14.7 Describe the process of text generation using Transformer-based language models.

#### Text Generation Using Transformer-Based Language Models

| **Step** | **Process** | **What Happens** |
| --- | --- | --- |
| **1. Input text** | User provides a prompt | Example: `"Machine learning is"` |
| **2. Tokenization** | Text → tokens | The prompt is split into tokens such as `["Machine", "learning", "is"]` |
| **3. Token embeddings** | Tokens → vectors | Each token is converted into a numerical vector |
| **4. Positional information** | Add position information | The model knows the order/position of each token |
| **5. Transformer processing** | Self-attention + neural network layers | The model analyzes relationships between tokens and builds contextual representations |
| **6. Predict next token** | Generate probability distribution | The model calculates probabilities for possible next tokens, e.g. `["important": 0.4, "useful": 0.3, ...]` |
| **7. Select next token** | Choose a token | The next token is selected using greedy decoding, sampling, temperature, top-k, top-p, etc. |
| **8. Add token to input** | Extend the sequence | The selected token is added to the existing text |
| **9. Repeat** | Predict the next token again | The model continues generating tokens one at a time |
| **10. Stop** | End generation | Generation stops when an end token, length limit, or stopping condition is reached |

#### Simple Flow

**Prompt → Tokenization → Embeddings → Transformer → Next-token probabilities → Token selection → Add token → Repeat → Generated text**

#### Example

**Prompt:**

> `The capital of Nepal is`

The model may predict:

> `Kathmandu`

Then the sequence becomes:

> `The capital of Nepal is Kathmandu`

The model can then predict the next token and continue until the generation stops.

### 14.8 What are Seq2Seq models?

**Seq2Seq (Sequence-to-Sequence) models** are neural network models that **convert one sequence into another sequence**, where the input and output can have different lengths.

| **Aspect** | **Explanation** |
| --- | --- |
| **Definition** | A model that maps an input sequence to an output sequence |
| **Architecture** | Traditionally consists of an **Encoder** and a **Decoder** |
| **Encoder** | Reads and represents the input sequence |
| **Decoder** | Generates the output sequence step by step |
| **Input/Output** | Input and output can have different lengths |
| **Example** | English: `"How are you?"` → French: `"Comment allez-vous ?"` |
| **Traditional models** | RNN, LSTM, GRU-based Seq2Seq |
| **Modern approach** | Transformers are widely used for Seq2Seq tasks |
| **Attention** | Helps the decoder focus on relevant parts of the input |
| **Applications** | Machine translation, text summarization, chatbot systems, question answering |
| **Main advantage** | Can handle variable-length input and output sequences |
| **Main limitation** | Traditional RNN-based models struggle with long sequences; Transformers address this much better |

#### Simple Example

**Input:**

`"I love machine learning"`

**Seq2Seq model:**

`Input sequence → Encoder → Decoder → Output sequence`

**Output:**

`"J'aime l'apprentissage automatique"`

### 14.9 Compare N-gram models and deep learning models (trade-offs).

| **Aspect** | **N-gram Models** | **Deep Learning Models** |
| --- | --- | --- |
| **Basic idea** | Predicts text using a fixed number of previous words | Uses neural networks to learn complex language patterns |
| **Context** | **Short context** — limited to $N$ words | Can capture **longer context**, especially Transformers |
| **Representation** | Word sequences/counts and probabilities | Learned numerical representations (embeddings) |
| **Learning method** | Statistical frequency/probability | Neural network training on large datasets |
| **Semantic understanding** | Very limited | Much better at capturing meaning and relationships |
| **Word order** | Considers local word order | Can model complex word relationships |
| **Long-range dependencies** | Poor | Much better |
| **Data requirement** | Relatively smaller datasets can work | Usually requires large datasets |
| **Computational cost** | Low | High, especially during training |
| **Training** | Simple counting/statistical estimation | Computationally intensive neural network training |
| **Handling unseen words** | Poor; suffers from data sparsity | Better, especially with subword/token-based models |
| **Examples** | Unigram, Bigram, Trigram | RNN, LSTM, GRU, Transformer, BERT, GPT |
| **Applications** | Basic autocomplete, spelling correction, simple language modeling | Translation, chatbots, text generation, QA, modern LLMs |
| **Main limitation** | Cannot effectively understand long-range context | Requires significant data and computational resources |
| **Overall** | **Simple and fast but limited** | **Powerful and context-aware but more expensive** |

### 14.10 What is the n-gram model?

#### N-gram Model

An **n-gram model** is an NLP language model that predicts the **next word/token based on the previous $N-1$ words**.

| **Aspect** | **Explanation** |
| --- | --- |
| **Definition** | A model that uses sequences of **N words/tokens** to predict or analyze text |
| **N** | Number of words/tokens in the sequence |
| **Unigram ($N=1$)** | Considers one word at a time |
| **Bigram ($N=2$)** | Considers two consecutive words |
| **Trigram ($N=3$)** | Considers three consecutive words |
| **Example sentence** | `"I love machine learning"` |
| **Bigrams** | `"I love"`, `"love machine"`, `"machine learning"` |
| **Trigrams** | `"I love machine"`, `"love machine learning"` |
| **Main purpose** | Predict the next word based on previous words |
| **Example** | `"I want to"` → predicts `"learn"` |
| **Advantages** | Simple, fast, easy to implement |
| **Limitations** | Struggles with long-range context and requires large amounts of data for larger $N$ |
| **Applications** | Text prediction, speech recognition, spelling correction, autocomplete, language modeling |

### 14.11 What is TF-IDF, and how does it differ from word embeddings?

#### What is TF-IDF (Term Frequency-Inverse Document Frequency)?

- **Core idea:** A statistical measure used to evaluate how important a word is to a document within a collection (corpus), balancing how often a word appears in a document against how common it is across all documents.
- **Term Frequency (TF):** Measures how frequently a word appears in a document.

  $$
  \text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total terms in } d}
  $$

- **Inverse Document Frequency (IDF):** Measures how rare/common a word is across the entire corpus — words appearing in many documents get a lower score (less informative).

  $$
  \text{IDF}(t) = \log \frac{\text{Total number of documents}}{\text{Number of documents containing term } t}
  $$

- **TF-IDF score:** $\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)$
  - High score → word is frequent in this document but rare across the corpus (likely important/distinctive).
  - Low score → word is either rare in the document or very common across all documents (e.g., "the," "is" — stopwords naturally get low scores).
- **Output:** Produces a sparse vector representation of a document, where each dimension corresponds to a word in the vocabulary, and the value is its TF-IDF score.

#### How TF-IDF Differs from Word Embeddings

| **Aspect** | **TF-IDF** | **Word Embeddings** |
| --- | --- | --- |
| **Basic idea** | Measures how important a word is in a document | Represents words as vectors that capture meaning |
| **Representation** | Sparse, high-dimensional vector | Dense, low-dimensional vector |
| **Vector size** | Vocabulary size | Fixed size, e.g. 100, 300, 768+ |
| **Semantic meaning** | Does not capture meaning well | Captures semantic relationships |
| **Word similarity** | `good` and `great` are separate features | `good` and `great` can have similar vectors |
| **Context** | Ignores word order and context | Can capture context, especially BERT/GPT-style embeddings |
| **How generated** | Calculated using word frequency statistics | Learned from large text datasets |
| **Training required** | No | Usually yes |
| **Computational cost** | Low and fast | Higher |
| **Interpretability** | High — easy to understand word importance | Lower — vector dimensions are not directly interpretable |
| **Typical use cases** | Search, keyword extraction, basic text classification | Semantic search, similarity, QA, classification, NLP applications |
| **Examples** | TF-IDF | Word2Vec, GloVe, FastText, BERT |
| **Key question** | **"How important is this word?"** | **"What does this word mean and how is it related to other words?"** |

### 14.12 What is Bag-of-Words?

**Bag-of-Words is a simple NLP technique that represents text as a collection of words and their frequencies, ignoring grammar and word order.**

| Aspect | Explanation |
| --- | --- |
| **Purpose** | Convert text into numerical features that ML models can process |
| **Representation** | A vector of word counts or frequencies |
| **Word order** | Ignored |
| **Grammar** | Ignored |
| **Meaning/context** | Not captured |
| **Common use** | Text classification, spam detection, sentiment analysis |
| **Advantage** | Simple and computationally efficient |
| **Disadvantage** | Loses word order and semantic/contextual information |

#### Example

Suppose we have two sentences:

- **"I love machine learning"**
- **"I love deep learning"**

Vocabulary:

`[I, love, machine, learning, deep]`

| Sentence | I | love | machine | learning | deep |
| --- | --- | --- | --- | --- | --- |
| I love machine learning | 1 | 1 | 1 | 1 | 0 |
| I love deep learning | 1 | 1 | 0 | 1 | 1 |

The text is converted into **numerical vectors**, which can then be used by machine-learning algorithms.

### 14.13 What is perplexity used for in NLP?

**Perplexity (PPL)** is a metric used to measure **how well a language model predicts a sequence of words/tokens**.

| Aspect | Explanation |
| --- | --- |
| **Purpose** | Evaluates how well a language model predicts text |
| **What it measures** | The model's uncertainty when predicting the next token |
| **Lower perplexity** | Better prediction / less uncertainty |
| **Higher perplexity** | Worse prediction / more uncertainty |
| **Used for** | Comparing and evaluating language models |
| **Common in** | LLMs, language modeling, text generation |
| **Example** | A model with PPL = 20 is generally better than one with PPL = 50 on the same dataset |

**Example:**

> "The cat is sitting on the \_\_\_."

If the model assigns a high probability to **"mat"**, it has low uncertainty → **lower perplexity**.

If it assigns similar probabilities to many unrelated words, it has high uncertainty → **higher perplexity**.

#### Interview Answer

> **Perplexity is an evaluation metric for language models that measures how well a model predicts the next token. Lower perplexity indicates that the model is less uncertain and generally predicts the text better.**

### 14.14 What is stemming vs. lemmatization?

| **Aspect** | **Stemming** | **Lemmatization** |
| --- | --- | --- |
| **Core idea** | Reduces words by chopping off prefixes/suffixes | Reduces words to their actual dictionary/base form |
| **Approach** | Rule-based suffix/prefix stripping | Dictionary + morphological analysis |
| **How it works** | Applies simple rules to modify the word | Uses vocabulary, morphology, and often Part-of-Speech (POS) context |
| **Output** | May not be a real word | Usually a valid dictionary word |
| **Meaning awareness** | No understanding of word meaning | Can consider word meaning and grammatical role |
| **Accuracy** | Lower | Higher |
| **Speed** | Faster | Slower |
| **Computational cost** | Low | Higher |
| **Example 1** | `running → run` | `running → run` |
| **Example 2** | `studies → studi` | `studies → study` |
| **Example 3** | `better → better` | `better → good` |
| **Common algorithms/tools** | Porter Stemmer, Snowball Stemmer, Lancaster Stemmer | WordNet Lemmatizer, spaCy, Stanford CoreNLP |
| **Main advantage** | Fast and simple | More linguistically accurate |
| **Main disadvantage** | Can produce meaningless words or miss relationships | More computationally expensive |
| **Best suited for** | Large-scale processing where speed is important | Tasks where linguistic accuracy and meaning matter |
| **Typical use cases** | Search engines, information retrieval, text indexing | Sentiment analysis, chatbots, NLP classification |

#### Stemming vs Lemmatization

Both **stemming** and **lemmatization** are NLP techniques used to reduce words to their **base or root form**, but they work differently.

| Aspect | Stemming | Lemmatization |
| --- | --- | --- |
| **Approach** | Removes prefixes/suffixes using rules | Uses vocabulary and linguistic analysis |
| **Output** | May produce an invalid word | Produces a valid dictionary word |
| **Accuracy** | Generally lower | Generally higher |
| **Speed** | Faster | Slower |
| **Context** | Usually does not consider context | Can consider context and part of speech |
| **Example** | `studies → studi` | `studies → study` |
| **Another example** | `running → run` | `running → run` |

#### Example

**Stemming:**

> `connection`, `connected`, `connecting` → `connect`

It simply removes word endings according to predefined rules.

**Lemmatization:**

> `better` → `good`

It uses linguistic knowledge to identify the correct **lemma** (base dictionary form).

### 14.15 What is Latent Semantic Indexing (LSI)?

**Latent Semantic Indexing (LSI)** is an NLP and information-retrieval technique that uses **Singular Value Decomposition (SVD)** to reduce a large **term-document matrix** into a smaller representation that captures the underlying **semantic relationships between words and documents**.

| Aspect | LSI |
| --- | --- |
| **Purpose** | Discover hidden/latent semantic relationships in text |
| **Input** | Term-document matrix, often based on TF-IDF |
| **Core technique** | Singular Value Decomposition (SVD) |
| **Dimensionality** | Reduces a high-dimensional matrix to fewer dimensions |
| **Semantic similarity** | Helps identify documents with related meanings even when they don't use exactly the same words |
| **Example** | “car” and “automobile” can be associated through their document usage patterns |
| **Applications** | Document retrieval, search, document similarity, clustering, topic discovery |
| **Main limitation** | Traditional LSI does not understand word meaning/context as deeply as modern embeddings or Transformers |

#### How LSI works

**Documents → Term-Document Matrix → TF-IDF → SVD → Reduced Latent-Semantic Representation**

For a term-document matrix $A$, SVD decomposes it as:

$$
A = U \Sigma V^T
$$

LSI keeps only the most important $k$ dimensions:

$$
A \approx U_k \Sigma_k V_k^T
$$

This creates a lower-dimensional **latent semantic space**.

#### Simple Example

Suppose documents contain:

- **D1:** “car engine vehicle”
- **D2:** “automobile engine vehicle”
- **D3:** “computer software hardware”

Even though **“car”** and **“automobile”** are different words, their occurrence patterns across documents can cause LSI to represent them as related concepts.

### 14.16 What is dependency parsing?

- **Core idea:** A syntactic analysis technique that identifies **grammatical relationships between words** in a sentence. It represents these relationships as **head–dependent pairs**, typically forming a directed dependency tree rather than a phrase-structure tree.
- **Key concepts:**
  - **Head:** The word that governs or determines the grammatical role of another word.
  - **Dependent:** A word that modifies or is grammatically related to the head.
  - **Dependency relation:** A label describing the relationship between the head and dependent, such as subject, object, modifier, or determiner.
- **Structure:** The output is usually a **dependency tree** with a single **root** and labeled directed edges connecting all other words. The root is often the main verb, although this depends on the sentence.
- **Example:**<br>Sentence: *“The dog chased the cat.”*
  - **“chased”** → root
  - **“dog”** → nominal subject (`nsubj`) of “chased”
  - **“cat”** → object (`obj`) of “chased”
  - **“The”** → determiner (`det`) of “dog”
  - **“the”** → determiner (`det`) of “cat”
- **Common Universal Dependencies labels:**
  - `nsubj` → nominal subject
  - `obj` → object
  - `amod` → adjectival modifier
  - `advmod` → adverbial modifier
  - `det` → determiner
  - `case` → case marker, often associated with prepositions
  - `conj` → conjunction
  - `aux` → auxiliary
  - `root` → root of the sentence

#### Dependency Parsing vs. Constituency Parsing

| Dependency Parsing | Constituency Parsing |
| --- | --- |
| Focuses on **word-to-word relationships** | Focuses on **phrase structure** |
| Uses **head–dependent relationships** | Uses **nested constituents/phrases** |
| No explicit phrase nodes are required | Explicit phrase nodes such as NP and VP |
| Output is a dependency tree | Output is a constituency/phrase-structure tree |
| Example: `dog → chased` | Example: `[The dog] [chased the cat]` |

#### How Dependency Parsing Is Computed

- **Traditional approaches:** Rule-based and statistical parsing methods.
- **Transition-based parsing:** Builds the dependency tree incrementally using actions such as shift and reduce/arc operations. Modern systems commonly use neural networks to predict these actions.
- **Graph-based parsing:** Scores possible head–dependent relationships and searches for the highest-scoring valid dependency tree.
- **Modern NLP:** Neural and Transformer-based models are commonly trained on manually annotated treebanks, such as **Universal Dependencies**.

#### Applications

- **Information extraction:** Extract subject–verb–object relationships and structured facts.
- **Question answering:** Understand grammatical roles to identify relevant entities and relationships.
- **Machine translation:** Help preserve grammatical relationships across different word orders.
- **Sentiment analysis:** Handle relationships such as negation, e.g., *“not good.”*
- **Relation extraction:** Identify relationships between entities based on their syntactic connections.
- **Text understanding:** Provide syntactic information for downstream NLP tasks.

### 14.17 What are some approaches for text summarization?

Text summarization approaches can be broadly divided into **extractive** and **abstractive** methods, with modern systems often combining both.

| Approach | How it works | Common techniques / models | Advantages | Limitations |
| --- | --- | --- | --- | --- |
| **Extractive summarization** | Selects the most important **sentences or phrases directly from the original text** | TF-IDF, TextRank, LexRank, graph-based methods | Simple, fast, preserves original wording, less likely to introduce new facts | May produce disconnected or repetitive summaries |
| **Abstractive summarization** | **Generates new text** that captures the main meaning rather than simply copying sentences | Seq2Seq, LSTM, Transformer, BART, T5, LLMs | More natural, concise, and human-like summaries | More computationally expensive and can hallucinate information |
| **Hybrid summarization** | Combines extraction and generation | Extract important content first, then use a generative model to rewrite it | Can improve relevance while reducing generation cost | More complex pipeline |
| **Query-focused summarization** | Generates a summary focused on information relevant to a **specific query or topic** | Retrieval + summarization, attention-based models, RAG | Useful when users need only specific information | Requires a well-defined query |
| **Single-document summarization** | Summarizes one document | TextRank, BART, T5, LLMs | Straightforward | Limited to information in one document |
| **Multi-document summarization** | Combines information from multiple documents into one summary | Clustering, retrieval, Transformer/LLM-based methods | Removes redundancy and combines information across sources | More difficult due to redundancy and conflicting information |

#### Extractive vs. Abstractive

| Feature | Extractive | Abstractive |
| --- | --- | --- |
| Output | Sentences/phrases from source | Newly generated text |
| Rewriting | No | Yes |
| Naturalness | Usually lower | Usually higher |
| Hallucination risk | Lower | Higher |
| Computational cost | Lower | Higher |
| Example methods | TextRank, LexRank | BART, T5, LLMs |

### 14.18 What are word embeddings?

**Word embeddings** are **dense, low-dimensional vector representations of words that capture their semantic and syntactic relationships**.

Instead of representing words as unrelated one-hot vectors, embeddings place words with similar meanings **close together in a vector space**.

| Aspect | Description |
| --- | --- |
| **Representation** | Dense numerical vector |
| **Purpose** | Capture word meaning and relationships |
| **Typical size** | 100–300+ dimensions for traditional embeddings |
| **Learning principle** | Words appearing in similar contexts tend to have similar meanings |
| **Similarity** | Often measured using cosine similarity |
| **Examples** | Word2Vec, GloVe, FastText, BERT, GPT |

#### Example

Words such as **"king"** and **"queen"** will generally have similar vectors, while **"king"** and **"banana"** will be much less similar.

Some embeddings can even capture relationships through vector arithmetic:

$$
\text{king} - \text{man} + \text{woman} \approx \text{queen}
$$

#### Static vs Contextual Embeddings

- **Static:** Word2Vec, GloVe, FastText — one fixed vector for each word.
- **Contextual:** BERT, GPT, ELMo — the representation changes depending on the surrounding context.

For example, **"bank"** has different meanings in:

- "I deposited money in the **bank**."
- "We sat beside the river **bank**."

Contextual embeddings can represent these meanings differently.

### 14.19 What is Word2Vec?

**Word2Vec** is a **neural-network-based word embedding technique** that learns dense vector representations of words from their surrounding context.

Its main idea is the **distributional hypothesis**:

> Words that appear in similar contexts tend to have similar meanings.

#### How Word2Vec Works

Word2Vec has two main architectures:

| Approach | Description | Example |
| --- | --- | --- |
| **CBOW (Continuous Bag of Words)** | Predicts a target word from its surrounding context words | `The cat ___ on the mat` → predicts `sat` |
| **Skip-gram** | Predicts surrounding context words from a target word | `cat` → predicts nearby words such as `the`, `sat`, `mat` |

#### Key Characteristics

- Produces **dense, low-dimensional vectors**.
- Learns semantic and syntactic relationships from large text corpora.
- Similar words tend to have similar vectors.
- Can capture relationships through vector arithmetic, such as: $\text{king} - \text{man} + \text{woman} \approx \text{queen}$
- Word2Vec embeddings are **static**, meaning each word normally has one vector regardless of its context.

#### Limitation

Word2Vec cannot properly distinguish different meanings of the same word.

For example:

- **"bank"** in "river bank"
- **"bank"** in "bank account"

Both receive essentially the **same embedding**.

Modern contextual models such as **BERT and GPT** address this by producing representations that depend on context.

### 14.20 What is t-SNE, and how is it used for NLP?

**t-SNE (t-distributed Stochastic Neighbor Embedding)** is a **dimensionality-reduction technique used to visualize high-dimensional data in 2D or 3D** while preserving local similarities between data points.

#### How t-SNE Works

1. Starts with high-dimensional vectors, such as **Word2Vec or BERT embeddings**.
2. Measures which data points are close to each other in the original high-dimensional space.
3. Maps them into a low-dimensional space, usually **2D**.
4. Optimizes the 2D representation so that similar points remain close together.

#### Use of t-SNE in NLP

| Application | Use |
| --- | --- |
| **Word embeddings** | Visualize Word2Vec, GloVe, or FastText vectors |
| **Sentence embeddings** | Explore relationships between sentences/documents |
| **Document clustering** | Identify groups of semantically similar documents |
| **Semantic analysis** | See whether words with related meanings form clusters |
| **Model analysis** | Explore how embeddings from NLP models are distributed |

#### Example

Suppose we have thousands of Word2Vec vectors:

`king, queen, prince, car, bus, truck, apple, banana...`

t-SNE can reduce these high-dimensional vectors to 2D so we can visualize whether related words form clusters, such as:

- **Royalty:** king, queen, prince
- **Vehicles:** car, bus, truck
- **Fruits:** apple, banana

The visualization helps us **explore patterns**, but the 2D distances should not be interpreted as an exact representation of the original embedding space.

#### Important Limitations

- Mainly used for **visualization**, not as a general-purpose feature transformation.
- Results can depend on parameters such as **perplexity** and random initialization.
- Can be computationally expensive for very large datasets.
- Primarily preserves **local structure**, so global distances between clusters may be misleading.

### 14.21 Explain ColBERT

- **ColBERT (Contextualized Late Interaction over BERT)** is a neural **information retrieval** model used to find relevant documents for a query.
- It uses **BERT to encode the query and documents separately**, producing an embedding for **each token** rather than one embedding for the entire text.
- **Documents are encoded and stored in advance**, making retrieval faster.
- At search time, ColBERT compares each **query token** with all document tokens and finds its **best match**.
- It uses **MaxSim (Maximum Similarity)** to combine these token-level matches into an overall relevance score.
- This is called **late interaction** because the query and document interact only **after they have already been independently encoded**.
- **Compared with other retrieval methods:**
  - **Bi-encoder:** one vector for the entire query/document → very fast, but less precise.
  - **Cross-encoder:** query and document processed together → highly accurate, but slow.
  - **ColBERT:** token-level representations + late interaction → a good balance between **speed and accuracy**.

### 14.22 Text encoding in Natural Language Processing?

**Text encoding** is the process of **converting human-readable text into a numerical representation that machine-learning models can process**.

Since ML models work with numbers rather than raw text, NLP systems first encode text into numerical forms such as tokens, vectors, or embeddings.

#### Common Text Encoding Techniques

| Technique | Description | Example |
| --- | --- | --- |
| **One-Hot Encoding** | Represents each word as a binary vector | `cat → [1,0,0,0]` |
| **Bag-of-Words (BoW)** | Represents text using word occurrence/counts | `[2,1,0,3]` |
| **TF-IDF** | Represents words based on their importance within documents | Higher value = more important word |
| **Word Embeddings** | Represents words as dense vectors capturing semantic relationships | Word2Vec, GloVe, FastText |
| **Contextual Embeddings** | Generates representations based on surrounding context | BERT, GPT |
| **Token Encoding** | Converts text into token IDs that a model's tokenizer understands | `"Hello world" → [15496, 995]` |

#### Example

Suppose we have:

> **"I love machine learning."**

A tokenizer might convert it into:

`["I", "love", "machine", "learning"]`

Then each token is mapped to a numerical **token ID**:

`[40, 1842, 4521, 7890]`

A Transformer can then convert these token representations into **embeddings** and process them using its neural network layers.

### 14.23 Why is an encoder-decoder model used for NLP?

#### Why Use an Encoder-Decoder Model for NLP?

- **Encoder:** Converts the input sequence into a meaningful **contextual representation**.
- **Decoder:** Generates the output sequence **token by token** based on that representation.
- **Handles variable-length sequences:** Input and output can have different lengths.
- **Captures context:** The encoder understands relationships between words in the input.
- **Suitable for sequence-to-sequence tasks:** Such as **machine translation, text summarization, and question answering**.
- **Example:**`English sentence → Encoder → Context representation → Decoder → French sentence`

### 14.24 Explain the BERT architecture.

**BERT** (Bidirectional Encoder Representations from Transformers, Google 2018) is an **encoder-only Transformer** pre-trained to produce deep bidirectional contextual embeddings.

**Architecture**

- **Input:** WordPiece tokens (30k vocab) with special tokens: `[CLS]` at the start (its final vector summarizes the sequence) and `[SEP]` between/after segments.
- **Embeddings:** token embedding + **learned absolute position embedding** (max 512 positions) + **segment embedding** (sentence A or B), summed and layer-normalized.
- **Stack of Transformer encoder blocks**, each with multi-head **bidirectional** self-attention (no causal mask, every token attends to every token) and a feed-forward network (GELU), with residual connections and Post-LayerNorm.
- **Sizes:** BERT-Base: 12 layers, hidden 768, 12 heads, ~110M parameters. BERT-Large: 24 layers, hidden 1024, 16 heads, ~340M.

**Pre-training objectives:**

- **Masked Language Modeling (MLM):** mask 15% of tokens (80% `[MASK]`, 10% random, 10% unchanged) and predict them using both left and right context.
- **Next Sentence Prediction (NSP):** predict whether sentence B follows A.

**Fine-tuning:** add a small head: `[CLS]` vector → classifier for sentence classification; per-token outputs for NER; start/end span prediction for extractive QA.

**Strengths and limits:** excellent for understanding tasks (classification, NER, retrieval, reranking as a cross-encoder); bidirectionality makes it unsuited to text generation; 512-token limit; quadratic attention. Successors: RoBERTa, ALBERT, DistilBERT, DeBERTa, ModernBERT. In production, BERT-family models remain the workhorses for embeddings, classifiers and rerankers because they're cheap compared with LLMs.

### 14.25 What is next-sentence prediction in BERT?

**Next-Sentence Prediction (NSP)** is BERT's second pre-training task (alongside masked language modeling). Its goal was to teach the model relationships **between** sentences, useful for tasks like question answering and natural language inference.

**How it works:**

- Build input pairs `[CLS] sentence A [SEP] sentence B [SEP]`, with segment embeddings marking A and B.
- 50% of the time B is the **actual next sentence** from the corpus (label `IsNext`); 50% it's a **random sentence** from another document (`NotNext`).
- The final hidden state of `[CLS]` goes through a binary classifier; the loss is binary cross-entropy, added to the MLM loss.

```text
[CLS] The man went to the store [SEP] He bought a gallon of milk [SEP]   -> IsNext
[CLS] The man went to the store [SEP] Penguins are flightless birds [SEP] -> NotNext
```

**Why it was later dropped:**

- **RoBERTa** (2019) showed removing NSP and training on contiguous full-length text matched or improved downstream performance.
- The task is too easy: random sentences usually come from a different topic, so the model learns **topic detection** rather than coherence.
- It also forced shorter segments, hurting long-range learning.
- **ALBERT** replaced it with **Sentence Order Prediction (SOP)**: the two segments are consecutive, but sometimes swapped, forcing the model to learn discourse coherence.

**Interview takeaway:** NSP is a good example of a pre-training objective that sounded useful but whose benefit came mostly from confounding signals; ablations (RoBERTa) revealed it.

### 14.26 How does BERT work, and what are its pre-training objectives (MLM, NSP)?

**Architecture:** BERT (Devlin et al., 2018) is a **Transformer encoder-only** stack: 12 layers, hidden size 768, 12 heads, about 110M parameters for Base; 24 layers, 1024, 16 heads, about 340M for Large. It uses full **bidirectional self-attention**, so every token sees left *and* right context.

**Input representation:** WordPiece tokens (about a 30k vocabulary), plus segment embeddings (sentence A/B), plus learned position embeddings (max 512). Special tokens: `[CLS]` at the start (its final vector is used for classification) and `[SEP]` between and after sentences.

```text
[CLS] my dog is [MASK] [SEP] he likes play ##ing [SEP]
```

**Pre-training objectives**

1. **Masked Language Modeling (MLM):** randomly select 15% of tokens. Of those, 80% become `[MASK]`, 10% become a random token, and 10% stay unchanged (this reduces the pretrain/fine-tune mismatch, since `[MASK]` never appears at fine-tuning time). The model predicts the original tokens with cross-entropy. This is what makes deep bidirectional context possible; a left-to-right LM cannot "see the answer".
2. **Next Sentence Prediction (NSP):** given sentences A and B, predict from `[CLS]` whether B actually follows A (50% real, 50% random). It was meant to teach inter-sentence relationships for QA and NLI.

Training data was BooksCorpus plus English Wikipedia (about 3.3B words).

**Fine-tuning:** add a small head (classification on `[CLS]`, span start/end for QA, per-token tags for NER) and fine-tune all weights briefly.

**Caveats worth mentioning:** RoBERTa later showed **NSP adds little or even hurts**. MLM only learns from about 15% of tokens per pass, which is sample-inefficient (ELECTRA addresses this). BERT also can't generate text naturally. It is still a strong, cheap backbone for classification, NER, reranking (cross-encoders) and embeddings.

### 14.27 How does RoBERTa differ from BERT?

**RoBERTa** ("Robustly optimized BERT approach", Facebook AI, 2019) keeps BERT's **architecture unchanged** and shows BERT was significantly *undertrained*. The gains come purely from the training recipe:

- **Removed NSP:** trains with MLM only, on full contiguous sentences packed up to 512 tokens (FULL-SENTENCES). Dropping NSP matched or improved downstream performance.
- **Dynamic masking:** a new mask pattern is generated every time a sequence is fed, instead of BERT's static masks fixed at preprocessing time.
- **Much more data:** about 160GB of text (adds CC-News, OpenWebText, Stories) versus BERT's about 16GB.
- **Longer training with bigger batches:** batch size of 8k sequences, more steps, and a tuned learning rate and Adam $\beta_2$. Large batches improved both perplexity and end-task accuracy.
- **Tokenizer:** byte-level BPE with about a 50k vocabulary (GPT-2 style) instead of a 30k WordPiece. It has no unknown tokens, but a slightly larger embedding matrix.

**Result:** state-of-the-art on GLUE, SQuAD and RACE at the time, with zero architecture changes.

**Interview takeaway:** data scale, training duration and objective choices can matter as much as architecture, and ablations should be done carefully (the NSP finding). In practice RoBERTa, and DeBERTa after it, became the default encoder backbones for classification, NER and cross-encoder rerankers.

### 14.28 How do you approach text summarization?

**1. Clarify requirements:** extractive or abstractive? Single or multiple documents? Length and format (bullets, TL;DR, structured fields)? Domain (legal, medical, meetings)? Faithfulness requirements (regulated domains allow zero hallucination)? Volume, latency, cost?

**2. Approaches**

- **Extractive:** select the key sentences (TextRank / LexRank, or a supervised sentence scorer such as BERTSum). Faithful by construction, cheap and explainable, but choppy and redundant.
- **Abstractive with seq2seq models:** fine-tuned BART, T5 or PEGASUS. Fluent and good for fixed domains with training data, but can hallucinate.
- **LLM prompting (the default today):** instruction-tuned LLMs with clear prompts (audience, length, format, "only use information in the text"), few-shot examples, and structured output (JSON fields such as decisions and action items).

**3. Long documents**

- **Map-reduce:** summarize chunks, then summarize the summaries.
- **Refine:** iteratively update a running summary chunk by chunk.
- **Hierarchical:** section → document → corpus.
- **Long-context models:** simpler, but watch for lost-in-the-middle effects and cost.
- **Query-focused summarization:** retrieve only the relevant parts first (RAG).

**4. Evaluation**

- ROUGE or BERTScore against references (weak signal for LLM outputs).
- **Faithfulness:** NLI-based checks (SummaC, AlignScore), QA-based checks (QAGS, QuestEval), LLM-as-judge with a rubric (coverage, faithfulness, conciseness, coherence).
- Human review on a sample, especially in high-stakes domains.

**5. Production concerns:** hallucinated numbers, names and dates (verify entities against the source, or add citations per bullet), omission of critical information, PII handling, prompt injection inside the documents being summarized, model choice by cost (a cheap model for bulk work, a stronger one for important documents), caching, and monitoring of user edits and feedback as a quality signal.

### 14.29 How are knowledge graphs used with NLP?

A **knowledge graph (KG)** stores facts as entities and typed relations, e.g. `(Marie Curie, wonPrize, Nobel Prize in Physics)`. NLP and KGs help each other in both directions.

**NLP → building KGs**

- **Named Entity Recognition** to find entities, and **entity linking / disambiguation** to map each mention to a canonical node ("Apple" the company vs. the fruit; Wikidata IDs).
- **Relation extraction / open information extraction** to pull (subject, relation, object) triples from text, now often done with LLMs and structured outputs.
- **Coreference resolution**, **entity resolution** (deduplicating nodes), and **schema/ontology alignment**.

**KGs → improving NLP**

- **Question answering over KGs (KGQA):** translate natural language to SPARQL or Cypher (text-to-query with LLMs), giving precise answers for multi-hop and aggregate questions.
- **GraphRAG / KG-augmented RAG:** retrieve entities and their neighborhoods or community summaries instead of (or as well as) text chunks. This is good for multi-hop ("which suppliers of our top customer had incidents?") and global sensemaking questions.
- **Grounding and hallucination reduction:** check generated claims against KG facts, and constrain outputs to valid entities.
- **Search and recommendation:** entity-aware query understanding, knowledge panels, query expansion via related entities, explainable recommendations (paths in the graph).
- **Knowledge-enhanced language models:** inject entity embeddings (ERNIE, KnowBERT), or use the KG for consistency in domain-specific models (biomedical, finance).

**Trade-offs:** KGs give precision, explainability, multi-hop reasoning and easy updates of facts. They are costly to build and maintain, have incomplete coverage, are brittle to extraction errors, and need schema governance.

**Production pattern:** combine them. Run LLM-based extraction with human validation for critical relations, store the result in a graph DB (Neo4j, Neptune), and use hybrid retrieval (vector + graph traversal) at query time.

### 14.30 How would you solve entity resolution?

**Entity resolution (ER)** identifies records that refer to the same real-world entity across or within datasets (customers, companies, products) despite typos, abbreviations, missing fields and format differences.

**Pipeline**

1. **Standardize / clean:** normalize case, punctuation, addresses (libpostal), phone numbers (E.164), names (nicknames such as Bob → Robert), company suffixes (Inc, Ltd).
2. **Blocking / candidate generation:** comparing all pairs is $O(n^2)$, which is infeasible. Group by blocking keys (zip code, first 3 letters plus year), sorted neighborhood, **LSH/MinHash**, or **embedding ANN search** to produce candidate pairs. Tune for high recall.
3. **Pairwise matching:** compute similarity features per field (Jaro-Winkler, Levenshtein, TF-IDF cosine, phonetic codes such as Soundex/Metaphone, numeric and date distances, embedding similarity). Then classify:
   - Rules or **probabilistic Fellegi-Sunter** models (Splink: unsupervised EM, explainable match weights), or
   - Supervised models (GBDT, or fine-tuned transformers such as Ditto on serialized record pairs), or
   - LLMs to adjudicate hard pairs (expensive, so use them for the uncertain band only).
4. **Clustering:** turn pairwise matches into entities via connected components, or correlation clustering or thresholded graph clustering to avoid "chain" errors ($A \approx B$, $B \approx C$, but $A \neq C$).
5. **Canonicalization / golden record:** survivorship rules decide which value wins (most recent, most trusted source), and lineage is kept back to the source records.

**Evaluation:** pairwise precision and recall, cluster metrics ($B^3$, pairwise F1) on a labeled sample. Active learning to label the most informative uncertain pairs. **Clerical review** queue for mid-confidence matches.

**Production concerns:** incremental or streaming ER for new records (match against the existing index rather than re-running everything), stable entity IDs over time, scalability (Spark/Splink, Zingg), PII and privacy (hashing, access controls), and explainability for regulated uses (KYC/AML).

**Trade-off:** false merges (combining two customers) are usually worse than missed matches, so set thresholds asymmetrically.

### 14.31 How does named entity recognition work, from rule-based systems to Transformer token classification?

**Named entity recognition (NER)** finds spans of text that name things (people, organisations, locations, dates, drugs, product codes) and assigns each span a type. Almost every modern system treats it as **sequence labelling**: one tag per token in a scheme such as BIO, which a decoder then turns back into spans.

**Tagging schemes**

| Scheme | Tags | Note |
| --- | --- | --- |
| **IO** | I-PER, O | Cannot separate two adjacent entities of the same type |
| **BIO (IOB2)** | B-PER, I-PER, O | The default; B marks the start of a span |
| **BIOES / BILOU** | adds E (end) and S (single) | Often slightly better, because boundaries are explicit |

Example: `Tim/B-PER Cook/I-PER visited/O Paris/B-LOC`.

**How the approaches evolved**

1. **Rules and gazetteers.** Regular expressions, dictionaries of known names, and patterns such as "Mr. + capitalised word". High precision, zero training data, easy to audit. They still win for structured entities (IBANs, emails, ICD-10 codes, order numbers). They break on new names, ambiguity ("Jordan" the person, country or brand) and noisy text.
2. **Feature-based statistical models.** HMMs, maximum-entropy taggers and especially **linear-chain CRFs** over hand-made features: the word, its shape (`Xxxx`, `dd-dd`), prefixes and suffixes, POS tags, gazetteer hits and neighbouring words. The CRF learns which tag transitions are legal, so `O → I-PER` is penalised.
3. **Neural sequence models.** Lample et al. (2016) replaced feature engineering with a **BiLSTM-CRF**: word embeddings plus a character-level BiLSTM (which captures capitalisation and morphology) feed a bidirectional LSTM, and a CRF layer on top decodes the best tag sequence. It reached about 91 F1 on CoNLL-2003 English without gazetteers.
4. **Transformer token classification.** Fine-tune a pretrained encoder (BERT, RoBERTa, DeBERTa-v3, ModernBERT, XLM-R for multilingual) with a linear layer that predicts a tag for every token, as proposed in the original BERT paper. Contextual representations largely remove the need for the CRF, although a CRF head can still add a little on small datasets.
5. **LLMs and generalist extractors.** Prompting an LLM to return JSON spans, or using span-based zero-shot models such as GLiNER, handles new entity types with no labelled data. They are useful for bootstrapping labels, but slower, costlier and less consistent on exact boundaries than a fine-tuned encoder.

**The subword alignment detail interviewers probe**

Encoders split words into subwords, but labels are per word. The standard recipe uses the fast tokenizer's `word_ids()` to map tokens back to words, labels the first subtoken of each word, and sets the rest and the special tokens to `-100` so the cross-entropy loss ignores them.

```python
def align(labels, word_ids):
    out, prev = [], None
    for w in word_ids:
        out.append(-100 if w is None or w == prev else labels[w])
        prev = w
    return out
```

At inference, take the prediction of each word's first subtoken, repair invalid sequences (an `I-` after `O`), and merge into spans.

**Evaluation:** entity-level, exact-match precision, recall and F1 (the CoNLL convention, as in `seqeval`), not token accuracy, which is inflated by the many `O` tags. Report per-type F1 and partial-match metrics where boundaries are fuzzy.

**Production issues**

- **Nested and overlapping entities** ("Bank of [England]") cannot be expressed in flat BIO; use span classification models.
- **Long documents** exceed 512 tokens, so use a sliding window with overlap and merge predictions, or a long-context encoder.
- **Annotation consistency** limits accuracy more than architecture: write guidelines for edge cases (titles, possessives, nested organisations) and measure inter-annotator agreement.
- **Hybrid systems** are common: rules for structured IDs, a fine-tuned model for names, and entity linking afterwards to map mentions to canonical records.

### 14.32 How do Hidden Markov Models and Conditional Random Fields work for sequence labelling?

Both assign a label sequence $y = (y_1, \ldots, y_n)$ to an input sequence $x$ (POS tags, BIO entity tags). An **HMM** is a *generative* model of the joint probability $P(x, y)$; a **linear-chain CRF** is a *discriminative* model of the conditional probability $P(y \mid x)$. That one difference explains most of their trade-offs.

**Hidden Markov Model**

An HMM assumes the hidden tags form a Markov chain and each word is generated only by its own tag:

$$
P(x, y) = \prod_{t=1}^{n} P(y_t \mid y_{t-1}) \, P(x_t \mid y_t)
$$

- **Parameters:** transition probabilities $P(y_t \mid y_{t-1})$ and emission probabilities $P(x_t \mid y_t)$, estimated by counting on labelled data (with smoothing), or with Baum-Welch (EM) when tags are unobserved.
- **Decoding:** the **Viterbi** algorithm finds the most likely tag sequence by dynamic programming in $O(n K^2)$ for $K$ tags. The **forward algorithm** sums over all paths to give $P(x)$.
- **Weakness:** the emission assumption means a word sees only its own tag. You cannot easily add overlapping features such as "is capitalised", "ends in -ing", "previous word is Mr." or "next word is Inc." without breaking the independence assumptions. Unknown words are handled poorly.

**Conditional Random Field (Lafferty, McCallum and Pereira, 2001)**

A linear-chain CRF scores the whole tag sequence with weighted feature functions that may look at **any part of the input**, then normalises over all possible tag sequences:

$$
P(y \mid x) = \frac{1}{Z(x)} \exp\left( \sum_{t=1}^{n} \sum_{k} \lambda_k f_k(y_{t-1}, y_t, x, t) \right)
$$

- **Features** can be arbitrary and overlapping: the current word, its shape, suffixes, neighbouring words, gazetteer membership, and the transition pair $(y_{t-1}, y_t)$.
- **Training** maximises the conditional log-likelihood, a convex objective (with L2 regularisation) solved with L-BFGS or SGD. Computing $Z(x)$ and the gradients uses the forward-backward algorithm, so training is $O(n K^2)$ per sentence.
- **Decoding** is again Viterbi.
- **Why not a MEMM?** Maximum-entropy Markov models normalise locally, per position, which causes the **label bias problem**: states with few outgoing transitions ignore the observation. The CRF normalises globally over the whole sequence, which was the paper's key contribution.

**Comparison**

| Aspect | HMM | Linear-chain CRF |
| --- | --- | --- |
| Model type | Generative, $P(x, y)$ | Discriminative, $P(y \mid x)$ |
| Features | Current word given tag only | Arbitrary, overlapping, whole input |
| Training | Counting (closed form) or EM | Convex optimisation, slower |
| Unlabelled data | Can use EM | Needs labels |
| Accuracy on NER/POS | Lower | Higher with good features |
| Decoding | Viterbi | Viterbi |

**Where they sit today**

- The CRF survives as an **output layer**: BiLSTM-CRF and BERT-CRF replace hand-made features with neural emission scores, while the CRF's learned transition matrix enforces consistent tags (no `I-ORG` straight after `B-PER`). With strong pretrained encoders the gain is usually small, but it helps on small datasets and strict-boundary tasks.
- Feature-based CRFs (CRFsuite, sklearn-crfsuite) are still sensible when you need a tiny, CPU-only, explainable model, for example parsing addresses or invoice fields.
- HMMs remain important for teaching dynamic programming, and their ideas live on in speech recognition, bioinformatics and constrained decoding.

**Interview point:** be ready to write Viterbi. Keep a table `score[t][k]` of the best path score ending in tag `k` at position `t`, plus back-pointers, then trace back from the best final tag.

### 14.33 How do you choose an approach for a text classification problem, from TF-IDF baselines to fine-tuned encoders and LLMs?

Start from the constraints, not the model: how many labelled examples you have, how many classes, the latency and cost budget, how often the labels change, and how much an error costs. Then climb a ladder, stopping at the first rung that meets the target, because every step up adds cost and operational burden.

**Clarify the problem first**

- **Shape:** binary, multi-class, **multi-label** (one ticket can be "billing" and "outage"), or hierarchical.
- **Labels:** how many exist, their quality and inter-annotator agreement, class imbalance, and whether the label set changes monthly.
- **Text:** length (tweets vs. contracts), languages, domain jargon, PII.
- **Serving:** volume, p95 latency, batch or online, CPU or GPU, explainability or regulatory needs.

**The ladder**

| Approach | Labels needed | Strengths | Weaknesses |
| --- | --- | --- | --- |
| **Rules / keywords** | None | Instant, auditable | Low recall, brittle |
| **TF-IDF + logistic regression or linear SVM** | Hundreds to thousands | Trains in seconds, CPU, interpretable weights, hard to beat on topical tasks | No semantics, misses paraphrases and negation |
| **Frozen embeddings + linear head** (sentence-transformer, SetFit) | Tens to hundreds per class | Strong few-shot accuracy, cheap | Ceiling below full fine-tuning |
| **Fine-tuned encoder** (DeBERTa-v3, RoBERTa, ModernBERT, XLM-R) | About 1,000+ | Best accuracy per dollar at scale, 5 to 50 ms on GPU | Needs labels, GPU training, retraining when labels change |
| **LLM zero- or few-shot** | None to a few examples | Handles new or changing labels and nuanced instructions, gives rationales | Costly per call, slower, less calibrated, can drift with model versions |
| **LLM-labelled data distilled into an encoder** | LLM labels plus a human-checked sample | LLM quality at encoder cost | Inherits LLM biases, needs an audit set |

**How I decide in practice**

1. **Build a gold test set first** (a few hundred human-labelled, stratified examples). Without it nothing else can be compared.
2. **Run TF-IDF + logistic regression** in an afternoon. It sets the floor and often reveals label noise.
3. **Try an LLM zero-shot** on the same test set to see the ceiling for unlabelled approaches and to find ambiguous label definitions.
4. **If volume is high or latency is tight**, fine-tune an encoder, using LLM-generated labels to bootstrap if human labels are scarce.
5. **Keep the LLM** for long-tail or low-volume classes, or as a fallback when the encoder's confidence is low (a cascade).

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

baseline = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=2000, class_weight="balanced"),
)
```

**Details that decide success**

- **Metrics:** macro-F1 for imbalanced classes, per-class precision and recall, and for multi-label tune a threshold per label rather than using 0.5.
- **Calibration:** routing and auto-close decisions need calibrated probabilities (temperature scaling for encoders; LLM verbalised confidence is unreliable).
- **Long documents:** chunk and aggregate, or use a long-context encoder such as ModernBERT (8,192 tokens).
- **Cost maths:** 10 million documents a day through an LLM API costs orders of magnitude more than a self-hosted encoder on one GPU.
- **Change management:** if the taxonomy changes often, an LLM with label descriptions in the prompt or an embedding plus nearest-centroid approach avoids constant retraining.

**Common failure:** jumping straight to an LLM, never measuring it against a TF-IDF baseline, and discovering in production that the baseline was within one F1 point at a thousandth of the cost.

### 14.34 What is aspect-based sentiment analysis, and why are negation, sarcasm and domain shift hard for sentiment models?

**Aspect-based sentiment analysis (ABSA)** identifies *what* is being judged and the sentiment towards *each* target, rather than one score per document. "The battery life is superb but the screen scratches easily" is mixed at document level, but positive for battery and negative for screen. That is the level product and operations teams can act on.

**The ABSA subtasks** (formalised in the SemEval 2014 to 2016 shared tasks on laptop and restaurant reviews)

| Subtask | Output for "The pasta was great but the waiter was rude" |
| --- | --- |
| **Aspect term extraction** | `pasta`, `waiter` |
| **Aspect term polarity** | pasta: positive, waiter: negative |
| **Aspect category detection** | FOOD#QUALITY, SERVICE#GENERAL |
| **Aspect category polarity** | FOOD: positive, SERVICE: negative |
| **Opinion term extraction / triplets** | (pasta, great, positive), (waiter, rude, negative) |

Implicit aspects matter too: "it died after two hours" is about the battery without naming it.

**Modelling approaches**

- **Pipeline:** sequence labelling (BIO tags) to extract aspects, then a classifier fed the sentence plus the aspect.
- **Sentence-pair encoders:** feed `[CLS] sentence [SEP] aspect [SEP]` to BERT or DeBERTa and classify polarity. This auxiliary-sentence framing works well for category polarity.
- **Joint / generative:** a seq2seq model (T5) or LLM emits all (aspect, category, opinion, polarity) tuples as structured output, avoiding error propagation between stages.
- **LLMs with a schema:** strong zero-shot quality and easy to add new categories; in production often used to label data and then distilled into a smaller encoder.

**Why negation is hard**

- Negation changes polarity, but its **scope** varies: "not bad" is mildly positive, "not only cheap but also durable" is positive, "I can't say I disliked it" is a double negative.
- Bag-of-words and lexicon models flip or ignore the wrong words. Contextual encoders do far better, but still fail on rare structures, and benchmarks with targeted negation tests (CheckList-style) expose it.
- Mitigations: include hard negations in training data, generate counterfactual pairs ("good" vs. "not good"), and test them explicitly.

**Why sarcasm is hard**

- The literal words are positive while the intent is negative: "Great, my flight is delayed again. Love it."
- Detecting it often needs world knowledge (delays are bad), conversation context, or the author's history, none of which is in the sentence. Even humans disagree, so label agreement is low and the accuracy ceiling is low.
- Signals that help: sentiment contrast between a positive word and a negative situation, emojis, punctuation, and thread context. LLMs handle common sarcasm better but are far from reliable.

**Why domain shift is hard**

- Word polarity depends on domain: "unpredictable" is positive for a film plot and negative for a car's steering; "small" is good for a phone, bad for a hotel room; "sick" is positive in some slang.
- Distributions drift: new products, features, competitors, memes and languages appear. A model trained on movie reviews loses a lot of accuracy on support tickets.
- Mitigations: label in-domain data (even a few hundred examples help a lot), continue pre-training on domain text, use active learning to label the most uncertain new examples, and monitor prediction distributions and accuracy on a refreshed human-labelled sample.

**Evaluation:** macro-F1 per aspect category, accuracy on a targeted challenge set (negation, sarcasm, comparisons such as "better than the old model"), and slice metrics by product and channel. Handle the "conflict" or mixed label explicitly instead of forcing it into neutral.

### 14.35 How do GloVe and fastText differ from Word2Vec, and how did contextual embeddings change NLP?

All three produce **static** word vectors (one vector per word type), but they learn from different signals: Word2Vec from **local context windows**, GloVe from **global co-occurrence counts**, and fastText from **context windows plus character n-grams**. Contextual embeddings (ELMo, then BERT) replaced the fixed vector with one computed from the whole sentence.

**GloVe (Pennington, Socher and Manning, 2014)**

- First builds a word-word co-occurrence matrix $X$ over the corpus, where $X_{ij}$ counts how often word $j$ appears near word $i$.
- Then fits vectors so that their dot product predicts the log co-occurrence, with a weighting function $f$ that caps the influence of very frequent pairs and ignores zero counts:

$$
J = \sum_{i,j} f(X_{ij}) \left( w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2
$$

- In the paper $f(x) = (x / x_{\max})^{3/4}$ below $x_{\max} = 100$ and 1 above it.
- **Intuition:** ratios of co-occurrence probabilities carry meaning ("ice" co-occurs with "solid" far more than "steam" does), and a log-bilinear model turns those ratios into vector differences, which is why analogies work.
- **Relation to Word2Vec:** skip-gram with negative sampling has been shown to implicitly factorise a shifted PMI matrix, so the two are closer than they look. GloVe makes the matrix explicit and trains on it in a single pass over non-zero entries.

**fastText (Bojanowski et al., 2017)**

- Extends skip-gram by representing each word as the **sum of its character n-gram vectors** (lengths 3 to 6, with `<` and `>` boundary markers) plus a vector for the whole word. "where" includes `<wh`, `whe`, `her`, `ere`, `re>` and `<where>`.
- N-grams are hashed into a fixed number of buckets (2 million in the paper) to bound memory.
- **Benefits:** vectors for **out-of-vocabulary** words and typos ("recieve"), shared information between morphological variants, and much better results for morphologically rich languages (Finnish, Turkish, German compounds). Pretrained vectors exist for 157 languages.
- **Cost:** a larger model and some noise from coincidental shared substrings.

**Comparison**

| Aspect | Word2Vec | GloVe | fastText |
| --- | --- | --- | --- |
| Training signal | Local windows, predictive | Global co-occurrence counts | Local windows plus subwords |
| Unseen words | No vector | No vector | Built from n-grams |
| Morphology | Ignored | Ignored | Captured |
| Typical strength | Fast, good analogies | Uses global statistics efficiently | Rare words, noisy text, rich morphology |

**How contextual embeddings changed NLP**

- **The problem:** static vectors average all senses of a word, so "bank" (river) and "bank" (money) share one vector, and the vector ignores syntax in the sentence.
- **ELMo (Peters et al., 2018):** a two-layer bidirectional LSTM language model over character-based inputs. Each token's representation is a learned, task-specific weighted sum of all layers. Lower layers carried more syntax, higher layers more semantics. Adding ELMo to existing models improved results across six benchmarks, including question answering, NER and sentiment.
- **BERT and successors:** deep Transformer encoders trained with masked language modelling, then **fine-tuned end to end** rather than used as frozen features.
- **The paradigm shift:** from "pretrain word vectors, then train a task architecture from scratch" to "pretrain a whole network, then fine-tune or prompt it". Task-specific architectures largely disappeared.

**Are static vectors still useful?** Yes, where cost and simplicity dominate: features for gradient-boosted models, fast lexical expansion, keyword clustering, very low-resource on-device settings, and fastText's own linear classifier, which trains on millions of documents in minutes on a CPU. For semantic search, use sentence-level embedding models instead.

### 14.36 How did Bahdanau and Luong attention improve sequence-to-sequence models?

Early RNN encoder-decoders compressed the entire source sentence into **one fixed-length vector**, and translation quality collapsed on long sentences. **Bahdanau et al. (2014)** let the decoder build a **different context vector at every output step** as a weighted average of all encoder states, with the weights learned jointly ("align and translate"). **Luong et al. (2015)** simplified and systematised the idea with cheaper scoring functions and global versus local variants. Together they are the direct ancestors of Transformer attention.

**The general mechanism**

At decoder step $t$, with encoder states $h_1, \ldots, h_S$ and decoder state $s$:

$$
e_{t,i} = \text{score}(s, h_i), \qquad \alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j} \exp(e_{t,j})}, \qquad c_t = \sum_{i} \alpha_{t,i} h_i
$$

The context vector $c_t$ is then combined with the decoder state to predict the next word.

**Bahdanau (additive) attention**

- **Score:** a small feed-forward network, $e_{t,i} = v^\top \tanh(W s_{t-1} + U h_i)$.
- Uses the **previous** decoder state $s_{t-1}$ to compute attention, and feeds $c_t$ into the RNN update that produces $s_t$.
- **Bidirectional encoder**, so each $h_i$ summarises words both before and after position $i$.
- **Result:** performance no longer fell sharply with sentence length, and the attention weights formed interpretable soft alignments (for example, handling the reversed adjective-noun order between English and French).

**Luong (multiplicative) attention**

- **Score options:** `dot` $s_t^\top h_i$, `general` $s_t^\top W h_i$, and `concat` (similar to Bahdanau).
- Uses the **current** decoder state $s_t$ (computed first), then forms an attentional state $\tilde{s}_t = \tanh(W_c [c_t ; s_t])$ to predict the word. This is a simpler computation path.
- **Global attention** attends to all source positions. **Local attention** predicts an aligned position $p_t$ and attends within a window around it with a Gaussian weighting, reducing cost on long inputs.
- **Input feeding:** $\tilde{s}_t$ is fed into the next step's input so the model remembers past alignment decisions (a rough form of coverage).
- Stacked LSTMs plus these ideas set new results on WMT English-German at the time.

**Comparison**

| Aspect | Bahdanau | Luong |
| --- | --- | --- |
| Score | Additive MLP | Dot, general (bilinear) or concat |
| Decoder state used | Previous, $s_{t-1}$ | Current, $s_t$ |
| Encoder | Bidirectional RNN | Stacked unidirectional LSTM |
| Variants | Global only | Global and local (windowed) |
| Cost | More parameters, slower | Dot product is a fast matrix multiply |

**Why it mattered**

- **Removed the bottleneck:** information flows directly from every source word to every output step, giving shorter gradient paths and better long-range handling.
- **Alignment for free:** attention maps became a debugging and analysis tool, and were used for unknown-word replacement by copying the aligned source word.
- **Led to the Transformer:** scaled dot-product attention is Luong's `dot` score divided by $\sqrt{d_k}$ to keep softmax gradients healthy, with queries, keys and values as separate projections. The Transformer then dropped the RNN altogether and applied attention to the sequence itself (self-attention), which made training parallel across positions.

**Limits that remained:** RNN computation was still sequential, attention cost grows with source length times target length, and soft attention was not guaranteed to cover every source word, causing over- and under-translation. Coverage models and later Transformers addressed these.

### 14.37 How does beam search work, and when should you use it instead of sampling?

**Beam search** is an approximate search for the **most probable output sequence**. Instead of committing to the single best token at each step (greedy decoding), it keeps the $B$ best partial sequences, extends each by every candidate token, and again keeps the top $B$ by cumulative log-probability. Use it when there is essentially one correct answer (translation, transcription, structured extraction); use sampling when variety and natural-sounding open-ended text matter.

**The algorithm**

1. Start with one hypothesis: the start token, score 0.
2. For each of the $B$ hypotheses, compute next-token log-probabilities and form candidates with score $\sum_{t} \log P(y_t \mid y_{<t}, x)$.
3. Keep the top $B$ candidates overall. Move any hypothesis that emits the end token into a finished list.
4. Stop when $B$ hypotheses have finished or no unfinished one can beat the best finished score. Return the best finished hypothesis.

Cost is roughly $B$ times greedy decoding in compute and KV-cache memory (inference engines share the prompt's cache across beams to reduce this). Typical beam sizes are 4 to 10.

**Length bias and normalisation**

Every extra token adds a negative log-probability, so raw scores favour **short outputs**. Common fixes:

- Divide the score by length, or by a length penalty. Google's GNMT used $lp(Y) = \frac{(5 + |Y|)^\alpha}{(5 + 1)^\alpha}$ with lpha on a development set (values around 0.6 are common), plus a **coverage penalty** to discourage ignoring source words.
- Set minimum and maximum lengths, and add a **no-repeat n-gram** constraint to stop loops.

**The beam search curse**

Larger beams find higher-probability sequences, but the quality metric (BLEU) often gets **worse** beyond a modest beam, and very large beams can return short or even empty outputs. The model's most probable sequence is not the best sequence, so a moderate beam acts as a useful regulariser.

**Why beam search fails for open-ended text**

Holtzman et al. ("The Curious Case of Neural Text Degeneration", 2019) showed that maximisation-based decoding produces **bland, generic and repetitive** text, while human text is not the highest-probability text; its per-token probability varies a lot. They proposed **nucleus (top-p) sampling**, which samples from the smallest token set covering probability mass $p$, cutting off the unreliable tail while keeping diversity.

**When to use which**

| Use case | Recommended decoding | Why |
| --- | --- | --- |
| Machine translation, speech recognition, OCR | Beam search ($B$ = 4 to 5) with length normalisation | One correct output; beam beats greedy by a measurable margin |
| Summarisation with smaller seq2seq models (BART, T5) | Beam search with no-repeat n-grams | Faithful, compact outputs |
| Structured extraction, code with tests | Greedy or beam; or sample many and verify | Determinism, or pass@k with a checker |
| Chat, creative writing, brainstorming | Temperature plus top-p or min-p sampling | Diversity, avoids degeneration |
| Self-consistency reasoning | Sample $N$ answers, majority vote | Diversity is the point |
| Large instruction-tuned LLMs | Usually greedy or low-temperature sampling | Sharp distributions; beam adds cost for little gain |

**Variants worth knowing**

- **Diverse beam search** splits beams into groups with a diversity penalty, giving genuinely different candidates for reranking.
- **Constrained beam search** forces required phrases or terminology (glossary terms in translation).
- **Sample-and-rerank** (generate $N$ samples and pick the best by a reward model, verifier or minimum Bayes risk) often outperforms beam search on quality in modern systems.

**Production notes:** beam search multiplies memory per request and complicates token streaming (the leading hypothesis can change), which is why most LLM serving APIs do not offer it. Always tune beam size and length penalty on a dev set with the real metric, not on log-likelihood.

### 14.38 How do topic models such as LDA and BERTopic work, and when would you use each?

A topic model finds recurring themes in a collection of documents without labels. **LDA** is a probabilistic bag-of-words model: each document is a mixture of topics, and each topic is a distribution over words. **BERTopic** is a clustering pipeline: it embeds documents with a transformer, clusters the embeddings, and then describes each cluster with its most distinctive words. Use LDA when you need interpretable mixed memberships on long documents with little compute. Use BERTopic when meaning matters more than word counts, especially for short text.

**How LDA works**

LDA assumes a generative story. For $K$ topics, each topic $k$ has a word distribution $\phi_k \sim \text{Dir}(\beta)$. Each document $d$ draws topic proportions $\theta_d \sim \text{Dir}(\alpha)$. Each word position first picks a topic $z \sim \text{Mult}(\theta_d)$, then a word $w \sim \text{Mult}(\phi_z)$. Training inverts this story: given only the words, infer $\theta$ and $\phi$ with variational Bayes (the original paper, and scikit-learn's online version) or collapsed Gibbs sampling (MALLET).

- A small $\alpha$ pushes each document towards a few topics; a small $\beta$ pushes each topic towards a few words.
- Output per document is a probability vector, such as 60% "billing", 30% "delivery", 10% "returns".
- It sees only counts, so preprocessing decides quality: stop words, lemmatisation, bigrams ("credit_card"), and pruning very rare and very common terms.

**How BERTopic works**

1. **Embed** each document with a sentence-transformer.
2. **Reduce** dimensions with UMAP, which keeps local neighbourhood structure so density clustering works.
3. **Cluster** with HDBSCAN, which picks the number of clusters itself and puts unclear documents in an outlier topic (labelled -1).
4. **Describe** each cluster with **class-based TF-IDF (c-TF-IDF)**: concatenate all documents in a cluster into one "class document" and score words that are frequent there but rare across other clusters.
5. Optionally refine labels with KeyBERT-style reranking or an LLM that writes a short topic name.

Every step is swappable (k-means instead of HDBSCAN, a domain embedding model), and the library supports topics over time, per class, guided seed topics and merging similar topics.

**Comparison**

| Aspect | LDA | BERTopic |
| --- | --- | --- |
| Input signal | Word counts | Contextual embeddings |
| Short text (tweets, tickets) | Weak, too few words per document | Strong |
| Document membership | Mixture of topics | One cluster per document (soft distribution is approximate) |
| Number of topics | Set $K$ yourself | Found by HDBSCAN, tunable via minimum cluster size |
| Compute | CPU, cheap | Embedding cost, GPU helps |
| Reproducibility | Fairly stable with fixed seed | UMAP is stochastic, results shift between runs |
| Main failure mode | Junk topics of stop words and boilerplate | Large outlier topic, one giant cluster |

**Evaluating topics**

- Perplexity does not track human judgement; prefer **topic coherence** (NPMI or $C_v$) plus **topic diversity** (share of unique words across top-word lists).
- The real test is human review: can an analyst name each topic, and are its example documents consistent?
- Check stability: rerun with different seeds and measure how many topics survive.

**Production pitfalls**

- With HDBSCAN, a large share of documents (in practice often a third or more) can land in the outlier topic. Lower `min_samples`, or reassign outliers to the nearest topic, and report the share you reassigned.
- Topics drift. Retraining renumbers them, so map new topics to old ones by centroid or top-word similarity before updating dashboards.
- Do not treat topics as classes. If the business needs a stable taxonomy ("complaint about delivery"), use topic modelling to *discover* the categories, then label data and train a classifier.

**Example:** for 200,000 short support tickets, I would run BERTopic with a multilingual embedding model, an LLM to name topics, and a human pass to merge and rename. For 10,000 long policy reports where each covers several themes, LDA or BERTopic's approximate topic distribution gives the mixture view that analysts want.

### 14.39 How do you build an information extraction system for entities and relations, and how have LLMs changed it?

Information extraction (IE) turns unstructured text into structured records: entities (people, drugs, contract parties), their types and canonical IDs, and typed relations between them, such as `(Acme Ltd, acquired, Beta GmbH, 2025-03-01)`. A production system is a pipeline with a schema, extraction models, normalisation and validation. LLMs have made the extraction step much faster to build, but have not removed the need for the rest.

**1. Start from the schema, not the model.** Define the entity types, relation types, argument types, cardinality and allowed values (for example, `acquired` takes two `Organisation` arguments and an optional date). Write annotation guidelines with edge cases. Most IE projects fail on vague schemas, not weak models.

**2. Classic pipeline.**

- **Preprocessing:** document parsing (PDF, HTML, tables), sentence splitting, tokenisation.
- **Named entity recognition:** sequence labelling with BIO tags, usually a fine-tuned encoder (BERT, RoBERTa, DeBERTa) with a token-classification head.
- **Coreference resolution:** link "the company" and "it" back to the entity, so relations stated with pronouns are not lost.
- **Entity linking:** map each mention to a canonical ID in a knowledge base or master data, using candidate generation (alias tables, embeddings) and a disambiguation model.
- **Relation extraction:** classify each candidate entity pair in a sentence or document into a relation type or "no relation". Typical models mark the entities with special tokens and classify the encoder output.
- **Normalisation:** dates, amounts, units and names into standard formats.

**3. Design choices.**

- **Pipeline vs joint models:** pipelines are modular and easy to debug, but errors cascade (a missed entity means a missed relation). Joint models extract entities and relations together and reduce cascading errors.
- **Sentence vs document level:** many relations span sentences (the buyer is named in paragraph 1, the price in paragraph 4). Document-level RE is much harder and needs coreference.
- **Closed vs open IE:** a fixed schema gives clean, queryable data; open IE extracts any (subject, predicate, object) triple but produces noisy, redundant relations that need canonicalisation.

**4. What LLMs changed.**

- **Zero/few-shot extraction:** describe the schema in a prompt, request JSON, and enforce it with structured outputs or constrained decoding. A working prototype now takes days, not months of labelling.
- **Long, messy documents:** LLMs handle cross-sentence relations, implicit relations and varied phrasing better than sentence-level classifiers.
- **Labelling:** LLMs pre-annotate data that humans correct, or label a corpus to train a small model.
- **Small generalist models:** GLiNER-style encoders take entity or relation type names as input and do zero-shot extraction at a fraction of LLM cost.

**What did not change:** zero-shot LLMs still trail well-tuned supervised models on benchmark NER and RE, especially for rare types and precise span boundaries. LLMs also hallucinate entities not in the text, normalise values inconsistently, and cost far more per document.

**5. Production pattern I would use.**

- Prototype with an LLM and the schema, measure on a gold set of 200–500 annotated documents.
- Require **evidence spans**: every extracted field must cite the exact text it came from, and a validator checks that the span exists in the source. This removes most hallucinated values.
- Validate types and business rules (dates parse, amounts are positive, both arguments have the right type).
- If volume is high, **distil**: label 20,000 documents with the LLM, review a sample, fine-tune an encoder for cheap inference, and keep the LLM for low-confidence cases.
- Route low-confidence or high-impact extractions to human review.

**Evaluation:** entity F1 with exact and partial span matching, relation F1 with strict matching (both arguments and type correct), and field-level accuracy for document templates. Track performance per relation type, because averages hide rare relations that the business often cares about most.

### 14.40 What is coreference resolution, and why does it matter in NLP pipelines?

Coreference resolution finds all expressions in a text that refer to the same real-world entity and groups them into clusters. In "Siemens reported results on Tuesday. The company said its margins improved, and they expect growth", the mentions "Siemens", "The company", "its" and "they" form one cluster. Without this step, any system that reads text sentence by sentence loses much of what is said about each entity.

**Types of reference it must handle**

- **Pronouns:** "she", "it", "they", "its".
- **Definite noun phrases:** "the company", "the drug", "the defendant".
- **Name variants:** "Microsoft Corporation", "Microsoft", "MSFT".
- **Harder cases:** event coreference ("the acquisition" referring to an earlier sentence), split antecedents ("Anna met Ben; they left"), and cataphora, where the pronoun comes first.
- It is different from **entity resolution**, which matches records across databases, and from **entity linking**, which maps a mention to a knowledge-base ID. Coreference works within one document.

**How it is solved**

- **Rule-based and statistical (pre-2017):** hand-crafted sieves (exact match, then head match, then pronouns) and mention-pair classifiers with features such as gender, number and distance.
- **End-to-end neural (Lee et al., 2017):** treat every text span up to a maximum length as a possible mention, score how likely it is to be a mention, prune, then for each remaining span score each earlier span as its antecedent (or "no antecedent"). Training uses only the gold clusters, with no separate mention detector or parser. Later work replaced the encoder with SpanBERT and Longformer and made the model cheaper.
- **Current encoder models:** recent encoder-only systems (for example LingMess, Maverick, and the fast library fastcoref) reach the low 80s CoNLL F1 on OntoNotes and run quickly on a single GPU.
- **LLMs:** a prompted LLM can resolve references reasonably well, and it is often done implicitly when the LLM answers questions over a document. But zero-shot LLMs still trail specialised models on benchmarks by double-digit F1, and they cost far more per document.

**Evaluation:** there is no single correct metric, so the standard is the **CoNLL F1**, the average of MUC (link-based), $B^3$ (mention-based) and CEAF (entity alignment). Report it alongside errors on the mention types your product depends on.

**Why it matters in pipelines**

- **Information extraction:** "It agreed to pay \$2 billion" only becomes the triple (Acme, pays, \$2B) if "It" resolves to Acme. Relation extraction recall can drop sharply without coreference.
- **RAG and chunking:** a chunk that starts "He later resigned" is useless when retrieved alone. Resolve references, or add context such as the entity name, before chunking.
- **Summarisation and QA:** both need to know who did what across sentences; errors cause wrong attributions.
- **Knowledge graphs and analytics:** entity sentiment, mention counts and timelines are undercounted when pronoun mentions are ignored.
- **Privacy:** redacting a patient name but leaving "she was admitted on..." still exposes information linked to the person.

**Production pitfalls**

- Domain shift: models trained on news struggle with clinical notes, legal contracts and chat logs.
- Long documents: errors increase with distance, and memory grows with document length. Process in overlapping windows and merge clusters.
- **Over-merging** two similar entities ("the buyer" and "the seller") is usually worse than missing a link, since it corrupts downstream facts. Use conservative thresholds for high-stakes uses.
- Language coverage: quality for low-resource languages is much lower than for English.

### 14.41 How do extractive, abstractive and open-domain question answering systems differ?

They differ in **where the answer comes from** and **what the system is given**. Extractive QA selects a span from a supplied passage. Abstractive QA generates the answer in its own words. Open-domain QA is not given the passage at all: it must first find relevant documents in a large corpus, then answer with an extractive or abstractive reader.

**Extractive QA**

- Input: question plus context passage. Output: start and end positions of the answer span.
- Model: an encoder (BERT, RoBERTa, DeBERTa) with two heads that score each token as span start and span end; the best valid span (start before end, limited length) wins.
- Benchmark: **SQuAD** (crowdsourced questions on Wikipedia paragraphs). SQuAD 2.0 added unanswerable questions, so the model must also learn to say "no answer", usually by comparing the best span score with a null score.
- Metrics: **Exact Match** and **token-level F1** against reference answers.
- Strengths: the answer is guaranteed to be in the source, which makes it faithful and easy to highlight. Weaknesses: cannot combine facts, compute, summarise or answer yes/no in a natural way.

**Abstractive (generative) QA**

- Input: question, optionally with context. Output: free text written by a seq2seq model (T5, BART) or a decoder LLM.
- Strengths: combines evidence from several passages, rephrases, handles "why" and "how" questions, and adapts tone and length.
- Weaknesses: can hallucinate content not in the source, and EM/F1 underrate correct paraphrases. Evaluation needs semantic metrics, LLM judges, and faithfulness checks against the context.

**Open-domain QA**

The classic design is **retriever plus reader**:

- **Retriever:** sparse (BM25) or dense. **Dense Passage Retrieval (DPR)** trains two BERT encoders, one for questions and one for passages, so that the dot product of their vectors ranks relevant passages first. It is trained with in-batch negatives plus a hard negative from BM25, and passage vectors are indexed with FAISS. On Natural Questions, DPR reached about 78% top-20 retrieval accuracy against about 59% for BM25, which showed that learned retrieval could beat strong lexical baselines.
- **Reader:** an extractive model over the top passages, or a generative one that reads many passages together (Fusion-in-Decoder).
- **Closed-book QA** is the opposite extreme: the model answers from its parameters with no retrieval. It is simple but cannot be updated without retraining and cannot cite sources.

Modern RAG is open-domain abstractive QA with an LLM as reader.

**Comparison**

| Aspect | Extractive | Abstractive | Open-domain |
| --- | --- | --- | --- |
| Given context? | Yes | Optional | No, must retrieve |
| Answer form | Exact span | Generated text | Span or generated |
| Main risk | Wrong span, no synthesis | Hallucination | Retrieval misses the evidence |
| Typical metrics | EM, F1 | Semantic similarity, LLM judge, faithfulness | Retrieval recall@k plus reader metrics |
| Good fit | Highlighting in contracts, forms | Assistants, explanations | Search over a knowledge base |

**How I decide**

- If users must see the exact source wording (legal, compliance, clinical), use extractive or require generated answers to quote spans that are verified against the source.
- If questions need synthesis across documents, use a generative reader with citations.
- In open-domain systems, **measure retrieval separately**: if the right passage is not in the top $k$, no reader can fix it. Most production errors I see are retrieval failures, not reader failures.
- Handle "no answer" explicitly. Models trained only on answerable questions will invent an answer from the closest passage.

### 14.42 How do weak supervision and LLM-assisted labelling help when labelled text data is scarce?

Both replace part of expensive expert labelling with cheaper, noisier sources and then control that noise. **Weak supervision** combines many imperfect heuristics into probabilistic labels. **LLM-assisted labelling** uses a large model as an annotator or pre-annotator. In both cases, a small, clean, human-labelled set is still essential, because it is the only way to measure how good the cheap labels are.

**Weak supervision (data programming, Snorkel)**

1. Experts write **labelling functions (LFs)**: small programs that vote a label or abstain. Examples: a keyword rule ("refund" → `BILLING`), a regex for order numbers, a lookup in a list of known product names (distant supervision), an old model's prediction, or an LLM prompt treated as one more LF.
2. Apply all LFs to the unlabelled corpus to get a matrix of votes, mostly abstains.
3. A **label model** estimates each LF's accuracy and correlations from their agreements and conflicts, **without ground truth**, and outputs a probabilistic label per example. This beats a simple majority vote when LFs differ in quality.
4. Train an **end model** (for example, a fine-tuned transformer) on the probabilistic labels. It generalises beyond the LFs, because it learns from context, not just the keywords that fired.

Key LF diagnostics: **coverage** (share of examples labelled), **overlap**, **conflict**, and empirical accuracy on a small dev set. The Snorkel paper's user study reported experts building models several times faster than with hand labelling, and with better accuracy than seven hours of manual labels.

Failure modes: LFs that are correlated (three rules on the same keyword count as three votes), low coverage of the minority class, and LFs that encode the same bias as the old process.

**LLM-assisted labelling**

- **LLM as annotator:** prompt with the label guide and a few examples, use low temperature, and request structured output with a short rationale. Studies (for example, on tweet and news classification) found zero-shot ChatGPT beat crowd workers on accuracy at a much lower cost, and its agreement with itself was high at low temperature.
- **LLM as pre-annotator:** the model proposes labels or spans and humans accept or correct them. In practice this can substantially cut the time per item, so measure it on a pilot batch.
- **LLM as one labelling function:** combine it with rules in a label model, so its errors are weighted by its estimated accuracy.
- **Distillation:** label tens of thousands of examples with the LLM, then fine-tune a small classifier for cheap, fast, private inference.
- **Synthetic data:** generate examples for rare classes. Check diversity and realism, because synthetic text is often easier and more uniform than real data.

**Risks to manage**

- LLM errors are **systematic, not random**: biases towards certain labels, positions in the list, dialects or demographic groups, and they repeat the same mistake consistently. Averaging does not remove them.
- Labels depend on model version and prompt. Pin versions, store prompts with labels, and re-validate when either changes.
- Data governance: sending customer text to an external API may breach privacy rules. Use an approved or self-hosted model.
- Never evaluate a model on labels made by the same LLM; the test set must be human-labelled.

**Workflow I would use**

1. Write a label guide; have experts label 300–500 examples as a gold set, and measure inter-annotator agreement (Cohen's kappa).
2. Run the LLM and rules on the gold set; measure per-class precision and recall for each source.
3. Combine sources (label model, or LLM with rules as checks); route low-confidence and disagreement cases to humans.
4. Train the end model, evaluate on the gold set, and compare with a baseline trained only on the gold labels.
5. Spend the remaining budget on the classes and slices with the largest error, found through error analysis.

**When not to use them:** if the task needs deep expertise and the LLM's agreement with experts is close to chance, or if a few thousand human labels are affordable and the label space is small, plain human labelling is often simpler and more reliable.

### 14.43 How do you detect near-duplicate documents at scale using MinHash and locality-sensitive hashing?

Represent each document as a set of shingles, compress each set into a short **MinHash signature** whose agreement rate estimates Jaccard similarity, then use **locality-sensitive hashing (LSH) banding** so that only documents likely to be similar are compared. This turns an $O(n^2)$ all-pairs comparison into roughly linear work, which is how billion-document web corpora are deduplicated.

**Step 1: Shingling.** Normalise the text (lowercase, collapse whitespace, strip punctuation and boilerplate), then split it into overlapping n-grams: word 5-grams or character 5–9-grams are common. Similarity between documents $A$ and $B$ is the Jaccard index of their shingle sets:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

**Step 2: MinHash.** Apply a hash function to every shingle and keep the minimum value. For a random hash function, the probability that two documents share the same minimum equals their Jaccard similarity:

$$
P\left[\min h(A) = \min h(B)\right] = J(A, B)
$$

Repeat with $k$ independent hash functions (commonly 128–256) to get a $k$-number signature. The share of positions where two signatures agree estimates $J$, with standard error about $\sqrt{J(1-J)/k}$. The signature size is fixed, whatever the document length.

**Step 3: LSH banding.** Split the signature into $b$ bands of $r$ rows ($b \cdot r = k$). Hash each band to a bucket; two documents become **candidates** if they match exactly in at least one band. The probability of becoming candidates is:

$$
P(\text{candidate}) = 1 - \left(1 - s^{r}\right)^{b}
$$

This is an S-curve with its steep part near $(1/b)^{1/r}$. More rows per band make the threshold stricter (fewer false positives); more bands make it more lenient (fewer missed duplicates). For example, with $k = 128$, choose $b$ and $r$ so that the curve rises near your target threshold, often 0.7–0.8 for training data deduplication.

**Step 4: Verify and cluster.** Optionally compute exact Jaccard for candidate pairs to remove false positives. Then build a graph of duplicate pairs and take **connected components** (union-find) to form clusters, keeping one document per cluster, for example the longest, newest or highest-quality one.

**Small example with datasketch**

```python
from datasketch import MinHash, MinHashLSH

def signature(text, k=128, n=5):
    words = text.lower().split()
    m = MinHash(num_perm=k)
    for i in range(max(1, len(words) - n + 1)):
        m.update(" ".join(words[i:i + n]).encode("utf8"))
    return m

lsh = MinHashLSH(threshold=0.8, num_perm=128)
for doc_id, text in docs:
    sig = signature(text)
    dupes = lsh.query(sig)          # candidate near-duplicates already indexed
    if not dupes:
        lsh.insert(doc_id, sig)     # keep the first copy only
```

**Scaling to billions of documents.** In Spark, Ray or Daft, compute signatures per partition, emit `(band_id, band_hash, doc_id)` rows, group by band, generate pairs within each bucket, and run distributed connected components. Cap or sample very large buckets, which come from boilerplate (cookie banners, templates) and can explode the number of pairs.

**Why it matters for LLMs.** Lee et al. ("Deduplicating Training Data Makes Language Models Better") found that common corpora such as C4 contain many near-duplicates and long repeated substrings. Deduplication with MinHash (document level) plus suffix arrays (exact repeated substrings) cut verbatim memorised output roughly tenfold, reached the same or better accuracy with fewer training steps, and exposed train-test overlap that inflated benchmark scores. Decontaminating evaluation sets against the training data uses the same tools.

**Trade-offs and alternatives**

| Method | Detects | Notes |
| --- | --- | --- |
| Exact hash (SHA-256) | Identical text | Cheap first pass |
| MinHash + LSH | High shingle overlap (Jaccard) | Standard for web-scale text |
| SimHash | Small edits (Hamming distance) | Compact 64-bit fingerprints |
| Suffix array | Repeated exact substrings | Finds copied paragraphs within different documents |
| Embedding similarity | Paraphrases, translations | Semantic, expensive, needs ANN search |

**Pitfalls:** shingle size changes what counts as "similar" (short shingles make unrelated documents look alike); threshold choice trades data volume against redundancy, so inspect sampled pairs at several thresholds; and all signatures must use the same number of permutations and seeds, or comparisons are meaningless.

---

← [Chapter 13. Deep Learning](13-deep-learning.md) · [Table of contents](../README.md) · [Chapter 15. Computer Vision](15-computer-vision.md) →
