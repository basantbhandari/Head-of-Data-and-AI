# Chapter 14. Natural Language Processing

_Part III — Machine Learning_

Text representation, classical NLP, BERT-family models and applied NLP systems.

This chapter contains **30 questions**.

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
| **Quadratic complexity with sequence length** | Self-attention computes pairwise interactions between all tokens, resulting in **O(n²)** time and memory complexity. This becomes expensive for long sequences. | Sparse attention, linear attention, FlashAttention, hierarchical/chunked processing |
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
| **Context** | **Short context** — limited to N words | Can capture **longer context**, especially Transformers |
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

An **n-gram model** is an NLP language model that predicts the **next word/token based on the previous N−1 words**.

| **Aspect** | **Explanation** |
| --- | --- |
| **Definition** | A model that uses sequences of **N words/tokens** to predict or analyze text |
| **N** | Number of words/tokens in the sequence |
| **Unigram (N=1)** | Considers one word at a time |
| **Bigram (N=2)** | Considers two consecutive words |
| **Trigram (N=3)** | Considers three consecutive words |
| **Example sentence** | `"I love machine learning"` |
| **Bigrams** | `"I love"`, `"love machine"`, `"machine learning"` |
| **Trigrams** | `"I love machine"`, `"love machine learning"` |
| **Main purpose** | Predict the next word based on previous words |
| **Example** | `"I want to"` → predicts `"learn"` |
| **Advantages** | Simple, fast, easy to implement |
| **Limitations** | Struggles with long-range context and requires large amounts of data for larger N |
| **Applications** | Text prediction, speech recognition, spelling correction, autocomplete, language modeling |

### 14.11 What is TF-IDF, and how does it differ from word embeddings?

#### What is TF-IDF (Term Frequency-Inverse Document Frequency)?

- **Core idea:** A statistical measure used to evaluate how important a word is to a document within a collection (corpus), balancing how often a word appears in a document against how common it is across all documents.
- **Term Frequency (TF):** Measures how frequently a word appears in a document.
  - `TF(t, d) = (Number of times term t appears in document d) / (Total terms in d)`
- **Inverse Document Frequency (IDF):** Measures how rare/common a word is across the entire corpus — words appearing in many documents get a lower score (less informative).
  - `IDF(t) = log(Total number of documents / Number of documents containing term t)`
- **TF-IDF score:** `TF-IDF(t, d) = TF(t, d) * IDF(t)`
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

For a term-document matrix A, SVD decomposes it as:

$$
A = U\Sigma V^T
$$

LSI keeps only the most important kk dimensions:

$$
A \approx U_k\Sigma_k V_k^T
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

`king − man + woman ≈ queen`

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
- Can capture relationships through vector arithmetic, such as:`king − man + woman ≈ queen`
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
4. **Clustering:** turn pairwise matches into entities via connected components, or correlation clustering or thresholded graph clustering to avoid "chain" errors (A≈B, B≈C, but A≠C).
5. **Canonicalization / golden record:** survivorship rules decide which value wins (most recent, most trusted source), and lineage is kept back to the source records.

**Evaluation:** pairwise precision and recall, cluster metrics (B³, pairwise F1) on a labeled sample. Active learning to label the most informative uncertain pairs. **Clerical review** queue for mid-confidence matches.

**Production concerns:** incremental or streaming ER for new records (match against the existing index rather than re-running everything), stable entity IDs over time, scalability (Spark/Splink, Zingg), PII and privacy (hashing, access controls), and explainability for regulated uses (KYC/AML).

**Trade-off:** false merges (combining two customers) are usually worse than missed matches, so set thresholds asymmetrically.

---

← [Chapter 13. Deep Learning](13-deep-learning.md) · [Table of contents](../README.md) · [Chapter 15. Computer Vision](15-computer-vision.md) →
