# Chapter 20. Retrieval-Augmented Generation

_Part IV — Generative AI and LLM Engineering_

RAG architecture, chunking, hybrid retrieval, reranking, advanced retrieval patterns, evaluation and failure analysis.

This chapter contains **70 questions**.

## Contents

- **20.1** [What is Retrieval-Augmented Generation (RAG), and why is it important?](#201-what-is-retrieval-augmented-generation-rag-and-why-is-it-important)
- **20.2** [Explain the architecture of a basic RAG system.](#202-explain-the-architecture-of-a-basic-rag-system)
- **20.3** [What are the key components of a RAG pipeline?](#203-what-are-the-key-components-of-a-rag-pipeline)
- **20.4** [What are chunking strategies, and how do you choose the right chunk size?](#204-what-are-chunking-strategies-and-how-do-you-choose-the-right-chunk-size)
- **20.5** [Compare fixed-size chunking, semantic chunking, and recursive chunking.](#205-compare-fixed-size-chunking-semantic-chunking-and-recursive-chunking)
- **20.6** [What are embedding models, and how do they convert text to vectors?](#206-what-are-embedding-models-and-how-do-they-convert-text-to-vectors)
- **20.7** [How do you choose an embedding model for your RAG system?](#207-how-do-you-choose-an-embedding-model-for-your-rag-system)
- **20.8** [Explain Agentic RAG.](#208-explain-agentic-rag)
- **20.9** [What is hybrid search, and why is it better than pure vector search?](#209-what-is-hybrid-search-and-why-is-it-better-than-pure-vector-search)
- **20.10** [What is re-ranking, and how does it improve RAG retrieval quality?](#2010-what-is-re-ranking-and-how-does-it-improve-rag-retrieval-quality)
- **20.11** [How do you handle multi-document and multi-hop questions in RAG?](#2011-how-do-you-handle-multi-document-and-multi-hop-questions-in-rag)
- **20.12** [What is the "lost in the middle" problem in RAG systems?](#2012-what-is-the-lost-in-the-middle-problem-in-rag-systems)
- **20.13** [How do you evaluate a RAG system? Explain faithfulness, relevance, and context precision/recall.](#2013-how-do-you-evaluate-a-rag-system-explain-faithfulness-relevance-and-context-precisionrecall)
- **20.14** [Explain Self-RAG. How does the model decide when to retrieve?](#2014-explain-self-rag-how-does-the-model-decide-when-to-retrieve)
- **20.15** [What is GraphRAG, and when would you use it over traditional RAG?](#2015-what-is-graphrag-and-when-would-you-use-it-over-traditional-rag)
- **20.16** [How do you handle structured data (tables, SQL databases) in a RAG pipeline?](#2016-how-do-you-handle-structured-data-tables-sql-databases-in-a-rag-pipeline)
- **20.17** [What are the common failure modes of RAG systems, and how do you debug them?](#2017-what-are-the-common-failure-modes-of-rag-systems-and-how-do-you-debug-them)
- **20.18** [How do you handle document updates and maintain freshness in a RAG system?](#2018-how-do-you-handle-document-updates-and-maintain-freshness-in-a-rag-system)
- **20.19** [How do you optimize RAG for latency in production?](#2019-how-do-you-optimize-rag-for-latency-in-production)
- **20.20** [What is the role of metadata filtering in RAG systems?](#2020-what-is-the-role-of-metadata-filtering-in-rag-systems)
- **20.21** [Compare RAG vs fine-tuning. When would you use each?](#2021-compare-rag-vs-fine-tuning-when-would-you-use-each)
- **20.22** [What is query transformation in RAG (HyDE, query decomposition, step-back prompting)?](#2022-what-is-query-transformation-in-rag-hyde-query-decomposition-step-back-prompting)
- **20.23** [How do you implement citation and source attribution in RAG?](#2023-how-do-you-implement-citation-and-source-attribution-in-rag)
- **20.24** [How do you scale a RAG system to millions of documents?](#2024-how-do-you-scale-a-rag-system-to-millions-of-documents)
- **20.25** [What is parent-child chunking, and how does it improve retrieval?](#2025-what-is-parent-child-chunking-and-how-does-it-improve-retrieval)
- **20.26** [Your RAG system is hallucinating despite having the right context. How do you fix it?](#2026-your-rag-system-is-hallucinating-despite-having-the-right-context-how-do-you-fix-it)
- **20.27** [Your RAG chunk overlap causes redundant results. How do you reduce redundancy?](#2027-your-rag-chunk-overlap-causes-redundant-results-how-do-you-reduce-redundancy)
- **20.28** [Your RAG retrieval is too slow with a large knowledge base. How do you speed it up?](#2028-your-rag-retrieval-is-too-slow-with-a-large-knowledge-base-how-do-you-speed-it-up)
- **20.29** [Your RAG system returns duplicate results. How do you deduplicate?](#2029-your-rag-system-returns-duplicate-results-how-do-you-deduplicate)
- **20.30** [Your RAG system needs per-user access control on internal documents. How do you implement it?](#2030-your-rag-system-needs-per-user-access-control-on-internal-documents-how-do-you-implement-it)
- **20.31** [Your RAG system fails on domain-specific jargon. How do you fix it?](#2031-your-rag-system-fails-on-domain-specific-jargon-how-do-you-fix-it)
- **20.32** [Your text-only RAG system now needs to handle images and tables. How do you extend it?](#2032-your-text-only-rag-system-now-needs-to-handle-images-and-tables-how-do-you-extend-it)
- **20.33** [Your RAG knowledge base gets updated frequently and needs versioning. How do you manage it?](#2033-your-rag-knowledge-base-gets-updated-frequently-and-needs-versioning-how-do-you-manage-it)
- **20.34** [Your RAG system fails on multi-hop questions that require combining multiple facts. How do you fix it?](#2034-your-rag-system-fails-on-multi-hop-questions-that-require-combining-multiple-facts-how-do-you-fix-it)
- **20.35** [Your enterprise RAG system returns contradictory answers from different source documents. How do you resolve conflicts?](#2035-your-enterprise-rag-system-returns-contradictory-answers-from-different-source-documents-how-do-you-resolve-conflicts)
- **20.36** [Your RAG system returns outdated answers from an evolving knowledge base. How do you keep it current?](#2036-your-rag-system-returns-outdated-answers-from-an-evolving-knowledge-base-how-do-you-keep-it-current)
- **20.37** [Your RAG system struggles with PDF documents containing tables and layouts. How do you fix PDF parsing?](#2037-your-rag-system-struggles-with-pdf-documents-containing-tables-and-layouts-how-do-you-fix-pdf-parsing)
- **20.38** [What is Retrieval-Augmented Generation (RAG)?](#2038-what-is-retrieval-augmented-generation-rag)
- **20.39** [What is a RAG pipeline, at a high level?](#2039-what-is-a-rag-pipeline-at-a-high-level)
- **20.40** [Why does chunking strategy matter so much in RAG?](#2040-why-does-chunking-strategy-matter-so-much-in-rag)
- **20.41** [What is hybrid search and why combine methods?](#2041-what-is-hybrid-search-and-why-combine-methods)
- **20.42** [What role does reranking play in a retrieval pipeline?](#2042-what-role-does-reranking-play-in-a-retrieval-pipeline)
- **20.43** [How should retrieval quality be judged?](#2043-how-should-retrieval-quality-be-judged)
- **20.44** [What are grounded citations?](#2044-what-are-grounded-citations)
- **20.45** [What is LlamaIndex primarily focused on?](#2045-what-is-llamaindex-primarily-focused-on)
- **20.46** [How do knowledge graphs/ontologies differ from flat text chunks for retrieval?](#2046-how-do-knowledge-graphsontologies-differ-from-flat-text-chunks-for-retrieval)
- **20.47** [Walk through the architecture of a production-grade RAG pipeline using hybrid retrieval and reranking.](#2047-walk-through-the-architecture-of-a-production-grade-rag-pipeline-using-hybrid-retrieval-and-reranking)
- **20.48** [Why combine dense vector search with keyword (sparse) search instead of using just one?](#2048-why-combine-dense-vector-search-with-keyword-sparse-search-instead-of-using-just-one)
- **20.49** [Why do you need a reranker if you already have a good retriever?](#2049-why-do-you-need-a-reranker-if-you-already-have-a-good-retriever)
- **20.50** [How do you evaluate and reduce hallucination in a RAG system?](#2050-how-do-you-evaluate-and-reduce-hallucination-in-a-rag-system)
- **20.51** [Explain a complete RAG architecture.](#2051-explain-a-complete-rag-architecture)
- **20.52** [Why can naive vector search produce poor results?](#2052-why-can-naive-vector-search-produce-poor-results)
- **20.53** [Compare dense retrieval vs sparse retrieval.](#2053-compare-dense-retrieval-vs-sparse-retrieval)
- **20.54** [What is reranking, and why does it improve RAG?](#2054-what-is-reranking-and-why-does-it-improve-rag)
- **20.55** [Compare bi-encoder vs cross-encoder retrieval.](#2055-compare-bi-encoder-vs-cross-encoder-retrieval)
- **20.56** [What is contextual compression?](#2056-what-is-contextual-compression)
- **20.57** [What is the lost-in-the-middle problem?](#2057-what-is-the-lost-in-the-middle-problem)
- **20.58** [How would you select an optimal chunk size?](#2058-how-would-you-select-an-optimal-chunk-size)
- **20.59** [What happens if chunks are too small?](#2059-what-happens-if-chunks-are-too-small)
- **20.60** [What happens if chunks are too large?](#2060-what-happens-if-chunks-are-too-large)
- **20.61** [Explain retrieval precision, retrieval recall, faithfulness, and answer relevance.](#2061-explain-retrieval-precision-retrieval-recall-faithfulness-and-answer-relevance)
- **20.62** [What is GraphRAG, and when is it better than traditional RAG?](#2062-what-is-graphrag-and-when-is-it-better-than-traditional-rag)
- **20.63** [How would you handle tables, PDFs, images, and scanned documents in RAG?](#2063-how-would-you-handle-tables-pdfs-images-and-scanned-documents-in-rag)
- **20.64** [How would you prevent an LLM from answering using its pretrained knowledge instead of retrieved documents?](#2064-how-would-you-prevent-an-llm-from-answering-using-its-pretrained-knowledge-instead-of-retrieved-documents)
- **20.65** [Your embeddings retrieve semantically similar but factually incorrect documents. How do you improve retrieval?](#2065-your-embeddings-retrieve-semantically-similar-but-factually-incorrect-documents-how-do-you-improve-retrieval)
- **20.66** [Your RAG system retrieves the correct document but the LLM gives the wrong answer. Where is the problem?](#2066-your-rag-system-retrieves-the-correct-document-but-the-llm-gives-the-wrong-answer-where-is-the-problem)
- **20.67** [The LLM answers correctly with 5 retrieved documents but incorrectly with 20. Why?](#2067-the-llm-answers-correctly-with-5-retrieved-documents-but-incorrectly-with-20-why)
- **20.68** [Explain HyDE, query decomposition, multi-query and step-back prompting.](#2068-explain-hyde-query-decomposition-multi-query-and-step-back-prompting)
- **20.69** [How would you debug an LLM/RAG system that hallucinates?](#2069-how-would-you-debug-an-llmrag-system-that-hallucinates)
- **20.70** [What is Open Knowledge Format (OKF), and how is it different from RAG?](#2070-what-is-open-knowledge-format-okf-and-how-is-it-different-from-rag)

---

### 20.1 What is Retrieval-Augmented Generation (RAG), and why is it important?

**Retrieval-Augmented Generation (RAG)** is an AI architecture that combines a **retrieval system** (which finds relevant information from external sources) with a **large language model (LLM)** (which generates natural language responses). Instead of relying only on what the model learned during training, RAG allows the model to access up-to-date or domain-specific information at the time of answering.

#### How RAG Works

A typical RAG pipeline looks like this:

1. **User asks a question**
   - Example: "What is our company's vacation policy?"
2. **Retrieve relevant information**
   - The system searches a knowledge source such as:
     - Internal documents
     - PDFs
     - Databases
     - Wikis
     - Documentation
     - Websites
3. **Augment the prompt**
   - The retrieved information is added to the user's question.
4. **Generate the answer**
   - The LLM uses both the user's query and the retrieved context to produce a response grounded in the retrieved information.

```txt
User Question
      │
      ▼
Retriever (Search/Vector Database)
      │
Relevant Documents
      │
      ▼
LLM + Retrieved Context
      │
      ▼
Accurate, Context-Aware Response
```

#### Why RAG Is Important

RAG addresses several limitations of standalone language models.

| Without RAG | With RAG |
| --- | --- |
| Relies only on training data | Uses current external knowledge |
| Knowledge can become outdated | Can access the latest information |
| More likely to hallucinate | Grounds answers in retrieved documents |
| Limited to general knowledge | Can answer organization-specific questions |
| Retraining is needed to update knowledge | Simply update the knowledge base |

#### Key Benefits

- **Reduces hallucinations:** Because answers are based on retrieved documents, the model is less likely to invent facts.
- **Keeps information current:** New documents can be indexed without retraining the model.
- **Supports private data:** Organizations can use internal documents without exposing them during model training.
- **Improves transparency:** Systems can often cite the documents used to generate an answer.
- **Cost-effective:** Updating a document store is generally much cheaper and faster than fine-tuning a large model.

#### Example

Imagine a customer support chatbot.

**Without RAG**

- User: "What is the warranty period for Product X?"
- Model: May answer based on outdated training data or guess.

**With RAG**

1. Searches the latest product manual.
2. Retrieves the warranty section.
3. Answers:
   > "Product X includes a 3-year limited warranty covering manufacturing defects," based on the current documentation.

#### Common Components of a RAG System

- **Embedding model:** Converts text into numerical vectors.
- **Vector database:** Stores embeddings for efficient similarity search (e.g., documents split into chunks).
- **Retriever:** Finds the most relevant chunks for a query.
- **LLM:** Generates the final response using the retrieved context.
- **Optional reranker:** Reorders retrieved results to improve relevance before generation.

#### Typical Use Cases

- Enterprise knowledge assistants
- Customer support chatbots
- Legal document search
- Medical information systems
- Research assistants
- Code assistants
- Technical documentation Q&A

#### RAG vs. Fine-Tuning

| RAG | Fine-Tuning |
| --- | --- |
| Adds external knowledge | Changes the model's learned parameters |
| Easy to update | Requires retraining to incorporate new knowledge |
| Best for factual, changing information | Best for changing the model's behavior, style, or specialized reasoning |
| Lower cost | Higher computational cost |

Many production AI systems use **both**:

- **Fine-tuning** to teach the model how to respond (tone, format, task-specific behavior).
- **RAG** to provide accurate, current, and organization-specific information.

### 20.2 Explain the architecture of a basic RAG system.

A **basic RAG (Retrieval-Augmented Generation) system** has two main phases: **indexing** and **retrieval + generation**.

#### 1. Document Ingestion & Indexing

Before users ask questions, documents are prepared and stored.

- **Load documents** — PDFs, web pages, databases, Word files, etc.
- **Chunking** — Split documents into smaller pieces.
- **Embedding** — Convert each chunk into a vector using an embedding model.
- **Store** — Save the vectors and their associated text/metadata in a vector database.

```txt
Documents
    ↓
Document Loader
    ↓
Chunking
    ↓
Embedding Model
    ↓
Vector Database
```

#### 2. Query & Retrieval

When the user asks a question:

- Convert the **user query into an embedding**.
- Search the vector database for the most semantically similar chunks.
- Retrieve the **Top-K relevant chunks**.

```txt
User Query
    ↓
Query Embedding
    ↓
Vector Database
    ↓
Top-K Relevant Chunks
```

#### 3. Augmentation

The retrieved chunks are combined with the user's question to create the context given to the LLM.

```txt
User Query + Retrieved Context
              ↓
          Prompt
```

For example:

```txt
System: Answer using the provided context.

Context:
[Retrieved document chunks]

Question:
What is the company's leave policy?
```

#### 4. Generation

The LLM receives the augmented prompt and generates the final answer.

```txt
Augmented Prompt
      ↓
     LLM
      ↓
Final Answer
```

#### Complete Basic RAG Architecture

```txt
             INDEXING PHASE

Documents
   ↓
Loader
   ↓
Chunking
   ↓
Embedding Model
   ↓
Vector Database
        │
        │
        │
        ▼
    RETRIEVAL PHASE

User Query
   ↓
Query Embedding
   ↓
Vector Search
   ↓
Top-K Documents
   ↓
Context + Query
   ↓
Prompt
   ↓
LLM
   ↓
Generated Answer
```

#### Key Components

- **Document Loader:** Reads source documents.
- **Chunker:** Splits documents into manageable pieces.
- **Embedding Model:** Converts text into vectors.
- **Vector Database:** Stores and searches embeddings.
- **Retriever:** Finds relevant chunks.
- **Prompt/Context Builder:** Combines retrieved information with the query.
- **LLM:** Generates the final response.

**In short:**

**RAG = Retrieve relevant knowledge → Augment the prompt → Generate an answer.**

### 20.3 What are the key components of a RAG pipeline?

#### Key Components of a RAG Pipeline

1. **Document Loader**
   - Loads data from PDFs, websites, databases, Word files, etc.
2. **Text Chunker**
   - Splits documents into smaller chunks for efficient retrieval.
3. **Embedding Model**
   - Converts text chunks and user queries into vector representations.
4. **Vector Database**
   - Stores embeddings and metadata and enables similarity search.
   - Examples: Pinecone, Weaviate, Milvus, Qdrant, pgvector.
5. **Retriever**
   - Searches the vector database and retrieves the most relevant chunks for the user's query.
6. **Reranker — Optional**
   - Re-ranks retrieved chunks based on their relevance to improve retrieval quality.
7. **Context Builder**
   - Combines the retrieved chunks with the user's query and constructs the prompt.
8. **LLM / Generator**
   - Uses the retrieved context to generate the final answer.
9. **Evaluation & Monitoring**
   - Measures retrieval and generation quality, such as relevance, faithfulness, latency, and hallucination rate.

#### Basic Flow

```txt
Documents
    ↓
Load → Chunk → Embed → Vector Database
                         ↑
                         │
User Query → Embed → Retrieve → Rerank
                              ↓
                         Context Builder
                              ↓
                             LLM
                              ↓
                         Final Answer
```

**In short:**

**Ingestion → Chunking → Embedding → Storage → Retrieval → Reranking → Context Construction → Generation → Evaluation**.

### 20.4 What are chunking strategies, and how do you choose the right chunk size?

#### Chunking Strategies

**Chunking** is the process of splitting large documents into smaller pieces before creating embeddings and storing them in a vector database. Good chunking improves retrieval accuracy and helps the LLM receive relevant context.

#### Common Chunking Strategies

1. **Fixed-size chunking**
   - Split text after a fixed number of tokens or characters.
   - Simple and fast.
   - Example: 500 tokens per chunk.
2. **Fixed-size with overlap**
   - Chunks have some shared content.
   - Example: 500-token chunks with 50-token overlap.
   - Helps preserve context across chunk boundaries.
3. **Sentence-based chunking**
   - Split text based on sentences.
   - Keeps complete sentences together.
4. **Paragraph-based chunking**
   - Uses paragraphs as natural boundaries.
   - Works well for structured documents.
5. **Recursive chunking**
   - Attempts to split using increasingly smaller separators, such as:
   - Document → sections → paragraphs → sentences → words.
   - Commonly used for general-purpose RAG systems.
6. **Semantic chunking**
   - Groups sentences based on their semantic similarity.
   - Creates chunks around changes in meaning rather than a fixed size.
7. **Structure-aware chunking**
   - Uses document structure such as headings, sections, tables, and lists.
   - Particularly useful for technical documentation and structured reports.

#### How to Choose Chunk Size

There is **no universally optimal chunk size**. It depends on the data and retrieval task.

Consider:

- **Document structure:** Structured documents often benefit from section- or paragraph-based chunks.
- **Information density:** Dense technical content may need smaller chunks.
- **Question complexity:** Simple factual questions can work well with smaller chunks; complex questions may require larger contextual chunks.
- **Embedding quality:** Chunks should contain enough meaningful information for the embedding to represent them accurately.
- **LLM context window:** Larger chunks consume more context and may introduce irrelevant information.
- **Retrieval precision:** Smaller chunks can improve precision but may lose surrounding context.
- **Retrieval recall:** Larger chunks preserve more context but may contain more irrelevant information.

#### Practical Starting Point

For a general text-based RAG system, you might start with:

```txt
Chunk size: 300–800 tokens
Overlap:    10–20%
```

Then evaluate different configurations using your actual dataset and queries.

#### Key Trade-off

```txt
Smaller Chunks
    ↓
Higher retrieval precision
Less context
More chunks

Larger Chunks
    ↓
More context
Lower retrieval precision
Fewer chunks
```

**Best practice:** Don't choose chunk size purely by a fixed rule. Start with a reasonable baseline, then evaluate **retrieval relevance, answer faithfulness, recall, and latency** on a representative test set and tune the chunk size accordingly.

### 20.5 Compare fixed-size chunking, semantic chunking, and recursive chunking.

#### Fixed-Size vs Semantic vs Recursive Chunking

| Strategy | How it works | Advantages | Disadvantages | Best for |
| --- | --- | --- | --- | --- |
| **Fixed-size** | Splits text into chunks of a fixed token/character size | Simple, fast, predictable | Can split sentences or concepts | Simple documents, baseline RAG |
| **Semantic** | Groups text based on meaning and semantic similarity | Preserves related concepts; often better contextual coherence | More computationally expensive; requires tuning | Research papers, technical content, complex knowledge bases |
| **Recursive** | Splits using a hierarchy of separators, e.g. sections → paragraphs → sentences | Preserves natural structure while controlling chunk size | More complex than fixed-size; may still split concepts | General-purpose RAG, documentation, mixed documents |

#### 1. Fixed-Size Chunking

Example:

```txt
Document
   ↓
500 tokens
   ↓
500 tokens
   ↓
500 tokens
```

Often combined with **overlap**, such as 500-token chunks with 50-token overlap.

**Pros**

- Very easy to implement
- Fast and predictable
- Good baseline

**Cons**

- May cut a concept or sentence in half
- Doesn't understand document structure

---

#### 2. Semantic Chunking

The system analyzes the meaning of neighboring sentences and creates a new chunk when the topic changes significantly.

```txt
Topic A → Topic A → Topic A
              ↓
           Chunk 1

Topic B → Topic B → Topic B
              ↓
           Chunk 2
```

**Pros**

- Better semantic coherence
- Can improve retrieval relevance
- Avoids arbitrary boundaries

**Cons**

- More expensive
- More computationally intensive
- Results can vary depending on the embedding model and thresholds

---

#### 3. Recursive Chunking

The system tries larger structural boundaries first and progressively splits when the resulting chunk is too large.

```txt
Document
   ↓
Sections
   ↓
Paragraphs
   ↓
Sentences
   ↓
Words
```

For example, it might first try to keep a complete paragraph together. If it's too large, it splits that paragraph into sentences.

**Pros**

- Preserves natural document structure
- Good balance between simplicity and quality
- Works well across many document types

**Cons**

- Not truly semantic; boundaries are primarily structural
- Requires separator and chunk-size configuration

### 20.6 What are embedding models, and how do they convert text to vectors?

#### What Are Embedding Models?

**Embedding models** are machine learning models that convert text, images, or other data into **numerical vectors** that capture their semantic meaning.

For text, the result is typically a high-dimensional vector:

```txt
"Machine learning is a subset of AI"
              ↓
        Embedding Model
              ↓
[0.21, -0.47, 0.83, ..., 0.12]
```

The vector may contain hundreds or thousands of dimensions.

#### How Text Is Converted to a Vector

A typical process is:

1. **Tokenization**
   - The text is divided into tokens.
   - Example:

   ```txt
   "I love machine learning"
          ↓
   ["I", "love", "machine", "learning"]
   ```
2. **Neural encoding**
   - The tokens are passed through a trained Transformer or similar neural network.
   - The model produces contextual representations for the text.
3. **Pooling / representation**
   - The token-level representations are combined into a single fixed-size representation for the whole text.
4. **Embedding vector**
   - The resulting representation is the text embedding.

```txt
Text
 ↓
Tokenizer
 ↓
Tokens
 ↓
Embedding / Transformer Model
 ↓
Contextual Representations
 ↓
Pooling
 ↓
Vector
```

#### Why Are Embeddings Useful?

Texts with similar meanings tend to have vectors that are **close together in vector space**.

For example:

```txt
"How do I reset my password?"
              ↓
          Vector A

"I forgot my password. How can I change it?"
              ↓
          Vector B
```

Even though the wording is different, their embeddings should be relatively close because their meanings are similar.

This allows systems to perform **semantic search** using measures such as cosine similarity or dot product.

#### Embeddings in RAG

In a RAG system:

```txt
Documents
   ↓
Chunking
   ↓
Embedding Model
   ↓
Document Vectors
   ↓
Vector Database

User Query
   ↓
Same Embedding Model
   ↓
Query Vector
   ↓
Similarity Search
   ↓
Relevant Chunks
   ↓
LLM
```

The important idea is:

**Embedding models transform text into vectors where semantic relationships can be measured mathematically, enabling semantic search, retrieval, clustering, and recommendation systems.**

### 20.7 How do you choose an embedding model for your RAG system?

#### How to Choose an Embedding Model for RAG

I would evaluate an embedding model based on **retrieval quality, domain fit, language support, performance, and cost**.

1. **Retrieval quality**
   - Evaluate how well the model retrieves the correct chunks for real user queries.
   - Measure metrics such as **Recall@K, Precision@K, MRR, and nDCG**.
2. **Domain fit**
   - General-purpose models work well for common text.
   - For specialized domains such as **legal, medical, scientific, or code**, choose a model that performs well on that domain.
3. **Multilingual support**
   - If the RAG system handles multiple languages, choose a strong multilingual embedding model.
   - Test cross-lingual retrieval if queries and documents can use different languages.
4. **Embedding dimensions**
   - Higher dimensions can potentially capture more information but increase **storage and search costs**.
   - Choose the smallest dimension that provides acceptable retrieval quality.
5. **Latency and throughput**
   - For real-time RAG, benchmark:
     - Embedding latency
     - Documents processed per second
     - Query throughput
   - Larger models generally require more computation.
6. **Context length**
   - Check how much text the model can embed in one request.
   - This matters when your chunks are relatively large.
7. **Cost and deployment**
   - Compare API pricing versus self-hosting.
   - Consider GPU/CPU requirements, memory usage, and operational complexity.
8. **Similarity compatibility**
   - Check whether the model is designed for **cosine similarity, dot product, or another distance metric**.
   - Configure the vector database accordingly.

#### Practical Selection Process

```txt
Candidate Models
       ↓
Benchmark on Your Dataset
       ↓
Measure Recall@K / MRR / nDCG
       ↓
Check Latency + Cost
       ↓
Test Multilingual / Domain Performance
       ↓
Choose Best Quality-Cost Trade-off
```

#### Important Interview Point

**Don't choose an embedding model solely based on benchmark scores.** Build a representative evaluation dataset containing real queries and known relevant documents, then compare candidate models on **retrieval quality, latency, cost, language/domain performance, and infrastructure requirements**.

For a production RAG system, **retrieval quality on your actual data is usually the most important criterion.**

### 20.8 Explain Agentic RAG.

#### Agentic RAG

**Agentic RAG** is an advanced RAG architecture where an **AI agent dynamically decides how, when, and where to retrieve information** instead of following a fixed retrieval pipeline.

In traditional RAG, the flow is usually:

```txt
Query
  ↓
Retrieve Top-K Documents
  ↓
LLM
  ↓
Answer
```

In **Agentic RAG**, the LLM/agent can reason about the task and perform multiple retrieval actions:

```txt
User Query
    ↓
   Agent
    ↓
Plan / Decide
    ↓
Choose Tool or Knowledge Source
    ↓
Retrieve Information
    ↓
Evaluate Results
    ↓
Need More Information?
   ↙          ↘
 Yes           No
  ↓             ↓
Retrieve      Generate
Again          Answer
```

#### Key Components

1. **Agent / Orchestrator**
   - Understands the user's request.
   - Decides what information is needed.
2. **Multiple Retrieval Tools**
   - Vector database
   - Keyword/BM25 search
   - SQL database
   - Web search
   - APIs
   - Internal knowledge bases
3. **Planning**
   - Breaks complex questions into smaller sub-questions.
   - Determines the order of retrieval steps.
4. **Iterative Retrieval**
   - Retrieves information.
   - Evaluates whether it is sufficient.
   - Performs additional searches if necessary.
5. **Reasoning / Synthesis**
   - Combines information from multiple sources.
   - Produces the final grounded response.

#### Example

Suppose the user asks:

> "Compare our company's Q1 and Q2 sales and explain why sales changed."

A basic RAG system might retrieve documents containing "Q1 sales" and "Q2 sales."

An **Agentic RAG** system could:

```txt
1. Find Q1 sales data
        ↓
2. Find Q2 sales data
        ↓
3. Query the sales database
        ↓
4. Compare the numbers
        ↓
5. Search company reports for explanations
        ↓
6. Determine whether the evidence is sufficient
        ↓
7. Generate the final analysis
```

#### Traditional RAG vs Agentic RAG

| Traditional RAG | Agentic RAG |
| --- | --- |
| Fixed retrieval pipeline | Dynamic retrieval |
| Usually one retrieval step | Can perform multiple retrieval steps |
| Often one knowledge source | Can use multiple tools/sources |
| Simple questions | Complex, multi-step questions |
| Lower complexity | Higher complexity |
| More predictable | More autonomous and flexible |

#### Main Advantage

The biggest advantage is **adaptive retrieval**.

Instead of assuming that the first Top-K results are enough, the agent can ask:

> "Do I have enough relevant information to answer this question?"

If not, it can **search again, use another tool, refine the query, or break the problem into sub-questions**.

### 20.9 What is hybrid search, and why is it better than pure vector search?

#### Hybrid Search

**Hybrid search** combines **vector (semantic) search** with **keyword-based search** to retrieve relevant documents.

```txt
User Query
    ↓
 ┌───────────────┐
 │               │
 ▼               ▼
Vector Search   Keyword Search
 │               │
 │               │
 └───────┬───────┘
         ↓
   Combine / Rerank
         ↓
   Relevant Results
```

#### Why Not Pure Vector Search?

Vector search is good at understanding **meaning**, but it can struggle with:

- Exact names
- Product IDs
- Error codes
- Technical terms
- Acronyms
- Numbers
- Rare keywords

For example:

> **"How do I fix error `ERR_CONNECTION_RESET`?"**

A keyword search can match the exact error code very effectively, while a vector search might retrieve documents about general network connection problems.

#### Hybrid Search Example

Query:

> "AWS Lambda timeout error 504"

**Vector search** may find:

- Articles about Lambda timeouts
- Serverless execution problems
- API Gateway timeout issues

**Keyword search** may specifically match:

- "AWS Lambda"
- "504"
- "timeout"
- Exact error messages

Combining both gives the retriever a better chance of finding the **right document**.

#### Hybrid Search vs Pure Vector Search

| Feature | Pure Vector Search | Hybrid Search |
| --- | --- | --- |
| Semantic similarity | Excellent | Excellent |
| Exact keyword matching | Weaker | Strong |
| IDs/error codes | Weaker | Strong |
| Rare technical terms | Can struggle | Better |
| Natural-language queries | Strong | Strong |
| Robustness | Moderate | Higher |

#### How It Is Usually Implemented

A common approach is:

```txt
Query
 ↓
 ├──→ Dense Vector Search → Results
 │
 └──→ BM25 / Keyword Search → Results
                         ↓
                  Score Fusion
                         ↓
                     Reranker
                         ↓
                  Top-K Results
```

The scores from both retrieval methods can be combined using approaches such as **weighted score fusion** or **Reciprocal Rank Fusion (RRF)**.

### 20.10 What is re-ranking, and how does it improve RAG retrieval quality?

#### Re-ranking in RAG

**Re-ranking** is a second-stage retrieval process that takes the initial search results and **reorders them based on how relevant they are to the user's query**.

Instead of sending the first retrieved documents directly to the LLM:

```txt
Query
  ↓
Vector / Hybrid Search
  ↓
Top 20 Results
  ↓
Reranker
  ↓
Best 5 Results
  ↓
LLM
```

#### How It Works

1. **Initial retrieval**
   - A vector database or hybrid search retrieves a relatively large candidate set, such as the top 20–100 chunks.
   - This stage is optimized for **speed and recall**.
2. **Re-ranking**
   - A more powerful model examines the **query and each retrieved chunk together**.
   - It assigns a relevance score to each pair.
   - Results are sorted by those scores.
3. **Final retrieval**
   - Only the highest-quality chunks are passed to the LLM.

#### Why It Improves RAG

A vector search model generally compares embeddings independently:

```txt
Query Embedding ↔ Document Embedding
```

A reranker can perform deeper query-document relevance analysis:

```txt
Query + Document
       ↓
   Reranker
       ↓
Relevance Score
```

This helps distinguish between documents that are **semantically similar but not actually relevant**.

#### Example

Query:

> "How can I reduce AWS Lambda cold-start latency?"

Initial retrieval might return:

1. Lambda cold starts — **highly relevant**
2. Lambda timeout configuration — somewhat relevant
3. Lambda pricing — related but irrelevant
4. Lambda memory optimization — potentially relevant
5. AWS EC2 startup time — less relevant

The reranker can push the most relevant chunks to the top before they reach the LLM.

#### Common Reranking Approaches

- **Cross-encoder reranker:** Processes query and document together and directly predicts relevance.
- **LLM-based reranking:** Uses an LLM to judge which retrieved documents best answer the query.
- **Score fusion:** Combines rankings from multiple retrieval systems.

#### Key Trade-off

| Initial Retrieval | Re-ranking |
| --- | --- |
| Fast | Slower |
| High recall | Higher precision |
| Retrieves many candidates | Selects the best candidates |
| Usually cheaper | More computationally expensive |

### 20.11 How do you handle multi-document and multi-hop questions in RAG?

#### Handling Multi-Document and Multi-Hop Questions in RAG

For complex questions, I use an **iterative retrieval and reasoning process** rather than relying on a single retrieval step.

1. **Query decomposition**
   - Break the question into smaller sub-questions.
   - Example: *"How did Company A's acquisition of Company B affect its revenue?"*
     - Retrieve acquisition information.
     - Retrieve Company B's financial information.
     - Retrieve Company A's revenue data.
2. **Independent retrieval**
   - Search the knowledge base separately for each sub-question.
   - Use **hybrid search** when both semantic and exact matching are important.
3. **Evidence linking**
   - Connect information retrieved from different documents.
   - Identify relationships between entities, dates, events, or facts.
4. **Iterative / multi-hop retrieval**
   - Use the result from one retrieval step to formulate the next query.
   - Continue until enough evidence is collected.
5. **Evidence validation**
   - Check whether the retrieved documents actually support the intermediate conclusions.
   - Remove conflicting or low-quality evidence.
6. **Final synthesis**
   - Pass the relevant evidence from all hops to the LLM.
   - Generate an answer that explains how the pieces connect.

#### Typical Architecture

```txt
Complex Query
     ↓
Query Decomposition
     ↓
Sub-queries
     ↓
Retrieve → Rerank → Evidence
     ↓
Intermediate Results
     ↓
Generate Next Query
     ↓
Retrieve Again
     ↓
Evidence Validation
     ↓
Final Synthesis
```

### 20.12 What is the "lost in the middle" problem in RAG systems?

#### Lost in the Middle

The **"lost in the middle" problem** occurs when an LLM receives a long context containing many retrieved documents but **fails to effectively use information located in the middle of that context**.

The model tends to pay more attention to information near the **beginning and end** of the context.

```txt
Context

[Important information]        ← Often remembered well
        ↓
[Important information]
[Important information]
[Important information]        ← May be overlooked
[Important information]
        ↓
[Important information]        ← Often remembered well
```

#### Why It Matters in RAG

If RAG retrieves many chunks and places them into a long prompt:

```txt
Query
 ↓
Top 20 Retrieved Chunks
 ↓
Large Context
 ↓
LLM
```

The **most relevant chunk might be placed in the middle**, causing the model to ignore it or give it less weight.

This can lead to:

- Incorrect answers
- Missing important evidence
- Lower faithfulness
- Poor performance despite good retrieval

#### How to Reduce It

- **Rerank** retrieved chunks so the most relevant evidence is prioritized.
- **Reduce the number of chunks** passed to the LLM.
- Put the **most important evidence near the beginning or end** of the context.
- Use **context compression** to remove irrelevant information.
- Use **query-focused retrieval** rather than retrieving large amounts of loosely related content.
- For complex questions, use **iterative retrieval** instead of putting everything into one huge prompt.

### 20.13 How do you evaluate a RAG system? Explain faithfulness, relevance, and context precision/recall.

#### Evaluating a RAG System

A RAG system should be evaluated at **both the retrieval and generation levels**. The key metrics are **context recall, context precision, relevance, and faithfulness**.

#### 1. Faithfulness

**Faithfulness measures whether the generated answer is actually supported by the retrieved context.**

Example:

```txt
Context:
"Product X has a 2-year warranty."

Answer:
"Product X has a 2-year warranty."
→ Faithful
```

If the model says:

```txt
"Product X has a 5-year warranty."
→ Not faithful
```

even if the answer sounds plausible.

**Goal:** Minimize hallucinations and unsupported claims.

---

#### 2. Answer Relevance

**Answer relevance measures whether the generated answer actually addresses the user's question.**

Example:

```txt
Question:
"What is the company's refund period?"

Answer:
"The refund period is 30 days."
→ Relevant
```

An answer discussing the company's shipping policy instead would have **low relevance**.

**Goal:** Generate answers that directly address the query.

---

#### 3. Context Precision

**Context precision measures how much of the retrieved context is actually relevant to the question.**

Suppose the retriever returns 5 chunks:

```txt
Chunk 1 → Relevant
Chunk 2 → Relevant
Chunk 3 → Irrelevant
Chunk 4 → Relevant
Chunk 5 → Irrelevant
```

Only **3/5** are relevant.

High context precision means the retriever is not bringing lots of unnecessary information into the LLM's context.

**Goal:** Retrieve fewer irrelevant chunks.

---

#### 4. Context Recall

**Context recall measures whether the retrieval system found all the important information needed to answer the question.**

Suppose the answer requires information from:

```txt
Document A
Document B
Document C
```

but retrieval only returns A and B.

The system has **low context recall** because an important piece of evidence was missed.

**Goal:** Avoid missing relevant information.

---

#### Precision vs Recall

```txt
Context Precision
→ "Of what I retrieved, how much was relevant?"

Context Recall
→ "Of everything relevant, how much did I retrieve?"
```

#### Putting It Together

```txt
                    RAG Evaluation
                         │
          ┌──────────────┴──────────────┐
          ↓                             ↓
     Retrieval                       Generation
          │                             │
   Precision / Recall          Faithfulness / Relevance
```

### 20.14 Explain Self-RAG. How does the model decide when to retrieve?

#### Self-RAG

**Self-RAG (Self-Reflective Retrieval-Augmented Generation)** is a RAG approach where the model **learns to decide whether it needs external information, retrieves when necessary, and evaluates the retrieved information and its own generated response**.

Unlike standard RAG, which usually retrieves for every query:

```txt
Query → Retrieve → Generate
```

Self-RAG makes retrieval **adaptive**:

```txt
Query
  ↓
Model decides: "Do I need retrieval?"
  ↓
 ┌───────────┴───────────┐
 No                      Yes
 ↓                        ↓
Generate              Retrieve
                           ↓
                    Evaluate Evidence
                           ↓
                    Generate / Revise
```

#### How Does It Decide When to Retrieve?

The model is trained to produce special **reflection signals/tokens** that indicate actions such as:

- **Retrieve** — external information is needed.
- **No retrieval** — the model can answer reliably from its existing knowledge.
- **Relevant** — the retrieved passage is useful.
- **Supported** — the generated claim is supported by the evidence.
- **Useful** — the response adequately answers the question.

For example:

```txt
Question: "What is the capital of France?"
        ↓
Model: No retrieval needed
        ↓
Answer: Paris
```

But:

```txt
Question: "What was Apple's revenue in the latest quarter?"
        ↓
Model: Retrieve
        ↓
Search current information
        ↓
Evaluate retrieved evidence
        ↓
Generate grounded answer
```

#### Why Self-RAG Is Useful

- **Reduces unnecessary retrieval**, improving latency and cost.
- **Improves factual accuracy** by retrieving when knowledge is insufficient.
- **Evaluates retrieved evidence** rather than blindly trusting it.
- **Reduces hallucinations** by checking whether generated claims are supported.
- **Allows iterative correction** when the initial evidence or answer is inadequate.

#### Self-RAG vs Traditional RAG

| Traditional RAG | Self-RAG |
| --- | --- |
| Usually always retrieves | Retrieves conditionally |
| Fixed pipeline | Adaptive pipeline |
| Doesn't inherently evaluate evidence | Reflects on retrieved evidence |
| Less control over generation | Can critique and revise its answer |
| Simpler | More complex |

### 20.15 What is GraphRAG, and when would you use it over traditional RAG?

#### GraphRAG

**GraphRAG (Graph-based Retrieval-Augmented Generation)** enhances traditional RAG by representing information as a **knowledge graph of entities and relationships**, then using those relationships during retrieval.

Instead of treating documents as independent chunks:

```txt
Documents
   ↓
Chunks
   ↓
Vector Database
   ↓
Retrieve Similar Chunks
   ↓
LLM
```

GraphRAG builds a graph:

```txt
Documents
    ↓
Entity & Relationship Extraction
    ↓
Knowledge Graph
    ↓
Graph Retrieval
    ↓
Relevant Entities + Relationships + Documents
    ↓
LLM
```

For example:

```txt
Microsoft
    │
 acquired
    ↓
 LinkedIn
    │
 headquartered_in
    ↓
 San Francisco
```

The graph captures **how pieces of information are connected**, not just how semantically similar they are.

#### Why Use GraphRAG?

Traditional vector RAG works very well for questions such as:

> "What is Microsoft's acquisition strategy?"

But it can struggle with questions requiring **relationships across multiple documents**.

For example:

> "Which companies acquired by Microsoft have subsidiaries in countries where Microsoft also has major offices?"

Answering this requires following multiple relationships:

```txt
Microsoft
   ↓ acquired
Company A
   ↓ subsidiary
Company B
   ↓ located_in
Country X
   ↓
Microsoft office
```

GraphRAG can traverse these relationships to gather the relevant evidence.

#### When to Use GraphRAG

Use GraphRAG when your data contains:

- **Many interconnected entities**
- Complex relationships
- Multi-hop questions
- Organizational hierarchies
- Supply-chain relationships
- Research-paper citations and relationships
- Financial/company relationships
- Large enterprise knowledge bases

#### Traditional RAG vs GraphRAG

| Traditional RAG | GraphRAG |
| --- | --- |
| Retrieves similar chunks | Retrieves entities and relationships |
| Primarily semantic similarity | Uses graph structure + semantics |
| Good for direct questions | Strong for multi-hop questions |
| Simpler and cheaper | More complex and expensive |
| Less explicit relationship modeling | Explicit relationship modeling |

#### Key Trade-off

GraphRAG isn't automatically better. **Building and maintaining the graph adds complexity and cost.**

For straightforward document Q&A, traditional RAG is usually sufficient.

For **relationship-heavy, multi-hop, and global knowledge questions**, GraphRAG can provide substantially better retrieval.

### 20.16 How do you handle structured data (tables, SQL databases) in a RAG pipeline?

#### Handling Structured Data in RAG

For structured data, I don't treat everything like ordinary text. I choose the retrieval method based on the **data format and question type**.

#### 1. Tables

For tables in PDFs, Excel, or documents:

- Extract the table while preserving **rows, columns, headers, and relationships**.
- Convert it into a structured representation such as JSON or Markdown.
- Store useful metadata such as table name, document, page, and columns.
- Embed meaningful textual representations for semantic retrieval.
- For numerical or exact queries, use **structured filtering/calculation** rather than relying only on embeddings.

```txt
Table
 ↓
Table Extraction
 ↓
Structured Representation
 ↓
Metadata + Embeddings
 ↓
Vector / Hybrid Search
 ↓
Relevant Table
 ↓
LLM
```

#### 2. SQL Databases

For databases, I generally use **Text-to-SQL** rather than embedding every database row.

```txt
User Question
      ↓
LLM
      ↓
SQL Generation
      ↓
SQL Database
      ↓
Query Results
      ↓
LLM
      ↓
Final Answer
```

For example:

> "What was our total revenue in Q2?"

The system generates an appropriate SQL query, executes it, and gives the result to the LLM to explain.

#### 3. Combine SQL + Vector RAG

Many production systems need both approaches:

```txt
                 User Query
                     ↓
                  Router
                 ↙      ↘
          Vector Search   SQL
              ↓            ↓
        Documents      Structured Data
                 ↘      ↙
                 Context
                    ↓
                   LLM
                    ↓
              Final Answer
```

For example:

> "Why did revenue decrease in Q2?"

- **SQL** retrieves the actual revenue numbers.
- **Vector RAG** retrieves financial reports explaining the change.
- The LLM combines both sources into the answer.

#### Important Considerations

- **Schema awareness:** Give the LLM relevant table/schema information rather than the entire database.
- **Query validation:** Validate generated SQL before execution.
- **Access control:** Enforce database permissions and row-level security.
- **Exact calculations:** Let SQL/database engines perform aggregation rather than asking the LLM to calculate from retrieved text.
- **Metadata:** Preserve table names, column names, timestamps, and source information.
- **Hybrid retrieval:** Combine structured filtering with semantic search when appropriate.

### 20.17 What are the common failure modes of RAG systems, and how do you debug them?

#### Common RAG Failure Modes

| Failure mode | Problem | How to debug/fix |
| --- | --- | --- |
| **Poor chunking** | Important context is split or chunks are too large | Test different chunk sizes, overlap, and structure-aware chunking |
| **Low retrieval recall** | Relevant documents aren't retrieved | Improve embeddings, query expansion, hybrid search, or increase Top-K |
| **Low retrieval precision** | Too many irrelevant chunks are retrieved | Use reranking, metadata filters, or reduce Top-K |
| **Wrong embeddings** | Similar concepts aren't mapped close together | Benchmark different embedding models on real queries |
| **Query mismatch** | User wording differs significantly from document wording | Use query rewriting, expansion, or hybrid search |
| **Lost in the middle** | Relevant context is buried in a long prompt | Reduce context, rerank, compress, and order evidence carefully |
| **Hallucination** | LLM generates unsupported information | Improve grounding, require evidence, and evaluate faithfulness |
| **Conflicting sources** | Retrieved documents contain contradictory information | Add source ranking, timestamps, conflict detection, and validation |
| **Stale knowledge** | Index contains outdated documents | Implement document refresh/re-indexing and metadata filtering |
| **Poor generation** | Retrieved context is good but answer is poor | Improve prompt design, model choice, and answer validation |
| **Context overflow** | Retrieved content exceeds the model's context limit | Reduce Top-K, compress chunks, or use hierarchical retrieval |

#### How I Debug a RAG System

I debug **retrieval and generation separately** rather than immediately changing the LLM.

```txt
User Query
    ↓
Query Processing
    ↓
Retrieval
    ↓
Retrieved Chunks
    ↓
Reranking
    ↓
Final Context
    ↓
LLM
    ↓
Answer
```

For a failed answer, I inspect each stage:

1. **Check the query**
   - Was the user's intent understood correctly?
   - Would query rewriting help?
2. **Inspect retrieved chunks**
   - Are the required documents present?
   - If not → **retrieval problem**.
3. **Measure retrieval**
   - Check **Recall@K, Precision@K, MRR, and nDCG**.
4. **Inspect ranking**
   - Is the relevant chunk being pushed below irrelevant results?
   - If yes → improve reranking.
5. **Inspect final context**
   - Is important evidence missing, truncated, or buried?
6. **Inspect the LLM response**
   - If the correct evidence is present but the answer is wrong → **generation/grounding problem**.
7. **Measure generation quality**
   - Evaluate **faithfulness, answer relevance, and completeness**.

#### The Most Important Debugging Principle

**First determine whether the failure is retrieval or generation.**

```txt
Correct information NOT retrieved
        → Retrieval problem

Correct information retrieved
        ↓
Wrong answer generated
        → Generation / grounding problem
```

This separation makes RAG debugging much faster because you can identify whether to change **chunking, embeddings, retrieval, reranking, context construction, or the LLM/prompt**.

### 20.18 How do you handle document updates and maintain freshness in a RAG system?

#### Handling Document Updates and Freshness in RAG

I would treat **freshness as an indexing and data-lifecycle problem**, not something the LLM itself solves.

1. **Track document versions**
   - Store metadata such as : `document_id`, `version`, `updated_at`, and source.
   - Detect when a document is created, modified, or deleted.
2. **Use incremental indexing**
   - Don't rebuild the entire vector database for every change.
   - Re-process only the changed documents or chunks.
3. **Re-embed changed content**
   - When a document changes:

   ```txt
   Updated Document
        ↓
      Chunking
        ↓
      Embedding
        ↓
   Update Vector Store
   ```
4. **Handle deletions**
   - Remove or deactivate embeddings belonging to deleted documents.
   - Otherwise, the retriever may return stale information.
5. **Use metadata filtering**
   - Store timestamps, versions, document status, and access permissions.
   - Retrieval can prioritize or filter based on freshness.
6. **Event-driven updates**
   - Use events such as file uploads, database changes, or CMS updates to trigger re-indexing automatically.

```txt
Document Store
     ↓
Change Detection
     ↓
Event / Queue
     ↓
Chunk + Embed
     ↓
Vector Database
     ↓
Fresh Knowledge
```

1. **Scheduled synchronization**
   - For sources that don't provide change events, periodically scan for updates.
2. **Freshness-aware retrieval**
   - For time-sensitive questions, prioritize newer sources.
   - For historical questions, retrieve documents appropriate to the requested time period instead.
3. **Monitor index freshness**
   - Track metrics such as:
     - Indexing delay
     - Percentage of stale documents
     - Last successful synchronization
     - Failed indexing jobs

#### Production Architecture

```txt
             Source Systems
                   ↓
            Change Detection
                   ↓
             Event / Queue
                   ↓
          Document Processing
           ↙              ↘
      Chunking          Metadata
           ↓
       Embeddings
           ↓
      Vector / Search DB
           ↓
       Hybrid Retrieval
           ↓
         Reranking
           ↓
            LLM
           ↓
      Grounded Answer
```

**Interview answer:**

**I maintain RAG freshness through versioned documents, incremental re-indexing, re-embedding changed content, deletion handling, metadata-based filtering, and event-driven or scheduled synchronization. I also monitor indexing latency and stale-document rates. For time-sensitive queries, I make retrieval freshness-aware so newer authoritative information is preferred.**

### 20.19 How do you optimize RAG for latency in production?

#### Optimizing RAG for Production Latency

I optimize each stage separately and focus on the **end-to-end latency budget**.

1. **Optimize retrieval**
   - Use efficient vector indexes such as **HNSW**.
   - Keep `Top-K` relatively small.
   - Use metadata filters to reduce the search space.
   - Use hybrid search carefully because running multiple retrieval methods adds latency.
2. **Use reranking selectively**
   - Retrieve a moderate candidate set, then rerank only when it improves quality enough to justify the extra latency.
   - Use a lightweight reranker when possible.
3. **Reduce context size**
   - Send only the most relevant chunks to the LLM.
   - Use context compression or smaller chunks where appropriate.
   - This reduces **LLM time-to-first-token and total generation time**.
4. **Use fast embedding models**
   - Choose an embedding model with a good quality/latency trade-off.
   - Batch document embeddings during ingestion.
   - Cache query embeddings when repeated queries are common.
5. **Cache aggressively**
   - Cache:
     - Query embeddings
     - Retrieval results
     - Frequently requested final responses
   - Use appropriate cache invalidation when documents change.
6. **Parallelize independent operations**
   - Run independent searches or tool calls concurrently.

```txt
Query
  ↓
 ┌──────────────┬──────────────┐
 ↓              ↓              ↓
Vector Search  Keyword Search  Other Sources
 └──────────────┴──────────────┘
                ↓
             Rerank
                ↓
               LLM
```

1. **Optimize the LLM**
   - Use a smaller model when the task doesn't require a large one.
   - Limit unnecessary output length.
   - Use streaming to improve perceived latency.
   - Consider optimized inference such as batching or quantization where appropriate.
2. **Optimize infrastructure**
   - Keep the vector database close to the application.
   - Use connection pooling.
   - Avoid unnecessary network hops.
   - Scale retrieval and inference independently.
3. **Measure before optimizing**
   - Track **p50, p95, and p99 latency** for:
     - Embedding
     - Retrieval
     - Reranking
     - LLM time-to-first-token
     - LLM generation
     - End-to-end response

#### Production Optimization Flow

```txt
User Query
    ↓
Fast Query Processing
    ↓
Cached / Efficient Retrieval
    ↓
Small Candidate Set
    ↓
Optional Reranking
    ↓
Minimal Relevant Context
    ↓
Fast LLM
    ↓
Streaming Response
```

### 20.20 What is the role of metadata filtering in RAG systems?

#### Metadata Filtering in RAG

**Metadata filtering** restricts retrieval results using information associated with each document or chunk, such as **date, source, department, document type, language, access permissions, or version**.

Instead of searching the entire knowledge base:

```txt
User Query
    ↓
Vector Search across everything
    ↓
Many results
```

you first narrow the search space:

```txt
User Query
    ↓
Metadata Filter
    ↓
Relevant subset
    ↓
Vector / Hybrid Search
    ↓
Top-K Results
```

#### Example

Suppose a company has documents from multiple departments and years.

User asks:

> "What is the 2026 marketing budget?"

Metadata could be:

```txt
department = "Marketing"
year = 2026
document_type = "Budget"
```

The retriever searches only that subset instead of the entire document collection.

#### Common Metadata Filters

- **Date/year** — retrieve only recent or historical information.
- **Department** — HR, Finance, Engineering, etc.
- **Document type** — policy, report, invoice, manual.
- **Source** — SharePoint, database, website, internal wiki.
- **Language** — English, Nepali, etc.
- **Version** — retrieve only the latest approved version.
- **Access permissions** — ensure users retrieve only documents they are authorized to access.

#### Why It Improves RAG

- **Higher retrieval precision** — removes obviously irrelevant documents.
- **Lower latency** — searches a smaller candidate space.
- **Better freshness** — can prioritize or restrict results by date/version.
- **Better security** — prevents unauthorized documents from entering the LLM context.
- **Less context noise** — gives the LLM more focused evidence.

### 20.21 Compare RAG vs fine-tuning. When would you use each?

#### RAG vs Fine-Tuning

| Aspect | RAG | Fine-Tuning |
| --- | --- | --- |
| Main purpose | Give the model external knowledge | Change the model's behavior/skills |
| Knowledge | Retrieved at runtime | Learned into model parameters |
| Updating knowledge | Easy — update the knowledge base | Requires additional training |
| Private/current data | Excellent | Less suitable |
| Hallucination control | Can improve grounding | Doesn't inherently provide current facts |
| Cost to update | Relatively low | Higher |
| Best for | Factual/domain knowledge | Behavior, style, task specialization |

#### When to Use RAG

Use **RAG** when the model needs access to information that is:

- Frequently changing
- Private or organization-specific
- Too large to put into the model's training data
- Required to be traceable to source documents

**Example:**

> "What is our latest HR leave policy?"

Retrieve the latest policy document and use it as context.

#### When to Use Fine-Tuning

Use **fine-tuning** when you want to change **how the model behaves**, such as:

- Following a specific response format
- Adapting to a specialized task
- Improving performance on a particular domain/task
- Learning a particular style or classification behavior
- Improving tool/function-calling behavior

**Example:**

Train a model to consistently convert customer requests into a specific JSON schema.

#### Best Approach: Combine Them

In many production systems, they complement each other:

```txt
Fine-Tuning
     ↓
Teach the model HOW to perform the task
     +
RAG
     ↓
Provide the model WHAT information to use
     ↓
Grounded Response
```

### 20.22 What is query transformation in RAG (HyDE, query decomposition, step-back prompting)?

#### Query Transformation in RAG

**Query transformation** means modifying the user's original query into a form that is **easier for the retriever to search effectively**.

A user query may be short, ambiguous, or conversational, while the relevant documents may use different terminology. Query transformation bridges that gap.

```txt
User Query
    ↓
Query Transformation
    ↓
Better Search Query
    ↓
Retriever
    ↓
Relevant Documents
    ↓
LLM
```

#### 1. HyDE — Hypothetical Document Embeddings

**HyDE** generates a hypothetical answer/document from the user's question first, then uses that generated text to perform retrieval.

```txt
Question
   ↓
LLM generates hypothetical answer
   ↓
Embed hypothetical answer
   ↓
Vector Search
   ↓
Relevant Documents
```

**Example:**

```txt
Question:
"What causes model overfitting?"

       ↓

Hypothetical document:
"Overfitting occurs when a model learns training-specific
patterns and noise, resulting in poor generalization..."

       ↓

Use this hypothetical document for retrieval
```

**Why:** The hypothetical answer can contain terminology that is closer to the actual documents than the original short question.

**Important:** The hypothetical answer is used for **retrieval**, not treated as factual evidence.

---

#### 2. Query Decomposition

**Query decomposition** breaks a complex question into smaller, independently searchable questions.

Example:

> "Compare the 2025 and 2026 revenue of Company A and explain the main reason for the change."

Decompose into:

```txt
1. What was Company A's 2025 revenue?
2. What was Company A's 2026 revenue?
3. What factors caused the revenue change?
```

Each sub-query is retrieved separately, and the results are combined for the final answer.

**Best for:** Multi-document and multi-hop questions.

---

#### 3. Step-Back Prompting

**Step-back prompting** first converts a specific question into a **more general conceptual question**, retrieves information about that broader concept, and then uses it to answer the original question.

Example:

```txt
Original:
"Why did our model's accuracy drop after adding feature X?"

        ↓

Step-back:
"What factors commonly cause ML model accuracy to decrease
after introducing new features?"

        ↓

Retrieve general knowledge
        ↓
Apply it to the original question
```

**Best for:** Questions where understanding the broader concept helps answer the specific question.

#### Comparison

| Technique | Main Idea | Best For |
| --- | --- | --- |
| **HyDE** | Generate hypothetical answer → retrieve using it | Query/document vocabulary mismatch |
| **Query decomposition** | Break one query into sub-queries | Complex multi-hop questions |
| **Step-back prompting** | Generalize the question before retrieval | Conceptual or reasoning-heavy questions |

### 20.23 How do you implement citation and source attribution in RAG?

#### Citation and Source Attribution in RAG

I would implement citations by **preserving source metadata throughout the entire retrieval pipeline** and linking each generated claim back to the retrieved chunk that supports it.

#### 1. Store Source Metadata

When indexing documents, store metadata with every chunk:

```txt
Chunk
 ├── document_id
 ├── document_name
 ├── source_url
 ├── page_number
 ├── section
 ├── chunk_id
 └── created_at
```

This metadata stays attached to the embedding in the vector/search database.

#### 2. Retrieve Chunks With Metadata

When the retriever returns relevant chunks, return both:

```txt
Retrieved Chunk
+
Source Metadata
```

For example:

```txt
Chunk: "The warranty period is 3 years..."

Source:
Document: Product Manual
Page: 12
Chunk ID: product_manual_12_04
```

#### 3. Pass Source IDs to the LLM

The context supplied to the LLM can include explicit source identifiers:

```txt
[Source 1]
Product Manual, Page 12
"The warranty period is 3 years..."

[Source 2]
Warranty Policy, Section 4
"Warranty claims must be submitted..."
```

The prompt instructs the model to cite the source supporting each factual claim.

#### 4. Generate Structured Output

For production systems, I prefer structured output rather than relying entirely on free-form citation generation.

```txt
{
  "answer": "...",
  "citations": [
    {
      "source_id": "product_manual_12_04",
      "page": 12
    }
  ]
}
```

The application can then convert these source IDs into clickable citations.

#### 5. Validate Citations

Don't blindly trust citations generated by the LLM.

Check that:

- The cited source actually exists.
- The source was actually retrieved.
- The cited chunk supports the claim.
- Citations aren't fabricated.
- The source has appropriate permissions.

A stronger system can perform a **claim-to-evidence verification** step after generation.

#### Production Flow

```txt
Documents
   ↓
Chunk + Metadata
   ↓
Embedding / Index
   ↓
Retrieve
   ↓
Chunks + Source Metadata
   ↓
Rerank
   ↓
Context + Source IDs
   ↓
LLM
   ↓
Answer + Citation IDs
   ↓
Citation Validation
   ↓
Final Answer with Sources
```

#### Important Design Principle

**Citations should be tied to retrieved evidence, not generated independently by the LLM.**

The LLM should select or reference known source IDs, while the application resolves those IDs to the actual **document, page, section, URL, or database record**.

### 20.24 How do you scale a RAG system to millions of documents?

To scale a **RAG system to millions of documents**, I would focus on **distributed ingestion, scalable vector search, efficient retrieval, and incremental updates**.

#### 1. Distributed document ingestion

- Process documents asynchronously using **Kafka, SQS, or similar queues**.
- Use distributed workers for parsing, chunking, and embedding.
- Process documents in batches rather than synchronously.
- Make the pipeline idempotent so failed jobs can be safely retried.

#### 2. Efficient chunking and indexing

- Avoid creating unnecessarily small chunks, which can cause billions of vectors.
- Store metadata such as `document_id`, `source`, `timestamp`, `tenant`, and `access_control`.
- Use **incremental indexing** so only new or modified documents are processed.

#### 3. Scalable vector database

Use a vector database/search engine that supports:

- Horizontal scaling and sharding
- Distributed indexes
- Approximate nearest-neighbor search such as **HNSW**
- Metadata filtering
- Replication and fault tolerance

Examples include OpenSearch, Elasticsearch, Milvus, Weaviate, and Pinecone.

#### 4. Hybrid retrieval

For millions of documents, combine:

- **Vector search** for semantic similarity
- **BM25/keyword search** for exact terms

Then use score fusion or reranking to improve precision.

#### 5. Reduce the search space

Don't search every vector blindly.

Use:

- Metadata filters
- Tenant/document permissions
- Document type
- Date/version filters
- Namespace partitioning
- Query routing

For example:

```txt
Millions of documents
        ↓
Metadata / permission filtering
        ↓
Candidate retrieval
        ↓
Top 50–100 documents
        ↓
Reranking
        ↓
Top 5–10 chunks
        ↓
LLM
```

#### 6. Optimize latency

- Cache frequent queries and embeddings.
- Parallelize keyword and vector retrieval.
- Limit `Top-K`.
- Rerank only a small candidate set.
- Use connection pooling.
- Monitor p50/p95/p99 latency.
- Keep retrieval services close to the vector database.

#### 7. Separate storage from retrieval

Keep original documents in cheap object storage such as S3 and store only the information needed for retrieval in the search system:

```txt
Object Storage
   ↓
Documents
   ↓
Distributed Processing
   ↓
Chunks + Embeddings + Metadata
   ↓
Distributed Vector / Hybrid Index
   ↓
Retriever
   ↓
Reranker
   ↓
LLM
```

#### 8. Handle updates efficiently

Don't rebuild the entire index whenever a document changes.

Use:

```txt
New document → index
Updated document → re-chunk + re-embed affected chunks
Deleted document → remove chunks
Unchanged document → skip
```

### 20.25 What is parent-child chunking, and how does it improve retrieval?

#### Parent-Child Chunking

**Parent-child chunking** stores documents at two levels:

- **Parent chunk:** A larger section containing broader context.
- **Child chunk:** Smaller, more focused pieces used for precise retrieval.
- The **child chunks are indexed for search**, while the corresponding **parent chunk is provided to the LLM** as context.

#### Example

```txt
Document
   │
   ├── Parent: "Company Leave Policy"
   │       │
   │       ├── Child: Annual leave rules
   │       ├── Child: Sick leave rules
   │       └── Child: Parental leave rules
   │
   └── Parent: "Employee Benefits"
           │
           ├── Child: Health insurance
           └── Child: Retirement benefits
```

If the user asks:

> "How many sick days can an employee take?"

The **sick-leave child chunk** may be retrieved because it closely matches the query, but the system can pass its **parent chunk** to the LLM to provide additional surrounding context.

#### Why it improves retrieval

- **Higher precision:** Small child chunks contain focused information, making semantic matching more accurate.
- **Better context:** The larger parent chunk gives the LLM surrounding information needed to understand the retrieved content.
- **Reduces irrelevant retrieval:** You don't need to search using large chunks containing many unrelated topics.
- **Balances precision and context:** Child chunks optimize **retrieval**, while parent chunks optimize **generation**.

### 20.26 Your RAG system is hallucinating despite having the right context. How do you fix it?

If the **right context is retrieved but the RAG system still hallucinates**, the problem is usually in **generation, grounding, or prompt design**, not retrieval.

#### How I would fix it

1. **Strengthen the grounding prompt**
   - Explicitly instruct the LLM to answer only from retrieved context.
   - Tell it to say *“I don't have enough information”* when the context doesn't support an answer.
2. **Require evidence-based answers**
   - Ask the model to associate each claim with a retrieved source/chunk.
   - Use structured output such as:

   ```txt
   answer
   supporting_sources
   ```
3. **Reduce unnecessary context**
   - Even correct context can contain distracting information.
   - Rerank and pass only the most relevant chunks.
4. **Improve context organization**
   - Clearly separate retrieved documents from instructions.
   - Include source IDs and document boundaries.
   - Put the most relevant evidence in prominent positions.
5. **Add answer validation**
   - Run a second LLM or verification step to check:
     - Is every claim supported?
     - Does the answer contradict the context?
     - Are citations actually supporting the claims?
6. **Use confidence/abstention**
   - If evidence is insufficient or conflicting, don't force an answer.
   - Return an uncertainty response instead.
7. **Test the LLM separately**
   - Feed the same retrieved context directly to the model without the retrieval pipeline.
   - If it still hallucinates, the issue is clearly in the **generation/grounding layer**.

#### Production flow

```txt
Query
  ↓
Retrieve correct context
  ↓
Rerank / compress
  ↓
Grounded prompt
  ↓
LLM
  ↓
Claim + Evidence validation
  ↓
Final answer
```

### 20.27 Your RAG chunk overlap causes redundant results. How do you reduce redundancy?

- **Reduce chunk overlap** — use a smaller overlap, since excessive overlap creates near-duplicate chunks.
- **Use similarity thresholds** — discard retrieved chunks that are too similar to a higher-ranked chunk.
- **Apply MMR (Maximal Marginal Relevance)** — balances relevance with diversity, so retrieved chunks cover different information.
- **Deduplicate after retrieval** — compare chunk IDs, text similarity, or document sections and remove duplicates.
- **Use parent-child retrieval** — retrieve small child chunks for precision, then return the relevant parent section instead of many overlapping chunks.
- **Tune `top-k`** — retrieving fewer chunks reduces the chance of redundant context.
- **Prefer semantic boundaries** — split by sections, paragraphs, or headings rather than arbitrary fixed-size windows.

**Best practical approach:** Use moderate/low overlap during chunking + **MMR or similarity-based deduplication during retrieval**.

### 20.28 Your RAG retrieval is too slow with a large knowledge base. How do you speed it up?

- **Use approximate nearest neighbor (ANN)** — use indexes such as HNSW, IVF, or PQ instead of exact vector search.
- **Reduce `top-k`** — retrieve only the number of chunks actually needed.
- **Add metadata filtering** — filter by document type, tenant, date, department, etc. before vector search.
- **Use hybrid search efficiently** — combine vector search with keyword search to narrow candidates.
- **Use a two-stage retrieval pipeline** — fast vector search first, then rerank only the top candidates.
- **Use a smaller embedding model** — reduces embedding and search latency while maintaining acceptable quality.
- **Cache frequent queries** — cache embeddings and retrieval results for repeated queries.
- **Partition or shard the index** — distribute very large knowledge bases across indexes or machines.
- **Optimize the vector database** — tune HNSW parameters, memory, indexing, batching, and hardware.
- **Use hierarchical retrieval** — first identify relevant documents/sections, then search within them.

**Best practical approach:** Metadata filtering → ANN vector search → retrieve a small `top-k` → rerank only the best candidates.

### 20.29 Your RAG system returns duplicate results. How do you deduplicate?

- **Remove exact duplicates** — compare chunk IDs, document IDs, or normalized text and keep only one copy.
- **Use similarity-based deduplication** — calculate similarity between retrieved chunks and remove chunks that are nearly identical.
- **Use MMR (Maximal Marginal Relevance)** — select results that are both relevant and diverse.
- **Group overlapping chunks** — if multiple chunks come from the same section, merge them or keep the most relevant one.
- **Deduplicate at ingestion time** — hash documents/chunks to prevent identical content from entering the vector database.
- **Use metadata** — track `document_id`, `section_id`, and `chunk_id` to identify duplicates reliably.
- **Limit results per document** — prevent a single document from dominating the retrieved context.

**Best practical approach:** Exact deduplication → similarity-based filtering/MMR → limit the number of chunks per document.

### 20.30 Your RAG system needs per-user access control on internal documents. How do you implement it?

- **Store ACLs/permissions as metadata** with every document/chunk, such as : `tenant_id`, `user_ids`, `roles`, `groups`, or `department`.
- **Authenticate the user** and obtain their identity and roles from the authorization system.
- **Apply authorization filtering during retrieval**, before returning chunks to the LLM. For example, filter by `tenant_id` and allowed `group_id`/`document_id`.
- **Never rely on the LLM to enforce access control**. Unauthorized documents should never enter the prompt/context.
- **Enforce access at multiple layers**:
  - Document ingestion/indexing
  - Vector/keyword retrieval
  - Document/source API
  - Final response validation if necessary
- **Handle permission changes** by updating or removing affected document/chunk permissions from the index.
- **Use tenant isolation** for multi-tenant systems to prevent cross-customer data leakage.
- **Audit access** by logging user ID, retrieved document IDs, and authorization decisions.
- **Test for authorization bypasses**, including prompt injection attempts such as asking the system to retrieve another user's documents.

**Key principle: Authorization must happen before retrieval, not after generation.**

### 20.31 Your RAG system fails on domain-specific jargon. How do you fix it?

- **Improve chunking:** Use domain-aware chunking so technical terms and their surrounding context stay together.
- **Expand queries:** Use query expansion or an LLM to add synonyms, abbreviations, acronyms, and domain-specific terminology.
- **Build a domain glossary:** Map jargon, abbreviations, product names, and aliases to their canonical terms.
- **Use domain-specific embeddings:** Choose or fine-tune an embedding model that performs well on the target domain.
- **Hybrid search:** Combine **vector search + BM25/keyword search** so exact technical terms are not lost by semantic retrieval.
- **Fine-tune retrieval:** Create domain-specific query–document pairs and fine-tune the embedding/reranking model.
- **Add a reranker:** Retrieve a larger candidate set, then use a cross-encoder/reranker trained or tuned for the domain.
- **Use metadata:** Filter by product, department, document type, version, or other domain-specific attributes.
- **Evaluate with domain queries:** Build a test set containing real jargon-heavy questions and measure retrieval recall, precision, and answer accuracy.

### 20.32 Your text-only RAG system now needs to handle images and tables. How do you extend it?

- **Use multimodal ingestion:** Extract text, images, and tables separately from PDFs, DOCX, PPTX, etc.
- **Handle tables specially:** Convert tables into structured representations such as Markdown/HTML/JSON, preserving headers, rows, and relationships. Create embeddings from the table content.
- **Handle images:** Use a vision-language model to generate detailed captions/descriptions containing objects, diagrams, labels, and relevant relationships. Embed those descriptions.
- **Keep relationships:** Store metadata such as `document_id`, `page_number`, `section`, `table_id`, and `image_id` so retrieved content can be traced back to the original source.
- **Use multimodal embeddings:** For systems requiring direct image understanding, use embeddings/models that support both text and visual content.
- **Retrieve across modalities:** A user query can retrieve text chunks, table representations, and image descriptions together.
- **Use the right model for generation:** Pass textual context to an LLM and relevant images directly to a vision-capable LLM when visual understanding is required.
- **Preserve citations:** Return the original document, page, table, or image reference so answers remain traceable.

**Typical pipeline:**

`Documents → Parse → Text/Table/Image extraction → Multimodal representation → Embeddings → Vector + Keyword retrieval → Reranking → LLM/VLM → Cited answer`

The key is **not treating images and tables as ordinary text**; preserve their structure and visual information during both indexing and retrieval.

### 20.33 Your RAG knowledge base gets updated frequently and needs versioning. How do you manage it?

- **Assign a version to every document** using fields such as : `document_id`, `version`, `created_at`, and `updated_at`.
- **Use immutable document versions:** Never overwrite an indexed version. Store each update as a new version.
- **Track chunk versions:** Each chunk should reference its `document_id` and `version`, so chunks from different versions are never mixed accidentally.
- **Incrementally re-index:** When a document changes, detect the changed sections/chunks and re-embed only those instead of rebuilding the entire knowledge base.
- **Use metadata filtering:** At query time, retrieve only the **currently active/published version** unless the user explicitly requests historical information.
- **Maintain an ingestion pipeline:**`Document update → Change detection → New version → Chunk → Embed → Index → Validate → Activate`
- **Support rollback:** Keep previous versions so you can quickly revert if a new version contains bad or incorrect information.
- **Handle deletions:** Mark old versions as inactive or deleted and remove/exclude their chunks from retrieval.
- **Use blue/green indexing for major updates:** Build a new index/version separately, validate retrieval quality, then switch traffic to it atomically.
- **Audit everything:** Record who changed the document, when, which version was indexed, and which version was used to answer a query.

### 20.34 Your RAG system fails on multi-hop questions that require combining multiple facts. How do you fix it?

- **Use query decomposition:** Break the multi-hop question into smaller sub-questions that can be retrieved independently.
- **Retrieve iteratively:** Retrieve evidence for the first step, use it to formulate the next query, and continue until all required facts are found.
- **Use multi-query retrieval:** Generate multiple related queries to improve recall across different documents.
- **Use a reranker:** Rerank retrieved chunks to select the most relevant evidence for each hop.
- **Build relationships:** Use document metadata, entity linking, or a **knowledge graph** to connect entities and facts across documents.
- **Keep intermediate context:** Store the evidence and intermediate conclusions from each hop rather than relying on a single retrieval call.
- **Use an agentic RAG workflow:** Let an orchestrator decide whether another retrieval step is needed before generating the final answer.
- **Validate the reasoning chain:** Check that every intermediate claim is supported by retrieved evidence and that no required hop is missing.
- **Evaluate specifically for multi-hop queries:** Create benchmark questions requiring 2–4 hops and measure retrieval recall and final-answer accuracy.

**Typical approach:**

`Question → Decompose → Retrieve hop 1 → Extract fact → Retrieve hop 2 → Combine evidence → Validate → Generate answer`

### 20.35 Your enterprise RAG system returns contradictory answers from different source documents. How do you resolve conflicts?

- **Establish source priority:** Define a hierarchy, e.g., approved policy > official documentation > internal wiki > older documents.
- **Use document freshness:** Prefer the latest valid version when documents contain time-sensitive information.
- **Add metadata:** Store `source`, `version`, `effective_date`, `owner`, `authority`, and `last_updated`.
- **Detect contradictions:** During retrieval, compare claims from multiple sources instead of blindly combining them.
- **Use a reranker:** Rank evidence based on relevance **and authority/freshness**, not relevance alone.
- **Resolve at retrieval time:** Filter out outdated or superseded documents before they reach the LLM.
- **Require evidence:** Prompt the LLM to cite the supporting source and avoid choosing between conflicting claims without evidence.
- **Expose uncertainty:** If two authoritative sources genuinely conflict, the system should say there is a conflict rather than inventing a resolution.
- **Create a source-governance process:** Assign document owners who can mark sources as deprecated, superseded, or authoritative.
- **Audit conflicts:** Log conflicting retrievals and use them to improve the knowledge base and retrieval rules.

**Typical approach:**

`Retrieve → Detect conflicting claims → Rank by authority + version + date → Select valid evidence → Generate with citations`

### 20.36 Your RAG system returns outdated answers from an evolving knowledge base. How do you keep it current?

- **Track document versions:** Store `document_id`, `version`, `updated_at`, and `effective_date`.
- **Use incremental indexing:** Detect changed documents and re-chunk/re-embed only the affected content.
- **Remove stale chunks:** Mark old versions as inactive or delete them from the retrieval index.
- **Filter by freshness:** At query time, prefer the latest **active/published** version using metadata filters.
- **Automate ingestion:** Trigger the RAG pipeline whenever source documents are created or updated.
- **Use source-of-truth systems:** Connect directly to authoritative databases, APIs, or document repositories instead of relying on manually copied content.
- **Cache carefully:** Use short TTLs for frequently changing information and invalidate caches when documents change.
- **Add effective dates:** For policies and procedures, retrieve the document that was valid for the relevant date.
- **Monitor freshness:** Track metrics such as indexing delay, stale-document rate, and last successful ingestion.
- **Support rollback:** Keep previous versions so bad updates can be reverted safely.

**Typical pipeline:**

`Source update → Change detection → Versioning → Re-chunk → Re-embed → Update index → Mark old version inactive → Validate`

### 20.37 Your RAG system struggles with PDF documents containing tables and layouts. How do you fix PDF parsing?

- **Use layout-aware PDF parsers:** Extract text while preserving page structure, columns, headings, and reading order.
- **Use dedicated table extraction:** Detect tables separately and convert them to structured formats such as Markdown, HTML, or JSON rather than treating them as plain text.
- **Handle scanned PDFs with OCR:** Apply OCR when the PDF contains images instead of selectable text.
- **Preserve layout metadata:** Store `page_number`, `section`, `table_id`, bounding boxes, and document coordinates for citations and retrieval.
- **Detect document types:** Use different parsing strategies for digital PDFs, scanned PDFs, reports, invoices, and complex multi-column documents.
- **Handle headers/footers:** Remove repeated headers, footers, and page numbers when they create retrieval noise.
- **Keep tables intact:** Avoid splitting a table across chunks without preserving its headers and row relationships.
- **Use multimodal models when necessary:** For complex diagrams, charts, or visually structured pages, render the page as an image and use a vision model to interpret it.
- **Validate extraction:** Compare extracted content against the original PDF and monitor table/text extraction accuracy.

**Typical pipeline:**

`PDF → Detect type → Layout analysis → Text/Table/OCR extraction → Preserve structure + metadata → Chunk → Embed → Index`

### 20.38 What is Retrieval-Augmented Generation (RAG)?

A technique where the model retrieves relevant documents/data from an external knowledge source at query time and uses that as context to generate more accurate, grounded answers, instead of relying only on what it memorized during training.

### 20.39 What is a RAG pipeline, at a high level?

It retrieves relevant documents for a query, injects them into the prompt as context, and grounds the model's answer in those real sources rather than relying purely on parametric knowledge.

### 20.40 Why does chunking strategy matter so much in RAG?

How documents are split — fixed-size, semantic, or header-based — directly determines what gets retrieved; poor chunking breaks context and degrades retrieval quality regardless of how good the embedding model is.

### 20.41 What is hybrid search and why combine methods?

It combines semantic (vector) search with keyword search (BM25) to get the best coverage — semantic search catches conceptual matches, keyword search catches exact terms vectors can miss.

### 20.42 What role does reranking play in a retrieval pipeline?

A second-pass model re-scores the initial candidate documents for relevance, refining the ranking beyond what the first-pass retriever produced.

### 20.43 How should retrieval quality be judged?

By whether the right documents were actually found, not by how confident the final model output sounds — confidence is not evidence of correctness.

### 20.44 What are grounded citations?

Forcing the model to produce traceable citations back to source material, so claims can be verified.

### 20.45 What is LlamaIndex primarily focused on?

Data ingestion and constructing RAG pipelines.

### 20.46 How do knowledge graphs/ontologies differ from flat text chunks for retrieval?

They capture relationships between entities, enabling relationship-aware retrieval instead of treating documents as isolated flat text.

### 20.47 Walk through the architecture of a production-grade RAG pipeline using hybrid retrieval and reranking.

1. **Ingestion**: chunk documents (semantic or fixed-size with overlap), generate dense embeddings (e.g., via an embedding model), and index into a vector DB; also build a keyword/sparse index (BM25) over the same chunks.
2. **Query time — hybrid retrieval**: run the query against both the dense (vector similarity) and sparse (keyword/BM25) indexes in parallel, then fuse results (e.g., Reciprocal Rank Fusion). This captures both semantic similarity and exact-term matches (important for names, IDs, jargon that embeddings can miss).
3. **Reranking**: pass the top-K fused candidates through a cross-encoder reranker (which jointly scores query+passage pairs, unlike bi-encoder embeddings) to reorder by true relevance — this is more accurate but more expensive, hence applied only to the shortlist.
4. **Context assembly**: select the top-N reranked chunks within a token budget, optionally with source metadata for citation.
5. **Generation**: pass context + query to the LLM with a prompt that instructs it to ground answers strictly in the provided context and cite sources.
6. **Grounding/hallucination checks**: optionally a post-hoc verification step (e.g., a smaller model or rule-based check) confirming claims are supported by retrieved context.

### 20.48 Why combine dense vector search with keyword (sparse) search instead of using just one?

Dense embeddings capture semantic/conceptual similarity but can miss exact matches on rare terms, product codes, acronyms, or numbers, since these get "smoothed over" in embedding space. Keyword search (BM25) is precise on exact terms but misses paraphrases and semantic relationships. Hybrid search gets the best of both — semantic recall plus lexical precision — and is especially important in technical or domain-specific corpora (legal, medical, enterprise data) where terminology precision matters.

### 20.49 Why do you need a reranker if you already have a good retriever?

Bi-encoder embedding search encodes the query and each document independently, so it can't model fine-grained interactions between them — it's fast but approximate. A cross-encoder reranker feeds the query and candidate together into one model, letting it attend to their interaction directly, which produces a much more accurate relevance score. It's too slow to run over the whole corpus, so the standard pattern is: cheap retrieval (bi-encoder + BM25) to get a shortlist of \~50-100 candidates, then expensive-but-accurate reranking to pick the true top-N for the LLM's context window.

### 20.50 How do you evaluate and reduce hallucination in a RAG system?

Evaluation: build a labeled eval set (question, gold answer, gold source chunks) and measure retrieval metrics (recall@k, MRR) separately from generation metrics (faithfulness/groundedness — does the answer only use retrieved facts — and answer relevance). Use LLM-as-judge or frameworks like RAGAS to automate this at scale. Reduction techniques: hybrid retrieval + reranking (better context = fewer hallucinations), explicit prompting to say "I don't know" when context is insufficient, citation-forcing (require the model to point to source chunks), and post-generation verification against retrieved context.

### 20.51 Explain a complete RAG architecture.

Typical flow:

```javascript
Documents
   ↓
Parsing
   ↓
Chunking
   ↓
Embedding
   ↓
Vector Database
   ↓
Query
   ↓
Retrieval
   ↓
Reranking
   ↓
Context Construction
   ↓
LLM
   ↓
Answer
```

#### RAG End-to-End

**Retrieval-Augmented Generation (RAG)** combines information retrieval with an LLM so the model can answer using external knowledge instead of relying only on its pretrained knowledge.

##### 1. Indexing / Ingestion

```txt
Documents
   ↓
Parsing / OCR
   ↓
Cleaning & Metadata
   ↓
Chunking
   ↓
Embedding Model
   ↓
Vector DB / Search Index
```

- Load documents from PDFs, web pages, databases, etc.
- Parse and clean the content.
- Split documents into meaningful chunks.
- Generate embeddings for each chunk.
- Store chunks, embeddings, and metadata in a vector database/search engine.

##### 2. Query Processing

```txt
User Query
    ↓
Query Rewriting / Expansion
    ↓
Retrieval
    ↓
Reranking
    ↓
Top-N Context
```

- Convert the query into an embedding.
- Retrieve relevant chunks using **dense, sparse, or hybrid retrieval**.
- Optionally use a **cross-encoder reranker** to improve precision.
- Select the best chunks for the LLM context.

##### 3. Generation

```txt
User Query + Retrieved Context
              ↓
             LLM
              ↓
        Answer + Citations
```

The LLM generates an answer based primarily on the retrieved evidence, ideally refusing to answer when sufficient evidence is unavailable.

---

#### How to Evaluate RAG

Evaluate **retrieval and generation separately**, then evaluate the complete system.

| Layer | Metrics | What it tells you |
| --- | --- | --- |
| **Retrieval** | Recall@K, Precision@K, MRR, nDCG | Did we retrieve the right documents? |
| **Context** | Context relevance, context precision, context recall | Is the retrieved context useful and sufficient? |
| **Generation** | Faithfulness/groundedness, answer correctness, relevance | Is the answer supported and correct? |
| **End-to-end** | Task success, human/LLM judge, user feedback | Does the complete RAG system solve the user's problem? |
| **Production** | p50/p95 latency, throughput, token usage, cost, errors | Is it efficient and reliable? |

##### Example debugging

If the answer is wrong:

```txt
Wrong Answer
    ↓
Was the correct document retrieved?
    ├── No → Retrieval problem
    │        → embeddings / chunking / query / search
    │
    └── Yes
         ↓
    Was the evidence sufficient and relevant?
         ├── No → Context construction problem
         │
         └── Yes
              ↓
         Did the LLM use the evidence correctly?
              └── No → Generation / prompting problem
```

##### Production evaluation loop

**Golden dataset → Retrieval evaluation → Generation evaluation → Human/LLM evaluation → Production monitoring → Regression tests**

### 20.52 Why can naive vector search produce poor results?

Naive vector search = embed the query, take top-k by cosine similarity, stuff into the prompt. It often underperforms because:

- **Similarity ≠ relevance:** "how to cancel a subscription" and "how to start a subscription" embed close together but answer different questions.
- **Exact terms are smoothed out:** product codes, error IDs, names, numbers and acronyms are poorly represented, so keyword matches are missed.
- **Query–document asymmetry:** short questions and long passages occupy different regions of the space.
- **Bad chunking:** chunks that split a fact or mix topics give diluted embeddings.
- **Domain mismatch:** general embedding models miss legal, medical or internal jargon.
- **No metadata awareness:** ignores recency, permissions, document type and version, so stale or unauthorised content surfaces.
- **ANN approximation and fixed k:** approximate indexes miss neighbours; a fixed k returns junk when few chunks are relevant and too little when many are.
- **Redundancy:** near-duplicates fill the context with the same information.

**Fixes:** hybrid search (BM25 + dense), cross-encoder reranking, query rewriting/HyDE/multi-query, structure-aware chunking with metadata, domain-adapted embeddings, metadata filters, MMR for diversity, and measuring recall@k on a labelled set.

### 20.53 Compare dense retrieval vs sparse retrieval.

#### Dense vs. Sparse vs. Hybrid Retrieval

| Feature | **Dense Retrieval** | **Sparse Retrieval** | **Hybrid Retrieval** |
| --- | --- | --- | --- |
| Representation | Dense embeddings | Sparse term-weight vectors | Dense + sparse |
| Matching | Semantic similarity | Keyword/token matching | Semantic + lexical |
| Good at | Meaning and paraphrases | Exact terms, IDs, names | Both |
| Common methods | Bi-encoder + vector DB | BM25, TF-IDF | BM25 + vector search |
| Weakness | Can miss exact terms | Poor semantic understanding | More complexity |

#### Core Mechanics

- **Dense:** Converts query and documents into embeddings and retrieves based on vector similarity.*Example:* `"How can I reset my password?"` can retrieve `"Steps to recover account credentials"`.
- **Sparse:** Matches important words/tokens between query and documents.*Example:* Searching `"ERR_CONNECTION_REFUSED"` benefits from exact lexical matching.
- **Hybrid:** Runs both retrieval methods and combines their rankings/scores, often using **Reciprocal Rank Fusion (RRF)** or a weighted score.

#### Production RAG

```txt
                ┌─ Dense Retrieval ─┐
User Query ─────┤                   ├─→ Fusion/Reranking → LLM
                └─ Sparse Retrieval ┘
```

**Key idea:**

**Dense = semantic meaning**

**Sparse = exact terminology**

**Hybrid = robust retrieval using both**

### 20.54 What is reranking, and why does it improve RAG?

**Reranking** is a second-stage scoring step: a cheap first-stage retriever (BM25, dense or hybrid) fetches a broad candidate set (e.g. top 50–200), then a more accurate, more expensive model re-scores each (query, passage) pair and only the top-N go to the LLM.

**Why it improves RAG:**
- Bi-encoders encode query and document independently, so they can't model fine-grained interactions. A **cross-encoder** (or LLM reranker) reads both together with full attention and judges whether the passage actually answers the question.
- It raises **precision at the top**, which matters because context is limited and noisy context causes hallucination and lost-in-the-middle failures. Fewer, better chunks also cut token cost.
- It decouples goals: set first-stage k high for **recall**, rely on the reranker for **precision**.

**Options:** cross-encoders (bge-reranker, ms-marco MiniLM), hosted rerankers (Cohere Rerank), late interaction (ColBERT), LLM listwise reranking.

**Tradeoffs:** extra latency (tens to hundreds of ms per ~100 pairs), cost and infra; mitigate with fewer candidates, batching, distilled rerankers and caching.

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base")
scores = reranker.predict([(query, c.text) for c in candidates])
top = [c for _, c in sorted(zip(scores, candidates), key=lambda x: -x[0])][:5]
```

Validate the gain with nDCG@k/MRR and end-to-end answer quality.

### 20.55 Compare bi-encoder vs cross-encoder retrieval.

#### Bi-Encoder vs. Cross-Encoder Reranking

| Feature | Bi-Encoder | Cross-Encoder |
| --- | --- | --- |
| **Input** | Query and document encoded independently | Query and document processed together |
| **Output** | Dense embedding vector | Relevance score |
| **Speed** | Fast | Slower |
| **Precomputation** | Document embeddings can be precomputed | Cannot precompute document-query scores |
| **Scalability** | Excellent for large-scale retrieval | Limited for large-scale retrieval |
| **Accuracy** | Good semantic matching | Usually better fine-grained relevance |
| **Typical role** | Initial retrieval | Reranking |

#### Core Mechanics

- **Bi-Encoder:** Encodes the query and documents separately, then compares their embeddings using **cosine similarity, dot product, or another distance metric**. It is optimized for **high recall and scalable retrieval**.
- **Cross-Encoder:** Processes the **query-document pair jointly**, allowing attention between query and document tokens. It produces a relevance score and is optimized for **ranking precision**.

#### Production RAG Pipeline

```txt
User Query
    ↓
Bi-Encoder
    ↓
Top-K Candidates
    ↓
Cross-Encoder Reranker
    ↓
Top-N Relevant Chunks
    ↓
LLM
```

**Key idea:**

**Bi-encoder = retrieve efficiently**

**Cross-encoder = rank accurately**

### 20.56 What is contextual compression?

**Contextual compression** shrinks retrieved documents to only what is relevant to the current query before they reach the LLM.

**Techniques:**
- **Filtering:** drop chunks below a relevance threshold (reranker score, embedding similarity, LLM yes/no).
- **Extractive compression:** keep only query-relevant sentences/spans.
- **Abstractive compression:** summarise each chunk with respect to the query.
- **Token-level compression:** e.g. LLMLingua removes low-information tokens.
- **Deduplication:** merge near-duplicate content.

**Benefits:** more distinct evidence fits in the window; less noise (fewer hallucinations and lost-in-the-middle failures); lower token cost and latency of the main call.

**Tradeoffs:**
- Extra model calls add latency/cost, sometimes more than they save.
- Risk of dropping a crucial qualifier or exception, hurting faithfulness.
- Abstractive summaries can hallucinate and lose exact wording needed for citations.

**Production tip:** start with cheap filtering and reranking; add LLM compression only when context budgets are tight, and judge it by faithfulness and correctness, not just token savings. LangChain implements this as `ContextualCompressionRetriever` wrapping a base retriever with a compressor.

### 20.57 What is the lost-in-the-middle problem?

#### Lost-in-the-Middle

LLMs may use information at the **beginning and end of a long context better than information buried in the middle**.

**Strategies:**

- Put the most relevant context near the **beginning/end**.
- Retrieve fewer, higher-quality chunks.
- **Rerank** retrieved documents before passing them to the LLM.
- Summarize or compress less-important context.
- Split complex queries into smaller retrieval tasks.

#### Long-Context Strategies

For large documents or long conversations:

- **Hierarchical retrieval:** retrieve documents → sections → relevant chunks.
- **Context compression:** remove redundant or irrelevant information.
- **Query-focused retrieval:** retrieve only information relevant to the current question.
- **Chunk strategically:** preserve semantic boundaries rather than using arbitrary sizes.
- **Summarization:** maintain compact summaries of older conversation/history.
- **Long-context model:** use when the task genuinely requires broad context.

### 20.58 How would you select an optimal chunk size?

There's no universal best size; chunk size is a hyperparameter tuned against retrieval and answer quality.

**Start from data and queries:**
- **Structure:** split on natural boundaries (headings, sections, paragraphs, table rows, functions) rather than fixed character counts.
- **Query type:** factoid lookups favour smaller chunks (~200–400 tokens); explanatory/synthesis questions favour larger (~500–1,000).
- **Embedding limits:** stay well within the model's max input; long inputs dilute embeddings.
- **Context budget:** chunk size × top-k must fit with the prompt and answer.

**Experiment:**
1. Build a labelled set (questions, gold passages, gold answers).
2. Try a grid, e.g. 256/512/1024 tokens with 10–20% overlap, plus a structure-aware splitter.
3. Measure recall@k/MRR, end-to-end faithfulness and correctness, cost and latency.
4. Choose the smallest size that keeps recall high and answers complete.

**Useful patterns:**
- **Small-to-big / parent-document retrieval:** match on small chunks, return the parent section to the LLM.
- **Contextual chunk headers:** prepend document title/section path to each chunk.
- Different strategies per document type (contracts vs FAQs vs code).

Re-tune when the corpus, embedding model or query mix changes.

### 20.59 What happens if chunks are too small?

**Chunks that are too small** (a sentence or < ~100 tokens):

- **Lost context:** "It must be renewed every 12 months" no longer says what "it" is, so both the embedding and the LLM lack meaning.
- **Fragmented answers:** needed facts are spread over many chunks; top-k may get only some, giving incomplete or wrong answers.
- **Noisier retrieval:** short texts produce less reliable embeddings that match on surface words.
- **Overhead:** many more vectors to store and search; the context fills with tiny, often redundant pieces.
- **Harder reasoning and citation:** the model must stitch fragments together, raising hallucination risk.

**Upside:** very precise matching for factoid lookups with little irrelevant text.

**Mitigations:** overlap, prepend document/section titles, parent-document (small-to-big) retrieval that sends the surrounding section, or larger top-k plus reranking.

### 20.60 What happens if chunks are too large?

**Chunks that are too large** (whole pages or thousands of tokens):

- **Diluted embeddings:** one vector represents many topics, so similarity to a specific query weakens and the right chunk ranks lower.
- **Noisy context:** lots of irrelevant text distracts the LLM, increasing hallucinations and lost-in-the-middle failures.
- **Less coverage:** only a few chunks fit the context budget, so fewer distinct sources are seen.
- **Higher cost and latency** from more input tokens.
- **Silent truncation** beyond the embedding model's max input length.
- **Coarse citations:** hard to point to the exact supporting passage.

**Upside:** more context per chunk, good for explanatory questions within one section.

**Mitigations:** structure-aware splitting, hierarchical retrieval (retrieve section, then sub-chunk), reranking and contextual compression, and tuning size on an eval set.

### 20.61 Explain retrieval precision, retrieval recall, faithfulness, and answer relevance.

These four metrics separate RAG quality into **retrieval** and **generation**:

**Retrieval precision** — of the retrieved chunks, how many are relevant?
$\text{Precision@k} = \frac{\#\text{relevant in top-}k}{k}$
Low precision = noisy context, more tokens, more distraction.

**Retrieval recall** — of all relevant chunks, how many were retrieved?
$\text{Recall@k} = \frac{\#\text{relevant in top-}k}{\#\text{relevant total}}$
Low recall = the answer isn't in the context, so the LLM can't answer correctly (or hallucinates). Usually the first thing to fix.

**Faithfulness (groundedness)** — are the answer's claims supported by the retrieved context? Split the answer into claims and check each against the context (NLI model or LLM judge): supported / total. Low faithfulness = hallucination, even if the answer happens to be true.

**Answer relevance** — does the answer address the user's question (not off-topic, evasive or padded)? Measured with an LLM judge, or (RAGAS) by generating questions from the answer and comparing them to the original via embeddings.

**Diagnosis:**
- Low recall → chunking, embeddings, hybrid search, query rewriting.
- High recall, low precision → reranking/filtering.
- Good retrieval, low faithfulness → prompt, grounding or model issue.
- Faithful but irrelevant → prompt or query understanding.

Also track **answer correctness** vs a gold answer. Tools: RAGAS, TruLens, DeepEval — with judges calibrated against human labels.

### 20.62 What is GraphRAG, and when is it better than traditional RAG?

|  | **Traditional RAG** | **GraphRAG** |
| --- | --- | --- |
| Knowledge representation | Document chunks + embeddings | **Entities + relationships + graph** |
| Retrieval | Similarity/vector search | Graph traversal + semantic search |
| Best for | Specific factual lookups | Multi-hop and relationship-based questions |
| Context | Individual relevant chunks | Connected entities, relationships, and summaries |
| Complexity | Lower | Higher |
| Storage | Vector DB | Graph DB + often vector index |

**Example:**

Question: *“How did Company A's acquisition of Company B affect its relationship with Supplier C?”*

- **Traditional RAG:** retrieves chunks mentioning the companies.
- **GraphRAG:** can traverse **Company A → acquired → Company B → supplied by → Supplier C** and reason over those relationships.

### 20.63 How would you handle tables, PDFs, images, and scanned documents in RAG?

Enterprise corpora are messy, so I build a **document-type-aware ingestion pipeline**, not "extract text and chunk".

**Digital PDFs:** layout-aware parsers (Docling, Unstructured, PyMuPDF, pdfplumber, Azure Document Intelligence, AWS Textract) to preserve reading order, headings and lists and strip headers/footers. Chunk by section; keep page numbers for citations.

**Scanned documents:** detect pages with no text layer, pre-process (deskew, denoise), OCR (Tesseract, PaddleOCR, cloud OCR or a VLM). Keep OCR confidence; route low-confidence pages to a stronger model or human review.

**Tables:**
- Extract as structured rows/columns, not flattened text.
- Store as Markdown/HTML for the LLM plus a natural-language summary for embedding.
- Keep small tables whole; chunk large ones by rows, repeating the header.
- For numeric/aggregate questions, load tables into SQL and let an agent query them.

**Images, charts, diagrams:** caption with a VLM and embed the caption (or use multimodal embeddings); store a link to the original so a VLM can inspect it at answer time.

**Cross-cutting:**
- Rich metadata (source, page, section, type, date, ACLs) for filtering and citations.
- Multi-vector indexing: summaries for retrieval, raw content for generation.
- Evaluate per document type — parsing failures are often the biggest source of bad answers.

### 20.64 How would you prevent an LLM from answering using its pretrained knowledge instead of retrieved documents?

You can't fully stop a model from using its priors, but you can constrain it strongly and detect violations.

- **Prompting:** "Answer only from the context below. If it isn't there, say you don't have enough information." Delimit context clearly, require **citations** (chunk IDs) for each claim, and include few-shot examples of abstaining.
- **Fix retrieval first:** most prior-knowledge answers happen when retrieval misses. Improve recall (hybrid, reranking, query rewriting) and abstain when the top relevance score is below a threshold.
- **Structured output:** e.g. `{"answer": ..., "citations": [...], "answerable": bool}`; reject answers without valid citations.
- **Verification:** an NLI model or LLM judge checks each claim against the cited chunks; unsupported answers are regenerated, trimmed or replaced with a refusal.
- **Model choice/tuning:** pick models that follow grounding instructions well; fine-tune on context-grounded QA including unanswerable examples; use low temperature.
- **Monitor:** track faithfulness on sampled traffic; keep "answer not in docs" cases in the eval set.

**Tradeoff:** stricter grounding raises refusal rates. Tune per use case (strict for legal/finance, looser for general assistants) and route abstentions to search results or a human.

### 20.65 Your embeddings retrieve semantically similar but factually incorrect documents. How do you improve retrieval?

This is the gap between **semantic similarity** and **factual relevance**: embeddings place "Q3 revenue rose 5%" near "Q3 revenue fell 5%", and an old policy near the current one. Fixes, roughly in order of payoff:

1. **Hybrid retrieval:** add BM25/sparse search so exact entities, numbers, IDs and dates count; fuse with RRF.
2. **Cross-encoder reranking:** rerank the top 50–100 candidates with a model that reads query and passage together and can catch negation, wrong entity or wrong period.
3. **Metadata filtering:** filter by date, version, product, jurisdiction or tenant before (or during) vector search; mark superseded documents and drop them.
4. **Better chunking:** keep the claim and its qualifiers (entity, period, units) in the same chunk; prepend titles/section headers ("contextual chunk headers").
5. **Domain-adapted embeddings:** fine-tune the embedding model with contrastive pairs, including **hard negatives** that look similar but are factually wrong.
6. **Query rewriting:** extract entities and constraints from the query and turn them into filters or keyword terms.
7. **Data hygiene:** deduplicate and remove stale or contradictory sources — retrieval cannot beat a corpus that contains wrong facts.

**Measure it:** build a labeled set with gold passages and track recall@k and precision@k (or nDCG) separately from answer quality, including adversarial "near miss" queries. Downstream, ask the generator to cite passages and verify that claims are supported.

### 20.66 Your RAG system retrieves the correct document but the LLM gives the wrong answer. Where is the problem?

If retrieval is correct, the failure is in **context construction or generation**. Debug in that order:

**1. Did the right passage actually reach the prompt?**
- The document was retrieved, but the relevant chunk may have been cut by the reranker, truncated by the token budget, or split across chunk boundaries.
- Log the exact final prompt — not just the retriever output.

**2. Is the context noisy or contradictory?**
- Other retrieved chunks may contradict the right one or distract the model; the key passage may be buried in the middle (*lost in the middle*).
- Fix: fewer, reranked chunks; put the best evidence first; deduplicate.

**3. Is the prompt telling the model to ground?**
- Without explicit instructions the model blends parametric knowledge with the context. Require "answer only from the context, cite sources, say 'I don't know' otherwise".

**4. Is the model capable of the reasoning?**
- Multi-hop reasoning, arithmetic over tables, or reading complex layouts (parsed PDFs) can exceed a small model. Try a stronger model, chain-of-thought, or tool use (a calculator, SQL).

**5. Parsing quality:** tables or numbers mangled during ingestion look "retrieved" but are unreadable.

**6. Decoding settings:** high temperature, or truncated outputs because of max tokens.

**How to tell:** measure **faithfulness/groundedness** (is the answer supported by the given context?) separately from **context recall**. High context recall plus low faithfulness pins the problem on generation. Use traces (Langfuse/LangSmith) to inspect failing cases and add them to a regression set.

### 20.67 The LLM answers correctly with 5 retrieved documents but incorrectly with 20. Why?

More context is not free. Likely reasons:

- **Distractors and noise:** documents 6–20 are less relevant; some are near-misses or contradict the right answer, and the model gets pulled toward them.
- **Lost in the middle:** LLMs attend best to the start and end of long contexts; the correct passage may now sit in the middle.
- **Attention dilution:** relevant evidence is a smaller fraction of the prompt, so the signal-to-noise ratio drops.
- **Conflicting information:** outdated versions or different entities introduce contradictions the model resolves wrongly.
- **Context-length limits:** the prompt may be truncated (silently, by the framework or the model limit), cutting the instructions or the key chunk.
- **Instruction drift:** long contexts push system instructions far from the question, weakening grounding behavior.

**Fixes**
- Retrieve more broadly (top-50) but **rerank** and pass only the top 3–8 to the model.
- Order chunks by relevance, putting the best near the start (or the end, next to the question).
- Deduplicate and filter by metadata (date, version).
- Use contextual compression: extract only the relevant sentences.
- Repeat the key instructions after the context.
- Tune *k* empirically: plot answer accuracy and faithfulness against *k* on an eval set — the curve usually peaks and then declines.

**Takeaway:** optimize for **precision of context**, not volume; retrieval recall matters, but the generator needs a short, clean, well-ordered context.

### 20.68 Explain HyDE, query decomposition, multi-query and step-back prompting.

#### RAG Query Optimization Techniques

- **HyDE (Hypothetical Document Embeddings):** Generate a hypothetical answer/document first, then embed it and use that embedding for retrieval. Useful when the **user query and relevant documents use different terminology**.
- **Query Decomposition:** Break a complex question into **smaller sub-questions**, retrieve answers for each, then combine them. Useful for **multi-step questions**.
- **Multi-Query:** Generate several alternative versions of the same query and retrieve documents for each. Improves **recall** when one wording may miss relevant documents.
- **Step-Back Prompting:** Transform a specific question into a **broader conceptual question** before retrieval/reasoning. Helps when answering requires **general principles or background knowledge**.

**Quick distinction:**

**HyDE → hypothetical document**

**Decomposition → smaller questions**

**Multi-query → different phrasings**

**Step-back → broader question**

### 20.69 How would you debug an LLM/RAG system that hallucinates?

#### Debugging Hallucinations in an LLM/RAG System

1. **Check retrieval**
   - Are the correct documents being retrieved?
   - Check **Recall@K, Precision@K, and context relevance**.
2. **Check retrieved context**
   - Is the required answer actually present?
   - Check for noisy, incomplete, outdated, or conflicting chunks.
3. **Check generation**
   - Verify whether the model's answer is supported by the retrieved context.
   - Reduce unnecessary generation freedom through better prompts, temperature, and structured output.
4. **Check the pipeline**
   - Inspect query rewriting, chunking, embeddings, reranking, and context assembly.
   - Trace the complete request to identify where incorrect information enters.
5. **Add guardrails**
   - Require citations/evidence.
   - Instruct the model to say **"I don't know"** when evidence is insufficient.
   - Add factuality/groundedness evaluation.

### 20.70 What is Open Knowledge Format (OKF), and how is it different from RAG?

**OKF** is a structured way to **represent and organize knowledge** so that machines and AI systems can easily understand, search, and exchange it.

It can represent:

- **Entities** — people, companies, products
- **Facts** — attributes and information
- **Relationships** — connections between entities
- **Sources/metadata** — where the information came from

#### OKF vs RAG

|  | OKF | RAG |
| --- | --- | --- |
| Purpose | Structure knowledge | Retrieve knowledge for an LLM |
| Focus | **Knowledge representation** | **Knowledge retrieval + generation** |
| Typical data | Entities, facts, relationships | Documents, chunks, vectors, databases |
| LLM required | No | Usually yes |

**Example:**

```txt
OKF:
Einstein → developed → Theory of Relativity
```

A RAG system could **retrieve this knowledge** when a user asks:

"What did Einstein develop?"

and provide it to the LLM to generate the answer.

---

← [Chapter 19. Embeddings and Vector Databases](19-embeddings-and-vector-databases.md) · [Table of contents](../README.md) · [Chapter 21. AI Agents and Agentic Systems](21-ai-agents-and-agentic-systems.md) →
