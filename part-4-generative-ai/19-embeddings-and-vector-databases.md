# Chapter 19. Embeddings and Vector Databases

_Part IV — Generative AI and LLM Engineering_

Embedding models, similarity search, ANN index internals (HNSW, IVF, product quantisation), late interaction and learned sparse retrieval, and sizing, securing and operating a vector store.

This chapter contains **37 questions**.

## Contents

- **19.1** [What are embeddings in the context of AI engineering?](#191-what-are-embeddings-in-the-context-of-ai-engineering)
- **19.2** [How do embedding models convert text to vectors?](#192-how-do-embedding-models-convert-text-to-vectors)
- **19.3** [What is the difference between sparse and dense embeddings?](#193-what-is-the-difference-between-sparse-and-dense-embeddings)
- **19.4** [Explain cosine similarity, dot product, and Euclidean distance for vector search.](#194-explain-cosine-similarity-dot-product-and-euclidean-distance-for-vector-search)
- **19.5** [What is a vector database, and how does it differ from a traditional database?](#195-what-is-a-vector-database-and-how-does-it-differ-from-a-traditional-database)
- **19.6** [How does Approximate Nearest Neighbor (ANN) search work?](#196-how-does-approximate-nearest-neighbor-ann-search-work)
- **19.7** [How do you choose the right embedding model for your use case?](#197-how-do-you-choose-the-right-embedding-model-for-your-use-case)
- **19.8** [What is embedding dimensionality, and how does it affect performance and cost?](#198-what-is-embedding-dimensionality-and-how-does-it-affect-performance-and-cost)
- **19.9** [How do you handle embedding drift when the embedding model is updated?](#199-how-do-you-handle-embedding-drift-when-the-embedding-model-is-updated)
- **19.10** [What are multi-modal embeddings, and how are they generated?](#1910-what-are-multi-modal-embeddings-and-how-are-they-generated)
- **19.11** [How do you index and query multi-tenant data in a vector database?](#1911-how-do-you-index-and-query-multi-tenant-data-in-a-vector-database)
- **19.12** [What is quantization of embeddings, and how does it reduce storage costs?](#1912-what-is-quantization-of-embeddings-and-how-does-it-reduce-storage-costs)
- **19.13** [How do you benchmark and evaluate embedding model quality?](#1913-how-do-you-benchmark-and-evaluate-embedding-model-quality)
- **19.14** [What is the role of metadata in vector databases?](#1914-what-is-the-role-of-metadata-in-vector-databases)
- **19.15** [How do you handle large-scale vector search with billions of vectors?](#1915-how-do-you-handle-large-scale-vector-search-with-billions-of-vectors)
- **19.16** [What is hybrid search (combining keyword search with vector search)?](#1916-what-is-hybrid-search-combining-keyword-search-with-vector-search)
- **19.17** [How do you fine-tune an embedding model for a specific domain?](#1917-how-do-you-fine-tune-an-embedding-model-for-a-specific-domain)
- **19.18** [Your vector database for RAG is consuming too much memory. How do you reduce it?](#1918-your-vector-database-for-rag-is-consuming-too-much-memory-how-do-you-reduce-it)
- **19.19** [Your vector database cannot scale to millions of embeddings. How do you fix the bottleneck?](#1919-your-vector-database-cannot-scale-to-millions-of-embeddings-how-do-you-fix-the-bottleneck)
- **19.20** [Your vector search returns irrelevant results despite high similarity scores. How do you fix it?](#1920-your-vector-search-returns-irrelevant-results-despite-high-similarity-scores-how-do-you-fix-it)
- **19.21** [You deployed a new embedding model, and search quality crashed overnight. How do you handle embedding drift?](#1921-you-deployed-a-new-embedding-model-and-search-quality-crashed-overnight-how-do-you-handle-embedding-drift)
- **19.22** [Your semantic search fails for short queries. How do you improve it?](#1922-your-semantic-search-fails-for-short-queries-how-do-you-improve-it)
- **19.23** [What is InfoNCE loss, and where is it used?](#1923-what-is-infonce-loss-and-where-is-it-used)
- **19.24** [Why can't you use one embedding model for every domain?](#1924-why-cant-you-use-one-embedding-model-for-every-domain)
- **19.25** [How do you choose between Pinecone, Weaviate, Qdrant, pgvector and other vector stores for a given client?](#1925-how-do-you-choose-between-pinecone-weaviate-qdrant-pgvector-and-other-vector-stores-for-a-given-client)
- **19.26** [How does the HNSW index work, and how do you tune it?](#1926-how-does-the-hnsw-index-work-and-how-do-you-tune-it)
- **19.27** [How do IVF and product quantisation work, and when would you use them instead of HNSW?](#1927-how-do-ivf-and-product-quantisation-work-and-when-would-you-use-them-instead-of-hnsw)
- **19.28** [How do you benchmark a vector index for recall, latency and cost?](#1928-how-do-you-benchmark-a-vector-index-for-recall-latency-and-cost)
- **19.29** [What are Matryoshka embeddings, and how do they reduce cost?](#1929-what-are-matryoshka-embeddings-and-how-do-they-reduce-cost)
- **19.30** [What is late interaction (ColBERT), and how does it compare with bi-encoders and cross-encoders?](#1930-what-is-late-interaction-colbert-and-how-does-it-compare-with-bi-encoders-and-cross-encoders)
- **19.31** [How does filtered vector search work, and why can filters hurt recall?](#1931-how-does-filtered-vector-search-work-and-why-can-filters-hurt-recall)
- **19.32** [What are learned sparse embeddings such as SPLADE, and how do they compare with BM25 and dense embeddings?](#1932-what-are-learned-sparse-embeddings-such-as-splade-and-how-do-they-compare-with-bm25-and-dense-embeddings)
- **19.33** [How do you keep a vector index fresh and consistent when source documents change frequently?](#1933-how-do-you-keep-a-vector-index-fresh-and-consistent-when-source-documents-change-frequently)
- **19.34** [Why do many embedding models need different prefixes or instructions for queries and documents?](#1934-why-do-many-embedding-models-need-different-prefixes-or-instructions-for-queries-and-documents)
- **19.35** [Why should you not use a fixed similarity score threshold to decide whether a result is relevant?](#1935-why-should-you-not-use-a-fixed-similarity-score-threshold-to-decide-whether-a-result-is-relevant)
- **19.36** [Are embeddings sensitive data, and how do you secure a vector database?](#1936-are-embeddings-sensitive-data-and-how-do-you-secure-a-vector-database)
- **19.37** [How would you estimate the size and cost of a vector database deployment?](#1937-how-would-you-estimate-the-size-and-cost-of-a-vector-database-deployment)

---

### 19.1 What are embeddings in the context of AI engineering?

#### What are embeddings?

- **Embeddings are numerical vector representations of data** such as text, images, audio, or code.
- They convert unstructured data into vectors that capture **semantic meaning and relationships**.
- Similar concepts tend to have vectors that are **close together** in the embedding space.
- For example, `"car"` and `"automobile"` would typically have similar embeddings, even though the words are different.

#### How are embeddings used?

- **Semantic search:** Find documents based on meaning rather than exact keywords.
- **RAG:** Convert documents and queries into embeddings, then retrieve the most relevant chunks.
- **Recommendation systems:** Find similar users, products, articles, or content.
- **Clustering:** Group semantically similar data.
- **Classification:** Use embeddings as features for downstream models.
- **Duplicate detection:** Identify semantically similar or repeated content.
- **Image/text matching:** Compare different modalities in a shared embedding space.

#### In a RAG system

1. Split documents into chunks.
2. Generate an embedding for each chunk.
3. Store the vectors in a **vector database** such as pgvector, Pinecone, or OpenSearch.
4. Convert the user's query into an embedding.
5. Perform a similarity search.
6. Retrieve the most relevant chunks.
7. Pass them to the LLM as context.

#### Key idea

> **Embedding = converting data into a vector that represents its meaning, so AI systems can efficiently measure similarity and retrieve related information.**

### 19.2 How do embedding models convert text to vectors?

#### How do embedding models convert text to vectors?

- **1. Tokenization**
  - The text is split into tokens using a tokenizer such as BPE or WordPiece.
  - Example: `"Machine learning is powerful"` → tokens representing the words/subwords.
- **2. Token IDs**
  - Each token is mapped to an integer ID from the model's vocabulary.
- **3. Transformer processing**
  - The token IDs are passed through a Transformer.
  - Self-attention allows each token to incorporate information from the surrounding context.
- **4. Contextual representations**
  - The Transformer produces a high-dimensional vector for each token.
  - The representation depends on context, so the same word can have different representations in different sentences.
- **5. Pooling**
  - The token representations are combined into a **single fixed-size vector** representing the entire text.
  - Common approaches include mean pooling or using a special pooled representation.
- **6. Normalization**
  - The resulting vector may be normalized, often to unit length, which makes similarity calculations more convenient.
- **7. Similarity comparison**
  - Vectors can then be compared using **cosine similarity**, dot product, or Euclidean distance.
  - Semantically similar texts tend to have similar vectors.

#### Simple flow

**Text → Tokenization → Token IDs → Transformer → Contextual token vectors → Pooling → Text embedding vector**

#### Important point

An embedding model doesn't simply assign a fixed vector to each word. **It learns to represent the meaning of the entire text in vector space**, using context and patterns learned during training.

### 19.3 What is the difference between sparse and dense embeddings?

#### Sparse vs Dense Embeddings

| Aspect | Sparse Embeddings | Dense Embeddings |
| --- | --- | --- |
| Representation | Mostly **zeros** | Most values are **non-zero** |
| Dimensions | Usually very high | Usually lower |
| Values | Often represent term presence/importance | Learned numerical representations |
| Captures | Mainly lexical relationships | Semantic relationships |
| Example | TF-IDF, BM25-style vectors | Transformer embeddings |
| Search | Excellent for exact keywords | Excellent for semantic similarity |
| Storage | Can be efficient with sparse storage | Requires storing many floating-point values |

#### Sparse embeddings

- Represent text using a large vocabulary where most dimensions are zero.
- Each dimension typically corresponds to a term or feature.
- Strong at matching **exact words and rare terms**.
- Example:
  - Query: `"neural network"`
  - Document containing exactly `"neural network"` gets strong lexical matching.

#### Dense embeddings

- Represent text as a relatively compact vector, such as 384, 768, or 1536 dimensions.
- Values are generally non-zero.
- Learned models capture **semantic meaning**.
- `"car"` and `"automobile"` can have similar vectors even without sharing the same word.

#### In modern RAG

A strong retrieval system often combines both:

- **Sparse retrieval:** captures exact keywords, names, IDs, technical terms.
- **Dense retrieval:** captures semantic meaning.
- **Hybrid search:** combines both scores to improve retrieval quality.

### 19.4 Explain cosine similarity, dot product, and Euclidean distance for vector search.

#### Cosine Similarity, Dot Product, and Euclidean Distance

These metrics measure how similar or different two vectors are in **vector search**, such as semantic search and RAG.

#### 1. Cosine Similarity

- Measures the **angle/direction** between two vectors.
- Focuses on whether vectors point in a similar direction, not their magnitude.
- Range: **$-1$ to $1$**.
  - $1$ → same direction
  - $0$ → orthogonal/unrelated
  - $-1$ → opposite direction
- Common for **text embeddings** and semantic search.
- Higher value means more similar.

#### 2. Dot Product

- Multiplies corresponding elements of two vectors and sums the results.
- Measures both **direction and magnitude**.
- Higher value generally indicates greater similarity.
- If vectors are **L2-normalized**, the dot product is equivalent to cosine similarity.
- Commonly supported by vector databases and retrieval systems.

#### 3. Euclidean Distance

- Measures the **straight-line distance** between two vectors.
- A smaller distance means the vectors are more similar.
- Unlike cosine similarity, it is affected by vector magnitude.
- Useful when the embedding space's actual geometric distance is meaningful.

#### Comparison

| Metric | Measures | Best match | Magnitude matters? |
| --- | --- | --- | --- |
| **Cosine similarity** | Direction/angle | Higher | No |
| **Dot product** | Direction + magnitude | Higher | Yes |
| **Euclidean distance** | Geometric distance | Lower | Yes |

#### Important for Vector Search

The best metric depends on the **embedding model and how it was trained**. You should generally use the similarity metric recommended by the embedding model rather than choosing one arbitrarily.

### 19.5 What is a vector database, and how does it differ from a traditional database?

#### What is a Vector Database?

- A **vector database** is a database designed to store, index, and search **high-dimensional vectors**, such as text embeddings, image embeddings, or audio embeddings.
- It enables **similarity search**: given a query vector, it finds the vectors that are most similar using metrics such as:
  - Cosine similarity
  - Dot product
  - Euclidean distance
- Commonly used in **RAG, semantic search, recommendation systems, and AI applications**.

#### How does it work?

1. Convert data into embeddings using an embedding model.
2. Store the embeddings along with metadata.
3. Build a vector index for efficient similarity search.
4. Convert a user query into an embedding.
5. Search for the nearest vectors.
6. Return the most relevant documents/items and metadata.

#### Vector Database vs Traditional Database

| Aspect | Vector Database | Traditional Database |
| --- | --- | --- |
| Primary data | Vectors + metadata | Structured data |
| Main search | Similarity/nearest-neighbor | Exact/filter/range queries |
| Query basis | Semantic similarity | Values, keys, conditions |
| Typical index | HNSW, IVF, etc. | B-tree, hash, etc. |
| AI use | RAG, semantic search, recommendations | Transactions, business applications |
| Example query | "Find documents similar to this query" | "Find users where age > 30" |

#### Important distinction

A vector database is **not necessarily a replacement for a traditional database**. Modern systems often combine both.

For example, **PostgreSQL + pgvector** can store:

- Regular relational data
- Embeddings
- Metadata
- Vector indexes

This allows an application to perform traditional SQL filtering together with vector similarity search.

### 19.6 How does Approximate Nearest Neighbor (ANN) search work?

#### How does Approximate Nearest Neighbor (ANN) search work?

- **ANN search** finds vectors that are *approximately* the nearest to a query vector without comparing the query against every vector in the database.
- This makes vector search much faster when dealing with **millions or billions of embeddings**.
- It trades a small amount of **recall/accuracy for significant speed and scalability**.

#### How it works

1. **Create embeddings**
   - Convert documents, images, or other data into vectors.
2. **Build an index**
   - Instead of storing vectors only for brute-force comparison, an ANN algorithm organizes them into a searchable structure.
   - Common approaches include:
     - **HNSW** — graph-based search
     - **IVF** — clusters vectors into partitions
     - **PQ** — compresses vectors to reduce memory usage
3. **Embed the query**
   - Convert the user's query into the same vector space.
4. **Search the index**
   - The algorithm explores only promising regions or candidates rather than the entire dataset.
5. **Rank candidates**
   - Calculate the actual similarity/distance for the selected candidates.
6. **Return top-k results**
   - Return the most similar vectors/documents.

#### ANN vs Exact Nearest Neighbor

| Aspect | Exact Search | ANN Search |
| --- | --- | --- |
| Comparison | Potentially every vector | Subset of vectors |
| Accuracy | Exact | Approximate |
| Speed | Slower at scale | Much faster |
| Recall | 100% | Usually slightly lower |
| Scalability | Limited for huge datasets | Excellent |

#### Example

If a database contains **10 million vectors**:

- Exact search may compare the query with all 10 million vectors.
- ANN builds an index and might evaluate only a much smaller candidate set.
- It can therefore provide results much faster while maintaining high recall.

### 19.7 How do you choose the right embedding model for your use case?

#### How do you choose the right embedding model?

- **1. Define the use case**
  - Semantic search, RAG, recommendations, clustering, classification, code search, multilingual search, etc.
  - Different tasks may require different embedding models.
- **2. Check domain relevance**
  - General-purpose models work well for common text.
  - For specialized domains such as legal, medical, finance, or technical content, prefer a model that performs well on similar data.
- **3. Evaluate retrieval quality**
  - Test candidate models on your **real queries and documents**.
  - Measure metrics such as:
    - **Recall@K**
    - **Precision@K**
    - **MRR**
    - **NDCG**
  - For RAG, also measure downstream **answer quality and faithfulness**.
- **4. Consider language support**
  - For multilingual applications, choose a model with strong performance across the required languages.
  - Don't assume an English-optimized model will work equally well for other languages.
- **5. Consider embedding dimensions**
  - Higher dimensions can potentially capture more information but require more storage and computation.
  - Lower dimensions can reduce cost and latency.
- **6. Consider latency and throughput**
  - Embedding generation happens frequently in production systems.
  - For large document collections or high query volume, benchmark inference speed and batching performance.
- **7. Consider infrastructure and cost**
  - Compare API-based models with self-hosted/open-source models.
  - Consider GPU/CPU requirements, API pricing, storage, and operational complexity.
- **8. Check context length**
  - Make sure the model can handle the size of the text chunks you intend to embed.
  - Chunking strategy and embedding model should be evaluated together.
- **9. Check similarity metric compatibility**
  - Verify whether the model expects cosine similarity, dot product, or another metric.
  - Follow the model's recommended normalization and indexing strategy.
- **10. Benchmark before deployment**
  - Build a representative evaluation dataset.
  - Compare several candidate models on **quality, latency, memory, and cost**.
  - Choose based on the overall production trade-off, not just benchmark scores.

#### Practical approach

**Start with a strong general-purpose model → test it on your real data → compare alternatives → evaluate retrieval + downstream RAG quality → benchmark cost/latency → deploy the best trade-off.**

### 19.8 What is embedding dimensionality, and how does it affect performance and cost?

#### What is Embedding Dimensionality?

- **Embedding dimensionality** is the number of numerical values in an embedding vector.
- For example:
  - 384-dimensional → $[x_1, x_2, \ldots, x_{384}]$
  - 768-dimensional → 768 values
  - 1536-dimensional → 1536 values
- Each dimension represents a learned feature of the data, although individual dimensions usually don't have simple human-interpretable meanings.

#### How does dimensionality affect performance?

- **Higher dimensions**
  - Can capture more complex semantic information.
  - May improve retrieval quality for some tasks.
  - Increase computation during similarity search.
  - Increase memory and storage requirements.
- **Lower dimensions**
  - Faster similarity calculations.
  - Smaller indexes and lower memory usage.
  - Lower storage requirements.
  - May lose some semantic information and reduce retrieval quality.

#### How does it affect cost?

Suppose you store **10 million embeddings**:

- 384 dimensions → 3.84 billion values
- 768 dimensions → 7.68 billion values
- 1536 dimensions → 15.36 billion values

If each value uses 4 bytes (FP32), raw vector storage is approximately:

- 384D → **15.4 GB**
- 768D → **30.7 GB**
- 1536D → **61.4 GB**

Actual vector-database storage will be higher because indexes and metadata also consume space.

#### Important trade-off

**Higher dimensionality ≠ automatically better embeddings.**

The best dimensionality is the one that provides sufficient retrieval quality at an acceptable **latency, memory, storage, and cost**.

### 19.9 How do you handle embedding drift when the embedding model is updated?

#### How do you handle embedding drift when the embedding model is updated?

**Embedding drift** occurs when a new embedding model produces vectors that are not directly comparable with vectors generated by the previous model.

#### Why is it a problem?

- Existing documents may use **old embeddings**.
- New queries may use **new embeddings**.
- The two embedding spaces may have different representations.
- Similarity search can therefore return poor or incorrect results.

#### How to handle it

1. **Version the embedding models**
   - Track the model name, version, dimensions, normalization, and configuration.
   - Never treat embeddings from different models as interchangeable.
2. **Create a new vector index**
   - Generate new embeddings for the existing document collection using the new model.
   - Store them in a separate index/collection.
3. **Dual-run during migration**
   - Temporarily generate/query both old and new embeddings.
   - Compare retrieval quality and production metrics.
4. **Evaluate retrieval quality**
   - Use a fixed evaluation dataset.
   - Compare **Recall@K, MRR, NDCG**, and downstream RAG answer quality.
5. **Migrate gradually**
   - If the new model performs better, progressively move traffic to the new index.
   - Keep the old index temporarily for rollback.
6. **Re-embed new documents consistently**
   - Once migration starts, make sure new documents are embedded with the selected model rather than mixing incompatible embedding spaces.
7. **Monitor production**
   - Track retrieval quality, zero-result rates, latency, and user feedback.
   - Watch for unexpected changes after migration.

#### Practical migration strategy

**Old Model → Old Index**

**New Model → Re-embed documents → New Index → Evaluate → Shadow/Dual Query → Gradual Cutover → Retire Old Index**

#### Important point

You generally **should not mix embeddings from different model versions in the same similarity space** unless the models are explicitly designed to be compatible.

#### When the new model has a different dimension

- If the old model produces 768-dimensional vectors and the new one 1536, most indexes cannot even store them together. Create a new index or collection configured for the new dimension, re-embed the corpus, and cut over as above.
- **Do not truncate or pad vectors** to force them to the same size. This destroys the representation and badly hurts retrieval. The one exception is a model trained with Matryoshka representation learning, whose vectors are designed to be truncated (and then re-normalised); even then, every vector in an index must use the same truncation.
- Update the ingestion pipeline at the same time, and store the model version and dimension with the index configuration.

### 19.10 What are multi-modal embeddings, and how are they generated?

#### What are Multi-Modal Embeddings?

- **Multi-modal embeddings** represent different types of data—such as **text, images, audio, and video**—as vectors in a **shared or aligned embedding space**.
- This allows an AI system to compare and retrieve information across different modalities.
- For example, a text query like `"a red sports car"` can be compared against **image embeddings** to find visually/semantically relevant images.

#### How are they generated?

1. **Separate modality encoders**
   - A model processes each type of input using an appropriate encoder.
   - Text → text encoder
   - Image → vision encoder
   - Audio → audio encoder
2. **Convert inputs into representations**
   - Each encoder produces a high-dimensional representation for its input.
3. **Project into a shared embedding space**
   - Projection layers transform the representations into compatible vector spaces.
4. **Align the modalities during training**
   - Models are trained so that related inputs have **similar embeddings**.
   - For example, an image of a dog and the text `"a dog"` are trained to have nearby vectors.
5. **Use similarity search**
   - Once embedded, vectors can be compared using cosine similarity, dot product, etc.

#### Example

**Image:** Photo of a mountain

**Text:** `"snow-covered mountain"`

Both can be converted into vectors:

`Image → [0.12, -0.31, 0.84, ...]`

`Text → [0.15, -0.28, 0.81, ...]`

Because they represent related concepts, their vectors should be relatively close.

#### Common applications

- **Text-to-image search**
- **Image-to-text retrieval**
- **Multimodal RAG**
- **Visual question answering**
- **Cross-modal recommendation**
- **Video and audio retrieval**
- **Document understanding**

### 19.11 How do you index and query multi-tenant data in a vector database?

#### Multi-Tenant Vector Search

**Multi-tenancy** means multiple customers or organizations share the same vector database while their data remains isolated.

#### How to index multi-tenant data

1. **Add tenant metadata**
   - Store a `tenant_id` with every vector.
   - Example:
     - `tenant_id = company_A`
     - `document_id = doc_123`
     - `embedding = [...]`
2. **Choose an isolation strategy**
   - **Shared index + metadata filtering:** All tenants share an index, but every query filters by `tenant_id`.
   - **Separate collections/namespaces:** Each tenant gets its own logical vector space.
   - **Separate databases/indexes:** Strongest isolation, but higher operational cost.
3. **Use metadata filters**
   - Always restrict retrieval to the authenticated tenant.
   - Conceptually:`similarity_search(query_vector, filter={"tenant_id": current_tenant})`
4. **Index for scale**
   - For many tenants, use ANN indexes such as HNSW or IVF.
   - Consider tenant size and query volume when deciding whether large tenants need dedicated indexes.

#### How to query

1. Authenticate the user.
2. Determine the user's `tenant_id` from the trusted authentication/session context.
3. Generate the query embedding.
4. Perform vector similarity search **with the tenant filter**.
5. Return only documents belonging to that tenant.

#### Important security consideration

**Never trust `tenant_id` supplied directly by the client.**

The backend should derive it from the authenticated user's identity and enforce the filter server-side. Otherwise, a malicious user could potentially query another tenant's vectors.

#### Shared vs Separate Indexes

| Strategy | Isolation | Cost | Scalability |
| --- | --- | --- | --- |
| Shared index + metadata filter | Good | Low | Excellent |
| Separate namespace/collection | Better | Medium | Good |
| Separate index/database | Strongest | High | Depends on infrastructure |

#### Practical approach

For most SaaS RAG systems:

> **Shared vector infrastructure + tenant metadata + mandatory server-side filtering** is a good default.

For highly sensitive or regulated workloads, stronger physical/logical isolation may be appropriate.

### 19.12 What is quantization of embeddings, and how does it reduce storage costs?

#### What is Embedding Quantization?

- **Embedding quantization** converts high-precision embedding values into **lower-precision representations**.
- For example, embeddings stored as:
  - **FP32** → 32 bits per value
  - **FP16** → 16 bits per value
  - **INT8** → 8 bits per value
- The goal is to reduce **storage, memory usage, and sometimes search computation** while keeping retrieval quality acceptable.

#### How does it reduce storage?

Suppose you have **10 million embeddings**, each with **768 dimensions**.

Using FP32:

- 768 × 4 bytes = **3,072 bytes per vector**
- 10M vectors ≈ **30.7 GB** of raw vector data

Using INT8:

- 768 × 1 byte = **768 bytes per vector**
- 10M vectors ≈ **7.68 GB**

So INT8 can reduce raw vector storage by roughly **75%** compared with FP32.

#### How does quantization work?

1. Start with high-precision floating-point values.
2. Determine an appropriate range or scaling scheme.
3. Map the continuous values to a smaller set of discrete values.
4. Store the quantized representation.
5. During search, use the quantized vectors to calculate approximate similarity.

#### Benefits

- Lower **storage cost**
- Lower **RAM usage**
- Smaller vector indexes
- Potentially faster similarity computation
- Allows larger datasets to fit into available hardware

#### Trade-off

- Quantization introduces **approximation error**.
- Excessive quantization can reduce **recall and ranking quality**.
- Therefore, you should benchmark retrieval quality before deploying it.

### 19.13 How do you benchmark and evaluate embedding model quality?

#### How do you benchmark and evaluate embedding model quality?

- **1. Define the retrieval task**
  - Determine what the embeddings will be used for: semantic search, RAG, recommendations, clustering, classification, etc.
  - Evaluation should reflect the actual production use case.
- **2. Create a representative evaluation dataset**
  - Collect realistic **queries and relevant documents**.
  - Include easy, difficult, ambiguous, long, short, and domain-specific examples.
  - Keep a separate test set that is not used during model selection.
- **3. Measure retrieval quality**
  - **Recall@K:** Is the relevant document retrieved within the top K?
  - **Precision@K:** How many of the top K results are relevant?
  - **MRR:** How highly is the first relevant result ranked?
  - **NDCG@K:** Measures ranking quality when relevance has different levels.
- **4. Compare candidate models**
  - Run the same evaluation dataset and retrieval configuration against each embedding model.
  - Compare retrieval metrics rather than relying only on public benchmarks.
- **5. Evaluate downstream RAG quality**
  - Retrieval quality is not the final goal.
  - Measure:
    - Answer correctness
    - Faithfulness/groundedness
    - Context relevance
    - Hallucination rate
  - A model with slightly lower Recall@K could still produce better overall RAG results depending on the retrieved context.
- **6. Measure operational performance**
  - Embedding generation latency
  - Query latency
  - Throughput
  - Memory usage
  - Storage/index size
  - API or infrastructure cost
- **7. Test robustness**
  - Evaluate spelling variations, paraphrases, abbreviations, long queries, multilingual inputs, domain terminology, and noisy text.
- **8. Test different retrieval configurations**
  - Compare chunk sizes, similarity metrics, normalization, and ANN parameters.
  - The embedding model should not be evaluated independently from the retrieval pipeline.

#### Example evaluation

Suppose you're building a technical-document RAG system:

**1000 real queries → known relevant documents → generate embeddings → ANN search → calculate Recall@5, MRR, NDCG → evaluate RAG answers → measure latency and cost**

Then compare several embedding models using the same pipeline.

### 19.14 What is the role of metadata in vector databases?

#### What is the role of Metadata in Vector Databases?

- **Metadata is additional information stored alongside each embedding.**
- It describes the source or properties of the vector, while the embedding represents its semantic content.

#### Example

A document chunk might contain:

- `embedding` → vector representation
- `document_id` → `doc_123`
- `tenant_id` → `company_A`
- `source` → `employee_handbook.pdf`
- `page` → `15`
- `department` → `HR`
- `created_at` → date
- `language` → `English`

#### Why is metadata important?

- **Filtering**
  - Restrict vector search to specific documents, users, departments, dates, or tenants.
  - Example: search only documents where `department = "HR"`.
- **Multi-tenancy**
  - Ensure users retrieve only vectors belonging to their own tenant.
- **Access control**
  - Apply permissions or security policies during retrieval.
- **Hybrid retrieval**
  - Combine semantic similarity with metadata filters.
  - Example: "Find documents semantically similar to this query, but only from 2026."
- **Result attribution**
  - Metadata tells the application where the retrieved content came from, such as the document name, URL, page, or chunk.
- **Data management**
  - Helps with updating, deleting, re-indexing, and tracking embeddings.

#### Important distinction

**Embedding = semantic representation**

**Metadata = structured information about the embedding**

For example, two documents could have similar embeddings but different `tenant_id`, `department`, or `access_level`. Metadata allows the system to enforce those constraints during retrieval.

### 19.15 How do you handle large-scale vector search with billions of vectors?

#### How do you handle large-scale vector search with billions of vectors?

At billion-scale, the main goals are **high recall, low latency, manageable memory, and reasonable cost**.

- **1. Use Approximate Nearest Neighbor (ANN) search**
  - Avoid brute-force comparison against billions of vectors.
  - Use indexes such as **HNSW, IVF, or DiskANN**.
  - Tune parameters to balance recall and latency.
- **2. Partition or shard the data**
  - Distribute vectors across multiple machines/nodes.
  - Partition by tenant, geography, category, or another suitable dimension when possible.
  - Route queries only to relevant partitions to reduce search work.
- **3. Use vector quantization**
  - Convert FP32 vectors to lower-precision formats such as INT8.
  - Techniques such as **Product Quantization (PQ)** can reduce memory substantially.
  - This is particularly important when the raw vector dataset cannot fit in RAM.
- **4. Use hierarchical retrieval**
  - First, perform a cheap coarse search to identify promising partitions/candidates.
  - Then perform a more accurate similarity search on the smaller candidate set.
- **5. Optimize indexing strategy**
  - HNSW is useful for high-recall in-memory search but can require substantial RAM.
  - IVF/PQ can provide better memory efficiency for very large collections.
  - Disk-based indexes can reduce RAM requirements.
- **6. Use metadata filtering**
  - Apply filters such as `tenant_id`, category, date, or permissions before/within vector retrieval.
  - Searching a smaller relevant subset can significantly improve efficiency.
- **7. Separate ingestion and query workloads**
  - Build/update indexes asynchronously.
  - Keep query-serving infrastructure isolated from heavy indexing workloads.
- **8. Cache frequently used queries/results**
  - Cache embeddings and frequently requested search results where appropriate.
  - This reduces repeated computation.
- **9. Monitor the right metrics**
  - **Recall@K**
  - Query latency / p95 / p99
  - Throughput (QPS)
  - Memory usage
  - Index size
  - Storage cost
  - CPU/GPU utilization

#### Practical architecture

**Billions of vectors → Sharding → ANN index → Quantization → Candidate retrieval → Re-ranking → Top-K results**

#### Key trade-off

You don't simply optimize for the **fastest search**. You tune the system for the required **recall vs latency vs memory vs cost**.

### 19.16 What is hybrid search (combining keyword search with vector search)?

#### What is Hybrid Search?

- **Hybrid search combines traditional keyword search with vector/semantic search** to retrieve more relevant results.
- It combines:
  - **Keyword search** → matches exact words and terms.
  - **Vector search** → matches semantic meaning.

#### Why combine them?

Each method has different strengths:

**Keyword search**

- Excellent for exact terms, names, IDs, product codes, error messages, and technical terms.
- Can miss relevant documents when different words express the same meaning.

**Vector search**

- Understands semantic similarity and paraphrasing.
- Can sometimes miss exact keywords, especially rare identifiers or precise terminology.

#### How does hybrid search work?

1. Convert the query into a **text query** for keyword search.
2. Convert the same query into an **embedding** for vector search.
3. Run both searches.
4. Combine their results or scores.
5. Rank the combined candidates.
6. Return the top-K results.

#### Example

Query:

> `"AWS Lambda timeout error"`

Keyword search may strongly match documents containing **"AWS Lambda"**, **"timeout"**, and **"error"**.

Vector search may retrieve a document saying:

> `"Serverless function execution exceeds the configured time limit"`

Even though the wording is different.

Hybrid search can retrieve **both exact terminology and semantically related content**.

#### Common score combination

A system can combine normalized scores:

$$
\text{Final Score} = \alpha \times \text{Keyword Score} + (1 - \alpha) \times \text{Vector Score}
$$

The weight $\alpha$ controls how much importance is given to keyword versus semantic retrieval.

Another common approach is **Reciprocal Rank Fusion (RRF)**, which combines rankings from the two retrieval systems rather than directly combining their raw scores.

#### Hybrid Search in RAG

A typical RAG pipeline is:

**User Query → Keyword Search + Vector Search → Combine/Rerank → Top-K Chunks → LLM**

This is particularly useful for enterprise RAG where documents contain **exact technical terms, IDs, names, and domain-specific terminology**.

### 19.17 How do you fine-tune an embedding model for a specific domain?

#### How do you fine-tune an embedding model for a specific domain?

- **1. Define the retrieval objective**
  - Decide what should be considered similar.
  - Examples: question → relevant document, product → similar product, query → relevant support article.
- **2. Collect domain-specific data**
  - Ideally, use **query–document relevance pairs** from the target domain.
  - Example:
    - Query: `"How do I reset my enterprise VPN password?"`
    - Positive: relevant IT support article
    - Negative: unrelated or less relevant article
- **3. Create high-quality positive and negative examples**
  - Positive pairs should be genuinely relevant.
  - Hard negatives are especially valuable: documents that look similar but are actually incorrect.
- **4. Choose a pretrained embedding model**
  - Start with a strong general-purpose model.
  - Fine-tuning is useful when the domain has specialized terminology or a domain-specific notion of relevance.
- **5. Fine-tune with a contrastive/ranking objective**
  - Train the model to place relevant pairs **closer together** and irrelevant pairs **farther apart**.
  - Common approaches include contrastive learning, triplet loss, and ranking losses.
- **6. Use parameter-efficient fine-tuning when appropriate**
  - LoRA/PEFT can reduce training memory and compute.
  - Full fine-tuning may be appropriate when sufficient data and compute are available.
- **7. Evaluate on held-out domain data**
  - Use metrics such as:
    - **Recall@K**
    - **MRR**
    - **NDCG**
    - **Precision@K**
  - Compare against the original embedding model.
- **8. Evaluate the complete RAG pipeline**
  - Better embeddings should improve retrieval and ultimately improve:
    - Context relevance
    - Answer correctness
    - Faithfulness
    - Hallucination rate
- **9. Monitor for regression**
  - Check whether domain fine-tuning damages general-purpose retrieval capabilities.
  - Keep a general-domain evaluation set alongside the domain-specific test set.

#### Practical flow

**Pretrained Embedding Model → Domain Query/Document Pairs → Hard Negatives → Contrastive/Ranking Fine-Tuning → Retrieval Evaluation → RAG Evaluation**

### 19.18 Your vector database for RAG is consuming too much memory. How do you reduce it?

#### How do you reduce memory usage in a RAG vector database?

- **1. Quantize embeddings**
  - Convert FP32 vectors to **FP16, INT8, or lower-precision formats**.
  - Can dramatically reduce vector memory while maintaining acceptable retrieval quality.
- **2. Use Product Quantization (PQ)**
  - Compress high-dimensional vectors into compact codes.
  - Particularly useful for very large collections.
  - Trade-off: some loss in retrieval accuracy.
- **3. Reduce embedding dimensionality**
  - Use a smaller embedding model or a model that supports lower-dimensional embeddings.
  - Lower dimensions mean less memory per vector.
- **4. Optimize the ANN index**
  - HNSW indexes can consume substantial memory.
  - Reduce parameters such as graph connectivity where supported, while monitoring Recall@K.
  - Consider IVF/PQ or disk-based indexes for large datasets.
- **5. Remove unnecessary vectors**
  - Deduplicate documents and near-duplicate chunks.
  - Remove obsolete or irrelevant content.
  - Avoid indexing content that will never be retrieved.
- **6. Optimize chunking**
  - Extremely small chunks can create millions of unnecessary vectors.
  - Use an appropriate chunk size and overlap to reduce vector count without hurting retrieval quality.
- **7. Move cold data to disk**
  - Keep frequently accessed vectors/index portions in memory.
  - Store less frequently accessed data on disk when the database supports it.
- **8. Optimize metadata**
  - Remove unnecessary metadata fields.
  - Avoid duplicating large text payloads inside the vector database if they can be stored elsewhere.
- **9. Apply metadata partitioning**
  - Partition data by tenant, domain, time, or another appropriate dimension.
  - This can reduce the amount of index data that needs to be actively searched.

#### Practical approach

**Profile memory → Reduce duplicate/chunk count → Quantize vectors → Optimize ANN index → Use disk-based storage → Benchmark Recall@K and latency**

### 19.19 Your vector database cannot scale to millions of embeddings. How do you fix the bottleneck?

#### How do you fix a vector database that cannot scale to millions of embeddings?

First, I would **profile the bottleneck** rather than immediately scaling infrastructure. The problem could be memory, indexing, query latency, ingestion, or poor partitioning.

- **1. Use ANN indexing**
  - Avoid brute-force similarity search.
  - Use **HNSW, IVF, DiskANN**, or another appropriate ANN algorithm.
- **2. Optimize the index**
  - Tune ANN parameters for the required **Recall@K vs. latency** trade-off.
  - An oversized HNSW index, for example, can consume significant memory.
- **3. Quantize embeddings**
  - Move from FP32 to **FP16/INT8** or use **Product Quantization (PQ)**.
  - Reduces memory and storage requirements substantially.
- **4. Reduce the number of vectors**
  - Remove duplicate and obsolete documents.
  - Review chunk size and overlap.
  - Excessive chunking can unnecessarily create millions of vectors.
- **5. Reduce embedding dimensionality**
  - Use a smaller embedding model or a lower-dimensional representation if retrieval quality remains acceptable.
- **6. Shard the database**
  - Distribute vectors across multiple nodes.
  - Partition by tenant, domain, geography, or another meaningful key where appropriate.
- **7. Use metadata filtering**
  - Narrow the search space before or during vector retrieval.
  - For example, search only within the relevant tenant or document category.
- **8. Separate ingestion from query serving**
  - Build and update indexes asynchronously.
  - Prevent large ingestion jobs from degrading query performance.
- **9. Use caching**
  - Cache frequently generated query embeddings and frequently requested search results.
- **10. Consider a distributed/vector-native system**
  - If a single-node deployment is the bottleneck, move to a vector database or architecture designed for distributed ANN search.

#### Practical troubleshooting flow

**Profile → Identify bottleneck → Optimize index → Quantize → Reduce vector count → Shard → Scale horizontally → Benchmark**

Measure:

- **Recall@K**
- p95/p99 query latency
- QPS/throughput
- Index build time
- Memory usage
- Storage usage
- Cost

### 19.20 Your vector search returns irrelevant results despite high similarity scores. How do you fix it?

#### How do you fix irrelevant results despite high similarity scores?

A **high similarity score does not necessarily mean high relevance**. I would debug the entire retrieval pipeline.

- **1. Verify the embedding model**
  - Check whether the model is appropriate for the domain and language.
  - Benchmark it using real query-document relevance data.
- **2. Check the similarity metric**
  - Ensure the vector database uses the metric expected by the embedding model.
  - Check normalization if using cosine similarity or dot product.
- **3. Improve chunking**
  - Poor chunks can contain incomplete or unrelated information.
  - Tune chunk size and overlap.
  - Avoid combining multiple unrelated topics into one chunk.
- **4. Use metadata filtering**
  - Restrict retrieval by tenant, document type, date, permissions, category, etc.
  - This prevents semantically similar but contextually irrelevant documents from being retrieved.
- **5. Add hybrid search**
  - Combine **vector search + keyword search**.
  - This helps when exact terms, names, IDs, or technical terminology matter.
- **6. Increase candidate retrieval and rerank**
  - Retrieve a larger candidate set, such as top 50–100.
  - Use a **cross-encoder/reranker** to perform deeper query-document relevance scoring.
  - Return only the best final results.
- **7. Improve the query**
  - Rewrite or expand vague user queries before retrieval.
  - Query decomposition can help with complex questions.
- **8. Check for bad data**
  - Remove duplicate, outdated, noisy, or incorrectly indexed documents.
  - Verify that metadata is correctly associated with each vector.
- **9. Tune ANN parameters**
  - ANN search may sacrifice recall for speed.
  - Increase search parameters where supported and compare against exact search to determine whether ANN is causing the problem.
- **10. Evaluate systematically**
  - Build a labeled evaluation set.
  - Measure **Recall@K, Precision@K, MRR, and NDCG**.
  - Compare each pipeline change against a baseline.

#### Practical debugging flow

**Check data → Check chunking → Check embeddings → Check metric/normalization → Check ANN → Add metadata filters → Hybrid retrieval → Rerank → Evaluate**

### 19.21 You deployed a new embedding model, and search quality crashed overnight. How do you handle embedding drift?

#### How do you handle embedding drift after a deployment?

If search quality **crashed overnight**, I would treat it as a production incident first, then investigate the embedding drift.

- **1. Roll back immediately**
  - Route queries back to the previous embedding model/index if available.
  - Restore search quality while investigating.
- **2. Check model compatibility**
  - Compare:
    - Embedding dimensions
    - Similarity metric
    - Normalization
    - Model version
    - Tokenization/context limits
  - A configuration mismatch can cause severe retrieval degradation.
- **3. Verify query/document consistency**
  - Ensure queries and indexed documents are embedded using the **same model/version**.
  - A common failure is new-model queries being compared against old-model document vectors.
- **4. Compare embedding distributions**
  - Sample old and new embeddings.
  - Check vector norms, similarity distributions, and nearest-neighbor behavior.
  - Look for unexpected distribution shifts.
- **5. Re-embed the corpus**
  - If the new model is genuinely better but uses a different embedding space, create a **new index** and re-embed all documents with the new model.
- **6. Run offline evaluation**
  - Compare old vs new models using a fixed labeled dataset.
  - Measure **Recall@K, Precision@K, MRR, and NDCG**.
- **7. Test the complete RAG pipeline**
  - Check context relevance, answer correctness, and faithfulness.
  - A retrieval metric alone may not reveal the full impact.
- **8. Canary deployment**
  - Before full rollout, send a small percentage of traffic to the new index.
  - Monitor retrieval quality, latency, and user feedback.
- **9. Version everything**
  - Store embedding model version, dimension, normalization, and index configuration.
  - Make embedding changes reproducible and rollback-safe.

#### Key rule

**Never mix incompatible embedding spaces.**

If the new model produces a different representation, maintain a separate index and migrate after validation.

### 19.22 Your semantic search fails for short queries. How do you improve it?

#### How to improve semantic search for short queries

- **Use hybrid search:** Combine dense vector search with BM25/keyword search. Short queries often contain important exact terms that embeddings may miss.
- **Query expansion:** Expand short queries with related terms or context before embedding.
- **Query rewriting:** Use an LLM to transform vague queries into more descriptive search queries.
- **Use a stronger embedding model:** Choose an embedding model that performs well on short queries and your domain.
- **Tune chunking:** Avoid overly large chunks. Short queries need chunks with focused, relevant content.
- **Retrieve more candidates:** Increase `top-K` initially, then use a **reranker/cross-encoder** to select the most relevant results.
- **Use metadata:** Apply filters such as document type, category, date, language, or tenant when available.
- **Use query context:** If the user has conversation history, incorporate relevant previous context into the search query.
- **Handle abbreviations and entities:** Normalize acronyms, product names, IDs, spelling variations, and domain-specific terminology.
- **Evaluate specifically on short queries:** Create a test set containing 1–5-word queries and measure Recall@K, MRR, NDCG, and Precision@K.

### 19.23 What is InfoNCE loss, and where is it used?

**InfoNCE** (from Contrastive Predictive Coding, van den Oord et al., 2018) is the standard **contrastive loss**. Given an anchor $q$, one positive $k^+$ and $N-1$ negatives, it is a softmax cross-entropy that asks the model to pick out the positive:

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(q, k^+)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(q, k_j)/\tau)}
$$

- $\text{sim}$ is usually cosine similarity (dot product of L2-normalized embeddings).
- $\tau$ is the **temperature**. A low $\tau$ sharpens the distribution and focuses on hard negatives; a high $\tau$ makes it smoother. It is often learned (CLIP).
- Minimizing it maximizes a lower bound on the mutual information between the anchor and positive views: $I \ge \log N - \mathcal{L}$, so **more negatives give a tighter bound**.

```python
import torch, torch.nn.functional as F
def info_nce(q, k, tau=0.07):           # q, k: [B, D], row i of k is the positive for row i of q
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    logits = q @ k.T / tau               # [B, B]; off-diagonal entries are in-batch negatives
    return F.cross_entropy(logits, torch.arange(q.size(0), device=q.device))
```

**Where it's used**

- **Self-supervised vision:** SimCLR (two augmentations of one image), MoCo (momentum queue of negatives).
- **Multimodal:** CLIP/ALIGN (a symmetric InfoNCE over image-text pairs).
- **Retrieval and embeddings:** dense retrievers (DPR), sentence embeddings (SimCSE, E5, BGE), with query/positive pairs plus in-batch and hard negatives.
- **Audio/speech:** wav2vec 2.0, CPC. **RL:** contrastive goal-conditioned objectives.

**Practical issues:** quality depends on **batch size / number of negatives** (hence large batches, memory queues, cross-device gathering), **hard-negative mining** (but watch for false negatives, such as duplicates that are actually positives), and tuning $\tau$.

### 19.24 Why can't you use one embedding model for every domain?

An embedding model encodes **one particular notion of similarity**: the one implied by its training data and objective. Two texts are "close" only in the sense the model was trained to treat as close. When the domain or task changes, that notion often stops matching what users mean by relevant.

**Why a single model falls short**

- **Vocabulary and meaning shift.** In law, "consideration" means something exchanged in a contract, not thoughtfulness. In medicine, abbreviations and drug names are rare in general web text, so a general model has weak representations for them and the tokenizer splits them into meaningless pieces.
- **Different definitions of similarity.** For duplicate detection, paraphrases should be close. For question answering, a question should be close to its *answer*, which may share few words. For code search, a natural-language description should be close to code. For e-commerce, "iPhone 15 case" should not be close to "iPhone 15", even though they share words. One training objective cannot serve all of these equally.
- **Asymmetry.** Search queries are short and documents are long. Models trained only on symmetric sentence pairs handle this poorly.
- **Language coverage.** English-heavy models degrade on other languages and scripts.
- **Modalities.** Text models cannot embed images or tables well; multimodal models are needed for those.

**What to do in practice**

- Benchmark candidate models on **your own queries and documents** (recall@k, NDCG), not only on public leaderboards such as MTEB, which may not match your domain.
- Use domain-specific or instruction-tuned models where they exist, and pass the task instruction or query prefix the model expects.
- Fine-tune a general model on in-domain pairs when the gap is large (see the fine-tuning question in this chapter).
- Use hybrid search so that exact domain terms, codes and IDs are matched lexically even when embeddings are weak.
- It is fine to run **different models for different collections** (for example, one for code, one for support articles), as long as each index only ever compares vectors from its own model.

### 19.25 How do you choose between Pinecone, Weaviate, Qdrant, pgvector and other vector stores for a given client?

Start from the client's requirements and existing infrastructure, not from a product list. For most projects, several options would work; the decision is about operations, cost, scale and features such as filtering and hybrid search.

**Questions to ask first**

- **Scale:** how many vectors now and in two years (100k, 10M, 1B)? What dimension? How many queries per second?
- **Latency and recall targets:** for example, p99 under 50 ms at recall@10 of 0.95.
- **Filtering:** heavy metadata filtering, multi-tenancy, per-document permissions?
- **Hybrid search:** is keyword search needed alongside vectors?
- **Update pattern:** mostly static, or constant upserts and deletes?
- **Operations:** does the client want a managed service, or must data stay inside its own cloud account or on premises?
- **Existing stack:** do they already run Postgres, Elasticsearch or OpenSearch, or a lakehouse?
- **Budget** and team skills.

**Options compared**

| Option | Strengths | Watch out for | Good fit |
| --- | --- | --- | --- |
| pgvector (Postgres) | No new system; SQL joins and filters; transactions with application data; HNSW and IVF indexes | Large indexes need lots of RAM; tuning filtered queries; scaling beyond tens of millions of vectors needs care | Already on Postgres, up to roughly tens of millions of vectors |
| Elasticsearch / OpenSearch | Mature keyword search plus vectors in one engine; strong filtering and aggregations | Operational weight; vector performance and memory need tuning | Already using them for search; hybrid search is central |
| Pinecone | Fully managed, serverless option, little to operate | Vendor lock-in, data leaves your account, cost at high volume | Teams who want no operations and accept SaaS |
| Weaviate | Open source or managed; built-in hybrid search and modules | Another system to run if self-hosted | Hybrid search with flexible hosting |
| Qdrant | Open source or managed; fast; strong payload filtering; quantisation support | Self-hosting needs operations skills | Heavy filtering, self-hosted performance |
| Milvus / Zilliz | Built for very large scale; many index types including GPU and disk-based | More complex architecture | Hundreds of millions to billions of vectors |
| LanceDB, or vectors in a lakehouse | Vectors stored alongside data in open formats; cheap storage | Lower query concurrency than dedicated servers | Batch, analytics and offline retrieval workloads |

**How I would decide**

1. Default to the **simplest option that meets the requirements**, usually the database the client already runs (pgvector, OpenSearch).
2. Shortlist two or three alternatives only if requirements exceed what the default can do.
3. Run a **benchmark on the client's data**: recall@k against exact search, p99 latency at target QPS, filtered-query performance, index build time, memory and cost.
4. Check non-functional needs: security (encryption, private networking, RBAC), backups, multi-region, compliance.
5. Keep an abstraction layer in the application and keep the source text and metadata in the system of record, so the vector store can be rebuilt or replaced.

### 19.26 How does the HNSW index work, and how do you tune it?

**HNSW (Hierarchical Navigable Small World)** is a graph-based approximate nearest neighbour index. It is the default in most vector databases (pgvector, Qdrant, Weaviate, Milvus, OpenSearch) because it gives high recall at low latency for in-memory search.

**Structure**

- Every vector is a node in a graph, connected to some of its nearest neighbours.
- The graph has **several layers**. Every node is in layer 0. Each node is also placed in higher layers with exponentially decreasing probability, so the top layers are sparse and contain few nodes with long-range links, like an express highway. This is similar in spirit to a skip list.

**Search**

1. Start at a fixed entry point in the top layer.
2. Greedily move to the neighbour closest to the query until no neighbour is closer. Then drop down one layer and repeat.
3. In layer 0, run a **beam search** that keeps a candidate list of size `ef_search`, expanding the closest unvisited candidates until the list stops improving.
4. Return the best `k` from the candidate list.

Search cost grows roughly logarithmically with the number of vectors, which is why HNSW scales well.

**Insertion:** a new vector picks its top layer at random, then is connected at each layer to its nearest neighbours, found with the same search procedure using `ef_construction`. A heuristic prefers diverse neighbours, so the graph keeps long links that help navigation.

**Parameters**

| Parameter | What it controls | Higher value means |
| --- | --- | --- |
| `M` | Links per node (often 2M at layer 0) | Better recall, more memory, slower build. Typical 16 to 64 |
| `ef_construction` | Candidate list size during build | Better graph quality, slower build. Typical 100 to 500 |
| `ef_search` | Candidate list size at query time (must be at least k) | Better recall, higher latency. Tune per workload, can change at runtime |

**How to tune**

1. Compute exact nearest neighbours for a sample of real queries as ground truth.
2. Fix `M` and `ef_construction` at reasonable values, build the index.
3. Sweep `ef_search` and plot recall@k against p99 latency and QPS. Pick the smallest value that meets the recall target.
4. If recall plateaus too low, increase `M` or `ef_construction` and rebuild.

**Costs and limitations**

- **Memory:** the full vectors plus the links, roughly `M × 2 × 4 bytes` per vector at layer 0. For 10M vectors at 768 dimensions in float32 that is about 31 GB of vectors plus about 1 to 3 GB of graph.
- **Build time** is significant for large collections.
- **Deletes** are usually tombstones; many deletes degrade the graph until it is rebuilt or compacted.
- **Restrictive filters** can disconnect the graph during search and reduce recall (see the filtered search question).
- When memory is the limit, combine HNSW with quantised vectors, or use a disk-based index such as DiskANN.

### 19.27 How do IVF and product quantisation work, and when would you use them instead of HNSW?

**IVF (inverted file index)** partitions the vector space into clusters:

1. Run k-means on a sample of vectors to get `nlist` centroids. A common starting point is `nlist` around $\sqrt{n}$ to $4\sqrt{n}$.
2. Assign every vector to its nearest centroid's list.
3. At query time, find the `nprobe` nearest centroids and search only the vectors in those lists.

Raising `nprobe` increases recall and latency. IVF has a small memory overhead and fast builds, but recall suffers when the nearest neighbours fall just across a cluster boundary, and the centroids should be retrained if the data distribution shifts.

**Product quantisation (PQ)** compresses vectors:

1. Split each $d$-dimensional vector into $m$ sub-vectors.
2. For each sub-space, learn 256 centroids with k-means, so each sub-vector is replaced by a one-byte code.
3. A 768-dimensional float32 vector (3,072 bytes) with $m = 96$ becomes 96 bytes, a 32 times reduction.
4. At query time, precompute a small lookup table of distances from the query's sub-vectors to every centroid. The distance to any compressed vector is then a sum of $m$ table lookups (asymmetric distance computation), which is very fast.

PQ distances are approximate, so production systems **retrieve more candidates with PQ and re-rank them with full-precision vectors** (kept on disk or in cheaper storage).

**IVF-PQ** combines both: clusters to limit how many vectors are scanned, PQ codes to make each scan cheap. Variants such as OPQ (a learned rotation before PQ) and ScaNN's anisotropic quantisation improve accuracy.

**DiskANN** is another option for very large collections: a graph index stored on SSD with PQ codes in memory for navigation, fetching full vectors from disk only for final re-ranking. It serves billions of vectors with modest RAM.

**When to choose what**

| Situation | Good choice |
| --- | --- |
| Up to tens of millions of vectors, RAM available, best latency and recall | HNSW (possibly with scalar quantisation) |
| Hundreds of millions to billions of vectors, memory is the constraint | IVF-PQ with re-ranking, or DiskANN |
| Frequent full rebuilds, batch workloads, GPU available | IVF or IVF-PQ (fast to build, GPU-friendly in FAISS) |
| Small collection (under about 100k vectors) or very selective filters | Exact (flat) search, which is often fast enough and has perfect recall |

### 19.28 How do you benchmark a vector index for recall, latency and cost?

Vector search is approximate, so you must measure how much accuracy you are trading for speed. Vendor benchmarks rarely match your data, filters and load, so run your own.

**1. Build a ground truth**

- Take a representative sample of real queries (or held-out documents used as queries), typically 1,000 to 10,000.
- Run **exact (brute-force) search** over the full collection to get the true top-k for each query.

**2. Measure ANN recall**

$\text{recall@k} = \frac{|\text{ANN top-}k \cap \text{exact top-}k|}{k}$

averaged over queries. This measures the index, not the embedding model. Retrieval relevance (whether the true top-k are actually useful) is a separate evaluation with labelled relevance data.

**3. Measure performance under realistic conditions**

- p50, p95 and p99 latency at the target QPS, with realistic concurrency, not one query at a time.
- Queries **with the filters** production will use, at realistic selectivity.
- Performance **during writes** (upserts and deletes) if the index is updated continuously.
- Index build time, memory use, disk use, and cost per million queries.

**4. Sweep parameters and plot trade-off curves**

- For each index configuration (HNSW `M`, `ef_search`; IVF `nlist`, `nprobe`; quantisation settings), plot recall@k against QPS or p99 latency.
- Choose the cheapest configuration that meets the recall target, for example recall@10 of at least 0.95.

**5. Check end-to-end impact**

- Confirm on the downstream task (RAG answer quality, search NDCG) that the recall loss does not matter in practice. Recall of 0.95 versus 0.99 at the index level is often invisible after reranking.

**Tools:** ann-benchmarks and VectorDBBench for comparing engines; FAISS for quickly building exact ground truth; your own load-testing harness for production-like tests.

**Common mistakes:** benchmarking with random vectors instead of real embeddings; testing without filters; single-threaded tests; comparing systems at different recall levels; and confusing index recall with retrieval relevance.

### 19.29 What are Matryoshka embeddings, and how do they reduce cost?

**Matryoshka Representation Learning (MRL)** trains an embedding model so that **the first $k$ dimensions of the vector are themselves a good embedding**, for several values of $k$ (for example 64, 128, 256, 512, 1024). Like nested Russian dolls, smaller embeddings sit inside the larger one.

**How it is trained:** the training loss (for example, a contrastive loss) is computed on several prefixes of the same vector and summed:

$\mathcal{L} = \sum_{k \in \{64, 128, \ldots, d\}} c_k \cdot \mathcal{L}(\mathbf{z}_{1:k})$

This forces the model to put the most important information in the earliest dimensions.

**How to use it**

- **Truncate** the vector to the size you need, then **re-normalise** it to unit length before using cosine or dot-product similarity.
- Several commercial and open models support it, for example OpenAI's text-embedding-3 models (through the `dimensions` parameter) and several open models (Nomic, Jina, and many recent models in the sentence-transformers ecosystem).

**Why it matters for cost**

- Storage and memory scale linearly with dimension: going from 1,024 to 256 dimensions cuts vector memory by 4 times, often with only a small loss in retrieval quality.
- **Adaptive retrieval (funnel search):** search a fast index of short vectors (for example 128 or 256 dimensions) to get a few hundred candidates, then re-rank those candidates with the full vectors. This gives most of the full-dimension quality at a fraction of the cost.
- Combine with quantisation (int8 or binary) for further savings.

**Caveats**

- Only works for models trained with MRL. Truncating an ordinary embedding model's vectors badly damages quality.
- Always measure the quality loss at each size on your own evaluation set; the loss varies by domain.
- Every vector in an index must use the same truncation size.

### 19.30 What is late interaction (ColBERT), and how does it compare with bi-encoders and cross-encoders?

There are three common ways to score how relevant a document is to a query with transformers. They trade quality against cost.

**Bi-encoder (dense retrieval)**

- Encode the query and the document **separately**, each into one vector, and score with a dot product.
- Document vectors are computed offline, so search is cheap and works with ANN indexes.
- Weakness: compressing a whole passage into one vector loses detail, especially for long or multi-topic text and exact terms.

**Cross-encoder (reranker)**

- Feed the query and document **together** into one transformer, which outputs a relevance score. Every query token can attend to every document token.
- The most accurate option, but nothing can be precomputed: each (query, document) pair needs a full forward pass. Only practical for re-ranking a short list (tens to hundreds of candidates).

**Late interaction (ColBERT)**

- Encode the query and document separately, but keep **one vector per token** (for example, 128 dimensions each) instead of one vector per text.
- Score with **MaxSim**: for each query token, find its most similar document token, and add up those maximum similarities:

$\text{score}(q, d) = \sum_{i \in q} \max_{j \in d} \mathbf{q}_i \cdot \mathbf{d}_j$

- Document token vectors are precomputed offline, so it is much cheaper than a cross-encoder, while fine-grained token matching makes it more accurate than a single-vector bi-encoder, and it generalises better to new domains.

| Approach | Precompute documents? | Quality | Cost | Typical use |
| --- | --- | --- | --- | --- |
| Bi-encoder | Yes (one vector) | Good | Lowest | First-stage retrieval over millions of documents |
| Late interaction | Yes (vector per token) | Better | Medium; much more storage | Retrieval or re-ranking where quality matters |
| Cross-encoder | No | Best | Highest per pair | Re-ranking the top 20 to 200 candidates |

**The storage problem:** one vector per token makes the index tens of times larger than a single-vector index. ColBERTv2 compresses token vectors with centroids plus quantised residuals, and the PLAID engine speeds up search. Some vector databases now support multi-vector fields natively, and the same idea is used for visual document retrieval (ColPali), where each image patch gets a vector.

**Typical production pipeline:** bi-encoder plus BM25 retrieval for recall, then late interaction or a cross-encoder for re-ranking.

### 19.31 How does filtered vector search work, and why can filters hurt recall?

Most real queries combine similarity with constraints: tenant ID, user permissions, date range, language, product category. How the engine combines the filter with the ANN search has a big effect on both recall and latency.

**Strategies**

| Strategy | How it works | Problem |
| --- | --- | --- |
| Post-filtering | Run ANN for top-k (or top-k × a factor), then drop results that fail the filter | With a selective filter, few or no results survive, so you return fewer than k results or miss relevant ones |
| Pre-filtering with exact search | Apply the filter first, then brute-force search the matching vectors | Exact and correct; fast only when the filtered set is small |
| Filtered ANN traversal | Search the graph but only accept nodes that pass the filter | With a restrictive filter the allowed nodes are sparse, the graph becomes effectively disconnected, and the search gets stuck, so recall drops |

**How good engines handle it**

- **Query planning by selectivity:** estimate how many vectors pass the filter. If few, use exact search over them; if many, use filtered ANN. Qdrant, Weaviate, Milvus and pgvector (with iterative scans in recent versions) do variations of this.
- **Filter-aware graphs:** add extra edges so the graph stays connected within common filter values, or expand to neighbours of neighbours when filtered nodes are sparse (the ACORN approach).
- **Partitioning:** for filters that are always present and high cardinality, such as tenant ID, put each tenant in its own partition, shard or collection, so the filter becomes free.
- **Payload indexes** on filter fields so the engine can evaluate filters quickly.

**What to do in a design**

1. Identify the filters production queries use and their selectivity distribution (a 1% filter behaves very differently from a 50% filter).
2. Benchmark recall@k **with those filters**, not just unfiltered.
3. Partition by tenant or other mandatory filters where possible.
4. For security filters (permissions), never rely on post-filtering that might return nothing; and always enforce access control in the query, never only in the application after retrieval.

### 19.32 What are learned sparse embeddings such as SPLADE, and how do they compare with BM25 and dense embeddings?

**Learned sparse retrieval** uses a transformer to produce a sparse vector over the vocabulary: one weight per vocabulary term, with most weights zero. It can be served from a standard **inverted index**, like BM25, but the weights are learned.

**How SPLADE works**

- Pass the text through a BERT-style model with its masked language modelling head, giving a score for every vocabulary term at every token position.
- For each vocabulary term, take the maximum over positions and apply $\log(1 + \text{ReLU}(\cdot))$ to get a non-negative weight.
- A sparsity regulariser during training pushes most weights to zero, so each document has a few hundred non-zero terms.
- Crucially, the model can give weight to terms **that do not appear in the text** (expansion): a passage about "car insurance" may get weight on "vehicle" and "policy". This reduces vocabulary mismatch, which is BM25's main weakness.

**Comparison**

| Aspect | BM25 | Learned sparse (SPLADE) | Dense (bi-encoder) |
| --- | --- | --- | --- |
| Training needed | No | Yes | Yes |
| Synonyms and paraphrase | No | Partly, through expansion | Yes |
| Exact terms, IDs, rare words | Strong | Strong | Often weak |
| Interpretability | High (term weights) | High (term weights) | Low |
| Index | Inverted index | Inverted index | ANN index |
| Out-of-domain robustness | Good | Good | Varies, often weaker |
| Cost | Very cheap | Model inference for documents (and queries, unless an inference-free query variant is used) | Model inference for both; ANN memory |

**Where it fits:** learned sparse retrieval is a strong option for hybrid search, combined with dense retrieval through reciprocal rank fusion, and it works well in engines that already have an inverted index (Elasticsearch's ELSER, OpenSearch neural sparse search, and sparse vector support in Qdrant, Milvus and others). It is particularly helpful in domains with technical vocabulary, where dense models struggle with exact terms.

### 19.33 How do you keep a vector index fresh and consistent when source documents change frequently?

In production, documents are created, edited and deleted constantly, and permissions change. A vector index that drifts out of sync with the source of truth returns deleted, outdated or unauthorised content, which is both a quality and a security problem.

**Principles**

- The **source system is the system of record**; the vector index is a derived, rebuildable view.
- Every vector carries metadata linking it back to its source: `doc_id`, `chunk_id`, source version or `updated_at`, content hash, embedding model version, and access-control fields.

**Pipeline**

```text
source systems ─► change events (CDC, webhooks, or an outbox table)
  ─► queue (Kafka / SQS) ─► ingestion workers: parse ─► chunk ─► embed ─► upsert
  ─► vector store (idempotent upserts keyed by doc_id + chunk_id)
nightly reconciliation job: compare source IDs and versions with the index ─► fix drift
```

**Handling the hard cases**

- **Edits:** a changed document may produce a different number of chunks. Delete all chunks for that `doc_id` (or for the old version), then insert the new ones, so stale chunks do not linger.
- **Idempotency:** deterministic chunk IDs and content hashes make retries safe and let you skip re-embedding unchanged chunks, which saves cost.
- **Deletes:** propagate quickly, especially for legal deletion requests (GDPR). Remember backups and caches.
- **Permission changes:** update ACL metadata without re-embedding; enforce permissions at query time.
- **Ordering:** use version numbers so a late, older event cannot overwrite a newer one.

**Index maintenance**

- Graph indexes (HNSW) degrade with many deletes and updates, which are stored as tombstones. Schedule compaction or periodic rebuilds.
- For full rebuilds (new embedding model, new chunking strategy), build a new index in the background and switch an **alias** atomically, keeping the old index for rollback (blue/green).

**Monitoring:** ingestion lag (time from source change to searchable), failed documents, count reconciliation between source and index, and the share of retrieved chunks whose source version is stale.

### 19.34 Why do many embedding models need different prefixes or instructions for queries and documents?

Retrieval is usually **asymmetric**: a short query ("refund policy for damaged items") is matched against long passages that answer it. The query and the passage do not say the same thing; one asks, the other answers. Many embedding models are trained to handle this by being told which role each text plays.

**Common conventions**

| Model family | Convention |
| --- | --- |
| E5 | Prefix queries with `query: ` and documents with `passage: ` |
| BGE (English v1.5) | An instruction prefix on queries for retrieval; no prefix on passages |
| Nomic embed | `search_query: ` and `search_document: ` (also `clustering: `, `classification: `) |
| Instruction-tuned models (e5-mistral, gte-Qwen and similar) | A task description on the query, such as `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ...` |
| Commercial APIs | An input-type parameter, for example `search_query` versus `search_document` |

**Why it matters**

- The model learned different behaviour for each role during training. Leaving out the prefix, or using the wrong one, **silently degrades retrieval**: nothing errors, recall just drops, sometimes by several points.
- Instructions let one model serve several tasks (retrieval, clustering, classification, duplicate detection), each with its own notion of similarity.
- For **symmetric** tasks such as duplicate detection, both sides use the same prefix or instruction.

**Engineering practice**

- Read the model card and encode the convention in **one shared embedding function** used by both ingestion and query code, so they cannot drift apart.
- Store the prefix or instruction as part of the embedding model version metadata; changing it requires re-embedding, just like changing the model.
- Include the prefix when evaluating candidate models, otherwise the comparison is unfair.

### 19.35 Why should you not use a fixed similarity score threshold to decide whether a result is relevant?

It is tempting to say "only return chunks with cosine similarity above 0.8". This usually fails, because embedding similarity scores are **not calibrated probabilities of relevance**.

**Reasons**

- **Each model has its own score distribution.** Many models are anisotropic: their embeddings occupy a narrow cone of the space, so even unrelated texts can score 0.6 to 0.8. Another model may score related texts at 0.4. A threshold tuned for one model is meaningless for another, including a new version of the same model.
- **Scores vary by query.** Short or generic queries tend to get high similarity with many documents; long, specific queries get lower scores overall. One global threshold will be too strict for some queries and too loose for others.
- **Similarity is not relevance.** A passage can be on the same topic without answering the question, for example a document about a product's older version.
- **Hubness:** in high-dimensional spaces, some vectors are close to many others and appear in many result lists regardless of the query.

**Better approaches**

- **Use a reranker score for thresholds.** Cross-encoder rerankers produce scores that track relevance much better; calibrate a threshold on labelled data.
- **Calibrate per model with labelled data:** plot precision against threshold on (query, document, relevant?) examples, choose the threshold for the precision you need, and recalibrate whenever the model changes.
- **Relative rules:** keep results within a margin of the top score, or look for a large gap in the score curve.
- **Let the generator decide:** in RAG, pass the top-k and instruct the model to answer only from relevant context and to say when the context is insufficient; evaluate this behaviour explicitly.
- **Monitor the distribution of top scores** over time; a shift often signals a data or model change.

### 19.36 Are embeddings sensitive data, and how do you secure a vector database?

Yes. Embeddings are often treated as anonymous numbers, but they should be protected **as carefully as the text they were created from**.

**Why embeddings are sensitive**

- **Inversion attacks:** research such as Vec2Text has shown that text can be reconstructed from embeddings with high accuracy for short passages, given access to the embedding model. Leaked embeddings can therefore leak the content, including names and other personal data.
- **Attribute and membership inference:** embeddings can reveal properties of the text (topic, sentiment, demographic signals) or confirm whether a particular record is in the collection.
- Vector stores usually also hold the **original chunk text and metadata**, so a breach exposes the source content directly.

**Controls**

- **Access control enforced at query time:** store ACL metadata (user, group, tenant) with each chunk and filter in the query, so retrieval only ever returns content the requester can see. Do not rely on the LLM or on application code after retrieval.
- **Tenant isolation:** separate collections, namespaces or databases per tenant for strong isolation where required.
- **Encryption** at rest and in transit, private networking, secrets management, and audit logs of queries.
- **Data minimisation:** redact or mask personal data before embedding when it is not needed for retrieval.
- **Deletion:** support deletion of every vector and chunk derived from a document (GDPR right to erasure), including replicas and backups.
- **Third-party embedding APIs:** check data retention and training terms, region, and contractual protections; self-host the embedding model for highly sensitive data.
- **Index poisoning and prompt injection:** documents added to the index can contain malicious instructions or misleading content aimed at the RAG system. Control who can add content, scan ingested content, and treat retrieved text as untrusted data in prompts.

### 19.37 How would you estimate the size and cost of a vector database deployment?

Capacity planning starts from four numbers: **number of vectors, dimensions, bytes per dimension, and query load**. Work through the estimate explicitly; interviewers look for the arithmetic and the levers.

**Worked example:** 20M documents, chunked into an average of 2.5 chunks each, so 50M vectors at 1,024 dimensions.

**1. Raw vectors**

- float32: 50M × 1,024 × 4 bytes ≈ **205 GB**
- int8 scalar quantisation: about **51 GB**
- binary quantisation: 50M × 1,024 bits ≈ **6.4 GB**
- Matryoshka truncation to 256 dimensions in float32: about **51 GB**

**2. Index overhead**

- HNSW with `M = 16` stores about 32 links at layer 0, roughly 32 × 4 bytes = 128 bytes per vector, so about **6.4 GB**, plus upper layers.

**3. Metadata and text**

- If chunk text (about 1.5 KB) and metadata are stored with the vectors, that adds about 75 GB, which can live on disk rather than in RAM.

**4. Replication and headroom**

- Two or three replicas for availability and read throughput, plus 30% to 50% headroom for growth and rebuilds.
- float32 HNSW in RAM: (205 + 6.4) × 2 replicas × 1.4 ≈ **590 GB of RAM**. With int8 vectors in RAM and full vectors on disk for re-scoring: (51 + 6.4) × 2 × 1.4 ≈ **160 GB**.

**5. Query load**

- Benchmark the queries per second one node sustains at the target recall and latency, then divide peak QPS by it and add headroom.

**6. Embedding cost**

- Initial ingestion: 50M chunks × about 300 tokens = 15B tokens. Multiply by the embedding API price per million tokens, or estimate GPU hours for a self-hosted model.
- Ongoing cost: daily changed documents plus query embeddings.
- **Re-embedding the whole corpus** for a model change costs the same as initial ingestion; budget for it.

**Levers to cut cost:** quantisation with re-scoring, Matryoshka truncation, disk-based indexes (DiskANN-style), tiered storage for rarely queried data, smaller chunks only where needed, and deduplicating near-identical chunks before embedding.

---

← [Chapter 18. Prompt and Context Engineering](18-prompt-and-context-engineering.md) · [Table of contents](../README.md) · [Chapter 20. Retrieval-Augmented Generation](20-retrieval-augmented-generation.md) →
