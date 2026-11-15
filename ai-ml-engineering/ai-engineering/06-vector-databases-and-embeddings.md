<!-- notion-id: 3d567418285480f18f61d0666de4f4d7 -->
# Vector Databases and Embeddings

Interview questions on embeddings and vector databases: similarity metrics, ANN indexing, model selection, drift, multi-tenancy, scaling and troubleshooting.

### Q1. What are embeddings in the context of AI engineering?

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

### Q2. How do embedding models convert text to vectors?

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

### Q3. What is the difference between sparse and dense embeddings?

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

### Q4. Explain cosine similarity, dot product, and Euclidean distance for vector search.

#### Cosine Similarity, Dot Product, and Euclidean Distance

These metrics measure how similar or different two vectors are in **vector search**, such as semantic search and RAG.

#### 1. Cosine Similarity

- Measures the **angle/direction** between two vectors.
- Focuses on whether vectors point in a similar direction, not their magnitude.
- Range: **1 to 1**.
  - `1` → same direction
  - `0` → orthogonal/unrelated
  - `1` → opposite direction
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

### Q5. What is a vector database, and how does it differ from a traditional database?

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

### Q6. How does Approximate Nearest Neighbor (ANN) search work?

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

### Q7. How do you choose the right embedding model for your use case?

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

### Q8. What is embedding dimensionality, and how does it affect performance and cost?

#### What is Embedding Dimensionality?

- **Embedding dimensionality** is the number of numerical values in an embedding vector.
- For example:
  - 384-dimensional → `[x₁, x₂, ..., x₃₈₄]`
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

### Q9. How do you handle embedding drift when the embedding model is updated?

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

### Q10. What are multi-modal embeddings, and how are they generated?

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

### Q11. How do you index and query multi-tenant data in a vector database?

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

### Q12. What is quantization of embeddings, and how does it reduce storage costs?

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

### Q13. How do you benchmark and evaluate embedding model quality?

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

### Q14. What is the role of metadata in vector databases?

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

### Q15. How do you handle large-scale vector search with billions of vectors?

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

### Q16. What is hybrid search (combining keyword search with vector search)?

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

`Final Score = α × Keyword Score + (1 − α) × Vector Score`

The weight `α` controls how much importance is given to keyword versus semantic retrieval.

Another common approach is **Reciprocal Rank Fusion (RRF)**, which combines rankings from the two retrieval systems rather than directly combining their raw scores.

#### Hybrid Search in RAG

A typical RAG pipeline is:

**User Query → Keyword Search + Vector Search → Combine/Rerank → Top-K Chunks → LLM**

This is particularly useful for enterprise RAG where documents contain **exact technical terms, IDs, names, and domain-specific terminology**.

### Q17. How do you fine-tune an embedding model for a specific domain?

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

### Q18. Your vector database for RAG is consuming too much memory. How do you reduce it?

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

### Q19. Your vector database cannot scale to millions of embeddings. How do you fix the bottleneck?

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

### Q20. Your new embedding model has different dimensions from the existing vectors in production. How do you handle the mismatch?

#### How do you handle an embedding dimension mismatch?

- **Do not mix the old and new vectors in the same index.**
  - If the existing vectors are 768-dimensional and the new model produces 1536-dimensional vectors, most vector indexes cannot compare them directly.
- **Create a new index/collection**
  - Configure it for the new embedding dimension.
  - Keep the existing production index unchanged during migration.
- **Re-embed the existing data**
  - Run the new embedding model over the existing documents/chunks.
  - Store the new vectors in the new index.
- **Dual-run both systems**
  - Send the same evaluation queries to both indexes.
  - Compare **Recall@K, MRR, NDCG, latency, and downstream RAG quality**.
- **Migrate gradually**
  - Once the new model is validated, route traffic to the new index.
  - Keep the old index available temporarily for rollback.
- **Update the ingestion pipeline**
  - Ensure all newly created documents use the new embedding model.
  - Store the model version and dimension with the indexing configuration.

#### What not to do

- **Don't simply truncate or pad vectors** to force them to the same dimension.
- This generally destroys the embedding model's intended representation and can significantly hurt retrieval quality.

#### Practical migration

**Old Model (768D) → Old Index**

**New Model (1536D) → Re-embed corpus → New Index → Evaluate → Gradual Cutover → Retire Old Index**

### Q21. Your vector search returns irrelevant results despite high similarity scores. How do you fix it?

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

### Q22. You deployed a new embedding model, and search quality crashed overnight. How do you handle embedding drift?

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

### Q23. Your semantic search fails for short queries. How do you improve it?

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
