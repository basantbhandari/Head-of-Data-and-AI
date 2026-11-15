# System Design and MLOps

Classic ML system design cases plus modelling, deployment, monitoring and MLOps questions. Every answer here is a drafted answer.

> The first 13 questions of this set (from "Design ChatGPT: Training to Serving (End-to-End)" to "Design an AI Coding Agent.") repeat Q1–Q13 of [AI System Design](01-ai-system-design.md), and "Design a Machine Learning System for Multimodal Search." repeats Q7 there. They are answered in that file.

## ML system design cases

### Q1. Design a Machine Learning System for YouTube Video Recommendation.
> ✍️ _Drafted answer (not from the original notes)._

**Clarify:** home-feed recommendations for about 2B users and 1B+ videos. Latency under 200 ms. The goal is long-term satisfaction, not just clicks.

**Framing:** a multi-stage retrieval-and-ranking problem. Optimise *expected watch time × satisfaction*. Labels come from implicit feedback: a click, watch fraction, likes and shares, and negatives such as "not interested", skips and dislikes. Raw CTR rewards clickbait, so use a multi-task objective.

**Features:** user (watch history embeddings, search history, demographics, device, time of day), video (topic, language, freshness, creator, engagement stats, content embeddings from frames, audio and title), and context (session sequence, surface).

**Model:**
```text
1B videos ──► Candidate generation (several sources, ~1000s)
               • two-tower retrieval (user tower vs. video tower, ANN via ScaNN/FAISS)
               • co-watch / item-item, subscriptions, trending, fresh uploads
          ──► Light ranker (~500, small DNN/GBDT)
          ──► Heavy ranker (~50, multi-task DNN: P(click), E[watch time], P(like), P(dismiss))
          ──► Re-ranking: diversity, freshness, policy filters, dedupe, creator caps
```
The final score is a weighted combination of the task heads. The weights are tuned through A/B tests.

**Metrics:** offline recall@k for retrieval, plus NDCG/AUC and watch-time-weighted log loss for ranking. Online: watch time per DAU, satisfied sessions, retention (D7), surveyed satisfaction, and dismiss rate.

**Serving:** a feature store gives the same features online and offline. Video embeddings are precomputed and indexed in ANN. The user tower is computed on request. Cache results for heavy users.

**Monitoring/retraining:** retrain daily or continuously, because new videos and new trends appear all the time. Watch feature drift and calibration.

**Pitfalls:** cold start for new videos (use content embeddings and exploration traffic) and new users (popular-by-region, onboarding). Position bias (add position as a training feature and set it to a constant at serving). Feedback loops and filter bubbles (exploration, diversity). Clickbait (weight watch time and survey signals).

### Q2. Design a Machine Learning System for YouTube Video Search.
> ✍️ _Drafted answer (not from the original notes)._

**Clarify:** text queries over 1B+ videos, multilingual, p99 latency under 300 ms. Results should be relevant first, then personalised.

**Framing:** retrieve, then rank. Relevance labels come from human raters (graded 0–4) plus implicit signals such as clicks and long watches. Click-through on search is position-biased, so debias it with inverse propensity weighting or click models.

**Data & features:** query (tokens, language, intent class, spelling-corrected form), video text (title, description, tags, ASR transcript, OCR from frames), visual and audio embeddings, quality (engagement, freshness, creator authority), and query-video interaction features (BM25, embedding similarity, historical CTR for the pair).

**Architecture:**
```text
query ─► spell-correct / normalize / intent ─► 
   ├─ lexical retrieval (inverted index, BM25 on title/desc/transcript)
   └─ semantic retrieval (dual encoder: query tower vs. video tower; ANN)
   ─► merge (~1000) ─► LTR ranker (GBDT or cross-encoder on top-100) ─► personalization + policy filters ─► results
```
Video embeddings are built offline when a video is uploaded (the text encoder uses the transcript; vision uses CLIP-style frame embeddings), so text queries can match visual content.

**Metrics:** offline NDCG@10 and MRR on rater sets, and recall@1000 for retrieval. Online: CTR at top positions, long-click rate, query reformulation rate (lower is better), and abandonment.

**Serving:** sharded inverted index plus ANN index. The cross-encoder runs only on the top-N. Cache popular head queries.

**Pitfalls:** tail queries have little click data, so rely on semantic retrieval. Freshness matters for news and trending queries, so add a freshness feature and a separate index for recent uploads. Spam titles and keyword stuffing call for quality and authority signals. Language mismatch can be handled with a multilingual encoder.

### Q3. Design a Machine Learning System for Personalized Content Feed.
> ✍️ _Drafted answer (not from the original notes)._

**Clarify:** a social feed (posts from friends, pages and groups, plus unconnected recommended content). Hundreds of millions of DAU. The feed refreshes on every app open.

**Framing:** rank the candidate posts by a value model: `score = Σ wᵢ · P(actionᵢ)`, where the actions are like, comment, share, dwell over N seconds, hide and report (with negative weights). This is multi-task learning with one head per action. Integrity classifiers act as filters or demotions.

**Features:** viewer-author affinity (past interactions, closeness), post features (type, age, media, engagement velocity, text and image embeddings), viewer features (interests, activity level), and context (time, device, network). Sequence features from recent sessions capture short-term interest.

**Architecture:**
```text
Inventory (friends' posts + recommended pool via two-tower retrieval)
  ─► filter (seen, blocked, integrity)
  ─► first-pass ranker (lightweight)
  ─► main ranker (multi-task DNN / DLRM-style with embedding tables)
  ─► final pass: diversity (no 3 videos in a row), author caps, ads insertion, freshness
```

**Metrics:** offline per-head AUC, normalised entropy and calibration. Online: meaningful interactions, time spent, DAU/retention, and hide/report rates. Use guardrails on integrity metrics and creator ecosystem health.

**Serving & freshness:** features come from a real-time stream (Kafka/Flink) into an online feature store. Train incrementally every few hours so trending content gets picked up.

**Pitfalls:** engagement-bait and polarising content (add negative-feedback weights and integrity models). Position bias. Feedback loops that narrow interests (exploration slots). Delayed labels for comments and shares (use a label-attribution window). Cold-start authors (boost via exploration).

### Q4. Design a Machine Learning System for Harmful Content Detection.
> ✍️ _Drafted answer (not from the original notes)._

**Clarify:** posts with text, images and video. Harm classes include violence, hate, nudity, self-harm, spam and misinformation. Some classes need proactive detection before anyone sees the post. Billions of posts a day. Actions are remove, demote, or send to human review.

**Framing:** multi-label classification, with a calibrated probability per harm class. Policies set the thresholds: high precision for automatic removal, and a lower threshold to route posts to human review. Labels come from human moderators, user reports (noisy) and appeals outcomes.

**Data & features:** content embeddings (a text transformer, an image or video encoder, an audio/ASR model, OCR for text in images), author signals (account age, prior violations), and engagement and report velocity. Fuse the modalities. Late fusion is robust when a modality is missing; early or intermediate fusion captures cross-modal meaning, such as a harmless image paired with a hateful caption.

**Architecture:**
```text
upload ─► hash matching (PhotoDNA/perceptual hashes for known bad content)
       ─► fast multimodal classifier (sync, blocks obvious violations)
       ─► async heavier models + re-scoring as reports/engagement arrive
       ─► decision engine (thresholds per class/region) ─► remove | demote | human review queue (prioritized by severity × reach)
```
Reviewer decisions flow back as training labels (active learning).

**Metrics:** per-class precision and recall, PR-AUC, and *prevalence* (the share of views that land on violating content, estimated by sampling). Also track appeal overturn rate and review-queue latency.

**Pitfalls:** heavy class imbalance (stratified sampling, focal loss). Adversarial evolution: text obfuscation, image perturbation and new slang call for frequent retraining and red-teaming. Policy drift when the definitions change. Fairness across dialects and languages (measure per-group false-positive rates). Moderator wellbeing (blur content, show reviewers only what is needed).

### Q5. Design a Machine Learning System for Similar Listings on Airbnb.
> ✍️ _Drafted answer (not from the original notes)._

**Clarify:** a "similar listings" carousel on a listing page. The goal is to raise bookings. Millions of listings, and the results must respect the viewer's dates and destination.

**Framing:** item-to-item retrieval and ranking. The core idea (from Airbnb's KDD 2018 paper) is to learn **listing embeddings** from user click sessions, treating each session as a "sentence" of listing IDs and training skip-gram (word2vec) on them. Listings clicked in the same session end up close together. The booked listing is added as a *global context* to every session that ended in a booking, which pushes embeddings toward conversion. Negatives are sampled from the same market, so the model learns within-market distinctions rather than only geography.

**Features for ranking:** embedding similarity, price difference, distance, room type, capacity, rating, availability for the dates, and host quality.

**Architecture:**
```text
offline: sessions ─► skip-gram training ─► listing embeddings ─► ANN index per market
online:  anchor listing ─► ANN top-k (same market, available for dates) ─► GBDT ranker ─► carousel
```

**Cold start:** a new listing with no sessions gets the average embedding of the 3 nearest listings by location, type and price, or a content-based embedding from photos and text.

**Metrics:** offline, the rank of the eventually booked listing among the candidates (hit rate@k). Online: carousel CTR, booking conversion, and revenue per search.

**Pitfalls:** availability changes daily, so filter at request time. Position bias in the carousel. Embedding staleness (retrain daily). Fairness to new hosts (an exposure boost).

### Q6. Design a Machine Learning System for Replacement Product Recommendation.
> ✍️ _Drafted answer (not from the original notes)._

**Clarify:** when an item is out of stock (e-commerce or grocery substitution), suggest a *substitute* that the customer will accept. Substitutes are not the same as complements: a different-brand milk is a substitute, cereal is a complement.

**Framing:** learn a substitutability score `s(original, candidate, user)`. Labels: accepted versus rejected substitutions (from grocery pickers or the customer approval flow), "viewed A then bought B instead" sessions (a substitute signal), and "bought together" pairs, which act as negatives for substitution because they are complements.

**Features:** product taxonomy (same leaf category), attributes (brand, size, unit price, dietary flags such as organic, gluten-free or vegan), text and image embeddings, price-per-unit ratio, user preferences (brand loyalty, dietary history), and stock at the store.

**Model:**
```text
candidate gen: same/adjacent category + embedding ANN (text/image) + co-view-not-co-buy graph
ranker: GBDT or pairwise DNN on (original, candidate, user) features
hard constraints: dietary/allergen rules, size bounds, price ceiling
```

**Metrics:** offline AUC and acceptance-rate prediction. Online: substitution acceptance rate, refund and return rate, order completion, and customer satisfaction.

**Pitfalls:** violating hard constraints such as allergens or halal (always use rules, never rely only on the model). Price jumps upset customers. Sparse labels for long-tail items (back off to category-level models). Store-level inventory must be checked in real time.

### Q7. Design a Machine Learning System for Event Recommendation.
> ✍️ _Drafted answer (not from the original notes)._

**Clarify:** an Eventbrite-like platform that recommends upcoming events. Events are short-lived (no long engagement history), location-bound and time-bound. The goal is ticket registrations.

**Framing:** learning-to-rank on (user, event) pairs. Labels: registration (positive), impression with no action (negative), plus intermediate signals such as clicks and saves. Because every event is essentially cold start, the model must rely on **content and context features rather than item IDs**.

**Features:**
- *Location:* distance from the user, walk score and travel-time buckets, same city.
- *Time:* time until the event, day of week, overlap with the user's usual free times.
- *Social:* friends registered, friends who are hosts, invitations.
- *User-event similarity:* category match with history, text embedding of the description versus user interest embedding.
- *Event quality and price:* host's past events, price relative to the user's typical spend.

**Model:** start with GBDT (strong on tabular features, easy to interpret), then move to a DNN with embeddings. Candidate generation filters by geography, date window and availability.

**Metrics:** offline mAP and NDCG@k. Online: registration rate per impression, revenue, attendance, and organiser-side metrics.

**Serving:** batch candidate generation per city every few hours. Online ranking computes the real-time features (friends registered, remaining capacity).

**Pitfalls:** cold start for new users (location plus popular events, onboarding interests). Class imbalance. Features such as "friends registered" change quickly, so the online and offline features need point-in-time correctness to avoid leakage.

### Q8. Design a Machine Learning System for Ad Click Prediction.
> ✍️ _Drafted answer (not from the original notes)._

**Clarify:** predict pCTR for ads shown in feed or search. Billions of impressions a day, latency under 10 ms per batch of candidates. pCTR feeds the auction (`bid × pCTR × quality`), so **calibration matters as much as ranking**.

**Framing:** binary classification on impressions. Labels: click within a time window. A post-click conversion model (pCVR) is usually trained separately.

**Features:** very high-cardinality categoricals (user ID, ad ID, advertiser, campaign, publisher, keyword), user demographics and interests, context (placement, device, time), and historical CTRs at several granularities. Feature crosses matter (user segment × ad category).

**Model evolution:**
```text
Logistic regression + hashed crosses (baseline, very fast)
 → GBDT leaves as features + LR (Facebook 2014)
 → Factorization Machines / FFM
 → Deep models: Wide & Deep, DeepFM, DCN-v2, DLRM (embedding tables + interaction layers)
```

**Training:** continuous or online learning, because ads and users change hourly. **Negative downsampling** of non-clicks cuts cost. After downsampling, recalibrate with `p = p' / (p' + (1−p')/w)`, where `w` is the sampling rate.

**Metrics:** offline normalised cross-entropy (log loss relative to a baseline), AUC, and calibration (predicted versus observed CTR per bucket). Online: revenue, CTR, advertiser ROI, and user experience guardrails such as ad hide rate.

**Serving:** embedding tables (often hundreds of GB) live in a parameter server or sharded GPU memory. Feature logging at serving time, joined with delayed click labels, avoids training/serving skew.

**Pitfalls:** delayed feedback and label leakage, position bias (use a position feature or a separate position model), cold-start ads (content features and an exploration budget), privacy constraints (fewer cross-site signals, aggregated APIs).

### Q9. Design a Machine Learning System to Estimate Delivery Time.
> ✍️ _Drafted answer (not from the original notes)._

**Clarify:** food-delivery ETA shown before ordering and updated during delivery. The business cost is asymmetric: late is worse than early. Strict latency limits.

**Framing:** regression on the total time. It is often better to decompose it into **food preparation time + courier assignment/pickup wait + travel time**, each with its own model. Labels come from the timestamps of historical orders. For an asymmetric cost, use quantile or pinball loss (for example P50 for display and P90 for promises) or Huber loss for robustness to outliers.

**Features:**
- *Restaurant:* historical prep time by hour, current order queue, cuisine, order size.
- *Courier and supply:* available couriers nearby, current utilisation.
- *Route:* distance, routing-engine travel time, traffic, weather.
- *Temporal:* hour, day of week, holidays, live demand surge.
Real-time features come from streaming aggregates.

**Model:** GBDT (LightGBM/XGBoost) is a strong baseline for tabular data. Graph or sequence models over road segments help the travel component. Re-predict at each state change (order accepted, food ready, courier picked up).

**Metrics:** MAE and RMSE, the percentage of orders within ±5 minutes, and the on-time rate against the promise. Online: conversion (a lower ETA raises orders), the late-order rate, customer support tickets, and cancellations.

**Serving:**
```text
order request ─► feature fetch (online store: restaurant load, courier supply) ─► model ─► ETA (+ buffer by uncertainty)
stream events (prep ready, GPS) ─► re-estimate ─► push update to app
```

**Pitfalls:** label noise (couriers mark "picked up" late). Distribution shift during storms and holidays. Feedback loops (shown ETAs change demand). Leakage from features that are only known after the order.

### Q10. Design a Machine Learning System for Image Search.
> ✍️ _Drafted answer (not from the original notes)._

**Clarify:** query by image (visual search, as in Pinterest Lens) and/or by text. Billions of images, results within 200 ms.

**Framing:** representation learning plus nearest-neighbour retrieval. Train an embedding model so that similar images, and matching text-image pairs, are close together. Labels: user engagement pairs (query image → clicked or saved image), plus self-supervised augmentation pairs.

**Model:** a CNN or ViT encoder trained with a **contrastive loss** (SimCLR-style for image-image; CLIP-style InfoNCE for text-image) or a triplet loss with hard-negative mining. Fine-tune a pretrained CLIP on domain engagement data.

**Architecture:**
```text
offline: images ─► encoder ─► embeddings ─► ANN index (FAISS IVF-PQ / HNSW), sharded
online:  query image ─► (object detection/crop) ─► encoder ─► ANN top-k ─► re-rank (engagement, freshness, dedupe, safety) ─► results
```
Product quantisation compresses embeddings: 512-dimensional float32 is 2 KB, and PQ brings it to about 64 bytes.

**Metrics:** offline recall@k, mAP and MRR on labelled query sets. Online: CTR, saves and repins, and time to first click.

**Pitfalls:** near-duplicates crowd the results (dedupe with perceptual hashing). Unsafe content (safety classifiers). Re-embedding billions of images after a model update (versioned indexes, backfill jobs, dual-serving during migration). Background clutter (crop detected objects first).

### Q11. Design a Machine Learning System for Friends Recommendation.
> ✍️ _Drafted answer (not from the original notes)._

**Clarify:** "People You May Know" on a social network with about 1B users. The goal is accepted connections that lead to engagement, not just sent requests.

**Framing:** link prediction on the social graph. Score `P(request sent ∧ accepted | viewer, candidate)`. Labels: past recommendations shown, then sent and accepted. Negatives are recommendations that were shown and ignored.

**Candidate generation:** the main source is **friends-of-friends (2-hop)**, which covers most accepted links. Add same school, employer or location, contact-book uploads, and graph-embedding neighbours (node2vec or GNN).

**Ranking features:** mutual friend count and their closeness, Adamic-Adar and Jaccard scores, shared groups and schools, geographic distance, profile similarity, candidate's acceptance rate, and recency of the graph changes. A GBDT ranker or GNN (GraphSAGE / PinSage-style) produces the score.

**Architecture:**
```text
nightly batch: graph processing (Spark/GraphX) ─► 2-hop candidates + features per user (precomputed top-1000)
online: fetch precomputed candidates ─► refresh real-time features ─► rank ─► filter (blocked, already requested) ─► display
```
2-hop expansion for heavy users is very expensive, so cap the fan-out and sample.

**Metrics:** offline AUC and recall@k of future connections. Online: accepted requests per user, downstream engagement between new friends, and the rate of rejected or ignored requests.

**Pitfalls:** privacy (do not surface sensitive links such as therapists or exes from contact uploads). Spam and fake accounts. Cold-start users (contacts, onboarding). Over-recommending popular users.

### Q12. Design a Machine Learning Product Recommendation System for an e-commerce platform.
> ✍️ _Drafted answer (not from the original notes)._

**Clarify:** surfaces are the homepage, product page ("similar"/"bought together") and cart. Catalogue of 10M+ SKUs, 100M users. The goals are GMV and conversion, with guardrails on returns.

**Framing:** retrieval plus ranking. Labels: purchase (strong), add-to-cart, click and view, weighted by value. Multi-task heads for P(click), P(add to cart) and P(purchase), because purchases are sparse.

**Candidate generation:**
- Collaborative: matrix factorisation / ALS, or two-tower on user-item interactions.
- Item-item co-view and co-purchase graphs (complements for the cart page).
- Session-based sequence models (GRU4Rec, SASRec) for short-term intent.
- Content-based retrieval for new items (text and image embeddings).

**Ranking:** a DNN or GBDT with user features (history embeddings, price affinity, brand affinity), item features (category, price, rating, stock, margin, delivery time), and context (surface, device, season, promotions).

**Re-ranking and business rules:** in-stock only, diversity across categories, sponsored slots, margin-aware blending, and suppressing items already bought (unless consumables).

**Metrics:** offline recall@k, NDCG and AUC on time-split data (never random splits, which leak the future). Online: CTR, conversion rate, GMV/revenue per session, return rate, and catalogue coverage.

**Serving:** precompute candidates for active users in batch. Do real-time session updates with streaming features. The ranker runs at request time with a feature store.

**Pitfalls:** popularity bias and long-tail coverage, cold-start items and users, seasonality, and the feedback loop with promotions.

### Q13. How would you build a system to detect fraudulent transactions?
> ✍️ _Drafted answer (not from the original notes)._

**Clarify:** card or payment transactions scored in real time (under 50 ms) before authorisation. Decisions: approve, decline, or step-up (3DS/OTP). Fraud base rate is about 0.1%. The costs are asymmetric: fraud loss versus declining a good customer.

**Framing:** binary classification with a cost-sensitive threshold. Labels come from chargebacks and confirmed fraud reports, and they arrive **with a delay of weeks**. Recent data therefore has incomplete labels; account for it with label maturity windows.

**Features (most of the signal):**
- *Velocity aggregates:* transactions per card in the last 1 min, 1 h and 24 h, amount sums, distinct merchants and countries.
- *Deviation from the user's profile:* amount z-score, new device, new geo, impossible travel.
- *Merchant and device risk:* merchant fraud rate, device fingerprint, IP reputation.
- *Graph features:* shared devices, emails and cards across accounts, which expose fraud rings.
These need a streaming feature pipeline (Kafka → Flink → online store) with point-in-time-correct offline backfill.

**Model:** GBDT (XGBoost/LightGBM) is the industry workhorse. Add unsupervised anomaly detection (isolation forest, autoencoders) for new patterns, GNNs for rings, and rules for known patterns and regulatory needs.

**Architecture:**
```text
txn ─► feature service (real-time aggregates) ─► rules engine + ML score ─► decision policy (thresholds per segment)
     ─► approve / step-up / decline ─► case management (analyst review) ─► labels back to training
```

**Metrics:** PR-AUC (not accuracy), recall at a fixed false-positive rate, dollar-weighted fraud caught, and the decline rate of good customers.

**Pitfalls:** class imbalance (class weights, careful undersampling). Adversarial drift as fraudsters adapt (frequent retraining, drift monitoring). Selection bias, because declined transactions never get labels (keep a small random holdout that is approved for counterfactual data). Explainability needs for disputes and regulators (reason codes via SHAP).

### Q14. Multimodal Fusion Techniques in Machine Learning: Early Fusion vs Late Fusion

> 🔁 _Already answered in [10-multimodal-ai.md → Q20](../../ai-ml-engineering/ai-engineering/10-multimodal-ai.md#q20-explain-multimodal-fusion-techniques-early-fusion-vs-late-fusion); skipped here to avoid duplication._

### Q15. How would you approach a time series forecasting problem?
> ✍️ _Drafted answer (not from the original notes)._

1. **Clarify the decision:** the forecast horizon (next hour or next quarter), granularity, number of series (1 or 100k SKUs), whether a point forecast or prediction intervals are needed, and the cost of over- versus under-forecasting.
2. **EDA:** trend, seasonality (daily, weekly, yearly), holidays and events, structural breaks such as COVID, missing data, outliers, and stationarity (ADF test). Decompose with STL.
3. **Baselines first:** naive (last value), seasonal naive, and moving average. Many fancy models fail to beat seasonal naive.
4. **Model families:**
   - *Statistical:* ETS and ARIMA/SARIMA, which are strong for a few well-behaved series. Prophet when there are many holidays and changepoints.
   - *ML on lag features:* GBDT with lags, rolling stats, calendar and promotion features. A single **global model** across many series often wins (M5 competition).
   - *Deep:* DeepAR, N-BEATS/N-HiTS, Temporal Fusion Transformer, and foundation models (TimesFM, Chronos) for zero-shot.
5. **Validation:** **time-based backtesting** (rolling-origin cross-validation), never random splits. Features must only use information available at forecast time.
6. **Metrics:** MAE, RMSE, MAPE (breaks near zero), sMAPE/MASE, WAPE for retail, and pinball loss for quantiles.
7. **Production:** retrain on a schedule, monitor forecast error and bias by segment, reconcile hierarchical forecasts (store → region → total), and let planners override with tracked adjustments.

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive
sf = StatsForecast(models=[AutoETS(season_length=7), SeasonalNaive(7)], freq="D")
cv = sf.cross_validation(df=df, h=14, n_windows=4)   # rolling-origin backtest
```

### Q16. How would you build a spam detection system?

> 🔁 _Already answered in [commonly-asked-ml-questions.md → Q119](../../ai-ml-engineering/commonly-asked-ml-questions.md#q119-how-do-you-build-a-spam-detection-system); skipped here to avoid duplication._

### Q17. Describe how you would implement an image classification system.
> ✍️ _Drafted answer (not from the original notes)._

1. **Define the problem:** the classes, single- or multi-label, accuracy versus latency targets, deployment target (cloud GPU or mobile), and data volume.
2. **Data:** collect and label (labelling vendors, active learning to prioritise uncertain samples), write clear labelling guidelines, audit inter-annotator agreement, and split by source or time to avoid leakage (near-duplicates across splits inflate metrics).
3. **Model: use transfer learning.** Start from a pretrained backbone (ResNet-50, EfficientNet, ConvNeXt, ViT). With little data, freeze the backbone and train the head, then unfreeze and fine-tune with a lower learning rate. With very little data, use zero-shot CLIP or a linear probe on CLIP embeddings.
4. **Training tricks:** augmentation (random resized crop, flips, colour jitter, MixUp/CutMix, RandAugment), label smoothing, a cosine LR schedule with warmup, mixed precision, early stopping. For class imbalance, use weighted loss, oversampling or focal loss.
5. **Evaluation:** per-class precision, recall and F1, the confusion matrix, top-k accuracy, and calibration (temperature scaling). Do error analysis on misclassified images to find label noise or data gaps.
6. **Deployment:** export to ONNX or TorchScript, then optimise (TensorRT, quantisation; MobileNet with TFLite/Core ML for mobile). Batch requests on a GPU server (Triton). Preprocessing must match training exactly.
7. **Monitoring:** input drift (brightness, resolution, new camera types), confidence distribution, and periodic human-labelled samples to estimate live accuracy. Feed back hard examples.

```python
import timm, torch
model = timm.create_model("convnext_tiny", pretrained=True, num_classes=12)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
```

### Q18. What approach would you take for a sentiment analysis task?
> ✍️ _Drafted answer (not from the original notes)._

**Clarify:** the granularity (document, sentence or **aspect-based**, e.g. "battery great, screen bad"), the labels (positive/negative/neutral or 1–5 stars), domain (tweets, reviews, support tickets), languages, volume and latency, and whether explainability is needed.

**Approach ladder, from cheapest up:**
1. **Lexicon / rule-based** (VADER): no training data needed, an instant baseline, but weak on sarcasm and domain slang.
2. **Classical ML:** TF-IDF n-grams with logistic regression or linear SVM. Fast, interpretable, and a surprisingly strong baseline.
3. **Fine-tuned transformer** (DistilBERT/RoBERTa, or XLM-R for multilingual): the best accuracy/cost balance with a few thousand labels.
4. **LLM zero- or few-shot prompting:** quick to start, handles aspect-based sentiment and nuance well. Expensive at scale, so often used to *label data* and then distilled into a small model.

**Data:** domain-labelled data matters more than model choice. Handle negation, emojis, sarcasm and mixed sentiment. Check the label balance, because neutral is often the majority.

**Evaluation:** macro-F1 (class imbalance), a confusion matrix (neutral versus mild positive confusion is common), and slice metrics by product, language and length.

**Production:** batch scoring for analytics dashboards and real-time scoring for ticket routing. Monitor drift, since new products and slang appear. Keep a human-labelled sample each month to track true accuracy.

```python
from transformers import pipeline
clf = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
clf(["Delivery was late but support fixed it fast"])
```

### Q19. How would you design a customer churn prediction model?
> ✍️ _Drafted answer (not from the original notes)._

1. **Define churn precisely:** contractual (cancelled subscription) or non-contractual (no purchase in 60 days). Fix a **prediction window** ("will churn in the next 30 days") and an **observation window** for features. Be clear about the action: a retention offer or a CSM call, which sets the precision and recall needs.
2. **Labels and leakage:** build snapshots. For each customer at date T, features use only data before T, and the label looks at T to T+30. Exclude signals that only appear after a churn decision, such as a "cancellation page visited" feature captured after the decision.
3. **Features:** tenure, plan and price, usage trends (week-over-week decline is very predictive), engagement recency/frequency/monetary value, support tickets and sentiment, payment failures, feature adoption, NPS, and contract renewal date.
4. **Model:** GBDT (XGBoost/LightGBM) on tabular snapshots is the default. Logistic regression gives an interpretable baseline. Survival models (Cox, or survival forests for time-to-churn) help when *when* matters.
5. **Evaluation:** time-based split, PR-AUC, lift and precision in the top decile (the customers you can actually contact), and calibration.
6. **Make it actionable:** churn probability alone is not enough. The aim is **uplift**: who will *stay because of* the intervention. Use uplift modelling or A/B-tested treatment policies. Otherwise discounts go to people who would have stayed anyway, and the "sleeping dogs" whom contact triggers to churn get woken up.
7. **Production:** a daily or weekly batch score pushed to the CRM, with SHAP reason codes ("usage down 40%, 2 failed payments") so account teams know what to do. Monitor drift and measure retention lift against a holdout.

### Q20. What would be your approach to ranking search results?
> ✍️ _Drafted answer (not from the original notes)._

**Multi-stage pipeline:**
```text
query understanding (spell, synonyms, intent, entities)
 ─► retrieval: lexical (BM25) + semantic (dense bi-encoder ANN) ─► ~1000 candidates
 ─► L1 ranker: GBDT LambdaMART on cheap features ─► ~100
 ─► L2 ranker: cross-encoder / deep model ─► top 10–20
 ─► business re-ranking: diversity, freshness, personalization, sponsored, policy
```

**Labels:** human relevance judgements (graded) give clean but expensive labels. Click logs give scale but are biased by position and presentation. Debias them with click models (cascade, DBN) or inverse propensity scoring estimated from randomisation experiments. Treat long clicks and conversions as stronger signals.

**Learning to rank:** pointwise (regression on relevance), pairwise (RankNet, LambdaRank), or listwise (LambdaMART, ListNet), which optimise NDCG directly. LambdaMART with LightGBM (`objective="lambdarank"`) is a strong production default.

**Features:** query-document text match (BM25 per field, embedding cosine), document quality (popularity, authority, freshness, ratings), personalisation (user history affinity), and context (location, device).

**Evaluation:** offline NDCG@10, MRR and recall on judged sets. Online: A/B tests on CTR, success rate (click with no quick return), conversion, query reformulation rate and zero-result rate. **Interleaving** finds a difference between rankers with far less traffic than A/B tests.

**Pitfalls:** feedback loops (the model only learns from what it showed, so add exploration). Training/serving skew in features. Over-personalisation. Latency budgets limit the cross-encoder to the top-N.

### Q21. How would you build a system to detect anomalies in network traffic?
> ✍️ _Drafted answer (not from the original notes)._

**Clarify:** the goal (intrusion/DDoS detection, data exfiltration, misconfigurations), the data (NetFlow/IPFIX, packet headers, firewall and DNS logs), the volume (millions of flows a second), and response time (seconds for DDoS). Labels are scarce, so this is **mostly unsupervised or semi-supervised**.

**Features:** aggregates per host, subnet and port in time windows: bytes and packets per second, connection count, distinct destination IPs and ports (scans), SYN/ACK ratios, flow duration, protocol mix, DNS query entropy (DGA/tunnelling), geo and ASN of peers, and deviation from each host's baseline by time of day.

**Models (layered):**
- *Statistical baselines:* per-entity seasonal baselines with z-score or EWMA thresholds. Cheap and explainable.
- *Unsupervised ML:* Isolation Forest, autoencoders (high reconstruction error means anomalous), or LSTM forecasters on per-host series.
- *Supervised:* GBDT trained on known attacks (CIC-IDS datasets plus labelled incidents) for known patterns.
- *Graph:* new or unusual edges in the host communication graph (lateral movement).

**Architecture:**
```text
flow collectors ─► Kafka ─► stream processing (Flink: windowed aggregates) ─► scoring service
   ─► alert correlation/dedup ─► SIEM / SOC dashboard ─► analyst feedback (true/false positive) ─► retraining & thresholds
raw data ─► data lake for forensics + offline training
```

**Metrics:** precision of alerts (alert fatigue kills these systems), detection latency, and recall on red-team or injected attacks.

**Pitfalls:** concept drift (new services, traffic growth), seasonality causing false alarms, encrypted traffic limiting content features, adversaries mimicking normal traffic, and the need for explainable alerts ("host X contacted 500 new IPs on port 445").

## Modelling, data & training

### Q22. How do you choose the right machine learning algorithm?
> ✍️ _Drafted answer (not from the original notes)._

Start from the problem and constraints, not the algorithm.

1. **Problem type:** classification, regression, ranking, clustering, forecasting, generation. Is ML needed at all, or will a rule or heuristic do?
2. **Data shape and size:**
   - *Tabular, small to medium:* gradient-boosted trees (XGBoost/LightGBM/CatBoost) are the default winner. Logistic or linear regression is the baseline.
   - *Images, audio, text:* deep learning, almost always **transfer learning** from pretrained models or foundation models.
   - *Very little labelled data:* pretrained embeddings plus a simple classifier, few-shot LLMs, or semi-supervised methods.
   - *Sequences and time series:* seasonal baselines, GBDT on lags, or sequence models.
3. **Constraints:** latency and memory (on-device means small models), interpretability and regulation (credit scoring favours monotonic GBDT or scorecards), training budget, and team skills.
4. **Empirical process:** build a trivial baseline, then a simple model, then iterate toward complexity only when validation metrics justify it. Use proper cross-validation (time-split when time matters) and compare on the *business metric*.
5. **Operational fit:** retraining cost, monitoring, explainability for stakeholders, and the cost of serving.

**Rule of thumb:** "No free lunch". Pick 2–3 candidates suited to the data type, benchmark them fairly, and prefer the simplest model within noise of the best. For a Head of Data and AI, the deciding factor is often maintainability and time-to-value, not +0.5% AUC.

### Q23. What is model drift, and how do you handle it?
> ✍️ _Drafted answer (not from the original notes)._

**Model drift** means a model's performance degrades in production because the world has moved away from its training data.

| Type | What changes | Example |
| --- | --- | --- |
| **Data / covariate drift** | P(X): input distribution | New user demographics, a new app version changes feature values |
| **Concept drift** | P(Y \| X): the relationship itself | Fraudsters change tactics; "normal" spending changes after inflation |
| **Label / prior drift** | P(Y): base rates | Churn rate jumps during a recession |
| **Upstream / data-quality issues** | Pipelines | A feature becomes null, units change from cents to dollars |

**Detection:**
- Monitor feature distributions against a training reference: PSI (PSI above 0.2 means significant), KS test, Jensen-Shannon/Wasserstein distance, and null rates.
- Monitor prediction distribution and confidence (a proxy when labels are delayed).
- Once labels arrive, track the true performance metrics by segment.
- Tools: Evidently, WhyLabs, Arize, or custom dashboards.

**Handling:**
- **Scheduled retraining** (daily or weekly) on a rolling window, or **triggered retraining** when drift or performance alerts fire.
- Online or incremental learning for fast-moving domains (ads, fraud).
- Weight recent data more heavily. Add robust features that are less sensitive to drift.
- Shadow-test or canary the retrained models before promotion. Keep the ability to roll back.
- Investigate the root cause first: many "drifts" are pipeline bugs, not changes in the world.

```python
import numpy as np
def psi(expected, actual, bins=10):
    q = np.quantile(expected, np.linspace(0, 1, bins + 1))
    e, a = np.histogram(expected, q)[0] / len(expected), np.histogram(actual, q)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return np.sum((a - e) * np.log(a / e))
```

### Q24. How would you handle large-scale data for training?
> ✍️ _Drafted answer (not from the original notes)._

**First ask whether all of it is needed.** Learning curves often flatten. Smart sampling (stratified, hard-example mining, negative downsampling with reweighting) can cut cost 10× with no loss.

**Storage and formats:** columnar formats (Parquet) on object storage with a table format (Delta/Iceberg) for versioning and time travel. Partition by date. For deep learning, stream sharded files (WebDataset, TFRecord, MosaicML StreamingDataset) instead of millions of small files.

**Processing:** distributed feature engineering with Spark, Dask, Ray Data or BigQuery. Precompute features once into a feature store or offline tables with point-in-time-correct joins.

**Data loading:** keep GPUs fed with parallel workers, prefetching, pinned memory, and caching preprocessed tensors. Profile: a GPU idle waiting on I/O is the most common waste.

**Distributed training:**
- *Data parallelism* (PyTorch DDP, Horovod): replicate the model, shard the batches.
- *Model and sharded parallelism* for big models: FSDP/ZeRO, tensor and pipeline parallelism (DeepSpeed, Megatron).
- For tabular data: distributed XGBoost/LightGBM on Spark or Dask, or out-of-core training.

**Incremental and online learning:** for streaming data, use `partial_fit` models, warm-starting from previous checkpoints, or continuous training pipelines.

**Operational concerns:** checkpoint often (spot instances), keep runs reproducible (data versioning, seeds), track experiments (MLflow/W&B), and control cost (spot or preemptible compute, autoscaling clusters).

### Q25. How do you deal with noisy data in machine learning models?
> ✍️ _Drafted answer (not from the original notes)._

Separate **feature noise** (measurement errors, outliers, missing values) from **label noise** (wrong annotations). Label noise is usually more damaging.

**Diagnose first:**
- Profile the data: distributions, impossible values, duplicates, nulls.
- Measure annotator agreement (Cohen's kappa) on a re-labelled sample.
- Use **confident learning** (cleanlab) or high-loss examples during training to surface likely mislabelled samples, then review them.

**Feature noise:**
- Validation rules and data contracts upstream (Great Expectations / dbt tests) so that garbage is caught at the source.
- Robust scaling, winsorising or clipping outliers, log transforms; median instead of mean imputation, plus missingness indicator features.
- Smoothing and aggregation (rolling windows) for sensor and time-series noise.
- Tree-based models are naturally fairly robust to outliers.

**Label noise:**
- Fix the labelling process: clearer guidelines, gold questions, consensus or majority vote, expert adjudication.
- Robust losses (MAE, Huber, generalized cross-entropy, symmetric CE), label smoothing.
- Down-weight or drop suspected noisy samples. Co-teaching or ensembles that disagree flag the doubtful ones.
- Weak supervision (Snorkel) models the accuracy of each noisy source explicitly.

**Modelling:** regularisation, early stopping (networks memorise noise late in training), and ensembles.

**Key principle (data-centric AI):** cleaning 5% of the labels often beats a month of model tuning. Keep a small, very clean **test set**, otherwise you cannot even measure progress.

### Q26. What strategies would you use to optimize the training time for a deep learning model?
> ✍️ _Drafted answer (not from the original notes)._

**Profile before optimising** (PyTorch Profiler, `nvidia-smi`, GPU utilisation). Find out whether the job is input-bound, compute-bound or communication-bound.

**Input pipeline:** more DataLoader workers, `pin_memory`, prefetching, preprocessed and cached datasets, GPU-side augmentation (DALI), and sharded streaming formats.

**Compute efficiency:**
- **Mixed precision** (bf16/fp16 with AMP): often a 2–3× speedup and half the memory.
- `torch.compile`, fused kernels and FlashAttention for transformers.
- Larger batch sizes to fill the GPU, with LR scaling and warmup (LARS/LAMB for very large batches). Use gradient accumulation if memory is tight.
- Gradient checkpointing to trade compute for memory, allowing larger batches or models.
- Channels-last memory format for CNNs, and tensor-core-friendly dimensions (multiples of 8).

**Algorithmic:**
- Transfer learning or fine-tuning instead of training from scratch, and parameter-efficient fine-tuning (LoRA).
- A better optimiser and schedule (AdamW with one-cycle or cosine), and early stopping.
- Smaller or distilled architectures. Progressive resizing (train on low resolution first).
- Data curation: deduplicate and drop low-value samples.

**Scale out:** DDP for data parallelism; FSDP/DeepSpeed ZeRO when the model doesn't fit; fast interconnects (NVLink, InfiniBand).

**Experiment efficiency:** tune hyperparameters on a subset or smaller model, use early-stopping search (ASHA/Hyperband via Ray Tune or Optuna), and use spot instances with checkpointing to cut cost.

```python
scaler = torch.cuda.amp.GradScaler()
with torch.autocast("cuda", dtype=torch.float16):
    loss = model(x).loss
scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
```

## Deployment, monitoring & scaling

### Q27. How do you deploy an ML model in production?
> ✍️ _Drafted answer (not from the original notes)._

**1. Choose the serving pattern from the requirements:**
- *Batch scoring* (nightly Spark or Airflow job writing predictions to a table): churn scores, recommendations. Cheapest and simplest.
- *Online, synchronous* (REST/gRPC microservice): fraud scoring, search ranking.
- *Streaming* (score events from Kafka/Flink): real-time alerting.
- *Edge or on-device:* privacy, offline use, latency.

**2. Package:** serialise the model plus the *exact* preprocessing (a sklearn Pipeline, a TorchScript/ONNX graph) and pin dependencies in a Docker image. Register it in a **model registry** (MLflow, SageMaker, Vertex) with lineage: data version, code commit, metrics.

**3. Serve:** FastAPI/BentoML for simple models; Triton, TorchServe, KServe or vLLM for GPU and LLM serving with dynamic batching. Run on Kubernetes or managed endpoints with autoscaling.

**4. Release safely:** CI/CD runs tests (unit, data validation, model quality gates against the current champion). Then **shadow deployment** (score live traffic with no impact), **canary** (1–5% of traffic), an A/B test on business metrics, and full rollout, with one-click rollback.

**5. Operate:** monitor latency, errors, throughput, input drift, prediction distribution and delayed-label performance. Log features and predictions for retraining and debugging. Retrain on a schedule or trigger.

```text
train ─► evaluate ─► registry (staging) ─► CI tests ─► shadow ─► canary ─► prod ─► monitor ─► retrain
```

**Key point:** the model is a small part of the system. Feature parity between training and serving (a feature store), versioning and monitoring are what make deployment reliable.

### Q28. How do you monitor a model's performance in production?
> ✍️ _Drafted answer (not from the original notes)._

Monitor in layers, from cheapest and fastest to most meaningful:

1. **Service health:** latency (p50/p95/p99), throughput, error rate, timeouts, CPU/GPU/memory. Standard SRE dashboards (Prometheus/Grafana) and alerts.
2. **Data quality:** schema violations, null and default-value rates, out-of-range values, missing upstream features, and freshness of streaming features. These catch most incidents, which are pipeline bugs.
3. **Data drift:** distribution of each important feature against the training baseline (PSI, KS, JS divergence). Weight alerts by feature importance.
4. **Prediction drift:** distribution of scores, predicted-class mix, and confidence. Shifts here are an early proxy when labels are delayed.
5. **Model quality (ground truth):** once labels arrive (minutes for clicks, weeks for chargebacks), compute AUC, precision/recall, calibration and RMSE, **sliced by segment** (region, device, new versus existing users) to catch hidden regressions and fairness issues.
6. **Business KPIs:** conversion, revenue, fraud losses, tickets. This is the outcome that matters, tracked against a holdout or control group where possible.

**Practices:**
- Log every request: features, model version, prediction, and later the label (an "inference log" joined with outcomes).
- Use human review samples for tasks without natural labels (moderation, LLM outputs).
- Alert with sensible thresholds and owners. Avoid alert fatigue.
- Tools: Evidently, Arize, WhyLabs, Fiddler, or SageMaker/Vertex Model Monitor.
- Connect monitoring to action: runbooks, automatic rollback, and retraining triggers.

### Q29. How would you deploy a model with low-latency requirements?
> ✍️ _Drafted answer (not from the original notes)._

Set the budget first (for example p99 under 20 ms end to end) and break it down: network, feature fetch, inference and post-processing. Then attack each part.

**Model side:**
- Pick or design a smaller model: GBDT or shallow networks, distillation from a big teacher.
- **Quantisation** (INT8/FP8), pruning, and operator fusion. Compile with ONNX Runtime, TensorRT or OpenVINO.
- Cascades: a cheap model handles most traffic, and only uncertain cases go to the heavy model.

**Serving side:**
- Keep the model loaded in memory and warm (no cold starts). Pre-allocate.
- Use an efficient runtime and protocol: gRPC instead of REST/JSON, Triton with **dynamic batching** tuned for latency (small max queue delay).
- Co-locate the model with the caller (sidecar or in-process) or deploy at the edge or in-region to cut network hops.
- Choose hardware to match: CPU with AVX-512 for small models, GPU or Inferentia for large models with batching.

**Features:**
- Precompute heavy features in batch or streaming jobs into a low-latency online store (Redis, DynamoDB, Feast online). Avoid live joins and SQL queries.
- Precompute embeddings and use ANN indexes.

**Caching and precomputation:** cache predictions for repeated inputs; fully precompute when the input space is enumerable (per-user recommendations refreshed hourly).

**Reliability:** timeouts with a fallback (last known good score, rule-based default), autoscaling on latency metrics, and load testing at peak QPS against p99, not the average.

### Q30. What are the common challenges when deploying ML models?
> ✍️ _Drafted answer (not from the original notes)._

- **Training/serving skew:** features computed differently offline (Spark SQL) and online (Java service), or preprocessing that doesn't match. *Fix:* shared feature definitions or a feature store, logging served features for training, and parity tests.
- **Data leakage discovered late:** a great offline AUC collapses online because a feature isn't available at prediction time. *Fix:* point-in-time-correct joins and time-based validation.
- **Data quality and pipeline fragility:** upstream schema changes, nulls, late data. *Fix:* data contracts, validation (Great Expectations), and alerting.
- **Drift and decay:** the world changes. *Fix:* monitoring and retraining pipelines.
- **Delayed or missing ground truth:** you can't measure accuracy quickly. *Fix:* proxy metrics, human labelling samples.
- **Latency, scalability and cost:** a big model is too slow or expensive. *Fix:* compression, batching, caching, right-sized hardware.
- **Reproducibility and versioning:** unknown data, code or hyperparameters behind the production model. *Fix:* model registry, data versioning, experiment tracking.
- **Dependency and environment issues:** "works on my notebook". *Fix:* containers, pinned environments, CI.
- **Offline metrics not matching business impact.** *Fix:* A/B tests and guardrail metrics.
- **Feedback loops:** the model's own outputs shape future training data. *Fix:* exploration and holdouts.
- **Governance:** explainability, fairness, privacy (GDPR), and audit trails, especially in regulated domains.
- **Organisational:** unclear ownership between data science and engineering, no on-call for models. *Fix:* MLOps platform and clear SLAs. This one is often the biggest blocker.

### Q31. Discuss scalability and latency requirements for ML systems.
> ✍️ _Drafted answer (not from the original notes)._

**Define the requirements up front:** peak QPS (not the average), p95/p99 latency SLOs, availability (99.9%?), data volume growth, freshness (how stale may features or models be), and cost per prediction. These choose the architecture.

**Latency tiers drive design:**
| Requirement | Pattern |
| --- | --- |
| Hours/days | Batch scoring, precomputed results in a table or KV store |
| Seconds | Async queue + workers, streaming |
| < 100 ms | Online service, precomputed features, optimized model |
| < 10 ms | In-process or edge inference, tiny model / lookup, heavy caching |

**Scalability levers:**
- *Horizontal scaling:* stateless inference services behind a load balancer, with Kubernetes HPA scaling on QPS, GPU utilisation or queue depth.
- *Batching:* dynamic batching raises GPU throughput at a small latency cost. This is the core throughput/latency trade-off.
- *Multi-stage funnels:* cheap retrieval over millions of items, expensive ranking over hundreds. Cost scales with the funnel, not the catalogue.
- *Caching:* results, embeddings and features. Use semantic caching for LLMs.
- *Sharding:* embedding tables, ANN indexes, feature stores.
- *Model efficiency:* quantisation, distillation, efficient serving runtimes (TensorRT, vLLM with continuous batching).

**Training scalability:** distributed data processing, distributed training, incremental retraining.

**Trade-offs to discuss:** accuracy against latency (bigger models are slower), freshness against cost (streaming against batch features), consistency against availability (serve a stale model or features instead of failing), and graceful degradation (fall back to a simpler model or popular items under load).

**Validate:** load tests at 2–3× expected peak, and capacity planning with cost per 1k predictions.

### Q32. How do you ensure your model is scalable and performs well with large datasets?
> ✍️ _Drafted answer (not from the original notes)._

**Training on large data:**
- Choose algorithms that scale: mini-batch SGD-trained models, GBDT with histogram binning (LightGBM), distributed implementations (XGBoost on Spark/Dask, PyTorch DDP/FSDP). Avoid O(n²) methods (kernel SVMs, exact kNN) at scale. Use approximate versions (ANN, random features).
- Efficient data handling: columnar formats, partitioning, streaming data loaders, out-of-core and incremental learning (`partial_fit`), sampling where learning curves show diminishing returns.
- Cache and reuse feature computation; vectorise; push heavy joins to Spark or the warehouse.

**Generalisation as the data grows:**
- Larger data usually supports higher-capacity models; re-tune regularisation and check learning curves.
- Validate on large, representative, time-split holdouts and check performance per segment, not just overall.
- Watch for label noise and class imbalance at scale.

**Inference at scale:**
- Make the model cheap: compression, efficient runtime, batching.
- Choose batch versus online serving deliberately. Precompute when possible.
- Stateless, horizontally scalable services with autoscaling; sharded embedding and ANN indexes.

**Engineering discipline:**
- Benchmark training time and inference latency/throughput as data grows (scaling tests on 10× synthetic data).
- Profile bottlenecks: I/O, CPU and GPU utilisation, memory.
- Monitor cost per training run and per prediction.
- Automate retraining pipelines so the scale stays manageable operationally.

**Example:** moving a churn model from pandas plus sklearn on a single node to Spark feature pipelines with distributed LightGBM cut training from 6 h to 40 min and allowed daily retraining on the full 2B-row history.

## Explainability, debugging & responsible AI

### Q33. What is model explainability? Why is it important?
> ✍️ _Drafted answer (not from the original notes)._

**Explainability** is the ability to understand *why* a model made a prediction (local explanation) and *how* it behaves overall (global explanation): which features drive it, in which direction, and with what interactions.

- **Interpretable by design:** linear and logistic regression, small decision trees, scorecards, GAMs/EBMs. The model *is* the explanation.
- **Post-hoc explanations** for black boxes: SHAP, LIME, permutation importance, partial dependence, counterfactuals.

**Why it matters:**
- **Trust and adoption:** business users, doctors and underwriters won't act on a black box. Reason codes ("declined: high utilisation, recent delinquencies") drive adoption.
- **Regulation and compliance:** credit decisions need adverse-action reasons (ECOA/FCRA in the US). GDPR covers automated decisions, and the EU AI Act requires transparency for high-risk systems.
- **Debugging:** explanations expose leakage (an ID or timestamp feature dominating), spurious correlations (a "husky vs wolf" model using snow), and data bugs.
- **Fairness:** detect reliance on protected attributes or their proxies (ZIP code as a proxy for race).
- **Actionability:** churn reason codes tell account managers *what* to do. Counterfactuals tell customers how to change an outcome.
- **Safety and accountability:** audit trails for incidents.

**Trade-off:** more interpretable models can lose some accuracy. On tabular data, GBDT with SHAP or EBMs often closes the gap. Post-hoc explanations are approximations and can be unstable or misleading, so validate them and don't over-trust them.

### Q34. What techniques would you use to make a model more interpretable?
> ✍️ _Drafted answer (not from the original notes)._

**1. Choose interpretable models when stakes are high:**
- Logistic regression with well-engineered features, scorecards (binned WoE features).
- Shallow decision trees and rule lists.
- **GAMs and Explainable Boosting Machines** (InterpretML): near-GBDT accuracy with per-feature shape plots.
- **Monotonic constraints** in XGBoost/LightGBM (risk must increase with debt), which make behaviour intuitive and defensible.

**2. Global post-hoc techniques:**
- Permutation feature importance (more reliable than impurity-based importance).
- SHAP summary plots: global importance plus direction.
- Partial dependence and ICE plots, ALE plots (better with correlated features).
- Global surrogate: a shallow tree trained to mimic the black box (report its fidelity).

**3. Local post-hoc techniques:**
- **SHAP values** (TreeSHAP is exact and fast for tree models): additive per-prediction attributions, which become reason codes.
- LIME: a local linear surrogate (less stable).
- **Counterfactual explanations** (DiCE): "if income were $5k higher, the loan would be approved".
- Anchors: rule-based local explanations.

**4. Deep learning:** Integrated Gradients, Grad-CAM saliency maps for vision, attention visualisation (with caution), concept-based methods (TCAV). For LLMs: citations and retrieval grounding, chain-of-thought rationales (not faithful by default).

**5. Simplify the input side:** fewer, meaningful features (feature selection), human-understandable feature engineering, and avoiding opaque hashed features.

```python
import shap
explainer = shap.TreeExplainer(xgb_model)
sv = explainer(X_valid)
shap.plots.beeswarm(sv)          # global
shap.plots.waterfall(sv[0])      # one prediction's reason codes
```

Always validate explanations with domain experts and sanity checks, such as randomised-label tests for saliency maps.

### Q35. Describe your approach to debugging an underperforming ML model.
> ✍️ _Drafted answer (not from the original notes)._

Work systematically, from the cheapest and most likely causes to the most expensive.

1. **Check the problem definition and metric.** Is the metric aligned with the business goal? Is the baseline sensible? Is the target right?
2. **Compare against baselines:** majority class, simple heuristics, logistic regression. If a complex model barely beats them, suspect the data or the features.
3. **Diagnose bias against variance with learning curves:**
   - *High train error* (underfitting): more capacity, better features, less regularisation, longer training.
   - *Low train error, high validation error* (overfitting): more data, regularisation, augmentation, simpler model, early stopping.
4. **Inspect the data:** label quality (re-label a sample), leakage, train/validation/test distribution mismatch, duplicates across splits, class imbalance, missing values, wrong joins, feature bugs. Most problems live here.
5. **Error analysis:** slice the errors by segment (region, device, class, input length). Look at the worst-scored examples manually and group failure modes. This shows whether you need more data for a slice, a new feature, or label fixes.
6. **Sanity checks on the pipeline:** overfit a tiny batch (if you can't, there's a bug), check that preprocessing matches between training and inference, shuffle labels (performance should drop to chance), check loss curves, the learning rate and gradient norms.
7. **Feature analysis:** importance and SHAP to spot leakage or features that do nothing; add domain features.
8. **Model and hyperparameters:** systematic tuning (Optuna), trying other model families or ensembles. Do this *after* the data checks.
9. **If it's production-only underperformance:** training/serving skew, drift, upstream pipeline changes, or feedback loops. Compare logged serving features with offline features.

Document each hypothesis and its result. Fix the largest error bucket first.

### Q36. How do you ensure fairness and reduce bias in ML models?

> 🔁 _Already answered in [05-optimization.md → Q7](../../ai-ml-engineering/ml-engineering/05-optimization.md#q7-how-do-you-ensure-fairness-and-reduce-bias-in-ml-models); skipped here to avoid duplication._

## MLOps & infrastructure

### Q37. Explain MLOps and its key components.
> ✍️ _Drafted answer (not from the original notes)._

**MLOps** applies DevOps and data-engineering practice to machine learning, so that models are built, deployed, monitored and improved **reliably, reproducibly and quickly**. The difference from DevOps is that an ML system depends on *code + data + model*, and all three change.

**Key components:**
| Component | Purpose | Example tools |
| --- | --- | --- |
| Data management & versioning | Reproducible datasets, lineage, quality checks | Delta/Iceberg, DVC, lakeFS, Great Expectations |
| Feature store | Consistent offline/online features, reuse | Feast, Tecton, Databricks/Vertex/SageMaker FS |
| Experiment tracking | Log params, metrics, artifacts; compare runs | MLflow, Weights & Biases |
| Training pipelines / orchestration | Automated, repeatable training workflows | Airflow, Kubeflow Pipelines, Vertex/SageMaker Pipelines, Dagster |
| Model registry | Versioned models, stage transitions, lineage, approvals | MLflow Registry, SageMaker/Vertex registry |
| CI/CD/CT | Test code, data and models; automated deploy; continuous training | GitHub Actions, Argo CD |
| Model serving | Batch, online, streaming, edge inference | KServe, Triton, BentoML, Seldon, vLLM |
| Monitoring & observability | Service health, drift, performance, business KPIs | Prometheus/Grafana, Evidently, Arize |
| Governance | Access control, audit, model cards, fairness, compliance | Unity Catalog, model cards |

**Maturity levels (Google):** level 0, manual notebooks and hand-offs. Level 1, automated training pipelines with continuous training. Level 2, full CI/CD of the pipelines themselves.

**Head-of perspective:** MLOps is a platform and team investment. Standardise paved roads (templates, a shared feature store and registry) so data scientists ship in days, not months. Measure time-to-production, the number of models in production, and incident rates.

### Q38. What is a feature store, and why is it important?
> ✍️ _Drafted answer (not from the original notes)._

A **feature store** is a central system for defining, computing, storing and serving ML features consistently for **training (offline)** and **inference (online)**.

**Core parts:**
- **Feature definitions / registry:** features defined once as code (entity, transformation, owner, freshness), discoverable and versioned.
- **Offline store** (warehouse or lake: BigQuery, Snowflake, Delta): full history for training, with **point-in-time-correct joins**, meaning each training row gets feature values as they were at that event's timestamp. This prevents leakage.
- **Online store** (Redis, DynamoDB, Cassandra): the latest feature values per entity with millisecond lookups for real-time inference.
- **Materialisation and ingestion:** batch jobs and streaming pipelines (Kafka/Flink) that compute features and keep both stores in sync.
- **Monitoring:** freshness, null rates, drift.

**Why it matters:**
1. **Eliminates training/serving skew:** the same logic and values serve both paths. This is the top cause of silent production failures.
2. **Prevents leakage** with time-travel joins.
3. **Reuse:** a "customer 30-day spend" feature built once is used by fraud, churn and marketing models, reducing duplicated pipelines.
4. **Low-latency serving** of precomputed and streaming features.
5. **Governance and discovery:** lineage, ownership, access control.

**Trade-offs:** operational complexity and cost. A small team with only batch models may not need one: well-modelled warehouse tables plus dbt may be enough. It pays off with many models, real-time features and several teams.

```python
from feast import FeatureStore
store = FeatureStore(repo_path=".")
training_df = store.get_historical_features(entity_df=events, features=["user_stats:spend_30d"]).to_df()
online = store.get_online_features(features=["user_stats:spend_30d"], entity_rows=[{"user_id": 42}]).to_dict()
```

### Q39. Cloud vs on-device Model Deployment.
> ✍️ _Drafted answer (not from the original notes)._

| Aspect | Cloud deployment | On-device (edge) deployment |
| --- | --- | --- |
| Compute | Effectively unlimited; large models, GPUs | Constrained CPU/NPU, memory, battery, thermals |
| Model size | Any (incl. large LLMs) | Small: compressed, quantized (e.g. ≤ few hundred MB) |
| Latency | Network round-trip (50–300 ms+) | Very low, no network |
| Offline use | Requires connectivity | Works offline |
| Privacy | Data leaves device; needs strong governance | Data stays local, which helps privacy and regulation |
| Cost | Pay per inference/server; scales with users | Inference cost borne by user hardware; near-zero marginal server cost |
| Updates | Instant, centralized, easy A/B tests and rollback | Requires app/model rollout; version fragmentation across devices |
| Monitoring | Full logging and observability | Limited telemetry; harder to debug and measure drift |
| Security | Model weights protected on server | Model can be extracted / reverse-engineered |
| Tooling | Triton, KServe, SageMaker, vLLM | TFLite/LiteRT, Core ML, ONNX Runtime Mobile, ExecuTorch, MediaPipe |

**When to choose:**
- *On-device:* real-time camera or AR, keyboard prediction, wake-word detection, privacy-sensitive health data, offline use, or a very high request volume where server costs would be large.
- *Cloud:* large or frequently updated models, heavy compute (big LLMs), need for central data and fast iteration.

**Hybrid is common:** a small on-device model handles common or simple requests (and privacy filtering), and hard requests escalate to a cloud model (Apple Intelligence's on-device plus Private Cloud Compute is one example). **Federated learning** can improve on-device models without centralising raw data.

### Q40. Tell about the Model Compression Techniques.
> ✍️ _Drafted answer (not from the original notes)._

Compression shrinks models to cut **latency, memory, cost and energy**, which matters for edge deployment and for cheap large-scale serving.

1. **Quantisation:** lower the numeric precision of weights and activations (FP32 → FP16/BF16 → INT8 → INT4/FP8).
   - *Post-training quantisation (PTQ):* fast, needs a small calibration set. LLM methods include GPTQ, AWQ, and bitsandbytes NF4.
   - *Quantisation-aware training (QAT):* simulates quantisation during training for better accuracy at low bit widths.
   - INT8 gives about 4× smaller weights and often 2–4× speedup on supported hardware, usually for under 1% accuracy loss.
2. **Pruning:** remove weights that matter little.
   - *Unstructured* (individual weights): high sparsity, but needs sparse kernels to get a speedup.
   - *Structured* (whole channels, heads, layers): real speedups on standard hardware. 2:4 semi-structured sparsity is supported on NVIDIA Ampere+.
3. **Knowledge distillation:** train a small *student* to mimic a large *teacher's* soft outputs or logits (and intermediate representations). Examples: DistilBERT (40% smaller, about 97% of BERT's performance), and distilling LLMs into small task models.
4. **Low-rank factorisation:** approximate weight matrices with low-rank products (SVD). LoRA uses the same idea for efficient fine-tuning.
5. **Weight sharing and clustering:** cluster weights into a small codebook (Deep Compression).
6. **Efficient architecture design / NAS:** MobileNet (depthwise separable convolutions), EfficientNet, SqueezeNet, and mixture-of-experts for sparse compute.
7. **Runtime and graph optimisation:** operator fusion, constant folding, compilation (TensorRT, ONNX Runtime, TVM, `torch.compile`). Not compression as such, but usually applied alongside it.

**Practical recipe:** distil or pick an efficient architecture, apply structured pruning if needed, then quantise (PTQ first, QAT if accuracy drops), then compile for the target hardware. Always benchmark accuracy *and* latency on the real target device, and check per-slice accuracy, because compression can hurt rare classes disproportionately.

```python
import torch
qmodel = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```
