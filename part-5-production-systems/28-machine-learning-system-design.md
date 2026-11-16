# Chapter 28. Machine Learning System Design

_Part V — Production Systems and Architecture_

End-to-end design cases for recommendation, search, ranking, fraud, forecasting and content understanding.

This chapter contains **27 questions**.

## Contents

- **28.1** [Design a Machine Learning System for YouTube Video Recommendation.](#281-design-a-machine-learning-system-for-youtube-video-recommendation)
- **28.2** [Design a Machine Learning System for YouTube Video Search.](#282-design-a-machine-learning-system-for-youtube-video-search)
- **28.3** [Design a Machine Learning System for Personalized Content Feed.](#283-design-a-machine-learning-system-for-personalized-content-feed)
- **28.4** [Design a Machine Learning System for Harmful Content Detection.](#284-design-a-machine-learning-system-for-harmful-content-detection)
- **28.5** [Design a Machine Learning System for Similar Listings on Airbnb.](#285-design-a-machine-learning-system-for-similar-listings-on-airbnb)
- **28.6** [Design a Machine Learning System for Replacement Product Recommendation.](#286-design-a-machine-learning-system-for-replacement-product-recommendation)
- **28.7** [Design a Machine Learning System for Event Recommendation.](#287-design-a-machine-learning-system-for-event-recommendation)
- **28.8** [Design a Machine Learning System for Ad Click Prediction.](#288-design-a-machine-learning-system-for-ad-click-prediction)
- **28.9** [Design a Machine Learning System to Estimate Delivery Time.](#289-design-a-machine-learning-system-to-estimate-delivery-time)
- **28.10** [Design a Machine Learning System for Image Search.](#2810-design-a-machine-learning-system-for-image-search)
- **28.11** [Design a Machine Learning System for Friends Recommendation.](#2811-design-a-machine-learning-system-for-friends-recommendation)
- **28.12** [Design a Machine Learning Product Recommendation System for an e-commerce platform.](#2812-design-a-machine-learning-product-recommendation-system-for-an-e-commerce-platform)
- **28.13** [How would you build a system to detect fraudulent transactions?](#2813-how-would-you-build-a-system-to-detect-fraudulent-transactions)
- **28.14** [How would you approach a time series forecasting problem?](#2814-how-would-you-approach-a-time-series-forecasting-problem)
- **28.15** [Describe how you would implement an image classification system.](#2815-describe-how-you-would-implement-an-image-classification-system)
- **28.16** [What approach would you take for a sentiment analysis task?](#2816-what-approach-would-you-take-for-a-sentiment-analysis-task)
- **28.17** [How would you design a customer churn prediction model?](#2817-how-would-you-design-a-customer-churn-prediction-model)
- **28.18** [What would be your approach to ranking search results?](#2818-what-would-be-your-approach-to-ranking-search-results)
- **28.19** [How would you build a system to detect anomalies in network traffic?](#2819-how-would-you-build-a-system-to-detect-anomalies-in-network-traffic)
- **28.20** [How do you build a spam detection system?](#2820-how-do-you-build-a-spam-detection-system)
- **28.21** [How do you build a harmful-content detection system?](#2821-how-do-you-build-a-harmful-content-detection-system)
- **28.22** [Design a real-time fraud detection system.](#2822-design-a-real-time-fraud-detection-system)
- **28.23** [Design a recommendation system for millions of users.](#2823-design-a-recommendation-system-for-millions-of-users)
- **28.24** [Design a large-scale document classification system.](#2824-design-a-large-scale-document-classification-system)
- **28.25** [Design a real-time anomaly detection pipeline.](#2825-design-a-real-time-anomaly-detection-pipeline)
- **28.26** [Design a personalized search engine.](#2826-design-a-personalized-search-engine)
- **28.27** [Design a model serving infrastructure supporting 10,000 requests/second.](#2827-design-a-model-serving-infrastructure-supporting-10000-requestssecond)

---

### 28.1 Design a Machine Learning System for YouTube Video Recommendation.

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

### 28.2 Design a Machine Learning System for YouTube Video Search.

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

### 28.3 Design a Machine Learning System for Personalized Content Feed.

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

### 28.4 Design a Machine Learning System for Harmful Content Detection.

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

### 28.5 Design a Machine Learning System for Similar Listings on Airbnb.

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

### 28.6 Design a Machine Learning System for Replacement Product Recommendation.

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

### 28.7 Design a Machine Learning System for Event Recommendation.

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

### 28.8 Design a Machine Learning System for Ad Click Prediction.

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

### 28.9 Design a Machine Learning System to Estimate Delivery Time.

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

### 28.10 Design a Machine Learning System for Image Search.

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

### 28.11 Design a Machine Learning System for Friends Recommendation.

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

### 28.12 Design a Machine Learning Product Recommendation System for an e-commerce platform.

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

### 28.13 How would you build a system to detect fraudulent transactions?

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

### 28.14 How would you approach a time series forecasting problem?

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

### 28.15 Describe how you would implement an image classification system.

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

### 28.16 What approach would you take for a sentiment analysis task?

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

### 28.17 How would you design a customer churn prediction model?

1. **Define churn precisely:** contractual (cancelled subscription) or non-contractual (no purchase in 60 days). Fix a **prediction window** ("will churn in the next 30 days") and an **observation window** for features. Be clear about the action: a retention offer or a CSM call, which sets the precision and recall needs.
2. **Labels and leakage:** build snapshots. For each customer at date T, features use only data before T, and the label looks at T to T+30. Exclude signals that only appear after a churn decision, such as a "cancellation page visited" feature captured after the decision.
3. **Features:** tenure, plan and price, usage trends (week-over-week decline is very predictive), engagement recency/frequency/monetary value, support tickets and sentiment, payment failures, feature adoption, NPS, and contract renewal date.
4. **Model:** GBDT (XGBoost/LightGBM) on tabular snapshots is the default. Logistic regression gives an interpretable baseline. Survival models (Cox, or survival forests for time-to-churn) help when *when* matters.
5. **Evaluation:** time-based split, PR-AUC, lift and precision in the top decile (the customers you can actually contact), and calibration.
6. **Make it actionable:** churn probability alone is not enough. The aim is **uplift**: who will *stay because of* the intervention. Use uplift modelling or A/B-tested treatment policies. Otherwise discounts go to people who would have stayed anyway, and the "sleeping dogs" whom contact triggers to churn get woken up.
7. **Production:** a daily or weekly batch score pushed to the CRM, with SHAP reason codes ("usage down 40%, 2 failed payments") so account teams know what to do. Monitor drift and measure retention lift against a holdout.

### 28.18 What would be your approach to ranking search results?

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

### 28.19 How would you build a system to detect anomalies in network traffic?

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

### 28.20 How do you build a spam detection system?

**1. Problem framing:** binary (or multi-class: spam, phishing, promo, ham) classification on messages or emails. **Precision matters most**: putting a legitimate email in spam is very costly. So pick an operating point such as precision ≥ 99% and maximize recall. Latency target is under about 50 ms per message at high volume. The setting is **adversarial**: spammers adapt.

**2. Data and labels:** user "report spam / not spam" actions (noisy, biased), honeypot accounts, analyst labels, blocklists. Use time-based splits, deduplicate near-identical campaigns (avoid leakage), and handle class imbalance by weighting rather than naive oversampling.

**3. Features / signals**

- **Content:** TF-IDF n-grams, embeddings (a fine-tuned DistilBERT or similar), URLs and their reputation, attachments, obfuscation (l33t speak, hidden text, Unicode homoglyphs).
- **Sender and metadata:** sender reputation and domain age, SPF/DKIM/DMARC results, IP reputation, sending volume and velocity, header anomalies.
- **Behavioral / graph:** recipient engagement, how many users received similar content (campaign clustering via MinHash), sender-recipient graph.

**4. Models (layered)**

- **Rules and blocklists** for known-bad items. Fast, explainable.
- **GBDT** (LightGBM) on reputation and metadata features, plus a **text model** (logistic regression on TF-IDF as a baseline, a transformer for quality). Combine via stacking.
- Optional: an **LLM** for borderline cases or a phishing-intent explanation, where cost allows.

**5. Serving:** a real-time scoring service with a feature store (precomputed reputations), thresholds for inbox / spam folder / quarantine / block, and per-user personalization (what one user calls spam is a newsletter to another).

**6. Evaluation and monitoring:** precision, recall, PR-AUC, false-positive rate on important mail, user complaint rate. Online A/B tests. Monitor **drift** closely: new campaigns appear daily.

**7. Adversarial loop:** frequent retraining (daily, or online learning), fast rule deployment for emerging campaigns, adversarial testing, rate limits on suspicious senders, and keep the exact features secret so they are harder to game.

### 28.21 How do you build a harmful-content detection system?

**1. Scope and policy first:** define the taxonomy (hate, harassment, violence, self-harm, sexual content, CSAM, extremism, spam, misinformation) with written policy guidelines and severity levels. The policy drives labeling, thresholds and actions. Cover multiple modalities (text, image, video, audio) and languages.

**2. Data and labeling:** expert annotators with clear guidelines, inter-annotator agreement tracking, annotator wellbeing safeguards, active learning to pick borderline examples, adversarial and red-team data, and rare-class oversampling. Severe classes are extremely imbalanced.

**3. Model stack (cascade for cost and latency)**

- **Hash matching** of known-bad content (PhotoDNA / perceptual hashes for CSAM and terror content, shared via industry hash databases). Exact and fast.
- **Lightweight classifiers** on everything: fine-tuned multilingual transformers (e.g., XLM-R) for text, CNN/ViT for images, multi-label, one head per policy.
- **Heavier models / LLM judges** (e.g., Llama Guard-style policy-conditioned classifiers, VLMs) for the uncertain middle band or new policies (zero/few-shot).
- **Context and behavioral signals:** user history, account age, report velocity, network or coordinated behavior.

**4. Decisioning:** per-class thresholds tied to actions: allow, down-rank, blur/interstitial, age-gate, remove, escalate to human review, report to authorities (CSAM, which is legally required). Severe harms get high-recall thresholds; borderline speech gets high-precision thresholds to limit over-moderation.

**5. Human in the loop:** review queues prioritized by severity and reach, appeals, reviewer decisions fed back as labels. Track reviewer accuracy.

**6. Evaluation and fairness:** per-class precision and recall, PR-AUC, prevalence estimation through sampling, latency. Do **bias audits across dialects and identity terms** (e.g., AAVE falsely flagged, reclaimed slurs) and per-language quality.

**7. Adversarial robustness and operations:** obfuscation (misspellings, emoji, text in images, so add OCR), evolving slang, coordinated attacks. Mitigate with frequent retraining, rapid rule updates, drift monitoring, and transparency reports.

### 28.22 Design a real-time fraud detection system.

**Requirements:** score card/payment transactions in < 100 ms p99 at thousands of TPS; high recall on fraud with tolerable false-positive rate; explainability and audit for regulators; adapt to evolving fraud patterns.

**Architecture:**
```txt
Payment API → Kafka → Feature service (online store: Redis/Feast)
                    → Scoring service (rules + model) → decision: approve / step-up (3DS/OTP) / decline / review
                    → Event log → offline lake (training, analytics)
```

**Features:**
- Real-time aggregates via streaming (Flink/Spark Streaming): txn count/amount per card in last 1 min/1 h/24 h, new device/IP/merchant flags, geo-velocity ("impossible travel").
- Static/profile: account age, historical spend, merchant risk.
- Graph features: shared devices/cards/emails across accounts (fraud rings).
- Same feature definitions offline and online (feature store) to avoid training-serving skew.

**Models:** a rules engine for known patterns and hard policy; a gradient-boosted model (XGBoost/LightGBM) as the main scorer — fast, strong on tabular data, explainable with SHAP; optionally GNN or sequence models offline, and anomaly detection for novel patterns.

**Key challenges:**
- **Extreme imbalance (~0.1%):** class weights, PR-AUC metrics, cost-sensitive thresholds (cost of fraud vs friction).
- **Label delay:** chargebacks arrive weeks later — use delayed labels plus analyst feedback; train on time-based splits.
- **Adversarial drift:** monitor score distributions and feature drift; retrain frequently; champion/challenger.

**Operations:** shadow-deploy new models, canary, latency budgets with fallback to rules if the model times out, case-management UI for analysts, full audit log of features, score and reason codes per decision.

### 28.23 Design a recommendation system for millions of users.

**Goal:** personalised top-N items for millions of users and items at low latency (~100 ms), optimising engagement while respecting business rules (diversity, freshness, inventory).

**Multi-stage funnel:**
```txt
Millions of items → Candidate generation (~1,000) → Ranking (~100) → Re-ranking/business rules → Top-N
```

1. **Candidate generation (high recall, cheap):**
   - Two-tower model: user tower and item tower produce embeddings; serve with ANN (FAISS, ScaNN, Milvus).
   - Collaborative filtering / matrix factorisation, item-to-item co-occurrence ("bought together"), trending/popular, content-based for new items.
   - Union multiple sources.
2. **Ranking (high precision):** GBDT or deep model (DLRM, DCN, Wide & Deep) with rich features — user history, item features, context (time, device), cross features; predicts multiple objectives (click, purchase, watch time) combined into a score.
3. **Re-ranking:** diversity (MMR), freshness, deduplication, fairness, sponsored items, filtering already-purchased or out-of-stock items.

**Data and infra:**
- Event streaming (Kafka) for impressions/clicks; feature store for online/offline consistency; batch-precompute embeddings and candidates for heavy users, real-time features for session intent.
- Logging of impressions (not just clicks) for unbiased training.

**Cold start:** content features, popularity by segment, onboarding preferences, exploration (bandits).

**Evaluation:** offline recall@k, NDCG, AUC; but decisions come from **online A/B tests** on business metrics (CTR, conversion, retention). Watch for feedback loops and popularity bias — keep some exploration traffic.

### 28.24 Design a large-scale document classification system.

**Scenario:** classify millions of documents (emails, contracts, tickets, invoices) into tens to hundreds of classes, possibly multi-label, with high throughput and auditability.

**Pipeline:**
```txt
Ingest (S3/queue) → Parse/OCR → Clean + language detect → Feature/embedding → Classifier → Confidence routing → Store labels + index
```

**Modelling choices (cheapest that works):**
- **Baseline:** TF-IDF + logistic regression/linear SVM — fast, strong, explainable.
- **Fine-tuned encoder** (DeBERTa/BERT/multilingual) for higher accuracy; truncate or chunk long docs and aggregate (max/mean pooling or hierarchical model).
- **Embeddings + lightweight classifier** for many classes and easy retraining.
- **LLMs** zero/few-shot for bootstrapping labels, rare classes or long-tail; distil into a smaller model for volume because LLMs are expensive at millions of docs.
- Hierarchical classification for large taxonomies.

**Labels:** start with weak supervision / LLM labelling, then human review via active learning (label the most uncertain samples).

**Scale and ops:**
- Batch inference on Spark/Ray with GPU workers; streaming path for new docs; ONNX/quantisation for throughput.
- **Confidence thresholds:** auto-accept high confidence, route low confidence to humans; their corrections feed retraining.
- Calibrate probabilities; per-class precision/recall and confusion matrix, macro-F1 for imbalanced classes.
- Monitor class distribution and drift; version models and data; keep audit trail of predictions.

### 28.25 Design a real-time anomaly detection pipeline.

**Use case:** detect anomalies in metrics, logs, transactions or IoT sensor streams within seconds, with low false-alert rates.

**Architecture:**
```txt
Sources → Kafka/Kinesis → Stream processor (Flink/Spark Streaming)
        → Feature windows → Detector(s) → Alert scoring/dedup → Alerting (PagerDuty/Slack) + dashboards
        → Raw + scored events to lake for training/backtesting
```

**Feature engineering on the stream:** sliding/tumbling windows (counts, means, percentiles, rates of change), seasonality-aware baselines (hour-of-day, day-of-week), per-entity state (per device/user/service).

**Detectors (layered, cheap first):**
- **Statistical:** z-score/robust MAD, EWMA, seasonal decomposition (STL), control charts — fast and explainable.
- **ML:** Isolation Forest, one-class models, autoencoders (reconstruction error), forecasting models (Prophet, LSTM) flagging large residuals.
- **Rules** for known critical conditions.
- Models trained offline, scored online; state in a fast store (Redis/RocksDB).

**Reducing noise:** require persistence (N anomalous windows), combine signals, group related alerts, severity scoring, suppression during deploys/maintenance, and feedback buttons so operators label alerts.

**Challenges:** few labels (mostly unsupervised; evaluate on injected anomalies and historical incidents with precision/recall and detection delay), concept drift (seasonality changes, new normal → periodic retraining or adaptive baselines), late/out-of-order events (watermarks), and scale (partition by entity key).

### 28.26 Design a personalized search engine.

**Goal:** return results ranked by relevance to the query **and** the individual user's preferences, within ~100–200 ms.

**Pipeline:**
```txt
Query → Query understanding → Retrieval (lexical + semantic) → Personalised ranking → Re-ranking → Results
```

1. **Query understanding:** spell correction, tokenisation, synonym expansion, intent and entity detection, category prediction, query embedding.
2. **Candidate retrieval (recall):** BM25 in Elasticsearch/OpenSearch plus dense retrieval via ANN; filters (availability, locale, permissions); a few hundred to a thousand candidates.
3. **Personalised ranking:** learning-to-rank (LambdaMART/GBDT or a neural ranker) with features:
   - Query–document: BM25 score, embedding similarity, field matches.
   - Document: popularity, quality, freshness, price.
   - **User:** embedding from history (clicks, purchases), preferred brands/categories, location, device, session context.
   - User–document cross features (affinity of the user to the item's category).
4. **Re-ranking:** diversity, business rules, de-duplication, fairness.

**Data:** click/purchase logs as implicit labels with **position-bias correction** (inverse propensity weighting, randomised interleaving); feature store for online/offline parity; user profiles updated in near real time.

**Evaluation:** offline NDCG/MRR on held-out logs; online A/B and interleaving tests on CTR, conversion, zero-result rate, time to click.

**Concerns:** cold-start users (fall back to non-personalised ranking), privacy and consent for using history, filter bubbles (keep some exploration), latency budget (precompute user embeddings, cache popular queries).

### 28.27 Design a model serving infrastructure supporting 10,000 requests/second.

**Targets:** 10k RPS, e.g. p99 < 100 ms, 99.9%+ availability, cost-efficient.

**Architecture:**
```txt
Clients → CDN/edge → API gateway / L7 load balancer (auth, rate limit)
        → Stateless inference service pods (Kubernetes, HPA/KEDA autoscaling)
        → Model server (Triton / TorchServe / vLLM for LLMs) on CPU/GPU nodes
        → Feature store (Redis) + cache · Metrics/tracing · Model registry
```

**Throughput techniques:**
- **Dynamic batching** in the model server (batch requests arriving within a few ms) — the biggest GPU win.
- **Model optimisation:** ONNX Runtime/TensorRT, quantisation (FP16/INT8), distillation, compiled graphs.
- **Caching:** response cache for repeated inputs, precomputed features/embeddings.
- **Async I/O** and gRPC between internal services; connection pooling.
- Right-size hardware: CPU for small tabular models, GPUs for deep models.

**Capacity planning:** benchmark one replica (e.g. 500 RPS at target latency) → ~20 replicas at peak, plus 30–50% headroom and N+1 across availability zones. Autoscale on RPS/queue depth/GPU utilisation, not just CPU.

**Reliability:** health/readiness probes, timeouts, retries with backoff, circuit breakers, load shedding and graceful degradation (fallback to a simpler model or cached/default result), multi-AZ/region.

**Deployment and ops:** model registry and versioned images; canary/blue-green rollout; shadow traffic for new models; monitor latency percentiles, error rate, throughput, saturation, prediction distributions and drift; load-test before launch.

---

← [Chapter 27. AI Infrastructure and Inference Optimization](27-ai-infrastructure-and-inference-optimization.md) · [Table of contents](../README.md) · [Chapter 29. Generative AI System Design](29-generative-ai-system-design.md) →
