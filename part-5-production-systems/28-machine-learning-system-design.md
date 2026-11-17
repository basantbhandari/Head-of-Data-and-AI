# Chapter 28. Machine Learning System Design

_Part V — Production Systems and Architecture_

A framework for answering design questions, then end-to-end design cases for recommendation, search, ranking, fraud, forecasting, pricing, risk, content understanding and the platforms that support them.

This chapter contains **37 questions**.

## Contents

- **28.1** [What framework do you use to answer an ML system design interview question?](#281-what-framework-do-you-use-to-answer-an-ml-system-design-interview-question)
- **28.2** [Design a Machine Learning System for YouTube Video Recommendation.](#282-design-a-machine-learning-system-for-youtube-video-recommendation)
- **28.3** [Design a Machine Learning System for YouTube Video Search.](#283-design-a-machine-learning-system-for-youtube-video-search)
- **28.4** [Design a Machine Learning System for Personalized Content Feed.](#284-design-a-machine-learning-system-for-personalized-content-feed)
- **28.5** [Design a Machine Learning System for Harmful Content Detection.](#285-design-a-machine-learning-system-for-harmful-content-detection)
- **28.6** [Design a Machine Learning System for Similar Listings on Airbnb.](#286-design-a-machine-learning-system-for-similar-listings-on-airbnb)
- **28.7** [Design a Machine Learning System for Replacement Product Recommendation.](#287-design-a-machine-learning-system-for-replacement-product-recommendation)
- **28.8** [Design a Machine Learning System for Event Recommendation.](#288-design-a-machine-learning-system-for-event-recommendation)
- **28.9** [Design a Machine Learning System for Ad Click Prediction.](#289-design-a-machine-learning-system-for-ad-click-prediction)
- **28.10** [Design a Machine Learning System to Estimate Delivery Time.](#2810-design-a-machine-learning-system-to-estimate-delivery-time)
- **28.11** [Design a Machine Learning System for Image Search.](#2811-design-a-machine-learning-system-for-image-search)
- **28.12** [Design a Machine Learning System for Friends Recommendation.](#2812-design-a-machine-learning-system-for-friends-recommendation)
- **28.13** [Design a Machine Learning Product Recommendation System for an e-commerce platform.](#2813-design-a-machine-learning-product-recommendation-system-for-an-e-commerce-platform)
- **28.14** [How would you build a system to detect fraudulent transactions?](#2814-how-would-you-build-a-system-to-detect-fraudulent-transactions)
- **28.15** [How would you approach a time series forecasting problem?](#2815-how-would-you-approach-a-time-series-forecasting-problem)
- **28.16** [Describe how you would implement an image classification system.](#2816-describe-how-you-would-implement-an-image-classification-system)
- **28.17** [What approach would you take for a sentiment analysis task?](#2817-what-approach-would-you-take-for-a-sentiment-analysis-task)
- **28.18** [How would you design a customer churn prediction model?](#2818-how-would-you-design-a-customer-churn-prediction-model)
- **28.19** [What would be your approach to ranking search results?](#2819-what-would-be-your-approach-to-ranking-search-results)
- **28.20** [How would you build a system to detect anomalies in network traffic?](#2820-how-would-you-build-a-system-to-detect-anomalies-in-network-traffic)
- **28.21** [How do you build a spam detection system?](#2821-how-do-you-build-a-spam-detection-system)
- **28.22** [Design a large-scale document classification system.](#2822-design-a-large-scale-document-classification-system)
- **28.23** [Design a real-time anomaly detection pipeline.](#2823-design-a-real-time-anomaly-detection-pipeline)
- **28.24** [Design a personalized search engine.](#2824-design-a-personalized-search-engine)
- **28.25** [Design a model serving infrastructure supporting 10,000 requests/second.](#2825-design-a-model-serving-infrastructure-supporting-10000-requestssecond)
- **28.26** [Design a feature store and feature platform for many ML teams.](#2826-design-a-feature-store-and-feature-platform-for-many-ml-teams)
- **28.27** [Design an experimentation (A/B testing) platform.](#2827-design-an-experimentation-ab-testing-platform)
- **28.28** [Design a dynamic pricing system.](#2828-design-a-dynamic-pricing-system)
- **28.29** [Design a demand forecasting and inventory replenishment system for a retailer.](#2829-design-a-demand-forecasting-and-inventory-replenishment-system-for-a-retailer)
- **28.30** [Design a customer lifetime value (LTV) prediction system.](#2830-design-a-customer-lifetime-value-ltv-prediction-system)
- **28.31** [Design a credit risk scoring system for loan applications.](#2831-design-a-credit-risk-scoring-system-for-loan-applications)
- **28.32** [Design a notification and push ranking system.](#2832-design-a-notification-and-push-ranking-system)
- **28.33** [Design a query autocomplete (typeahead) system.](#2833-design-a-query-autocomplete-typeahead-system)
- **28.34** [Design an entity resolution system to deduplicate customer records across sources.](#2834-design-an-entity-resolution-system-to-deduplicate-customer-records-across-sources)
- **28.35** [Design a predictive maintenance system for industrial equipment.](#2835-design-a-predictive-maintenance-system-for-industrial-equipment)
- **28.36** [Design a data labelling and human-in-the-loop annotation platform.](#2836-design-a-data-labelling-and-human-in-the-loop-annotation-platform)
- **28.37** [Design a homepage content optimisation system using contextual bandits.](#2837-design-a-homepage-content-optimisation-system-using-contextual-bandits)

---

### 28.1 What framework do you use to answer an ML system design interview question?

ML system design interviews are open-ended on purpose. Interviewers score structure, trade-offs and judgement more than any single model choice. A repeatable framework keeps the 45 minutes on track and makes sure nothing important is skipped.

| Step | Time | What to cover |
| --- | --- | --- |
| 1. Clarify the problem | 5 min | Business goal, users, surfaces, scale (users, items, QPS), latency budget, constraints (privacy, regulation, cost) |
| 2. Frame it as ML | 5 min | Is ML needed at all? The ML task (classification, ranking, regression, retrieval), the objective, and how labels are obtained |
| 3. Data and features | 8 min | Data sources, label quality and delay, feature groups (user, item, context, cross), freshness, leakage risks |
| 4. Model | 8 min | A simple baseline first, then the production model, multi-stage design if the candidate set is large, loss and training setup |
| 5. Evaluation | 5 min | Offline metrics tied to the objective, the split strategy (usually by time), online A/B metrics and guardrails |
| 6. Serving and architecture | 8 min | Batch or real time, feature store, latency budget per stage, caching, scaling, fallbacks |
| 7. Monitoring and iteration | 4 min | Data and prediction drift, retraining cadence, feedback loops, failure modes, fairness and safety |

**Habits that score well**

- **State assumptions with numbers.** "Assume 100M daily users, 10k candidate items per request, 200 ms p99." Numbers drive architecture choices.
- **Start simple.** A heuristic or logistic regression baseline shows judgement and gives something to beat.
- **Separate the offline and online metrics,** and explain how they connect to the business goal (AUC up does not always mean revenue up).
- **Name the trade-offs** explicitly: precision against recall, freshness against cost, model size against latency, personalisation against privacy.
- **Cover the unglamorous parts:** label delay, position bias, cold start, training and serving skew, and what happens when a dependency is down.
- **Drive the conversation,** but check in: "I will go deeper on ranking unless you would prefer serving."

**Common mistakes:** jumping straight to a deep model; no clarifying questions; ignoring how labels are obtained; random train and test splits on time-dependent data; and no plan for monitoring or retraining.

### 28.2 Design a Machine Learning System for YouTube Video Recommendation.

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

### 28.3 Design a Machine Learning System for YouTube Video Search.

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

### 28.4 Design a Machine Learning System for Personalized Content Feed.

**Clarify:** a social feed (posts from friends, pages and groups, plus unconnected recommended content). Hundreds of millions of DAU. The feed refreshes on every app open.

**Framing:** rank the candidate posts by a value model: $\text{score} = \sum_i w_i \cdot P(\text{action}_i)$, where the actions are like, comment, share, dwell over N seconds, hide and report (with negative weights). This is multi-task learning with one head per action. Integrity classifiers act as filters or demotions.

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

### 28.5 Design a Machine Learning System for Harmful Content Detection.

**See also:** 29.16 designs LLM-based content moderation.

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
Reviewer decisions flow back as training labels (active learning). For the uncertain middle band, and for new policies with few labels, a heavier policy-conditioned classifier or LLM judge (Llama Guard-style, or a vision-language model for images) can score zero-shot or few-shot from the written policy text.

**Policy and decisioning:** the written policy comes first. It defines the taxonomy, severity levels and examples, and it drives labelling guidelines, thresholds and actions. Actions are graded: allow, down-rank, blur or interstitial, age-gate, remove, escalate to human review, and report to authorities where the law requires it (CSAM). Severe harms get high-recall thresholds; borderline speech gets high-precision thresholds to limit over-moderation.

**Metrics:** per-class precision and recall, PR-AUC, and *prevalence* (the share of views that land on violating content, estimated by sampling). Also track appeal overturn rate and review-queue latency.

**Pitfalls:** heavy class imbalance (stratified sampling, focal loss). Adversarial evolution: text obfuscation, image perturbation and new slang call for frequent retraining and red-teaming. Policy drift when the definitions change. Fairness across dialects and languages (measure per-group false-positive rates; for example, African American English and reclaimed slurs are often falsely flagged). Text inside images and video (add OCR and ASR). Moderator wellbeing (blur content, show reviewers only what is needed).

### 28.6 Design a Machine Learning System for Similar Listings on Airbnb.

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

### 28.7 Design a Machine Learning System for Replacement Product Recommendation.

**Clarify:** when an item is out of stock (e-commerce or grocery substitution), suggest a *substitute* that the customer will accept. Substitutes are not the same as complements: a different-brand milk is a substitute, cereal is a complement.

**Framing:** learn a substitutability score $s(\text{original}, \text{candidate}, \text{user})$. Labels: accepted versus rejected substitutions (from grocery pickers or the customer approval flow), "viewed A then bought B instead" sessions (a substitute signal), and "bought together" pairs, which act as negatives for substitution because they are complements.

**Features:** product taxonomy (same leaf category), attributes (brand, size, unit price, dietary flags such as organic, gluten-free or vegan), text and image embeddings, price-per-unit ratio, user preferences (brand loyalty, dietary history), and stock at the store.

**Model:**
```text
candidate gen: same/adjacent category + embedding ANN (text/image) + co-view-not-co-buy graph
ranker: GBDT or pairwise DNN on (original, candidate, user) features
hard constraints: dietary/allergen rules, size bounds, price ceiling
```

**Metrics:** offline AUC and acceptance-rate prediction. Online: substitution acceptance rate, refund and return rate, order completion, and customer satisfaction.

**Pitfalls:** violating hard constraints such as allergens or halal (always use rules, never rely only on the model). Price jumps upset customers. Sparse labels for long-tail items (back off to category-level models). Store-level inventory must be checked in real time.

### 28.8 Design a Machine Learning System for Event Recommendation.

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

### 28.9 Design a Machine Learning System for Ad Click Prediction.

**Clarify:** predict pCTR for ads shown in feed or search. Billions of impressions a day, latency under 10 ms per batch of candidates. pCTR feeds the auction ($\text{bid} \times \text{pCTR} \times \text{quality}$), so **calibration matters as much as ranking**.

**Framing:** binary classification on impressions. Labels: click within a time window. A post-click conversion model (pCVR) is usually trained separately.

**Features:** very high-cardinality categoricals (user ID, ad ID, advertiser, campaign, publisher, keyword), user demographics and interests, context (placement, device, time), and historical CTRs at several granularities. Feature crosses matter (user segment × ad category).

**Model evolution:**
```text
Logistic regression + hashed crosses (baseline, very fast)
 → GBDT leaves as features + LR (Facebook 2014)
 → Factorization Machines / FFM
 → Deep models: Wide & Deep, DeepFM, DCN-v2, DLRM (embedding tables + interaction layers)
```

**Training:** continuous or online learning, because ads and users change hourly. **Negative downsampling** of non-clicks cuts cost. After downsampling, recalibrate with $p = \frac{p'}{p' + (1 - p')/w}$, where $w$ is the sampling rate.

**Metrics:** offline normalised cross-entropy (log loss relative to a baseline), AUC, and calibration (predicted versus observed CTR per bucket). Online: revenue, CTR, advertiser ROI, and user experience guardrails such as ad hide rate.

**Serving:** embedding tables (often hundreds of GB) live in a parameter server or sharded GPU memory. Feature logging at serving time, joined with delayed click labels, avoids training/serving skew.

**Pitfalls:** delayed feedback and label leakage, position bias (use a position feature or a separate position model), cold-start ads (content features and an exploration budget), privacy constraints (fewer cross-site signals, aggregated APIs).

### 28.10 Design a Machine Learning System to Estimate Delivery Time.

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

### 28.11 Design a Machine Learning System for Image Search.

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

### 28.12 Design a Machine Learning System for Friends Recommendation.

**Clarify:** "People You May Know" on a social network with about 1B users. The goal is accepted connections that lead to engagement, not just sent requests.

**Framing:** link prediction on the social graph. Score $P(\text{request sent} \land \text{accepted} \mid \text{viewer}, \text{candidate})$. Labels: past recommendations shown, then sent and accepted. Negatives are recommendations that were shown and ignored.

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

### 28.13 Design a Machine Learning Product Recommendation System for an e-commerce platform.

**See also:** 29.17 designs a real-time AI recommendation system, and Chapter 32 covers recommender techniques.

**Clarify:** surfaces are the homepage, product page ("similar"/"bought together") and cart. Catalogue of 10M+ SKUs, 100M users. The goals are GMV and conversion, with guardrails on returns.

**Framing:** retrieval plus ranking. Labels: purchase (strong), add-to-cart, click and view, weighted by value. Multi-task heads for P(click), P(add to cart) and P(purchase), because purchases are sparse.

**Candidate generation:**
- Collaborative: matrix factorisation / ALS, or two-tower on user-item interactions.
- Item-item co-view and co-purchase graphs (complements for the cart page).
- Session-based sequence models (GRU4Rec, SASRec) for short-term intent.
- Content-based retrieval for new items (text and image embeddings).

**Ranking:** a DNN or GBDT with user features (history embeddings, price affinity, brand affinity), item features (category, price, rating, stock, margin, delivery time), and context (surface, device, season, promotions).

**Re-ranking and business rules:** in-stock only, diversity across categories (for example, maximal marginal relevance), sponsored slots, margin-aware blending, and suppressing items already bought (unless consumables).

**Logging:** log every impression, not just clicks and purchases, with the position and the model version. Without impressions there are no true negatives, and training data reflects only what the old model chose to show.

**Metrics:** offline recall@k, NDCG and AUC on time-split data (never random splits, which leak the future). Online: CTR, conversion rate, GMV/revenue per session, return rate, and catalogue coverage.

**Serving:** precompute candidates for active users in batch. Do real-time session updates with streaming features. The ranker runs at request time with a feature store.

**Pitfalls:** popularity bias and long-tail coverage (reserve exploration traffic, for example with bandits), cold-start items and users (content embeddings, popularity by segment, onboarding preferences), seasonality, and the feedback loop with promotions. Offline metrics only shortlist models; the decision comes from online A/B tests on conversion, GMV and retention.

### 28.14 How would you build a system to detect fraudulent transactions?

**See also:** 29.20 covers where LLMs add value in fraud detection.

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

**Operations:** shadow-deploy new models and compare decisions before switching, then canary and champion/challenger. Keep a strict latency budget: if the model or feature service times out, fall back to the rules engine rather than blocking payments. Store a full audit log per decision (feature values, model version, score, reason codes) for disputes and regulators, and give analysts a case-management UI whose outcomes become labels.

**Pitfalls:** class imbalance (class weights, careful undersampling). Adversarial drift as fraudsters adapt (frequent retraining, drift monitoring). Selection bias, because declined transactions never get labels (keep a small random holdout that is approved for counterfactual data). Explainability needs for disputes and regulators (reason codes via SHAP).

### 28.15 How would you approach a time series forecasting problem?

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

### 28.16 Describe how you would implement an image classification system.

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

### 28.17 What approach would you take for a sentiment analysis task?

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

### 28.18 How would you design a customer churn prediction model?

1. **Define churn precisely:** contractual (cancelled subscription) or non-contractual (no purchase in 60 days). Fix a **prediction window** ("will churn in the next 30 days") and an **observation window** for features. Be clear about the action: a retention offer or a CSM call, which sets the precision and recall needs.
2. **Labels and leakage:** build snapshots. For each customer at date T, features use only data before T, and the label looks at T to T+30. Exclude signals that only appear after a churn decision, such as a "cancellation page visited" feature captured after the decision.
3. **Features:** tenure, plan and price, usage trends (week-over-week decline is very predictive), engagement recency/frequency/monetary value, support tickets and sentiment, payment failures, feature adoption, NPS, and contract renewal date.
4. **Model:** GBDT (XGBoost/LightGBM) on tabular snapshots is the default. Logistic regression gives an interpretable baseline. Survival models (Cox, or survival forests for time-to-churn) help when *when* matters.
5. **Evaluation:** time-based split, PR-AUC, lift and precision in the top decile (the customers you can actually contact), and calibration.
6. **Make it actionable:** churn probability alone is not enough. The aim is **uplift**: who will *stay because of* the intervention. Use uplift modelling or A/B-tested treatment policies. Otherwise discounts go to people who would have stayed anyway, and the "sleeping dogs" whom contact triggers to churn get woken up.
7. **Production:** a daily or weekly batch score pushed to the CRM, with SHAP reason codes ("usage down 40%, 2 failed payments") so account teams know what to do. Monitor drift and measure retention lift against a holdout.

### 28.19 What would be your approach to ranking search results?

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

### 28.20 How would you build a system to detect anomalies in network traffic?

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

### 28.21 How do you build a spam detection system?

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

### 28.22 Design a large-scale document classification system.

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

### 28.23 Design a real-time anomaly detection pipeline.

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

### 28.24 Design a personalized search engine.

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

### 28.25 Design a model serving infrastructure supporting 10,000 requests/second.

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

### 28.26 Design a feature store and feature platform for many ML teams.

**Clarify:** dozens of teams and hundreds of models, a mix of batch models (daily scoring) and real-time models (fraud, ranking) with online lookups under about 10 ms p99. Problems today: each team rebuilds the same features, training and serving use different code, and point-in-time mistakes leak the future into training data.

**Goals:** define a feature once and use it for both training and serving; point-in-time-correct training sets; low-latency online reads; discovery and reuse; governance (ownership, lineage, access control).

**Architecture:**
```text
feature definitions (code, versioned in git: entity, source, transformation, TTL, owner)
   ─► registry / catalog (metadata, lineage, owners, docs, usage)
batch sources (warehouse / lakehouse) ─► batch transforms (Spark / SQL / dbt) ─► offline store (Iceberg / Delta, time-stamped)
stream sources (Kafka) ─► stream transforms (Flink / Spark Streaming: windowed aggregates) ─┬─► online store (Redis / DynamoDB / Cassandra)
                                                                                         └─► offline store (same values, logged)
materialisation jobs: offline ─► online for batch features
training: entity rows + event timestamps ─► point-in-time join ─► training dataset
serving: model request ─► online store lookup by entity key (+ on-demand features computed at request time)
```

**Key design decisions**

- **Point-in-time joins:** for each training example at time T, take the latest feature value with a timestamp at or before T (an "as of" join). This prevents leakage and is the main reason to have a feature store.
- **One definition, two paths:** the same transformation code produces offline and online values. Where that is impossible (streaming aggregates), log the online feature values at serving time and train on the logs.
- **Freshness tiers:** batch (daily), near real time (minutes, micro-batch), and real time (streaming or on-demand at request time). Each costs more than the one before; pick per feature.
- **Online store sizing:** keys times features times bytes. For example, 200M users × 300 features × 8 bytes is about 480 GB before overhead, so the store is sharded with replication and TTLs.
- **Backfills:** new features must be computable over history, or they cannot be used for training until enough history accumulates.

**Build or buy:** Feast (open source), Tecton, or the platform-native stores in Databricks, SageMaker and Vertex AI. Most companies buy or adopt and invest their own effort in feature definitions, standards and governance.

**Metrics:** feature reuse rate, time to ship a new feature, online read latency and availability, freshness SLA adherence, and the number of training and serving skew incidents.

**Pitfalls:** building a platform before there are two or three real use cases; letting the registry become a dumping ground of unowned features (require owners and deprecate unused ones); ignoring PII (tag sensitive features and enforce access control); and expensive online storage for features only batch models use.

### 28.27 Design an experimentation (A/B testing) platform.

**Clarify:** a company running hundreds of concurrent experiments on web, mobile and backend services (including ML models). Needs: reliable randomisation, trustworthy statistics, self-service setup, and protection against harmful launches.

**Components:**
```text
experiment config service (UI + API: hypothesis, variants, traffic %, targeting, metrics, owner, duration)
   ─► assignment service / SDK: variant = hash(user_id, experiment_salt) mod 10,000 buckets  (deterministic, no lookup)
   ─► exposure logging (only when the user actually encounters the change)
event pipeline (Kafka ─► lakehouse) ─► metric definitions (semantic layer, versioned)
   ─► analysis engine (daily batch + near real-time guardrails)
   ─► results UI: effect, confidence interval, p-value, guardrails, SRM check, segment breakdowns
```

**Statistical design**

- **Randomisation unit:** usually the user. Use sessions only when user-level effects do not matter; use clusters (cities, markets) when there is interference between users, such as marketplaces or social networks.
- **Power analysis** before launch: minimum detectable effect, baseline variance, and required sample size and duration (cover full weekly cycles).
- **Variance reduction:** CUPED, which uses pre-experiment behaviour as a covariate, often cuts the required sample size by 30% to 50%.
- **Ratio metrics** (clicks per session) need the delta method or bootstrap for correct confidence intervals.
- **Multiple testing:** many metrics and segments inflate false positives. Pre-register one primary metric, and use corrections (Benjamini-Hochberg) for the rest.
- **Peeking:** checking results daily and stopping when significant inflates false positives. Use fixed horizons or sequential testing methods (for example, always-valid confidence sequences) if early stopping is needed.

**Trust checks:** sample ratio mismatch (SRM) detection with a chi-squared test on assignment counts; A/A tests to validate the pipeline; and automatic guardrails (latency, crashes, revenue, unsubscribes) that can stop an experiment.

**Concurrency:** layers or namespaces so that experiments on the same surface are mutually exclusive, while independent experiments overlap orthogonally.

**ML-specific use:** model launches go through shadow mode, then an A/B test on business metrics. Interleaving is useful for ranking because it detects differences with far less traffic.

**Metrics for the platform:** experiments run per quarter, time from idea to result, share of experiments with SRM or invalid results, and decisions that changed because of experiments.

**Pitfalls:** novelty and primacy effects (run long enough, look at trends over time); network effects violating independence; metric definitions that differ from the finance view; and launching on a non-significant "positive trend".

### 28.28 Design a dynamic pricing system.

**See also:** 29.42 designs an AI dynamic pricing engine from the generative AI angle.

**Clarify:** for example, nightly prices for a hotel or short-term rental marketplace, or prices for a retailer's catalogue. The objective is usually revenue or margin, subject to constraints: price bounds, fairness and legal limits, competitor parity, and brand perception. Hosts or merchants may accept or override suggestions.

**Framing:** two parts.
1. **Demand model:** predict demand as a function of price and context: $P(\text{booking} \mid \text{price}, x)$ or units sold $q(p, x)$.
2. **Optimisation:** choose the price that maximises expected revenue $p \cdot q(p, x)$ or margin $(p - c) \cdot q(p, x)$, subject to constraints and inventory (a unit unsold tonight is lost forever, so perishability matters).

**Data and features:** historical prices and outcomes, seasonality, day of week, lead time, local events, competitor prices, listing or product quality, inventory remaining, search and view demand signals, and macro factors.

**The causal problem:** historical prices were not random. Prices were raised when demand was high, so a naive model learns that higher prices go with more bookings. Remedies:
- Controlled price experiments (small randomised price perturbations) to estimate elasticity.
- Causal methods (double machine learning, instrumental variables) on observational data.
- Monotonic constraints in the model so demand cannot rise with price.

**Model:**
```text
features ─► demand model (GBDT with monotonic price constraint, or a parametric elasticity model per segment)
         ─► price optimiser (grid search over candidate prices, or a bandit for exploration)
         ─► business rules (min/max, max daily change, parity, legal) ─► suggested price
         ─► host/merchant accept or override ─► outcomes logged ─► retrain
```
Contextual bandits (for example, Thompson sampling over price levels) balance exploiting the best known price with exploring to learn elasticity.

**Metrics:** offline demand-model log loss and calibration across price levels; online revenue per available unit (RevPAR in hotels), occupancy or sell-through, margin, conversion, host adoption of suggestions, and complaints.

**Pitfalls:** confounding between price and demand; competitors' algorithms reacting to yours (price wars); fairness and regulation (no discrimination by protected attributes, care with personalised pricing); price instability that upsets customers; and surge pricing during emergencies, which causes reputational damage.

### 28.29 Design a demand forecasting and inventory replenishment system for a retailer.

**Clarify:** 5,000 stores × 50,000 SKUs is about 250 million store-SKU series, forecast daily for the next 1 to 28 days, used to place replenishment orders. Costs: stockouts (lost sales, unhappy customers) against overstock (holding cost, waste for perishables).

**Framing:** probabilistic forecasting. Replenishment needs a distribution or quantiles, not just a mean: the order quantity is chosen from a service-level quantile.

**Data and features:** sales history (which is censored: when stock is zero, sales understate demand), prices and promotions, calendar and holidays, weather, store attributes, product hierarchy, new product launches, and stock levels.

**Model:**
```text
history + features ─► global GBDT (LightGBM) with lag, rolling and calendar features, quantile or Tweedie loss
                   (alternatives: DeepAR / TFT, hierarchical reconciliation)
                 ─► forecasts per store-SKU-day (P50, P90)
                 ─► reconciliation across hierarchy (store ─► region ─► total; SKU ─► category)
                 ─► replenishment policy: order = target stock − on hand − on order,
                    target = forecast over lead time + safety stock at the chosen service level
                 ─► orders to suppliers / DCs, with planner overrides
```

- A single **global model** across all series shares patterns and handles sparse series better than millions of local models.
- **Intermittent demand** (many zeros for slow movers) fits Tweedie or Croston-style methods.
- **Cold start** for new products: borrow from similar products by attributes and category.
- **Censored demand:** do not train on stockout days as true demand; impute or exclude them.

**Evaluation:** rolling-origin backtests. Metrics: WAPE and bias at the decision level (store-SKU over the lead time), pinball loss for quantiles, and business simulation of stockout rate, inventory days and waste.

**Serving:** a nightly batch on Spark or Ray across all series, finishing before the ordering cut-off. Monitor forecast error and bias by category and region; alert on input data problems (missing sales feeds) before the forecasts reach orders.

**Pitfalls:** promotion cannibalisation and halo effects; supply constraints feeding back into the sales data; planner overrides that are never measured (track forecast value added); and optimising forecast accuracy instead of business cost.

### 28.30 Design a customer lifetime value (LTV) prediction system.

**Clarify:** what will the prediction be used for? Common uses are marketing bid caps (how much to pay to acquire a customer), segmenting customers for retention and service levels, and finance planning. The horizon is usually 12 to 36 months. Non-contractual businesses (retail, apps) are harder than contractual ones (subscriptions).

**Framing:** predict future value (revenue or margin) per customer over the horizon. The target distribution is unusual: many customers are worth zero, and a few are worth a lot.

**Modelling options:**
- **Probabilistic models** for non-contractual settings: BG/NBD (or Pareto/NBD) predicts the number of future purchases from recency, frequency and customer age; Gamma-Gamma predicts spend per purchase. They are interpretable and work with little data.
- **ML models:** GBDT on rich features (behaviour, product mix, channel, engagement, support history). Handle the many zeros with a two-part model ($P(\text{purchase}) \times E[\text{value} \mid \text{purchase}]$) or with a zero-inflated lognormal loss.
- **Early LTV:** predict from the first 7 to 30 days of behaviour to support acquisition decisions quickly.

**Data:** a snapshot design. At cutoff date T, features use only data before T; the label is value from T to T plus the horizon. Train on several historical cutoffs. Use margin, not revenue, when returns and discounts are large.

**Evaluation:** time-based holdouts; normalised Gini or a lift chart (are high predictions really high-value customers?); decile calibration (predicted against actual totals per decile); and total predicted versus actual for finance.

**Serving:** weekly batch scores written to the warehouse, CRM and ad platforms (as conversion values for value-based bidding).

**Pitfalls:** long horizons mean labels arrive late, so validate with shorter proxy horizons; changes in pricing or product make history less representative; feedback loops (high predicted LTV gets more retention spend, which raises actual LTV); and privacy rules when sending predictions to ad platforms.

### 28.31 Design a credit risk scoring system for loan applications.

**Clarify:** consumer loans; decision in seconds at application; outcomes are approve, decline, or approve with a different limit or rate. It is a **regulated** domain: decisions must be explainable to applicants (adverse action reasons), fair across protected groups, documented, and reviewed by model risk management.

**Framing:** predict probability of default (PD) within a fixed window (for example, 90+ days past due within 12 months). The decision policy combines PD with expected loss: $\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$ (loss given default and exposure at default), compared against the expected profit from the loan.

**Data and features:** credit bureau data (history, utilisation, delinquencies, inquiries), application data (income, employment, requested amount), internal history for existing customers, and, where allowed, cash-flow data from bank transactions with consent. Protected attributes (race, religion and others) must not be used as features, and proxies for them need testing.

**Model:**
- The traditional choice is a logistic regression scorecard with binned, weight-of-evidence features: very interpretable and easy for regulators.
- GBDT with **monotonic constraints** (for example, risk cannot decrease as delinquencies increase) usually improves accuracy. SHAP-based reason codes provide adverse action explanations.
- **Calibration** is essential, because PD feeds pricing and capital calculations.

**Reject inference:** the training data only has outcomes for approved applicants, so the model learns from a biased sample. Remedies: a small, controlled approval of below-threshold applicants (where allowed), augmentation or parcelling methods, and bureau outcomes from loans taken elsewhere.

**Evaluation:** AUC and Gini, KS statistic, calibration by score band, stability over time (population stability index), and fairness metrics across groups (approval rate differences, equal opportunity) with documented trade-offs.

**Governance and operations:** model documentation, independent validation, champion/challenger testing, monitoring of PSI, approval rates and default rates by vintage, human review for borderline cases, and a full audit trail of every decision with model version and reason codes.

**Pitfalls:** long label delay (12 months), so monitor early delinquency proxies; economic cycles shifting default rates; proxy discrimination through features like postcode; and features that applicants can easily manipulate.

### 28.32 Design a notification and push ranking system.

**See also:** 29.39 designs LLM-based notification prioritisation.

**Clarify:** an app with many notification types (social activity, recommendations, reminders, promotions). The goal is long-term engagement and retention, not opens today. Too many notifications cause users to disable notifications or uninstall the app, which is a permanent loss.

**Framing:** for each candidate notification, predict the value of sending it now, minus its cost:

$\text{score} = P(\text{open}) \cdot V(\text{open}) - P(\text{disable or uninstall}) \cdot C$

Then decide **whether** to send, **which** notification, and **when**, under a per-user volume budget.

**Features:** user activity level and recency, historical open rate by notification type, time-of-day responsiveness, notifications already received today, content relevance (affinity with the actor or item), and device and time zone.

**Architecture:**
```text
event triggers (likes, new content, reminders) ─► candidate notifications queue
  ─► dedupe / aggregate ("5 people liked your post") ─► ranking model (multi-task: open, downstream engagement, disable)
  ─► policy: per-user daily cap, quiet hours, minimum spacing, type quotas, send-time optimisation
  ─► delivery (APNs / FCM) ─► open and disable events ─► training data
```

**Evaluation:** offline AUC per head and calibration; online A/B tests measured over weeks, on daily active users, retention, notification-disable rate and uninstalls, not just open rate. Hold out a group with fewer notifications to measure the true incremental effect.

**Pitfalls:** optimising open rate drives notification spam and long-term harm; delayed negative signals (users turn off notifications days later); unfair volume between types owned by different product teams (central arbitration is needed); and time-sensitive notifications that lose value if delayed.

### 28.33 Design a query autocomplete (typeahead) system.

**Clarify:** suggestions appear as the user types in a search box. Very high QPS (every keystroke), latency under 50 ms end to end, results should be popular, fresh, personalised and safe.

**Framing:** given a prefix and context, rank candidate completions by $P(\text{completion selected} \mid \text{prefix}, \text{user}, \text{context})$.

**Candidate generation:**
- A **trie** (or finite-state transducer) of historical queries with their frequencies, giving the top-k completions per prefix. Precompute top-k at each node so lookup is $O(\text{prefix length})$.
- Fresh trending queries from a streaming counter (recent time-decayed counts), merged with the historical trie.
- Fuzzy matching for typos (edit distance one or two), and a language model to generate completions for rare prefixes.

**Ranking features:** global popularity, recency and trend, prefix match quality, user history (past queries, clicked categories), location and language, and the expected success of the completed query (does it lead to clicks, or to zero results?).

**Architecture:**
```text
query logs ─► batch aggregation (daily) ─► trie build ─► distributed in-memory servers (sharded by prefix, replicated)
trending stream (Kafka ─► Flink counts) ─► fresh suggestions overlay
request: prefix ─► candidates (trie + fresh + personal) ─► light ranker ─► safety filter ─► top 5–10
client: debounce keystrokes, cache prefix results locally
```

**Metrics:** suggestion acceptance rate, keystrokes saved, latency p99, search success rate after a suggestion, and the rate of offensive or harmful suggestions (must be near zero).

**Pitfalls:** offensive, defamatory or dangerous completions (blocklists, classifiers and human review for popular prefixes); manipulation by coordinated queries; privacy (never suggest rare queries that could expose one user's data, so apply a minimum-count threshold); and stale suggestions during breaking news.

### 28.34 Design an entity resolution system to deduplicate customer records across sources.

**Clarify:** customer records arrive from a CRM, e-commerce accounts, a loyalty programme and support tools, with no shared ID. Tens of millions of records. Goal: a single customer view (a "golden record") used by analytics, marketing and ML. Mistaken merges are worse than missed merges, especially when they mix two people's personal data.

**Pipeline:**
```text
sources ─► standardise (names, addresses, phones to E.164, emails lower-cased, dates)
  ─► blocking (cheap candidate pairs): same email, same phone, same postcode + name prefix, LSH on name embeddings
  ─► pairwise matching model: P(same entity | record A, record B)
  ─► clustering (connected components with thresholds, or correlation clustering) ─► entity IDs
  ─► survivorship rules ─► golden record per entity ─► publish to warehouse / MDM / CRM
  ─► human review queue for uncertain pairs ─► labels back to the model
```

**Blocking** is essential because comparing every pair is impossible ($n^2$ for $10^7$ records is $10^{14}$ pairs). Good blocking keeps recall high while cutting pairs to millions. Measure blocking recall separately.

**Matching features:** string similarity per field (Jaro-Winkler for names, token-based for addresses), exact match flags on email and phone, nickname and transliteration handling, date-of-birth agreement, geographic distance, and embedding similarity. A GBDT classifier trained on labelled pairs works well; the Fellegi-Sunter probabilistic model is the classic unsupervised option (open-source tools such as Splink implement it).

**Clustering pitfalls:** transitive chaining (A matches B, B matches C, but A and C are clearly different people). Limit cluster size, use stricter thresholds for merges that bridge clusters, and check cluster consistency.

**Evaluation:** pairwise precision and recall on a labelled sample, cluster-level metrics (B-cubed), and manual audits of the largest clusters.

**Operations:** incremental matching for new records, stable entity IDs over time (avoid renumbering), the ability to unmerge, audit history of merges, and GDPR handling (deletion requests must find every record for a person).

### 28.35 Design a predictive maintenance system for industrial equipment.

**Clarify:** thousands of machines (pumps, turbines, conveyors) with sensors streaming vibration, temperature, pressure and current. The goal is to predict failures early enough to schedule maintenance, reducing unplanned downtime. A missed failure is costly; too many false alarms waste technician time and erode trust.

**Framing:** several options, depending on labels.
- **Classification:** will this machine fail within the next N days?
- **Remaining useful life (RUL) regression:** how many cycles or hours until failure? Survival models handle machines that have not failed yet (censored data).
- **Anomaly detection:** when failure labels are very rare, learn normal behaviour and flag deviations.

**Data and features:** sensor time series aggregated over windows (mean, standard deviation, trend, kurtosis), frequency-domain vibration features (FFT bands, which reveal bearing faults), operating conditions (load, speed) to normalise readings, maintenance logs and failure records (often free text and inconsistent), machine age and type.

**Architecture:**
```text
sensors ─► edge gateway (filtering, downsampling, local alerts) ─► IoT hub / Kafka
  ─► stream processing (window features) ─► time-series store + lakehouse
  ─► models (per machine type): anomaly score + failure risk / RUL
  ─► alerting with explanation ("vibration band 2 up 40% over 7 days")
  ─► maintenance system (work orders) ─► technician findings ─► labels
```

**Evaluation:** time-based splits by machine; precision and recall of alerts at the chosen lead time; lead time distribution (how early did we warn?); and business metrics: unplanned downtime, maintenance cost, and false alarm rate per technician per week.

**Pitfalls:** very few failures, and each machine type fails differently; labels from maintenance logs are noisy; preventive maintenance removes failures from the data before they happen; sensor faults look like machine faults (detect sensor drift separately); and models must be explainable for engineers to act.

### 28.36 Design a data labelling and human-in-the-loop annotation platform.

**Clarify:** several ML teams need labels for text, images and documents; some labels need domain experts (medical, legal); budgets are limited; label quality directly limits model quality.

**Components:**
```text
data selection ─► (active learning: uncertainty, diversity, error-prone slices; model pre-labels)
  ─► task design (guidelines with examples and edge cases, label schema)
  ─► workforce routing (in-house experts / vendor / crowd, by task difficulty and sensitivity)
  ─► annotation UI (pre-filled suggestions, shortcuts, QA flags)
  ─► quality control: gold questions, overlap (2–3 annotators on a sample), agreement metrics, adjudication
  ─► versioned label store (who, when, guideline version) ─► training datasets
  ─► model errors and production disagreements ─► back to data selection
```

**Quality techniques:**
- **Clear guidelines** are the biggest lever. Revise them when annotators disagree, and version them.
- **Inter-annotator agreement** (Cohen's or Fleiss' kappa, Krippendorff's alpha) per class, to find ambiguous classes.
- **Gold questions** with known answers, mixed in to measure each annotator's accuracy.
- **Aggregation** of multiple labels by majority vote or annotator-reliability models (Dawid-Skene).
- **Model-assisted labelling:** humans correct pre-labels, which is faster, but watch for anchoring bias (spot-check without pre-labels).
- **LLM labelling** for simpler tasks, validated against a human gold set, with humans for hard and sensitive cases.

**Metrics:** cost per label, throughput, label accuracy against gold, agreement, model improvement per labelling dollar.

**Pitfalls:** privacy and security of sensitive data sent to vendors (redaction, contracts, secure environments); annotator wellbeing for harmful content; labels silently changing meaning when guidelines change; and spending budget on easy examples the model already gets right.

### 28.37 Design a homepage content optimisation system using contextual bandits.

**Clarify:** a homepage with several slots (hero banner, carousels, promotions) and a frequently changing pool of content. Many items are new and have no history. The goal is engagement or conversion, and we want to learn quickly which content works for which users.

**Why bandits:** a standard supervised model trained on logged data only learns about content the old system chose to show. Bandits explicitly balance **exploitation** (show what looks best) with **exploration** (show uncertain items to learn), which suits fast-changing inventory and cold start.

**Framing:** a contextual bandit. For each request, observe context $x$ (user features, time, device), choose an action $a$ (a content item or layout), and observe reward $r$ (click, conversion). Learn a policy that maximises expected reward.

**Algorithms:**
- **Epsilon-greedy:** simple; explore uniformly a fraction of the time.
- **LinUCB:** a linear reward model per action with an upper confidence bound bonus for uncertainty.
- **Thompson sampling:** sample model parameters from the posterior and act greedily on the sample; strong in practice and easy to reason about.
- Neural or GBDT reward models with approximate uncertainty for richer features.

**Architecture:**
```text
request ─► context features ─► policy service: score candidates, sample/choose action, record propensity P(a | x)
        ─► render ─► reward events (click, conversion within window)
        ─► join (context, action, propensity, reward) ─► update policy (minutes to hours)
offline: logged data ─► off-policy evaluation (IPS / doubly robust) of candidate policies before launch
```

**Logging propensities** is critical: with $P(a \mid x)$ recorded, new policies can be evaluated offline with inverse propensity scoring or doubly robust estimators before any A/B test.

**Metrics:** cumulative reward and regret, click or conversion rate against a control, exploration cost, and coverage of new content.

**Pitfalls:** delayed rewards (conversions arrive hours later); non-stationarity (use decay or sliding windows); slot interactions on a page with multiple slots (slate bandits or ranking approaches); optimising clicks instead of long-term value; and business constraints such as contractual promotion exposure.

---

← [Chapter 27. AI Infrastructure and Inference Optimization](27-ai-infrastructure-and-inference-optimization.md) · [Table of contents](../README.md) · [Chapter 29. Generative AI System Design](29-generative-ai-system-design.md) →
