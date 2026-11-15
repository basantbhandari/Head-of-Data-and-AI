# Computer Vision

Computer vision interview questions: core tasks (classification, detection, segmentation, OCR, tracking), feature extraction, CNNs vs. dense networks, data augmentation, frameworks, Vision Transformers, world models and JEPA.

### Q1. What is computer vision, and why is it important?

#### Computer Vision

- **Core idea:** A field of AI that enables machines to interpret, analyze, and understand visual information from the world (images, videos) — essentially teaching computers to "see" and derive meaningful understanding from visual data, similar to human vision.
- **Core tasks:**
  - **Image classification:** Assigning a label/category to an entire image (e.g., "cat" vs. "dog").
  - **Object detection:** Identifying and localizing multiple objects within an image (bounding boxes + class labels).
  - **Semantic/Instance segmentation:** Classifying each pixel in an image into a category (semantic), or distinguishing individual object instances (instance segmentation).
  - **Image generation:** Creating new images (e.g., via GANs, diffusion models).
  - **Optical Character Recognition (OCR):** Extracting text from images.
  - **Pose estimation, face recognition, tracking:** Identifying human poses, faces, or tracking objects across video frames.
- **How it works (typical pipeline):** Raw pixel data → feature extraction (traditionally hand-crafted features like edges/corners; now mostly learned via CNNs/Vision Transformers) → task-specific output (classification, detection, etc.).

---

##### Why It's Important

- **Automates visual understanding tasks** that previously required human judgment — enabling scale and consistency that manual review can't match.
- **Wide range of real-world applications:**
  - **Healthcare:** Medical image analysis (X-rays, MRIs) for disease detection/diagnosis.
  - **Autonomous vehicles:** Detecting pedestrians, other vehicles, traffic signs, lane markings for self-driving systems.
  - **Retail/manufacturing:** Quality control/defect detection on production lines, automated checkout systems.
  - **Security/surveillance:** Facial recognition, anomaly detection in video feeds.
  - **Agriculture:** Crop monitoring, disease detection in plants via drone/satellite imagery.
  - **Augmented/Virtual Reality:** Real-time scene understanding for AR overlays.
  - **Document processing:** OCR for digitizing/extracting information from scanned documents.
- **Economic/societal impact:** Drives efficiency gains, cost reduction, and new capabilities across industries — from enabling self-driving cars to speeding up medical diagnosis, making it one of the most commercially impactful areas of AI alongside NLP.
- **Foundational for multimodal AI:** As AI systems increasingly need to understand the world across multiple senses (not just text), computer vision is a core building block for multimodal models (e.g., GPT-4V, Gemini) that combine visual and textual understanding.

### Q2. What is image segmentation, and what are its applications?

#### Image Segmentation

- **Core idea:** The task of partitioning an image into meaningful regions by classifying **individual pixels**, rather than just labeling the whole image (classification) or drawing bounding boxes (detection) — providing pixel-level precision about object boundaries and shapes.

---

##### Types of Segmentation

- **Semantic segmentation:** Classifies every pixel into a category (e.g., "road," "car," "person"), but doesn't distinguish between different instances of the same class (all cars get the same label, not separated individually).
- **Instance segmentation:** Identifies and separates individual object instances, even of the same class (e.g., distinguishing "car 1" from "car 2"), combining classification with per-object boundary delineation.
- **Panoptic segmentation:** Combines both — labels every pixel with a semantic category AND separates distinct object instances, giving a complete scene understanding.

---

##### Common Approaches

- **Traditional methods:** Thresholding, edge detection, region growing, clustering (e.g., k-means on pixel color/intensity).
- **Deep learning methods:**
  - **FCN (Fully Convolutional Networks):** Adapts CNNs to output pixel-wise predictions instead of a single class label.
  - **U-Net:** Encoder-decoder architecture with skip connections, widely used especially in medical imaging.
  - **Mask R-CNN:** Extends object detection (Faster R-CNN) to also predict a segmentation mask for each detected object — standard for instance segmentation.
  - **Vision Transformer-based models (e.g., SegFormer, Segment Anything Model/SAM):** Use attention mechanisms for improved segmentation, with SAM notably enabling flexible, prompt-based segmentation of arbitrary objects.

---

##### Applications

- **Medical imaging:** Segmenting tumors, organs, or lesions in X-rays/MRIs/CT scans for diagnosis and treatment planning.
- **Autonomous driving:** Segmenting roads, lanes, pedestrians, vehicles for scene understanding and navigation.
- **Satellite/aerial imagery:** Land use classification, deforestation monitoring, urban planning.
- **Agriculture:** Identifying crop boundaries, detecting diseased plants/leaves.
- **Photo/video editing:** Background removal, object isolation for editing (e.g., portrait mode blur, virtual backgrounds).
- **Retail:** Analyzing product placement/shelf inventory via segmented images.
- **Robotics:** Enabling robots to understand object boundaries for manipulation/grasping tasks.

### Q3. What is object detection, and how does it differ from image classification?

#### Object Detection vs. Image Classification

##### Image Classification

- **Task:** Assigns a single label to an **entire image**, identifying what the dominant object/category present is (e.g., "this image contains a dog").
- **Output:** One class label (optionally with a confidence score) per image — no information about location or multiple objects.

##### Object Detection

- **Task:** Identifies **what** objects are present AND **where** they are located within an image — can detect **multiple objects** of different classes in a single image.
- **Output:** For each detected object — a class label + a **bounding box** (coordinates defining a rectangle around the object) + confidence score.

---

##### Key Difference

- **Classification** answers: _"What is in this image?"_ (single global label)
- **Detection** answers: _"What objects are in this image, and where exactly are they?"_ (multiple labeled, localized objects)

---

##### How Object Detection Works (General Approach)

- **Two-stage detectors (e.g., R-CNN, Fast R-CNN, Faster R-CNN):**
  1. Generate candidate regions likely to contain objects (region proposals).
  2. Classify each proposed region and refine its bounding box.

  - More accurate but slower.
- **One-stage detectors (e.g., YOLO, SSD, RetinaNet):**
  - Directly predict class labels and bounding boxes in a single pass over the image (treating detection as a regression problem), without a separate region-proposal step.
  - Faster, suitable for real-time applications, though historically slightly less accurate than two-stage methods (the gap has narrowed significantly in modern versions).
- **Transformer-based detectors (e.g., DETR):** Use a Transformer encoder-decoder to directly predict a set of objects (class + bounding box) end-to-end, removing the need for hand-designed components like anchor boxes or non-max suppression.

---

##### Applications of Object Detection

- Autonomous driving (detecting pedestrians, vehicles, traffic signs).
- Surveillance/security (detecting people, suspicious objects).
- Retail (shelf/inventory monitoring, automated checkout).
- Medical imaging (localizing tumors/abnormalities, not just classifying presence).
- Robotics (identifying and localizing objects for manipulation).

### Q4. What are the steps to build an image recognition system?

#### Steps to Build an Image Recognition System

- **1. Define the problem/task**
  - Clarify the goal: classification, detection, or segmentation.
  - Define classes/categories and success metrics (accuracy, precision/recall, mAP, etc.).
- **2. Collect and prepare data**
  - Gather a labeled dataset (public datasets like ImageNet, COCO, or custom-collected/labeled images).
  - Ensure sufficient data volume and class balance; use tools/annotators for labeling if building custom data.
- **3. Preprocess the data**
  - Resize/normalize images to a consistent input size and pixel value range.
  - Apply **data augmentation** (rotation, flipping, cropping, color jitter) to improve generalization and reduce overfitting.
  - Split data into train/validation/test sets.
- **4. Choose a model architecture**
  - For small datasets/simple tasks: use **transfer learning** with a pretrained CNN (ResNet, EfficientNet) or Vision Transformer (ViT).
  - For detection/segmentation: use architectures like YOLO, Faster R-CNN, Mask R-CNN, or SAM as appropriate.
  - For training from scratch (rare, needs large data): design/select a CNN or ViT architecture suited to the task.
- **5. Train the model**
  - Fine-tune the pretrained model (or train from scratch) on the prepared dataset.
  - Choose appropriate loss function (cross-entropy for classification, IoU-based losses for detection/segmentation).
  - Tune hyperparameters (learning rate, batch size, epochs) using the validation set.
- **6. Evaluate the model**
  - Assess performance on the held-out test set using relevant metrics (accuracy, F1-score, mAP for detection, IoU for segmentation).
  - Analyze failure cases/confusion matrix to identify weaknesses (e.g., specific classes performing poorly).
- **7. Optimize and iterate**
  - Address issues found in evaluation: gather more data for weak classes, adjust augmentation, try different architectures, tune hyperparameters further.
  - Consider techniques like regularization, dropout, or learning rate scheduling to improve generalization.
- **8. Optimize for deployment**
  - Apply model compression techniques if needed (quantization, pruning, distillation) for faster inference/smaller size, especially for edge/mobile deployment.
  - Convert model to deployment-friendly formats (ONNX, TensorRT, TensorFlow Lite) as needed.
- **9. Deploy the model**
  - Integrate into the target application (mobile app, web service, embedded device).
  - Set up an inference pipeline (API endpoint, batch processing, or on-device inference).
- **10. Monitor and maintain**
  - Track real-world performance post-deployment (accuracy drift, edge cases).
  - Periodically retrain/update the model with new data to handle data drift or expand capabilities.

### Q5. What are the challenges in real-time object tracking?

#### Challenges in Real-Time Object Tracking

- **Occlusion**
  - Objects can be partially or fully blocked by other objects/obstacles temporarily, causing the tracker to lose the object or confuse it with another when it reappears.
- **Real-time speed/computational constraints**
  - Tracking must run fast enough (high FPS) for real-time applications (e.g., autonomous driving, surveillance), creating a trade-off between accuracy (complex models) and speed (lightweight models), especially on edge/embedded devices with limited compute.
- **Scale and appearance variation**
  - Objects change size (moving closer/farther from camera) and appearance (lighting changes, rotation, deformation — e.g., a person changing pose) over time, making consistent identification difficult.
- **Identity switches (ID switches)**
  - When multiple similar-looking objects are close together or cross paths, the tracker may incorrectly swap their identities/IDs.
- **Fast motion/motion blur**
  - Rapidly moving objects can cause significant frame-to-frame displacement or blur, making it hard to associate detections across frames reliably.
- **Cluttered backgrounds / similar-looking objects**
  - Distinguishing a tracked object from visually similar background elements or other objects of the same class (e.g., tracking one specific person in a crowd) is challenging.
- **Illumination and environmental changes**
  - Changing lighting conditions (day/night, shadows, weather) can alter object appearance significantly, affecting detection/tracking consistency.
- **Multi-object tracking (MOT) complexity**
  - Tracking many objects simultaneously requires solving a data association problem (matching detections across frames to existing tracks) efficiently, which grows more complex as object count increases.
- **New object entry/object exit (track initiation & termination)**
  - Deciding when a new object should start being tracked (avoiding false positives) and when a tracked object has truly left the scene (avoiding lingering "ghost" tracks) requires careful handling.
- **Camera motion (non-static camera scenarios)**
  - When the camera itself moves (e.g., drone, vehicle-mounted camera), separating object motion from camera motion adds complexity to tracking algorithms.
- **Long-term tracking drift**
  - Errors can accumulate over time in tracking algorithms that rely on frame-to-frame updates, causing the tracker to gradually drift away from the true object location.
- **Common solutions/approaches to address these:**
  - Combining **detection + tracking** (tracking-by-detection paradigm) using strong object detectors (YOLO) with tracking algorithms (SORT, DeepSORT, ByteTrack) that use motion prediction (Kalman filters) plus appearance features (re-identification embeddings) to handle occlusion and ID switches better.
  - Using **Kalman filters** or **particle filters** for motion prediction during brief occlusions.
  - Leveraging **re-identification (Re-ID) embeddings** to re-match objects after occlusion based on appearance features, not just spatial proximity.

### Q6. What is feature extraction in computer vision?

#### Feature Extraction in Computer Vision

- **Core idea:** The process of transforming raw pixel data (high-dimensional, often noisy) into a more compact, informative representation that captures the important visual patterns needed for a given task (classification, detection, etc.), while discarding irrelevant/redundant information.
- **Why it's needed:** Raw pixels alone don't directly convey meaningful structure (shapes, edges, textures, objects) in a form that's easy for models to use — feature extraction bridges this gap by highlighting discriminative visual patterns.

---

##### Traditional (Hand-Crafted) Feature Extraction Methods

- **Edge detection:** Identifies boundaries/edges in an image (e.g., **Sobel**, **Canny** edge detectors) based on sharp intensity changes.
- **Corner detection:** Identifies keypoints/corners in an image (e.g., **Harris Corner Detector**) useful for tracking and matching across images.
- **SIFT (Scale-Invariant Feature Transform):** Extracts keypoints and descriptors that are invariant to scale, rotation, and partially invariant to illumination changes — widely used for object recognition and image matching.
- **HOG (Histogram of Oriented Gradients):** Captures the distribution of gradient orientations in localized regions of an image, historically effective for pedestrian/object detection.
- **SURF, ORB:** Faster alternatives to SIFT for keypoint detection and description.
- **Color histograms/texture features (e.g., GLCM):** Capture color distribution or texture patterns within an image.

---

##### Deep Learning-Based Feature Extraction

- **Convolutional Neural Networks (CNNs):** Automatically learn hierarchical features directly from data during training — early layers learn simple features (edges, colors), middle layers learn textures/patterns, and deeper layers learn complex, task-relevant features (object parts, whole objects).
- **Pretrained CNNs as feature extractors:** Common practice is to use a CNN pretrained on a large dataset (e.g., ResNet pretrained on ImageNet), remove the final classification layer, and use the output of an intermediate/penultimate layer as a general-purpose feature vector for a new task (this is essentially transfer learning applied via feature extraction).
- **Vision Transformers (ViT):** Extract features by treating image patches as tokens and applying self-attention, learning global relationships between different parts of the image from the start, rather than the local-to-global hierarchy CNNs build via stacked convolutions.

---

##### Key Difference: Traditional vs. Deep Learning Approaches

- **Traditional methods:** Features are manually designed by domain experts based on prior knowledge of what patterns might be useful (hand-crafted) — interpretable but limited in capturing complex patterns and require task-specific tuning.
- **Deep learning methods:** Features are **learned automatically** from data during training — far more powerful and adaptable across tasks, but less interpretable and require more data/compute.

---

##### Applications

- Used as an intermediate step (or standalone output) for downstream tasks: image classification, object detection, image retrieval/similarity search, face recognition, image matching/stitching (panorama creation), and as embeddings feeding into other models (e.g., multimodal systems combining vision and text features).

### Q7. What is OCR, and what are its main applications?

#### OCR (Optical Character Recognition)

- **Core idea:** A computer vision technology that detects and extracts printed or handwritten text from images or scanned documents, converting it into a machine-readable/editable text format.

---

##### How It Works (General Pipeline)

- **Preprocessing:** Image cleanup — noise removal, binarization (converting to black/white), skew correction (straightening tilted text), contrast enhancement.
- **Text detection/localization:** Identifying regions in the image that contain text (bounding boxes around lines/words), often using object detection-style models.
- **Character/text segmentation:** Separating text into individual characters, words, or lines for recognition.
- **Character recognition:** Classifying each detected character/word into actual text — traditionally via pattern matching or feature-based classifiers; modern systems use deep learning (CNNs for feature extraction + RNN/Transformer-based sequence models like CRNN, or end-to-end Transformer OCR models).
- **Post-processing:** Applying language models/spell-checking to correct recognition errors and improve output accuracy based on context.

---

##### Modern Approaches

- **CRNN (Convolutional Recurrent Neural Network):** Combines CNN (feature extraction) + RNN/LSTM (sequence modeling) + CTC (Connectionist Temporal Classification) loss to handle variable-length text without needing explicit character segmentation.
- **Transformer-based OCR (e.g., TrOCR):** Uses a Vision Transformer encoder + Transformer decoder for end-to-end text recognition, achieving strong performance especially on complex/handwritten text.
- **End-to-end multimodal models:** Modern large vision-language models (e.g., GPT-4V) can perform OCR-like text extraction as part of broader visual understanding, without a dedicated separate OCR pipeline.

---

##### Main Applications

- **Document digitization:** Converting scanned paper documents, books, or historical archives into searchable, editable digital text.
- **Data entry automation:** Extracting information from forms, invoices, receipts, and ID cards to automate data entry (e.g., invoice processing in finance/accounting).
- **License plate recognition:** Used in traffic monitoring, toll systems, and parking management.
- **Handwriting recognition:** Digitizing handwritten notes, historical manuscripts, or postal address recognition for mail sorting.
- **Accessibility tools:** Reading text aloud for visually impaired users (e.g., text-to-speech apps using OCR to first extract text from images/signs).
- **Translation apps:** Real-time translation of text captured via camera (e.g., translating restaurant menus or street signs).
- **Search and indexing:** Making scanned/image-based content searchable in databases or document management systems.
- **Banking/finance:** Check processing, extracting data from financial documents and KYC (Know Your Customer) verification documents.

### Q8. How does a CNN differ from traditional neural networks in computer vision?

#### CNN vs. Traditional Neural Networks (Fully Connected/Dense Networks) in Computer Vision

- **Connectivity pattern:**
  - **Traditional NN (fully connected):** Every neuron connects to every neuron in the next layer — for images, this means flattening the image into a 1D vector, losing spatial structure entirely.
  - **CNN:** Uses **local connectivity** — each neuron connects only to a small local region (receptive field) of the input, preserving spatial relationships between pixels.
- **Parameter sharing:**
  - **Traditional NN:** Each connection has its own unique weight — for images, this leads to an enormous number of parameters (e.g., a 224x224x3 image flattened would need millions of weights just for one layer).
  - **CNN:** Uses **shared filters/kernels** that slide across the entire image — the same small set of weights (filter) is reused at every spatial location, drastically reducing parameter count.
- **Spatial structure preservation:**
  - **Traditional NN:** Flattening destroys 2D spatial relationships (neighboring pixels lose their proximity information).
  - **CNN:** Processes images as 2D (or 3D with channels) grids throughout, explicitly preserving spatial relationships between pixels/features.
- **Translation invariance:**
  - **Traditional NN:** No inherent invariance — if an object shifts position in the image, the network sees a completely different input pattern.
  - **CNN:** Naturally achieves translation invariance/equivariance — a filter detecting an edge or pattern can recognize it regardless of where it appears in the image, due to shared weights sliding across the image (further reinforced by pooling layers).
- **Hierarchical feature learning:**
  - **Traditional NN:** No built-in mechanism for hierarchical feature composition — features are learned globally without a structured hierarchy.
  - **CNN:** Naturally builds a hierarchy through stacked layers — early layers learn simple features (edges, colors), middle layers learn textures/shapes, deeper layers learn complex object parts/whole objects.
- **Computational efficiency:**
  - **Traditional NN:** Extremely inefficient for images due to massive parameter counts from full connectivity — impractical for any reasonably sized image.
  - **CNN:** Far more efficient — parameter sharing and local connectivity make it feasible to train on high-resolution images with manageable computational/memory requirements.
- **Pooling layers (CNN-specific):**
  - CNNs typically include **pooling layers** (max pooling, average pooling) to progressively reduce spatial dimensions, providing some robustness to small translations/distortions and reducing computation — traditional NNs have no equivalent concept.
- **Performance on visual tasks:**
  - Due to all the above properties, CNNs dramatically outperform traditional fully connected networks on image-related tasks (classification, detection, segmentation), which is why CNNs became the dominant architecture in computer vision until Vision Transformers emerged as a competitive alternative.

### Q9. What is data augmentation, and what techniques are commonly used?

#### Data Augmentation

- **Core idea:** Artificially expanding a training dataset by applying transformations to existing data, creating new variations without collecting additional labeled samples — helps improve model generalization and reduce overfitting.
- **Why it's used:** Deep learning models need large amounts of data; augmentation increases effective dataset size/diversity, makes models more robust to real-world variations, and helps with limited/imbalanced datasets.

---

##### Common Techniques (Images)

- **Geometric transformations:** Rotation, flipping (horizontal/vertical), cropping, scaling, translation, shearing.
- **Color/photometric transformations:** Brightness, contrast, saturation, hue adjustments; color jitter.
- **Noise injection:** Adding random Gaussian noise or blur to simulate real-world imperfections.
- **Cutout/Random Erasing:** Randomly masking/removing rectangular patches from an image to force the model to learn from partial information.
- **Mixup:** Blending two images (and their labels) together in weighted proportions.
- **CutMix:** Cutting a patch from one image and pasting it onto another, mixing labels proportionally to patch area.
- **Random affine/perspective transforms:** Simulating different viewing angles.
- **GAN/diffusion-based augmentation:** Generating entirely new synthetic samples using generative models.

---

##### Common Techniques (Text/NLP)

- **Synonym replacement:** Swapping words with synonyms.
- **Back-translation:** Translating text to another language and back to generate paraphrased variants.
- **Random insertion/deletion/swap:** Randomly modifying words in a sentence.
- **Contextual augmentation:** Using language models (e.g., BERT) to substitute words based on context.

---

##### Other Domains

- **Audio:** Pitch shifting, time stretching, adding background noise, time masking.
- **Time series:** Jittering, scaling, window slicing/warping.

---

##### Key Considerations

- Augmentations should be **label-preserving** (e.g., flipping a "6" horizontally could turn it into something resembling a different digit — inappropriate for digit recognition).
- Often applied **on-the-fly during training** (randomly per batch) rather than precomputing a fixed augmented dataset, maximizing variety across epochs.

### Q10. What are some popular deep learning frameworks for computer vision?

#### Popular Deep Learning Frameworks for Computer Vision

- **PyTorch**
  - Most widely used in research and increasingly in production; dynamic computation graph, intuitive Pythonic syntax.
  - Rich ecosystem: **torchvision** (pretrained models, datasets, transforms), **Detectron2** (Facebook AI's detection/segmentation library), **timm** (large collection of pretrained vision models).
- **TensorFlow**
  - Widely used in production/industry settings; supports static and dynamic graphs (via Keras and eager execution).
  - **Keras** (high-level API) simplifies model building; **TensorFlow Hub** provides pretrained vision models; **TF Lite** for mobile/edge deployment.
- **Keras**
  - High-level API, now integrated with TensorFlow, but also has multi-backend support (TensorFlow, JAX, PyTorch as of Keras 3) — easy to use for quick prototyping.
- **OpenCV**
  - Not a deep learning framework per se, but the standard library for classical computer vision operations (image processing, filtering, feature extraction, camera calibration) — often used alongside deep learning frameworks for preprocessing/postprocessing.
- **JAX**
  - Google's framework focused on high-performance numerical computing with automatic differentiation; gaining traction in research for its speed and functional programming style, used in some cutting-edge vision research.
- **MXNet**
  - Used by AWS, supports multiple languages, though less popular now compared to PyTorch/TensorFlow.
- **Caffe / Caffe2**
  - Older framework, historically popular for CNNs, largely succeeded by PyTorch (Caffe2 was merged into PyTorch).
- **MMDetection / MMSegmentation (OpenMMLab)**
  - Popular open-source toolboxes built on PyTorch, providing many pretrained state-of-the-art models specifically for object detection and segmentation tasks.
- **Hugging Face Transformers**
  - Originally NLP-focused, now also provides many pretrained vision and multimodal models (ViT, CLIP, DETR, SAM) with easy-to-use APIs.
- **Ultralytics (YOLO)**
  - Popular library providing easy-to-use implementations of YOLO models for real-time object detection.
- **Common choice today:** **PyTorch** dominates research and is increasingly standard in production too, often paired with **Hugging Face** for pretrained models and **OpenCV** for classical image processing utilities.

### Q11. How can Transformers be used for computer vision tasks?

#### Transformers for Computer Vision

- **Vision Transformer (ViT):** Splits an image into fixed-size patches (e.g., 16x16), flattens and linearly embeds each patch as a "token" (analogous to word tokens in NLP), adds positional embeddings, and feeds the sequence into a standard Transformer encoder.
- **Classification token (`[CLS]`):** A learnable token is prepended to the patch sequence; its final representation is used for image classification via a simple head.
- **Self-attention for global context:** Unlike CNNs (local receptive fields, need many layers to see the whole image), self-attention lets every patch attend to every other patch from the very first layer — capturing global relationships early.
- **Data requirements:** ViTs lack CNNs' built-in inductive biases (locality, translation invariance), so they typically need large-scale pretraining data (e.g., JFT-300M, ImageNet-21k) to match/exceed CNN performance; they perform less well on small datasets without this pretraining.
- **Hybrid approaches:** Combine CNN backbones (for local feature extraction) with Transformer layers (for global context) — e.g., **DETR** (CNN + Transformer for object detection).
- **Task-specific adaptations:**
  - **Object detection:** **DETR** — Transformer encoder-decoder directly predicts object sets (class + bounding box), removing hand-crafted components like anchors/NMS.
  - **Segmentation:** **SegFormer**, **Mask2Former**, **Segment Anything Model (SAM)** — use Transformer-based architectures for pixel-level and prompt-based segmentation.
  - **Video understanding:** **VideoMAE**, **TimeSformer** — extend attention across both spatial and temporal dimensions.
- **Multimodal integration:** Transformers naturally unify vision and language (e.g., **CLIP**, **GPT-4V**) since both modalities can be tokenized and processed with the same attention mechanism, enabling joint vision-language tasks (captioning, VQA, zero-shot classification).
- **Efficiency variants:** **Swin Transformer** (hierarchical, shifted-window attention) reduces the quadratic cost of full self-attention, making Transformers more practical for high-resolution vision tasks and denser prediction tasks (detection/segmentation).

### Q12. What is a world model?

A **world model** is an internal representation an agent (biological or artificial) builds to predict how its environment behaves — essentially a mental simulation of "how the world works" that lets it anticipate outcomes without having to physically try everything.

**The core idea**

Instead of reacting only to immediate inputs, a system with a world model can answer questions like:

- "If I do X, what will happen next?"
- "What does the world look like right now, even the parts I can't currently see?"
- "What's likely to happen if I do nothing?"

This is the difference between a thermostat (pure reaction) and a chess player (mentally simulating several moves ahead before acting).

**Where the term shows up**

1. **Cognitive science/psychology** — Humans and animals seem to build internal models of physics, other people's intentions (theory of mind), and spatial layouts, which we use for planning and imagination.
2. **Robotics and reinforcement learning** — A "world model" is a learned function that predicts the next state of the environment given the current state and an action. An agent can then "imagine" trajectories inside this model and plan or train without needing real-world trial and error (e.g., Ha & Schmidhuber's 2018 "World Models" paper, or DeepMind's Dreamer/MuZero-style agents).
3. **Modern AI / LLM discourse** — More recently, people ask whether large language models have a "world model" — i.e., whether they've learned an implicit representation of physical, causal, or social reality (as opposed to just learning surface statistical patterns of text). This is a hot and unresolved debate: skeptics argue LLMs often make basic physical or spatial errors that suggest a shallow or inconsistent world model; others point to emergent behaviors (like tracking game board states from text) as evidence of something world-model-like forming internally.
4. **Video/world generation models** — Systems like Genie, Sora, or Google's Genie 2/World Labs' models are sometimes called "world models" because they generate consistent, interactive simulated environments frame-by-frame, effectively learning the rules of a visual world.

**Why it matters**

A good world model lets an agent:

- Plan by simulating consequences mentally (cheaper/safer than acting in reality)
- Generalize to new situations it hasn't directly experienced
- Fill in missing information (e.g., predicting what's behind you even though you can't see it)

### Q13. what JEPA?

**JEPA** (Joint Embedding Predictive Architecture) is a self-supervised learning framework proposed by Yann LeCun (Meta's former Chief AI Scientist) as a candidate architecture for building AI systems with actual world models — his answer to "how do we get machines to understand the world the way humans and animals do, without needing millions of labeled examples or exhaustive trial and error."

**The core idea**

Instead of predicting raw pixels or tokens (like generative models such as GANs, VAEs, or autoregressive LLMs do), JEPA predicts in an **abstract embedding (representation) space**.

Concretely:

1. Take an input (e.g., an image or video) and mask or hide part of it.
2. Encode the visible part into an embedding.
3. Encode the hidden/target part into another embedding (using a separate encoder).
4. Train a **predictor** to guess the target embedding from the context embedding — not the raw pixels themselves.

**Why predict in embedding space instead of pixel space?**

LeCun's argument: predicting raw pixels forces the model to waste capacity on irrelevant, unpredictable details (exact leaf positions in windy grass, video noise, texture minutiae). By predicting a compressed, abstract representation instead, the model is encouraged to throw away noise and irrelevant detail and focus on the predictable, semantic structure of the world — which is closer to what "understanding" looks like.

This also sidesteps a big failure mode of generative approaches: they can waste enormous effort trying to generate exact pixel-perfect detail for inherently unpredictable things.

**Key implementations**

- **I-JEPA** (Image JEPA, 2023) — learns image representations by predicting embeddings of masked regions from visible regions of the same image.
- **V-JEPA / V-JEPA 2** — extends this to video, learning to predict masked spatiotemporal regions, aimed at capturing physical dynamics and intuitive physics (e.g., object permanence, gravity) from watching video, without labels.

**How it fits LeCun's broader vision**

JEPA is one piece of his proposed roadmap toward more human/animal-like AI (sometimes called "Autonomous Machine Intelligence" or AMI), alongside:

- A **world model** (JEPA-based) for prediction
- A **cost/critic module** to evaluate action outcomes
- An **actor** that plans by simulating action sequences through the world model
- A hierarchical planning structure, rather than the reactive/autoregressive nature of current LLMs

LeCun has been fairly vocal that he thinks pure autoregressive LLMs are a limited path toward general intelligence, and JEPA-style architectures are his bet on a better route toward machines with genuine predictive world models.

Want me to go into how V-JEPA 2 specifically performs on physical reasoning benchmarks, or how JEPA compares architecturally to diffusion/generative video models?
