# Chapter 15. Computer Vision

_Part III — Machine Learning_

CNNs, modern vision backbones, detection and segmentation, augmentation and vision pipelines.

This chapter contains **47 questions**.

## Contents

- **15.1** [What is computer vision, and why is it important?](#151-what-is-computer-vision-and-why-is-it-important)
- **15.2** [What is image segmentation, and what are its applications?](#152-what-is-image-segmentation-and-what-are-its-applications)
- **15.3** [What is object detection, and how does it differ from image classification?](#153-what-is-object-detection-and-how-does-it-differ-from-image-classification)
- **15.4** [What are the steps to build an image recognition system?](#154-what-are-the-steps-to-build-an-image-recognition-system)
- **15.5** [What are the challenges in real-time object tracking?](#155-what-are-the-challenges-in-real-time-object-tracking)
- **15.6** [What is feature extraction in computer vision?](#156-what-is-feature-extraction-in-computer-vision)
- **15.7** [What is OCR, and what are its main applications?](#157-what-is-ocr-and-what-are-its-main-applications)
- **15.8** [How does a CNN differ from traditional neural networks in computer vision?](#158-how-does-a-cnn-differ-from-traditional-neural-networks-in-computer-vision)
- **15.9** [What is data augmentation, and what techniques are commonly used?](#159-what-is-data-augmentation-and-what-techniques-are-commonly-used)
- **15.10** [What are some popular deep learning frameworks for computer vision?](#1510-what-are-some-popular-deep-learning-frameworks-for-computer-vision)
- **15.11** [How can Transformers be used for computer vision tasks?](#1511-how-can-transformers-be-used-for-computer-vision-tasks)
- **15.12** [What is a world model?](#1512-what-is-a-world-model)
- **15.13** [what JEPA?](#1513-what-jepa)
- **15.14** [Compare SAM and YOLO.](#1514-compare-sam-and-yolo)
- **15.15** [How would you approach building an object detection pipeline for a client, and what tools would you use?](#1515-how-would-you-approach-building-an-object-detection-pipeline-for-a-client-and-what-tools-would-you-use)
- **15.16** [What's the difference between object detection, image classification, and segmentation, and when would you use each?](#1516-whats-the-difference-between-object-detection-image-classification-and-segmentation-and-when-would-you-use-each)
- **15.17** [Why do CNNs work well for images?](#1517-why-do-cnns-work-well-for-images)
- **15.18** [Explain convolution, stride, padding, dilation, and receptive field.](#1518-explain-convolution-stride-padding-dilation-and-receptive-field)
- **15.19** [What is the difference between same and valid convolution?](#1519-what-is-the-difference-between-same-and-valid-convolution)
- **15.20** [How does pooling provide translation invariance?](#1520-how-does-pooling-provide-translation-invariance)
- **15.21** [Why did ResNet outperform many earlier CNN architectures?](#1521-why-did-resnet-outperform-many-earlier-cnn-architectures)
- **15.22** [Compare ResNet, EfficientNet, Vision Transformer, and ConvNeXt.](#1522-compare-resnet-efficientnet-vision-transformer-and-convnext)
- **15.23** [Explain IoU, precision, recall, and mAP for object detection.](#1523-explain-iou-precision-recall-and-map-for-object-detection)
- **15.24** [How does NMS work?](#1524-how-does-nms-work)
- **15.25** [What is data augmentation, and why does it improve generalization?](#1525-what-is-data-augmentation-and-why-does-it-improve-generalization)
- **15.26** [When can augmentation actually hurt model performance?](#1526-when-can-augmentation-actually-hurt-model-performance)
- **15.27** [Explain the architecture of CLIP.](#1527-explain-the-architecture-of-clip)
- **15.28** [Why does CLIP use contrastive learning?](#1528-why-does-clip-use-contrastive-learning)
- **15.29** [How would you use CLIP for zero-shot image classification?](#1529-how-would-you-use-clip-for-zero-shot-image-classification)
- **15.30** [What is ImageNet?](#1530-what-is-imagenet)
- **15.31** [What is AlexNet and why is it important?](#1531-what-is-alexnet-and-why-is-it-important)
- **15.32** [How does a Vision Transformer (ViT) work?](#1532-how-does-a-vision-transformer-vit-work)
- **15.33** [How do two-stage and one-stage object detectors differ, and what are anchor-based and anchor-free detectors?](#1533-how-do-two-stage-and-one-stage-object-detectors-differ-and-what-are-anchor-based-and-anchor-free-detectors)
- **15.34** [How does DETR perform object detection with Transformers, and what problems did later variants solve?](#1534-how-does-detr-perform-object-detection-with-transformers-and-what-problems-did-later-variants-solve)
- **15.35** [What are semantic, instance and panoptic segmentation, and which architectures are used for each?](#1535-what-are-semantic-instance-and-panoptic-segmentation-and-which-architectures-are-used-for-each)
- **15.36** [Which loss functions and metrics are used for image segmentation, and when do you choose each?](#1536-which-loss-functions-and-metrics-are-used-for-image-segmentation-and-when-do-you-choose-each)
- **15.37** [How do you detect small objects and handle severe class imbalance in object detection?](#1537-how-do-you-detect-small-objects-and-handle-severe-class-imbalance-in-object-detection)
- **15.38** [What is open-vocabulary detection and segmentation, and how do models like Grounding DINO and SAM 2 work?](#1538-what-is-open-vocabulary-detection-and-segmentation-and-how-do-models-like-grounding-dino-and-sam-2-work)
- **15.39** [How do multi-object tracking algorithms such as SORT, DeepSORT and ByteTrack work?](#1539-how-do-multi-object-tracking-algorithms-such-as-sort-deepsort-and-bytetrack-work)
- **15.40** [How would you debug a computer vision model that performs well offline but poorly in production?](#1540-how-would-you-debug-a-computer-vision-model-that-performs-well-offline-but-poorly-in-production)
- **15.41** [How does self-supervised learning work for vision, and how do contrastive, distillation-based and masked-image methods compare?](#1541-how-does-self-supervised-learning-work-for-vision-and-how-do-contrastive-distillation-based-and-masked-image-methods-compare)
- **15.42** [How do you fine-tune a vision model when you have very little labelled data?](#1542-how-do-you-fine-tune-a-vision-model-when-you-have-very-little-labelled-data)
- **15.43** [How do classical feature detectors such as Harris, SIFT and ORB work, and when does classical computer vision still beat deep learning?](#1543-how-do-classical-feature-detectors-such-as-harris-sift-and-orb-work-and-when-does-classical-computer-vision-still-beat-deep-learning)
- **15.44** [How do monocular depth estimation, NeRF and 3D Gaussian Splatting work, and when would you use each?](#1544-how-do-monocular-depth-estimation-nerf-and-3d-gaussian-splatting-work-and-when-would-you-use-each)
- **15.45** [How do video understanding models capture motion and time?](#1545-how-do-video-understanding-models-capture-motion-and-time)
- **15.46** [How do you optimise and deploy a computer vision model on edge devices?](#1546-how-do-you-optimise-and-deploy-a-computer-vision-model-on-edge-devices)
- **15.47** [How do you design a labelling strategy for a computer vision project, including active learning and handling label noise?](#1547-how-do-you-design-a-labelling-strategy-for-a-computer-vision-project-including-active-learning-and-handling-label-noise)

---

### 15.1 What is computer vision, and why is it important?

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

### 15.2 What is image segmentation, and what are its applications?

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

### 15.3 What is object detection, and how does it differ from image classification?

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

### 15.4 What are the steps to build an image recognition system?

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

### 15.5 What are the challenges in real-time object tracking?

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

### 15.6 What is feature extraction in computer vision?

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

### 15.7 What is OCR, and what are its main applications?

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

### 15.8 How does a CNN differ from traditional neural networks in computer vision?

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

### 15.9 What is data augmentation, and what techniques are commonly used?

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

### 15.10 What are some popular deep learning frameworks for computer vision?

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

### 15.11 How can Transformers be used for computer vision tasks?

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

### 15.12 What is a world model?

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

### 15.13 what JEPA?

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

### 15.14 Compare SAM and YOLO.

They solve different problems and are often combined rather than competing.

| | **YOLO** (You Only Look Once) | **SAM** (Segment Anything Model, Meta) |
| --- | --- | --- |
| Task | Object **detection** (boxes + class labels); newer versions add instance segmentation | Promptable **segmentation** (pixel masks), class-agnostic |
| Input | Image | Image + prompt (points, boxes, or masks; SAM 2 also handles video) |
| Output | Boxes, classes, confidences | High-quality masks, no class names |
| Training | Supervised on a fixed label set (e.g., COCO); fine-tune for new classes | Trained on SA-1B (1B+ masks); zero-shot on new objects |
| Speed | Real-time, edge-friendly (single-stage CNN) | Heavy ViT image encoder; the mask decoder is light, but encoding is the bottleneck (MobileSAM / EfficientSAM help) |

**When to use which**

- Need to know *what* and *where* fast (counting cars, safety-gear detection on a camera feed): **YOLO**.
- Need precise object outlines with no task-specific labels (annotation tooling, medical or industrial masks, background removal): **SAM**.

**Common production pattern:** YOLO (or Grounding DINO for open-vocabulary text prompts) proposes boxes, and SAM turns each box into a precise mask. This "detect then segment" pipeline is also a fast way to bootstrap segmentation labels: SAM-assisted labeling cuts annotation time a lot, and a human only corrects the masks.

**Trade-off:** YOLO needs labeled data for your classes but runs cheaply. SAM is zero-shot for shapes but gives you no semantics and costs more compute per image.

### 15.15 How would you approach building an object detection pipeline for a client, and what tools would you use?

Clarify the requirement first: real-time vs. batch, accuracy vs. latency tradeoffs, classes needed (pretrained-sufficient vs. needs fine-tuning), and deployment target (edge vs. cloud). Typical stack: YOLO (v8/v11) for real-time detection given its strong speed/accuracy tradeoff, OpenCV for pre/post-processing (resizing, NMS, drawing boxes), and PyTorch for any custom training/fine-tuning on client-specific data. If classes are non-standard, fine-tune on a labeled dataset (possibly bootstrapped via a foundation vision model or semi-supervised labeling). Package the trained model into a Docker container with a serving layer (e.g., TorchServe, FastAPI + ONNX Runtime, or Triton) for deployment.

### 15.16 What's the difference between object detection, image classification, and segmentation, and when would you use each?

Classification assigns a single label to the whole image (e.g., "this image contains a defect"). Object detection localizes and classifies multiple objects with bounding boxes (e.g., "3 defects at these coordinates"). Segmentation goes further — pixel-level masks, either semantic (label each pixel's class) or instance (separate each object instance). Use classification when only presence/absence matters, detection when you need location/counting, and segmentation when precise shape/area matters (e.g., measuring defect size, medical imaging).

### 15.17 Why do CNNs work well for images?

CNNs encode **priors that match the structure of natural images**:

- **Locality** — a pixel is most related to its neighbours, so small kernels (3×3) look at local patches first.
- **Weight sharing** — the same filter slides over the image, so an edge detector learned in one place works everywhere. This gives **translation equivariance** and cuts parameters massively (a 3×3×64×64 conv has ~37k weights regardless of image size, versus millions for a dense layer).
- **Hierarchy** — stacking layers grows the receptive field, so features compose: edges → textures → parts → objects.
- **Pooling / striding** — adds some tolerance to small shifts and reduces compute.

**Tradeoffs:** these inductive biases make CNNs very **data-efficient**, which is why they beat Vision Transformers on small/medium datasets. ViTs have weaker priors and need more data or pretraining, but scale better and model global context earlier. Modern designs (ConvNeXt, hybrid CNN-ViT) borrow from both.

**Production angle:** CNNs are still the default for edge and real-time vision (YOLO, MobileNet, EfficientNet) because convolution is extremely well optimised on GPUs, NPUs and mobile accelerators.

### 15.18 Explain convolution, stride, padding, dilation, and receptive field.

- **Convolution** — slide a learned kernel (e.g. $3 \times 3 \times C_{\text{in}}$) over the input and take a dot product at each position, producing one output channel per kernel. It detects the same local pattern everywhere.
- **Stride** — how far the kernel moves each step. Stride 2 halves the spatial resolution (cheap downsampling).
- **Padding** — adds border pixels (usually zeros) so the kernel can be centred on edge pixels and the output size is controlled.
- **Dilation** — spaces out kernel taps (dilation 2 means a 3×3 kernel covers a 5×5 area). It enlarges the receptive field without more parameters or downsampling; used in segmentation (DeepLab) and WaveNet.
- **Receptive field** — the region of the input that affects one output unit. It grows with depth, kernel size, stride and dilation.

**Output size:**

$$
O = \left\lfloor \frac{I + 2P - D(K-1) - 1}{S} \right\rfloor + 1
$$

**Receptive field** (recursive):

$$
r_l = r_{l-1} + (K_l - 1)\,D_l \prod_{i<l} S_i
$$

**Example:** three stacked 3×3 convs (stride 1) have a 7×7 receptive field with fewer parameters and more nonlinearity than one 7×7 conv, which is the VGG insight.

**Practical note:** the *effective* receptive field is smaller than the theoretical one (centre pixels dominate), which is one reason global-context modules and attention help on large objects.

### 15.19 What is the difference between same and valid convolution?

- **Valid convolution** — no padding. The kernel only sits where it fully fits, so the output shrinks: $O = I - K + 1$ (stride 1). A 32×32 input with a 3×3 kernel gives 30×30.
- **Same convolution** — pad so the output has the **same spatial size** as the input (stride 1). For odd kernel $K$, pad $P = (K-1)/2$ on each side (3×3 → pad 1).

**Tradeoffs:**

- *Valid* has no artificial border values, but features shrink every layer, which limits depth and loses border information.
- *Same* keeps shapes stable, which makes deep stacks and skip connections (ResNet, U-Net) easy to wire, at the cost of edge artefacts from the zero padding. Reflect or replicate padding can reduce those.

```python
import torch.nn as nn
nn.Conv2d(64, 64, kernel_size=3, padding=0)       # valid
nn.Conv2d(64, 64, kernel_size=3, padding="same")  # same (stride 1)
```

Note that with stride > 1, "same" in TensorFlow means $O = \lceil I/S \rceil$, and the padding can be asymmetric, which is a common source of PyTorch/TF porting bugs.

### 15.20 How does pooling provide translation invariance?

Strictly, pooling gives only **local, approximate** translation invariance.

- Convolution is **equivariant**: shift the input and the feature map shifts by the same amount.
- **Max pooling** takes the maximum over a window (e.g. 2×2). If a feature moves by a pixel but stays inside the same window, the pooled output is unchanged. So small shifts are absorbed.
- Stacking conv + pool layers compounds this, and **global average pooling** at the end discards position entirely, so the classifier sees "is the feature present" rather than "where".

**Caveats worth raising in an interview:**

- Invariance is only to small shifts. Strided pooling causes **aliasing**, and a 1-pixel shift can noticeably change CNN outputs (Zhang 2019, "Making CNNs Shift-Invariant Again" fixes this with blur-pooling).
- Invariance is not always wanted. Detection and segmentation need location, so they rely on skip connections or avoid aggressive pooling.
- Most modern robustness to translation actually comes from **data augmentation** (random crops and shifts) and global pooling, not from the pooling layers themselves.

### 15.21 Why did ResNet outperform many earlier CNN architectures?

**Problem:** before ResNet, making plain networks deeper (beyond ~20 layers) made **training** error worse. This is the *degradation problem*: an optimisation difficulty, not overfitting.

**Key idea — residual learning:** each block learns a residual $F(x)$ and outputs $y = F(x) + x$ through an identity skip connection.

- If the best mapping is close to identity, the block only has to push $F(x)$ towards 0, which is easy.
- Gradients flow straight through the identity path ($\partial y/\partial x = \partial F/\partial x + I$), which eases vanishing gradients and lets 100+ (even 1000) layer networks train.
- The network behaves like an ensemble of many shallower paths, which smooths the loss landscape.

**Other ingredients:** BatchNorm after every conv, bottleneck blocks (1×1 → 3×3 → 1×1) for cheap depth, and global average pooling instead of huge FC layers.

**Result:** ResNet-152 won ImageNet 2015 with 3.57% top-5 error, with fewer FLOPs than VGG.

**Legacy:** skip connections are now everywhere: Transformers (residual stream), U-Net, DenseNet and diffusion models. ResNet-50 remains a standard production and benchmark backbone because it is simple and well supported by hardware.

### 15.22 Compare ResNet, EfficientNet, Vision Transformer, and ConvNeXt.

| Model | Core idea | Strengths | Weaknesses |
| --- | --- | --- | --- |
| **ResNet** (2015) | Residual blocks with skip connections | Simple, robust, well optimised everywhere, good transfer | Less accurate per FLOP than newer models |
| **EfficientNet** (2019) | Compound scaling of depth, width and resolution; MBConv + squeeze-excitation | Excellent accuracy per parameter/FLOP; good for mobile | Depthwise convs are memory-bound, so it is often slower on GPU than FLOPs suggest; harder to train |
| **Vision Transformer (ViT)** (2020) | Split image into 16×16 patches, treat them as tokens, apply a standard Transformer | Global context from layer 1, scales very well with data, shares stack with LLMs (CLIP, multimodal) | Weak inductive bias, so it needs large pretraining or heavy augmentation; quadratic cost in patches at high resolution |
| **ConvNeXt** (2022) | A ResNet "modernised" with Transformer design choices: 7×7 depthwise convs, LayerNorm, GELU, inverted bottlenecks, AdamW training | Matches Swin/ViT accuracy while staying a pure CNN; efficient at high resolution; easy to deploy | Less natural for multimodal token pipelines |

**How I would choose:**

- **Limited data, edge deployment** → EfficientNet or MobileNet-style, or ResNet for simplicity.
- **Large pretraining or multimodal** → ViT (often a CLIP or DINOv2 backbone).
- **Dense prediction at high resolution, strong general backbone** → ConvNeXt or Swin.

**Takeaway:** much of the ViT vs CNN gap came from training recipes (augmentation, AdamW, long schedules), which is exactly what ConvNeXt showed.

### 15.23 Explain IoU, precision, recall, and mAP for object detection.

- **IoU (Intersection over Union)** — overlap between a predicted box and a ground-truth box: $\text{IoU} = \frac{\lvert A \cap B \rvert}{\lvert A \cup B \rvert}$. A prediction counts as a **true positive** if $\text{IoU} \geq$ a threshold (e.g. 0.5) and the class matches. Each ground truth can be matched only once.
- **Precision** — $\text{TP} / (\text{TP} + \text{FP})$: of the boxes predicted, how many are correct. Duplicates and wrong boxes are FPs.
- **Recall** — $\text{TP} / (\text{TP} + \text{FN})$: of the real objects, how many were found.
- **AP (Average Precision)** — sort detections by confidence, sweep the threshold, plot the precision-recall curve, and take the area under it (interpolated). Computed **per class**.
- **mAP** — the mean of AP over classes.
  - **PASCAL VOC:** mAP@0.5.
  - **COCO:** mAP@[0.5:0.95], the average over IoU thresholds 0.5, 0.55, …, 0.95. This rewards tight localisation. COCO also reports AP_small, AP_medium and AP_large.

**Production angle:** mAP is a ranking metric that summarises all thresholds. In deployment you pick **one** confidence threshold, so also report precision and recall at the operating point, per-class performance (rare classes hide inside the mean) and latency. Business cost often depends on FP vs FN asymmetry, for example missed defects vs false alarms.

### 15.24 How does NMS work?

**Non-Maximum Suppression** removes duplicate detections of the same object.

**Algorithm (per class):**

1. Drop boxes below a confidence threshold.
2. Sort the remaining boxes by score.
3. Take the highest-scoring box and keep it.
4. Remove every remaining box whose IoU with it exceeds a threshold (e.g. 0.5).
5. Repeat until no boxes are left.

```python
def nms(boxes, scores, iou_thr=0.5):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        ious = iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_thr]
    return keep
```

**Tradeoffs and variants:**

- **Threshold too low** suppresses true objects that overlap (crowds, stacked items). **Too high** leaves duplicates.
- **Soft-NMS** decays the scores of overlapping boxes instead of deleting them, which helps crowded scenes.
- **Class-agnostic NMS** suppresses across classes; **batched NMS** runs per class efficiently.
- **DIoU-NMS** also considers centre distance.
- **NMS-free detectors** (DETR, YOLOv10) learn one-to-one matching, removing this heuristic, its latency and its hard-to-export post-processing.

**Production:** NMS can dominate latency on edge devices with many candidate boxes, so use the framework's fused implementation (`torchvision.ops.nms`, TensorRT's NMS plugin).

### 15.25 What is data augmentation, and why does it improve generalization?

**Data augmentation** creates label-preserving variants of training examples so the model sees more of the input distribution.

- **Vision:** random crop/resize, flips, rotation, colour jitter, blur, Cutout, **Mixup/CutMix**, RandAugment.
- **Text:** back-translation, synonym replacement, token dropout, paraphrasing with an LLM.
- **Audio:** time shift, noise, SpecAugment.
- **Tabular:** SMOTE, noise injection (use with care).

**Why it improves generalization:**

- **Encodes invariances** you know are true ("a flipped cat is still a cat"), so the model does not have to learn them from data.
- **Acts as a regulariser** — more effective data and fewer spurious cues to memorise, which reduces variance and overfitting.
- **Covers distribution shift** — lighting, camera and orientation differences in production.
- Mixup and CutMix also improve **calibration** and smooth decision boundaries.

**Production angle:** design augmentations from the **failure modes** you see in production (motion blur, glare, low light) rather than a generic list. Apply them on the fly on GPU (DALI, Kornia) to avoid data-loader bottlenecks. Never augment validation or test sets, except deliberately for test-time augmentation.

### 15.26 When can augmentation actually hurt model performance?

Augmentation hurts when it **breaks the label, breaks the distribution, or adds noise the model cannot use**.

- **Label-changing transforms** — horizontal flip on text or digits ("b" vs "d"), rotating a 6 into a 9, flipping left/right in medical images where laterality matters, or vertical flips for street scenes.
- **Destroying the signal** — colour jitter when colour is the label (ripe vs unripe fruit, skin lesions), or aggressive crops that remove the object, which produces mislabeled samples.
- **Distribution mismatch** — augmentations that never occur in production move capacity away from the real distribution. Strong augmentation on a small model can cause **underfitting**.
- **Too strong for the capacity or the schedule** — heavy RandAugment or Mixup on small models or short training lowers accuracy. The best strength usually scales with model size.
- **Leakage** — augmenting before the train/test split puts near-duplicates in both sets and inflates metrics.
- **Detection and segmentation** — transforming the image but not the boxes or masks consistently.
- **Text** — synonym swaps can flip sentiment or meaning ("not bad").
- **Class imbalance** — oversampling plus augmentation of a rare class can overfit to augmentation artefacts.

**Mitigation:** treat augmentation as a hyperparameter, run ablations, check per-class metrics, and visually inspect augmented batches before training.

### 15.27 Explain the architecture of CLIP.

**CLIP (Contrastive Language-Image Pretraining, OpenAI 2021)** is a **dual-encoder** model trained on ~400M image-text pairs from the web.

- **Image encoder** — a ResNet or ViT that maps an image to a vector.
- **Text encoder** — a Transformer that maps a caption to a vector (the EOS token representation).
- **Projection heads** — linear layers map both into a **shared embedding space** (e.g. 512-d), followed by L2 normalisation.
- **Training** — for a batch of $N$ pairs, compute the $N \times N$ cosine-similarity matrix scaled by a learned temperature. The matching pairs are on the diagonal. Apply a **symmetric cross-entropy (InfoNCE)** loss over rows (image→text) and columns (text→image).

```python
logits = (img_emb @ txt_emb.T) * logit_scale.exp()
labels = torch.arange(N)
loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
```

**Properties:**

- Encoders are independent, so image and text embeddings can be **precomputed and indexed**, which suits retrieval at scale.
- Enables zero-shot classification, text-to-image search, and serves as a backbone for multimodal LLMs (e.g. the LLaVA vision encoder) and the text conditioning in diffusion models.

**Limitations:** weak at counting, spatial relations and fine-grained text in images; inherits web-data biases; a bag-of-concepts behaviour because there is no cross-attention between modalities.

### 15.28 Why does CLIP use contrastive learning?

**Because the supervision is noisy, open-ended web captions, and contrastive learning turns them into an efficient, scalable training signal.**

- **Predicting exact captions is too hard.** The OpenAI authors first tried generating captions; it learned about 3× slower for zero-shot accuracy. The model wastes capacity on wording, not on meaning.
- **A contrastive objective only asks "which caption matches this image?"** among the batch. That is a much easier proxy task that still forces the embeddings to capture semantics.
- **No fixed label set** — any text becomes a class, which is what makes zero-shot transfer possible.
- **Aligned shared space** — matching pairs pulled together and non-matching pairs pushed apart gives embeddings where cosine similarity is meaningful across modalities, which is ideal for retrieval.
- **Scales well** — every other sample in the batch is a free negative. CLIP used a 32k batch, so each step had ~1 billion negative pairs.

**Tradeoffs:** quality depends on large batches (expensive multi-GPU all-gather), false negatives in the batch (two near-identical captions), and the temperature setting. Follow-ups address this: **SigLIP** uses a pairwise sigmoid loss that works with smaller batches; **ALIGN** scales to noisier data; and **BLIP/CoCa** add captioning losses back for generative ability.

### 15.29 How would you use CLIP for zero-shot image classification?

**Turn class names into text prompts, embed everything, and pick the nearest text.**

1. For each class, write prompts like `"a photo of a {label}."`. **Prompt ensembling** (averaging the embeddings of many templates: "a blurry photo of a…", "a close-up of a…") usually adds 1–5% accuracy.
2. Encode the prompts with the text encoder, L2-normalise them, and cache the class-embedding matrix.
3. Encode the image, normalise it, compute cosine similarities, scale by the temperature, and apply softmax.

```python
import torch, open_clip
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tok = open_clip.get_tokenizer("ViT-B-32")
labels = ["cat", "dog", "car"]
with torch.no_grad():
    t = model.encode_text(tok([f"a photo of a {l}." for l in labels]))
    t = t / t.norm(dim=-1, keepdim=True)
    i = model.encode_image(preprocess(img).unsqueeze(0))
    i = i / i.norm(dim=-1, keepdim=True)
    probs = (100 * i @ t.T).softmax(dim=-1)
```

**Practical points:**

- Class names matter. Use descriptive, unambiguous names ("crane (bird)"), and add domain context ("a satellite photo of…").
- Zero-shot is weak on specialised domains (medical, industrial defects). Improve it cheaply with a **linear probe** on frozen embeddings, few-shot adapters (Tip-Adapter, CoOp), or fine-tuning.
- Calibrate a reject threshold for out-of-set images, because softmax over a closed label list always picks something.

### 15.30 What is ImageNet?

**ImageNet** is a large-scale image dataset and benchmark that played a major role in the development of modern deep learning for computer vision.

It contains millions of labeled images organized into thousands of object categories.

#### Why is ImageNet important?

It helped establish:

- Large-scale supervised computer vision
- CNN-based image classification
- Transfer learning
- Standardized benchmarking

The **ImageNet Large Scale Visual Recognition Challenge (ILSVRC)** was particularly influential in the development of CNN architectures.

ImageNet's importance is not just the dataset itself; it demonstrated that **large datasets + deep neural networks + GPU computation** could dramatically outperform traditional computer-vision pipelines.

### 15.31 What is AlexNet and why is it important?

**AlexNet** is a deep convolutional neural network that achieved a breakthrough result in the 2012 ImageNet competition.

#### Important techniques

- Deep CNN architecture
- **ReLU** activations
- GPU-based training
- Dropout
- Data augmentation
- Max pooling
- Local response normalization

Its success demonstrated the effectiveness of deep CNNs for large-scale image recognition.

#### Why was ReLU important?

Compared with sigmoid/tanh, ReLU reduces saturation-related gradient problems and makes deep networks easier to optimize.

AlexNet was a major milestone that helped trigger the modern deep-learning revolution in computer vision.

### 15.32 How does a Vision Transformer (ViT) work?

A ViT treats an image as a sequence of words. It cuts the image into fixed-size patches, projects each patch into a vector, adds position information, and feeds the sequence to a standard Transformer encoder. There are no convolutions in the backbone. The model has to learn locality and spatial structure from data, which is why ViTs are data-hungry but scale very well.

**1. Patch embedding.** An image $x \in \mathbb{R}^{H \times W \times C}$ is split into non-overlapping $P \times P$ patches. The number of patches, which is the sequence length, is:

$$
N = \frac{H \cdot W}{P^2}
$$

A $224 \times 224$ image with $P = 16$ gives $N = 196$ tokens; with $P = 14$ it gives 256. Each patch is flattened to a vector of length $P^2 C$ and multiplied by a learned matrix $E \in \mathbb{R}^{(P^2 C) \times D}$. In code this is a single `Conv2d` with kernel size and stride both equal to $P$.

**2. Class token and position embeddings.** A learnable `[CLS]` token is prepended, and learned position embeddings are added so the model knows where each patch came from:

$$
z_0 = [\,x_{\text{cls}};\ x_p^1 E;\ x_p^2 E;\ \dots;\ x_p^N E\,] + E_{\text{pos}}, \qquad E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}
$$

**3. Encoder.** $L$ pre-norm Transformer blocks, each with multi-head self-attention and an MLP, both wrapped in residual connections. The final `[CLS]` state (or the mean of the patch tokens) goes to a linear classification head. ViT-B/16 has 12 layers, $D = 768$ and about 86M parameters.

**Why the cost matters:** self-attention is $O(N^2 D)$. Halving the patch size or doubling the resolution multiplies $N$ by four and attention cost by sixteen. At a new resolution, the position embeddings must be interpolated in 2D before fine-tuning.

**Inductive bias versus data.** A CNN builds in locality and translation equivariance. A ViT only has the patch grid. Every patch can attend to every other patch from layer 1, which gives global context but nothing to prevent overfitting. The original paper found that ViT trained on ImageNet-1k alone underperformed comparable ResNets. Pretrained on ImageNet-21k or the 300M-image JFT dataset, it overtook them. The lesson: with little data, priors win; with lots of data, learned structure wins.

**The family that fixed the weaknesses:**

| Model | Key idea | What it solved |
| --- | --- | --- |
| **DeiT** | Strong augmentation and regularisation, plus a *distillation token* that learns from a CNN teacher's predictions alongside the `[CLS]` token | Competitive ViTs trained on ImageNet-1k only, on a single 8-GPU node |
| **Swin** | Attention within local windows, with windows shifted between layers so information crosses boundaries; patch merging builds a 4-stage hierarchy | Cost linear in image size; multi-scale feature maps for detection and segmentation |
| **MAE** | Mask about 75% of patches; the encoder sees only visible patches and a light decoder reconstructs pixels | Cheap, scalable self-supervised pretraining with no labels |
| **DINO / DINOv2** | Self-distillation: a student matches an EMA teacher across augmented crops; DINOv2 adds masked-token objectives and a curated 142M-image dataset | General-purpose frozen features that work with a linear probe for classification, retrieval, depth and segmentation |

DINOv3 (2025) scaled the same recipe to a 7B-parameter teacher with an extra regularizer to keep dense features sharp. Register tokens, extra learnable tokens that soak up global information, remove artefact spikes in ViT attention maps.

**When I would use a ViT:**

- **Default for transfer learning:** a pretrained DINOv2/DINOv3 or CLIP/SigLIP ViT as a frozen or fine-tuned backbone. With pretrained weights the data-hunger argument mostly disappears.
- **Multimodal systems:** vision-language models use ViT encoders because patch tokens slot directly into an LLM token stream.
- **Dense prediction at high resolution:** Swin or a hierarchical ViT, or ConvNeXt, since plain ViT attention is quadratic.
- **Small dataset, training from scratch, or edge hardware:** a CNN usually wins on accuracy per FLOP, latency and simplicity of quantization.

**Common pitfalls:** training from scratch on a few thousand images; forgetting to interpolate position embeddings when changing resolution; using a patch size too coarse for small objects; and judging ViT against a CNN without matching the training recipe.

### 15.33 How do two-stage and one-stage object detectors differ, and what are anchor-based and anchor-free detectors?

A **two-stage** detector first proposes a few hundred to a few thousand candidate regions, then classifies and refines each one. A **one-stage** detector predicts classes and boxes densely over a feature map in a single pass. **Anchor-based** detectors regress offsets from a fixed set of prior boxes. **Anchor-free** detectors predict boxes directly from points or centres. These are two separate design choices, so all four combinations exist.

**Two-stage (Faster R-CNN family)**

- A backbone with an FPN produces multi-scale features. A **Region Proposal Network (RPN)** slides over them, scores anchors as object or background and regresses them into proposals.
- **RoIAlign** (RoIPool in the original) crops a fixed-size feature for each proposal. A second head classifies it and refines the box.
- Strengths: accurate localisation and a natural extension to masks (Mask R-CNN) and keypoints. Weaknesses: more latency, harder to export, and the proposal step is a separate thing to tune.

**One-stage (SSD, YOLO, RetinaNet, FCOS)**

- Every feature-map location predicts class scores and box parameters at the same time. SSD predicts from several feature scales. YOLO started with a coarse grid and has since moved to multi-scale, anchor-free heads.
- The weakness was always **class imbalance**: tens of thousands of easy background locations swamp a handful of objects. RetinaNet's **focal loss** closed most of the accuracy gap with two-stage detectors.

**Anchor-based vs anchor-free**

| | Anchor-based | Anchor-free |
| --- | --- | --- |
| Examples | Faster R-CNN, SSD, RetinaNet, YOLOv2–v5, YOLOv7 | CornerNet, CenterNet, FCOS, YOLOX, YOLOv8 and later |
| Prediction | Offsets from preset boxes (scales × aspect ratios) | Distances from a point to the four box sides, or a centre heatmap plus size |
| Positives | IoU between anchor and ground truth above a threshold | Points inside the box or centre region, or learned assignment (SimOTA, TAL) |
| Pain points | Anchor sizes must match the dataset; many hyperparameters | Ambiguous points where objects overlap; needs good assignment |

- **FCOS** predicts the distances (l, t, r, b) to the box edges at every location and adds a **centre-ness** branch so that low-quality boxes far from the centre get lower scores.
- **CenterNet** ("Objects as Points") predicts a heatmap of object centres plus width and height. Peak extraction replaces most of the NMS work.

**Where the field is now.** The real accuracy driver turned out to be **label assignment**, not anchors. ATSS showed that once positives are chosen the same way, anchor-based and anchor-free RetinaNet/FCOS perform about the same. Modern YOLOs use task-aligned assignment and decoupled heads. YOLOv10 added consistent one-to-one assignment so that NMS can be dropped, and Ultralytics YOLO26 ships NMS-free by default and also removes Distribution Focal Loss to simplify export. DETR-style detectors (15.34) go further and drop anchors and NMS altogether.

**How to choose in practice**

- Real-time or edge with standard objects: a modern one-stage anchor-free detector (YOLO, RTMDet) or a real-time DETR.
- Highest accuracy, small datasets, or you need masks and keypoints from the same model: a two-stage detector or a DETR-family model with a strong backbone.
- Unusual aspect ratios (text lines, poles, ships in satellite images): anchor-free, or anchors re-clustered on your own boxes with k-means.

### 15.34 How does DETR perform object detection with Transformers, and what problems did later variants solve?

DETR (DEtection TRansformer) treats detection as **set prediction**. A CNN backbone and a Transformer encoder build global image features. A decoder turns a fixed set of learned **object queries** (100 in the paper) into exactly that many predictions, and each one is a class (or "no object") plus a box. Training uses **Hungarian matching**, a one-to-one assignment between predictions and ground truths, so no anchors and no NMS are needed.

**How it works**

1. Backbone (ResNet) → features, flattened into tokens with positional encodings.
2. Encoder: self-attention over all tokens gives global context, which helps with large objects and reasoning about the whole scene.
3. Decoder: object queries cross-attend to the image tokens and self-attend to each other, so they learn not to predict the same object twice.
4. Matching: the Hungarian algorithm finds the cheapest one-to-one assignment, using a cost built from class probability, L1 box distance and generalised IoU. Matched queries get classification and box losses. Unmatched queries learn "no object".

**Problems with the original DETR**

- **Very slow convergence:** around 500 epochs on COCO, against roughly 36 for Faster R-CNN. Each query starts by attending to the whole image and only slowly learns where to look.
- **Weak on small objects:** global attention is quadratic in the number of tokens, so DETR uses only a low-resolution feature map and has no multi-scale features.
- **Unstable matching early in training:** a query's target can switch between epochs.

**What later variants fixed**

| Variant | Key idea | Problem addressed |
| --- | --- | --- |
| Deformable DETR | Each query attends to a few learned sampling points (4 per head per level) around a reference point, across several feature scales | Convergence (about 10× fewer epochs) and small objects |
| Conditional DETR, DAB-DETR | Queries carry explicit spatial priors; DAB uses 4D anchor boxes refined layer by layer | Makes queries interpretable and speeds up training |
| DN-DETR | Feed noised ground-truth boxes to the decoder and train it to reconstruct them | Stabilises bipartite matching |
| DINO | Contrastive denoising, mixed query selection, look-forward-twice box refinement | Became the strongest closed-set DETR; basis for Grounding DINO |
| H-DETR, Co-DETR | Add auxiliary one-to-many matching during training only | One-to-one matching gives too few positive samples |
| RT-DETR | Efficient hybrid encoder (attention within one scale, CNN fusion across scales), IoU-aware query selection | Real-time speed that beats YOLOs of similar size, with no NMS |
| RT-DETRv2, D-FINE, DEIM | Bag of training tricks; fine-grained distribution refinement of box edges; denser matching | Better accuracy and faster training at the same latency |

**Why it matters in production**

- **NMS-free output** has a fixed shape, which makes it easy to export to TensorRT or ONNX, gives deterministic latency and avoids crowd-scene suppression errors.
- **The number of queries caps detections.** A scene with 500 objects (dense retail shelves, cells) needs more queries or a different model.
- **Training cost** is still higher than for a YOLO. For fine-tuning, start from a strong pretrained checkpoint (DINO, RT-DETR, D-FINE) and use the framework's recipes rather than training from scratch.
- DETR's query-based design spread to segmentation (Mask2Former), tracking (MOTR) and open-vocabulary detection (Grounding DINO). This shared design is a big reason it matters beyond COCO numbers.

### 15.35 What are semantic, instance and panoptic segmentation, and which architectures are used for each?

**Semantic** segmentation gives every pixel a class but does not separate objects of the same class. **Instance** segmentation gives a separate mask to each countable object ("thing") and ignores background. **Panoptic** segmentation does both: every pixel gets a class, and pixels of things also get an instance ID. Amorphous regions like sky or road ("stuff") get only a class.

| Task | Output | Typical metric | Classic architectures | Modern choices |
| --- | --- | --- | --- | --- |
| Semantic | H×W class map | mIoU | FCN, U-Net, DeepLabv3+, PSPNet | SegFormer, Mask2Former, UperNet on ViT/ConvNeXt |
| Instance | Set of (mask, class, score) | Mask AP | Mask R-CNN, YOLACT | Mask2Former, YOLO-seg, RTMDet-Ins |
| Panoptic | Class map + instance IDs, no overlaps | PQ | Panoptic FPN, Panoptic-DeepLab | Mask2Former, OneFormer |

**Semantic segmentation architectures**

- **U-Net:** a symmetric encoder-decoder with skip connections that concatenate high-resolution encoder features into the decoder. It works with little data and is still the default in medical imaging. nnU-Net automates its configuration and is a hard baseline to beat.
- **DeepLabv3+:** **atrous (dilated) convolutions** keep resolution without losing receptive field. **ASPP** (Atrous Spatial Pyramid Pooling) captures context at several rates, and a light decoder sharpens boundaries using low-level features.
- **SegFormer:** a hierarchical Transformer encoder with an all-MLP decoder. Strong accuracy for its size, and it handles different input resolutions well.

**Instance segmentation architectures**

- **Mask R-CNN:** Faster R-CNN plus a small FCN mask head that predicts a 28×28 mask per RoI. **RoIAlign** (bilinear sampling with no rounding) was critical for mask quality. Masks are predicted per class, which separates mask quality from classification.
- **Real-time:** YOLO-seg variants predict mask prototypes plus per-instance coefficients (the YOLACT idea), so mask cost is almost constant.
- Weaknesses: the 28×28 masks are coarse at boundaries (PointRend refines them), and the box-first design fails when the box is wrong.

**Panoptic segmentation architectures**

- **Panoptic FPN:** a Mask R-CNN instance branch plus a semantic branch on the same FPN, merged by heuristics that resolve overlaps.
- **Mask2Former:** one architecture for all three tasks. A pixel decoder produces per-pixel embeddings. A Transformer decoder with **masked attention** (each query attends only inside its current predicted mask) produces N queries, and each query outputs a class and a mask. Changing the task means changing only the training labels and the inference merge. OneFormer adds a task token so one trained model can do all three.

**How to decide**

- Need area, coverage or boundaries but not counts (land cover, road surface, tumour volume): **semantic**.
- Need to count or measure individual objects (cells, fruit, defects): **instance**.
- Robotics and autonomous driving need full scene understanding: **panoptic**.
- With little labelled data, combine a strong pretrained backbone with SAM-assisted labelling (15.38) before considering architecture changes. Label quality at boundaries usually limits mIoU more than the model does.

### 15.36 Which loss functions and metrics are used for image segmentation, and when do you choose each?

Start with **cross-entropy**, add a **region-overlap loss** (Dice or Lovász) when the foreground is small or the metric is IoU-like, and add **boundary-aware** terms only when boundary accuracy is the business requirement. Choose the metric that matches the decision: mIoU or Dice for area, boundary F-score for outlines, mask AP or PQ for instances.

**Loss functions**

- **Pixel-wise cross-entropy (CE):** stable and well-calibrated. Every pixel counts equally, so with 98% background the model can score well while missing small structures. Fixes include class weights (e.g. inverse frequency or median frequency balancing) and **OHEM** (train only on the hardest pixels).
- **Focal loss:** down-weights easy pixels:

$$
\text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)
$$

  With $\gamma = 2$, a pixel predicted correctly at $p_t = 0.9$ contributes 100× less than under CE.
- **Dice loss:** directly optimises overlap and ignores true negatives, so it is robust to foreground/background imbalance:

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}
$$

  Drawbacks: noisy gradients on tiny or empty masks, and poor calibration. Compute it per image or per batch, and keep the $\epsilon$ smoothing.
- **Tversky / Focal Tversky:** weight false negatives and false positives differently ($\alpha$, $\beta$). Use when a miss costs much more than a false alarm, for example lesions.
- **Lovász-Softmax:** a convex surrogate that directly optimises the Jaccard index. Good for squeezing the last mIoU points during fine-tuning.
- **Boundary / Hausdorff-distance losses:** penalise distance from the true contour. Use as an add-on later in training for thin structures (vessels, cracks, roads).
- **Compound losses:** CE + Dice is the default in nnU-Net and many medical pipelines. Mask2Former-style models use per-query CE/focal + Dice on masks after Hungarian matching.

```python
loss = 1.0 * F.cross_entropy(logits, target, weight=class_w) \
     + 1.0 * dice_loss(logits.softmax(1), one_hot(target))
```

**Metrics**

| Metric | What it measures | Use when |
| --- | --- | --- |
| Pixel accuracy | Fraction of pixels correct | Almost never alone; inflated by background |
| IoU / mIoU | Overlap per class, averaged over classes | Standard for semantic segmentation (Cityscapes, ADE20K) |
| Dice / F1 | $2\lvert A \cap B \rvert / (\lvert A \rvert + \lvert B \rvert)$, monotonic with IoU | Medical imaging convention |
| Boundary F-score, Boundary IoU | Agreement within a few pixels of contours | Outlines matter (matting, measurement) |
| Hausdorff distance (HD95) | Worst-case contour distance, 95th percentile | Surgical or radiotherapy margins |
| Mask AP | Detection-style AP using mask IoU | Instance segmentation (COCO) |
| PQ | Segmentation quality × recognition quality | Panoptic segmentation |

**Pitfalls seen in practice**

- **Averaging choices change the number.** mIoU per image vs over the whole dataset, and how empty classes are handled, can shift results by several points. Fix the protocol before comparing models.
- **Empty masks:** Dice is undefined or 0/0 when both prediction and ground truth are empty. Decide the rule explicitly (usually count it as 1).
- **Mean hides the tail.** Report per-class IoU and slices by object size, because a 0.85 mIoU can hide a 0.3 on the class that matters.
- **Optimised loss ≠ product metric.** If the product measures area (e.g. corrosion percentage), also track absolute area error, not just IoU.

### 15.37 How do you detect small objects and handle severe class imbalance in object detection?

Small objects fail because they **vanish in downsampled features** and get **few positive training samples**. Imbalance fails because **background and frequent classes dominate the gradient**. Fix resolution and feature scale first, then sampling and the loss, then inference tricks such as tiling. Measure each step on AP_small and per-class AP, not overall mAP.

**Small objects: why they are hard**

- A 12×12 pixel object becomes less than one cell at stride 32. The backbone has averaged it away.
- IoU is very sensitive to small shifts: a 3-pixel offset on a 10-pixel box drops IoU sharply, so assignment rarely picks such objects as positives.
- Labels for tiny objects are noisier and less consistent.

**Fixes, roughly in order of payoff**

- **Increase input resolution**, or keep native resolution and **tile**. Resizing a 4K image down to 640 throws away the object.
- **High-resolution feature levels:** an **FPN** (top-down path with lateral connections) gives semantically strong features at fine strides. Add a P2 (stride 4) head for very small targets, accepting the extra compute.
- **Slicing-aided inference (SAHI):** run the detector on overlapping tiles (e.g. 640×640 with 20% overlap) plus optionally the full image, map the boxes back and merge them with NMS or NMM (non-maximum merging). Use **slicing-aided fine-tuning** too, so that training sees objects at the scale they appear in tiles.
- **Better assignment and box loss:** ATSS, TAL or SimOTA give small objects a fair number of positives. Normalised Wasserstein Distance or similar penalties are smoother than IoU for tiny boxes.
- **Augmentation:** copy-paste of small instances, mosaic, and scale jitter that does not shrink objects further.
- **Anchors** (if used): re-cluster anchor sizes on your own data.

**Severe class imbalance**

There are two kinds: **foreground vs background** (the detector's built-in imbalance) and **between classes** (long-tailed datasets like LVIS or rare defect types).

- **Focal loss** (RetinaNet) handles the first: $\text{FL} = -\alpha_t (1 - p_t)^{\gamma} \log p_t$ with $\gamma = 2$ and $\alpha = 0.25$ as the paper defaults. Initialise the classifier bias so the starting foreground probability is about 0.01, or training diverges early.
- **Repeat factor sampling** (as used for LVIS) oversamples images that contain rare classes. It is simple and works well.
- **Loss re-weighting:** equalisation loss or seesaw loss reduce the suppressive gradients that frequent classes send to rare ones.
- **Data:** targeted collection, synthetic or copy-pasted rare instances, and hard negative mining for false-alarm classes.
- **Decouple representation from classifier:** train normally, then re-balance only the classification head.
- **Thresholds per class:** after training, pick operating points per class on validation data, because a single global confidence threshold suits frequent classes.

**Evaluation discipline**

- Report AP_small/medium/large, per-class AP and recall at the operating confidence, plus the frequent/common/rare split for long-tailed sets.
- Tiling raises recall but also false positives at tile edges and multiplies latency by the number of tiles. Budget for it: a 4K frame at 640 tiles with overlap can be 20+ forward passes.
- **Example:** in aerial vehicle detection, moving from full-image inference at 1024 to a P2 head plus tiled training and SAHI inference typically gives large gains in AP_small, while overall mAP moves much less. That is why you look at the size slices.

### 15.38 What is open-vocabulary detection and segmentation, and how do models like Grounding DINO and SAM 2 work?

**Open-vocabulary** models detect or segment categories described in **free text at inference time**, including ones never labelled in their detection training data. They do this by aligning region features with a language embedding space. **Grounding DINO** detects boxes for any text phrase. **SAM 2** segments and tracks whatever you point at in images and videos, but it is class-agnostic. Combining them gives text-prompted masks and tracks.

**The core idea**

- A closed-set detector ends in a fixed classifier with K outputs. An open-vocabulary detector replaces it with a **similarity between region embeddings and text embeddings**, so any phrase can be a class.
- The knowledge comes from **image-text pretraining** (CLIP-style) and **grounding data** (phrases linked to boxes), often expanded with pseudo-labels.

**OWL-ViT and OWLv2**

- Take a CLIP-style ViT and drop the final pooling. **Every output token** predicts a box and a class embedding, and the class logits are dot products with text-query embeddings. Image queries (an example crop) also work, which is useful for one-shot detection.
- OWLv2 scales training with **self-training**: an existing detector pseudo-labels about a billion web images, which improves rare-class performance substantially.

**Grounding DINO**

- Built on the DINO detector (15.34). An image backbone (Swin) and a text backbone (BERT) are fused **early and deeply**:
  1. A **feature enhancer** applies image-to-text and text-to-image cross-attention.
  2. **Language-guided query selection** picks the image tokens most relevant to the text as decoder queries.
  3. A **cross-modality decoder** attends to both image and text features. Each output box is scored against each text token.
- Trained on detection, grounding and caption data. It is strong zero-shot (the paper reports 52.5 AP on COCO without COCO training data) and handles referring expressions ("the red cup on the left"). Prompt with short, period-separated class names. Long prompts and rare words degrade results.

**SAM 2**

- Extends SAM (promptable segmentation) to video. Prompts are points, boxes or masks on any frame.
- A **Hiera** image encoder runs once per frame. **Memory attention** conditions the current frame on a **memory bank**: memories from recent frames and from prompted frames, plus **object pointers** (compact per-object tokens). A memory encoder writes each new prediction into the bank, and an occlusion head predicts whether the object is visible.
- It runs as a **streaming** model, so it processes arbitrarily long videos frame by frame, and a user can add a corrective click on any frame. Trained on the SA-V dataset of tens of thousands of videos with hundreds of thousands of masklets.
- Meta's **SAM 3** (late 2025) adds **concept prompts**: give a short noun phrase and it returns masks and IDs for every matching instance, which brings open-vocabulary detection into the SAM family directly.

**Production patterns**

- **Auto-labelling:** Grounding DINO boxes → SAM masks → human review → fine-tune a small closed-set model (YOLO, RT-DETR) that is cheap to serve. This is the most common real use.
- **Text-prompted tracking:** detect with text on keyframes, then SAM 2 propagates masks.
- **Caveats:** heavy compute; scores are not calibrated across prompts, so thresholds must be tuned per class; prompt wording changes results; weak on domain-specific vocabulary (part numbers, medical terms) and small objects. Open-vocabulary models are excellent for bootstrapping and long-tail coverage. When the class list is fixed and latency matters, a fine-tuned closed-set detector usually wins.

### 15.39 How do multi-object tracking algorithms such as SORT, DeepSORT and ByteTrack work?

All three follow **tracking-by-detection**: run a detector on every frame, **predict** where each existing track will be, **associate** detections to tracks, then **update**, start or end tracks. They differ in which cues they use for association and which detections they allow into it.

**SORT (Simple Online and Realtime Tracking)**

- **Motion model:** each track has a **Kalman filter** with a constant-velocity state (box centre, area, aspect ratio and their velocities).
- **Association:** build a cost matrix of 1 − IoU between predicted track boxes and detections, solve it with the **Hungarian algorithm**, and reject matches below an IoU threshold.
- **Life cycle:** unmatched detections start tentative tracks. Tracks unmatched for `max_age` frames are deleted.
- Very fast. Weakness: IoU alone causes **identity switches** after occlusions or when objects cross paths.

**DeepSORT**

- Adds an **appearance embedding** from a re-identification (Re-ID) CNN for each detection. Each track keeps a gallery of recent embeddings.
- Cost combines **cosine distance** of appearance with **Mahalanobis distance** from the Kalman prediction, which is used to gate implausible matches.
- **Matching cascade:** recently seen tracks get priority, and an IoU stage handles the rest.
- Much fewer ID switches across occlusions. Costs a Re-ID forward pass per detection, and appearance fails on look-alike objects (uniforms, identical products).

**ByteTrack**

- Key insight: **low-confidence detections are often real objects that are occluded**. Throwing them away (as most trackers did) creates fragmented tracks.
- **Two-stage association:**
  1. Match **high-score** detections to all tracks (IoU, optionally with appearance).
  2. Match **low-score** detections only to tracks left unmatched, using IoU. Unmatched low-score boxes are discarded as background.
- New tracks start only from high-score detections. Simple, fast, and strong on MOT17/MOT20.

**Later improvements**

- **OC-SORT:** observation-centric fixes for Kalman drift during occlusion. It re-updates the track along a virtual path when the object reappears and uses motion direction consistency. Better for non-linear motion (dancing, sports).
- **BoT-SORT:** ByteTrack plus **camera motion compensation** (global motion estimated between frames and applied to predictions), a better Kalman state, and optional Re-ID. A default tracker in Ultralytics.
- **End-to-end trackers** (MOTR, MOTRv2) carry DETR queries across frames. Elegant, but heavier, and usually not better than a strong detector plus ByteTrack in practice.

**Metrics**

- **MOTA** is dominated by detection errors (FP, FN), with ID switches a small term.
- **IDF1** measures how consistently identities are kept.
- **HOTA** balances detection and association and is now the preferred single number.

**Production tips**

- **Detector quality dominates.** A better detector improves tracking more than any tracker change.
- Tune the **detection thresholds** (high and low), **track buffer** (how long to keep lost tracks) and **match thresholds** to the frame rate. At lower FPS, motion per frame is larger and IoU matching fails.
- Moving cameras (drones, vehicles) need camera motion compensation. Long occlusions or re-entry into the scene need Re-ID.
- For counting (people crossing a line), measure **ID switches** on your own footage, because each switch can double-count.

### 15.40 How would you debug a computer vision model that performs well offline but poorly in production?

Assume the **input is different** until proven otherwise. Most offline-to-production gaps in computer vision come from **preprocessing mismatches**, **data drift** (camera, lighting, scene), or a **metric that did not match the product requirement**. Real model bugs are less common. Debug in order: reproduce, compare inputs, check the serving pipeline, then the data, then the model.

**1. Reproduce on real production inputs**

- Log raw production frames (sampled, with privacy controls) together with model inputs, outputs and metadata (camera ID, time, firmware).
- Label a few hundred production samples. Without labelled production data you are guessing. Compute the same metrics as offline, sliced by camera, time of day and object size.

**2. Training-serving skew in preprocessing (the most common cause)**

| Check | Typical bug |
| --- | --- |
| Colour order | BGR from OpenCV vs RGB in training |
| Normalisation | Missing mean/std, or values left in 0–255 vs 0–1 |
| Resize | Letterbox padding in training vs stretch in serving; different interpolation |
| Orientation | EXIF rotation ignored; portrait images sideways |
| Compression | Training on PNG, serving on heavily compressed JPEG or H.264 video frames |
| Box mapping | Coordinates not mapped back from letterboxed input to the original frame |
| Precision | FP16/INT8 export loses accuracy; quantisation calibrated on unrepresentative images |

A fast test: run the **same image** through the training code and the production endpoint and compare the input tensors and outputs. Any difference beyond small float noise is a bug.

**3. Serving and post-processing**

- Confidence and NMS thresholds differ between the eval script and the serving config.
- Export bugs in ONNX or TensorRT: operations not supported, or NMS handled differently. Compare outputs layer by layer if needed.
- Batching or padding effects, dropped frames under load, timeouts that silently return empty results.

**4. Data and domain drift**

- **Covariate shift:** new cameras, lenses, resolutions, night/IR mode, weather, motion blur, dirty lenses, different mounting angles.
- **Label shift:** class frequencies in production differ from the validation set, so the thresholds are wrong even when the model is fine.
- **Evaluation leakage offline:** near-duplicate frames from the same video in train and test inflate offline scores. Split by video, site or camera, not by frame.
- Monitoring signals without labels: embedding drift between production and training images, confidence histograms, detections per frame, and image statistics (brightness, blur score).

**5. Model weaknesses exposed by production**

- Error analysis on a labelled production sample: group errors into missed small objects, confusions between classes, localisation errors, duplicates and background false positives. Tools like FiftyOne help.
- Fix with **targeted data**, not blind retraining: collect hard cases (active learning on low-confidence or disagreement frames), use augmentation that simulates the real conditions (blur, compression, low light), then fine-tune.

**6. Prevent recurrence**

- Ship preprocessing **inside** the exported model or as one shared library used by both training and serving.
- Golden-image tests in CI that compare training and serving outputs.
- Shadow deployment on live traffic before switching, and per-camera dashboards with alerts on drift.
- Hold-out sets defined by deployment conditions (new site, night, rain), and gating on those slices before release.

**Illustrative example:** a defect detector scores 0.92 mAP offline and misses half the defects on the line. The cause is the production camera's JPEG quality setting plus a stretch resize that changed the defect's aspect ratio. Fixing preprocessing recovered most of the gap. Fine-tuning on 400 labelled line images with compression augmentation recovered the rest.

### 15.41 How does self-supervised learning work for vision, and how do contrastive, distillation-based and masked-image methods compare?

Self-supervised learning (SSL) builds a training signal from the images themselves. You pick a pretext task that can only be solved if the network learns useful structure, such as "these two crops come from the same photo" or "fill in the hidden patches". Then you throw away the pretext head and keep the encoder. The three main families differ in what they predict and in how they stop the encoder from *collapsing* to a constant output.

**1. Contrastive (SimCLR, MoCo).** Two augmented views of the same image are positives, and every other image in the batch is a negative. The InfoNCE loss pulls positives together and pushes negatives apart:

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}
$$

- **SimCLR** showed that the augmentation recipe (random crop plus strong colour jitter) and a projection MLP after the encoder matter more than architecture. It needs very large batches (4096+) to get enough negatives.
- **MoCo** decouples negatives from batch size. It keeps a queue of keys produced by a *momentum encoder* whose weights are an exponential moving average (EMA) of the query encoder, so the queued keys stay consistent.
- **Weakness:** negatives that are really the same class ("false negatives") get pushed apart, and the learned invariances are only as good as the augmentations you hand-pick.

**2. Distillation / non-contrastive (BYOL, DINO, DINOv2).** No negatives at all. A student network predicts the output of an EMA teacher on a different view. Collapse is avoided by asymmetry, not by repulsion: BYOL adds a predictor head on the student only, and DINO centres and sharpens the teacher's softmax output. DINO also uses multi-crop: the teacher sees only global crops and the student also sees small local crops, which forces local-to-global correspondence. That is why DINO attention maps segment objects without any labels. DINOv2 adds a masked-patch objective (from iBOT), a KoLeo regulariser to spread features out, and careful data curation.

**3. Masked image modelling (MAE, BEiT, I-JEPA).** Hide most of the patches and predict what is missing.

- **MAE** reconstructs raw pixels. Images are highly redundant, so a high mask ratio is needed to make the task hard; it also makes the encoder cheap because it only processes visible tokens.
- **I-JEPA** predicts the *embeddings* of masked blocks from a context block, using an EMA target encoder. It skips pixel detail and needs no hand-crafted augmentations. The paper reports a ViT-H/14 trained in under 1,200 GPU hours, more than 10 times cheaper than MAE at similar linear-probe accuracy.

**How they compare in practice:**

| Family | Signal | Frozen features (linear probe, k-NN, retrieval) | After full fine-tuning | Main cost |
| --- | --- | --- | --- | --- |
| Contrastive | Instance discrimination with negatives | Good, strongly global | Good | Large batches or queues, augmentation tuning |
| Distillation | Match EMA teacher across views | Best; also strong dense features | Very good | Fiddly to stabilise (centring, temperature, EMA schedule) |
| Masked pixels | Reconstruct pixels | Weak; features need fine-tuning | Very good | Cheap per step, long schedules |
| Masked embeddings | Predict latent of masked region | Good | Very good | Masking strategy matters |

**The key insight:** contrastive and distillation methods learn *invariance*, so they give semantic, linearly separable features. Pixel reconstruction learns *what images look like*, keeping low-level detail that fine-tuning later turns into semantics. That explains MAE's well-known gap between poor linear-probe and strong fine-tune accuracy.

**How I would choose in industry:** rarely pretrain from scratch. Start from a public DINOv2/DINOv3 backbone for frozen features, retrieval, or dense tasks. Run *domain-adaptive* SSL only when your images are far from web photos (medical slides, satellite, industrial X-ray) and you have millions of unlabelled images. MAE-style continued pretraining is often the cheapest win there because it needs no augmentation design. Watch for augmentations that destroy the signal you care about: colour jitter on a task where colour is the defect will teach the model to ignore exactly that.

### 15.42 How do you fine-tune a vision model when you have very little labelled data?

Start from the strongest pretrained backbone you can afford, change as few weights as possible, and use every unlabelled image you have. With 10 to 1,000 labels the real enemies are overfitting and an unreliable validation set, not model capacity.

**A ladder of options, cheapest first:**

1. **Zero-shot or open-vocabulary.** A CLIP/SigLIP classifier or an open-vocabulary detector (Grounding DINO, OWL-ViT) may already be good enough, and it gives you a baseline.
2. **Frozen features plus a light head.** Extract DINOv2 embeddings and train logistic regression, k-NN, or a prototype classifier (class mean embeddings). This is robust with 5 to 50 examples per class, trains in seconds, and is hard to overfit.
3. **Linear probe, then fine-tune (LP-FT).** Train the head with the backbone frozen first, then unfreeze with a small learning rate. Fine-tuning straight away with a random head sends large gradients into the backbone and distorts pretrained features, which hurts out-of-distribution accuracy.
4. **Parameter-efficient fine-tuning.** LoRA on attention layers, adapters, visual prompt tuning, or tuning only biases and norm layers. You train less than 1% of the weights, which acts as strong regularisation and lets you keep one adapter per customer or site.
5. **Full fine-tuning** only with a few thousand labels or a big domain shift, using layer-wise learning rate decay so early layers move less than later ones.

**Regularisation that actually matters at small scale:**

- Strong but *label-preserving* augmentation (RandAugment, Mixup/CutMix for classification, copy-paste for detection and segmentation).
- Early stopping on a held-out set, weight decay, label smoothing, and fewer epochs than you think.
- Keep BatchNorm statistics frozen with tiny batches, or use a backbone with LayerNorm/GroupNorm.

**Using unlabelled data:**

- **Self-training / pseudo-labelling:** train a teacher, label the unlabelled pool, keep only high-confidence predictions, and retrain a student with strong augmentation. FixMatch combines weak-augmentation pseudo-labels with strong-augmentation consistency.
- **Domain-adaptive pretraining:** continue SSL (MAE or DINO-style) on your unlabelled domain images before fine-tuning.
- **Foundation-model labelling:** use SAM, an open-vocabulary detector or a VLM to draft labels and have humans correct them, which is faster than labelling from scratch.
- **Synthetic data:** rendered or diffusion-generated images help for rare defects, but always validate on real images only.

**Evaluation is the hard part.** With 50 labels, one mislabelled test image moves accuracy by 2 points. Use repeated stratified k-fold cross-validation, report confidence intervals, split by *source* (patient, camera, production batch) to avoid leakage between near-duplicate frames, and look at every error by hand.

**Concrete example:** a factory has 80 images of a new scratch defect and 20,000 unlabelled parts. I would start with DINOv2 features and a logistic-regression head plus an anomaly-detection score (PatchCore-style nearest-neighbour distance on normal parts), add LoRA fine-tuning once labels reach a few hundred, and use the model's uncertain cases to choose the next images for labelling.

**Failure modes:** the model learns a shortcut such as background, scanner, or a ruler in medical images; validation leaks through duplicates; and pseudo-labels reinforce early mistakes (confirmation bias), especially on minority classes.

### 15.43 How do classical feature detectors such as Harris, SIFT and ORB work, and when does classical computer vision still beat deep learning?

Classical detectors find points that can be located precisely and found again in another image, then describe each point with a compact vector. They are hand-designed from image gradients, need no training data, run on a CPU, and come with geometric guarantees. That is why they still sit underneath SLAM, panorama stitching, structure-from-motion and camera calibration.

**Harris corner detector.** A corner is a point where shifting a small window in *any* direction changes the intensity a lot. Harris summarises local gradients in a $2 \times 2$ structure tensor and scores each pixel:

$$
M = \sum_{\text{window}} w(x,y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}, \qquad R = \det(M) - k\,(\operatorname{tr} M)^2
$$

Two large eigenvalues mean a corner, one large eigenvalue means an edge, and two small ones mean a flat region. Harris is rotation invariant but **not scale invariant**: a corner at one zoom level becomes a curve at another. Shi-Tomasi ("good features to track") uses the smaller eigenvalue instead and feeds the KLT tracker.

**SIFT (Scale-Invariant Feature Transform).**

1. **Scale space:** blur the image at increasing scales and take differences of Gaussians (DoG), which approximate the Laplacian.
2. **Keypoints:** local extrema across position *and* scale, refined to sub-pixel accuracy; low-contrast and edge-like points are rejected.
3. **Orientation:** the dominant gradient direction in the neighbourhood is assigned, which makes the descriptor rotation invariant.
4. **Descriptor:** a 4×4 grid of 8-bin gradient histograms gives a 128-dimensional vector, normalised for illumination robustness.

Matching uses nearest neighbours plus Lowe's **ratio test**: accept a match only if the best distance is clearly smaller than the second best.

**ORB (Oriented FAST and Rotated BRIEF).** Built for speed and free use. FAST finds corners by checking a ring of 16 pixels around a candidate, Harris scores rank them, an image pyramid gives some scale coverage, and an intensity centroid gives orientation. BRIEF describes the patch with 256 binary intensity comparisons, so matching uses Hamming distance (XOR plus popcount). It is one to two orders of magnitude faster than SIFT and powers ORB-SLAM.

**The geometry step is what makes them useful.** Matches are fed to RANSAC to fit a homography, fundamental or essential matrix, rejecting outliers, and then bundle adjustment. Each step is interpretable and gives measurable reprojection error.

**When classical still wins:**

- **Precise geometry:** calibration, metrology, stitching and visual odometry need sub-pixel accuracy and explicit geometric models.
- **Tiny compute budgets:** microcontrollers, battery-powered sensors, or 1 kHz industrial lines where a 2 ms CPU pipeline beats any network.
- **Controlled environments:** fixed lighting and fixtures, where thresholding, template matching, blob analysis or Hough transforms solve the task deterministically.
- **No training data, or certification:** a deterministic algorithm is easier to validate in regulated settings.
- **Within-scene matching:** SIFT remains a strong baseline for matching the same rigid scene.

**Where deep learning wins:** large viewpoint or lighting changes, day-to-night, textureless surfaces, and anything semantic. Learned matchers (SuperPoint with SuperGlue or LightGlue, dense matchers such as LoFTR) are now the default in modern structure-from-motion.

**The production answer is usually hybrid:** a learned detector or matcher feeding classical RANSAC and bundle adjustment, or a CNN for detection followed by classical subpixel edge fitting for measurement.

### 15.44 How do monocular depth estimation, NeRF and 3D Gaussian Splatting work, and when would you use each?

They answer different questions. **Monocular depth** guesses per-pixel distance from a single image. **NeRF** and **3D Gaussian Splatting (3DGS)** reconstruct one specific scene from many posed photos so you can render it from new viewpoints.

**Monocular depth estimation (MiDaS, Depth Anything).** Depth from one image is mathematically ambiguous, so the network learns priors such as perspective, object size, occlusion and texture gradients.

- **MiDaS** made the field practical by training on many mixed datasets with a *scale-and-shift-invariant* loss, so datasets with incompatible depth units could be combined. The output is **relative** (affine-invariant inverse) depth: correct ordering and shape, but no metres.
- **Depth Anything** scaled this with a DINOv2 encoder and a DPT decoder, using a teacher to pseudo-label tens of millions of unlabelled images. **V2** trains the teacher only on synthetic images, whose depth labels are precise on thin structures and edges, and then distils into smaller students through pseudo-labelled real images.
- **Metric depth** (Depth Anything metric heads, Metric3D, Depth Pro, UniDepth) needs fine-tuning on metric data or knowledge of the camera intrinsics, because focal length and real-world scale are ambiguous.
- **Use it for:** augmented-reality occlusion, background blur, robotics priors, and 2D-to-3D effects. One forward pass, real time. **Do not** trust it for safety-critical metric distance without a LiDAR, stereo or known-object-size anchor.

**NeRF (Neural Radiance Fields).** A small MLP maps a 3D position and viewing direction to colour and density. To render a pixel, you cast a ray, sample points along it, query the MLP, and alpha-composite:

$$
C(\mathbf{r}) = \sum_{i} T_i \left(1 - e^{-\sigma_i \delta_i}\right) \mathbf{c}_i, \qquad T_i = \exp\Big(-\sum_{j<i} \sigma_j \delta_j\Big)
$$

Training minimises the photometric error against the input photos. Positional encoding (Fourier features) lets the MLP represent high frequencies. Camera poses come from structure-from-motion (COLMAP). The original was slow: hours to train and seconds per frame. Instant-NGP's hash-grid encoding cut training to minutes, and Mip-NeRF 360 handled unbounded scenes.

**3D Gaussian Splatting.** It replaces the implicit MLP with millions of explicit 3D Gaussians, each with a position, an anisotropic covariance (scale and rotation), opacity, and view-dependent colour stored as spherical harmonics. They are initialised from the SfM point cloud, projected to 2D and alpha-blended by a fast tile-based rasteriser that is fully differentiable. Adaptive density control clones and splits Gaussians where reconstruction error is high and prunes transparent ones. The result is NeRF-level quality with real-time rendering (30+ fps at 1080p) and training in tens of minutes.

**Comparison:**

| | Monocular depth | NeRF | 3DGS |
| --- | --- | --- | --- |
| Input | One image | 50 to 300 posed images of one scene | Same as NeRF |
| Output | Depth map | Implicit field, novel views | Explicit Gaussians, novel views |
| Generalises to new scenes | Yes | No, per-scene optimisation | No, per-scene optimisation |
| Rendering speed | One pass | Slow (fast with hash grids) | Real time, even in browsers and on mobile |
| Storage | Model only | Small (MLP weights) | Large (hundreds of MB, compressible) |
| Editing and integration | Not applicable | Hard | Easier: explicit primitives, game-engine plugins |

**How I would choose:** monocular depth for any single-image, real-time or per-frame need. 3DGS as the default for capturing a real place or product for viewing (digital twins, real estate, e-commerce, VFX). NeRF-style fields when you need a smooth continuous field, compact storage, or clean surface extraction (SDF variants such as NeuS).

**Production failure modes:** bad SfM poses ruin both NeRF and 3DGS; moving objects, reflections and changing exposure cause floaters; 3DGS can look great from training views and break from novel angles that were not covered. Plan the capture (overlap, fixed exposure, no motion blur) as carefully as the model.

### 15.45 How do video understanding models capture motion and time?

A video model must learn appearance (what is in each frame) and dynamics (how it changes). Averaging per-frame features recognises "a kitchen" but cannot tell "opening a drawer" from "closing a drawer". The architectures differ in where and how cheaply they mix information across time.

**1. Two-stream networks.** One CNN sees RGB frames and another sees stacked optical flow, with predictions fused at the end. Explicit flow gives motion directly, but computing flow is expensive and the two streams fuse late.

**2. 3D convolutions (C3D, I3D).** Kernels extend over time (for example 3×3×3), so motion is learned directly. **I3D** "inflated" ImageNet-pretrained 2D filters into 3D by repeating them along time and dividing by the temporal depth, which lets a video model start from image weights. The Kinetics dataset made this training practical. Cost grows with frames times pixels. **R(2+1)D** factorises a 3D conv into a spatial 2D conv followed by a temporal 1D conv, which is cheaper and easier to optimise.

**3. SlowFast.** Inspired by the primate visual system. A **slow pathway** samples few frames with many channels to capture semantics. A **fast pathway** samples about 8 times more frames ($\alpha = 8$) with only about 1/8 of the channels ($\beta = 1/8$) to capture fast motion cheaply. Lateral connections feed fast into slow. The fast path adds roughly 20% of compute.

**4. Video transformers (TimeSformer, ViViT, Video Swin).** Videos become spatiotemporal patch tokens. Full joint attention over $T \times N$ tokens is too expensive, so **TimeSformer** uses *divided space-time attention*: in each block, a patch first attends to the same location in other frames (temporal), then to other patches in its own frame (spatial). This cuts cost from $O((TN)^2)$ to $O(TN(T+N))$ and performed best among their variants. ViViT factorises the encoder instead; Video Swin uses local 3D windows.

**5. Self-supervised video pretraining (VideoMAE, V-JEPA).** **VideoMAE** masks 90 to 95% of tokens with *tube masking*, the same spatial mask across all frames. Without tubes the model could copy a patch from the next frame instead of reasoning about motion. It is very data-efficient: it works on datasets of a few thousand clips. V-JEPA predicts masked regions in embedding space instead of pixels.

**6. Frame sampling plus a language model.** Vision-language models encode a sparse set of frames and let the LLM reason over time. Good for description and question answering, weak on fine-grained motion.

**Appearance bias is the big trap.** On Kinetics, a single frame often reveals the action (a pool means swimming). On **Something-Something V2**, where labels like "pushing something left to right" depend only on motion, frame-based models fail. Always test whether your model beats a shuffled-frame or single-frame baseline; if it does not, it is not using time.

**Practical trade-offs:**

| Need | Good choice |
| --- | --- |
| Real-time on edge, short actions | 2D CNN with temporal shift module or X3D |
| Fine-grained motion | SlowFast or VideoMAE-pretrained ViT |
| Long videos (minutes to hours) | Clip-level features plus a temporal model over clips, or memory-augmented transformers |
| Open-ended questions | Frame sampling plus VLM |
| Temporal localisation | Clip features plus a temporal action detection head |

**Production details:** video decoding is often the real bottleneck, so decode on GPU (NVDEC) and sample frames before decoding where possible. Match frame rate and clip length between training and serving; a model trained on 16 frames at 2-frame stride behaves differently at stride 8.

### 15.46 How do you optimise and deploy a computer vision model on edge devices?

Treat it as co-design between the model, the compiler and the target chip. Fix the hardware and latency budget first, choose an architecture that runs well on that chip, then compress and compile, and measure on the real device at every step. FLOPs are a poor guide to latency at the edge.

**1. Pin down the constraints.** Target device and accelerator (Jetson GPU, phone NPU, Intel CPU or iGPU, Coral or Hailo NPU, microcontroller), latency budget per frame including pre- and post-processing, memory, power and thermal limits, and the accuracy floor.

**2. Pick an edge-friendly architecture.** MobileNetV3/V4, EfficientNet-Lite, RepVGG-style re-parameterised nets, YOLO nano and small variants, or small ViT hybrids such as MobileViT and EfficientViT. Avoid operators the accelerator does not support; an unsupported op can push the whole graph back to the CPU. Depthwise convolutions are low in FLOPs but memory-bound on some GPUs. Lowering input resolution is often the single biggest win.

**3. Compress.**

- **Quantisation.** FP32 to FP16 is nearly free. INT8 makes weights 4 times smaller than FP32 and, as a rule of thumb, gives 2 to 4 times faster inference on hardware with integer units; always measure on the target device.
  - *Post-training quantisation (PTQ)* with a few hundred representative calibration images. Use per-channel weight scales; calibrate on real production-like images, not random ones.
  - *Quantisation-aware training (QAT)* when PTQ loses more than about 1 point: fake-quant nodes simulate rounding during fine-tuning.
  - Keep sensitive layers (first conv, detection head, softmax or attention) in higher precision when needed.
- **Pruning.** Structured pruning (whole channels or heads) gives real speedups; unstructured sparsity rarely does, except for NVIDIA's 2:4 pattern on Ampere and newer.
- **Knowledge distillation.** Train the small student to match a big teacher's logits or features. This often recovers most of the accuracy lost by shrinking.

**4. Compile for the runtime.**

| Target | Runtime | Notes |
| --- | --- | --- |
| NVIDIA Jetson or GPU | TensorRT | Layer fusion, kernel auto-tuning, FP16/INT8 with calibration; engines are tied to GPU and TensorRT version |
| Intel CPU, iGPU, NPU | OpenVINO | NNCF for PTQ and QAT |
| Android and iOS, microcontrollers | LiteRT (formerly TensorFlow Lite), Core ML | GPU or NNAPI-style delegates; Core ML for the Apple Neural Engine |
| Cross-platform | ONNX Runtime | Execution providers for CUDA, TensorRT, OpenVINO, CoreML, QNN |
| Dedicated NPUs | Vendor SDKs (Hailo, Qualcomm QNN, Rockchip) | Strict operator sets |

The usual path is PyTorch → ONNX (fixed opset, static shapes where possible) → vendor runtime.

```python
import torch
model.eval()
dummy = torch.randn(1, 3, 640, 640)
torch.onnx.export(model, dummy, "detector.onnx", opset_version=17,
                  input_names=["images"], output_names=["boxes"])
# then, on the Jetson:
# trtexec --onnx=detector.onnx --int8 --calib=calib.cache --saveEngine=detector.plan
```

**5. Optimise the whole pipeline, not just the model.** Resize, colour conversion and normalisation on the CPU in Python can cost more than inference. Use hardware decoders and zero-copy camera buffers, move pre-processing onto the GPU (or fold normalisation into the first layer), run NMS on the accelerator, batch across cameras, and pipeline capture, inference and post-processing on separate threads.

**6. Validate and operate.**

- Compare compiled-model outputs against the PyTorch reference on a golden set (mAP, not just a few images); quantisation errors often hit small objects first.
- Benchmark p50 and p99 latency on the device under sustained load, because thermal throttling appears after minutes.
- Ship models with version pinning and over-the-air updates with rollback, and log low-confidence samples back for retraining.

**Common failures:** pre-processing mismatch between training and device (BGR vs RGB, resize method, normalisation constants), unsupported operators silently falling back to CPU, calibrating INT8 on unrepresentative images, and dynamic shapes that disable optimisations.

### 15.47 How do you design a labelling strategy for a computer vision project, including active learning and handling label noise?

Treat labels as a product with a specification, a budget, a quality process and a feedback loop. Do not treat them as a one-off purchase. Most CV projects are limited by label consistency rather than model choice, so the strategy decides which images get labelled, how, by whom, and how errors are caught.

**1. Write the labelling specification before anyone labels.**

- Decide the **task granularity** the product needs: image tags are cheapest, boxes cost several times more, and polygons or masks cost much more again. Do not buy masks if boxes answer the business question.
- Define every class with positive and negative examples, plus the hard rules: occlusion, truncation at the image edge, minimum object size, groups ("crowd" flag), ambiguous cases, and an explicit "unsure" option.
- Run a **pilot**: 200 images labelled by 3 or more annotators, measure agreement (Cohen's or Fleiss' kappa for classes, IoU between annotators for boxes), and revise the guide until agreement stabilises. Inter-annotator agreement is roughly the ceiling for model performance.

**2. Choose what to label first.**

- Start with a **diverse** seed set, not random frames: deduplicate near-identical video frames, then cluster embeddings (DINOv2 or CLIP) and sample across clusters, sites, cameras, times of day and weather.
- Over-sample rare but costly cases deliberately.
- Freeze a **gold test set** early, labelled by experts with adjudication, and never use it for selection.

**3. Speed up labelling with models.** Pre-label with the current model, SAM for masks, or an open-vocabulary detector, and have humans correct. This often cuts cost by half or more. The risk is *anchoring*: annotators accept plausible but wrong pre-labels. Audit a sample labelled from scratch to measure that bias.

**4. Active learning loop.** Train, score the unlabelled pool, send the most valuable images to annotators, retrain, repeat.

- **Uncertainty sampling:** lowest max-probability, smallest margin between top two classes, highest entropy. For detection, aggregate box-level uncertainty per image.
- **Disagreement:** ensembles or MC-dropout (query-by-committee), or augmentation inconsistency.
- **Diversity:** core-set selection or k-means over embeddings, so a batch does not contain 500 copies of the same hard case.
- **Hybrid** (uncertainty within diverse clusters, as in BADGE) is the robust default.
- **Pitfalls:** pure uncertainty sampling picks outliers, blurred and unlabelable images; the selected set is biased, so evaluate only on the random gold set; and with a strong pretrained backbone, random sampling is a surprisingly hard baseline. Measure whether active learning beats random on your data before building the machinery.

**5. Handle label noise.**

- **Prevent:** clear guide, annotator training, qualification tests, gold questions mixed into tasks, and consensus or review for hard classes.
- **Detect:** *confident learning* (cleanlab-style) flags examples where out-of-fold predicted probability disagrees strongly with the given label; high-loss examples, annotator disagreement, and embedding outliers or duplicates with conflicting labels.
- **Review, do not auto-delete:** send flagged items back for adjudication; many "noisy" labels are actually rare valid cases.
- **Train robustly** when noise remains: label smoothing, early stopping (networks memorise noise late in training), robust losses such as generalised cross-entropy, or co-teaching with two networks selecting small-loss samples for each other.
- **Track noise per annotator and per class**; systematic confusion (two similar defect types) usually means the guide is wrong, not the people.

**6. Run it as a feedback loop.** Version labels with the guide version they followed, track cost per label and agreement over time, and route production low-confidence predictions, user corrections and drift alerts back into the queue.

**Concrete budget example:** 50,000 unlabelled images and money for 5,000 box labels. I would spend about 500 on a diverse seed set and gold test set, about 3,500 over five active-learning rounds of 700, and keep about 1,000 for re-labelling flagged noisy items and new classes that appear after launch.

---

← [Chapter 14. Natural Language Processing](14-natural-language-processing.md) · [Table of contents](../README.md) · [Chapter 16. Reinforcement Learning](16-reinforcement-learning.md) →
