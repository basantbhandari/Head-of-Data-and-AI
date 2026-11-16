# Chapter 15. Computer Vision

_Part III — Machine Learning_

CNNs, modern vision backbones, detection and segmentation, augmentation and vision pipelines.

This chapter contains **32 questions**.

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

---

← [Chapter 14. Natural Language Processing](14-natural-language-processing.md) · [Table of contents](../README.md) · [Chapter 16. Reinforcement Learning](16-reinforcement-learning.md) →
