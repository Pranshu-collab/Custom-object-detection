# Custom-object-detection
## 1. Problem Definition and Motivation
The goal of this project is to design and implement a fully custom object detection model. Unlike approaches that rely on pretrained backbones or off-the-shelf detectors (e.g., YOLO, SSD, Faster R-CNN), this work focuses on building every component from scratch, including:
* A custom convolutional backbone
* Feature hierarchy design
* Skip connections and upsampling
* A grid-based detection head
* Bounding box regression formulation
* Post-processing (decoding and NMS)
* Custom evaluation using mAP
The motivation behind this design is to achieve full architectural control, enabling deep understanding, debugging, and adaptation for industrial deployment where pretrained models may not generalize well.

## 2. Overall System Architecture 
The proposed system follows a single-stage detection pipeline consisting of three main components:
* Custom CNN Backbone – for hierarchical feature extraction
* Detection Head with Upsampling and Skip Fusion – for spatial refinement
* Post-processing Pipeline – for converting raw predictions into final detections
The model takes a fixed-size RGB image as input and produces a dense grid of predictions, where each grid cell independently predicts object presence, bounding box geometry, and class probabilities.

## 3. Custom CNN Backbone Design
3.1 Input Specification
* Input shape: 224 × 224 × 3
* RGB images normalized before inference
* Fixed resolution chosen to ensure:
* Consistent tensor shapes
* Stable grid size in the detection head
* Predictable receptive field growth
  
3.2 Backbone Layer-by-Layer Design
The backbone is composed of four convolutional blocks, each designed to progressively increase semantic richness while reducing spatial resolution.
Block 1
* Convolution with a small number of filters
* Same padding to preserve spatial dimensions
* Batch normalization to stabilize training
* Max pooling to reduce resolution by a factor of 2
* Purpose:
  Capture low-level features such as edges, textures, and contrast variations that are critical for defect detection.
Block 2
* Increased number of convolution filters
* Same normalization and pooling strategy
* Purpose:
  Learn mid-level patterns such as contours, small shapes, and defect boundaries.
Block 3
* Further increase in channel depth
* Output of this block is explicitly stored as skip features
* Purpose:
  This layer retains high-resolution spatial information that would otherwise be lost in deeper layers. It is later reused during upsampling to improve localization accuracy.
Block 4
* Highest number of filters
* Final max pooling layer
* Purpose:
  Extract high-level semantic features indicating the presence or absence of defects.
  
3.3 Backbone Outputs
The backbone produces two outputs:
* Deep feature map – low spatial resolution, high semantic content
* Skip feature map – higher spatial resolution, lower semantic depth
This dual-output design enables multi-scale feature fusion without relying on pretrained feature pyramids.

## 4. Detection Head Design
4.1 Motivation for Upsampling
After multiple pooling operations, the deep feature map becomes too coarse for precise localization. To address this:
An upsampling layer is applied to increase spatial resolution
This enables the detector to operate on a denser grid
The final detection grid size is 28 × 28, meaning the image is divided into 784 cells.

4.2 Skip Feature Alignment
The skip features extracted from the backbone are:
* Passed through a 1×1 convolution to align channel dimensions
* Batch-normalized and activated
Reasoning:
* 1×1 convolution reduces channel mismatch
* Batch normalization ensures stable fusion
* Activation introduces non-linearity before fusion

4.3 Feature Fusion via Concatenation
The upsampled deep features and processed skip features are concatenated along the channel axis.
Advantages:
* Combines semantic understanding with spatial precision
* Improves small defect localization
* Avoids expensive multi-branch architectures
  
4.4 Final Prediction Layers
After fusion:
* A convolution layer refines the combined features
* A final 1×1 convolution produces the detection tensor
Each grid cell predicts:
* 4 bounding box parameters
* 1 objectness score
* C class logits
This results in an output tensor of shape:
28 × 28 × (5 + number_of_classes)
  
## 5. Bounding Box Representation and Prediction Logic
* Each grid cell predicts bounding boxes using a center-based parameterization:
* Center offsets are constrained using sigmoid activation
* Width and height are predicted using exponential scaling
* All coordinates are normalized relative to the image size
Why this formulation was chosen:
* Prevents negative widths and heights
* Ensures bounding boxes remain within image bounds
* Makes predictions resolution-independent

## 6. Post-Processing Pipeline
6.1 Decoding Predictions
Raw network outputs are transformed into real-world bounding boxes by:
* Applying activation functions
* Converting grid-relative predictions to image-relative coordinates
* Filtering predictions using a confidence threshold

6.2 Non-Maximum Suppression (NMS)
Since multiple grid cells may predict overlapping boxes:
* NMS removes redundant detections
* Highest-confidence boxes are retained
This step is essential for producing clean and interpretable detection outputs.

## 7. Training Methodology
7.1 Loss Function Structure
The training objective combines:
* Localization loss (bounding box regression)
* Objectness confidence loss
* Classification loss
Each component is weighted to balance:
* Accurate localization
* Robust background rejection
* Correct class assignment
  
7.2 Optimization Strategy
* Adaptive optimizer used for stable convergence
* Fixed batch size to maintain consistent gradient statistics
* Training performed for multiple epochs with validation monitoring

## 8. Evaluation Methodology
8.1 Intersection over Union (IoU)
IoU is computed using normalized coordinates output by the decoder. This ensures consistency between predictions and ground-truth annotations.

8.2 Mean Average Precision (mAP)
mAP is computed at a fixed IoU threshold:
* True positives identified using IoU matching
* Precision and recall computed per image
* Average precision aggregated across the dataset
Low initial mAP values were observed despite decreasing training loss, highlighting the difficulty of:
* Learning objectness scores
* Achieving stable localization early in training

## References:
* https://www.kaggle.com/code/muthumeenalv/object-detection/notebook
