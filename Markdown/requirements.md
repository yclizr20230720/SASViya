# Comprehensive Wafer Defect Analysis System - Requirements Specification

**Document Version:** 2.0 (Production-Ready)

**Author:** Manus AI

**Date:** January 17, 2026

**Status:** Ready for Development

---

## Executive Summary

This document provides a comprehensive, production-ready requirements specification for an AI-driven semiconductor wafer defect analysis system. It builds upon the initial draft requirements and adds critical technical specifications, architectural considerations, performance metrics, and real-world manufacturing constraints necessary for successful deployment in high-volume semiconductor fabrication facilities.

The system integrates deep learning models (YOLOv10, Vision Transformers, ResNet50), GAN-based synthetic data generation, GPU acceleration, and continuous learning capabilities to enable automated, high-accuracy defect pattern recognition at scale.

---

## 1. System Overview and Context

### 1.1 Problem Statement

Semiconductor manufacturers face critical challenges in wafer defect analysis:

- **Manual Inspection Limitations**: Human inspectors can only maintain peak performance for 2-3 hours before fatigue reduces accuracy to 85-90%
- **Data Scarcity**: Rare defect patterns lack sufficient training examples (often <100 samples per class)
- **Class Imbalance**: Common defects (random, local) vastly outnumber rare defects (edge-ring, scratch)
- **Process Variability**: Different tools, recipes, and process steps create diverse defect manifestations
- **Time Pressure**: Root cause analysis must occur within hours to enable rapid corrective actions
- **Scalability**: Modern fabs process 1000+ wafers per hour, making manual analysis infeasible

### 1.2 System Objectives

The system shall:

1. **Automate defect classification** with ≥98% accuracy on benchmark datasets
2. **Reduce false positives** from 40-50% (legacy systems) to <10%
3. **Enable real-time analysis** with <5 second latency per wafer
4. **Handle class imbalance** through synthetic data generation and advanced augmentation
5. **Provide explainability** for all classification decisions
6. **Support continuous learning** from production data and expert feedback
7. **Scale horizontally** to support 1000+ wafers/hour throughput
8. **Integrate seamlessly** with Manufacturing Execution Systems (MES)

---

## 2. Refined Requirement 1: Wafer Map Data Ingestion and Preprocessing

### 2.1 Data Format Support

**Primary Formats (MUST Support):**

| Format | Standard | Source Equipment | Priority |
| :--- | :--- | :--- | :--- |
| SEMI SECS/GEM | SEMI E5-0211 | Universal (all major tools) | Critical |
| CSV (Structured) | Custom | Generic inspection tools | Critical |
| HDF5 (Binary) | ISO/IEC 14596 | High-volume AOI systems | High |
| NetCDF | CF Conventions | Metrology systems | High |
| TIFF/PNG (Image) | ISO/IEC 12234 | E-beam, SEM systems | Medium |

**Secondary Formats (SHOULD Support):**

- KLA Proprietary Format (KLA Tencor inspection data)
- Applied Materials Proprietary Format (AMAT tools)
- ASML Proprietary Format (Lithography tools)
- Synopsys Odyssey Export Format
- Custom JSON/XML schemas

**Technical Specifications:**

```yaml
Data_Ingestion_Specifications:
  File_Size_Handling:
    Max_Single_File: 5 GB
    Batch_Upload: 100 GB per session
    Streaming_Support: Yes (for real-time feeds)
  
  Coordinate_Systems:
    - Cartesian (X, Y)
    - Polar (R, Theta) - for circular wafers
    - Die-based indexing (Row, Column)
    - Normalized [0, 1] range
  
  Metadata_Extraction:
    Required_Fields:
      - Lot ID (alphanumeric, max 20 chars)
      - Wafer ID (numeric, 1-25)
      - Process Step (layer name, max 50 chars)
      - Timestamp (ISO 8601 format)
      - Equipment ID (tool name, max 30 chars)
      - Recipe Name (process recipe, max 50 chars)
    
    Optional_Fields:
      - Wafer Diameter (200mm, 300mm, 450mm)
      - Die Size (microns)
      - Defect Density (defects/cm²)
      - Inspection Sensitivity Setting
      - Temperature/Humidity during inspection
      - Operator ID
      - Inspection Duration (seconds)
  
  Data_Quality_Checks:
    - File integrity verification (MD5/SHA256)
    - Coordinate range validation
    - Metadata completeness check
    - Duplicate detection (same lot/wafer/step)
    - Temporal consistency (timestamp ordering)
```

### 2.2 Preprocessing Pipeline

**Normalization and Coordinate Transformation:**

```yaml
Preprocessing_Steps:
  Step_1_Coordinate_Normalization:
    Input: Raw die coordinates (X, Y) in microns
    Process:
      - Identify wafer notch/flat orientation
      - Rotate to standard reference frame (notch at bottom)
      - Translate to center-based coordinates
      - Scale to [0, 1] range
    Output: Normalized coordinates
    Accuracy_Requirement: ±0.1% of wafer diameter
  
  Step_2_Die_Indexing:
    Input: Normalized coordinates
    Process:
      - Map to die row/column indices
      - Account for die size variations
      - Handle edge dies (partial dies)
    Output: Die-indexed defect locations
  
  Step_3_Image_Generation:
    Input: Die-indexed defects with bin codes
    Process:
      - Create 512x512 or 1024x1024 pixel wafer map image
      - Render defects as colored pixels (color = bin code)
      - Apply anti-aliasing for smooth edges
      - Generate heatmap overlay
    Output: Wafer map image (PNG/HDF5)
    Resolution: 1 pixel = 1 die (for standard 300mm wafer)
  
  Step_4_Metadata_Enrichment:
    Input: Extracted metadata
    Process:
      - Validate against equipment database
      - Map recipe to process layer
      - Lookup historical defect rates
      - Identify process window violations
    Output: Enriched metadata JSON
  
  Step_5_Data_Quality_Assessment:
    Input: Preprocessed data
    Checks:
      - Missing defect coordinates: Flag if >5%
      - Outlier detection: Identify unusual defect densities
      - Temporal anomalies: Check for retroactive data
      - Spatial anomalies: Detect impossible defect locations
    Output: Quality score (0-100) and anomaly flags
```

### 2.3 Batch Processing

**Specifications:**

```yaml
Batch_Processing:
  Batch_Size_Options:
    - Small: 1-10 wafers (interactive mode)
    - Medium: 10-100 wafers (daily processing)
    - Large: 100-1000 wafers (shift processing)
    - Continuous: Real-time stream (production mode)
  
  Progress_Tracking:
    - Real-time progress bar (percentage complete)
    - Estimated time remaining (ETA)
    - Current file being processed
    - Number of files completed/failed
    - Throughput (wafers/minute)
  
  Error_Handling:
    - Log all errors with timestamps
    - Continue processing on individual file failures
    - Generate error report with:
      - File name
      - Error type (parsing, validation, format)
      - Error message and stack trace
      - Suggested remediation
    - Retry mechanism (configurable, default 3 attempts)
  
  Parallelization:
    - Process multiple files concurrently (default 8 threads)
    - Configurable thread pool size
    - Memory-aware batching (prevent OOM)
    - GPU-accelerated preprocessing (optional)
```

### 2.4 Acceptance Criteria (Refined)

**AC 1.1: Multi-Format Support**
- GIVEN: User uploads wafer map in SEMI SECS/GEM format
- WHEN: System processes the file
- THEN: All required metadata extracted and coordinates normalized within ±0.1% accuracy
- VERIFICATION: Automated test suite with 50+ sample files from different equipment

**AC 1.2: Data Quality Metrics**
- GIVEN: Preprocessed wafer map
- WHEN: Quality assessment is performed
- THEN: System generates quality score (0-100) with breakdown:
  - Metadata completeness: 0-25 points
  - Coordinate validity: 0-25 points
  - Defect density plausibility: 0-25 points
  - Temporal consistency: 0-25 points
- VERIFICATION: Quality scores validated against manual inspection

**AC 1.3: Batch Processing Performance**
- GIVEN: Batch of 100 wafer maps (300mm, 512x512 images)
- WHEN: Batch processing is initiated
- THEN: All wafers processed within 60 seconds (600 ms/wafer)
- VERIFICATION: Performance benchmarks on target hardware

---

## 3. Refined Requirement 2: Defect Pattern Recognition and Classification

### 3.1 Defect Pattern Taxonomy

**Standard Defect Classes (9 Classes):**

| Class ID | Pattern Name | Characteristics | Typical Root Cause |
| :--- | :--- | :--- | :--- |
| 0 | None | No defects | N/A |
| 1 | Center | Circular cluster at wafer center | Lithography focus issue |
| 2 | Donut | Ring-shaped defect around center | Lithography aberration |
| 3 | Edge-Loc | Defects localized at wafer edge | Edge bead removal issue |
| 4 | Edge-Ring | Ring of defects near wafer edge | Centrifugal force effect |
| 5 | Loc (Local) | Small isolated clusters | Particle contamination |
| 6 | Random | Randomly scattered defects | Process instability |
| 7 | Scratch | Linear defect patterns | Mechanical damage |
| 8 | Near-Full | Defects covering >80% of wafer | Process failure |

**Mixed-Type Patterns (29 Additional Classes):**

Combinations of above patterns (e.g., Center+Donut, Edge-Ring+Random, etc.)

**Unknown/Novel Patterns:**

- Patterns not matching any trained class
- Confidence score <60% for all classes
- Require expert review and potential model retraining

### 3.2 Model Architecture Specifications

**Primary Model: YOLOv10-Medium**

```yaml
Model_Specification:
  Architecture: YOLOv10-Medium
  Input_Size: 640x640 pixels
  Output_Format:
    - Bounding boxes: [x1, y1, x2, y2, confidence, class_id]
    - Number of detections: Variable (0-100 per wafer)
  
  Performance_Requirements:
    Accuracy_Metrics:
      - mAP@50: ≥98.0%
      - mAP@50-95: ≥95.0%
      - Precision: ≥97.0%
      - Recall: ≥97.0%
    
    Speed_Metrics:
      - Inference_Time: ≤12 ms (batch size 32)
      - Throughput: ≥2,500 images/second
      - Latency_P99: ≤50 ms
    
    Hardware_Requirements:
      - GPU_Memory: 2-4 GB VRAM
      - GPU_Type: NVIDIA H100, A100, or L40
  
  Training_Configuration:
    - Epochs: 300
    - Batch_Size: 32
    - Learning_Rate: 0.001 (initial), cosine annealing
    - Optimizer: SGD with momentum 0.937
    - Weight_Decay: 0.0005
    - Data_Augmentation: Mosaic, mixup, HSV augmentation
```

**Secondary Model: DeiT-Tiny (Classification)**

```yaml
Model_Specification:
  Architecture: Data-Efficient Image Transformer (DeiT-Tiny)
  Input_Size: 224x224 pixels
  Output_Format:
    - Class_Probabilities: [p_0, p_1, ..., p_8] (9 classes)
    - Predicted_Class: argmax(probabilities)
    - Confidence: max(probabilities)
  
  Performance_Requirements:
    Accuracy_Metrics:
      - Top-1_Accuracy: ≥96.0%
      - Top-2_Accuracy: ≥99.0%
      - Per_Class_F1_Score: ≥0.94 (minimum)
    
    Speed_Metrics:
      - Inference_Time: ≤8 ms (batch size 32)
      - Throughput: ≥4,000 images/second
  
  Training_Configuration:
    - Epochs: 400
    - Batch_Size: 256
    - Learning_Rate: 0.001
    - Optimizer: AdamW
    - Weight_Decay: 0.05
    - Stochastic_Depth: 0.1
    - Dropout: 0.0
```

### 3.3 Multi-Pattern Detection

**Specifications:**

```yaml
Multi_Pattern_Detection:
  Approach: Ensemble voting with confidence thresholding
  
  Process:
    Step_1_Primary_Detection:
      - Run YOLOv10 to detect all defect regions
      - Extract bounding boxes with confidence scores
      - Filter detections with confidence > 0.5
    
    Step_2_Region_Classification:
      - For each detected region, extract 224x224 patch
      - Classify with DeiT-Tiny
      - Generate per-class probabilities
    
    Step_3_Pattern_Ranking:
      - Combine YOLOv10 detection confidence with DeiT classification confidence
      - Rank patterns by combined confidence score
      - Output top-3 patterns with scores
    
    Step_4_Spatial_Heatmap_Generation:
      - Create 512x512 heatmap image
      - Render each detected region with class-specific color
      - Overlay confidence scores
      - Generate spatial distribution statistics
  
  Output_Format:
    - Pattern_1: {class: "Center", confidence: 0.98, bbox: [x1, y1, x2, y2]}
    - Pattern_2: {class: "Donut", confidence: 0.92, bbox: [...]}
    - Pattern_3: {class: "Random", confidence: 0.87, bbox: [...]}
    - Heatmap_Image: PNG (512x512)
    - Spatial_Stats: {density, distribution, symmetry, ...}
```

### 3.4 Unknown Pattern Detection

**Specifications:**

```yaml
Unknown_Pattern_Detection:
  Trigger_Conditions:
    - Confidence_Score_All_Classes < 0.60
    - Entropy_of_Predictions > 2.0 (high uncertainty)
    - Spatial_Pattern_Anomaly_Score > 0.7
    - Visual_Dissimilarity_to_Training_Data > 0.8
  
  Handling_Process:
    - Flag wafer as "Requires_Expert_Review"
    - Store original wafer map with all model outputs
    - Create ticket in expert review queue
    - Assign priority based on defect density
    - Enable one-click labeling interface for expert
    - Upon labeling, add to retraining dataset
  
  Retraining_Trigger:
    - Accumulate 50+ labeled unknown patterns
    - Automatically trigger model retraining
    - Validate new model on held-out test set
    - Deploy if accuracy improvement ≥1%
```

### 3.5 Root Cause Mapping

**Specifications:**

```yaml
Root_Cause_Mapping:
  Mapping_Table:
    Center:
      - Primary: "Lithography focus error"
      - Secondary: "Resist uniformity issue"
      - Tertiary: "Exposure dose variation"
      - Recommended_Actions:
        - Check focus offset on stepper
        - Verify resist spin parameters
        - Inspect reticle cleanliness
    
    Donut:
      - Primary: "Lithography spherical aberration"
      - Secondary: "Lens contamination"
      - Tertiary: "Resist thickness variation"
      - Recommended_Actions:
        - Perform aberration correction
        - Schedule lens cleaning
        - Check resist deposition uniformity
    
    Edge_Loc:
      - Primary: "Edge bead removal (EBR) issue"
      - Secondary: "Wafer edge contamination"
      - Tertiary: "Spin speed miscalibration"
      - Recommended_Actions:
        - Adjust EBR parameters
        - Clean wafer edge
        - Verify spin speed calibration
    
    # ... (similar for other patterns)
  
  Confidence_Scoring:
    - Base confidence from model prediction
    - Adjust based on historical correlation
    - Consider process window violations
    - Output: Root_Cause_Score (0-100) per category
```

### 3.6 Acceptance Criteria (Refined)

**AC 2.1: Multi-Pattern Detection**
- GIVEN: Wafer map with 3 distinct defect patterns
- WHEN: Pattern recognition is performed
- THEN: All 3 patterns detected and ranked by confidence
- AND: Each pattern confidence score ≥0.85
- VERIFICATION: Manual validation on 100 test wafers

**AC 2.2: Accuracy Requirement**
- GIVEN: Test dataset of 1000 wafers (WM-811K)
- WHEN: Model inference is performed
- THEN: Overall accuracy ≥98.0%
- AND: Per-class accuracy ≥95.0% (minimum)
- AND: F1-score ≥0.94 (minimum across classes)
- VERIFICATION: Cross-validation on stratified splits

**AC 2.3: Unknown Pattern Handling**
- GIVEN: Novel defect pattern not in training data
- WHEN: System analyzes wafer map
- THEN: Pattern flagged as "Unknown" with confidence <0.60
- AND: Wafer routed to expert review queue
- AND: Expert can label and trigger retraining
- VERIFICATION: Manual injection of synthetic novel patterns

---

## 4. Refined Requirement 3: GAN-Based Synthetic Data Generation

### 4.1 GAN Architecture Selection

**Recommended Architecture: Conditional StyleGAN2**

**Rationale:**
- Superior image quality compared to DCGAN/WGAN-GP
- Supports class conditioning for targeted generation
- Proven performance on similar industrial datasets
- Better control over generated attributes (defect density, spatial distribution)

**Alternative Architectures (Fallback Options):**

1. **WGAN-GP (Wasserstein GAN with Gradient Penalty)**
   - Pros: Stable training, better convergence
   - Cons: Slower generation, lower quality
   - Use Case: When StyleGAN2 training fails

2. **Diffusion Models (DDPM)**
   - Pros: Highest quality, better diversity
   - Cons: Slower inference (100-500 steps)
   - Use Case: Offline batch generation

3. **Conditional GAN (cGAN)**
   - Pros: Simple, fast, good control
   - Cons: Lower quality, mode collapse risk
   - Use Case: Quick prototyping

### 4.2 GAN Training Specifications

```yaml
GAN_Training_Configuration:
  Generator_Architecture:
    Input: [Latent_Vector_512, Class_Embedding_128]
    Layers:
      - Dense: 512 → 4x4x512
      - StyleConv: 4x4 → 8x8 (512 channels)
      - StyleConv: 8x8 → 16x16 (512 channels)
      - StyleConv: 16x16 → 32x32 (256 channels)
      - StyleConv: 32x32 → 64x64 (128 channels)
      - StyleConv: 64x64 → 128x128 (64 channels)
      - StyleConv: 128x128 → 256x256 (32 channels)
      - StyleConv: 256x256 → 512x512 (16 channels)
      - Conv: 512x512 → 512x512 (3 channels, RGB output)
    Output: 512x512 RGB wafer map image
  
  Discriminator_Architecture:
    Input: 512x512 RGB image
    Layers:
      - Conv: 512x512 (3 channels) → 256x256 (16 channels)
      - Conv: 256x256 → 128x128 (32 channels)
      - Conv: 128x128 → 64x64 (64 channels)
      - Conv: 64x64 → 32x32 (128 channels)
      - Conv: 32x32 → 16x16 (256 channels)
      - Conv: 16x16 → 8x8 (512 channels)
      - Conv: 8x8 → 4x4 (512 channels)
      - Dense: 4x4x512 → 1 (real/fake score)
      - Dense: 1 → 9 (class logits)
    Output: [Authenticity_Score, Class_Logits]
  
  Training_Parameters:
    Batch_Size: 32
    Learning_Rate_Generator: 0.002
    Learning_Rate_Discriminator: 0.002
    Optimizer: Adam (beta1=0.0, beta2=0.99)
    Epochs: 500-1000
    Gradient_Penalty_Coefficient: 10
    Class_Loss_Weight: 0.1
  
  Conditioning_Mechanism:
    Defect_Density_Parameter:
      Range: [0.0, 1.0] (0% to 100% defect coverage)
      Embedding_Dimension: 64
      Injection_Points: All StyleConv layers
    
    Spatial_Distribution_Parameter:
      Options: ["center", "edge", "ring", "random", "local"]
      Embedding_Dimension: 64
      Injection_Points: All StyleConv layers
    
    Class_Embedding:
      Dimension: 128
      Method: One-hot encoding → Dense layer
      Injection_Points: Generator input, Discriminator final layer
```

### 4.3 Synthetic Data Quality Metrics

**Specifications:**

```yaml
Quality_Metrics:
  Fréchet_Inception_Distance_FID:
    Description: Measures distribution similarity between real and synthetic images
    Calculation: Distance between feature distributions (Inception V3)
    Threshold: FID < 50 (acceptable), FID < 30 (good), FID < 20 (excellent)
    Frequency: Compute every 50 training iterations
  
  Inception_Score_IS:
    Description: Measures image quality and diversity
    Calculation: KL divergence of class predictions
    Threshold: IS > 7.0 (acceptable), IS > 8.0 (good)
    Frequency: Compute every 50 training iterations
  
  Domain_Specific_Metrics:
    Defect_Density_Accuracy:
      Description: Generated defect density matches target ±5%
      Calculation: Count white pixels / total pixels
      Threshold: Accuracy ≥95%
    
    Spatial_Distribution_Fidelity:
      Description: Generated patterns match spatial characteristics
      Calculation: Compare spatial statistics (mean, variance, skewness)
      Threshold: Correlation ≥0.90 with real data
    
    Physical_Plausibility_Score:
      Description: Generated wafers satisfy manufacturing constraints
      Checks:
        - No defects in edge exclusion zone (5mm from edge)
        - Defect clustering follows Poisson distribution
        - Defect sizes within realistic range (1-100 microns)
      Threshold: Score ≥0.95 (95% of samples pass)
  
  Classifier_Performance_on_Synthetic:
    Description: Trained classifier accuracy on synthetic data
    Calculation: Run validation model on synthetic samples
    Threshold: Accuracy ≥95% (synthetic should be realistic)
    Purpose: Detect mode collapse or distribution shift
```

### 4.4 Data Provenance Tracking

**Specifications:**

```yaml
Data_Provenance_Tracking:
  Metadata_Per_Sample:
    - Sample_ID: Unique identifier (UUID)
    - Source_Type: "real" or "synthetic"
    - Generation_Date: Timestamp
    - GAN_Model_Version: Version hash
    - Conditioning_Parameters:
      - Defect_Density: 0.0-1.0
      - Spatial_Distribution: "center" | "edge" | ...
      - Class: 0-8 (defect class)
    - Quality_Metrics:
      - FID_Score: Float
      - IS_Score: Float
      - Physical_Plausibility: 0-1
    - Training_Usage:
      - Epoch_Used: Integer
      - Batch_Index: Integer
      - Loss_Contribution: Float
  
  Dataset_Composition_Tracking:
    - Real_Sample_Count: Integer
    - Synthetic_Sample_Count: Integer
    - Real_to_Synthetic_Ratio: Float (e.g., 0.5 = 1:2 ratio)
    - Per_Class_Distribution: Array[9]
    - Generation_Date_Range: [start_date, end_date]
    - GAN_Models_Used: List of model versions
  
  Versioning:
    - Dataset_Version: v1.0, v1.1, v2.0, etc.
    - Model_Version: Linked to training run
    - Changelog: Record of modifications
    - Reproducibility: Random seed, hyperparameters
```

### 4.5 Acceptance Criteria (Refined)

**AC 3.1: Synthetic Data Quality**
- GIVEN: GAN trained on real wafer map dataset
- WHEN: 1000 synthetic wafer maps are generated
- THEN: FID score <50 (acceptable) or <30 (good)
- AND: Physical plausibility score ≥0.95
- AND: Spatial distribution correlation ≥0.90 with real data
- VERIFICATION: Quantitative metrics + manual visual inspection

**AC 3.2: Classifier Performance on Synthetic**
- GIVEN: Trained defect classifier
- WHEN: Classifier evaluates 500 synthetic wafer maps
- THEN: Classification accuracy ≥95%
- VERIFICATION: Confusion matrix analysis

**AC 3.3: Data Provenance**
- GIVEN: Training dataset with mixed real/synthetic samples
- WHEN: Model is trained
- THEN: System maintains complete provenance for each sample
- AND: Can generate report showing real-to-synthetic ratio per epoch
- VERIFICATION: Audit trail verification

---

## 5. Refined Requirement 4: Advanced Data Augmentation Techniques

### 5.1 Augmentation Pipeline

**Specifications:**

```yaml
Augmentation_Pipeline:
  Geometric_Transformations:
    Rotation:
      - Angles: [0°, 90°, 180°, 270°] (wafer symmetry-aware)
      - Probability: 0.5
      - Interpolation: Bilinear
    
    Flipping:
      - Horizontal_Flip: Probability 0.5
      - Vertical_Flip: Probability 0.5
      - Note: Preserve defect semantics (e.g., "Edge-Loc" remains valid)
    
    Scaling:
      - Scale_Range: [0.8, 1.2]
      - Probability: 0.3
      - Preserve_Aspect_Ratio: True
  
  Defect_Aware_Augmentation:
    Defect_Density_Variation:
      - Multiply defect count by factor in [0.7, 1.3]
      - Probability: 0.4
      - Maintain spatial distribution pattern
    
    Spatial_Jittering:
      - Shift defect coordinates by ±5% of wafer diameter
      - Probability: 0.5
      - Preserve cluster structure
    
    Cluster_Size_Modulation:
      - Adjust defect cluster radius by factor in [0.8, 1.2]
      - Probability: 0.3
      - Maintain cluster density
  
  VAE_Based_Augmentation:
    Approach: Variational Autoencoder for latent space sampling
    Process:
      - Encode real wafer map to latent vector z
      - Sample z' from latent space (Gaussian perturbation)
      - Decode z' to generate variant wafer map
    Probability: 0.2
    Output: Diverse variants of input wafer
  
  Mixup_and_CutMix:
    Mixup:
      - Blend two wafer maps: w = λ * w1 + (1-λ) * w2
      - λ sampled from Beta(α=1.0, β=1.0)
      - Label: Weighted combination of labels
      - Probability: 0.3
    
    CutMix:
      - Cut rectangular region from w2, paste into w1
      - Region size: Random, 10-50% of image
      - Label: Weighted by region area
      - Probability: 0.3
  
  SMOTE_for_Spatial_Data:
    Approach: Synthetic Minority Oversampling Technique
    Process:
      - Identify minority class samples
      - Find k-nearest neighbors (k=5)
      - Interpolate between sample and neighbor
      - Generate synthetic samples
    Threshold: Apply when class has <100 samples
    Output: Balanced class distribution
  
  Augmentation_Strategy_Configuration:
    Per_Class_Customization:
      Center:
        - Geometric_Transforms: [Rotation, Scaling]
        - Defect_Aware: [Density_Variation, Cluster_Modulation]
        - Probability: 0.7
      
      Edge_Loc:
        - Geometric_Transforms: [Flipping]
        - Defect_Aware: [Spatial_Jittering]
        - Probability: 0.8 (rare class, more augmentation)
      
      Random:
        - Geometric_Transforms: [All]
        - Defect_Aware: [All]
        - Probability: 0.5
      
      # ... (similar for other classes)
```

### 5.2 Augmentation Validation

**Specifications:**

```yaml
Augmentation_Validation:
  Semantic_Preservation_Check:
    - Verify augmented sample maintains class label
    - Check defect pattern characteristics unchanged
    - Validate spatial distribution plausibility
    - Threshold: ≥95% of augmented samples pass
  
  Distribution_Matching:
    - Compare augmented vs. real data distributions
    - Metrics: Mean, variance, skewness, kurtosis
    - Threshold: KL divergence <0.1
  
  Classifier_Robustness_Test:
    - Train model with augmentation
    - Compare accuracy on:
      - Original test set
      - Augmented test set
      - Real production data
    - Threshold: Accuracy improvement ≥2%
```

### 5.3 Acceptance Criteria (Refined)

**AC 4.1: Augmentation Effectiveness**
- GIVEN: Training dataset with class imbalance (1:100 ratio)
- WHEN: Augmentation pipeline is applied
- THEN: Minority class samples increased by 5-10x
- AND: Augmented samples maintain semantic validity
- VERIFICATION: Manual inspection + classifier performance

**AC 4.2: Distribution Preservation**
- GIVEN: Augmented dataset
- WHEN: Statistical distributions compared
- THEN: KL divergence between augmented and real <0.1
- VERIFICATION: Statistical tests (Kolmogorov-Smirnov)

---

## 6. Refined Requirement 5: Model Training and Validation Framework

### 6.1 Training Pipeline

**Specifications:**

```yaml
Training_Pipeline:
  Data_Splitting_Strategy:
    Method: Stratified K-Fold (k=5)
    Stratification_By: Defect class
    
    Split_Ratios:
      Training_Set: 70%
      Validation_Set: 15%
      Test_Set: 15%
    
    Temporal_Consideration:
      - If temporal data available, use time-based split
      - Training: Older data
      - Validation: Recent data
      - Test: Most recent data
  
  Cross_Validation_Strategy:
    Method: 5-Fold Stratified Cross-Validation
    Metrics_Tracked:
      - Mean accuracy ± std deviation
      - Per-fold confusion matrices
      - Per-class F1 scores
    
    Threshold: Std deviation <2% (stable model)
  
  Class_Balancing_Techniques:
    Weighted_Loss_Function:
      - Compute class weights: w_i = N_total / (N_classes * N_i)
      - Apply to cross-entropy loss
      - Example: Rare class gets 10x weight
    
    Focal_Loss:
      - Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
      - α_t: Class weight
      - γ: Focusing parameter (default 2.0)
      - Reduces loss for easy examples
    
    Class_Balanced_Sampling:
      - Oversample minority classes
      - Undersample majority classes
      - Ratio: Configurable (default 1:1)
  
  Hyperparameter_Optimization:
    Method: Bayesian Optimization (Optuna)
    Search_Space:
      Learning_Rate: [1e-4, 1e-2]
      Batch_Size: [16, 128]
      Weight_Decay: [1e-5, 1e-3]
      Dropout_Rate: [0.0, 0.5]
      Optimizer: [Adam, SGD, AdamW]
    
    Optimization_Metric: Validation F1 score
    Number_of_Trials: 100
    Pruning_Strategy: Successive halving
```

### 6.2 Validation and Evaluation Metrics

**Specifications:**

```yaml
Evaluation_Metrics:
  Standard_Metrics:
    Accuracy: (TP + TN) / (TP + TN + FP + FN)
    Precision: TP / (TP + FP)
    Recall: TP / (TP + FN)
    F1_Score: 2 * (Precision * Recall) / (Precision + Recall)
    
    Per_Class_Metrics:
      - Compute for each of 9 defect classes
      - Report minimum, mean, maximum
      - Threshold: Minimum F1 ≥0.94
  
  Confusion_Matrix:
    - 9x9 matrix for 9 defect classes
    - Identify common misclassifications
    - Highlight problematic class pairs
  
  Domain_Specific_Metrics:
    Pattern_Detection_Rate:
      - Percentage of wafers with correct primary pattern identified
      - Threshold: ≥98%
    
    False_Alarm_Rate:
      - Percentage of wafers incorrectly flagged as defective
      - Threshold: <5%
    
    Root_Cause_Attribution_Accuracy:
      - Percentage of patterns correctly mapped to root cause
      - Threshold: ≥90%
    
    Multi_Pattern_Detection_Accuracy:
      - Percentage of wafers with multiple patterns correctly identified
      - Threshold: ≥95%
  
  Robustness_Metrics:
    Synthetic_vs_Real_Performance_Gap:
      - Accuracy on real test set - Accuracy on synthetic test set
      - Threshold: Gap <2% (indicates no overfitting to synthetic)
    
    Adversarial_Robustness:
      - Accuracy under adversarial perturbations (ε=0.05)
      - Threshold: ≥90% (robust to noise)
    
    Out_of_Distribution_Detection:
      - Percentage of novel patterns correctly flagged as unknown
      - Threshold: ≥95%
```

### 6.3 Model Versioning and Tracking

**Specifications:**

```yaml
Model_Versioning:
  Version_Format: v{major}.{minor}.{patch}
  
  Metadata_Per_Version:
    Model_ID: UUID
    Version_Number: String (e.g., "v2.1.3")
    Creation_Date: ISO 8601 timestamp
    Training_Duration: Hours
    
    Training_Data_Composition:
      Total_Samples: Integer
      Real_Samples: Integer
      Synthetic_Samples: Integer
      Per_Class_Distribution: Array[9]
      Real_to_Synthetic_Ratio: Float
    
    Hyperparameters:
      Learning_Rate: Float
      Batch_Size: Integer
      Optimizer: String
      Weight_Decay: Float
      Epochs: Integer
      Early_Stopping_Patience: Integer
    
    Performance_Metrics:
      Overall_Accuracy: Float
      Per_Class_F1_Scores: Array[9]
      Confusion_Matrix: 9x9 Array
      FID_Score: Float (if synthetic data used)
      Validation_Loss: Float
    
    Hardware_Info:
      GPU_Type: String (e.g., "NVIDIA H100")
      GPU_Count: Integer
      Training_Time: Hours
      Inference_Speed: ms/image
    
    Dataset_References:
      Training_Dataset_Version: String
      Validation_Dataset_Version: String
      Test_Dataset_Version: String
    
    Git_Commit_Hash: String (for code reproducibility)
    Docker_Image_Hash: String (for environment reproducibility)
  
  Model_Registry:
    - Central repository of all model versions
    - Metadata queryable (filter by date, accuracy, etc.)
    - Ability to compare versions side-by-side
    - Rollback capability (revert to previous version)
    - Deprecation tracking (mark old versions as obsolete)
```

### 6.4 Acceptance Criteria (Refined)

**AC 5.1: Cross-Validation Stability**
- GIVEN: 5-fold cross-validation on training data
- WHEN: Model is trained
- THEN: Standard deviation of fold accuracies <2%
- VERIFICATION: Cross-validation report

**AC 5.2: Per-Class Performance**
- GIVEN: Trained model on balanced dataset
- WHEN: Evaluated on test set
- THEN: Minimum per-class F1 score ≥0.94
- AND: No class with recall <0.92
- VERIFICATION: Per-class confusion matrix

**AC 5.3: Synthetic Data Validation**
- GIVEN: Model trained with 50% synthetic data
- WHEN: Evaluated on real-only test set
- THEN: Accuracy on real data ≥98%
- AND: Gap between synthetic and real test accuracy <2%
- VERIFICATION: Separate evaluation on real vs. synthetic test sets

---

## 7. Refined Requirement 6: Real-Time Inference and Integration

### 7.1 Inference Performance Requirements

**Specifications:**

```yaml
Inference_Performance:
  Latency_Requirements:
    Single_Image_Latency: ≤5 seconds (end-to-end)
    Breakdown:
      - Preprocessing: ≤500 ms
      - YOLOv10_Detection: ≤1000 ms
      - DeiT_Classification: ≤800 ms
      - Post_Processing: ≤500 ms
      - Heatmap_Generation: ≤1200 ms
    
    Batch_Processing_Latency:
      - Batch_Size_32: ≤12 ms per image (average)
      - Batch_Size_64: ≤15 ms per image (average)
      - Throughput: ≥2500 images/second
    
    P99_Latency: ≤50 ms per image (batch mode)
  
  Throughput_Requirements:
    Minimum_Throughput: 1000 wafers/hour
    Calculation: 1000 wafers/hour = 278 wafers/second = 0.278 seconds/wafer
    With_Preprocessing: ≤5 seconds/wafer (acceptable)
    
    Peak_Throughput: 2000+ wafers/hour (with dual GPU)
  
  Resource_Utilization:
    GPU_Memory: ≤4 GB per model (YOLOv10 + DeiT)
    CPU_Utilization: ≤50% (preprocessing/postprocessing)
    Memory_Bandwidth: ≤500 GB/s (acceptable on H100)
    Power_Consumption: ≤400W (dual H100)
```

### 7.2 Model Optimization and Deployment

**Specifications:**

```yaml
Model_Optimization:
  TensorRT_Optimization:
    Precision_Options:
      - FP32: Baseline (100% accuracy)
      - FP16: 1.5-2x speedup, <0.5% accuracy loss
      - INT8: 3-4x speedup, 1-2% accuracy loss
    
    Recommended_Configuration:
      - YOLOv10: FP16 (balance speed/accuracy)
      - DeiT: FP16 (classification less sensitive to precision)
      - Combined: 2-2.5x speedup
    
    Optimization_Process:
      1. Export model to ONNX format
      2. Build TensorRT engine with FP16 precision
      3. Calibrate with representative data (100 images)
      4. Benchmark on target GPU
      5. Validate accuracy drop <0.5%
  
  Deployment_Options:
    Option_1_TensorRT_Runtime:
      - Direct TensorRT engine execution
      - Fastest inference
      - Requires NVIDIA GPU
      - Recommended for production
    
    Option_2_ONNX_Runtime:
      - Cross-platform compatibility
      - Slightly slower than TensorRT
      - Supports CPU fallback
      - Recommended for flexibility
    
    Option_3_NVIDIA_Triton:
      - Multi-model serving
      - Dynamic batching
      - Load balancing
      - Recommended for high-volume production
```

### 7.3 API and Integration

**Specifications:**

```yaml
RESTful_API:
  Base_URL: https://wafer-analysis.fab.internal/api/v1
  
  Endpoints:
    POST_/inference/submit:
      Description: Submit wafer map for analysis
      Request_Body:
        - wafer_map_file: Binary (PNG/HDF5)
        - lot_id: String
        - wafer_id: Integer
        - process_step: String
        - priority: Enum [low, medium, high, critical]
      Response:
        - job_id: UUID
        - status: "queued"
        - estimated_wait_time: Seconds
      HTTP_Status: 202 Accepted
    
    GET_/inference/results/{job_id}:
      Description: Retrieve inference results
      Response:
        - job_id: UUID
        - status: Enum [queued, processing, completed, failed]
        - results: Object (if completed)
          - primary_pattern: String
          - confidence: Float
          - patterns: Array[{class, confidence, bbox}]
          - heatmap_url: URL
          - root_cause: String
          - recommended_actions: Array[String]
        - error_message: String (if failed)
      HTTP_Status: 200 OK
    
    GET_/inference/batch_status/{batch_id}:
      Description: Check batch processing status
      Response:
        - batch_id: UUID
        - total_wafers: Integer
        - completed: Integer
        - failed: Integer
        - in_progress: Integer
        - progress_percentage: Float
      HTTP_Status: 200 OK
    
    POST_/feedback/label:
      Description: Submit expert feedback/correction
      Request_Body:
        - job_id: UUID
        - corrected_pattern: String
        - confidence: Float
        - notes: String
      Response:
        - feedback_id: UUID
        - status: "recorded"
      HTTP_Status: 201 Created
  
  Authentication:
    - OAuth 2.0 with JWT tokens
    - Role-based access control (RBAC)
    - Roles: operator, engineer, data_scientist, admin
  
  Rate_Limiting:
    - 1000 requests/minute per API key
    - 100 concurrent jobs per user
    - Burst allowance: 50 requests/second
  
  Error_Handling:
    HTTP_Status_Codes:
      - 400: Bad request (invalid input)
      - 401: Unauthorized (authentication failed)
      - 403: Forbidden (insufficient permissions)
      - 404: Not found (job_id doesn't exist)
      - 429: Too many requests (rate limit exceeded)
      - 500: Internal server error
      - 503: Service unavailable
    
    Error_Response_Format:
      {
        "error_code": "INVALID_FILE_FORMAT",
        "error_message": "Uploaded file is not a valid wafer map",
        "details": "Expected PNG or HDF5, received JPEG",
        "timestamp": "2026-01-17T10:30:00Z"
      }
```

### 7.4 MES Integration

**Specifications:**

```yaml
MES_Integration:
  Integration_Protocol: REST API + Message Queue (RabbitMQ)
  
  Data_Flow:
    1_Wafer_Inspection_Complete:
      - MES sends inspection completion event
      - Includes: lot_id, wafer_id, process_step, wafer_map_file_path
      - Triggers: Automatic submission to inference system
    
    2_Inference_Processing:
      - System processes wafer map
      - Generates results with confidence scores
      - Identifies root cause category
    
    3_Results_Publication:
      - System publishes results to MES message queue
      - Includes: job_id, pattern, confidence, root_cause, recommended_actions
      - MES updates wafer record with results
    
    4_Alert_Generation:
      - If confidence <0.70 or unknown pattern: Route to expert review
      - If critical pattern detected: Generate high-priority alert
      - MES triggers corrective action workflow
  
  Message_Format:
    Inference_Request:
      {
        "request_id": "UUID",
        "timestamp": "ISO 8601",
        "lot_id": "LOT20260117001",
        "wafer_id": 5,
        "process_step": "LITHO_LAYER_5",
        "wafer_map_file": "s3://fab-data/wafer_maps/LOT20260117001_W05.hdf5",
        "priority": "high"
      }
    
    Inference_Result:
      {
        "request_id": "UUID",
        "job_id": "UUID",
        "timestamp": "ISO 8601",
        "status": "completed",
        "results": {
          "primary_pattern": "Center",
          "confidence": 0.98,
          "patterns": [
            {"class": "Center", "confidence": 0.98, "bbox": [...]},
            {"class": "Donut", "confidence": 0.85, "bbox": [...]}
          ],
          "root_cause": "Lithography focus error",
          "recommended_actions": [
            "Check focus offset on stepper",
            "Verify resist spin parameters"
          ],
          "heatmap_url": "s3://fab-data/results/job_UUID_heatmap.png"
        }
      }
  
  Alert_Configuration:
    Critical_Alerts:
      - Pattern: Near_Full (defects >80%)
      - Pattern: Scratch (linear defects)
      - Unknown patterns
      - Action: Immediate halt of wafer processing
    
    High_Priority_Alerts:
      - Confidence <0.70
      - Multiple patterns detected
      - Unusual root cause
      - Action: Route to expert review within 5 minutes
    
    Standard_Alerts:
      - All other patterns
      - Action: Log in MES, include in daily report
```

### 7.5 Acceptance Criteria (Refined)

**AC 6.1: Inference Latency**
- GIVEN: Single 300mm wafer map (512x512 image)
- WHEN: Submitted to inference system
- THEN: Results returned within 5 seconds
- AND: Breakdown: Preprocessing ≤500ms, Detection ≤1000ms, Classification ≤800ms
- VERIFICATION: End-to-end latency benchmarks

**AC 6.2: Throughput**
- GIVEN: Batch of 1000 wafers
- WHEN: Processed in parallel (batch size 32)
- THEN: All wafers processed within 400 seconds (2.5 wafers/second)
- AND: GPU utilization ≥85%
- VERIFICATION: Throughput benchmarks on target hardware

**AC 6.3: API Integration**
- GIVEN: MES system submits wafer map via REST API
- WHEN: Inference is performed
- THEN: Results published back to MES within 5 seconds
- AND: Alert generated for confidence <0.70
- VERIFICATION: End-to-end integration testing

---

## 8. Refined Requirement 7: Explainability and Root Cause Analysis

### 8.1 Explainability Techniques

**Specifications:**

```yaml
Explainability_Methods:
  Grad_CAM:
    Description: Gradient-weighted Class Activation Mapping
    Process:
      - Compute gradients of class score w.r.t. feature maps
      - Weight feature maps by gradients
      - Generate heatmap highlighting important regions
    Output: Spatial heatmap (same size as input image)
    Interpretation: Bright regions = high influence on classification
    
    Implementation:
      - Layer: Final convolutional layer before classification
      - Visualization: Overlay on original wafer map
      - Color_Map: Hot (red = high importance)
  
  SHAP_Values:
    Description: SHapley Additive exPlanations
    Process:
      - Compute contribution of each pixel to prediction
      - Use game theory to assign feature importance
      - Generate force plot showing positive/negative contributions
    Output: Per-pixel importance scores
    Interpretation: Positive values push toward predicted class
    
    Implementation:
      - Algorithm: KernelSHAP (model-agnostic)
      - Background_Data: 100 random wafer maps
      - Computation_Time: ~10 seconds per image
  
  Attention_Maps:
    Description: Visualization of transformer attention weights
    Process:
      - Extract attention weights from Vision Transformer
      - Visualize attention patterns across patches
      - Aggregate across attention heads
    Output: Attention heatmap
    Interpretation: Bright regions = model focus areas
    
    Implementation:
      - Layer: Final attention layer in DeiT
      - Aggregation: Average across 12 attention heads
      - Visualization: Overlay on wafer map
  
  Feature_Importance_Ranking:
    Description: Rank features by importance
    Process:
      - Extract features from penultimate layer
      - Compute correlation with prediction
      - Rank by absolute correlation
    Output: Top-10 most important features
    Interpretation: Features with highest influence on decision
```

### 8.2 Root Cause Analysis

**Specifications:**

```yaml
Root_Cause_Analysis:
  Historical_Case_Matching:
    Process:
      1. Extract features from current wafer map
      2. Search historical database for similar cases
      3. Retrieve top-5 most similar historical cases
      4. Display root causes from historical cases
      5. Show resolution actions taken
    
    Similarity_Metric: Cosine similarity in feature space
    Threshold: Similarity ≥0.85
    Database_Size: 10,000+ historical cases
  
  Statistical_Correlation_Analysis:
    Process:
      1. Identify pattern type and characteristics
      2. Query database for similar patterns
      3. Compute correlation with known root causes
      4. Rank root causes by correlation strength
      5. Display correlation confidence intervals
    
    Correlation_Metrics:
      - Pearson correlation coefficient
      - Chi-square test for categorical data
      - Mutual information for non-linear relationships
    
    Threshold: Correlation ≥0.70 (significant)
  
  Process_Window_Analysis:
    Process:
      1. Extract process parameters from metadata
      2. Compare against process window specifications
      3. Identify parameters outside nominal range
      4. Suggest parameter adjustments
    
    Parameters_Monitored:
      - Temperature (±2°C tolerance)
      - Pressure (±5% tolerance)
      - Flow rates (±3% tolerance)
      - Exposure dose (±5% tolerance)
      - Focus offset (±0.1 microns)
  
  Equipment_Correlation:
    Process:
      1. Extract equipment ID from metadata
      2. Query defect history for this equipment
      3. Identify patterns specific to this tool
      4. Suggest equipment maintenance if needed
    
    Maintenance_Triggers:
      - Defect rate increase >20% vs. baseline
      - New pattern type not seen before
      - Spatial clustering on specific chamber
```

### 8.3 Explainability Visualization

**Specifications:**

```yaml
Visualization_Dashboard:
  Layout:
    - Left_Panel: Original wafer map (512x512)
    - Center_Panel: Grad-CAM heatmap overlay
    - Right_Panel: Classification results + confidence
    - Bottom_Panel: Root cause analysis + recommendations
  
  Interactive_Features:
    - Hover_Over_Region: Show pixel-level importance
    - Click_On_Defect: Display historical similar cases
    - Toggle_Heatmap: Switch between Grad-CAM, SHAP, Attention
    - Zoom_In: Inspect specific defect regions
    - Export_Report: Generate PDF with all visualizations
  
  Quantitative_Features_Display:
    - Defect_Density: X defects/cm²
    - Spatial_Distribution: Center, Edge, Ring, Random (%)
    - Pattern_Symmetry: Symmetry score (0-1)
    - Clustering_Index: Spatial clustering strength
    - Confidence_Interval: 95% CI for classification
```

### 8.4 Active Learning Integration

**Specifications:**

```yaml
Active_Learning:
  Feedback_Collection:
    Interface:
      - Display model prediction + confidence
      - Show Grad-CAM heatmap
      - Provide correction interface
      - Allow free-text notes
    
    Feedback_Types:
      1. Confirm_Prediction: Expert agrees with model
      2. Correct_Prediction: Expert provides correct label
      3. Partial_Correction: Model partially correct
      4. Flag_as_Unknown: Pattern not in training set
  
  Feedback_Weighting:
    - Correct_Feedback: Weight = 1.0
    - Partial_Feedback: Weight = 0.5
    - Confirm_Feedback: Weight = 0.1
    - Unknown_Feedback: Weight = 0.0 (don't use for training)
  
  Retraining_Trigger:
    - Accumulate 100+ weighted feedback samples
    - OR: Accuracy drop detected on validation set
    - OR: New pattern type detected (10+ samples)
    - Automatic trigger for retraining workflow
  
  Retraining_Process:
    1. Merge feedback samples with training dataset
    2. Apply weighted sampling during training
    3. Train new model with updated data
    4. Validate on held-out test set
    5. If accuracy improvement ≥1%, deploy new model
    6. Archive old model for rollback capability
```

### 8.5 Acceptance Criteria (Refined)

**AC 7.1: Explainability Visualization**
- GIVEN: Classified wafer map with confidence 0.95
- WHEN: Explainability report is generated
- THEN: Grad-CAM heatmap highlights relevant defect regions
- AND: SHAP values show pixel-level importance
- AND: Attention maps show transformer focus areas
- VERIFICATION: Manual inspection of visualizations

**AC 7.2: Root Cause Mapping**
- GIVEN: Defect pattern classified as "Center"
- WHEN: Root cause analysis is performed
- THEN: Primary root cause identified as "Lithography focus error"
- AND: Confidence score ≥0.80
- AND: Similar historical cases displayed (top-5)
- VERIFICATION: Comparison with expert manual analysis

**AC 7.3: Active Learning**
- GIVEN: Expert provides feedback on 100 classifications
- WHEN: Feedback is accumulated and retraining triggered
- THEN: New model trained with updated data
- AND: Accuracy improvement ≥1% on validation set
- VERIFICATION: Model performance comparison

---

## 9. Refined Requirement 8: Continuous Learning and Model Improvement

### 9.1 Data Repository and Versioning

**Specifications:**

```yaml
Data_Repository:
  Storage_Architecture:
    - Primary: Cloud storage (AWS S3, Azure Blob, or on-prem equivalent)
    - Backup: Redundant storage (2x replication, different locations)
    - Archive: Long-term storage (Glacier/Archive tier)
    - Cache: Local SSD for frequently accessed data
  
  Data_Organization:
    Directory_Structure:
      /wafer_data/
        /raw/
          /2026-01/
            /LOT20260101001/
              W01_raw.hdf5
              W02_raw.hdf5
              ...
        /processed/
          /v1.0/
            /train/
              /class_0/ (None)
              /class_1/ (Center)
              ...
            /validation/
            /test/
        /synthetic/
          /gan_v1.0/
            /generated_2026_01_15/
              ...
        /feedback/
          /labeled_corrections/
          /expert_reviews/
  
  Metadata_Tracking:
    Per_Sample_Metadata:
      - Sample_ID: UUID
      - Source_Type: "real" | "synthetic" | "augmented"
      - Acquisition_Date: ISO 8601
      - Equipment_ID: String
      - Process_Step: String
      - Lot_ID: String
      - Wafer_ID: Integer
      - Ground_Truth_Label: Integer (0-8)
      - Labeling_Confidence: Float (0-1)
      - Expert_Reviewer: String (if manually labeled)
      - Review_Date: ISO 8601
      - Feedback_Count: Integer
      - Last_Used_In_Training: String (model version)
    
    Dataset_Version_Metadata:
      - Dataset_Version: String (v1.0, v1.1, v2.0)
      - Creation_Date: ISO 8601
      - Total_Samples: Integer
      - Per_Class_Distribution: Array[9]
      - Real_Samples: Integer
      - Synthetic_Samples: Integer
      - Augmented_Samples: Integer
      - Data_Quality_Score: Float (0-100)
      - Changelog: String (description of changes from previous version)
      - GAN_Model_Version: String (if synthetic data included)
```

### 9.2 Automated Retraining Workflow

**Specifications:**

```yaml
Retraining_Workflow:
  Trigger_Conditions:
    Condition_1_Sufficient_New_Data:
      - Accumulate 500+ new labeled samples
      - Trigger: Automatic check daily
      - Action: Initiate retraining
    
    Condition_2_Model_Drift_Detection:
      - Monitor validation accuracy on recent data
      - Threshold: Accuracy drop >2% over 1 week
      - Trigger: Automatic alert
      - Action: Initiate retraining
    
    Condition_3_New_Pattern_Emergence:
      - Detect 10+ samples of unknown pattern
      - Trigger: Automatic detection
      - Action: Initiate retraining with new class
    
    Condition_4_Manual_Trigger:
      - Data scientist manually initiates retraining
      - Trigger: Manual request
      - Action: Immediate retraining
  
  Retraining_Process:
    Step_1_Data_Preparation:
      - Merge new data with existing training set
      - Apply stratified train/val/test split
      - Generate new augmentation samples
      - Version new dataset
    
    Step_2_Model_Training:
      - Train YOLOv10 and DeiT models
      - Apply class balancing (weighted loss)
      - Monitor training/validation loss
      - Save checkpoints every 10 epochs
    
    Step_3_Validation:
      - Evaluate on validation set
      - Compute per-class metrics
      - Check for overfitting (training vs. validation gap)
      - Validate on real-only test set (if synthetic data used)
    
    Step_4_A_B_Testing:
      - Deploy new model to shadow environment
      - Run in parallel with production model
      - Collect inference results on 1000 wafers
      - Compare metrics:
        - Accuracy: New vs. Old
        - Confidence scores: Distribution comparison
        - False alarm rate: New vs. Old
      - Threshold: New model must improve ≥1% on at least one metric
    
    Step_5_Deployment_Decision:
      - If A/B test passes: Deploy new model to production
      - If A/B test fails: Archive new model, keep old model
      - Notify stakeholders of deployment
      - Log deployment metadata
    
    Step_6_Monitoring:
      - Monitor production accuracy for 1 week
      - If accuracy drops >1%: Rollback to previous model
      - Generate performance report
  
  Retraining_Frequency:
    - Minimum: Weekly (if sufficient data available)
    - Maximum: Daily (if model drift detected)
    - Typical: Bi-weekly (balanced approach)
```

### 9.3 Transfer Learning and Few-Shot Learning

**Specifications:**

```yaml
Transfer_Learning:
  Approach_1_Fine_Tuning:
    Process:
      1. Load pre-trained model (trained on WM-811K)
      2. Replace final classification layer
      3. Freeze early layers (feature extractor)
      4. Train only final layers on new data
      5. Gradually unfreeze layers if needed
    
    Advantages:
      - Requires fewer samples (100-500 per class)
      - Faster training (1-2 hours vs. 24+ hours)
      - Better generalization
    
    Use_Case: New process node or equipment type
  
  Approach_2_Few_Shot_Learning:
    Process:
      1. Use prototypical networks or matching networks
      2. Learn from few examples (5-10 per class)
      3. Adapt model to new classes quickly
    
    Advantages:
      - Minimal labeling effort
      - Rapid model updates
    
    Use_Case: Rare defect patterns, novel process variations
  
  Approach_3_Domain_Adaptation:
    Process:
      1. Train on source domain (existing fab data)
      2. Adapt to target domain (new fab or equipment)
      3. Use adversarial domain adaptation
      4. Minimize domain shift
    
    Advantages:
      - Transfers knowledge across fabs
      - Reduces labeling burden
    
    Use_Case: Multi-fab deployment
```

### 9.4 Model Drift Detection

**Specifications:**

```yaml
Model_Drift_Detection:
  Metrics_Monitored:
    Accuracy_Drift:
      - Compute accuracy on recent data (last 100 wafers)
      - Compare to baseline (training accuracy)
      - Threshold: Drop >2% triggers alert
    
    Confidence_Distribution_Shift:
      - Monitor distribution of confidence scores
      - Compute KL divergence vs. training distribution
      - Threshold: KL divergence >0.1 triggers alert
    
    Per_Class_Performance_Drift:
      - Monitor F1 score per class
      - Threshold: Drop >3% for any class triggers alert
    
    Defect_Density_Shift:
      - Monitor defect density distribution
      - Threshold: Shift >20% triggers alert
  
  Alert_Actions:
    Severity_Level_1_Minor_Drift:
      - Accuracy drop 1-2%
      - Action: Log alert, monitor closely
    
    Severity_Level_2_Moderate_Drift:
      - Accuracy drop 2-3%
      - Action: Alert data scientist, schedule retraining
    
    Severity_Level_3_Severe_Drift:
      - Accuracy drop >3%
      - Action: Immediate alert, consider rollback
    
    Severity_Level_4_Critical_Drift:
      - Accuracy drop >5%
      - Action: Automatic rollback to previous model
```

### 9.5 GAN Model Retraining

**Specifications:**

```yaml
GAN_Retraining:
  Trigger_Conditions:
    - New real data accumulated (500+ samples)
    - GAN quality degradation detected (FID >60)
    - New defect class requires synthetic samples
    - Scheduled retraining (monthly)
  
  Retraining_Process:
    Step_1_Data_Preparation:
      - Collect all new real wafer maps
      - Preprocess and normalize
      - Split into train/validation
    
    Step_2_GAN_Training:
      - Initialize from previous GAN weights
      - Train for 100-200 epochs
      - Monitor FID and IS scores
      - Save checkpoints every 20 epochs
    
    Step_3_Quality_Validation:
      - Compute FID on validation set
      - Threshold: FID <50 (acceptable)
      - Generate 100 synthetic samples
      - Visual inspection by data scientist
    
    Step_4_Synthetic_Data_Generation:
      - Generate synthetic samples for minority classes
      - Target: 5-10x augmentation for rare classes
      - Version new synthetic dataset
    
    Step_5_Integration:
      - Add synthetic data to training pool
      - Retrain classification models
      - Validate improvement on real test set
```

### 9.6 Acceptance Criteria (Refined)

**AC 8.1: Continuous Learning**
- GIVEN: 500 new labeled wafer maps accumulated
- WHEN: Retraining workflow is triggered
- THEN: New model trained within 24 hours
- AND: A/B testing performed on 1000 wafers
- AND: New model improves accuracy ≥1% before deployment
- VERIFICATION: Retraining workflow logs

**AC 8.2: Model Drift Detection**
- GIVEN: Production model deployed
- WHEN: Accuracy drops 2.5% on recent data
- THEN: Drift alert generated within 1 hour
- AND: Data scientist notified
- AND: Retraining recommended
- VERIFICATION: Alert system testing

**AC 8.3: Few-Shot Learning**
- GIVEN: 10 samples of new defect pattern
- WHEN: Few-shot learning applied
- THEN: Model adapts to new class within 2 hours
- AND: Accuracy ≥90% on new class
- VERIFICATION: Few-shot learning validation

---

## 10. Refined Requirement 9: System Performance and Scalability

### 10.1 Performance Requirements

**Specifications:**

```yaml
Performance_Requirements:
  Throughput:
    Minimum_Throughput: 1000 wafers/hour
    Peak_Throughput: 2000+ wafers/hour (dual GPU)
    Calculation: 1000 wafers/hour ÷ 3600 seconds = 0.278 wafers/second
    
    Batch_Processing_Throughput:
      - Batch_Size_32: 2500+ images/second
      - Batch_Size_64: 2000+ images/second
      - Batch_Size_128: 1500+ images/second
  
  Latency:
    End_to_End_Latency: ≤5 seconds per wafer
    Breakdown:
      - Preprocessing: ≤500 ms
      - Detection (YOLOv10): ≤1000 ms
      - Classification (DeiT): ≤800 ms
      - Post_Processing: ≤500 ms
      - Heatmap_Generation: ≤1200 ms
    
    P99_Latency: ≤50 ms (batch mode)
    P95_Latency: ≤30 ms (batch mode)
  
  Resource_Utilization:
    GPU_Memory: ≤4 GB per model
    GPU_Utilization: ≥85% (production)
    CPU_Utilization: ≤50% (preprocessing/postprocessing)
    Memory_Bandwidth: ≤500 GB/s
    Power_Consumption: ≤400W (dual H100)
  
  Availability:
    System_Uptime: ≥99.5% (monthly)
    Mean_Time_Between_Failures: ≥720 hours
    Mean_Time_To_Recovery: ≤5 minutes
    Failover_Time: ≤30 seconds (automatic)
```

### 10.2 Scalability Architecture

**Specifications:**

```yaml
Scalability_Architecture:
  Horizontal_Scaling:
    Containerization:
      - Docker containers for each service
      - Kubernetes orchestration
      - Auto-scaling based on load
      - Load balancing across instances
    
    Service_Decomposition:
      - Preprocessing_Service: Handles data normalization
      - Detection_Service: Runs YOLOv10
      - Classification_Service: Runs DeiT
      - PostProcessing_Service: Generates heatmaps
      - API_Service: REST endpoint
      - Database_Service: Results storage
    
    Scaling_Strategy:
      - Detection_Service: Scale by GPU count (1 GPU per instance)
      - Classification_Service: Scale by GPU count
      - Preprocessing_Service: Scale by CPU count (8 instances per GPU)
      - API_Service: Scale by request rate (1 instance per 100 req/s)
  
  Vertical_Scaling:
    GPU_Options:
      - Single_GPU: NVIDIA L40 (24 GB VRAM)
      - Dual_GPU: NVIDIA A100 (80 GB VRAM each)
      - Quad_GPU: NVIDIA H100 (141 TFLOPS each)
    
    CPU_Options:
      - Intel Xeon Platinum (16-32 cores)
      - AMD EPYC (32-64 cores)
    
    Memory_Options:
      - 128-512 GB DDR5
      - Scales with GPU count
  
  Load_Balancing:
    Strategy: Round-robin with health checks
    Health_Check_Interval: 5 seconds
    Failover_Threshold: 3 consecutive failures
    Sticky_Sessions: No (stateless design)
```

### 10.3 Storage and Data Management

**Specifications:**

```yaml
Storage_Management:
  Data_Storage:
    Hot_Storage (SSD):
      - Recent data (last 30 days)
      - Capacity: 10 TB
      - Access_Latency: <10 ms
      - Use_Case: Active processing, model training
    
    Warm_Storage (HDD):
      - Historical data (30 days - 1 year)
      - Capacity: 100 TB
      - Access_Latency: 50-100 ms
      - Use_Case: Archive, occasional queries
    
    Cold_Storage (Cloud Archive):
      - Long-term archive (>1 year)
      - Capacity: Unlimited
      - Access_Latency: Hours
      - Use_Case: Compliance, rare queries
  
  Compression:
    Wafer_Map_Images:
      - Format: PNG (lossless)
      - Compression_Ratio: 10:1 (typical)
      - Size_Per_Wafer: 50-200 KB
    
    Metadata:
      - Format: JSON (gzip compressed)
      - Compression_Ratio: 5:1
      - Size_Per_Wafer: 5-10 KB
    
    Total_Storage_Per_Wafer: 60-220 KB
    Annual_Storage_Requirement: 60-220 GB (for 1M wafers/year)
  
  Backup_Strategy:
    Frequency: Hourly incremental, daily full
    Replication: 3x (different geographic locations)
    Recovery_Time_Objective: 1 hour
    Recovery_Point_Objective: 15 minutes
```

### 10.4 Resource Prioritization

**Specifications:**

```yaml
Resource_Prioritization:
  Priority_Queue:
    Critical_Priority:
      - Wafers from high-value products
      - Wafers from critical process steps
      - Wafers with unknown patterns
      - Queue_Position: Front
      - Processing_Delay: <2 seconds
    
    High_Priority:
      - Wafers from new products
      - Wafers with suspected issues
      - Queue_Position: Second
      - Processing_Delay: <5 seconds
    
    Standard_Priority:
      - Regular production wafers
      - Queue_Position: Normal FIFO
      - Processing_Delay: <30 seconds
    
    Low_Priority:
      - Batch reprocessing
      - Historical data analysis
      - Queue_Position: Last
      - Processing_Delay: <5 minutes
  
  Resource_Allocation:
    Critical_Priority: 40% of GPU resources
    High_Priority: 30% of GPU resources
    Standard_Priority: 20% of GPU resources
    Low_Priority: 10% of GPU resources
    
    Dynamic_Reallocation:
      - If critical queue empty: Reallocate to high priority
      - If high queue empty: Reallocate to standard priority
      - Ensures resource utilization ≥90%
```

### 10.5 Acceptance Criteria (Refined)

**AC 9.1: Throughput Requirement**
- GIVEN: 1000 wafers submitted for processing
- WHEN: System processes all wafers
- THEN: All wafers processed within 3600 seconds (1 hour)
- AND: Average throughput ≥278 wafers/second
- VERIFICATION: Throughput benchmarks

**AC 9.2: Horizontal Scaling**
- GIVEN: System deployed on Kubernetes
- WHEN: Load increases from 500 to 2000 wafers/hour
- THEN: Additional GPU instances auto-provisioned
- AND: Processing latency remains <5 seconds
- AND: No wafers dropped or delayed
- VERIFICATION: Kubernetes metrics, load testing

**AC 9.3: Resource Prioritization**
- GIVEN: Mixed priority wafers in queue
- WHEN: System processes queue
- THEN: Critical priority wafers processed first
- AND: Critical wafers processed within 2 seconds
- AND: Standard priority wafers processed within 30 seconds
- VERIFICATION: Queue processing logs

---

## 11. Non-Functional Requirements

### 11.1 Security Requirements

**Specifications:**

```yaml
Security_Requirements:
  Authentication:
    - OAuth 2.0 with JWT tokens
    - Multi-factor authentication (MFA) for admin users
    - LDAP/Active Directory integration
  
  Authorization:
    - Role-based access control (RBAC)
    - Roles: operator, engineer, data_scientist, admin
    - Fine-grained permissions per API endpoint
  
  Data_Encryption:
    - In_Transit: TLS 1.3 (all network communication)
    - At_Rest: AES-256 (all stored data)
    - Key_Management: Hardware security module (HSM)
  
  Audit_Logging:
    - Log all API calls with user ID, timestamp, parameters
    - Log all model deployments and retraining events
    - Retention: 2 years
    - Tamper_Detection: Digital signatures on logs
  
  Compliance:
    - GDPR: Personal data anonymization
    - SOC 2 Type II: Security and availability controls
    - ISO 27001: Information security management
```

### 11.2 Reliability and Fault Tolerance

**Specifications:**

```yaml
Reliability:
  Redundancy:
    - Active-passive failover for critical services
    - Automatic failover within 30 seconds
    - No data loss during failover
  
  Error_Handling:
    - Graceful degradation (partial functionality if component fails)
    - Retry logic with exponential backoff
    - Circuit breaker pattern for external services
  
  Health_Monitoring:
    - Health checks every 5 seconds
    - Alerting for component failures
    - Automatic recovery attempts
```

### 11.3 Maintainability

**Specifications:**

```yaml
Maintainability:
  Code_Quality:
    - Code review for all changes
    - Unit test coverage ≥80%
    - Integration test coverage ≥60%
    - Static code analysis (SonarQube)
  
  Documentation:
    - API documentation (OpenAPI/Swagger)
    - Architecture documentation (C4 diagrams)
    - Deployment runbooks
    - Troubleshooting guides
  
  Logging:
    - Structured logging (JSON format)
    - Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    - Centralized log aggregation (ELK stack)
    - Log retention: 90 days
```

---

## 12. Implementation Roadmap

### Phase 1: Foundation (Months 1-2)

1. **Data Ingestion Module**
   - Support SEMI SECS/GEM and CSV formats
   - Implement preprocessing pipeline
   - Set up data repository

2. **Primary Model Development**
   - Train YOLOv10-Medium on WM-811K dataset
   - Achieve ≥98% accuracy
   - Optimize with TensorRT

3. **API Framework**
   - Implement REST API endpoints
   - Set up authentication/authorization
   - Deploy to development environment

### Phase 2: Core Functionality (Months 3-4)

1. **Classification Model**
   - Train DeiT-Tiny for defect classification
   - Achieve ≥96% accuracy
   - Integrate with detection model

2. **GAN-Based Synthetic Data**
   - Implement StyleGAN2 for wafer map generation
   - Generate synthetic data for minority classes
   - Validate synthetic data quality

3. **Advanced Augmentation**
   - Implement geometric transformations
   - Implement defect-aware augmentation
   - Implement VAE-based augmentation

### Phase 3: Production Readiness (Months 5-6)

1. **Explainability**
   - Implement Grad-CAM visualization
   - Implement SHAP value computation
   - Create explainability dashboard

2. **MES Integration**
   - Implement message queue integration
   - Set up alert generation
   - Test end-to-end integration

3. **Continuous Learning**
   - Implement feedback collection system
   - Implement automated retraining workflow
   - Set up model versioning and registry

### Phase 4: Optimization and Scaling (Months 7-8)

1. **Performance Optimization**
   - Profile and optimize inference latency
   - Implement batch processing
   - Achieve ≥2500 images/second throughput

2. **Scalability**
   - Deploy on Kubernetes
   - Implement auto-scaling policies
   - Test horizontal scaling

3. **Production Deployment**
   - Deploy to production environment
   - Set up monitoring and alerting
   - Conduct user training

---

## 13. Success Metrics and KPIs

| Metric | Target | Measurement |
| :--- | :--- | :--- |
| Classification Accuracy | ≥98% | Validation on test set |
| False Alarm Rate | <5% | Production monitoring |
| Inference Latency (P99) | ≤50 ms | Performance benchmarks |
| Throughput | ≥1000 wafers/hour | Production metrics |
| System Availability | ≥99.5% | Uptime monitoring |
| Model Retraining Frequency | Bi-weekly | Automated workflows |
| Expert Review Reduction | 50% | Manual review logs |
| Root Cause Accuracy | ≥90% | Expert validation |

---

## 14. Conclusion

This comprehensive requirements specification provides a detailed, production-ready blueprint for implementing an AI-driven wafer defect analysis system. The system integrates state-of-the-art deep learning models, advanced data augmentation techniques, GPU acceleration, and continuous learning capabilities to enable automated, high-accuracy defect pattern recognition at scale.

By following this specification, semiconductor manufacturers can achieve significant improvements in yield management, reduce manual inspection burden, and accelerate root cause analysis cycles.
