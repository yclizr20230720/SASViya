# Training Data Structure & Usage Guide

## Overview
This document explains what data is used for model training in the AI Wafer Map Pattern Recognition System.

---

## Directory Structure

```
C:\VSMC\VSMC-FAB_Investigayion_Tool\wafer-defect-ap\data\wafer_images\
├── 📁 augmented/              # GAN-generated synthetic images (240+ files)
├── 📁 training/               # Exported & labeled images for training
├── 📄 Edge-Ring2.png          # Original wafer maps (12 files)
├── 📄 M12345.01_M93242.01_test_wafer.png
├── 📄 M34264.01_center1.png
├── 📄 M34264.01_Donut1.png
├── 📄 M34264.02_Edge_Loc1.png
├── 📄 M34264.03_Edge_Ring1.png
├── 📄 M34264.04_loc1.png
├── 📄 M34264.05_Near_Full1.png
├── 📄 M34264.06_Random1.png
├── 📄 M34264.07_Scrach1.png
├── 📄 M93242.01_M93242.01_test_wafer.png
└── 📄 M93242.01_test_wafer.png
```

---

## Data Categories

### 1. Original Wafer Maps (Root Directory)
**Location**: `C:\VSMC\VSMC-FAB_Investigayion_Tool\wafer-defect-ap\data\wafer_images\`

**Purpose**: Original seed images for GAN training and reference

**Files** (12 images):
- `Edge-Ring2.png` - Edge ring defect pattern
- `M12345.01_M93242.01_test_wafer.png` - Test wafer
- `M34264.01_center1.png` - Center defect pattern
- `M34264.01_Donut1.png` - Donut defect pattern
- `M34264.02_Edge_Loc1.png` - Edge local defect
- `M34264.03_Edge_Ring1.png` - Edge ring defect
- `M34264.04_loc1.png` - Local defect
- `M34264.05_Near_Full1.png` - Near full defect
- `M34264.06_Random1.png` - Random defect pattern
- `M34264.07_Scrach1.png` - Scratch pattern
- `M93242.01_M93242.01_test_wafer.png` - Test wafer
- `M93242.01_test_wafer.png` - Test wafer

**Usage**: 
- ❌ NOT directly used for model training
- ✅ Used as seed data for GAN to generate synthetic images
- ✅ Used for reference and validation

---

### 2. Augmented/Synthetic Images (augmented/)
**Location**: `C:\VSMC\VSMC-FAB_Investigayion_Tool\wafer-defect-ap\data\wafer_images\augmented\`

**Purpose**: GAN-generated synthetic wafer maps for data augmentation

**Files** (240+ images):
- `Center2_aug0.png` to `Center2_aug19.png` (20 images)
- `Edge-Ring2_aug0.png` to `Edge-Ring2_aug19.png` (20 images)
- `M12345.01_M93242.01_test_wafer_aug0.png` to `..._aug19.png` (20 images)
- `M34264.01_center1_aug0.png` to `..._aug19.png` (20 images)
- `M34264.01_Donut1_aug0.png` to `..._aug19.png` (20 images)
- `M34264.02_Edge_Loc1_aug0.png` to `..._aug19.png` (20 images)
- `M34264.03_Edge_Ring1_aug0.png` to `..._aug19.png` (20 images)
- `M34264.04_loc1_aug0.png` to `..._aug19.png` (20 images)
- `M34264.05_Near_Full1_aug0.png` to `..._aug19.png` (20 images)
- `M34264.06_Random1_aug0.png` to `..._aug19.png` (20 images)
- `M34264.07_Scrach1_aug0.png` to `..._aug19.png` (20 images)
- `M93242.01_M93242.01_test_wafer_aug0.png` to `..._aug19.png` (20 images)
- `M93242.01_test_wafer_aug0.png` to `..._aug19.png` (20 images)

**Naming Convention**: `{OriginalName}_aug{N}.png`
- Example: `Center2_aug5.png` = 5th augmented version of Center2

**Usage**:
- ⚠️ **REQUIRES LABELING** before training
- ✅ View in Image Labeling UI
- ✅ Label with defect pattern
- ✅ Validate labels
- ✅ Export to training folder

**Current Status**: 
- 📊 **Unlabeled**: Most images need labels
- 📊 **Labeled**: Some images have been labeled
- 📊 **Validated**: Some images have been validated
- 📊 **Exported**: 2 images exported to training folder

---

### 3. Training Dataset (training/)
**Location**: `C:\VSMC\VSMC-FAB_Investigayion_Tool\wafer-defect-ap\data\wafer_images\training\`

**Purpose**: ✅ **ACTUAL DATA USED FOR MODEL TRAINING**

**Files** (Currently 2 images):
- `Center_Center2_aug1.png` - Labeled as "Center" pattern
- `Center_Center2_aug14.png` - Labeled as "Center" pattern

**Naming Convention**: `{Pattern}_{OriginalFilename}.png`
- Pattern prefix indicates the ground truth label
- Example: `Center_Center2_aug1.png` = Center pattern, from Center2_aug1.png

**Usage**:
- ✅ **DIRECTLY USED** for model training
- ✅ Contains labeled and validated images
- ✅ Organized by pattern prefix for easy dataset creation
- ✅ Ready for PyTorch DataLoader

**How Images Get Here**:
1. User labels images in `augmented/` folder via Image Labeling UI
2. User validates the labels
3. User selects images with checkboxes
4. User clicks "Export Selected" button
5. System copies images to `training/` with pattern prefix
6. System marks images as "exported" to prevent duplicates

---

## Training Data Workflow

### Step 1: Generate Synthetic Data (GAN)
```
Original Images (12 files)
    ↓ [GAN Generation]
Augmented Images (240+ files in augmented/)
```

**Tools**:
- GAN Training UI: `/training/gan-training`
- Synthetic Generation UI: `/training/synthetic-generation`
- Script: `scripts/generate_synthetic_wafers.py`

---

### Step 2: Label & Validate Images
```
Augmented Images (unlabeled)
    ↓ [Manual Labeling]
Labeled Images (with pattern + confidence)
    ↓ [Validation]
Validated Images (ready for export)
```

**Tools**:
- Image Labeling UI: `/training/image-labeling`
- Actions: Label → Validate → Select → Export

**Defect Patterns** (9 classes):
1. Center
2. Edge-Ring
3. Edge-Loc
4. Loc (Local)
5. Random
6. Scratch
7. Donut
8. Near-Full
9. None (Good)

---

### Step 3: Export to Training Dataset
```
Validated Images (in augmented/)
    ↓ [Checkbox Selection + Export]
Training Images (in training/)
```

**Export Process**:
1. Filter by "Labeled" or "Validated" status
2. Select images using checkboxes
3. Click "Export Selected (N)" button
4. Images copied to `training/` with pattern prefix
5. Export metadata saved to `data/metadata/export_*.json`

**Export Metadata Example**:
```json
{
  "exported_at": "2026-01-22T10:40:00",
  "destination": "training",
  "total_exported": 2,
  "images": [
    {
      "original": "Center2_aug1.png",
      "destination": "Center_Center2_aug1.png",
      "label": "Center",
      "confidence": 0.95
    }
  ]
}
```

---

### Step 4: Model Training
```
Training Images (in training/)
    ↓ [PyTorch DataLoader]
Model Training
    ↓ [Checkpoints]
Trained Model (in data/models/)
```

**Training Script**: `scripts/train_model.py`

**Dataset Loading**:
```python
from app.ml.dataset import WaferDefectDataset

# Load training data
dataset = WaferDefectDataset(
    data_dir='data/wafer_images/training',
    metadata_file='data/processed/train_metadata.json',
    transform=WaferPreprocessor(target_size=(224, 224))
)

# Create DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

**Metadata Format** (`train_metadata.json`):
```json
[
  {
    "image_filename": "Center_Center2_aug1.png",
    "image_path": "data/wafer_images/training/Center_Center2_aug1.png",
    "pattern_class": "Center",
    "root_cause": "Process",
    "confidence": 0.95,
    "labeled_by": "current_user",
    "labeled_at": "2026-01-22T10:30:00"
  }
]
```

---

## Current Training Data Status

### Summary Statistics
```
📊 Original Images:     12 files
📊 Augmented Images:    240+ files
📊 Training Images:     2 files ⚠️ (NEEDS MORE DATA)
📊 Labeled Images:      ~10 files
📊 Validated Images:    ~5 files
📊 Unlabeled Images:    ~230 files
```

### ⚠️ **CRITICAL**: Training Dataset is Too Small!

**Current**: Only 2 images in training folder
**Minimum Required**: 100-200 images per class (900-1800 total)
**Recommended**: 500+ images per class (4500+ total)

**Action Required**:
1. ✅ Label more images in `augmented/` folder
2. ✅ Validate the labels
3. ✅ Export to training folder
4. ✅ Repeat until sufficient data

---

## Data Requirements for Training

### Minimum Dataset Size
| Pattern Class | Minimum | Recommended | Current |
|--------------|---------|-------------|---------|
| Center       | 100     | 500         | 2 ⚠️    |
| Edge-Ring    | 100     | 500         | 0 ⚠️    |
| Edge-Loc     | 100     | 500         | 0 ⚠️    |
| Loc          | 100     | 500         | 0 ⚠️    |
| Random       | 100     | 500         | 0 ⚠️    |
| Scratch      | 100     | 500         | 0 ⚠️    |
| Donut        | 100     | 500         | 0 ⚠️    |
| Near-Full    | 100     | 500         | 0 ⚠️    |
| None         | 100     | 500         | 0 ⚠️    |
| **TOTAL**    | **900** | **4500**    | **2** ⚠️ |

### Data Split Ratios
- **Training**: 70% (630-3150 images)
- **Validation**: 15% (135-675 images)
- **Testing**: 15% (135-675 images)

---

## How to Prepare Training Data

### Quick Start Guide

#### 1. Access Image Labeling UI
```
http://localhost:5173/training/image-labeling
```

#### 2. Label Images
1. Filter by "Unlabeled"
2. Click "Label" on each image
3. Select defect pattern from dropdown
4. Set confidence (0.8-1.0 for clear patterns)
5. Add notes if needed
6. Click "Save Label"

#### 3. Validate Labels
1. Filter by "Labeled"
2. Review each labeled image
3. Click ✓ (Validate) if correct
4. Click ✗ (Reject) if incorrect
5. Click "Edit" to modify label

#### 4. Export to Training
1. Filter by "Validated"
2. Check boxes to select images
3. Or click "Select All" for all on page
4. Click "Export Selected (N)"
5. Confirm export success

#### 5. Repeat Until Sufficient Data
- Target: 100+ images per class minimum
- Recommended: 500+ images per class
- Monitor statistics dashboard for progress

---

## Training Data Quality Guidelines

### Good Training Images ✅
- Clear defect patterns
- High confidence labels (0.9-1.0)
- Validated by human reviewer
- Diverse variations within pattern class
- Good image quality (no artifacts)

### Poor Training Images ❌
- Ambiguous patterns
- Low confidence labels (<0.7)
- Not validated
- Duplicate or near-duplicate images
- Poor image quality (blurry, corrupted)

### Labeling Best Practices
1. **Consistency**: Use same criteria for all images
2. **Confidence**: Be honest about uncertainty
3. **Notes**: Document unusual cases
4. **Validation**: Always validate before export
5. **Review**: Periodically review exported data

---

## Training Pipeline Integration

### Automatic Training Trigger
Once sufficient data is exported, the training pipeline can be triggered:

```python
# Check if enough data for training
from pathlib import Path

training_dir = Path('data/wafer_images/training')
image_count = len(list(training_dir.glob('*.png')))

if image_count >= 900:  # Minimum threshold
    print("✅ Sufficient data for training!")
    # Trigger training job
else:
    print(f"⚠️ Need {900 - image_count} more images")
```

### Training Job Creation
```bash
# Via API
curl -X POST http://localhost:5000/api/v1/training/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "wafer_defect_v1",
    "data_dir": "data/wafer_images/training",
    "epochs": 50,
    "batch_size": 32
  }'

# Via UI
http://localhost:5173/training/job-queue
```

---

## Metadata Files

### Image Labels (`data/metadata/image_labels.json`)
Tracks all labeled images with status and export history:
```json
{
  "images": [
    {
      "filename": "Center2_aug1.png",
      "folder": "augmented",
      "label": "Center",
      "confidence": 0.95,
      "status": "validated",
      "labeled_by": "current_user",
      "labeled_at": "2026-01-22T10:30:00",
      "validated_by": "current_user",
      "validated_at": "2026-01-22T10:35:00",
      "exported": true,
      "exported_at": "2026-01-22T10:40:00",
      "notes": "Clear center defect pattern"
    }
  ]
}
```

### Training Metadata (`data/processed/train_metadata.json`)
Generated from exported images for PyTorch DataLoader:
```json
[
  {
    "image_filename": "Center_Center2_aug1.png",
    "image_path": "data/wafer_images/training/Center_Center2_aug1.png",
    "pattern_class": "Center",
    "root_cause": "Process",
    "confidence": 0.95,
    "labeled_by": "current_user",
    "labeled_at": "2026-01-22T10:30:00"
  }
]
```

### Export History (`data/metadata/export_training_*.json`)
Audit trail of all exports:
```json
{
  "exported_at": "2026-01-22T10:40:00",
  "destination": "training",
  "total_exported": 2,
  "images": [
    {
      "original": "Center2_aug1.png",
      "destination": "Center_Center2_aug1.png",
      "label": "Center",
      "confidence": 0.95
    }
  ]
}
```

---

## Summary

### ✅ What IS Used for Training
- **Training Folder**: `data/wafer_images/training/`
- **Files**: Images with pattern prefix (e.g., `Center_*.png`)
- **Current Count**: 2 images ⚠️ (INSUFFICIENT)
- **Required**: 900+ images minimum, 4500+ recommended

### ❌ What is NOT Used for Training
- **Original Images**: Root directory (12 files) - used for GAN seed only
- **Augmented Images**: `augmented/` folder (240+ files) - requires labeling first

### 🎯 Action Items
1. **Label** 240+ images in augmented folder
2. **Validate** all labeled images
3. **Export** validated images to training folder
4. **Monitor** progress in statistics dashboard
5. **Train** model once sufficient data is ready

### 📊 Progress Tracking
- Use Image Labeling UI statistics dashboard
- Target: 100+ images per class (900+ total)
- Current: 2 images (0.2% of minimum)
- **Completion Rate**: 0.2% ⚠️

---

## Quick Commands

### Check Training Data Count
```bash
# Windows
dir /b data\wafer_images\training\*.png | find /c ".png"

# PowerShell
(Get-ChildItem data\wafer_images\training\*.png).Count
```

### Check Augmented Data Count
```bash
# Windows
dir /b data\wafer_images\augmented\*.png | find /c ".png"

# PowerShell
(Get-ChildItem data\wafer_images\augmented\*.png).Count
```

### View Export History
```bash
# List all exports
dir data\metadata\export_*.json

# View latest export
type data\metadata\export_training_*.json
```

---

## Conclusion

**Training data comes from**: `data/wafer_images/training/` folder

**Current status**: Only 2 images - **INSUFFICIENT FOR TRAINING**

**Next steps**: 
1. Label images in Image Labeling UI
2. Export to training folder
3. Repeat until 900+ images
4. Start model training

The system is ready, but needs more labeled data to train an effective model!
