# Image Labeling & Ground Truth Management System - Implementation Complete

## Overview
Professional system for viewing, labeling, validating, and selectively exporting GAN-generated wafer map images for model training.

## Features Implemented

### 1. Image Display & Navigation
- **5×2 Grid Layout**: Displays exactly 10 images per page (5 columns × 2 rows)
- **Pagination**: Navigate through large image collections
- **Status Filtering**: Filter by All, Unlabeled, Labeled, Validated, Rejected
- **Image Zoom**: Click any image to view full-size in dialog
- **Error Handling**: Graceful fallback for missing images

### 2. Labeling Workflow
- **Label Dialog**: Interactive dialog with:
  - Pattern dropdown (9 defect patterns)
  - Confidence slider (0-1)
  - Notes field for additional context
- **Status Indicators**: Color-coded chips showing image status
- **Action Buttons**: Context-aware buttons based on status:
  - Unlabeled: "Label" button
  - Labeled: Validate (✓), Reject (✗), Edit buttons
  - Validated: "Re-label" + Undo (⚠️) buttons
  - Rejected: "Re-label" + Undo (✓) buttons

### 3. Validation System
- **Validate**: Mark labeled images as validated (ready for training)
- **Reject**: Mark images as rejected (quality issues)
- **Undo**: Reset validated/rejected images back to labeled status
- **Edit**: Modify existing labels

### 4. Checkbox Selection & Export ✨ NEW
- **Selective Export**: Choose which images to export to training
- **Checkbox Selection**: 
  - Individual checkboxes on each selectable image (top-left corner)
  - Only labeled and validated images can be selected
  - Unlabeled and rejected images are not selectable
- **Select All**: 
  - Checkbox in filter bar to select/deselect all selectable images on current page
  - Shows count of selectable images
  - Indeterminate state when some (but not all) are selected
- **Export Badge**: 
  - Green "Exported" badge on images that have been exported
  - Tracks export history with timestamp
- **Export Button**: 
  - Shows count of selected images: "Export Selected (3)"
  - Disabled when no images selected
  - Exports only selected images to training dataset
  - Clears selection after successful export

### 5. Statistics Dashboard
- **Total Images**: Count of all images in system
- **Unlabeled**: Count of images needing labels
- **Validated**: Count of validated images
- **Completion Rate**: Percentage with progress bar

### 6. Real-time Updates
- Statistics refresh after every action
- Image list refreshes after labeling/validation
- Selection state persists within page
- Selection clears when changing page/filter

## API Endpoints

### Backend (`wafer-defect-ap/app/api/v1/labeling.py`)

1. **GET /api/v1/labeling/images**
   - List images with pagination and filtering
   - Returns: filename, url, status, label, exported status, etc.

2. **GET /api/v1/labeling/image/{folder}/{filename}**
   - Serve image file

3. **POST /api/v1/labeling/label**
   - Label an image with pattern, confidence, notes

4. **POST /api/v1/labeling/batch-label**
   - Label multiple images at once

5. **POST /api/v1/labeling/validate/{filename}**
   - Validate, reject, or unlabel an image

6. **POST /api/v1/labeling/export** ✨ UPDATED
   - Export selected images to training dataset
   - Request body: `{ "filenames": ["img1.png", "img2.png"], "destination": "training" }`
   - Marks exported images with timestamp
   - Only exports labeled or validated images
   - Copies files with label prefix: `{pattern}_{filename}`

7. **GET /api/v1/labeling/statistics**
   - Get labeling statistics

8. **GET /api/v1/labeling/patterns**
   - Get available defect patterns

## File Structure

```
wafer-defect-gui/src/pages/training/
└── ImageLabeling.tsx          # Main component with checkbox selection

wafer-defect-ap/app/api/v1/
└── labeling.py                # API endpoints with selective export

wafer-defect-ap/data/
├── wafer_images/
│   ├── augmented/             # Source images (absolute path supported)
│   └── training/              # Exported training images
└── metadata/
    ├── image_labels.json      # Label data with export tracking
    └── export_*.json          # Export history logs
```

## Image Path Configuration

The system supports both relative and absolute paths:

**Absolute Path** (Windows):
```
C:\VSMC\VSMC-FAB_Investigayion_Tool\wafer-defect-ap\data\wafer_images\augmented
```

**Relative Path** (fallback):
```
wafer-defect-ap/data/wafer_images/augmented
```

Path detection is automatic in `config.py`.

## Usage Workflow

### Step 1: Label Images
1. Navigate to "Image Labeling" in Training menu
2. Filter by "Unlabeled" status
3. Click "Label" button on each image
4. Select defect pattern from dropdown
5. Adjust confidence if needed
6. Add notes (optional)
7. Click "Save Label"

### Step 2: Validate Labels
1. Filter by "Labeled" status
2. Review each labeled image
3. Click ✓ (Validate) if correct
4. Click ✗ (Reject) if incorrect
5. Click "Edit" to modify label

### Step 3: Select & Export for Training ✨ NEW
1. Filter by "Labeled" or "Validated" status
2. Use checkboxes to select images:
   - Click individual checkboxes on images
   - Or use "Select All" to select all on page
3. Review selection count in export button
4. Click "Export Selected (N)" button
5. Images are copied to training folder with label prefix
6. Exported images show green "Exported" badge
7. Selection is cleared after export

### Step 4: Monitor Progress
- Check statistics dashboard for completion rate
- Use filters to find images needing attention
- Refresh to see latest updates

## Defect Patterns

1. **Center**: Defects concentrated in wafer center
2. **Edge Ring**: Defects around wafer edge
3. **Edge Local**: Localized edge defects
4. **Local**: Localized defects
5. **Random**: Randomly distributed defects
6. **Scratch**: Scratch pattern
7. **Donut**: Ring-shaped defect pattern
8. **Near Full**: Nearly full wafer defects
9. **None/Good**: No defects detected

## Data Storage

### image_labels.json Structure
```json
{
  "images": [
    {
      "filename": "Center2_aug0.png",
      "folder": "augmented",
      "label": "Center",
      "confidence": 0.95,
      "status": "validated",
      "labeled_by": "current_user",
      "labeled_at": "2026-01-22T10:30:00",
      "validated_by": "current_user",
      "validated_at": "2026-01-22T10:35:00",
      "notes": "Clear center defect",
      "exported": true,
      "exported_at": "2026-01-22T10:40:00"
    }
  ]
}
```

### Export Metadata (export_training_*.json)
```json
{
  "exported_at": "2026-01-22T10:40:00",
  "destination": "training",
  "total_exported": 5,
  "images": [
    {
      "original": "Center2_aug0.png",
      "destination": "Center_Center2_aug0.png",
      "label": "Center",
      "confidence": 0.95
    }
  ]
}
```

## Key Features of Checkbox Selection

### Selection Rules
- ✅ **Can Select**: Labeled and Validated images
- ❌ **Cannot Select**: Unlabeled and Rejected images
- 📊 **Visual Feedback**: Checkboxes only appear on selectable images
- 🔄 **State Management**: Selection persists within page, clears on page/filter change

### Select All Behavior
- Selects only selectable images (labeled/validated)
- Shows count: "Select All (8 selectable)"
- Indeterminate state when partially selected
- Toggle: Click once to select all, click again to deselect all

### Export Badge
- Green "Exported" badge appears after export
- Positioned next to checkbox (if selectable) or top-left
- Tracks export history with timestamp
- Helps avoid duplicate exports

### Export Button
- Dynamic label: "Export Selected (N)"
- Disabled when N = 0
- Validates that only labeled/validated images are exported
- Shows success message with count
- Clears selection after successful export

## Technical Implementation

### Frontend State Management
```typescript
const [selectedImages, setSelectedImages] = useState<Set<string>>(new Set());

// Toggle individual image
const handleToggleImage = (filename: string, status: string) => {
  if (status !== 'labeled' && status !== 'validated') return;
  const newSelected = new Set(selectedImages);
  if (newSelected.has(filename)) {
    newSelected.delete(filename);
  } else {
    newSelected.add(filename);
  }
  setSelectedImages(newSelected);
};

// Select/deselect all
const handleSelectAll = () => {
  const selectableImages = images.filter(
    img => img.status === 'labeled' || img.status === 'validated'
  );
  if (selectedImages.size === selectableImages.length) {
    setSelectedImages(new Set());
  } else {
    setSelectedImages(new Set(selectableImages.map(img => img.filename)));
  }
};
```

### Backend Export Logic
```python
# Filter by filenames and validate status
images_to_export = [
    img for img in labels_data['images'] 
    if img['filename'] in filenames and img['status'] in ['labeled', 'validated']
]

# Mark as exported
img['exported'] = True
img['exported_at'] = datetime.now().isoformat()

# Save updated labels
storage.write('image_labels.json', labels_data)
```

## Testing

### Manual Testing Checklist
- [x] Images display in 5×2 grid
- [x] Pagination works correctly
- [x] Status filtering works
- [x] Label dialog opens and saves
- [x] Validate/Reject actions work
- [x] Undo actions work
- [x] Statistics update in real-time
- [x] Image zoom works
- [x] Checkboxes appear only on selectable images
- [x] Individual checkbox selection works
- [x] Select All checkbox works
- [x] Export button shows correct count
- [x] Export only exports selected images
- [x] Exported badge appears after export
- [x] Selection clears after export
- [x] Selection clears when changing page/filter

## Integration with Training Pipeline

Exported images are ready for model training:

1. **File Naming**: `{Pattern}_{OriginalFilename}.png`
   - Example: `Center_Center2_aug0.png`
   - Pattern prefix enables easy dataset organization

2. **Destination Folder**: `data/wafer_images/training/`
   - Separate from augmented images
   - Ready for training script consumption

3. **Metadata Tracking**: Export logs in `data/metadata/`
   - Audit trail of all exports
   - Timestamp and file list for each export

4. **Training Script Integration**:
   ```python
   # Training script can read from training folder
   training_dir = Path('data/wafer_images/training')
   for img_path in training_dir.glob('*.png'):
       # Extract label from filename prefix
       label = img_path.name.split('_')[0]
       # Load and train...
   ```

## Future Enhancements (Optional)

1. **Bulk Operations**: Select and label multiple images at once
2. **Keyboard Shortcuts**: Arrow keys for navigation, hotkeys for actions
3. **Advanced Filters**: Filter by label, confidence, date range
4. **Export Presets**: Save common export configurations
5. **Annotation Tools**: Draw regions of interest on images
6. **Collaborative Review**: Multi-user labeling with conflict resolution
7. **Quality Metrics**: Inter-rater agreement, confidence distributions
8. **Auto-labeling**: ML-assisted pre-labeling suggestions

## Conclusion

The Image Labeling & Ground Truth Management System is now complete with full checkbox selection and selective export functionality. Users can efficiently label, validate, and export wafer map images for training with fine-grained control over which images are included in the training dataset.

**Key Achievement**: Professional-grade labeling system with selective export that prevents duplicate exports and provides full audit trail.
