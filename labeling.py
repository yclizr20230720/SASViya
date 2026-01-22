"""
Image Labeling API Endpoints
Allows users to label and validate synthetic wafer maps
"""
from flask import request, jsonify, current_app, send_file
from app.api.v1 import api_v1_bp
from app.utils.json_storage import JSONStorage
from pathlib import Path
from datetime import datetime
import os
import shutil


@api_v1_bp.route('/labeling/images', methods=['GET'])
def list_unlabeled_images():
    """
    List all unlabeled synthetic wafer map images
    
    Query Parameters:
        - folder: Folder to scan (default: augmented)
        - status: Filter by status (unlabeled, labeled, validated, rejected)
        - limit: Number of images to return (default: 50)
        - offset: Pagination offset (default: 0)
    
    Returns:
        JSON response with list of images and metadata
    """
    try:
        storage = JSONStorage(current_app.config['METADATA_FOLDER'])
        
        # Get query parameters
        folder = request.args.get('folder', 'augmented')
        status = request.args.get('status', 'unlabeled')
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        # Get image directory - try configured path first
        image_dir = Path(current_app.config['UPLOAD_FOLDER']) / folder
        
        # If not found, try absolute path
        if not image_dir.exists():
            abs_path = Path(r'C:\VSMC\VSMC-FAB_Investigayion_Tool\wafer-defect-ap\data\wafer_images') / folder
            if abs_path.exists():
                image_dir = abs_path
        
        if not image_dir.exists():
            current_app.logger.error(f"Image directory not found: {image_dir}")
            return jsonify({'error': 'Image directory not found', 'path': str(image_dir)}), 404
        
        # Load existing labels
        labels_data = storage.read('image_labels.json', default={'images': []})
        labeled_images = {img['filename']: img for img in labels_data['images']}
        
        # Scan directory for images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(image_dir.glob(ext))
        
        current_app.logger.info(f"Found {len(image_files)} images in {image_dir}")
        
        # Build response
        images = []
        for img_path in sorted(image_files):
            filename = img_path.name
            
            # Get label info if exists
            label_info = labeled_images.get(filename, {})
            img_status = label_info.get('status', 'unlabeled')
            
            # Filter by status
            if status != 'all' and img_status != status:
                continue
            
            images.append({
                'filename': filename,
                'path': str(img_path),
                'url': f'/labeling/image/{folder}/{filename}',  # Removed /api/v1 prefix since blueprint already has it
                'size': img_path.stat().st_size,
                'created_at': datetime.fromtimestamp(img_path.stat().st_ctime).isoformat(),
                'status': img_status,
                'label': label_info.get('label'),
                'confidence': label_info.get('confidence'),
                'labeled_by': label_info.get('labeled_by'),
                'labeled_at': label_info.get('labeled_at'),
                'notes': label_info.get('notes'),
                'exported': label_info.get('exported', False),
                'exported_at': label_info.get('exported_at')
            })
        
        # Pagination
        total = len(images)
        images = images[offset:offset + limit]
        
        return jsonify({
            'status': 'success',
            'images': images,
            'total': total,
            'limit': limit,
            'offset': offset,
            'has_more': offset + limit < total
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"List images error: {str(e)}")
        return jsonify({"error": f"Failed to list images: {str(e)}"}), 500


@api_v1_bp.route('/labeling/image/<folder>/<filename>', methods=['GET'])
def get_image(folder, filename):
    """
    Serve wafer map image file
    
    Parameters:
        - folder: Image folder (augmented, original, etc.)
        - filename: Image filename
    
    Returns:
        Image file
    """
    try:
        # Try configured path first
        image_path = Path(current_app.config['UPLOAD_FOLDER']) / folder / filename
        
        # If not found, try absolute path
        if not image_path.exists():
            # Try absolute path for Windows
            abs_path = Path(r'C:\VSMC\VSMC-FAB_Investigayion_Tool\wafer-defect-ap\data\wafer_images') / folder / filename
            if abs_path.exists():
                image_path = abs_path
        
        if not image_path.exists():
            current_app.logger.error(f"Image not found: {image_path}")
            return jsonify({'error': 'Image not found'}), 404
        
        return send_file(str(image_path), mimetype='image/png')
    
    except Exception as e:
        current_app.logger.error(f"Get image error: {str(e)}")
        return jsonify({"error": f"Failed to get image: {str(e)}"}), 500


@api_v1_bp.route('/labeling/label', methods=['POST'])
def label_image():
    """
    Label a wafer map image
    
    Request Body:
        {
            "filename": "image.png",
            "folder": "augmented",
            "label": "Center",
            "confidence": 0.95,
            "labeled_by": "user@example.com",
            "notes": "Clear center defect pattern"
        }
    
    Returns:
        JSON response with success status
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required = ['filename', 'label']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        storage = JSONStorage(current_app.config['METADATA_FOLDER'])
        
        # Load existing labels
        labels_data = storage.read('image_labels.json', default={'images': []})
        
        # Create label record
        label_record = {
            'filename': data['filename'],
            'folder': data.get('folder', 'augmented'),
            'label': data['label'],
            'confidence': data.get('confidence', 1.0),
            'status': 'labeled',
            'labeled_by': data.get('labeled_by', 'unknown'),
            'labeled_at': datetime.now().isoformat(),
            'notes': data.get('notes', ''),
            'metadata': {
                'original_filename': data.get('original_filename'),
                'generation_method': data.get('generation_method', 'manual')
            }
        }
        
        # Update or add label
        existing_idx = next(
            (i for i, img in enumerate(labels_data['images']) 
             if img['filename'] == data['filename']), 
            None
        )
        
        if existing_idx is not None:
            labels_data['images'][existing_idx] = label_record
        else:
            labels_data['images'].append(label_record)
        
        # Save labels
        storage.write('image_labels.json', labels_data)
        
        return jsonify({
            'status': 'success',
            'message': 'Image labeled successfully',
            'label': label_record
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Label image error: {str(e)}")
        return jsonify({"error": f"Failed to label image: {str(e)}"}), 500


@api_v1_bp.route('/labeling/batch-label', methods=['POST'])
def batch_label_images():
    """
    Label multiple images at once
    
    Request Body:
        {
            "labels": [
                {
                    "filename": "image1.png",
                    "label": "Center",
                    "confidence": 0.95
                },
                ...
            ],
            "labeled_by": "user@example.com"
        }
    
    Returns:
        JSON response with batch results
    """
    try:
        data = request.get_json()
        
        if 'labels' not in data or not isinstance(data['labels'], list):
            return jsonify({'error': 'Invalid request: labels array required'}), 400
        
        storage = JSONStorage(current_app.config['METADATA_FOLDER'])
        labels_data = storage.read('image_labels.json', default={'images': []})
        
        results = []
        for label_item in data['labels']:
            try:
                label_record = {
                    'filename': label_item['filename'],
                    'folder': label_item.get('folder', 'augmented'),
                    'label': label_item['label'],
                    'confidence': label_item.get('confidence', 1.0),
                    'status': 'labeled',
                    'labeled_by': data.get('labeled_by', 'unknown'),
                    'labeled_at': datetime.now().isoformat(),
                    'notes': label_item.get('notes', '')
                }
                
                # Update or add
                existing_idx = next(
                    (i for i, img in enumerate(labels_data['images']) 
                     if img['filename'] == label_item['filename']), 
                    None
                )
                
                if existing_idx is not None:
                    labels_data['images'][existing_idx] = label_record
                else:
                    labels_data['images'].append(label_record)
                
                results.append({
                    'filename': label_item['filename'],
                    'status': 'success'
                })
            
            except Exception as e:
                results.append({
                    'filename': label_item.get('filename', 'unknown'),
                    'status': 'error',
                    'error': str(e)
                })
        
        # Save all labels
        storage.write('image_labels.json', labels_data)
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        
        return jsonify({
            'status': 'success',
            'message': f'Labeled {success_count}/{len(results)} images',
            'results': results
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Batch label error: {str(e)}")
        return jsonify({"error": f"Failed to batch label: {str(e)}"}), 500


@api_v1_bp.route('/labeling/validate/<filename>', methods=['POST'])
def validate_label(filename):
    """
    Validate or reject a labeled image
    
    Request Body:
        {
            "action": "validate", "reject", or "unlabel",
            "validated_by": "user@example.com",
            "notes": "Validation notes"
        }
    
    Returns:
        JSON response with validation status
    """
    try:
        data = request.get_json()
        action = data.get('action', 'validate')
        
        if action not in ['validate', 'reject', 'unlabel']:
            return jsonify({'error': 'Invalid action'}), 400
        
        storage = JSONStorage(current_app.config['METADATA_FOLDER'])
        labels_data = storage.read('image_labels.json', default={'images': []})
        
        # Find image
        img_idx = next(
            (i for i, img in enumerate(labels_data['images']) 
             if img['filename'] == filename), 
            None
        )
        
        if img_idx is None:
            return jsonify({'error': 'Image not found'}), 404
        
        # Update status
        if action == 'unlabel':
            # Reset to labeled status
            labels_data['images'][img_idx]['status'] = 'labeled'
            labels_data['images'][img_idx]['validated_by'] = None
            labels_data['images'][img_idx]['validated_at'] = None
            labels_data['images'][img_idx]['validation_notes'] = None
        else:
            labels_data['images'][img_idx]['status'] = 'validated' if action == 'validate' else 'rejected'
            labels_data['images'][img_idx]['validated_by'] = data.get('validated_by', 'unknown')
            labels_data['images'][img_idx]['validated_at'] = datetime.now().isoformat()
            labels_data['images'][img_idx]['validation_notes'] = data.get('notes', '')
        
        storage.write('image_labels.json', labels_data)
        
        return jsonify({
            'status': 'success',
            'message': f'Image {action}d successfully',
            'image': labels_data['images'][img_idx]
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Validate label error: {str(e)}")
        return jsonify({"error": f"Failed to validate: {str(e)}"}), 500


@api_v1_bp.route('/labeling/export', methods=['POST'])
def export_labeled_data():
    """
    Export labeled images to training dataset
    
    Request Body:
        {
            "filenames": ["image1.png", "image2.png"],  # Specific files to export
            "destination": "training"  # or "validation", "test"
        }
    
    Returns:
        JSON response with export results
    """
    try:
        data = request.get_json()
        filenames = data.get('filenames', [])
        destination = data.get('destination', 'training')
        
        if not filenames:
            return jsonify({
                'status': 'error',
                'message': 'No filenames provided',
                'exported': 0
            }), 400
        
        storage = JSONStorage(current_app.config['METADATA_FOLDER'])
        labels_data = storage.read('image_labels.json', default={'images': []})
        
        # Filter images by filenames and ensure they are labeled or validated
        images_to_export = [
            img for img in labels_data['images'] 
            if img['filename'] in filenames and img['status'] in ['labeled', 'validated']
        ]
        
        if not images_to_export:
            return jsonify({
                'status': 'success',
                'message': 'No valid images to export (must be labeled or validated)',
                'exported': 0
            }), 200
        
        # Create destination directory
        dest_dir = Path(current_app.config['UPLOAD_FOLDER']) / destination
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Export images and create metadata
        exported = []
        for img in images_to_export:
            try:
                # Try configured path first
                src_path = Path(current_app.config['UPLOAD_FOLDER']) / img['folder'] / img['filename']
                
                # If not found, try absolute path
                if not src_path.exists():
                    abs_path = Path(r'C:\VSMC\VSMC-FAB_Investigayion_Tool\wafer-defect-ap\data\wafer_images') / img['folder'] / img['filename']
                    if abs_path.exists():
                        src_path = abs_path
                
                if not src_path.exists():
                    current_app.logger.warning(f"Source image not found: {src_path}")
                    continue
                
                # Copy to destination with label prefix
                label_prefix = img['label'].replace(' ', '_').replace('-', '_')
                dest_filename = f"{label_prefix}_{img['filename']}"
                dest_path = dest_dir / dest_filename
                
                shutil.copy2(src_path, dest_path)
                
                # Mark as exported
                img['exported'] = True
                img['exported_at'] = datetime.now().isoformat()
                
                exported.append({
                    'original': img['filename'],
                    'destination': dest_filename,
                    'label': img['label'],
                    'confidence': img['confidence']
                })
            
            except Exception as e:
                current_app.logger.error(f"Export error for {img['filename']}: {str(e)}")
        
        # Save updated labels with export status
        storage.write('image_labels.json', labels_data)
        
        # Save export metadata
        export_metadata = {
            'exported_at': datetime.now().isoformat(),
            'destination': destination,
            'total_exported': len(exported),
            'images': exported
        }
        
        storage.write(f'export_{destination}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', export_metadata)
        
        return jsonify({
            'status': 'success',
            'message': f'Exported {len(exported)} images to {destination}',
            'exported': len(exported),
            'images': exported
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Export error: {str(e)}")
        return jsonify({"error": f"Failed to export: {str(e)}"}), 500


@api_v1_bp.route('/labeling/statistics', methods=['GET'])
def get_labeling_statistics():
    """
    Get labeling statistics
    
    Returns:
        JSON response with statistics
    """
    try:
        storage = JSONStorage(current_app.config['METADATA_FOLDER'])
        labels_data = storage.read('image_labels.json', default={'images': []})
        
        # Calculate statistics
        total = len(labels_data['images'])
        by_status = {}
        by_label = {}
        
        for img in labels_data['images']:
            status = img.get('status', 'unlabeled')
            label = img.get('label', 'Unknown')
            
            by_status[status] = by_status.get(status, 0) + 1
            by_label[label] = by_label.get(label, 0) + 1
        
        return jsonify({
            'status': 'success',
            'statistics': {
                'total_images': total,
                'by_status': by_status,
                'by_label': by_label,
                'completion_rate': (by_status.get('validated', 0) / total * 100) if total > 0 else 0
            }
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Statistics error: {str(e)}")
        return jsonify({"error": f"Failed to get statistics: {str(e)}"}), 500


@api_v1_bp.route('/labeling/patterns', methods=['GET'])
def get_available_patterns():
    """
    Get list of available defect patterns for labeling
    
    Returns:
        JSON response with pattern list
    """
    try:
        patterns = [
            {'value': 'Center', 'label': 'Center', 'description': 'Defects concentrated in wafer center'},
            {'value': 'Edge-Ring', 'label': 'Edge Ring', 'description': 'Defects around wafer edge'},
            {'value': 'Edge-Loc', 'label': 'Edge Local', 'description': 'Localized edge defects'},
            {'value': 'Loc', 'label': 'Local', 'description': 'Localized defects'},
            {'value': 'Random', 'label': 'Random', 'description': 'Randomly distributed defects'},
            {'value': 'Scratch', 'label': 'Scratch', 'description': 'Scratch pattern'},
            {'value': 'Donut', 'label': 'Donut', 'description': 'Ring-shaped defect pattern'},
            {'value': 'Near-Full', 'label': 'Near Full', 'description': 'Nearly full wafer defects'},
            {'value': 'None', 'label': 'None/Good', 'description': 'No defects detected'}
        ]
        
        return jsonify({
            'status': 'success',
            'patterns': patterns
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Get patterns error: {str(e)}")
        return jsonify({"error": f"Failed to get patterns: {str(e)}"}), 500
