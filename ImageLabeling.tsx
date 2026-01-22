import { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  TextField,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  LinearProgress,
  Pagination,
  Stack,
  Tooltip,
  Checkbox,
  Badge,
} from '@mui/material';
import {
  CheckCircle,
  Cancel,
  Label as LabelIcon,
  FilterList,
  Refresh,
  Upload,
  CheckBox as CheckBoxIcon,
  CheckBoxOutlineBlank as CheckBoxOutlineBlankIcon,
} from '@mui/icons-material';

interface WaferImage {
  filename: string;
  path: string;
  url: string;
  size: number;
  created_at: string;
  status: 'unlabeled' | 'labeled' | 'validated' | 'rejected';
  label?: string;
  confidence?: number;
  labeled_by?: string;
  labeled_at?: string;
  notes?: string;
  exported?: boolean;
  exported_at?: string;
}

interface Pattern {
  value: string;
  label: string;
  description: string;
}

const STATUS_COLORS = {
  unlabeled: 'default',
  labeled: 'info',
  validated: 'success',
  rejected: 'error',
} as const;

export default function ImageLabeling() {
  const [images, setImages] = useState<WaferImage[]>([]);
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState<WaferImage | null>(null);
  const [selectedPattern, setSelectedPattern] = useState('');
  const [confidence, setConfidence] = useState(1.0);
  const [notes, setNotes] = useState('');
  const [statusFilter, setStatusFilter] = useState('unlabeled');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [statistics, setStatistics] = useState<any>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [zoomDialogOpen, setZoomDialogOpen] = useState(false);
  const [selectedImages, setSelectedImages] = useState<Set<string>>(new Set());

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api/v1';
  const ITEMS_PER_PAGE = 10; // 5 columns × 2 rows

  // Fetch available patterns
  useEffect(() => {
    fetchPatterns();
    fetchStatistics();
  }, []);

  // Fetch images when filter or page changes
  useEffect(() => {
    fetchImages();
    setSelectedImages(new Set()); // Clear selection when changing page/filter
  }, [statusFilter, page]);

  const fetchPatterns = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/labeling/patterns`);
      const data = await response.json();
      if (data.status === 'success') {
        setPatterns(data.patterns);
      }
    } catch (error) {
      console.error('Failed to fetch patterns:', error);
    }
  };

  const fetchImages = async () => {
    setLoading(true);
    try {
      const offset = (page - 1) * ITEMS_PER_PAGE;
      const response = await fetch(
        `${API_BASE_URL}/labeling/images?status=${statusFilter}&limit=${ITEMS_PER_PAGE}&offset=${offset}`
      );
      const data = await response.json();
      
      if (data.status === 'success') {
        setImages(data.images);
        setTotalPages(Math.ceil(data.total / ITEMS_PER_PAGE));
      }
    } catch (error) {
      console.error('Failed to fetch images:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchStatistics = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/labeling/statistics`);
      const data = await response.json();
      if (data.status === 'success') {
        setStatistics(data.statistics);
      }
    } catch (error) {
      console.error('Failed to fetch statistics:', error);
    }
  };

  const handleLabelImage = async () => {
    if (!selectedImage || !selectedPattern) return;

    try {
      const response = await fetch(`${API_BASE_URL}/labeling/label`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: selectedImage.filename,
          folder: 'augmented',
          label: selectedPattern,
          confidence: confidence,
          labeled_by: 'current_user',
          notes: notes,
        }),
      });

      const data = await response.json();
      
      if (data.status === 'success') {
        setDialogOpen(false);
        setSelectedImage(null);
        setSelectedPattern('');
        setNotes('');
        fetchImages();
        fetchStatistics();
      }
    } catch (error) {
      console.error('Failed to label image:', error);
    }
  };

  const handleValidate = async (filename: string, action: 'validate' | 'reject' | 'unlabel') => {
    try {
      if (action === 'unlabel') {
        // Reset to labeled status
        const response = await fetch(`${API_BASE_URL}/labeling/validate/${filename}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'unlabel',
            validated_by: 'current_user',
            notes: 'Undoing validation/rejection',
          }),
        });

        const data = await response.json();
        
        if (data.status === 'success') {
          fetchImages();
          fetchStatistics();
        }
      } else {
        const response = await fetch(`${API_BASE_URL}/labeling/validate/${filename}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: action,
            validated_by: 'current_user',
            notes: '',
          }),
        });

        const data = await response.json();
        
        if (data.status === 'success') {
          fetchImages();
          fetchStatistics();
        }
      }
    } catch (error) {
      console.error('Failed to validate image:', error);
    }
  };

  const handleExport = async () => {
    try {
      // Get list of selected filenames
      const filenames = Array.from(selectedImages);
      
      if (filenames.length === 0) {
        alert('Please select at least one image to export');
        return;
      }

      const response = await fetch(`${API_BASE_URL}/labeling/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filenames: filenames,
          destination: 'training',
        }),
      });

      const data = await response.json();
      
      if (data.status === 'success') {
        alert(`Exported ${data.exported} images to training dataset`);
        setSelectedImages(new Set()); // Clear selection after export
        fetchImages(); // Refresh to show exported status
        fetchStatistics();
      }
    } catch (error) {
      console.error('Failed to export images:', error);
    }
  };

  const handleSelectAll = () => {
    // Only select labeled or validated images (not unlabeled or rejected)
    const selectableImages = images.filter(
      img => img.status === 'labeled' || img.status === 'validated'
    );
    
    if (selectedImages.size === selectableImages.length) {
      // Deselect all
      setSelectedImages(new Set());
    } else {
      // Select all selectable images
      setSelectedImages(new Set(selectableImages.map(img => img.filename)));
    }
  };

  const handleToggleImage = (filename: string, status: string) => {
    // Only allow selection of labeled or validated images
    if (status !== 'labeled' && status !== 'validated') {
      return;
    }

    const newSelected = new Set(selectedImages);
    if (newSelected.has(filename)) {
      newSelected.delete(filename);
    } else {
      newSelected.add(filename);
    }
    setSelectedImages(newSelected);
  };

  const isImageSelectable = (status: string) => {
    return status === 'labeled' || status === 'validated';
  };

  const selectableCount = images.filter(img => isImageSelectable(img.status)).length;
  const allSelectableSelected = selectableCount > 0 && selectedImages.size === selectableCount;

  const openLabelDialog = (image: WaferImage) => {
    setSelectedImage(image);
    setSelectedPattern(image.label || '');
    setConfidence(image.confidence || 1.0);
    setNotes(image.notes || '');
    setDialogOpen(true);
  };

  const openZoomDialog = (image: WaferImage) => {
    setSelectedImage(image);
    setZoomDialogOpen(true);
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 600 }}>
          Image Labeling & Ground Truth Management
        </Typography>
        <Stack direction="row" spacing={2}>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={() => {
              fetchImages();
              fetchStatistics();
            }}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<Upload />}
            onClick={handleExport}
            disabled={selectedImages.size === 0}
          >
            Export Selected ({selectedImages.size})
          </Button>
        </Stack>
      </Box>

      {/* Statistics */}
      {statistics && (
        <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap' }}>
          <Card sx={{ flex: '1 1 calc(25% - 12px)', minWidth: '200px' }}>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Total Images
              </Typography>
              <Typography variant="h4">{statistics.total_images}</Typography>
            </CardContent>
          </Card>
          <Card sx={{ flex: '1 1 calc(25% - 12px)', minWidth: '200px' }}>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Unlabeled
              </Typography>
              <Typography variant="h4">{statistics.by_status?.unlabeled || 0}</Typography>
            </CardContent>
          </Card>
          <Card sx={{ flex: '1 1 calc(25% - 12px)', minWidth: '200px' }}>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Validated
              </Typography>
              <Typography variant="h4" color="success.main">
                {statistics.by_status?.validated || 0}
              </Typography>
            </CardContent>
          </Card>
          <Card sx={{ flex: '1 1 calc(25% - 12px)', minWidth: '200px' }}>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Completion
              </Typography>
              <Typography variant="h4">{statistics.completion_rate?.toFixed(1)}%</Typography>
              <LinearProgress
                variant="determinate"
                value={statistics.completion_rate}
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Box>
      )}

      {/* Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stack direction="row" spacing={2} alignItems="center">
            <FilterList />
            <FormControl sx={{ minWidth: 200 }}>
              <InputLabel>Status Filter</InputLabel>
              <Select
                value={statusFilter}
                label="Status Filter"
                onChange={(e) => {
                  setStatusFilter(e.target.value);
                  setPage(1);
                }}
              >
                <MenuItem value="all">All</MenuItem>
                <MenuItem value="unlabeled">Unlabeled</MenuItem>
                <MenuItem value="labeled">Labeled</MenuItem>
                <MenuItem value="validated">Validated</MenuItem>
                <MenuItem value="rejected">Rejected</MenuItem>
              </Select>
            </FormControl>
            
            {selectableCount > 0 && (
              <Box sx={{ display: 'flex', alignItems: 'center', ml: 'auto' }}>
                <Checkbox
                  checked={allSelectableSelected}
                  indeterminate={selectedImages.size > 0 && !allSelectableSelected}
                  onChange={handleSelectAll}
                  icon={<CheckBoxOutlineBlankIcon />}
                  checkedIcon={<CheckBoxIcon />}
                />
                <Typography variant="body2" sx={{ ml: 1 }}>
                  Select All ({selectableCount} selectable)
                </Typography>
              </Box>
            )}
          </Stack>
        </CardContent>
      </Card>

      {/* Loading */}
      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Image Grid - Fixed 5 columns using flexbox */}
      <Box
        sx={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 2,
          mb: 2,
        }}
      >
        {images.map((image) => (
          <Box
            key={image.filename}
            sx={{
              width: 'calc(20% - 12.8px)', // 5 columns with gap
              minWidth: '200px',
            }}
          >
            <Card sx={{ height: '100%', position: 'relative' }}>
              {/* Checkbox for selection */}
              {isImageSelectable(image.status) && (
                <Checkbox
                  checked={selectedImages.has(image.filename)}
                  onChange={() => handleToggleImage(image.filename, image.status)}
                  onClick={(e) => e.stopPropagation()}
                  icon={<CheckBoxOutlineBlankIcon />}
                  checkedIcon={<CheckBoxIcon />}
                  sx={{
                    position: 'absolute',
                    top: 8,
                    left: 8,
                    zIndex: 2,
                    backgroundColor: 'rgba(255, 255, 255, 0.9)',
                    borderRadius: '4px',
                    padding: '4px',
                    '&:hover': {
                      backgroundColor: 'rgba(255, 255, 255, 1)',
                    },
                  }}
                />
              )}
              
              {/* Export badge */}
              {image.exported && (
                <Badge
                  badgeContent="Exported"
                  color="success"
                  sx={{
                    position: 'absolute',
                    top: 8,
                    left: isImageSelectable(image.status) ? 48 : 8,
                    zIndex: 2,
                    '& .MuiBadge-badge': {
                      fontSize: '0.65rem',
                      height: '18px',
                      minWidth: '18px',
                      padding: '0 6px',
                    },
                  }}
                >
                  <Box sx={{ width: 0, height: 0 }} />
                </Badge>
              )}
              
              <Box
                sx={{
                  position: 'relative',
                  paddingTop: '100%', // Square aspect ratio
                  backgroundColor: '#f5f5f5',
                  cursor: 'pointer',
                  overflow: 'hidden',
                }}
                onClick={() => openZoomDialog(image)}
              >
                <Box
                  component="img"
                  src={`${API_BASE_URL}${image.url}`}
                  alt={image.filename}
                  onError={(e) => {
                    // Fallback if image fails to load
                    const target = e.target as HTMLImageElement;
                    target.style.display = 'none';
                    const parent = target.parentElement;
                    if (parent) {
                      parent.style.display = 'flex';
                      parent.style.alignItems = 'center';
                      parent.style.justifyContent = 'center';
                      parent.innerHTML = '<div style="color: #999; text-align: center; padding: 20px;">Image not found</div>';
                    }
                  }}
                  sx={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                    padding: '8px',
                  }}
                />
                <Chip
                  label={image.status}
                  color={STATUS_COLORS[image.status]}
                  size="small"
                  sx={{ position: 'absolute', top: 8, right: 8, zIndex: 1 }}
                />
              </Box>
              <CardContent sx={{ p: 1.5 }}>
                <Typography 
                  variant="caption" 
                  noWrap 
                  title={image.filename} 
                  sx={{ 
                    display: 'block', 
                    mb: 1,
                    fontSize: '0.75rem',
                  }}
                >
                  {image.filename}
                </Typography>
                {image.label && (
                  <Chip
                    label={image.label}
                    size="small"
                    color="primary"
                    sx={{ mb: 1, fontSize: '0.7rem', height: 20 }}
                  />
                )}
                <Stack direction="row" spacing={0.5}>
                  {image.status === 'unlabeled' && (
                    <Button
                      size="small"
                      variant="contained"
                      startIcon={<LabelIcon sx={{ fontSize: 14 }} />}
                      onClick={(e) => {
                        e.stopPropagation();
                        openLabelDialog(image);
                      }}
                      fullWidth
                      sx={{ fontSize: '0.7rem', py: 0.5 }}
                    >
                      Label
                    </Button>
                  )}
                  {image.status === 'labeled' && (
                    <>
                      <Tooltip title="Validate">
                        <IconButton
                          size="small"
                          color="success"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleValidate(image.filename, 'validate');
                          }}
                          sx={{ p: 0.5 }}
                        >
                          <CheckCircle fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Reject">
                        <IconButton
                          size="small"
                          color="error"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleValidate(image.filename, 'reject');
                          }}
                          sx={{ p: 0.5 }}
                        >
                          <Cancel fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={(e) => {
                          e.stopPropagation();
                          openLabelDialog(image);
                        }}
                        fullWidth
                        sx={{ fontSize: '0.65rem', py: 0.5 }}
                      >
                        Edit
                      </Button>
                    </>
                  )}
                  {image.status === 'validated' && (
                    <>
                      <Button
                        size="small"
                        variant="outlined"
                        color="success"
                        onClick={(e) => {
                          e.stopPropagation();
                          openLabelDialog(image);
                        }}
                        sx={{ fontSize: '0.7rem', py: 0.5, flex: 1 }}
                      >
                        Re-label
                      </Button>
                      <Tooltip title="Undo Validation">
                        <IconButton
                          size="small"
                          color="warning"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleValidate(image.filename, 'unlabel');
                          }}
                          sx={{ p: 0.5 }}
                        >
                          <Cancel fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </>
                  )}
                  {image.status === 'rejected' && (
                    <>
                      <Button
                        size="small"
                        variant="outlined"
                        color="error"
                        onClick={(e) => {
                          e.stopPropagation();
                          openLabelDialog(image);
                        }}
                        sx={{ fontSize: '0.7rem', py: 0.5, flex: 1 }}
                      >
                        Re-label
                      </Button>
                      <Tooltip title="Undo Rejection">
                        <IconButton
                          size="small"
                          color="warning"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleValidate(image.filename, 'unlabel');
                          }}
                          sx={{ p: 0.5 }}
                        >
                          <CheckCircle fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </>
                  )}
                </Stack>
              </CardContent>
            </Card>
          </Box>
        ))}
      </Box>

      {/* Pagination */}
      {totalPages > 1 && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
          <Pagination
            count={totalPages}
            page={page}
            onChange={(_, value) => setPage(value)}
            color="primary"
          />
        </Box>
      )}

      {/* Label Dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Label Wafer Map Image</DialogTitle>
        <DialogContent>
          {selectedImage && (
            <Box>
              <Box
                component="img"
                src={`${API_BASE_URL}${selectedImage.url}`}
                alt={selectedImage.filename}
                sx={{ width: '100%', maxHeight: 400, objectFit: 'contain', mb: 3 }}
              />
              <Typography variant="body2" color="text.secondary" gutterBottom>
                {selectedImage.filename}
              </Typography>
              
              <FormControl fullWidth sx={{ mt: 2 }}>
                <InputLabel>Defect Pattern</InputLabel>
                <Select
                  value={selectedPattern}
                  label="Defect Pattern"
                  onChange={(e) => setSelectedPattern(e.target.value)}
                >
                  {patterns.map((pattern) => (
                    <MenuItem key={pattern.value} value={pattern.value}>
                      <Box>
                        <Typography>{pattern.label}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          {pattern.description}
                        </Typography>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <TextField
                fullWidth
                label="Confidence (0-1)"
                type="number"
                value={confidence}
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
                inputProps={{ min: 0, max: 1, step: 0.05 }}
                sx={{ mt: 2 }}
              />

              <TextField
                fullWidth
                label="Notes"
                multiline
                rows={3}
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                sx={{ mt: 2 }}
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleLabelImage}
            disabled={!selectedPattern}
          >
            Save Label
          </Button>
        </DialogActions>
      </Dialog>

      {/* Zoom Dialog */}
      <Dialog
        open={zoomDialogOpen}
        onClose={() => setZoomDialogOpen(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          {selectedImage?.filename}
          {selectedImage?.label && (
            <Chip label={selectedImage.label} color="primary" size="small" sx={{ ml: 2 }} />
          )}
        </DialogTitle>
        <DialogContent>
          {selectedImage && (
            <Box
              component="img"
              src={`${API_BASE_URL}${selectedImage.url}`}
              alt={selectedImage.filename}
              sx={{ width: '100%', objectFit: 'contain' }}
            />
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setZoomDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
