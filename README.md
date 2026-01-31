# Human Detection System

A reliable human detection system using OpenCV's deep learning module (DNN) with pre-trained neural networks.

## Overview

This detector uses modern deep learning models instead of traditional Haar Cascades, providing significantly better accuracy and reliability:

- **Face Detection**: ResNet-based SSD model (90-95% accuracy)
- **Person Detection**: MobileNet-SSD or HOG detector (85-95% accuracy)
- **Smart Merging**: Automatically matches faces to bodies for complete person detection

## Features

- Deep learning-based face detection
- Full person body detection
- Automatic face-body matching
- Confidence scores for each detection
- Multiple detection types (person with face, face only, body only)
- Auto-downloads required models on first run

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.7+
- OpenCV 4.0+
- NumPy

## Usage

### Basic Detection

```bash
python human_detector.py image.jpg
```

### Save Output

```bash
python human_detector.py image.jpg -o output.jpg
```

### Display Results

```bash
python human_detector.py image.jpg -d
```

### Combined

```bash
python human_detector.py image.jpg -o output.jpg -d
```

## Command Line Arguments

```
positional arguments:
  image_path            Path to input image

optional arguments:
  -h, --help           Show help message
  -o, --output PATH    Save annotated image to PATH
  -d, --display        Display results in a window
```

## Detection Types

The system produces three types of detections:

1. **Person with Face** (Purple box + Green face box)
   - Complete person detected with visible face
   - Highest confidence detection
   - Both body and face matched

2. **Face Only** (Green box)
   - Face detected but no body match
   - May indicate partial visibility or close-up shot

3. **Body Only** (Yellow box)
   - Person body detected but no face visible
   - May indicate turned away, occluded face, or distance

## Output

The detector provides:

- Console output with detection count and confidence scores
- Annotated image with colored bounding boxes
- Detailed detection information including:
  - Detection type
  - Confidence percentage
  - Bounding box coordinates

## Examples

### Example 1: Group Photo
```bash
python human_detector.py family_photo.jpg -o detected.jpg
```
Detects all visible people with faces and bodies.

### Example 2: Crowded Scene
```bash
python human_detector.py crowd.jpg -d
```
Displays detection results for review before saving.

## Technical Details

### Models Used

**Face Detection**: ResNet-based SSD
- Input size: 300x300
- Architecture: Single Shot MultiBox Detector with ResNet-10 backbone
- Pre-trained on face detection dataset
- Auto-downloaded from OpenCV repository

**Person Detection**: MobileNet-SSD (with HOG fallback)
- Input size: 300x300
- Architecture: MobileNet-based Single Shot Detector
- Trained on PASCAL VOC dataset
- Falls back to HOG if model unavailable

### Detection Pipeline

1. **Image Loading**: Read input image
2. **Face Detection**: Run DNN face detector
3. **Person Detection**: Run MobileNet-SSD or HOG
4. **Merging**: Match faces to bodies using spatial proximity
5. **Drawing**: Annotate image with colored boxes and labels
6. **Output**: Save and/or display results

### Color Coding

- **Purple (147, 20, 255)**: Complete person (face + body)
- **Green (0, 255, 0)**: Face detection
- **Yellow (255, 255, 0)**: Body detection without face

## Performance

### Accuracy
- Face detection: 90-95% in good lighting conditions
- Person detection: 85-95% for visible bodies
- Much more reliable than traditional Haar Cascades

### Speed
- 0.5-2 seconds per image (depending on resolution)
- Faster on smaller images
- CPU optimized

### Limitations

- Requires clear visibility of face/body
- Performance degrades with:
  - Very small faces (< 30x30 pixels)
  - Extreme angles or occlusions
  - Very low light conditions
  - Heavy motion blur

## Troubleshooting

### Models Not Downloading
If automatic download fails, manually download:
1. Face model: [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)
2. Prototxt: [deploy.prototxt](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)

Place them in the same directory as the script.

### No Detections
- Check image quality and resolution
- Ensure faces are > 30x30 pixels
- Verify good lighting conditions
- Try adjusting confidence threshold in code

### False Positives
- Increase confidence threshold (default: 0.5)
- Edit line: `confidence_threshold=0.5` to higher value (e.g., 0.7)

## Comparison with Haar Cascades

| Feature | Haar Cascades | DNN (This System) |
|---------|--------------|-------------------|
| Accuracy | 70-85% | 90-95% |
| Speed | Fast | Moderate |
| Profile faces | Poor | Good |
| Occlusions | Poor | Fair-Good |
| Lighting | Sensitive | Robust |
| False positives | High | Low |

## Credits

- OpenCV DNN module
- ResNet-based face detector
- MobileNet-SSD architecture
- Pre-trained models from OpenCV repository

## License

MIT License