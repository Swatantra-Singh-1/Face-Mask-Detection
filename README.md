<<<<<<< HEAD
# Face Mask Detection

Face mask detection using deep learning with MobileNetV2 for mask classification and SSD (Single Shot Detector) for face detection.

## Features
- Train mask detector on with/without mask dataset
- Real-time video mask detection
- Uses OpenCV, TensorFlow/Keras, imutils

## Setup
1. Clone repo: `git clone https://github.com/Swatantra-Singh-1/Face-Mask-Detection`
2. Install dependencies:
```
pip install -r requirements.txt
```
Note: TensorFlow 1.15+, Keras 2.3.1 (older versions for compatibility).

3. Download pre-trained model:
   - `mask_detector.model` (train with `train_mask_detector.py` or download)
   - Face detector: `face_detector/res10_300x300_ssd_iter_140000.caffemodel` (from OpenCV zoo)

## Usage
### Train
```bash
python train_mask_detector.py
```

### Detect in video/webcam
```bash
python detect_mask_video.py --image <path>  # single image
python detect_mask_video.py --video <path>  # video file
# or webcam (default)
```

## Dataset
- `dataset/with_mask/` and `dataset/without_mask/` (excluded from repo, download separately)
- Images augmented during training.

## Models
- Mask classifier: MobileNetV2 fine-tuned
- Face detector: ResNet-SSD prototxt + caffemodel

## Notes
- Older TF/Keras versions for compatibility with pre-trained models.
- Update to TF2+ possible but requires model conversion.
=======
# Face-Mask-Detection
This program detects if you have wore mask or not
>>>>>>> 77330d9f6d56a9a447ec1bb2eacdb253d5c729f5
