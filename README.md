# Language Classification using YOLOv8

## Overview

This project performs language classification from speech audio using a YOLOv8 model.
The audio is converted into a Mel Spectrogram image, then classified.

Supported languages:

* English
* Spanish
* French

## Pipeline

1. Input a `.wav` audio file
2. Remove non-speech using WebRTC VAD
3. Convert audio to Mel Spectrogram
4. Classify using YOLOv8

## Project Structure

```
.
├── demon_GUI.py              # GUI application
├── train_model.ipynb         # Training notebook
├── predict.ipynb             # Inference notebook
├── preprocessing_*.ipynb     # Audio preprocessing
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt   # Trained model
```

## Requirements

Install dependencies:

```
pip install ultralytics librosa webrtcvad matplotlib numpy
```

## Usage

### Run GUI

```
python demon_GUI.py
```

Steps:

1. Select a `.wav` file
2. The system processes and shows the predicted language

### Use Notebooks

* `train_model.ipynb` for training
* `predict.ipynb` for prediction

## Model

The model uses YOLOv8 to classify Mel Spectrogram images.

Model path:

```
runs/detect/train/weights/best.pt
```

## Notes

* Supports only `.wav` files
* Model path is hardcoded
* Requires `librosa.display`
* Uses YOLO as an image classifier

## Limitations

* YOLO is designed for object detection
* Only three languages
* Not production-ready

## Future Improvements

* Use audio-specific models such as CNN or wav2vec2
* Add more languages
* Support more audio formats
* Build an API
