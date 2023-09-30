# Sario Reader

## Table of Contents

- [Sario Reader](#sario-reader)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [How the Code Works](#how-the-code-works)
  - [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Configuration](#configuration)
  - [Development](#development)
    - [Pre-commit Hooks](#pre-commit-hooks)
    - [Flake8](#flake8)
    - [Coverage](#coverage)
  - [Usage](#usage)
    - [Object Detection](#object-detection)
    - [OCR](#ocr)
  - [License](#license)

## Introduction

Sario Reader is an AI/ML assisted tool designed to analyze videos of industrial environments and recognize objects with number plates. This project is part of a thesis work and aims to provide a robust solution for object detection and recognition.

## How the Code Works

The Sario Reader application is primarily divided into two main components: object detection and Optical Character Recognition (OCR). The object detection is handled by the `SarioDetector` class, which uses the YOLO model from the Ultralytics library to identify objects in video frames. Once the objects are detected, their regions of interest (ROIs) are extracted for further processing. The OCR part is managed by the `srOCR` class, which uses the Tesseract-OCR engine to recognize text from the extracted ROIs. This class also includes preprocessing steps like contrast adjustment and binarization to improve OCR accuracy. Both components are orchestrated in the `__main__.py` script, where video frames are read, processed, and the results are logged.

## Installation

To install Sario Reader, you can use Poetry:

```shell
poetry install
```

## Dependencies

Here are the major dependencies of the project:

| Dependency    | Version   | Description                               |
| ------------- | --------- | ----------------------------------------- |
| clearvision   | ^0.3.1    | Image processing toolkit                  |
| ultralytics   | ^8.0.184  | YOLO model for object detection           |
| opencv-python | ^4.8.0.76 | OpenCV bindings for Python                |
| numpy         | >=1.24.0  | Numerical computing library               |
| pytesseract   | ^0.3.10   | Python binding for Google's Tesseract-OCR |

Dependencies should not be tampered with as the packages themselves have strict requirements.

## Configuration

The only configuration required is the path to the video. It can also be a stream as long as it is compatible with OpenCV.

## Development

### Pre-commit Hooks

The project uses pre-commit hooks to maintain code quality. The configuration can be found in `.pre-commit-config.yaml`.

### Flake8

Flake8 is used for linting the code. The configuration can be found in `.flake8`.

### Coverage

Code coverage settings are in `.coveragerc`.

## Usage

To run the main application:

```shell
python src/sarioreader "path/to/video"
```

Or using poetry:

```shell
poetry run python sarioreader "path/to/video"
```

The app will process the video frame by frame. It will first be passed by the Object Detector and if anything is returned it will be send to the OCR model. The output will be a list of text that was found in the ROI returned by the Object Detector. The ROI is passed through an image pre-processing algorithm provided by the clearvision package that performs histogram equalization for better text clarity.

The app also includes a preview of each ROI as it gets executed. The code runs fast enough to process the video near real-time.

### Object Detection

The `SarioDetector` class in [`sario.py`](https://github.com/chmaikos/video-analysis-thesis/blob/main/src/sarioreader/sario.py) is used for object detection.

It utilizes the YOLOv8 library from the Ultralytics package and a custom model that is able to detect objects of interest in the video.

The model right now is not modular unless you modify the __init__ method of the package.

### OCR

The `srOCR` class in [`ocr.py`](https://github.com/chmaikos/video-analysis-thesis/blob/main/src/sarioreader/ocr.py) is used for Optical Character Recognition.

It is based on the clearvision project that is also created by me. More info [here](https://github.com/chmaikos/clearvision)

## License

This project is licensed under the GPL-3.0 License.
