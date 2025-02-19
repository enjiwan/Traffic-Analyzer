# Traffic Analyzer

A professional traffic analysis application leveraging the power of YOLOv8 for vehicle detection and PyQt5 for a seamless graphical user experience.

## Key Features
- **Real-time Vehicle Detection:** Identify and track vehicles from a webcam feed or video files.
- **Speed Calculation:** Use calibration points to measure vehicle speed with high accuracy.
- **Report Export:** Generate and export detailed traffic reports in CSV format.
- **Customizable Interface:** Toggle between night mode and light mode for optimal viewing experience.

## System Requirements
```
ultralytics
opencv-python-headless
numpy
PyQt5
```

## Installation
```bash
# Create a virtual environment (recommended)
python -m venv env
source env/bin/activate  # Linux/macOS
env\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Calibration Guide
- Select two points in the video frame that are a known distance apart to calibrate the pixel-to-meter ratio.
- This calibration ensures precise speed measurement and traffic analytics.

## Exporting Reports
- After video processing, click **Export Report** to save detected vehicle speeds and timestamps as a CSV file.

## Contributing
We welcome contributions! Feel free to fork this repository, create issues, or submit pull requests to enhance the functionality and accuracy of the Traffic Analyzer.

## License
This project is licensed under the MIT License — see the LICENSE file for details.

---
🚀 **Stay ahead of the traffic with AI-powered insights!**
