import sys
import csv
import math
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QFileDialog, QMessageBox, QInputDialog,
    QSlider, QAction, QMenuBar, QMenu
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeyEvent, QIcon, QColor
from PyQt5.QtCore import Qt, QTimer, QSize, QPoint, pyqtSignal
import cv2
from ultralytics import YOLO


class SpeedDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = YOLO("yolov8n.pt")
        self.tracker = VehicleTracker()
        self.initUI()
        self.cap = None
        self.is_webcam = False
        self.report_data = []
        self.calibration_points = []
        self.setFont(QFont("Segoe UI", 10))
        self.night_mode = True
        self.video_sources = []
        self.current_video_index = 0
        self.paused = False

    def initUI(self):
        self.setWindowTitle("🚗 Traffic Analyzer ")
        self.setGeometry(100, 100, 1400, 900)
        self.setWindowIcon(QIcon("icon.png"))

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Video Panel
        video_panel = QVBoxLayout()
        self.video_label = ClickableLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1E1E1E; border-radius: 8px;")
        video_panel.addWidget(self.video_label)

        # Playback Controls
        playback_controls = QHBoxLayout()
        self.btn_play = self.create_button("▶ Play", self.toggle_play)
        self.btn_pause = self.create_button("⏸ Pause", self.toggle_pause)
        self.btn_rewind = self.create_button("⏪ Rewind", self.rewind)
        self.btn_fast_forward = self.create_button("⏩ Fast Forward", self.fast_forward)
        playback_controls.addWidget(self.btn_play)
        playback_controls.addWidget(self.btn_pause)
        playback_controls.addWidget(self.btn_rewind)
        playback_controls.addWidget(self.btn_fast_forward)
        video_panel.addLayout(playback_controls)

        # Control Panel
        control_panel = QVBoxLayout()
        control_panel.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("Traffic Analytics Dashboard")
        header.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header.setStyleSheet("color: #3498DB; padding-bottom: 20px;")
        control_panel.addWidget(header)

        # Buttons
        self.btn_webcam = self.create_button("🌐 Start Webcam", self.toggle_webcam)
        self.btn_file = self.create_button("📁 Load Video", self.load_video)
        self.btn_calibrate = self.create_button("📏 Calibrate", self.calibrate)
        self.btn_export = self.create_button("📤 Export Report", self.export_report)
        self.btn_theme = self.create_button("🌙 Toggle Theme", self.toggle_theme)
        control_panel.addWidget(self.btn_webcam)
        control_panel.addWidget(self.btn_file)
        control_panel.addWidget(self.btn_calibrate)
        control_panel.addWidget(self.btn_export)
        control_panel.addWidget(self.btn_theme)

        # Vehicle List
        vehicle_list_header = QLabel("Detected Vehicles")
        vehicle_list_header.setFont(QFont("Segoe UI", 12))
        vehicle_list_header.setStyleSheet("color: #FFFFFF; padding: 10px 0;")
        control_panel.addWidget(vehicle_list_header)

        self.vehicle_list = QListWidget()
        self.vehicle_list.setStyleSheet("""
            QListWidget {
                background-color: #2A2A2A;
                color: #FFFFFF;
                border-radius: 6px;
                padding: 8px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #3A3A3A;
            }
            QListWidget::item:hover {
                background-color: #3498DB20;
            }
        """)
        control_panel.addWidget(self.vehicle_list)

        # Status Bar
        self.status_bar = QLabel("Ready")
        self.status_bar.setStyleSheet("color: #7F8C8D; padding-top: 20px;")
        control_panel.addWidget(self.status_bar)

        # Assemble Layouts
        main_layout.addLayout(video_panel, 70)
        main_layout.addLayout(control_panel, 30)

        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)

        # Style
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QPushButton {
                background-color: #3498DB;
                color: white;
                padding: 12px;
                border-radius: 6px;
                min-width: 120px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #2980B9; }
        """)

    def create_button(self, text, callback):
        btn = QPushButton(text)
        btn.clicked.connect(callback)
        btn.setIconSize(QSize(24, 24))
        return btn

    def toggle_webcam(self):
        self.is_webcam = not self.is_webcam
        if self.is_webcam:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.btn_webcam.setText("🌐 Stop Webcam")
            self.update_status("Webcam activated")
        else:
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.btn_webcam.setText("🌐 Start Webcam")
            self.update_status("Webcam disconnected")

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName()
        if path:
            self.video_sources.append(path)
            self.current_video_index = len(self.video_sources) - 1
            self.cap = cv2.VideoCapture(path)
            self.timer.start(30)
            self.update_status(f"Loaded: {path.split('/')[-1]}")

    def calibrate(self):
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self, "Warning", "Please load a video first!")
            return

        self.calibration_points = []
        self.video_label.clicked.connect(self.handle_calibration_click)
        self.update_status("Click two points a known distance apart (e.g., 10 meters).")

    def handle_calibration_click(self, pos):
        if len(self.calibration_points) < 2:
            self.calibration_points.append((pos.x(), pos.y()))
            if len(self.calibration_points) == 2:
                # Calculate pixel-to-meter conversion factor
                pixel_distance = math.hypot(
                    self.calibration_points[1][0] - self.calibration_points[0][0],
                    self.calibration_points[1][1] - self.calibration_points[0][1]
                )
                real_distance = 10  # Assume 10 meters between points
                self.tracker.px_to_m = real_distance / pixel_distance
                self.video_label.clicked.disconnect()
                self.update_status(f"Calibration complete. Scale: {self.tracker.px_to_m:.4f} m/px")

    def process_frame(self):
        if self.paused:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.update_status("Video playback completed")
            return

        results = self.model.track(frame, persist=True, classes=[2, 3, 5, 7])
        annotated_frame = results[0].plot()

        for box in results[0].boxes:
            if box.id is not None:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                speed = self.tracker.update(box.id, (x1, y1, x2, y2))

                self.update_vehicle_list(box.id, speed)
                self.report_data.append({
                    "id": int(box.id),
                    "speed": speed,
                    "timestamp": cv2.getTickCount() / cv2.getTickFrequency()
                })

                # Speed color coding
                color = (0, 255, 0) if speed < self.tracker.speed_limit else (0, 0, 255)
                cv2.putText(annotated_frame, f"{speed} km/h",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        self.show_image(annotated_frame)

    def update_vehicle_list(self, vehicle_id, speed):
        entry = f"🚗 Vehicle {vehicle_id}: {speed} km/h"
        items = [self.vehicle_list.item(i).text() for i in range(self.vehicle_list.count())]
        if entry not in items:
            self.vehicle_list.addItem(entry)

    def show_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def export_report(self):
        path, _ = QFileDialog.getSaveFileName(filter="CSV Files (*.csv)")
        if path:
            try:
                with open(path, 'w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=["id", "speed", "timestamp"])
                    writer.writeheader()
                    writer.writerows(self.report_data)
                self.update_status(f"Report saved to {path}")
                QMessageBox.information(self, "Success", "Report exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def update_status(self, message):
        self.status_bar.setText(f"🔹 {message}")

    def toggle_theme(self):
        self.night_mode = not self.night_mode
        if self.night_mode:
            self.setStyleSheet("""
                QMainWindow { background-color: #121212; }
                QPushButton {
                    background-color: #3498DB;
                    color: white;
                    padding: 12px;
                    border-radius: 6px;
                    min-width: 120px;
                    font-size: 14px;
                }
                QPushButton:hover { background-color: #2980B9; }
                QLabel { color: #FFFFFF; }
                QListWidget {
                    background-color: #2A2A2A;
                    color: #FFFFFF;
                    border-radius: 6px;
                    padding: 8px;
                }
                QListWidget::item {
                    padding: 8px;
                    border-bottom: 1px solid #3A3A3A;
                }
                QListWidget::item:hover {
                    background-color: #3498DB20;
                }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow { background-color: #FFFFFF; }
                QPushButton {
                    background-color: #3498DB;
                    color: white;
                    padding: 12px;
                    border-radius: 6px;
                    min-width: 120px;
                    font-size: 14px;
                }
                QPushButton:hover { background-color: #2980B9; }
                QLabel { color: #000000; }
                QListWidget {
                    background-color: #F0F0F0;
                    color: #000000;
                    border-radius: 6px;
                    padding: 8px;
                }
                QListWidget::item {
                    padding: 8px;
                    border-bottom: 1px solid #D0D0D0;
                }
                QListWidget::item:hover {
                    background-color: #3498DB20;
                }
            """)

    def toggle_play(self):
        self.paused = False
        self.timer.start(30)

    def toggle_pause(self):
        self.paused = True
        self.timer.stop()

    def rewind(self):
        if self.cap:
            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 30))

    def fast_forward(self):
        if self.cap:
            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + 30)


class VehicleTracker:
    def __init__(self):
        self.positions = {}
        self.speed_limit = 60  # km/h
        self.px_to_m = 0.1  # Will be updated by calibration
        self.calibrated = False

    def update(self, vehicle_id, bbox):
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        x1, y1, x2, y2 = bbox
        current_pos = ((x1 + x2) // 2, (y1 + y2) // 2)

        if vehicle_id not in self.positions:
            self.positions[vehicle_id] = {
                'positions': [current_pos],
                'timestamps': [current_time],
                'filtered_speed': 0
            }
            return 0

        # Calculate speed
        prev_pos = self.positions[vehicle_id]['positions'][-1]
        prev_time = self.positions[vehicle_id]['timestamps'][-1]

        distance_px = math.hypot(current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1])
        distance_m = distance_px * self.px_to_m
        time_diff = current_time - prev_time

        if time_diff == 0:
            return self.positions[vehicle_id]['filtered_speed']

        speed_mps = distance_m / time_diff
        speed_kph = speed_mps * 3.6  # Convert m/s to km/h

        # Apply low-pass filter for smoothing
        alpha = 0.7
        filtered_speed = alpha * speed_kph + (1 - alpha) * self.positions[vehicle_id]['filtered_speed']

        self.positions[vehicle_id]['positions'].append(current_pos)
        self.positions[vehicle_id]['timestamps'].append(current_time)
        self.positions[vehicle_id]['filtered_speed'] = filtered_speed

        return round(filtered_speed, 1)


class ClickableLabel(QLabel):
    clicked = pyqtSignal(QPoint)

    def mousePressEvent(self, event):
        self.clicked.emit(event.pos())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = SpeedDetectionApp()
    window.show()
    sys.exit(app.exec_())