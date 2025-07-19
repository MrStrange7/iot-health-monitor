#INCOMPLETE HAI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, timedelta
import json
import csv
from scipy.signal import find_peaks, butter, filtfilt
import threading
import time
from collections import deque


class RealTimeHealthMonitor:
    """Real-time health monitoring interface for ECG and vital signs"""

    def __init__(self, health_model):
        self.health_model = health_model
        self.is_monitoring = False
        self.current_data = {}
        self.data_buffer = deque(maxlen=1000)  # Store last 1000 readings
        self.predictions_history = deque(maxlen=100)

        # Initialize GUI
        self.setup_gui()

        # Data simulation thread
        self.simulation_thread = None

    def setup_gui(self):
        """Setup the graphical user interface"""
        self.root = tk.Tk()
        self.root.title("Real-Time Health Monitoring System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # Create main frames
        self.create_control_frame()
        self.create_vitals_frame()
        self.create_predictions_frame()
        self.create_alerts_frame()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("System Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var,
                                    relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_control_frame(self):
        """Create control panel"""
        control_frame = ttk.LabelFrame(self.root, text="Control Panel", padding="10")
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Buttons
        self.start_button = ttk.Button(control_frame, text="Start Monitoring",
                                       command=self.start_monitoring)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop Monitoring",
                                      command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.load_button = ttk.Button(control_frame, text="Load ECG Data",
                                      command=self.load_ecg_data)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.save_button = ttk.Button(control_frame, text="Save Report",
                                      command=self.save_report)
        self.save_button.pack(side=tk.LEFT, padx=5)

        # Patient info
        patient_frame = ttk.Frame(control_frame)
        patient_frame.pack(side=tk.RIGHT, padx=20)

        ttk.Label(patient_frame, text="Patient Info:").pack(side=tk.LEFT)

        self.age_var = tk.StringVar(value="45")
        ttk.Label(patient_frame, text="Age:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(patient_frame, textvariable=self.age_var, width=5).pack(side=tk.LEFT, padx=2)

        self.gender_var = tk.StringVar(value="Male")
        ttk.Label(patient_frame, text="Gender:").pack(side=tk.LEFT, padx=(10, 2))
        gender_combo = ttk.Combobox(patient_frame, textvariable=self.gender_var,
                                    values=["Male", "Female"], width=8)
        gender_combo.pack(side=tk.LEFT, padx=2)
        gender_combo.state(['readonly'])

    def create_vitals_frame(self):
        """Create vital signs display"""
        vitals_frame = ttk.LabelFrame(self.root, text="Current Vital Signs", padding="10")
        vitals_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create grid for vital signs
        self.vitals_labels = {}
        self.vitals_values = {}

        vitals = [
            ("Heart Rate", "bpm", "heart_rate"),
            ("SpO2", "%", "spo2"),
            ("Temperature", "°C", "temperature"),
            ("Systolic BP", "mmHg", "systolic_bp"),
            ("Diastolic BP", "mmHg", "diastolic_bp"),
            ("RR Interval", "s", "ecg_rr_interval"),
            ("PR Interval", "s", "ecg_pr_interval"),
            ("QT Interval", "s", "ecg_qt_interval"),
            ("ST Elevation", "mm", "ecg_st_elevation")
        ]

        for i, (name, unit, key) in enumerate(vitals):
            row, col = divmod(i, 3)

            frame = ttk.Frame(vitals_frame)
            frame.grid(row=row, column=col, sticky="ew", padx=10, pady=5)
            vitals_frame.columnconfigure(col, weight=1)

            label = ttk.Label(frame, text=name, font=("Arial", 10, "bold"))
            label.pack()

            value_label = ttk.Label(frame, text="--", font=("Arial", 14), foreground="blue")
            value_label.pack()

            unit_label = ttk.Label(frame, text=unit, font=("Arial", 8))
            unit_label.pack()

            self.vitals_values[key] = value_label

    def create_predictions_frame(self):
        """Create predictions display"""
        pred_frame = ttk.LabelFrame(self.root, text="Health Condition Predictions", padding="10")
        pred_frame.pack(fill=tk.X, padx=10, pady=5)

        # Create scrollable frame for predictions
        canvas = tk.Canvas(pred_frame, height=150)
        scrollbar = ttk.Scrollbar(pred_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.predictions_frame = scrollable_frame
        self.prediction_labels = {}

        for condition in self.health_model.condition_names:
            frame = ttk.Frame(self.predictions_frame)
            frame.pack(fill=tk.X, pady=2)

            name_label = ttk.Label(frame, text=condition.replace('_', ' ').title(),
                                   width=20, anchor="w")
            name_label.pack(side=tk.LEFT)

            status_label = ttk.Label(frame, text="--", width=15, anchor="center")
            status_label.pack(side=tk.LEFT, padx=10)

            prob_label = ttk.Label(frame, text="--", width=10, anchor="center")
            prob_label.pack(side=tk.LEFT)

            self.prediction_labels[condition] = {
                'status': status_label,
                'probability': prob_label
            }

    def create_alerts_frame(self):
        """Create alerts display"""
        alerts_frame = ttk.LabelFrame(self.root, text="Health Alerts", padding="10")
        alerts_frame.pack(fill=tk.X, padx=10, pady=5)

        # Text widget for alerts
        self.alerts_text = tk.Text(alerts_frame, height=6, wrap=tk.WORD,
                                   font=("Arial", 10))
        alerts_scrollbar = ttk.Scrollbar(alerts_frame, orient="vertical",
                                         command=self.alerts_text.yview)
        self.alerts_text.configure(yscrollcommand=alerts_scrollbar.set)

        self.alerts_text.pack(side="left", fill="both", expand=True)
        alerts_scrollbar.pack(side="right", fill="y")

        # Configure text tags for different alert levels
        self.alerts_text.tag_configure("normal", foreground="green")
        self.alerts_text.tag_configure("warning", foreground="orange")
        self.alerts_text.tag_configure("critical", foreground="red", background="yellow")

    def start_monitoring(self):
        """Start real-time monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set("Monitoring Active")

            # Start simulation thread
            self.simulation_thread = threading.Thread(target=self.simulate_data_stream)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()

            self.add_alert("Monitoring started", "normal")

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        if self.is_monitoring:
            self.is_monitoring = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_var.set("Monitoring Stopped")

            self.add_alert("Monitoring stopped", "normal")

    def simulate_data_stream(self):
        """Simulate real-time data stream"""
        while self.is_monitoring:
            # Generate realistic vital signs with some variation
            current_time = datetime.now()

            # Base values with realistic variations
            age = float(self.age_var.get())
            gender = 1 if self.gender_var.get() == "Male" else 0

            # Add some realistic temporal variations
            time_factor = np.sin(time.time() * 0.1) * 0.1  # Slow variation
            stress_factor = np.random.normal(0, 0.05)  # Random stress

            vital_signs = {
                'heart_rate': max(50, min(150, 75 + age * 0.2 + time_factor * 10 + stress_factor * 20)),
                'spo2': max(85, min(100, 98 - age * 0.05 + time_factor * 2 + stress_factor * 3)),
                'temperature': max(35, min(40, 36.8 + time_factor * 0.5 + stress_factor * 1)),
                'systolic_bp': max(80, min(200, 120 + age * 0.5 + time_factor * 10 + stress_factor * 20)),
                'diastolic_bp': max(50, min(120, 80 + age * 0.3 + time_factor * 5 + stress_factor * 10)),
                'ecg_rr_interval': 60 / max(50, min(150, 75 + time_factor * 10)),
                'ecg_pr_interval': max(0.1, min(0.3, 0.16 + stress_factor * 0.02)),
                'ecg_qt_interval': max(0.3, min(0.5, 0.40 + stress_factor * 0.03)),
                'ecg_st_elevation': max(0, min(8, 0.5 + abs(stress_factor) * 2)),
                'age': age,
                'gender': gender
            }

            # Update display and get predictions
            self.root.after(0, self.update_display, vital_signs)

            time.sleep(2)  # Update every 2 seconds

    def update_display(self, vital_signs):
        """Update the GUI display with new data"""
        # Update vital signs display
        for key, value in vital_signs.items():
            if key in self.vitals_values:
                if key in ['ecg_rr_interval', 'ecg_pr_interval', 'ecg_qt_interval']:
                    display_value = f"{value:.3f}"
                elif key in ['temperature', 'ecg_st_elevation']:
                    display_value = f"{value:.1f}"
                else:
                    display_value = f"{value:.0f}"

                self.vitals_values[key].config(text=display_value)

                # Color coding for abnormal values
                color = self.get_vital_color(key, value)
                self.vitals_values[key].config(foreground=color)

        # Get predictions
        predictions = self.health_model.predict_health_status(vital_signs)

        if predictions:
            # Update predictions display
            for condition, result in predictions.items():
                status = "POSITIVE" if result['prediction'] else "NEGATIVE"
                prob = f"{result['probability']:.2f}"

                self.prediction_labels[condition]['status'].config(text=status)
                self.prediction_labels[condition]['probability'].config(text=prob)

                # Color coding
                if result['prediction']:
                    if result['probability'] > 0.8:
                        color = "red"
                        bg_color = "yellow"
                    elif result['probability'] > 0.6:
                        color = "orange"
                        bg_color = "white"
                    else:
                        color = "blue"
                        bg_color = "white"
                else:
                    color = "green"
                    bg_color = "white"

                self.prediction_labels[condition]['status'].config(foreground=color, background=bg_color)

            # Check for alerts
            self.check_alerts(vital_signs, predictions)

            # Store data
            data_point = {
                'timestamp': datetime.now(),
                'vitals': vital_signs.copy(),
                'predictions': predictions.copy()
            }
            self.data_buffer.append(data_point)
            self.predictions_history.append(predictions.copy())

    def get_vital_color(self, vital_key, value):
        """Get color for vital sign based on normal ranges"""
        normal_ranges = {
            'heart_rate': (60, 100),
            'spo2': (95, 100),
            'temperature': (36.1, 37.8),
            'systolic_bp': (90, 140),
            'diastolic_bp': (60, 90),
            'ecg_rr_interval': (0.6, 1.0),
            'ecg_pr_interval': (0.12, 0.20),
            'ecg_qt_interval': (0.35, 0.45),
            'ecg_st_elevation': (0, 2)
        }

        if vital_key in normal_ranges:
            min_val, max_val = normal_ranges[vital_key]
            if min_val <= value <= max_val:
                return "green"
            elif value < min_val * 0.8 or value > max_val * 1.2:
                return "red"
            else:
                return "orange"

        return "blue"

    def check_alerts(self, vital_signs, predictions):
        """Check for critical alerts"""
        alerts = []

        # Critical vital signs
        if vital_signs['heart_rate'] < 50 or vital_signs['heart_rate'] > 120:
            alerts.append(("Critical heart rate detected!", "critical"))

        if vital_signs['spo2'] < 90:
            alerts.append(("Severe hypoxia detected!", "critical"))

        if vital_signs['temperature'] > 39:
            alerts.append(("High fever detected!", "critical"))

        if vital_signs['systolic_bp'] > 180 or vital_signs['diastolic_bp'] > 110:
            alerts.append(("Hypertensive crisis!", "critical"))

        if vital_signs['ecg_st_elevation'] > 3:
            alerts.append(("Significant ST elevation!", "critical"))

        # Condition predictions
        critical_conditions = ['myocardial_infarction', 'heart_failure']
        for condition in critical_conditions:
            if predictions[condition]['prediction'] and predictions[condition]['probability'] > 0.7:
                alerts.append((f"High risk of {condition.replace('_', ' ')}!", "critical"))

        # Add alerts to display
        for alert_text, alert_type in alerts:
            self.add_alert(alert_text, alert_type)

    def add_alert(self, message, alert_type="normal"):
        """Add alert to the alerts display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        alert_message = f"[{timestamp}] {message}\n"

        self.alerts_text.insert(tk.END, alert_message, alert_type)
        self.alerts_text.see(tk.END)

        # Limit alerts text size
        if int(self.alerts_text.index('end-1c').split('.')[0]) > 100:
            self.alerts_text.delete('1.0', '10.0')

    def load_ecg_data(self):
        """Load ECG data from file"""
        file_path = filedialog.askopenfilename(
            title="Select ECG Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            try:
                # Load and process ECG data
                df = pd.read_csv(file_path)