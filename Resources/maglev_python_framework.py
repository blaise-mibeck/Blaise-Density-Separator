"""
MagLev Density Separation Control System
========================================

This framework provides tools for:
1. USB microscope control and image acquisition
2. Automated calibration using density standards
3. Real-time particle tracking and density measurement
4. Data logging and analysis
5. Future peristaltic pump integration

Dependencies:
pip install opencv-python numpy pandas matplotlib scipy scikit-image pyserial
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import time
from scipy import ndimage
from skimage import measure, segmentation
import serial
import threading
import queue

@dataclass
class DensityStandard:
    """Reference standard for calibration"""
    name: str
    density: float  # g/cm³
    color: str  # for identification
    diameter: float  # mm, if known

@dataclass
class ParticleData:
    """Data structure for tracked particles"""
    id: int
    position: Tuple[float, float]  # (x, y) in pixels
    height: float  # levitation height in mm
    density: float  # calculated density
    area: float  # particle area in pixels
    timestamp: datetime

class MagLevMicroscope:
    """USB Microscope controller for MagLev system"""
    
    def __init__(self, camera_index=0, resolution=(1920, 1080)):
        self.camera_index = camera_index
        self.resolution = resolution
        self.cap = None
        self.is_connected = False
        self.current_frame = None
        self.background = None
        
    def connect(self):
        """Initialize camera connection"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.is_connected = True
                print(f"✅ Camera connected at resolution {self.resolution}")
                return True
            else:
                print("❌ Failed to connect to camera")
                return False
        except Exception as e:
            print(f"❌ Camera connection error: {e}")
            return False
    
    def capture_frame(self):
        """Capture a single frame"""
        if not self.is_connected:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            return frame
        return None
    
    def set_background(self, num_frames=10):
        """Capture background image for particle detection"""
        if not self.is_connected:
            return False
        
        frames = []
        for i in range(num_frames):
            frame = self.capture_frame()
            if frame is not None:
                frames.append(frame.astype(np.float32))
            time.sleep(0.1)
        
        if frames:
            self.background = np.mean(frames, axis=0).astype(np.uint8)
            print(f"✅ Background captured from {len(frames)} frames")
            return True
        return False
    
    def disconnect(self):
        """Close camera connection"""
        if self.cap:
            self.cap.release()
            self.is_connected = False
            print("📷 Camera disconnected")

class MagLevCalibration:
    """Calibration system using density standards"""
    
    def __init__(self, microscope: MagLevMicroscope):
        self.microscope = microscope
        self.standards = []
        self.calibration_data = []
        self.height_to_density_func = None
        self.pixel_to_mm_ratio = None
        
    def add_standard(self, standard: DensityStandard):
        """Add a density standard for calibration"""
        self.standards.append(standard)
        print(f"Added standard: {standard.name} ({standard.density} g/cm³)")
    
    def load_default_standards(self):
        """Load common density standards"""
        defaults = [
            DensityStandard("PMMA", 1.18, "clear", 3.0),
            DensityStandard("Polystyrene", 1.05, "white", 3.0),
            DensityStandard("Nylon", 1.14, "white", 3.0),
            DensityStandard("Glass", 2.50, "clear", 2.0),
            DensityStandard("Aluminum", 2.70, "silver", 2.0),
        ]
        for std in defaults:
            self.add_standard(std)
    
    def calibrate_scale(self, known_distance_mm, pixel_distance):
        """Calibrate pixel to mm conversion"""
        self.pixel_to_mm_ratio = known_distance_mm / pixel_distance
        print(f"✅ Scale calibrated: {self.pixel_to_mm_ratio:.4f} mm/pixel")
    
    def perform_calibration(self, magnet_separation_mm=45):
        """Perform full system calibration"""
        print("🔧 Starting calibration sequence...")
        
        if not self.microscope.is_connected:
            print("❌ Microscope not connected")
            return False
        
        # Set background
        print("📸 Capturing background...")
        if not self.microscope.set_background():
            print("❌ Failed to capture background")
            return False
        
        # Manual calibration points
        calibration_points = []
        
        for standard in self.standards:
            print(f"\n🎯 Place {standard.name} standard and press Enter...")
            input()
            
            frame = self.microscope.capture_frame()
            if frame is None:
                continue
            
            # Detect particle position
            position = self._detect_particle_position(frame)
            if position:
                height_mm = self._pixel_to_height(position[1], magnet_separation_mm)
                calibration_points.append((height_mm, standard.density))
                print(f"✅ Recorded: height={height_mm:.2f}mm, density={standard.density}")
        
        if len(calibration_points) >= 2:
            # Create linear calibration function
            heights, densities = zip(*calibration_points)
            coeffs = np.polyfit(heights, densities, 1)
            self.height_to_density_func = np.poly1d(coeffs)
            
            # Save calibration
            self._save_calibration(calibration_points)
            print("✅ Calibration completed successfully!")
            return True
        else:
            print("❌ Insufficient calibration points")
            return False
    
    def _detect_particle_position(self, frame):
        """Detect particle position in frame"""
        if self.microscope.background is None:
            return None
        
        # Background subtraction
        diff = cv2.absdiff(frame, self.microscope.background)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Threshold and find contours
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (assumed to be the particle)
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
        return None
    
    def _pixel_to_height(self, y_pixel, magnet_separation_mm):
        """Convert pixel position to height in mm"""
        if self.pixel_to_mm_ratio is None:
            # Assume full frame height corresponds to magnet separation
            self.pixel_to_mm_ratio = magnet_separation_mm / self.microscope.resolution[1]
        
        # Convert y-position to height from bottom magnet
        height_mm = (self.microscope.resolution[1] - y_pixel) * self.pixel_to_mm_ratio
        return height_mm
    
    def _save_calibration(self, calibration_points):
        """Save calibration data to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"maglev_calibration_{timestamp}.json"
        
        data = {
            "timestamp": timestamp,
            "calibration_points": calibration_points,
            "pixel_to_mm_ratio": self.pixel_to_mm_ratio,
            "coefficients": self.height_to_density_func.coefficients.tolist()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Calibration saved to {filename}")

class MagLevAnalyzer:
    """Real-time analysis and particle tracking"""
    
    def __init__(self, microscope: MagLevMicroscope, calibration: MagLevCalibration):
        self.microscope = microscope
        self.calibration = calibration
        self.particles = []
        self.particle_counter = 0
        self.data_log = []
        self.is_recording = False
        
    def start_analysis(self, duration_seconds=None):
        """Start real-time particle analysis"""
        print("🔍 Starting real-time analysis...")
        self.is_recording = True
        start_time = time.time()
        
        try:
            while self.is_recording:
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    break
                
                frame = self.microscope.capture_frame()
                if frame is None:
                    continue
                
                # Detect and analyze particles
                detected_particles = self._detect_particles(frame)
                self._update_particle_tracking(detected_particles)
                
                # Display frame with annotations
                annotated_frame = self._annotate_frame(frame, detected_particles)
                cv2.imshow('MagLev Analysis', annotated_frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.033)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\n⏹️ Analysis stopped by user")
        finally:
            self.is_recording = False
            cv2.destroyAllWindows()
            self._save_analysis_data()
    
    def _detect_particles(self, frame):
        """Detect all particles in current frame"""
        if self.microscope.background is None:
            return []
        
        # Background subtraction and preprocessing
        diff = cv2.absdiff(frame, self.microscope.background)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Noise reduction and thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        particles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 5000:  # Filter by reasonable particle size
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate density if calibrated
                    height_mm = self.calibration._pixel_to_height(cy, 45)  # Assuming 45mm separation
                    density = self._calculate_density(height_mm)
                    
                    particle = ParticleData(
                        id=self.particle_counter,
                        position=(cx, cy),
                        height=height_mm,
                        density=density,
                        area=area,
                        timestamp=datetime.now()
                    )
                    particles.append(particle)
                    self.particle_counter += 1
        
        return particles
    
    def _calculate_density(self, height_mm):
        """Calculate density from height using calibration"""
        if self.calibration.height_to_density_func is not None:
            return float(self.calibration.height_to_density_func(height_mm))
        return 0.0
    
    def _update_particle_tracking(self, detected_particles):
        """Update particle tracking and logging"""
        for particle in detected_particles:
            self.data_log.append({
                'timestamp': particle.timestamp.isoformat(),
                'particle_id': particle.id,
                'x_position': particle.position[0],
                'y_position': particle.position[1],
                'height_mm': particle.height,
                'density_g_cm3': particle.density,
                'area_pixels': particle.area
            })
    
    def _annotate_frame(self, frame, particles):
        """Add annotations to frame for display"""
        annotated = frame.copy()
        
        # Draw reference lines and scale
        height, width = frame.shape[:2]
        
        # Draw horizontal reference lines every 5mm
        if self.calibration.pixel_to_mm_ratio:
            for h in range(0, 50, 5):  # 0-50mm in 5mm increments
                y_pos = height - int(h / self.calibration.pixel_to_mm_ratio)
                if 0 <= y_pos <= height:
                    cv2.line(annotated, (0, y_pos), (width, y_pos), (0, 255, 0), 1)
                    cv2.putText(annotated, f"{h}mm", (10, y_pos-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw particles and their data
        for particle in particles:
            x, y = particle.position
            
            # Draw particle circle
            cv2.circle(annotated, (int(x), int(y)), 5, (0, 0, 255), 2)
            
            # Draw density label
            density_text = f"{particle.density:.3f} g/cm³"
            cv2.putText(annotated, density_text, (int(x)+10, int(y)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw height label
            height_text = f"{particle.height:.2f}mm"
            cv2.putText(annotated, height_text, (int(x)+10, int(y)+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add status information
        status_text = f"Particles: {len(particles)} | Recording: {self.is_recording}"
        cv2.putText(annotated, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def _save_analysis_data(self):
        """Save analysis data to CSV file"""
        if self.data_log:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"maglev_analysis_{timestamp}.csv"
            
            df = pd.DataFrame(self.data_log)
            df.to_csv(filename, index=False)
            print(f"💾 Analysis data saved to {filename}")
            
            # Generate summary statistics
            self._generate_summary_report(df)
    
    def _generate_summary_report(self, df):
        """Generate analysis summary report"""
        print("\n📊 Analysis Summary:")
        print(f"Total particles detected: {len(df)}")
        print(f"Unique particles: {df['particle_id'].nunique()}")
        
        if 'density_g_cm3' in df.columns and len(df) > 0:
            densities = df['density_g_cm3'][df['density_g_cm3'] > 0]
            if len(densities) > 0:
                print(f"Density range: {densities.min():.3f} - {densities.max():.3f} g/cm³")
                print(f"Mean density: {densities.mean():.3f} g/cm³")
                print(f"Std deviation: {densities.std():.3f} g/cm³")

class PeristalticPumpController:
    """Controller for peristaltic pumps (future implementation)"""
    
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.is_connected = False
        
    def connect(self):
        """Connect to pump controller"""
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            self.is_connected = True
            print(f"✅ Pump controller connected on {self.port}")
            return True
        except Exception as e:
            print(f"❌ Pump connection failed: {e}")
            return False
    
    def pump_volume(self, volume_ml, flow_rate_ml_min=1.0):
        """Pump specified volume at given flow rate"""
        if not self.is_connected:
            print("❌ Pump not connected")
            return False
        
        # Calculate pump time
        pump_time = (volume_ml / flow_rate_ml_min) * 60  # seconds
        
        try:
            # Send pump command (protocol depends on specific pump)
            command = f"PUMP {volume_ml} {flow_rate_ml_min}\n"
            self.serial_connection.write(command.encode())
            
            print(f"🔄 Pumping {volume_ml}ml at {flow_rate_ml_min}ml/min...")
            time.sleep(pump_time)
            
            return True
        except Exception as e:
            print(f"❌ Pump operation failed: {e}")
            return False
    
    def stop_pump(self):
        """Emergency stop"""
        if self.is_connected:
            self.serial_connection.write(b"STOP\n")
            print("⏹️ Pump stopped")
    
    def disconnect(self):
        """Disconnect from pump"""
        if self.serial_connection:
            self.serial_connection.close()
            self.is_connected = False
            print("🔌 Pump disconnected")

class MagLevBatchProcessor:
    """Automated batch processing system"""
    
    def __init__(self, microscope, analyzer, pump_controller=None):
        self.microscope = microscope
        self.analyzer = analyzer
        self.pump_controller = pump_controller
        self.batch_results = []
        
    def run_batch_separation(self, sample_volumes_ml, collection_densities, analysis_time=60):
        """Run automated batch separation"""
        print("🏭 Starting batch separation process...")
        
        for i, volume in enumerate(sample_volumes_ml):
            print(f"\n--- Processing Sample {i+1} ({volume}ml) ---")
            
            # Load sample
            if self.pump_controller:
                self.pump_controller.pump_volume(volume, flow_rate_ml_min=2.0)
                time.sleep(5)  # Allow settling
            else:
                print(f"💧 Manually load {volume}ml sample and press Enter...")
                input()
            
            # Analyze sample
            print("🔍 Analyzing sample...")
            self.analyzer.start_analysis(duration_seconds=analysis_time)
            
            # Collect fractions based on density
            fractions = self._collect_density_fractions(collection_densities)
            
            # Record batch results
            self.batch_results.append({
                'sample_id': i+1,
                'volume_ml': volume,
                'fractions': fractions,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"✅ Sample {i+1} processing complete")
        
        self._save_batch_results()
        print("🎉 Batch processing complete!")
    
    def _collect_density_fractions(self, target_densities):
        """Collect particles in different density ranges"""
        # This would interface with collection system
        # For now, just simulate the process
        
        fractions = {}
        for i, density in enumerate(target_densities):
            if self.pump_controller:
                # Move to collection position for this density
                # This would require coordination with the magnetic field position
                print(f"📦 Collecting fraction {i+1} (density ~{density} g/cm³)")
                self.pump_controller.pump_volume(1.0)  # Collect 1ml
                fractions[f"fraction_{i+1}"] = {
                    'target_density': density,
                    'collected_volume': 1.0
                }
            else:
                print(f"📦 Manually collect fraction {i+1} for density ~{density} g/cm³")
                fractions[f"fraction_{i+1}"] = {
                    'target_density': density,
                    'collected_volume': 'manual'
                }
        
        return fractions
    
    def _save_batch_results(self):
        """Save batch processing results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"maglev_batch_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.batch_results, f, indent=2)
        print(f"💾 Batch results saved to {filename}")

def main():
    """Main application entry point"""
    print("🧲 MagLev Density Separation System")
    print("===================================")
    
    # Initialize components
    microscope = MagLevMicroscope(camera_index=0)
    calibration = MagLevCalibration(microscope)
    
    # Connect microscope
    if not microscope.connect():
        print("❌ Cannot proceed without microscope connection")
        return
    
    try:
        while True:
            print("\n🎛️ Main Menu:")
            print("1. Perform system calibration")
            print("2. Start real-time analysis")
            print("3. Run batch processing")
            print("4. Load calibration standards")
            print("5. Test pump controller")
            print("6. Exit")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                calibration.load_default_standards()
                if calibration.perform_calibration():
                    print("✅ System ready for analysis")
                else:
                    print("❌ Calibration failed")
            
            elif choice == '2':
                if calibration.height_to_density_func is None:
                    print("⚠️ System not calibrated. Running calibration first...")
                    calibration.load_default_standards()
                    if not calibration.perform_calibration():
                        print("❌ Cannot start analysis without calibration")
                        continue
                
                analyzer = MagLevAnalyzer(microscope, calibration)
                print("Press 'q' in the video window to stop analysis")
                analyzer.start_analysis()
            
            elif choice == '3':
                if calibration.height_to_density_func is None:
                    print("❌ System must be calibrated before batch processing")
                    continue
                
                analyzer = MagLevAnalyzer(microscope, calibration)
                pump = PeristalticPumpController()
                
                # Try to connect pump (optional)
                pump.connect()
                
                batch_processor = MagLevBatchProcessor(microscope, analyzer, pump)
                
                # Example batch run
                volumes = [2.0, 2.0, 2.0]  # 3 samples of 2ml each
                densities = [1.0, 1.2, 1.4]  # Target collection densities
                
                batch_processor.run_batch_separation(volumes, densities, analysis_time=30)
            
            elif choice == '4':
                calibration.load_default_standards()
                print("✅ Default calibration standards loaded")
            
            elif choice == '5':
                pump = PeristalticPumpController()
                if pump.connect():
                    volume = float(input("Enter volume to pump (ml): "))
                    rate = float(input("Enter flow rate (ml/min): "))
                    pump.pump_volume(volume, rate)
                    pump.disconnect()
            
            elif choice == '6':
                break
            
            else:
                print("❌ Invalid option")
    
    except KeyboardInterrupt:
        print("\n⏹️ Program interrupted by user")
    
    finally:
        # Cleanup
        microscope.disconnect()
        print("👋 MagLev system shutdown complete")

if __name__ == "__main__":
    main()