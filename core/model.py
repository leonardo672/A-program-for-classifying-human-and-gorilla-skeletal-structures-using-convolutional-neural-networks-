import torch
import cv2
import numpy as np
from .config import MODEL_PATH

class SkeletonModel:
    """Headless model manager for skeleton analysis"""
    
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
    
    def load_model(self):
        """Load YOLOv5 model - headless operation"""
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                      path=self.model_path, force_reload=True)
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def analyze_image(self, image_path):
        """Analyze single image - returns dict with results"""
        if self.model is None:
            if not self.load_model():
                return None
        
        try:
            results = self.model(image_path)
            return {
                'success': True,
                'predictions': results.pandas().xyxy[0].to_dict('records'),
                'image_with_detections': np.squeeze(results.render()),
                'summary': str(results),
                'image_path': image_path
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path
            }
    
    def analyze_images_batch(self, image_paths):
        """Analyze multiple images"""
        results = []
        for image_path in image_paths:
            result = self.analyze_image(image_path)
            if result:
                results.append(result)
        return results
    
    def analyze_video(self, video_path, output_callback=None):
        """Analyze video frame by frame - headless operation"""
        if self.model is None:
            if not self.load_model():
                return []
        
        cap = cv2.VideoCapture(video_path)
        detected_objects = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if output_callback:
                output_callback(f"Processing frame {frame_count}")
            
            # Resize for processing
            frame = cv2.resize(frame, (640, 480))
            
            # Analyze frame
            results = self.model(frame)
            detected_objects_frame = results.xyxy[0]
            
            # Process detections
            for obj in detected_objects_frame:
                x1, y1, x2, y2, confidence, class_id = obj
                if confidence > 0.5:  # Confidence threshold
                    object_image = frame[int(y1):int(y2), int(x1):int(x2)]
                    detected_objects.append({
                        'frame': frame_count,
                        'confidence': float(confidence),
                        'class_id': int(class_id),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'image': object_image
                    })
        
        cap.release()
        return detected_objects