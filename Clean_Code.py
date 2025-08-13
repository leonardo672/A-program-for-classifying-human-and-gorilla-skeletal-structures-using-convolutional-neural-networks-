"""
Object Detection System with YOLOv5 and PyQt5 GUI

Features:
- Image and video analysis using YOLOv5 model
- Database integration for storing results
- Comprehensive GUI for visualization and control
"""

import os
import sys
import cv2
import numpy as np
import torch
import pyodbc
import configparser
from typing import Optional, List, Tuple, Dict, Any

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QMainWindow, QPushButton, QFileDialog,
    QTableWidget, QTableWidgetItem, QLabel, QGridLayout, QScrollArea, QMessageBox, 
    QDialog, QLineEdit, QHBoxLayout
)
from PyQt5.QtGui import (
    QPixmap, QFont, QIcon, QImage, QPalette, QBrush, QImageReader
)
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QSize
from PyQt5.QtWidgets import QHeaderView

# Constants
CONFIG_FILE = "config.ini"
DEFAULT_IMAGE_SIZE = QSize(400, 400)
BUTTON_STYLE = """
    QPushButton {{
        background-color: {bg_color}; 
        color: black; 
        border: 2px solid gray; 
        margin-bottom: 0px; 
        margin-top: 0px;
        border-radius: 10px;
        font-family: 'Times New Roman';
        font-weight: bold;
        font-size: {font_size}px;
        padding: {padding}px;
    }}
    QPushButton:hover {{
        background-color: lightgray; 
    }}
"""

class ConfigManager:
    """Handles configuration loading and management"""
    
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        self.load_config()
        
    def load_config(self) -> None:
        """Load configuration from file"""
        if not os.path.exists(self.config_file):
            self.create_default_config()
        self.config.read(self.config_file)
        
    def create_default_config(self) -> None:
        """Create default configuration file"""
        self.config['PATHS'] = {
            'model_weights': 'C:/Users/L/yolov5/runs/train/exp2/weights/last.pt',
            'image_folder': 'D:/My_Diploma/System2/skeletal2',
            'output_folder': 'D:/My_Diploma/System2/skeletal3',
            'database_path': 'D:/My_Diploma/System2/Homam_SK.accdb',
            'background_image': 'D:/My_Diploma/is-machine-learning-hard-a-guide-to-getting-started-scaled-1-scaled.jpeg',
            'structure_image': 'D:/My_Diploma/System2/SFD.png'
        }
        
        self.config['UI'] = {
            'window_width': '1900',
            'window_height': '900',
            'button_height': '70',
            'font_size': '14',
            'button_padding': '10'
        }
        
        with open(self.config_file, 'w') as f:
            self.config.write(f)
            
    def get_path(self, key: str) -> str:
        """Get a path from configuration"""
        return self.config['PATHS'].get(key, '')
        
    def get_ui_setting(self, key: str, default: Any = None) -> Any:
        """Get a UI setting from configuration"""
        return self.config['UI'].get(key, default)


class DetectionModel:
    """Wrapper for YOLOv5 model operations"""
    
    def __init__(self, model_path: str):
        self.model = torch.hub.load(
            'ultralytics/yolov5', 
            'custom', 
            path=model_path, 
            force_reload=True
        )
        
    def detect_objects(self, image: np.ndarray) -> Any:
        """Perform object detection on an image"""
        return self.model(image)
        
    def process_results(self, results: Any) -> Tuple[str, np.ndarray]:
        """Process detection results"""
        results_output = str(results)
        rendered_image = np.squeeze(results.render())
        return results_output, rendered_image


class DatabaseManager:
    """Handles database operations"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        
    def __enter__(self):
        self.conn = pyodbc.connect(self.connection_string)
        self.cursor = self.conn.cursor()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.conn.commit()
        self.conn.close()
        
    def save_detection(
        self, 
        image_name: str, 
        results: str, 
        detected_image_path: str, 
        original_image_path: str
    ) -> None:
        """Save detection results to database"""
        detected_image_link = self._create_hyperlink(detected_image_path)
        original_image_link = self._create_hyperlink(original_image_path)
        
        query = """
            INSERT INTO Detections 
            (Image_name, Results, Detected_image_link, Original_image_link)
            VALUES (?, ?, ?, ?)
        """
        self.cursor.execute(query, (
            image_name, 
            results, 
            detected_image_link, 
            original_image_link
        ))
        
    def _create_hyperlink(self, path: str) -> str:
        """Create a hyperlink string for database"""
        return f'=HYPERLINK("{os.path.abspath(path).replace("/")}", "Click to open image")'
        
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user credentials"""
        query = "SELECT Username FROM Users WHERE Username=? AND Password=?"
        self.cursor.execute(query, (username, password))
        return self.cursor.fetchone() is not None
        
    def register_user(
        self, 
        username: str, 
        password: str, 
        email: str, 
        phone: str
    ) -> bool:
        """Register a new user"""
        # Check if username exists
        self.cursor.execute("SELECT Username FROM Users WHERE Username=?", (username,))
        if self.cursor.fetchone():
            return False
            
        # Insert new user
        query = """
            INSERT INTO Users 
            (Username, Password, Email, Phone_Number) 
            VALUES (?, ?, ?, ?)
        """
        self.cursor.execute(query, (username, password, email, phone))
        return True


class ImageProcessor:
    """Handles image processing operations"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.model = DetectionModel(config.get_path('model_weights'))
        
    def process_image_folder(self) -> None:
        """Process all images in configured folder"""
        image_folder = self.config.get_path('image_folder')
        output_folder = self.config.get_path('output_folder')
        os.makedirs(output_folder, exist_ok=True)
        
        conn_str = f'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={self.config.get_path("database_path")};'
        
        with DatabaseManager(conn_str) as db:
            for image_file in self._get_image_files(image_folder):
                self._process_single_image(image_folder, output_folder, image_file, db)
                
    def _get_image_files(self, folder: str) -> List[str]:
        """Get list of image files in folder"""
        return [
            f for f in os.listdir(folder) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
    def _process_single_image(
        self, 
        input_folder: str, 
        output_folder: str, 
        image_file: str, 
        db: DatabaseManager
    ) -> None:
        """Process a single image and save results"""
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to load image: {img_path}")
            return
            
        results = self.model.detect_objects(img)
        results_output, rendered_image = self.model.process_results(results)
        
        # Save output image
        output_path = os.path.join(output_folder, f'detected_{image_file}')
        cv2.imwrite(output_path, rendered_image)
        
        # Save to database
        db.save_detection(
            image_file, 
            results_output, 
            output_path, 
            img_path
        )


class BaseForm(QWidget):
    """Base class for forms with common functionality"""
    
    def __init__(self, title: str, config: ConfigManager):
        super().__init__()
        self.config = config
        self.setWindowTitle(title)
        self.init_ui()
        
    def init_ui(self) -> None:
        """Initialize UI components"""
        raise NotImplementedError("Subclasses must implement init_ui")
        
    def set_background(self, image_path: str) -> None:
        """Set background image for the form"""
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path).scaled(
                self.size(), 
                Qt.KeepAspectRatioByExpanding
            )
            palette = self.palette()
            palette.setBrush(QPalette.Background, QBrush(pixmap))
            self.setPalette(palette)
            
    def create_button(
        self, 
        text: str, 
        position: Tuple[int, int, int, int], 
        bg_color: str = "#B0BBC1",
        font_size: int = 14,
        padding: int = 10
    ) -> QPushButton:
        """Create a styled button"""
        button = QPushButton(text, self)
        button.setGeometry(*position)
        button.setStyleSheet(BUTTON_STYLE.format(
            bg_color=bg_color,
            font_size=font_size,
            padding=padding
        ))
        button.setFont(QFont("Times New Roman", font_size, QFont.Bold))
        return button
        
    def show_image_grid(self, folder_path: str) -> None:
        """Display images from folder in a grid layout"""
        # Clear existing widgets
        for i in reversed(range(self.layout().count())):
            self.layout().itemAt(i).widget().setParent(None)
            
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedSize(1400, 600)
        
        # Create scroll widget
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background-color: black;")
        scroll_area.setWidget(scroll_widget)
        
        # Create grid layout
        grid_layout = QGridLayout(scroll_widget)
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(20)
        
        # Add images to grid
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        num_columns = 3
        for index, image_name in enumerate(images):
            row = index // num_columns
            col = index % num_columns
            image_path = os.path.join(folder_path, image_name)
            
            if os.path.exists(image_path):
                image_label = QLabel()
                pixmap = QPixmap(image_path).scaled(1200, 600, Qt.KeepAspectRatio)
                image_label.setPixmap(pixmap)
                grid_layout.addWidget(image_label, row, col)
                
        self.layout().addWidget(scroll_area)


class ModelOverviewForm(BaseForm):
    """Form for displaying model performance metrics"""
    
    def __init__(self, config: ConfigManager):
        super().__init__('Model Performance Overview', config)
        
    def init_ui(self) -> None:
        """Initialize UI components"""
        self.set_background(self.config.get_path('background_image'))
        self.setGeometry(50, 50, 1500, 1200)
        
        # Create buttons for different model metrics
        buttons = [
            ("Confusion Matrix", (30, 6, 220, 60), self.show_confusion_matrix),
            ("F1 Curve", (260, 6, 150, 60), self.show_f1_curve),
            ("Labels", (420, 6, 150, 60), self.show_labels),
            ("Labels Correlogram", (585, 6, 200, 60), self.show_labels_correlogram),
            ("P Curve", (800, 6, 150, 60), self.show_p_curve),
            ("PR Curve", (960, 6, 150, 60), self.show_pr_curve),
            ("R Curve", (1120, 6, 150, 60), self.show_r_curve),
            ("Results", (1280, 6, 150, 60), self.show_results)
        ]
        
        for text, geometry, handler in buttons:
            button = self.create_button(text, geometry)
            button.clicked.connect(handler)
            
    def show_confusion_matrix(self) -> None:
        """Show confusion matrix images"""
        folder = os.path.join(self.config.get_path('image_folder'), 'model/confusion_matrix')
        self.show_image_grid(folder)
        
    def show_f1_curve(self) -> None:
        """Show F1 curve images"""
        folder = os.path.join(self.config.get_path('image_folder'), 'model/F1_curve')
        self.show_image_grid(folder)
        
    def show_labels(self) -> None:
        """Show labels images"""
        folder = os.path.join(self.config.get_path('image_folder'), 'model/labels')
        self.show_image_grid(folder)
        
    def show_labels_correlogram(self) -> None:
        """Show labels correlogram images"""
        folder = os.path.join(self.config.get_path('image_folder'), 'model/labels_correlogram')
        self.show_image_grid(folder)
        
    def show_p_curve(self) -> None:
        """Show P curve images"""
        folder = os.path.join(self.config.get_path('image_folder'), 'model/P_curve')
        self.show_image_grid(folder)
        
    def show_pr_curve(self) -> None:
        """Show PR curve images"""
        folder = os.path.join(self.config.get_path('image_folder'), 'model/PR_curve')
        self.show_image_grid(folder)
        
    def show_r_curve(self) -> None:
        """Show R curve images"""
        folder = os.path.join(self.config.get_path('image_folder'), 'model/R_curve')
        self.show_image_grid(folder)
        
    def show_results(self) -> None:
        """Show results images"""
        folder = os.path.join(self.config.get_path('image_folder'), 'model/results')
        self.show_image_grid(folder)


class StructureOfObjectsForm(BaseForm):
    """Form for displaying object structure information"""
    
    def __init__(self, config: ConfigManager):
        super().__init__('Object Structure', config)
        
    def init_ui(self) -> None:
        """Initialize UI components"""
        self.set_background(self.config.get_path('structure_image'))
        self.setGeometry(100, 100, 800, 700)
        
        # Create buttons for different object types
        human_button = self.create_button(
            "Human Skeleton", 
            (130, 6, 290, 60),
            "#9F8B79"
        )
        human_button.clicked.connect(self.show_human_skeletal)
        
        gorilla_button = self.create_button(
            "Gorilla Skeleton", 
            (450, 6, 245, 60),
            "#9F8B79"
        )
        gorilla_button.clicked.connect(self.show_gorilla_skeletal)
        
    def show_human_skeletal(self) -> None:
        """Show human skeletal images"""
        folder = os.path.join(self.config.get_path('image_folder'), 'Human_Skeletal')
        self.show_image_grid(folder)
        
    def show_gorilla_skeletal(self) -> None:
        """Show gorilla skeletal images"""
        folder = os.path.join(self.config.get_path('image_folder'), 'Gorilla_Skeletal')
        self.show_image_grid(folder)


class ResultsWindow(QMainWindow):
    """Window for displaying detection results"""
    
    def __init__(
        self, 
        config: ConfigManager,
        image_path: Optional[str] = None, 
        folder_path: Optional[str] = None, 
        video_path: Optional[str] = None,
        image_size: Tuple[int, int] = (400, 400)
    ):
        super().__init__()
        self.config = config
        self.image_size = image_size
        self.model = DetectionModel(config.get_path('model_weights'))
        self.setWindowTitle('Detection Results')
        
        self.init_ui(image_path, folder_path, video_path)
        
    def init_ui(
        self,
        image_path: Optional[str],
        folder_path: Optional[str],
        video_path: Optional[str]
    ) -> None:
        """Initialize UI based on input type"""
        layout = QVBoxLayout()
        
        if image_path:
            self.analyze_single_image(image_path, layout)
        elif folder_path:
            self.analyze_images_in_folder(folder_path, layout)
        elif video_path:
            self.analyze_video(video_path, layout)
            
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
    def analyze_single_image(self, image_path: str, layout: QVBoxLayout) -> None:
        """Analyze a single image and display results"""
        results = self.model.detect_objects(image_path)
        self.display_results(results, layout)
        
    def analyze_images_in_folder(self, folder_path: str, layout: QVBoxLayout) -> None:
        """Analyze all images in folder and display results"""
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.png')):
                image_path = os.path.join(folder_path, filename)
                results = self.model.detect_objects(image_path)
                self.display_results(results, layout)
                
    def analyze_video(self, video_path: str, layout: QVBoxLayout) -> None:
        """Analyze video and display results"""
        processor = VideoAnalyzer(self.config)
        processor.detect_and_save_objects(video_path)
        
    def display_results(self, results: Any, layout: QVBoxLayout) -> None:
        """Display detection results in the UI"""
        if results:
            # Display text results
            text_label = QLabel(str(results), self)
            layout.addWidget(text_label)
            
            # Display rendered image
            img_array = np.squeeze(results.render())
            img_array_resized = cv2.resize(img_array, self.image_size)
            
            height, width, channel = img_array_resized.shape
            bytes_per_line = 3 * width
            
            q_img = QImage(
                img_array_resized.data, 
                width, 
                height, 
                bytes_per_line, 
                QImage.Format_RGB888
            )
            
            pixmap = QPixmap.fromImage(q_img)
            img_label = QLabel(self)
            img_label.setPixmap(pixmap)
            layout.addWidget(img_label)


class VideoAnalyzer(QObject):
    """Handles video analysis operations"""
    
    video_analysis_finished = pyqtSignal()
    
    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        self.model = DetectionModel(config.get_path('model_weights'))
        
    def detect_and_save_objects(
        self, 
        input_path: str, 
        new_width: int = 640, 
        new_height: int = 480
    ) -> None:
        """Process video file and save detected objects"""
        cap = cv2.VideoCapture(input_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, (new_width, new_height))
            results = self.model.detect_objects(frame)
            detected_objects = results.xyxy[0]
            
            for obj in detected_objects:
                x1, y1, x2, y2, confidence, class_id = obj
                object_image = frame[int(y1):int(y2), int(x1):int(x2)]
                
                output_path = os.path.join(
                    self.config.get_path('output_folder'),
                    f"object_{int(x1)}_{int(y1)}.jpg"
                )
                cv2.imwrite(output_path, object_image)
                
            cv2.imshow('Video Analysis', np.squeeze(results.render()))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.video_analysis_finished.emit()


class ImageDisplayForm(BaseForm):
    """Form for displaying analyzed images"""
    
    def __init__(self, config: ConfigManager):
        super().__init__('Analyzed Images Database', config)
        
    def init_ui(self) -> None:
        """Initialize UI components"""
        self.setGeometry(200, 200, 800, 600)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background-color: black;")
        scroll_area.setWidget(scroll_widget)
        
        grid_layout = QGridLayout(scroll_widget)
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(20)
        
        folder_path = os.path.join(self.config.get_path('image_folder'), 'On_DataBase')
        self._populate_image_grid(grid_layout, folder_path)
        
        layout = QVBoxLayout()
        layout.addWidget(scroll_area)
        self.setLayout(layout)
        
    def _populate_image_grid(
        self, 
        grid_layout: QGridLayout, 
        folder_path: str
    ) -> None:
        """Populate grid with images from folder"""
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        num_columns = 3
        for index, image_name in enumerate(images):
            row = index // num_columns
            col = index % num_columns
            image_path = os.path.join(folder_path, image_name)
            
            image_label = QLabel()
            pixmap = QPixmap(image_path).scaled(400, 400, Qt.KeepAspectRatio)
            image_label.setPixmap(pixmap)
            grid_layout.addWidget(image_label, row, col)


class RealTimeDatabaseForm(BaseForm):
    """Form for displaying real-time database images"""
    
    def __init__(self, config: ConfigManager):
        super().__init__('Real-Time Database', config)
        
    def init_ui(self) -> None:
        """Initialize UI components"""
        self.setGeometry(200, 200, 800, 600)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background-color: black;")
        scroll_area.setWidget(scroll_widget)
        
        grid_layout = QGridLayout(scroll_widget)
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(20)
        
        folder_path = os.path.join(self.config.get_path('image_folder'), 'Our_DataBase')
        self._populate_image_grid(grid_layout, folder_path)
        
        layout = QVBoxLayout()
        layout.addWidget(scroll_area)
        self.setLayout(layout)
        
    def _populate_image_grid(
        self, 
        grid_layout: QGridLayout, 
        folder_path: str
    ) -> None:
        """Populate grid with images from folder"""
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        num_columns = 3
        for index, image_name in enumerate(images):
            row = index // num_columns
            col = index % num_columns
            image_path = os.path.join(folder_path, image_name)
            
            image_label = QLabel()
            pixmap = QPixmap(image_path).scaled(300, 300, Qt.KeepAspectRatio)
            image_label.setPixmap(pixmap)
            grid_layout.addWidget(image_label, row, col)


class DatabaseForm(QMainWindow):
    """Form for displaying detection database"""
    
    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        self.setWindowTitle('Detections Database')
        self.setGeometry(200, 200, 800, 600)
        self.setStyleSheet("background-color: #f0f0f0;")
        
        self.init_ui()
        
    def init_ui(self) -> None:
        """Initialize UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        self.table_widget = QTableWidget()
        self.table_widget.setStyleSheet("background-color: #B0BBC1;")
        layout.addWidget(self.table_widget)
        
        self.load_database_data()
        
        central_widget.setLayout(layout)
        
    def load_database_data(self) -> None:
        """Load data from database into table"""
        conn_str = (
            f'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};'
            f'DBQ={self.config.get_path("database_path")};'
        )
        
        with DatabaseManager(conn_str) as db:
            db.cursor.execute("SELECT * FROM Detections")
            rows = db.cursor.fetchall()
            columns = [column[0] for column in db.cursor.description]
            
            self.table_widget.setRowCount(len(rows))
            self.table_widget.setColumnCount(len(columns))
            self.table_widget.setHorizontalHeaderLabels(columns)
            self.table_widget.horizontalHeader().setStyleSheet(
                "background-color: #d9d9d9;"
            )
            self.table_widget.horizontalHeader().setSectionResizeMode(
                QHeaderView.Stretch
            )
            
            for i, row in enumerate(rows):
                for j, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    self.table_widget.setItem(i, j, item)


class LoginDialog(QDialog):
    """Login and registration dialog"""
    
    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        self.setWindowTitle('Login / Registration')
        self.setGeometry(200, 200, 1400, 600)
        self.setStyleSheet("background-color: #1d1d1d; color: #f9f9f9;")
        
        self.init_ui()
        
    def init_ui(self) -> None:
        """Initialize UI components"""
        layout = QGridLayout()
        
        # Login section
        login_username_label = self.create_label('Username:')
        login_password_label = self.create_label('Password:')
        
        self.login_username_edit = self.create_line_edit()
        self.login_password_edit = self.create_line_edit()
        self.login_password_edit.setEchoMode(QLineEdit.Password)
        
        login_button = self.create_button('Login', self.login)
        
        layout.addWidget(login_username_label, 0, 0)
        layout.addWidget(self.login_username_edit, 0, 1)
        layout.addWidget(login_password_label, 1, 0)
        layout.addWidget(self.login_password_edit, 1, 1)
        layout.addWidget(login_button, 2, 1)
        
        # Registration section
        register_username_label = self.create_label('Username:')
        register_email_label = self.create_label('Email:')
        register_phone_label = self.create_label('Phone:')
        register_password_label = self.create_label('Password:')
        
        self.register_username_edit = self.create_line_edit()
        self.register_email_edit = self.create_line_edit()
        self.register_phone_edit = self.create_line_edit()
        self.register_password_edit = self.create_line_edit()
        self.register_password_edit.setEchoMode(QLineEdit.Password)
        
        register_button = self.create_button('Register', self.register)
        
        layout.addWidget(register_username_label, 0, 2)
        layout.addWidget(self.register_username_edit, 0, 3)
        layout.addWidget(register_email_label, 1, 2)
        layout.addWidget(self.register_email_edit, 1, 3)
        layout.addWidget(register_phone_label, 2, 2)
        layout.addWidget(self.register_phone_edit, 2, 3)
        layout.addWidget(register_password_label, 3, 2)
        layout.addWidget(self.register_password_edit, 3, 3)
        layout.addWidget(register_button, 4, 3)
        
        # Background image
        image_path = os.path.join(
            self.config.get_path('image_folder'), 
            'human-skeletal.jpg'
        )
        if os.path.exists(image_path):
            image_label = QLabel()
            pixmap = QPixmap(image_path).scaled(600, 600, Qt.KeepAspectRatio)
            image_label.setPixmap(pixmap)
            layout.addWidget(image_label, 0, 4, 5, 1)
            
        self.setLayout(layout)
        
    def create_label(self, text: str) -> QLabel:
        """Create a styled label"""
        label = QLabel(text)
        label.setStyleSheet("font: 25px 'Times New Roman'; color: #f9f9f9;")
        return label
        
    def create_line_edit(self) -> QLineEdit:
        """Create a styled line edit"""
        line_edit = QLineEdit()
        line_edit.setStyleSheet(
            "font-size: 25px; padding: 8px; "
            "font: bold 25px 'Times New Roman'; "
            "border: 2px solid #ccc; border-radius: 10px; "
            "background-color: #444; color: #f9f9f9;"
        )
        line_edit.setFixedWidth(230)
        return line_edit
        
    def create_button(self, text: str, handler) -> QPushButton:
        """Create a styled button"""
        button = QPushButton(text)
        button.setStyleSheet(
            "QPushButton {"
            "   background-color: #858585; "
            "   color: white; "
            "   padding: 15px 30px; "
            "   border: none; "
            "   border-radius: 10px; "
            "   font-family: 'Times New Roman'; "
            "   font-weight: bold; "
            "   font-size: 25px; "
            "}"
            "QPushButton:hover {"
            "   background-color: #606060; "
            "}"
        )
        button.clicked.connect(handler)
        return button
        
    def login(self) -> None:
        """Handle login attempt"""
        username = self.login_username_edit.text()
        password = self.login_password_edit.text()
        
        if not username or not password:
            QMessageBox.warning(
                self, 
                'Login Failed', 
                'Please enter both username and password.'
            )
            return
            
        conn_str = (
            f'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};'
            f'DBQ={self.config.get_path("database_path")};'
        )
        
        with DatabaseManager(conn_str) as db:
            if db.authenticate_user(username, password):
                self.accept()
            else:
                QMessageBox.warning(
                    self, 
                    'Login Failed', 
                    'Invalid username or password.'
                )
                
    def register(self) -> None:
        """Handle registration attempt"""
        username = self.register_username_edit.text()
        password = self.register_password_edit.text()
        email = self.register_email_edit.text()
        phone = self.register_phone_edit.text()
        
        if not all([username, password, email, phone]):
            QMessageBox.warning(
                self, 
                'Registration Failed', 
                'Please fill all fields.'
            )
            return
            
        conn_str = (
            f'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};'
            f'DBQ={self.config.get_path("database_path")};'
        )
        
        with DatabaseManager(conn_str) as db:
            if db.register_user(username, password, email, phone):
                QMessageBox.information(
                    self, 
                    'Registration Successful', 
                    'User registered successfully.'
                )
                self.clear_registration_fields()
            else:
                QMessageBox.warning(
                    self, 
                    'Registration Failed', 
                    'Username already exists.'
                )
                
    def clear_registration_fields(self) -> None:
        """Clear registration form fields"""
        self.register_username_edit.clear()
        self.register_password_edit.clear()
        self.register_email_edit.clear()
        self.register_phone_edit.clear()


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        self.setWindowTitle('Object Detection System')
        
        self.init_ui()
        
    def init_ui(self) -> None:
        """Initialize UI components"""
        # Set window size from config
        width = int(self.config.get_ui_setting('window_width', 1900))
        height = int(self.config.get_ui_setting('window_height', 900))
        self.setGeometry(100, 100, width, height)
        
        # Set background
        bg_image_path = os.path.join(
            self.config.get_path('image_folder'), 
            'thumbnail_1.jpg'
        )
        if os.path.exists(bg_image_path):
            pixmap = QPixmap(bg_image_path).scaled(width, height)
            palette = self.palette()
            palette.setBrush(QPalette.Background, QBrush(pixmap))
            self.setPalette(palette)
            
        # Create buttons
        button_height = int(self.config.get_ui_setting('button_height', 70))
        font_size = int(self.config.get_ui_setting('font_size', 14))
        padding = int(self.config.get_ui_setting('button_padding', 10))
        
        buttons = [
            ("Image Analysis", (50, 9, 235, button_height), 
             self.open_image_results, font_size, padding),
            ("Video Analysis", (300, 9, 200, button_height), 
             self.open_video_results, font_size, padding),
            ("Detections Database", (515, 9, 225, button_height), 
             self.open_database_form, font_size, padding),
            ("Real-Time Database", (750, 9, 240, button_height), 
             self.open_realtime_database, font_size, padding),
            ("Analyzed Images", (1000, 9, 380, button_height), 
             self.open_analyzed_images, font_size, padding),
            ("Object Structure", (1390, 9, 230, button_height), 
             self.open_structure_form, font_size, padding),
            ("Model Overview", (1630, 9, 250, button_height), 
             self.open_model_overview, font_size, padding)
        ]
        
        for text, geometry, handler, fs, pad in buttons:
            button = self.create_button(text, geometry, fs, pad)
            button.clicked.connect(handler)
            
    def create_button(
        self, 
        text: str, 
        geometry: Tuple[int, int, int, int],
        font_size: int,
        padding: int
    ) -> QPushButton:
        """Create a styled button"""
        button = QPushButton(text, self)
        button.setGeometry(*geometry)
        button.setStyleSheet(BUTTON_STYLE.format(
            bg_color="#B0BBC1",
            font_size=font_size,
            padding=padding
        ))
        button.setFont(QFont("Times New Roman", font_size, QFont.Bold))
        return button
        
    def open_image_results(self) -> None:
        """Open image analysis dialog"""
        options = QFileDialog.Options()
        filenames, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select Image(s)", 
            "", 
            "Images (*.jpg *.png)", 
            options=options
        )
        
        if filenames:
            self.results_window = ResultsWindow(
                self.config,
                image_path=filenames[0],
                image_size=(800, 600)
            )
            self.results_window.show()
            
    def open_video_results(self) -> None:
        """Open video analysis dialog"""
        options = QFileDialog.Options()
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video",
            "",
            "Videos (*.mp4 *.avi)",
            options=options
        )

        if filenames:
            self.video_analyzer = VideoAnalyzer(self.config)
            self.video_analyzer.video_analysis_finished.connect(
                self.open_realtime_database
            )
            self.video_analyzer.detect_and_save_objects(filenames[0])

    def open_database_form(self) -> None:
        """Open detections database form"""
        self.database_form = DatabaseForm(self.config)
        self.database_form.show()

    def open_realtime_database(self) -> None:
        """Open real-time database form"""
        self.realtime_db_form = RealTimeDatabaseForm(self.config)
        self.realtime_db_form.show()

    def open_analyzed_images(self) -> None:
        """Open analyzed images form"""
        self.analyzed_images_form = ImageDisplayForm(self.config)
        self.analyzed_images_form.show()

    def open_structure_form(self) -> None:
        """Open object structure form"""
        self.structure_form = StructureOfObjectsForm(self.config)
        self.structure_form.show()

    def open_model_overview(self) -> None:
        """Open model overview form"""
        self.model_overview = ModelOverviewForm(self.config)
        self.model_overview.show()


def main():
    """Main application entry point"""
    # Initialize application
    app = QApplication(sys.argv)
    
    # Load configuration
    config = ConfigManager()
    
    # Show login dialog
    login_dialog = LoginDialog(config)
    if login_dialog.exec_() == QDialog.Accepted:
        # If login successful, show main window
        main_window = MainWindow(config)
        main_window.show()
        sys.exit(app.exec_())


if __name__ == '__main__':
    main()