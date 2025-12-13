import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QMainWindow, QPushButton, QFileDialog,
    QTableWidget, QTableWidgetItem, QLabel, QGridLayout, QScrollArea, QMessageBox, QDialog,
    QLineEdit, QHBoxLayout, QHeaderView
)
from PyQt5.QtGui import QPixmap, QFont, QPalette, QBrush
from PyQt5.QtCore import Qt

# Import headless core functionality
from core.config import *
from core.model import SkeletonModel
from core.database import DatabaseManager
from core.preprocessing import ImageManager
from core.evaluation import ModelEvaluator

class ImageDisplayMixin:
    """GUI mixin for displaying images"""
    
    def display_images_in_grid(self, image_paths, scroll_area_size, image_size, columns=3):
        """Display images in grid layout - GUI only"""
        # Clear existing layout
        for i in reversed(range(self.layout().count())):
            widget = self.layout().itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedSize(*scroll_area_size)
        self.layout().addWidget(scroll_area)

        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)
        scroll_widget.setStyleSheet("background-color: black;")

        grid_layout = QGridLayout(scroll_widget)
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(20)

        # Display images
        for index, image_path in enumerate(image_paths):
            row = index // columns
            col = index % columns
            image_label = QLabel()
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(*image_size)
                image_label.setPixmap(pixmap)
                grid_layout.addWidget(image_label, row, col)

        self.adjustSize()

class ModelEvaluationForm(QWidget, ImageDisplayMixin):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Обзор силы модели')
        self.initUI()
        
    def initUI(self):
        self.setGeometry(50, 50, 1500, 1200)

        # Setup background using core config
        original_pixmap = QPixmap(BACKGROUND_IMAGE_MAIN)
        pixmap = original_pixmap.scaled(1500, 800)
        brush = QBrush(pixmap)
        palette = self.palette()
        palette.setBrush(QPalette.Background, brush)
        self.setPalette(palette)
        self.setFixedSize(pixmap.size())
        
        layout = QVBoxLayout(self)
        
        # Get available evaluations from core
        available_evals = ModelEvaluator.get_available_evaluations()
        
        # Create buttons for available evaluations
        button_configs = {
            'confusion_matrix': ('Матрица путаницы', 30, 220),
            'f1_curve': ('Кривая F1', 260, 150),
            'labels': ('Метки', 420, 150),
            'labels_correlogram': ('Коррелограмма меток', 585, 200),
            'p_curve': ('Кривая P', 800, 150),
            'pr_curve': ('Кривая PR', 960, 150),
            'r_curve': ('Кривая R', 1120, 150),
            'results': ('Результаты', 1280, 150)
        }
        
        for eval_type, (text, x, width) in button_configs.items():
            if eval_type in available_evals:
                button = QPushButton(text, self)
                button.setGeometry(x, 6, width, 60)
                button.setStyleSheet(
                    "QPushButton {"
                    "   background-color: #B0BBC1; "
                    "   color: black; "
                    "   border: 2px solid gray; border-radius: 10px;"
                    "}"
                    "QPushButton:hover { background-color: lightgray; }"
                )
                button.setFont(QFont("Times New Roman", 14, QFont.Bold))
                button.clicked.connect(lambda checked, et=eval_type: self.open_evaluation(et))
        
        self.setLayout(layout)
    
    def open_evaluation(self, evaluation_type):
        """Open evaluation using core logic"""
        image_paths = ModelEvaluator.get_evaluation_images(evaluation_type)
        if image_paths:
            self.display_images_in_grid(image_paths, (1400, 600), (1200, 600))
        else:
            QMessageBox.warning(self, 'Ошибка', f'Нет изображений для {evaluation_type}')

class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Вход или регистрация')
        self.setGeometry(200, 200, 1400, 600)
        self.init_ui()
    
    def init_ui(self):
        # ... (your existing login UI code)
        # Use DatabaseManager from core for authentication
        pass
    
    def login(self):
        username = self.login_username_edit.text()
        password = self.login_password_edit.text()
        
        # Use core database manager for headless authentication
        if DatabaseManager.validate_user(username, password):
            self.accept()
        else:
            QMessageBox.warning(self, 'Ошибка', 'Неверные данные для входа')

# ... (Include other GUI classes: StructureForm, MainWindow, etc.)
# They all import from core for headless operations

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.model = SkeletonModel()  # Core model
        self.initUI()
    
    def analyze_image(self):
        """Use core model for analysis"""
        filenames, _ = QFileDialog.getOpenFileNames(self, "Выберите изображение", "", "Images (*.jpg *.png)")
        if filenames:
            results = self.model.analyze_image(filenames[0])
            if results and results['success']:
                self.show_results(results)
            else:
                QMessageBox.warning(self, 'Ошибка', 'Не удалось проанализировать изображение')
    
    def analyze_video(self):
        """Use core model for video analysis"""
        filenames, _ = QFileDialog.getOpenFileNames(self, "Выберите видео", "", "Videos (*.mp4 *.avi)")
        if filenames:
            # This would use headless video analysis
            QMessageBox.information(self, 'Информация', 'Видео анализ запущен в фоновом режиме')

def run_gui():
    """Run the GUI application"""
    app = QApplication(sys.argv)
    login_window = LoginWindow()
    
    if login_window.exec_() == QDialog.Accepted:
        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec_())