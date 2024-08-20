# Часть 1:

import os
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import pyodbc
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/L/yolov5/runs/train/exp2/weights/last.pt', force_reload=True)
image_folder = 'D:/My_Diploma/System2/skeletal2'
output_folder = 'D:/My_Diploma/System2/skeletal3'
os.makedirs(output_folder, exist_ok=True)
db_path = r'D:\My_Diploma\System2\Homam_SK.accdb'
conn_str = f'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db_path};'
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()



image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    
    img_path = os.path.join(image_folder, image_file)
    img = cv2.imread(img_path)
    
    
    results = model(img)
    
   
    print(f"Results for image: {image_file}")
    print(results)
    results.print()
    
    
    plt.imshow(np.squeeze(results.render()))
    plt.title(f'Detection for {image_file}')
    plt.show()
    
    
    output_path = os.path.join(output_folder, f'detected_{image_file}')
    cv2.imwrite(output_path, np.squeeze(results.render()))
    results_output = results.__str__()

    
    detected_image_link = '=HYPERLINK("' + os.path.abspath(output_path).replace("\\", "/") + '", "Click to open detected image")'

   
    original_image_link = f'=HYPERLINK("{img_path}", "Click to open original image")'

    
    cursor.execute('''INSERT INTO Detections (Image_name, Results, Detected_image_link, Original_image_link)
                  VALUES (?, ?, ?, ?)''', (image_file, results_output, detected_image_link, img_path))


conn.commit()


conn.close()

# Часть 2: 

import sys
import cv2
import numpy as np
import torch
import os
import pyodbc
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QMainWindow, QPushButton, QFileDialog,
    QTableWidget, QTableWidgetItem, QLabel, QGridLayout, QScrollArea, QMessageBox, QDialog,
    QLineEdit
)
from PyQt5.QtGui import QPixmap, QFont, QIcon, QImage, QPalette, QBrush
from PyQt5.QtCore import QObject, pyqtSignal, Qt
import pandas as pd
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtWidgets import QDialog, QGridLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout, QMes-sageBox, QVBoxLayout, QHeaderView
from PyQt5.QtGui import QImage, QPixmap


class Overviewofmodelform(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Обзор силы модели')
        self.initUI()
        
    def initUI(self):

        self.setGeometry(50, 50, 1500, 1200)

      
        bg_image_path = "D:/My_Diploma/is-machine-learning-hard-a-guide-to-getting-started-scaled-1-scaled.jpeg"  
        original_pixmap = QPixmap(bg_image_path)
     
        pixmap = original_pixmap.scaled(1500, 800)

   
        brush = QBrush(pixmap)

      
        palette = self.palette()
        palette.setBrush(QPalette.Background, brush)

     
        self.setPalette(palette)

    
        self.setFixedSize(pixmap.size())
        
        layout = QVBoxLayout(self)
        

        self.confusion_matrix_button = QPushButton('Матрица путаницы', self)
        self.confusion_matrix_button.setGeometry(30, 6, 220, 60)
        self.confusion_matrix_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.confusion_matrix_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.confusion_matrix_button.clicked.connect(self.open_confusion_matrix_image)
        #.....
        self.F1_curve_button = QPushButton('Кривая F1', self)
        self.F1_curve_button.setGeometry(260, 6, 150, 60)
        self.F1_curve_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.F1_curve_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.F1_curve_button.clicked.connect(self.open_F1_curve_image)
        #.....
        self.labels_button = QPushButton('Метки', self)
        self.labels_button.setGeometry(420, 6, 150, 60)
        self.labels_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.labels_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.labels_button.clicked.connect(self.open_labels_image)
         #.....
        self.labels_correlogram_button = QPushButton('Коррелограмма \n меток', self)
        self.labels_correlogram_button.setGeometry(585, 6, 200, 60)
        self.labels_correlogram_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.labels_correlogram_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.labels_correlogram_button.clicked.connect(self.open_labels_correlogram_image)
        #.....
        self.P_curve_button = QPushButton('Кривая P', self)
        self.P_curve_button.setGeometry(800, 6, 150, 60)
        self.P_curve_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.P_curve_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.P_curve_button.clicked.connect(self.open_P_curve_image)
        #.....
        self.PR_curve_button = QPushButton('Кривая PR', self)
        self.PR_curve_button.setGeometry(960, 6, 150, 60)
        self.PR_curve_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.PR_curve_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.PR_curve_button.clicked.connect(self.open_PR_curve_image)
        #.....
        self.R_curve_button = QPushButton('Кривая R', self)
        self.R_curve_button.setGeometry(1120, 6, 150, 60)
        self.R_curve_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.R_curve_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.R_curve_button.clicked.connect(self.open_R_curve_image)
        #.....
        self.results_curve_button = QPushButton('Результаты', self)
        self.results_curve_button.setGeometry(1280, 6, 150, 60)
        self.results_curve_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.results_curve_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.results_curve_button.clicked.connect(self.open_results_image)
        
        self.setLayout(layout)

        
    def open_confusion_matrix_image(self):
        folder_path = "D:/My_Diploma/System2/model/confusion_matrix"
        self.open_folder_images(folder_path)

    def open_F1_curve_image(self):
        folder_path = "D:/My_Diploma/System2/model/F1_curve"
        self.open_folder_images(folder_path)
        
    def open_labels_image(self):
        folder_path = "D:/My_Diploma/System2/model/labels"
        self.open_folder_images(folder_path)
        
    def open_labels_correlogram_image(self):
        folder_path = "D:/My_Diploma/System2/model/labels_correlogram"
        self.open_folder_images(folder_path)
    
    def open_P_curve_image(self):
        folder_path = "D:/My_Diploma/System2/model/P_curve"
        self.open_folder_images(folder_path)
    
    def open_PR_curve_image(self):
        folder_path = "D:/My_Diploma/System2/model/PR_curve"
        self.open_folder_images(folder_path)
    
    def open_R_curve_image(self):
        folder_path = "D:/My_Diploma/System2/model/R_curve"
        self.open_folder_images(folder_path)
        
    def open_results_image(self):
        folder_path = "D:/My_Diploma/System2/model/results"
        self.open_folder_images(folder_path)

    def open_folder_images(self, folder_path):
      
        for i in reversed(range(self.layout().count())):
            self.layout().itemAt(i).widget().setParent(None)

    
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  
        scroll_area.setFixedSize(1400, 600)  
        scroll_area.move(200, 25)  

        self.layout().addWidget(scroll_area)

        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)
        scroll_widget.setStyleSheet("background-color: black;")

        grid_layout = QGridLayout(scroll_widget)
        grid_layout.setHorizontalSpacing(20) 
        grid_layout.setVerticalSpacing(20)    
        grid_layout.setColumnStretch(1, 1)     

    
        images = os.listdir(folder_path)
        num_columns = 3  
        for index, image_name in enumerate(images):
            row = index // num_columns
            col = index % num_columns
            image_path = os.path.join(folder_path, image_name)
            image_label = QLabel()
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(1200, 600)  
            image_label.setPixmap(pixmap)
            grid_layout.addWidget(image_label, row, col)

        self.adjustSize()  
        
        
class StructureOfObjectsForm(QWidget): 
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Структура объектов')

        self.initUI()
        
    def initUI(self):
         
        self.setGeometry(100, 100, 800, 700)

      
        bg_image_path = "D:/My_Diploma/System2/SFD.png"  
        original_pixmap = QPixmap(bg_image_path)
    
        pixmap = original_pixmap.scaled(800, 700)

       
        brush = QBrush(pixmap)

      
        palette = self.palette()
        palette.setBrush(QPalette.Background, brush)

       
        self.setPalette(palette)

     
        self.setFixedSize(pixmap.size())
        
        layout = QVBoxLayout(self)
        

        self.bayraktar_button = QPushButton('Человеческий скелет', self)
        self.bayraktar_button.setGeometry(130, 6, 290, 60)
        self.bayraktar_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #9F8B79; "
            "   color: black; margin-bottom: 0px; margin-top: 0px; "
            "   border: 2px solid; "
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: #F4E6D9; "
            "}"
        )
        self.bayraktar_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.bayraktar_button.clicked.connect(self.open_Human_Skeletal_images)

      
        self.phantom_button = QPushButton('Скелет гориллы', self)
        self.phantom_button.setGeometry(450, 6, 245, 60)
        self.phantom_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #9F8B79; "
            "   color: black; "
            "   border: 2px solid; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: #F4E6D9; "
            "}"
        )
        self.phantom_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.phantom_button.clicked.connect(self.open_Gorilla_Skeletal_images)
        
        self.setLayout(layout)

        
    def open_Human_Skeletal_images(self):
        folder_path = "D:/My_Diploma/System2/Human_Skeletal"
        self.open_folder_images(folder_path)

    def open_Gorilla_Skeletal_images(self):
        folder_path = "D:/My_Diploma/System2/Gorilla_Skeletal"
        self.open_folder_images(folder_path)

    def open_folder_images(self, folder_path):
       
        for i in reversed(range(self.layout().count())):
            self.layout().itemAt(i).widget().setParent(None)

      
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  
        
        scroll_area.setFixedSize(800, 500)  
        
        self.layout().addWidget(scroll_area)

        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)
        scroll_widget.setStyleSheet("background-color: black;")

        grid_layout = QGridLayout(scroll_widget)
        grid_layout.setHorizontalSpacing(20) 
        grid_layout.setVerticalSpacing(20)   
        grid_layout.setColumnStretch(1, 1)     

      
        images = os.listdir(folder_path)
        num_columns = 3  
        for index, image_name in enumerate(images):
            row = index // num_columns
            col = index % num_columns
            image_path = os.path.join(folder_path, image_name)
            image_label = QLabel()
            image = QImage(image_path)
            pixmap = QPixmap.fromImage(image)
            pixmap = pixmap.scaled(600, 600)  
            image_label.setPixmap(pixmap)
            grid_layout.addWidget(image_label, row, col)

        self.adjustSize()  
        
        
class ResultsWindow(QMainWindow):
    def __init__(self, image_path=None, folder_path=None, video_path=None, model_path=None, im-age_size=(400, 400)):
        super().__init__()
        self.setWindowTitle('Результаты анализа')
        self.image_size = image_size

        layout = QVBoxLayout()

     
        if model_path:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        else:
            self.model = None

       
        if image_path:
            self.analyze_single_image(image_path, layout)

      
        if folder_path:
            self.analyze_images_in_folder(folder_path, layout)

      
        if video_path:
            self.analyze_video(video_path, layout)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def analyze_single_image(self, image_path, layout):
       
        results = self.model(image_path) if self.model else None

      
        self.display_results(results, layout)

    def analyze_images_in_folder(self, folder_path, layout):
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(folder_path, filename)
                results = self.model(image_path) if self.model else None
                self.display_results(results, layout)

    def analyze_video(self, video_path, layout):
       
        detect_and_save_objects(video_path)
        
    def display_results(self, results, layout):
        if results:
           
            text_results = str(results)
            text_label = QLabel(text_results, self)
            layout.addWidget(text_label)

           
            img_array = np.squeeze(results.render())
            img_array_resized = cv2.resize(img_array, self.image_size)
            height, width, channel = img_array_resized.shape
            bytesPerLine = 3 * width
            qImg = QImage(img_array_resized.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            img_label = QLabel(self)
            img_label.setPixmap(pixmap)
            layout.addWidget(img_label)

class VideoAnalyzer(QObject):
    video_analysis_finished = pyqtSignal()

    def detect_and_save_objects(self, input_path, new_width=640, new_height=480):
     
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/L/yolov5/runs/train/exp2/weights/last.pt', force_reload=True)

      
        cap = cv2.VideoCapture(input_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

          
            frame = cv2.resize(frame, (new_width, new_height))

           
            results = model(frame)

            
            detected_objects = results.xyxy[0]

            
            for obj in detected_objects:
                x1, y1, x2, y2, confidence, class_id = obj
                object_image = frame[int(y1):int(y2), int(x1):int(x2)]

          
                object_filename = f"D:/My_Diploma/System2/Our_DataBase/object_{int(x1)}_{int(y1)}.jpg"
                cv2.imwrite(object_filename, object_image)

         
            cv2.imshow('Видеоанализ', np.squeeze(results.render()))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

       
        cap.release()

     
        cv2.destroyAllWindows()
        
       
        self.video_analysis_finished.emit()

class ImageDisplayForm(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('База данных проанализированных изображений')
        self.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout()

       
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  
        layout.addWidget(scroll_area)

        
        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)

       
        scroll_widget.setStyleSheet("background-color: black;")

      
        grid_layout = QGridLayout(scroll_widget)

       
        folder_path = "D:/My_Diploma/System2/On_DataBase/"
        images = os.listdir(folder_path)
        row, col = 0, 0
        for image_name in images:
            image_path = os.path.join(folder_path, image_name)
            image_label = QLabel()
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(400, 400)  
            image_label.setPixmap(pixmap)
            grid_layout.addWidget(image_label, row, col)
            col += 1
            if col == 3:  
                col = 0
                row += 1

        self.setLayout(layout)

        
class ImagesChildForm(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Базе данных реального времени')
        self.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout()

      
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  
        layout.addWidget(scroll_area)

       
        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)

       
        scroll_widget.setStyleSheet("background-color: black;")

       
        grid_layout = QGridLayout(scroll_widget)
        grid_layout.setHorizontalSpacing(20)  
        grid_layout.setVerticalSpacing(20)    
        grid_layout.setColumnStretch(1, 1)    


      
        folder_path = "D:/My_Diploma/System2/Our_DataBase"
        images = os.listdir(folder_path)
        num_columns = 3  
        for index, image_name in enumerate(images):
            row = index // num_columns
            col = index % num_columns
            image_path = os.path.join(folder_path, image_name)
            image_label = QLabel()
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(300, 300)  
            image_label.setPixmap(pixmap)
            grid_layout.addWidget(image_label, row, col)

        self.setLayout(layout)
        
    def display_images(self, layout):
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(self.folder_path, filename)
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    pixmap = pixmap.scaledToWidth(600)  
                    label = QLabel()
                    label.setPixmap(pixmap)
                    layout.addWidget(label)
                    
    def open_image_display_form(self):
       
        self.image_display_form = ImagesChildForm(self.folder_path)
        self.image_display_form.show()
        
        
class AccessDatabaseForm(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('База данных обнаружений')
        self.setGeometry(200, 200, 800, 600)
        self.setStyleSheet("background-color: #f0f0f0;")  

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()

      
        self.table_widget = QTableWidget()
        self.table_widget.setStyleSheet("background-color: #B0BBC1;")  
        layout.addWidget(self.table_widget)

     
        self.load_access_data()

        self.central_widget.setLayout(layout)

    def load_access_data(self):
     
        db_path = r'D:\My_Diploma\System2\Homam_SK.accdb'
        conn_str = f'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db_path};'
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        
        cursor.execute("SELECT * FROM Detections")
        rows = cursor.fetchall()

       
        columns = [column[0] for column in cursor.description]

   
        self.table_widget.setRowCount(len(rows))
        self.table_widget.setColumnCount(len(columns))
        self.table_widget.setHorizontalHeaderLabels(columns)
        self.table_widget.horizontalHeader().setStyleSheet("background-color: #d9d9d9;")  
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  

        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                cell_value = str(value)
                item = QTableWidgetItem(cell_value)
                self.table_widget.setItem(i, j, item)

       
        conn.close()
        
        
        
class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Вход или регистрация')
        self.setGeometry(200, 200, 1400, 600) 

        layout = QGridLayout()  

      
        self.setStyleSheet("background-color: #1d1d1d; color: #f9f9f9;")

      
        login_username_label = QLabel('Имя пользователя:')
        login_password_label = QLabel('Пароль:')
        login_username_edit = QLineEdit()
        login_password_edit = QLineEdit()
        login_password_edit.setEchoMode(QLineEdit.Password)
        login_button = QPushButton('Вход')

      
        labels = [login_username_label, login_password_label]
        for label in labels:
            label_style = "font: 25px 'Times New Roman'; color: #f9f9f9; margin-bottom: 5px;"
            label.setStyleSheet(label_style)
            
        login_username_edit.setStyleSheet("font-size: 25px; padding: 8px; font: bold 25px 'Times New Roman'; border: 2px solid #ccc; border-radius: 10px; background-color: #444; color: #f9f9f9;")
        login_password_edit.setStyleSheet("font-size: 25px; padding: 8px; font: bold 25px 'Times New Roman'; border: 2px solid #ccc; border-radius: 10px; background-color: #444; color: #f9f9f9;")

        login_username_edit.setFixedWidth(230)
        login_password_edit.setFixedWidth(230)
    
        login_button.setStyleSheet(
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

        layout.addWidget(login_username_label, 0, 0)
        layout.addWidget(login_username_edit, 0, 1)
        layout.addWidget(login_password_label, 1, 0)
        layout.addWidget(login_password_edit, 1, 1)
        layout.addWidget(login_button, 2, 1)

        registration_layout = QHBoxLayout()
        
 
        register_username_label = QLabel('Имя пользователя:')
        register_email_label = QLabel('Электронная почта:')
        register_phone_label = QLabel('Номер телефона:')

        register_password_label = QLabel('Пароль:')

        register_username_edit = QLineEdit()
        register_email_edit = QLineEdit()
        register_phone_edit = QLineEdit()
        register_password_edit = QLineEdit()
        register_password_edit.setEchoMode(QLineEdit.Password)
        register_button = QPushButton('Регистрация')


        registration_layout.addWidget(register_username_label)
        registration_layout.addWidget(register_username_edit)
        registration_layout.addWidget(register_email_label)
        registration_layout.addWidget(register_email_edit)
        registration_layout.addWidget(register_phone_label)
        registration_layout.addWidget(register_phone_edit)
        registration_layout.addWidget(register_password_label)
        registration_layout.addWidget(register_password_edit)
        registration_layout.addWidget(register_button)
        
        
        register_labels = [register_username_label, register_email_label, register_phone_label, regis-ter_password_label]
        for label in register_labels:
            label_style = "font: 25px 'Times New Roman'; color: #f9f9f9; margin-bottom: 5px;"
            label.setStyleSheet(label_style)

        register_edit_style = "font: bold 25px 'Times New Roman'; padding: 8px; border: 2px solid #ccc; border-radius: 10px; background-color: #444; color: #f9f9f9;"
        register_username_edit.setStyleSheet(register_edit_style)
        register_email_edit.setStyleSheet(register_edit_style)
        register_phone_edit.setStyleSheet(register_edit_style)
        register_password_edit.setStyleSheet(register_edit_style)

        register_username_edit.setFixedWidth(300)
        register_email_edit.setFixedWidth(300)
        register_phone_edit.setFixedWidth(300)
        register_password_edit.setFixedWidth(300)
        
        register_button.setStyleSheet(
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

        layout.addWidget(register_username_label, 0, 2)
        layout.addWidget(register_username_edit, 0, 3)
        layout.addWidget(register_email_label, 1, 2)
        layout.addWidget(register_email_edit, 1, 3)
        layout.addWidget(register_phone_label, 2, 2)
        layout.addWidget(register_phone_edit, 2, 3)
        layout.addWidget(register_password_label, 3, 2)
        layout.addWidget(register_password_edit, 3, 3)
        layout.addWidget(register_button, 4, 3)

    
        pixmap = QPixmap("D:/My_Diploma/System2/human-skeletal.jpg")  
        pixmap = pixmap.scaled(600, 600)  
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        layout.addWidget(image_label, 0, 4, 5, 1)  

        self.setLayout(layout)

        
        self.username_edit = login_username_edit
        self.password_edit = login_password_edit
        self.register_username_edit = register_username_edit
        self.register_password_edit = register_password_edit
        self.register_email_edit = register_email_edit
        self.register_phone_edit = register_phone_edit

        
        login_button.clicked.connect(self.login)
        register_button.clicked.connect(self.register)

    def login(self):
        
        username = self.username_edit.text()
        password = self.password_edit.text()

        if not username or not password:  
            QMessageBox.warning(self, 'Вход не выполнен', 'Пожалуйста, введите как имя пользователя, так и пароль.')
            return

        
        connection_string = r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=D:\My_Diploma\System2\Homam_SK.accdb;'
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        try:
            
            cursor.execute("SELECT Username, Password FROM Users WHERE Username=? AND Password=?", (username, password))
            user = cursor.fetchone()
            if user:
                self.accept()  
            else:
                QMessageBox.warning(self, 'Вход не выполнен', 'неправильное имя пользователя или пароль')
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f'Произошла ошибка: {str(e)}')
        finally:
           
            cursor.close()
            conn.close()

    def register(self):
      
        new_username = self.register_username_edit.text()
        new_password = self.register_password_edit.text()
        new_email = self.register_email_edit.text()
        new_phone = self.register_phone_edit.text()

        if not all([new_username, new_password, new_email, new_phone]):  
            QMessageBox.warning(self, 'Регистрация не удалас', 'Пожалуйста заполните все поля.')
            return

      
        connection_string = r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=D:\My_Diploma\System2\Homam_SK.accdb;'
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        try:
           
            cursor.execute("SELECT Username FROM Users WHERE Username=?", (new_username,))
            existing_user = cursor.fetchone()
            if existing_user:
                QMessageBox.warning(self, 'Регистрация не удалась', 'Имя пользователя уже занято')
            else:
               
                cursor.execute("INSERT INTO Users (Username, Password, Email, Phone_Number) VALUES (?, ?, ?, ?)", (new_username, new_password, new_email, new_phone))
                conn.commit()
                QMessageBox.information(self, 'Регистрация прошла успешно', 'Новый пользователь зарегистри-рован успешно')
             
                self.register_username_edit.clear()
                self.register_password_edit.clear()
                self.register_email_edit.clear()
                self.register_phone_edit.clear()
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f'Произошла ошибка: {str(e)}')
        finally:
           
            cursor.close()
            conn.close()
            
class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
      
        self.initUI()
        

    def initUI(self):
        
        self.setGeometry(100, 100, 1900, 900)

        
        button_width = int((self.width() - 100) / 7)
        
        
        button_height = 70

        
        bg_image_path = "D:/My_Diploma/System2/thumbnail_1.jpg"  # Specify the path to your background im-age
        original_pixmap = QPixmap(bg_image_path)
 
        pixmap = original_pixmap.scaled(1900, 900)


        brush = QBrush(pixmap)


        palette = self.palette()
        palette.setBrush(QPalette.Background, brush)


        self.setPalette(palette)

 
        self.setFixedSize(pixmap.size())


        self.image_button = QPushButton('Анализ изображений', self)
        self.image_button.setGeometry(50, 9, 235, button_height)
        self.image_button.setStyleSheet("background-color: white; color: black;")
        self.image_button.setFont(QFont("Times New Roman", 14, QFont.Bold))  
        self.image_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.image_button.clicked.connect(self.open_image_results_window)
        
        self.video_button = QPushButton('Видео анализ', self)
        self.video_button.setGeometry(300, 9, 200, button_height)
        self.video_button.setStyleSheet("background-color: white; color: black;")
        self.video_button.setFont(QFont("Times New Roman", 14, QFont.Bold))  
        self.video_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.video_button.clicked.connect(self.open_video_results_window)
        
        self.database_button = QPushButton('База данных\nобнаружений', self)
        self.database_button.setStyleSheet("QPushButton { text-align: left; }")
        self.database_button.setGeometry(515, 9, 225, button_height)
        self.database_button.setStyleSheet("background-color: white; color: black;")
        self.database_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.database_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.database_button.clicked.connect(self.open_database_form)

        
        self.child_form_button = QPushButton('База данных\nреального времени', self)
        self.child_form_button.setStyleSheet("QPushButton { text-align: left; }")

        self.child_form_button.setGeometry(750, 9, 240, button_height)
        self.child_form_button.setStyleSheet("background-color: white; color: black;")
        self.child_form_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.child_form_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.child_form_button.clicked.connect(self.open_child_form)
        

        self.image_display_button = QPushButton('База данных\nпроанализированных изображений', self)
        self.image_display_button.setStyleSheet("QPushButton { text-align: left; }")

        self.image_display_button.setGeometry(1000, 9, 380, button_height)
        self.image_display_button.setStyleSheet("background-color: white; color: black;")  
        self.image_display_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.image_display_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.image_display_button.clicked.connect(self.open_image_display_form)
        
        
        self.structure_button = QPushButton('Структура объектов', self)
        self.structure_button.setGeometry(1390, 9, 230, button_height)
        self.structure_button.setStyleSheet("background-color: white; color: black;")
        self.structure_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.structure_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.structure_button.clicked.connect(self.open_structure_form)
        
        
        self.model_button = QPushButton('Обзор силы модели', self)
        self.model_button.setGeometry(1630, 9, 250, button_height)
        self.model_button.setStyleSheet("background-color: white; color: black;")
        self.model_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.model_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.model_button.clicked.connect(self.open_model_form)

        
        self.image_display_label = QLabel(self)
        self.image_display_label.setGeometry(100, 100, 1800, 600)  
    
       
        
        self.setWindowTitle('Проект обнаружения дронов')
        self.show()

    def open_model_form(self):
        self.Overview_of_model_form = Overviewofmodelform()
        self.Overview_of_model_form.show()
        
    def open_image_results_window(self):
        
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Images (*.jpg *.png)")
        file_dialog.setViewMode(QFileDialog.Detail)
        filenames, _ = file_dialog.getOpenFileNames(self, "Выберите изображение или папку", "", "Images (*.jpg *.png);;All Files (*)", options=options)

        if filenames:
            
            model_path = 'C:/Users/L/yolov5/runs/train/exp2/weights/last.pt'
            image_size = (800, 600)

            
            self.results_window = ResultsWindow(image_path=filenames[0], folder_path=None, video_path=None, model_path=model_path, image_size=image_size)
            self.results_window.show()

    def open_video_results_window(self):

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Videos (*.mp4 *.avi)")
        file_dialog.setViewMode(QFileDialog.Detail)
        filenames, _ = file_dialog.getOpenFileNames(self, "Выберите видео", "", "Videos (*.mp4 *.avi)", op-tions=options)

        if filenames:

            self.video_analyzer = VideoAnalyzer()
            self.video_analyzer.video_analysis_finished.connect(self.open_child_form)
            self.video_analyzer.detect_and_save_objects(filenames[0])

    def open_child_form(self):

        self.child_form = ImagesChildForm()
        self.child_form.show()
        

    def open_database_form(self):
        self.database_form = AccessDatabaseForm()
        self.database_form.show()
        

        
    def open_image_display_form(self):
        self.image_display_form = ImageDisplayForm()
        self.image_display_form.show()
        
    def open_structure_form(self):
        self.structure_of_objects_form = StructureOfObjectsForm()
        self.structure_of_objects_form.show()
        
    def show_structure_of_objects(self):
        self.structure_of_objects_form = StructureOfObjectsForm()
        self.structure_of_objects_form.show()

    def show_image_display(self):
        self.image_display_form = ImageDisplayForm()
        self.image_display_form.show()

    def display_images(self, folder_path):
      
        self.image_display_label.clear()

      
        image_files = os.listdir(folder_path)
        if image_files:
            
            layout = QVBoxLayout()

            
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    pixmap = pixmap.scaledToWidth(400)  
                    label = QLabel()
                    label.setPixmap(pixmap)
                    layout.addWidget(label)

          
            self.image_display_label.setLayout(layout)
        

        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    login_window = LoginWindow()  
    if login_window.exec_() == QDialog.Accepted:  
        main_window = MyWindow()  
        main_window.show()  
        sys.exit(app.exec_())

