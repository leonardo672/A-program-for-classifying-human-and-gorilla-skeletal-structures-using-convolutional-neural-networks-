## **A-human-gorilla-skeleton-cnn**

### Description
A Python project for classifying and analyzing human and gorilla skeletal structures using deep learning.  
Supports headless execution, GUI (PyQt5), Docker, and Jupyter-based experimentation.

### ⚙ Primary Execution Environment
This project is **primarily designed to be executed and demonstrated in a Jupyter Notebook environment**.
The Jupyter implementation represents the **reference and stable execution flow** used for:
- New skeletal pattern recognition
- Dataset exploration and testing
- Model validation and comparison
- End-to-end pipeline demonstration and reproducibility

-----

### 🐳 Docker Execution Note
A Docker-based pipeline is included to illustrate the intended deployment architecture.
However, due to upstream compatibility constraints between YOLOv5, PyTorch, and the Ultralytics package, container execution may require additional version alignment or dependency pinning depending on the runtime environment.
This limitation **does not affect:**
- The core project logic.
- The model architecture or trained weights.
- The Jupyter-based execution (reference implementation).

---

#### **This repository includes:**
- **Core code**: model inference, preprocessing, evaluation, and database logic (`core/`).  
- **GUI (PyQt5)**: user-facing interface for loading images & videos and visualizing results (`gui/`).  
- **Jupyter Notebook**: full pipeline demo and reproducible tutorial (`skeleton-analyzer.ipynb`).  
- **Docker & docker-compose**: isolated execution for headless and deployment-oriented use cases.  
- **Data folders & example images**: for testing inference and evaluation workflows.


#### Install Project Dependencies
``` 
pip install --upgrade pip
pip install -r requirements.txt 
```

#### Running the Project (Python)
```# Headless mode
python main.py --mode headless

# GUI mode (requires local display)
python main.py --mode gui

# Auto mode (detects GUI or headless automatically)
python main.py --mode auto
```
**Note:**
Headless mode is suitable for servers, CI/CD pipelines, or testing without a display.
GUI mode requires PyQt5 and a local display environment.

#### Docker Setup
##### 1. Build the Docker Image:

The ``` Dockerfile```  defines a reusable image called ```skeleton-analyzer```. To build it:

``` 
# From the project root directory
docker-compose build
``` 
This will:
- Install Python dependencies (```requirements.txt```).
- Set up the project structure.
- Prepare the image for headless execution.    
After building, the image will be reusable and stored locally.  
##### 2. Run the Project in a Container: 
To run the project in headless mode inside a container:
``` 
docker-compose run skeleton-analyzer 
``` 
**What this does:**
- Creates a container from the skeleton-analyzer image.
- Mounts your local Data/ folder into /app/Data in the container.
- Runs the project in headless mode.
- Prints environment validation, available evaluations, and skeletal images.    

##### 3. Optional: Running GUI Mode:
Running GUI inside Docker is not recommended for Windows without additional setup (X11, VNC). For GUI testing, it’s easier to run the project locally using Python:
``` 
python main.py --mode gui
``` 
**Note:** For full demonstration, the Jupyter notebook (skeleton-analyzer.ipynb) provides a complete GUI-like layout and workflow. You can use it to test the GUI and pipeline without running the PyQt5 application inside Docker.


##### 4. Stop and Remove Containers:   
After testing, you can remove containers while keeping the image reusable:
``` 
docker-compose down
``` 
Your ``` skeleton-analyzer``` image remains locally and can be reused to create new containers.
 
#### Testing the Project

##### 1. Test single Image Inference:
```
from core.model import SkeletonModel
model = SkeletonModel()
model.load_model()
result = model.analyze_image("Data/Human_Skeletal/sample.jpg")
print(result['predictions'])
```

##### 2. Test batch Image Inference:
```
from core.preprocessing import ImageManager
image_paths = ImageManager.get_skeletal_images('human')
for img_path in image_paths:
    result = model.analyze_image(img_path)
    print(result['predictions'])

```
##### 3. Test video Inference:
```
detections = model.analyze_video("Data/sample_video.mp4")
print(f"Total detections: {len(detections)}")

```
##### 4. Test model Evaluation Utilities:
```
from core.evaluation import ModelEvaluator
available = ModelEvaluator.get_available_evaluations()
print(available)

for eval_type in available:
    images = ModelEvaluator.get_evaluation_images(eval_type)
    print(f"{eval_type}: {len(images)} images")

```
##### 5. Test GUI (PyQt5):
```
from gui.app import run_gui
run_gui()

```
GUI allows image/video selection, evaluation display, and login management.


#### 📒 Jupyter Notebook (skeleton-analyzer.ipynb)
##### The file skeleton-analyzer.ipynb demonstrates the full pipeline in one place: 
- Load skeletal datasets.
- Run training and inference.
- Generate evaluation metrics and plots.
- Perform image & video analysis.     

👉 This notebook serves as a tutorial / reproducibility showcase.



---

#### **The initial outline of the system before creating the user interfaces:**
<img width="700" height="400" alt="10" src="https://github.com/user-attachments/assets/a6a6265d-c9c1-4dd6-81c6-88ece85064ad" />

##### **Figure 1 – Registration and login form:**
![image](https://github.com/user-attachments/assets/d80ff996-1757-4ef0-ae69-18bd8d6e4233)


##### **Figure 2 – Main window of the system:**
![image](https://github.com/user-attachments/assets/80ca9b51-a2dc-4516-bbee-a5c78ff601a7)


##### **Figure 3 – Image selection procedure:**
![image](https://github.com/user-attachments/assets/f8da182b-7069-4c12-8359-8f4fed402e53)

##### **Figure 4 – Presentation of the results of the analysis of the gorilla skeleton:**
![image](https://github.com/user-attachments/assets/1aba26bb-56b6-4f78-a05c-55d63f381988)


##### **Figure 5 – Presentation of the results of the analysis of the human skeleton:**
![image](https://github.com/user-attachments/assets/c0a20c37-9183-4d79-ad2e-19d11a713579)


##### **Figure 6 – Selecting a video file from a local folder:**
![image](https://github.com/user-attachments/assets/871f54ed-a6f6-4ce5-b0cb-c23292e1e166)


##### **Figure 7 – Start of live analysis in real time on video:**
![image](https://github.com/user-attachments/assets/b25acb5c-6232-48df-85cb-037a338c4154)


##### **Figure 8 – Appearance of the real-time database after the end of the video stream:**
![image](https://github.com/user-attachments/assets/6bbfcac5-a414-4da5-809c-fa2fdb6bbe0a)


##### **Figure 9 – Detection database:**
![image](https://github.com/user-attachments/assets/84d3a96f-a538-47e4-b0a9-97803870918d)


##### **Figure 10 – Database of analyzed images:**
![image](https://github.com/user-attachments/assets/4e02e359-555c-4b8d-beac-9785af80731a)


##### **Figure 11 – Structure of interface objects:**
![image](https://github.com/user-attachments/assets/9935153f-ae3f-4857-8126-8412a5913d61)


##### **Figure 12 – Structure of the human skeleton:**
![image](https://github.com/user-attachments/assets/b1191ece-1a2f-4b2e-a8cb-514badd656dc)


##### **Figure 13 – Structure of the gorilla skeleton:**
![image](https://github.com/user-attachments/assets/d0e11e5d-5161-47e1-8613-3f41151c01a7)


##### **Figure 14 – Model Strength Overview:**
![image](https://github.com/user-attachments/assets/4eb415af-f405-42df-b881-f16dbb722211)


##### **Figure 15 – Confusion Matrix View:**
![image](https://github.com/user-attachments/assets/6515d9d1-4def-42cc-987c-3e894643ce44)

##### **Figure 16 – Representation Curve F1:**
![image](https://github.com/user-attachments/assets/8381a8d6-4b03-4e7c-9cc7-1cd20593c339)

##### **Figure 17 – Presentation of labels:**
![image](https://github.com/user-attachments/assets/7e260b8b-66df-4fe1-9956-8feeed2f035d)


##### **Figure 18 – Mark Correlogram view:**
![image](https://github.com/user-attachments/assets/5d395acc-5910-4732-8b73-0b3b93b37e74)


##### **Figure 19 – Curve P representation:**
![image](https://github.com/user-attachments/assets/a9e60e3b-73b4-42dc-bb72-bc01224fcedb)


##### **Figure 20 – R Curve Representation:**
![image](https://github.com/user-attachments/assets/efc15de5-aaaf-40d9-aa3b-c14d2a5f6db6)


##### **Figure 21 – PR Curve view:**
![image](https://github.com/user-attachments/assets/1173efa5-78e0-4243-b07c-32add0298ba3)


##### **Figure 22 – Presentation of results:**
![image](https://github.com/user-attachments/assets/9f75a39b-2b84-42e5-b96d-e6469c56d0b0)







