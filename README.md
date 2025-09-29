### A program for classifying human and gorilla skeletal structures using convolutional neural networks (CNN) and the You Only Look Once (YOLO) method


# Skeleton Analyzer Pro 🦴

A project for classifying **human and gorilla skeletal structures** using Convolutional Neural Networks (CNNs) and YOLO, with both GUI and headless modes.  

---

## 📋 Description

This repository includes:

- **Core code**: model inference, preprocessing, evaluation, and database logic (`core/`).  
- **GUI (PyQt5)**: user-facing interface for loading images & videos and seeing results (`gui/`).  
- **Jupyter Notebook**: full pipeline demo and reproducible tutorial (`skeleton-analyzer.ipynb`).  
- **Docker & docker-compose**: for running in isolated, reproducible environments (especially headless mode).  
- **Data folders & example images**: to test image/video inference and evaluation.  

---

## 📂 Repository Structure

skeleton-analyzer/
├── Data/ # Example skeletal datasets
├── core/ # Core logic: model, config, database, preprocessing, evaluation
│ ├── config.py
│ ├── model.py
│ ├── preprocessing.py
│ ├── evaluation.py
│ └── database.py
│
├── gui/ # GUI code (PyQt5)
│ └── app.py
│
├── skeleton-analyzer.ipynb # Notebook demonstrating full pipeline
├── Dockerfile # Docker setup for headless execution
├── docker-compose.yml # Compose file to run Docker + bind Data
├── main.py # Entry point (choose headless or GUI mode)
├── requirements.txt # Python dependencies
├── .gitattributes # Overrides for GitHub language stats
├── .gitignore
└── README.md # This file

yaml
Copy code

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/leonardo672/A-program-for-classifying-human-and-gorilla-skeletal-structures-using-convolutional-neural-networks-.git
cd A-program-for-classifying-human-and-gorilla-skeletal-structures-using-convolutional-neural-networks-
2. Local Python setup (without Docker)
Ensure you have Python 3.9+.

(Optional) Create a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate         # On macOS/Linux
venv\Scripts\activate            # On Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
3. Using Docker (recommended)
bash
Copy code
docker build -t skeleton-analyzer .
docker-compose up
This mounts your Data/ folder into the container.

Runs in headless mode by default.

🧪 How to Run / Test
🔹 Headless Mode
bash
Copy code
python main.py --mode headless
Validates environment and data paths

Lists available evaluation results

Reports how many human/gorilla skeletal images are found

🔹 Single Image Inference
python
Copy code
from core.model import SkeletonModel
model = SkeletonModel()
model.load_model()
result = model.analyze_image("Data/Human_Skeletal/sample.jpg")
print(result['predictions'])
🔹 Batch Image Inference
python
Copy code
from core.preprocessing import ImageManager
paths = ImageManager.get_skeletal_images("human")
results = model.analyze_images_batch(paths)
for r in results:
    print(r["image_path"], len(r["predictions"]))
🔹 Video Inference
python
Copy code
detections = model.analyze_video("Data/sample_video.mp4")
print("Detections:", len(detections))
🔹 Model Evaluation Utilities
python
Copy code
from core.evaluation import ModelEvaluator
print("Available evaluations:", ModelEvaluator.get_available_evaluations())
imgs = ModelEvaluator.get_evaluation_images("confusion_matrix")
print("Images in confusion matrix:", imgs)
🔹 GUI (PyQt5 Interface)
bash
Copy code
python main.py --mode gui
Opens GUI for interactive testing

Allows you to select images or videos for analysis

Displays results visually

⚠️ If the GUI window appears blank:

Ensure paths in core/config.py are correct

Check your PyQt5 installation

📒 Jupyter Notebook Demo
The file skeleton-analyzer.ipynb demonstrates the full pipeline in one place:

Load skeletal datasets

Run training and inference

Generate evaluation metrics and plots

Perform image & video analysis

👉 This notebook serves as a tutorial & reproducibility showcase.
You can run it locally with JupyterLab or upload it to Google Colab.

🧮 GitHub Language Stats & .gitattributes
This repository includes a .gitattributes file to ensure GitHub properly counts Dockerfile and YAML files in the repository’s language breakdown.

If you notice that Docker/YAML are not showing in the GitHub “Languages” section, .gitattributes is used to force GitHub Linguist to recognize them.

✅ Why This Design Is Professional
Separation of concerns → core logic, GUI, data, and Docker are modular.

Notebook → gives newcomers a complete walkthrough in a single file.

Docker → ensures reproducibility and simple setup on any machine.

Professional structure → CI/CD ready, scalable, and maintainable.

Both GUI and headless support → works for interactive users and automated workflows.

📌 Notes / Known Issues
Make sure paths in core/config.py match your local file structure.

GUI requires a display environment (won’t work in headless Docker without X11 forwarding).

If you see:

vbnet
Copy code
ModuleNotFoundError: No module named 'yaml'
Fix it by installing PyYAML:

bash
Copy code
pip install pyyaml
The Jupyter notebook is provided for demonstration & clarity; the core modular code is what production workflows should rely on.

⚡ Quick Demo Commands
bash
Copy code
# Headless mode test
python main.py --mode headless

# Test single image inference
python -c "from core.model import SkeletonModel; m=SkeletonModel(); m.load_model(); print(m.analyze_image('Data/Human_Skeletal/sample.jpg')['predictions'])"

# Test video inference
python -c "from core.model import SkeletonModel; m=SkeletonModel(); m.load_model(); print(len(m.analyze_video('Data/sample_video.mp4')))"

# Launch GUI
python main.py --mode gui


#### The initial outline of the system before creating the user interfaces:
<img width="700" height="400" alt="10" src="https://github.com/user-attachments/assets/a6a6265d-c9c1-4dd6-81c6-88ece85064ad" />

#### Figure 1 – Registration and login form:
![image](https://github.com/user-attachments/assets/d80ff996-1757-4ef0-ae69-18bd8d6e4233)


#### Figure 2 – Main window of the system:
![image](https://github.com/user-attachments/assets/80ca9b51-a2dc-4516-bbee-a5c78ff601a7)


#### Figure 3 – Image selection procedure:
![image](https://github.com/user-attachments/assets/f8da182b-7069-4c12-8359-8f4fed402e53)

#### Figure 4 – Presentation of the results of the analysis of the gorilla skeleton:
![image](https://github.com/user-attachments/assets/1aba26bb-56b6-4f78-a05c-55d63f381988)


#### Figure 5 – Presentation of the results of the analysis of the human skeleton:
![image](https://github.com/user-attachments/assets/c0a20c37-9183-4d79-ad2e-19d11a713579)


#### Figure 6 – Selecting a video file from a local folder:
![image](https://github.com/user-attachments/assets/871f54ed-a6f6-4ce5-b0cb-c23292e1e166)


#### Figure 7 – Start of live analysis in real time on video:
![image](https://github.com/user-attachments/assets/b25acb5c-6232-48df-85cb-037a338c4154)


#### Figure 8 – Appearance of the real-time database after the end of the video stream:
![image](https://github.com/user-attachments/assets/6bbfcac5-a414-4da5-809c-fa2fdb6bbe0a)


#### Figure 9 – Detection database:
![image](https://github.com/user-attachments/assets/84d3a96f-a538-47e4-b0a9-97803870918d)


#### Figure 10 – Database of analyzed images:
![image](https://github.com/user-attachments/assets/4e02e359-555c-4b8d-beac-9785af80731a)


#### Figure 11 – Structure of interface objects:
![image](https://github.com/user-attachments/assets/9935153f-ae3f-4857-8126-8412a5913d61)


#### Figure 12 – Structure of the human skeleton:
![image](https://github.com/user-attachments/assets/b1191ece-1a2f-4b2e-a8cb-514badd656dc)


#### Figure 13 – Structure of the gorilla skeleton:
![image](https://github.com/user-attachments/assets/d0e11e5d-5161-47e1-8613-3f41151c01a7)


#### Figure 14 – Model Strength Overview:
![image](https://github.com/user-attachments/assets/4eb415af-f405-42df-b881-f16dbb722211)


#### Figure 15 – Confusion Matrix View:
![image](https://github.com/user-attachments/assets/6515d9d1-4def-42cc-987c-3e894643ce44)

#### Figure 16 – Representation Curve F1:
![image](https://github.com/user-attachments/assets/8381a8d6-4b03-4e7c-9cc7-1cd20593c339)

#### Figure 17 – Presentation of labels:
![image](https://github.com/user-attachments/assets/7e260b8b-66df-4fe1-9956-8feeed2f035d)


#### Figure 18 – Mark Correlogram view:
![image](https://github.com/user-attachments/assets/5d395acc-5910-4732-8b73-0b3b93b37e74)


#### Figure 19 – Curve P representation:
![image](https://github.com/user-attachments/assets/a9e60e3b-73b4-42dc-bb72-bc01224fcedb)


#### Figure 20 – R Curve Representation:
![image](https://github.com/user-attachments/assets/efc15de5-aaaf-40d9-aa3b-c14d2a5f6db6)


#### Figure 21 – PR Curve view:
![image](https://github.com/user-attachments/assets/1173efa5-78e0-4243-b07c-32add0298ba3)


#### Figure 22 – Presentation of results:
![image](https://github.com/user-attachments/assets/9f75a39b-2b84-42e5-b96d-e6469c56d0b0)







