import os

# Path configuration - works in both Docker and local
BASE_PATH = "/app/Data"  # Docker path
LOCAL_BASE_PATH = "D:/My_Diploma/System2"  # Local development path

def get_base_path():
    """Determine the correct base path based on environment"""
    if os.path.exists(BASE_PATH):
        return BASE_PATH  # Docker environment
    elif os.path.exists(LOCAL_BASE_PATH):
        return LOCAL_BASE_PATH  # Local Windows development
    else:
        # Fallback - use current directory Data folder
        local_data = os.path.join(os.path.dirname(__file__), "..", "Data")
        if os.path.exists(local_data):
            return local_data
        else:
            os.makedirs(local_data, exist_ok=True)
            return local_data

# Dynamic path configuration
BASE_PATH = get_base_path()

# Image paths
BACKGROUND_IMAGE_MAIN = os.path.join(BASE_PATH, "2.jpeg")
BACKGROUND_IMAGE_STRUCTURE = os.path.join(BASE_PATH, "SFD.png")
LOGIN_IMAGE = os.path.join(BASE_PATH, "human-skeletal.jpg")
THUMBNAIL_IMAGE = os.path.join(BASE_PATH, "thumbnail_1.jpg")

# Model path
MODEL_PATH = os.path.join(BASE_PATH, "exp2/weights/last.pt")

# Data paths
HUMAN_SKELETAL_PATH = os.path.join(BASE_PATH, "Human_Skeletal")
GORILLA_SKELETAL_PATH = os.path.join(BASE_PATH, "Gorilla_Skeletal")
OUR_DATABASE_PATH = os.path.join(BASE_PATH, "Our_DataBase")
ON_DATABASE_PATH = os.path.join(BASE_PATH, "On_DataBase")
DATABASE_PATH = os.path.join(BASE_PATH, "Homam_SK.accdb")

# Model evaluation paths
MODEL_PATHS = {
    'confusion_matrix': os.path.join(BASE_PATH, "model/confusion_matrix"),
    'f1_curve': os.path.join(BASE_PATH, "model/F1_curve"),
    'labels': os.path.join(BASE_PATH, "model/labels"),
    'labels_correlogram': os.path.join(BASE_PATH, "model/labels_correlogram"),
    'p_curve': os.path.join(BASE_PATH, "model/P_curve"),
    'pr_curve': os.path.join(BASE_PATH, "model/PR_curve"),
    'r_curve': os.path.join(BASE_PATH, "model/R_curve"),
    'results': os.path.join(BASE_PATH, "model/results")
}

def validate_environment():
    """Validate that required paths exist"""
    print(f"Base path: {BASE_PATH}")
    
    required_paths = [
        HUMAN_SKELETAL_PATH,
        GORILLA_SKELETAL_PATH,
        OUR_DATABASE_PATH,
        ON_DATABASE_PATH
    ]
    
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
            print(f"⚠️  Missing path: {path}")
        else:
            print(f"✅ Path exists: {path}")
    
    return len(missing_paths) == 0