import pyodbc
import pandas as pd
from .config import DATABASE_PATH

class DatabaseManager:
    """Headless database operations"""
    
    @staticmethod
    def get_connection():
        """Get database connection"""
        try:
            if not DATABASE_PATH or not DATABASE_PATH.endswith('.accdb'):
                return None
                
            conn_str = f'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={DATABASE_PATH};'
            return pyodbc.connect(conn_str)
        except Exception as e:
            print(f"Database connection error: {e}")
            return None
    
    @staticmethod
    def validate_user(username, password):
        """Validate user credentials - headless"""
        conn = DatabaseManager.get_connection()
        if not conn:
            return False
            
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT Username FROM Users WHERE Username=? AND Password=?", 
                         (username, password))
            user = cursor.fetchone()
            conn.close()
            return user is not None
        except Exception as e:
            print(f"User validation error: {e}")
            return False
    
    @staticmethod
    def register_user(username, password, email, phone):
        """Register new user - headless"""
        conn = DatabaseManager.get_connection()
        if not conn:
            return False, "Database connection failed"
            
        try:
            cursor = conn.cursor()
            
            # Check if username exists
            cursor.execute("SELECT Username FROM Users WHERE Username=?", (username,))
            if cursor.fetchone():
                return False, "Username already exists"
            
            # Insert new user
            cursor.execute(
                "INSERT INTO Users (Username, Password, Email, Phone_Number) VALUES (?, ?, ?, ?)",
                (username, password, email, phone)
            )
            conn.commit()
            conn.close()
            return True, "User registered successfully"
            
        except Exception as e:
            return False, f"Registration error: {str(e)}"
    
    @staticmethod
    def get_detections():
        """Get all detections - headless"""
        conn = DatabaseManager.get_connection()
        if not conn:
            return [], []
            
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM Detections")
            rows = cursor.fetchall()
            columns = [column[0] for column in cursor.description]
            conn.close()
            return rows, columns
        except Exception as e:
            print(f"Data retrieval error: {e}")
            return [], []
    
    @staticmethod
    def save_detection(detection_data):
        """Save detection to database - headless"""
        conn = DatabaseManager.get_connection()
        if not conn:
            return False
            
        try:
            cursor = conn.cursor()
            # Implement based on your detection table structure
            # cursor.execute("INSERT INTO Detections ...", detection_data)
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Save detection error: {e}")
            return False