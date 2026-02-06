import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import hashlib

class FaceAttendanceSystem:
    def __init__(self):
        self.known_faces_dir = "known_faces"
        self.attendance_file = "attendance.csv"
        self.students_file = "students.csv"
        self.known_face_data = {}
        self.student_ids = {}
        
        # Initialize ORB detector with more features
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Create directories
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
        
        # Initialize student database
        self.init_student_database()
        
        # Webcam variables
        self.video_capture = None
        self.is_webcam_running = False
        self.marked_today = set()
        
        # Load existing faces
        self.load_known_faces()
        
        # Create GUI
        self.create_gui()
    
    def generate_student_id(self, name):
        """Generate unique student ID"""
        # Create hash from name and timestamp
        unique_string = f"{name}_{datetime.now().isoformat()}"
        hash_object = hashlib.md5(unique_string.encode())
        hash_hex = hash_object.hexdigest()[:6].upper()
        return f"STU{hash_hex}"
    
    def init_student_database(self):
        """Initialize student database with IDs"""
        if not os.path.exists(self.students_file):
            df = pd.DataFrame(columns=['StudentID', 'Name', 'RegisteredDate'])
            df.to_csv(self.students_file, index=False)
        else:
            df = pd.read_csv(self.students_file)
            for _, row in df.iterrows():
                self.student_ids[row['Name']] = row['StudentID']
    
    def add_student_to_database(self, name):
        """Add student to database with unique ID"""
        if name in self.student_ids:
            return self.student_ids[name]
        
        student_id = self.generate_student_id(name)
        self.student_ids[name] = student_id
        
        # Add to CSV
        df = pd.read_csv(self.students_file)
        new_record = pd.DataFrame({
            'StudentID': [student_id],
            'Name': [name],
            'RegisteredDate': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        })
        df = pd.concat([df, new_record], ignore_index=True)
        df.to_csv(self.students_file, index=False)
        
        return student_id
    
    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("Face Attendance System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2c3e50")
        
        # Title
        title_frame = tk.Frame(self.root, bg="#34495e", height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="Face Attendance System",
            font=("Arial", 24, "bold"),
            bg="#34495e",
            fg="white"
        )
        title_label.pack(pady=20)
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_frame, bg="#34495e", width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        controls_label = tk.Label(
            left_panel,
            text="Controls",
            font=("Arial", 16, "bold"),
            bg="#34495e",
            fg="white"
        )
        controls_label.pack(pady=20)
        
        # Buttons
        btn_style = {
            "font": ("Arial", 11),
            "bg": "#3498db",
            "fg": "white",
            "relief": tk.FLAT,
            "cursor": "hand2",
            "height": 2
        }
        
        self.btn_register = tk.Button(
            left_panel,
            text=" Register New Face ",
            command=self.open_register_window,
            **btn_style
        )
        self.btn_register.pack(fill=tk.X, padx=20, pady=5)
        
        self.btn_recognize = tk.Button(
            left_panel,
            text=" Recognize Face",
            command=self.open_recognize_window,
            **btn_style
        )
        self.btn_recognize.pack(fill=tk.X, padx=20, pady=5)
        
        self.btn_webcam = tk.Button(
            left_panel,
            text=" Live Attendance Mode",
            command=self.toggle_webcam,
            **btn_style
        )
        self.btn_webcam.pack(fill=tk.X, padx=20, pady=5)
        
        self.btn_view = tk.Button(
            left_panel,
            text=" View Attendance",
            command=self.view_attendance_window,
            **btn_style
        )
        self.btn_view.pack(fill=tk.X, padx=20, pady=5)
        
        self.btn_list = tk.Button(
            left_panel,
            text=" Registered Persons",
            command=self.show_registered_persons,
            **btn_style
        )
        self.btn_list.pack(fill=tk.X, padx=20, pady=5)
        
        # Status label
        status_frame = tk.Frame(left_panel, bg="#34495e")
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)
        
        tk.Label(
            status_frame,
            text="Registered Persons:",
            font=("Arial", 10),
            bg="#34495e",
            fg="white"
        ).pack()
        
        self.registered_count = tk.Label(
            status_frame,
            text=str(len(self.known_face_data)),
            font=("Arial", 20, "bold"),
            bg="#34495e",
            fg="#2ecc71"
        )
        self.registered_count.pack()
        
        # Right panel - Display
        right_panel = tk.Frame(main_frame, bg="#34495e")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.display_label = tk.Label(
            right_panel,
            text=" Camera Feed / Recognition Display",
            font=("Arial", 14),
            bg="#34495e",
            fg="white",
            compound=tk.TOP
        )
        self.display_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Log panel
        log_frame = tk.Frame(self.root, bg="#34495e", height=150)
        log_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        log_frame.pack_propagate(False)
        
        tk.Label(
            log_frame,
            text="Activity Log",
            font=("Arial", 12, "bold"),
            bg="#34495e",
            fg="white"
        ).pack(anchor=tk.W, padx=10, pady=5)
        
        self.log_text = tk.Text(
            log_frame,
            height=6,
            bg="#2c3e50",
            fg="#ecf0f1",
            font=("Courier", 9),
            relief=tk.FLAT
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.log("System initialized successfully")
        self.log(f"Loaded {len(self.known_face_data)} registered faces")
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def extract_face_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        if len(faces) == 0:
            return None
        
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(gray.shape[1] - x, w + 2 * padding)
        h = min(gray.shape[0] - y, h + 2 * padding)
        
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (250, 250))
        
        # Apply Gaussian blur to reduce noise
        face_roi = cv2.GaussianBlur(face_roi, (5, 5), 0)
        
        keypoints, descriptors = self.orb.detectAndCompute(face_roi, None)
        
        if descriptors is None or len(descriptors) < 10:
            return None
        
        return {'face_roi': face_roi, 'descriptors': descriptors, 'keypoints': keypoints}
    
    def compare_faces(self, features1, features2, threshold=50):
        if features1 is None or features2 is None:
            return False, 100
        
        desc1 = features1['descriptors']
        desc2 = features2['descriptors']
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        try:
            matches = bf.knnMatch(desc1, desc2, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) > 15:
                avg_distance = sum([m.distance for m in good_matches[:20]]) / min(20, len(good_matches))
                return avg_distance < threshold, avg_distance
            else:
                return False, 100
        except:
            return False, 100
    
    def load_known_faces(self):
        self.known_face_data = {}
        if not os.path.exists(self.known_faces_dir):
            return
        
        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith('.npy'):
                name = filename[:-4]
                img_path = os.path.join(self.known_faces_dir, f"{name}.jpg")
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    features = self.extract_face_features(img)
                    if features is not None:
                        self.known_face_data[name] = features
    
    def mark_attendance(self, name):
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d")
        time_string = now.strftime("%H:%M:%S")
        
        # Get or create student ID
        if name not in self.student_ids:
            student_id = self.add_student_to_database(name)
        else:
            student_id = self.student_ids[name]
        
        # Check if attendance file exists
        if os.path.exists(self.attendance_file):
            df = pd.read_csv(self.attendance_file)
            if not df.empty:
                today_attendance = df[(df['Name'] == name) & (df['Date'] == date_string)]
                if not today_attendance.empty:
                    self.log(f"{name} (ID: {student_id}) already marked attendance today")
                    return False
        else:
            df = pd.DataFrame(columns=['StudentID', 'Name', 'Date', 'Time'])
        
        new_record = pd.DataFrame({
            'StudentID': [student_id],
            'Name': [name],
            'Date': [date_string],
            'Time': [time_string]
        })
        df = pd.concat([df, new_record], ignore_index=True)
        df.to_csv(self.attendance_file, index=False)
        
        self.log(f" Attendance marked: {name} (ID: {student_id}) at {time_string}")
        return True
    
    def open_register_window(self):
        register_win = tk.Toplevel(self.root)
        register_win.title("Register New Face")
        register_win.geometry("500x300")
        register_win.configure(bg="#34495e")
        register_win.resizable(False, False)
        
        tk.Label(
            register_win,
            text="Register New Person",
            font=("Arial", 16, "bold"),
            bg="#34495e",
            fg="white"
        ).pack(pady=20)
        
        method_frame = tk.Frame(register_win, bg="#34495e")
        method_frame.pack(pady=10)
        
        tk.Label(
            method_frame,
            text="Choose Registration Method:",
            font=("Arial", 11, "bold"),
            bg="#34495e",
            fg="white"
        ).pack(pady=10)
        
        btn_frame = tk.Frame(method_frame, bg="#34495e")
        btn_frame.pack(pady=10)
        
        tk.Button(
            btn_frame,
            text=" Capture from Webcam",
            command=lambda: self.register_from_webcam(register_win),
            bg="#9b59b6",
            fg="white",
            font=("Arial", 11, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            width=22,
            height=2
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            btn_frame,
            text=" Upload Photo",
            command=lambda: self.register_from_file(register_win),
            bg="#3498db",
            fg="white",
            font=("Arial", 11, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            width=22,
            height=2
        ).pack(side=tk.LEFT, padx=10)
    
    def register_from_webcam(self, parent_win):
        parent_win.destroy()
        
        webcam_win = tk.Toplevel(self.root)
        webcam_win.title("Register from Webcam")
        webcam_win.geometry("700x650")
        webcam_win.configure(bg="#34495e")
        
        tk.Label(
            webcam_win,
            text="Capture Face from Webcam",
            font=("Arial", 16, "bold"),
            bg="#34495e",
            fg="white"
        ).pack(pady=10)
        
        name_frame = tk.Frame(webcam_win, bg="#34495e")
        name_frame.pack(pady=10)
        
        tk.Label(
            name_frame,
            text="Name:",
            font=("Arial", 11),
            bg="#34495e",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        name_entry = tk.Entry(name_frame, font=("Arial", 11), width=30)
        name_entry.pack(side=tk.LEFT, padx=5)
        
        video_label = tk.Label(webcam_win, bg="#2c3e50")
        video_label.pack(pady=10)
        
        status_label = tk.Label(
            webcam_win,
            text="Position your face in the frame. Keep steady for best results.",
            font=("Arial", 11),
            bg="#34495e",
            fg="#f39c12"
        )
        status_label.pack(pady=5)
        
        btn_frame = tk.Frame(webcam_win, bg="#34495e")
        btn_frame.pack(pady=10)
        
        captured_image = [None]
        webcam_active = [True]
        photo_captured = [False]
        
        def submit_registration():
            name = name_entry.get().strip()
            if not name:
                messagebox.showerror("Error", "Please enter a name first")
                return
            
            if not photo_captured[0]:
                messagebox.showerror("Error", "Please capture a photo first")
                return
            
            if captured_image[0] is not None:
                features = self.extract_face_features(captured_image[0])
                
                if features is None:
                    messagebox.showerror("Error", "No face detected. Please try again with better lighting.")
                    photo_captured[0] = False
                    capture_btn.config(state=tk.NORMAL, bg="#9b59b6")
                    submit_btn.config(state=tk.DISABLED, bg="#95a5a6")
                    return
                
                save_path = os.path.join(self.known_faces_dir, f"{name}.jpg")
                cv2.imwrite(save_path, captured_image[0])
                
                features_path = os.path.join(self.known_faces_dir, f"{name}.npy")
                np.save(features_path, features['descriptors'])
                
                self.known_face_data[name] = features
                student_id = self.add_student_to_database(name)
                self.registered_count.config(text=str(len(self.known_face_data)))
                
                self.log(f"✓ Registered from webcam: {name} (ID: {student_id})")
                messagebox.showinfo("Success", f"{name} registered successfully!\nStudent ID: {student_id}")
                webcam_active[0] = False
                webcam_win.destroy()
        
        def capture_photo():
            if captured_image[0] is not None:
                features = self.extract_face_features(captured_image[0])
                
                if features is None:
                    messagebox.showwarning("Warning", "No clear face detected. Please adjust position and lighting.")
                    return
                
                photo_captured[0] = True
                capture_btn.config(state=tk.DISABLED, bg="#95a5a6")
                submit_btn.config(state=tk.NORMAL, bg="#2ecc71")
                status_label.config(
                    text="✓ Photo captured successfully! Enter name and click Submit",
                    fg="#2ecc71"
                )
        
        capture_btn = tk.Button(
            btn_frame,
            text="📸 Capture Photo",
            command=capture_photo,
            bg="#9b59b6",
            fg="white",
            font=("Arial", 11, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            width=15,
            height=2
        )
        capture_btn.pack(side=tk.LEFT, padx=5)
        
        submit_btn = tk.Button(
            btn_frame,
            text=" Submit",
            command=submit_registration,
            bg="#95a5a6",
            fg="white",
            font=("Arial", 11, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            width=15,
            height=2,
            state=tk.DISABLED
        )
        submit_btn.pack(side=tk.LEFT, padx=5)
        
        def close_webcam():
            webcam_active[0] = False
            webcam_win.destroy()
        
        tk.Button(
            btn_frame,
            text=" Cancel",
            command=close_webcam,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 11, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            width=15,
            height=2
        ).pack(side=tk.LEFT, padx=5)
        
        cap = cv2.VideoCapture(0)
        
        def update_frame():
            if not webcam_active[0]:
                cap.release()
                return
            
            ret, frame = cap.read()
            if ret:
                captured_image[0] = frame.copy()
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Face Detected", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                if len(faces) > 0:
                    if not photo_captured[0]:
                        status_label.config(text="✓ Face detected! Click Capture Photo button", fg="#2ecc71")
                else:
                    if not photo_captured[0]:
                        status_label.config(text="⚠ No face detected. Adjust position and lighting", fg="#f39c12")
                
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display_frame = cv2.resize(display_frame, (640, 480))
                img_pil = Image.fromarray(display_frame)
                img_tk = ImageTk.PhotoImage(img_pil)
                
                video_label.config(image=img_tk)
                video_label.image = img_tk
            
            if webcam_active[0]:
                webcam_win.after(10, update_frame)
        
        update_frame()
        
        def on_close():
            webcam_active[0] = False
            cap.release()
            webcam_win.destroy()
        
        webcam_win.protocol("WM_DELETE_WINDOW", on_close)
    
    def register_from_file(self, parent_win):
        parent_win.destroy()
        
        file_win = tk.Toplevel(self.root)
        file_win.title("Register from File")
        file_win.geometry("400x250")
        file_win.configure(bg="#34495e")
        file_win.resizable(False, False)
        
        tk.Label(
            file_win,
            text="Register from Photo File",
            font=("Arial", 16, "bold"),
            bg="#34495e",
            fg="white"
        ).pack(pady=20)
        
        frame = tk.Frame(file_win, bg="#34495e")
        frame.pack(pady=10)
        
        tk.Label(frame, text="Name:", font=("Arial", 11), bg="#34495e", fg="white").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        name_entry = tk.Entry(frame, font=("Arial", 11), width=25)
        name_entry.grid(row=0, column=1, padx=10, pady=10)
        
        tk.Label(frame, text="Image:", font=("Arial", 11), bg="#34495e", fg="white").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        image_path_var = tk.StringVar()
        image_entry = tk.Entry(frame, textvariable=image_path_var, font=("Arial", 11), width=25)
        image_entry.grid(row=1, column=1, padx=10, pady=10)
        
        def browse_image():
            filename = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png")]
            )
            if filename:
                image_path_var.set(filename)
        
        tk.Button(
            frame,
            text="Browse",
            command=browse_image,
            bg="#3498db",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2"
        ).grid(row=1, column=2, padx=5)
        
        def register():
            name = name_entry.get().strip()
            image_path = image_path_var.get().strip()
            
            if not name or not image_path:
                messagebox.showerror("Error", "Please fill all fields")
                return
            
            if not os.path.exists(image_path):
                messagebox.showerror("Error", "Image file not found")
                return
            
            img = cv2.imread(image_path)
            if img is None:
                messagebox.showerror("Error", "Could not read image")
                return
            
            features = self.extract_face_features(img)
            if features is None:
                messagebox.showerror("Error", "No face found in image. Please use a clear face photo.")
                return
            
            save_path = os.path.join(self.known_faces_dir, f"{name}.jpg")
            cv2.imwrite(save_path, img)
            
            features_path = os.path.join(self.known_faces_dir, f"{name}.npy")
            np.save(features_path, features['descriptors'])
            
            self.known_face_data[name] = features
            student_id = self.add_student_to_database(name)
            self.registered_count.config(text=str(len(self.known_face_data)))
            
            self.log(f"✓ Registered from file: {name} (ID: {student_id})")
            messagebox.showinfo("Success", f"{name} registered successfully!\nStudent ID: {student_id}")
            file_win.destroy()
        
        tk.Button(
            file_win,
            text="Register",
            command=register,
            bg="#2ecc71",
            fg="white",
            font=("Arial", 12, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            width=15,
            height=2
        ).pack(pady=20)
    
    def open_recognize_window(self):
        recognize_win = tk.Toplevel(self.root)
        recognize_win.title("Recognize Face")
        recognize_win.geometry("500x300")
        recognize_win.configure(bg="#34495e")
        recognize_win.resizable(False, False)
        
        tk.Label(
            recognize_win,
            text="Recognize Face",
            font=("Arial", 16, "bold"),
            bg="#34495e",
            fg="white"
        ).pack(pady=20)
        
        method_frame = tk.Frame(recognize_win, bg="#34495e")
        method_frame.pack(pady=10)
        
        tk.Label(
            method_frame,
            text="Choose Recognition Method:",
            font=("Arial", 11, "bold"),
            bg="#34495e",
            fg="white"
        ).pack(pady=10)
        
        btn_frame = tk.Frame(method_frame, bg="#34495e")
        btn_frame.pack(pady=10)
        
        tk.Button(
            btn_frame,
            text=" Capture from Webcam",
            command=lambda: self.recognize_from_webcam_capture(recognize_win),
            bg="#9b59b6",
            fg="white",
            font=("Arial", 11, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            width=22,
            height=2
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            btn_frame,
            text=" Upload Photo",
            command=lambda: [recognize_win.destroy(), self.recognize_from_image()],
            bg="#3498db",
            fg="white",
            font=("Arial", 11, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            width=22,
            height=2
        ).pack(side=tk.LEFT, padx=10)
    
    def recognize_from_webcam_capture(self, parent_win):
        if len(self.known_face_data) == 0:
            messagebox.showwarning("Warning", "No registered faces. Please register faces first.")
            return
        
        parent_win.destroy()
        
        webcam_win = tk.Toplevel(self.root)
        webcam_win.title("Recognize from Webcam")
        webcam_win.geometry("700x650")
        webcam_win.configure(bg="#34495e")
        
        tk.Label(
            webcam_win,
            text="Capture Face for Recognition",
            font=("Arial", 16, "bold"),
            bg="#34495e",
            fg="white"
        ).pack(pady=10)
        
        video_label = tk.Label(webcam_win, bg="#2c3e50")
        video_label.pack(pady=10)
        
        status_label = tk.Label(
            webcam_win,
            text="Position your face in the frame",
            font=("Arial", 11),
            bg="#34495e",
            fg="#f39c12"
        )
        status_label.pack(pady=5)
        
        result_label = tk.Label(
            webcam_win,
            text="",
            font=("Arial", 13, "bold"),
            bg="#34495e",
            fg="white"
        )
        result_label.pack(pady=5)
        
        btn_frame = tk.Frame(webcam_win, bg="#34495e")
        btn_frame.pack(pady=10)
        
        captured_image = [None]
        webcam_active = [True]
        
        def recognize_captured():
            if captured_image[0] is not None:
                test_features = self.extract_face_features(captured_image[0])
                
                if test_features is None:
                    result_label.config(text="❌ No face detected in captured image", fg="#e74c3c")
                    messagebox.showwarning("No Face", "No face detected. Please try again with better lighting.")
                    return
                
                best_match = None
                best_distance = float('inf')
                
                for name, known_features in self.known_face_data.items():
                    is_match, distance = self.compare_faces(test_features, known_features)
                    if is_match and distance < best_distance:
                        best_match = name
                        best_distance = distance
                
                if best_match:
                    confidence = 100 - best_distance
                    student_id = self.student_ids.get(best_match, "N/A")
                    result_label.config(
                        text=f"✅ Recognized: {best_match} (ID: {student_id})\nConfidence: {confidence:.1f}%",
                        fg="#2ecc71"
                    )
                    self.log(f"✓ Recognized: {best_match} (ID: {student_id}, {confidence:.1f}% confidence)")
                    self.mark_attendance(best_match)
                    messagebox.showinfo("Recognition Success", f"Recognized: {best_match}\nStudent ID: {student_id}\nAttendance marked!")
                else:
                    result_label.config(text="❌ Face not recognized", fg="#e74c3c")
                    self.log("✗ Face not recognized")
                    messagebox.showwarning("Unknown", "Face not recognized in database")
        
        recognize_btn = tk.Button(
            btn_frame,
            text=" Recognize",
            command=recognize_captured,
            bg="#2ecc71",
            fg="white",
            font=("Arial", 11, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            width=15,
            height=2
        )
        recognize_btn.pack(side=tk.LEFT, padx=10)
        
        def close_webcam():
            webcam_active[0] = False
            webcam_win.destroy()
        
        tk.Button(
            btn_frame,
            text=" Close",
            command=close_webcam,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 11, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            width=15,
            height=2
        ).pack(side=tk.LEFT, padx=10)
        
        cap = cv2.VideoCapture(0)
        
        def update_frame():
            if not webcam_active[0]:
                cap.release()
                return
            
            ret, frame = cap.read()
            if ret:
                captured_image[0] = frame.copy()
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Face Detected", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                if len(faces) > 0:
                    status_label.config(text="✓ Face detected! Click Recognize to identify", fg="#2ecc71")
                else:
                    status_label.config(text="⚠ No face detected. Please adjust position", fg="#f39c12")
                
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display_frame = cv2.resize(display_frame, (640, 480))
                img_pil = Image.fromarray(display_frame)
                img_tk = ImageTk.PhotoImage(img_pil)
                
                video_label.config(image=img_tk)
                video_label.image = img_tk
            
            if webcam_active[0]:
                webcam_win.after(10, update_frame)
        
        update_frame()
        
        def on_close():
            webcam_active[0] = False
            cap.release()
            webcam_win.destroy()
        
        webcam_win.protocol("WM_DELETE_WINDOW", on_close)
    
    def recognize_from_image(self):
        if len(self.known_face_data) == 0:
            messagebox.showwarning("Warning", "No registered faces. Please register faces first.")
            return
        
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not image_path:
            return
        
        self.log(f"Processing image: {os.path.basename(image_path)}")
        
        img = cv2.imread(image_path)
        if img is None:
            messagebox.showerror("Error", "Could not read image")
            return
        
        test_features = self.extract_face_features(img)
        if test_features is None:
            messagebox.showwarning("No Face", "No face detected in image")
            self.log("✗ No face detected")
            return
        
        best_match = None
        best_distance = float('inf')
        
        for name, known_features in self.known_face_data.items():
            is_match, distance = self.compare_faces(test_features, known_features)
            if is_match and distance < best_distance:
                best_match = name
                best_distance = distance
        
        # Display result
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            color = (0, 255, 0) if best_match else (255, 0, 0)
            cv2.rectangle(display_img, (x, y), (x+w, y+h), color, 3)
            text = best_match if best_match else "Unknown"
            cv2.putText(display_img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        display_img = cv2.resize(display_img, (640, 480))
        img_pil = Image.fromarray(display_img)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.display_label.config(image=img_tk)
        self.display_label.image = img_tk
        
        if best_match:
            confidence = 100 - best_distance
            student_id = self.student_ids.get(best_match, "N/A")
            self.log(f" Recognized: {best_match} (ID: {student_id}, {confidence:.1f}% confidence)")
            self.mark_attendance(best_match)
            messagebox.showinfo("Recognition Success", f"Recognized: {best_match}\nStudent ID: {student_id}\nAttendance marked!")
        else:
            self.log(" Face not recognized")
            messagebox.showwarning("Unknown", "Face not recognized")
    
    def toggle_webcam(self):
        if not self.is_webcam_running:
            self.start_webcam()
        else:
            self.stop_webcam()
    
    def start_webcam(self):
        if len(self.known_face_data) == 0:
            messagebox.showwarning("Warning", "No registered faces. Please register faces first.")
            return
        
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return
        
        self.is_webcam_running = True
        self.btn_webcam.config(text=" Stop Live Mode", bg="#e74c3c")
        self.marked_today.clear()
        self.log("Live attendance mode started")
        
        threading.Thread(target=self.webcam_loop, daemon=True).start()
    
    def stop_webcam(self):
        self.is_webcam_running = False
        if self.video_capture:
            self.video_capture.release()
        self.btn_webcam.config(text=" Live Attendance Mode", bg="#3498db")
        self.log("Live attendance mode stopped")
    
    def webcam_loop(self):
        frame_count = 0
        while self.is_webcam_running:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            
            frame_count += 1
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            # Process every 10th frame for recognition
            if frame_count % 10 == 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    face_img = frame.copy()
                    test_features = self.extract_face_features(face_img)
                    
                    if test_features:
                        best_match = None
                        best_distance = float('inf')
                        
                        for name, known_features in self.known_face_data.items():
                            is_match, distance = self.compare_faces(test_features, known_features)
                            if is_match and distance < best_distance:
                                best_match = name
                                best_distance = distance
                        
                        if best_match:
                            student_id = self.student_ids.get(best_match, "N/A")
                            cv2.putText(frame, f"{best_match} ({student_id})", (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            if best_match not in self.marked_today:
                                self.mark_attendance(best_match)
                                self.marked_today.add(best_match)
                        else:
                            cv2.putText(frame, "Unknown", (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Just draw rectangles without recognition
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.putText(frame, "Live Attendance Mode - Auto Recognition", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frame = cv2.resize(display_frame, (640, 480))
            img_pil = Image.fromarray(display_frame)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            self.display_label.config(image=img_tk)
            self.display_label.image = img_tk
    
    def view_attendance_window(self):
        if not os.path.exists(self.attendance_file):
            messagebox.showinfo("No Records", "No attendance records found")
            return
        
        df = pd.read_csv(self.attendance_file)
        
        view_win = tk.Toplevel(self.root)
        view_win.title("Attendance Records")
        view_win.geometry("700x500")
        view_win.configure(bg="#34495e")
        
        tk.Label(
            view_win,
            text="Attendance Records",
            font=("Arial", 16, "bold"),
            bg="#34495e",
            fg="white"
        ).pack(pady=10)
        
        # Filter frame
        filter_frame = tk.Frame(view_win, bg="#34495e")
        filter_frame.pack(pady=5)
        
        tk.Label(filter_frame, text="Filter by Date:", font=("Arial", 10), bg="#34495e", fg="white").pack(side=tk.LEFT, padx=5)
        
        date_var = tk.StringVar()
        date_entry = tk.Entry(filter_frame, textvariable=date_var, font=("Arial", 10), width=15)
        date_entry.pack(side=tk.LEFT, padx=5)
        
        def apply_filter():
            date_filter = date_var.get().strip()
            if date_filter:
                filtered_df = df[df['Date'] == date_filter]
            else:
                filtered_df = df
            
            # Clear tree
            for item in tree.get_children():
                tree.delete(item)
            
            # Repopulate
            for _, row in filtered_df.iterrows():
                tree.insert("", tk.END, values=(row['StudentID'], row['Name'], row['Date'], row['Time']))
        
        tk.Button(
            filter_frame,
            text="Apply",
            command=apply_filter,
            bg="#3498db",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            filter_frame,
            text="Show All",
            command=lambda: [date_var.set(""), apply_filter()],
            bg="#95a5a6",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=5)
        
        frame = tk.Frame(view_win, bg="#34495e")
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tree = ttk.Treeview(frame, columns=("StudentID", "Name", "Date", "Time"), show="headings", height=15)
        tree.heading("StudentID", text="Student ID")
        tree.heading("Name", text="Name")
        tree.heading("Date", text="Date")
        tree.heading("Time", text="Time")
        
        tree.column("StudentID", width=120)
        tree.column("Name", width=200)
        tree.column("Date", width=120)
        tree.column("Time", width=100)
        
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        for _, row in df.iterrows():
            tree.insert("", tk.END, values=(row['StudentID'], row['Name'], row['Date'], row['Time']))
        
        tk.Label(
            view_win,
            text=f"Total Records: {len(df)}",
            font=("Arial", 11),
            bg="#34495e",
            fg="white"
        ).pack(pady=10)
        
        # Export button
        def export_to_excel():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
            )
            if file_path:
                if file_path.endswith('.xlsx'):
                    df.to_excel(file_path, index=False)
                else:
                    df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Attendance exported to {file_path}")
        
        tk.Button(
            view_win,
            text=" Export Attendance",
            command=export_to_excel,
            bg="#27ae60",
            fg="white",
            font=("Arial", 11, "bold"),
            relief=tk.FLAT,
            cursor="hand2"
        ).pack(pady=10)
    
    def show_registered_persons(self):
        if not self.known_face_data:
            messagebox.showinfo("No Persons", "No registered persons found")
            return
        
        persons_win = tk.Toplevel(self.root)
        persons_win.title("Registered Persons")
        persons_win.geometry("500x600")
        persons_win.configure(bg="#34495e")
        
        tk.Label(
            persons_win,
            text="Registered Persons",
            font=("Arial", 16, "bold"),
            bg="#34495e",
            fg="white"
        ).pack(pady=10)
        
        frame = tk.Frame(persons_win, bg="#34495e")
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tree = ttk.Treeview(frame, columns=("StudentID", "Name"), show="headings", height=20)
        tree.heading("StudentID", text="Student ID")
        tree.heading("Name", text="Name")
        
        tree.column("StudentID", width=150)
        tree.column("Name", width=300)
        
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        for name in self.known_face_data.keys():
            student_id = self.student_ids.get(name, "N/A")
            tree.insert("", tk.END, values=(student_id, name))
        
        tk.Label(
            persons_win,
            text=f"Total Registered: {len(self.known_face_data)}",
            font=("Arial", 11),
            bg="#34495e",
            fg="white"
        ).pack(pady=10)
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        if self.is_webcam_running:
            self.stop_webcam()
        self.root.destroy()


if __name__ == "__main__":
    app = FaceAttendanceSystem()
    app.run()