Face Attendance System

> Real-time biometric identity verification system — registers faces, recognises them live via webcam, and logs timestamped attendance records with unique student IDs.

**Security relevance:** Implements identity registration, biometric verification, access logging, anomaly flagging (unknown face detection), and exportable audit trails — core principles of enterprise identity management.

---

## What it does

- **Registers** new identities by capturing facial features using ORB (Oriented FAST and Rotated BRIEF) keypoint detection
- **Verifies** identities in real time from a live webcam feed using feature matching
- **Logs** attendance automatically with StudentID, Name, Date, and Time on successful match
- **Flags** unrecognised faces as "Unknown" — separating verified identities from intruders
- **Exports** full attendance records to CSV or Excel for audit purposes
- **Generates** unique hashed StudentIDs (MD5-based) for each registered person — no manual ID assignment

---

## Architecture

```
Face Registration
    └── Capture image → Extract ORB features → Store in known_faces/

Live Recognition Loop (threaded)
    └── Webcam frame → Haar Cascade face detection
        └── ORB feature extraction → Compare against registered faces
            ├── Match found → Log attendance + display name + StudentID
            └── No match → Flag as "Unknown"

Attendance Storage
    └── attendance.csv (StudentID, Name, Date, Time)
    └── students.csv (StudentID, Name, RegisteredDate)
```

---

## Tech stack

- Python · OpenCV (cv2) · Tkinter · Pandas · Pillow
- ORB feature detection · Haar Cascade classifier · MD5 hashing

---

## How to run

```bash
pip install opencv-python numpy pandas pillow
python face.py
```

**Requirements:** Webcam connected · Python 3.8+

---

## Security design notes

| Feature | Implementation |
|---------|---------------|
| Identity registration | ORB keypoint extraction + persistent storage |
| Real-time verification | Feature matching with distance threshold |
| Unknown face handling | Flagged visually, not logged as valid attendance |
| Audit trail | Full CSV log with timestamps, exportable to Excel |
| ID generation | MD5 hash of name + timestamp — collision-resistant unique IDs |

---

## Topics

`face-recognition` `identity-management` `python` `opencv` `security` `biometrics` `attendance-system` `tkinter` `orb-features` `access-logging`
