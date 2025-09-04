# 🎯 Multi-User Face Recognition Implementation Plan

## Overview
Implement AI-powered face recognition to identify registered users and detect unknown users during posture monitoring.

## 🎯 System Requirements

### Core Features
1. **User Registration** → Store face embeddings when users register
2. **Face Detection** → Detect faces in camera feed during monitoring
3. **Face Recognition** → Compare detected faces with stored embeddings
4. **User Identification** → Identify registered users or mark as "Unknown User"
5. **Multi-User Support** → Handle multiple people in camera simultaneously

## 📋 Technical Implementation

### 1. Dependencies Required
```bash
pip install face-recognition opencv-python numpy
```

### 2. Database Schema Updates
```sql
-- Add face embeddings table
CREATE TABLE user_face_embeddings (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    face_encoding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

### 3. Core Face Recognition Module
```python
# face_recognition_module.py
import face_recognition
import cv2
import numpy as np
from typing import List, Tuple, Optional

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
    
    def register_user_face(self, user_id: int, face_image) -> bool:
        """Register a new user's face"""
        try:
            # Detect face in image
            face_locations = face_recognition.face_locations(face_image)
            if not face_locations:
                return False
            
            # Extract face encoding
            face_encoding = face_recognition.face_encodings(face_image, face_locations)[0]
            
            # Store in database
            self.save_face_encoding(user_id, face_encoding)
            return True
        except Exception as e:
            print(f"Error registering face: {e}")
            return False
    
    def identify_users_in_frame(self, frame) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """Identify all users in a frame"""
        # Find all faces in frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        identified_users = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            
            if True in matches:
                # Find the matching user
                match_index = matches.index(True)
                user_name = self.known_face_names[match_index]
                identified_users.append((user_name, face_location))
            else:
                # Unknown user
                identified_users.append(("Unknown User", face_location))
        
        return identified_users
```

### 4. Backend API Endpoints

#### Register User Face
```python
@app.route('/register-face', methods=['POST'])
@login_required
def register_user_face():
    """Register user's face during profile setup"""
    if 'face_image' not in request.files:
        return jsonify({'success': False, 'message': 'No face image provided'})
    
    file = request.files['face_image']
    user_id = session['user']['id']
    
    # Process face image
    success = face_recognition_system.register_user_face(user_id, file)
    
    if success:
        return jsonify({'success': True, 'message': 'Face registered successfully'})
    else:
        return jsonify({'success': False, 'message': 'Failed to register face'})
```

#### Update Video Feed with Face Recognition
```python
def gen_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Identify users in frame
        identified_users = face_recognition_system.identify_users_in_frame(frame)
        
        # Draw identification boxes
        for user_name, face_location in identified_users:
            top, right, bottom, left = face_location
            
            # Draw box around face
            if user_name == "Unknown User":
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for known
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Add label
            cv2.putText(frame, user_name, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Continue with posture detection for identified users
        # ...
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
```

### 5. Frontend Updates

#### Profile Page - Face Registration
```html
<!-- Add to profile.html -->
<div class="face-registration-section">
  <h3>Face Recognition Setup</h3>
  <div class="face-capture-area">
    <video id="face-video" autoplay></video>
    <canvas id="face-canvas" style="display: none;"></canvas>
    <button onclick="captureFace()">Capture Face</button>
  </div>
  <div class="face-preview">
    <img id="captured-face" style="display: none;">
  </div>
</div>
```

#### Dashboard - User Identification Display
```html
<!-- Add to dashboard.html -->
<div class="user-identification">
  <div class="identified-users">
    <h4>Users Detected:</h4>
    <div id="users-list">
      <!-- Dynamically populated -->
    </div>
  </div>
</div>
```

## 🔄 User Experience Flow

### 1. Registration Process
1. User registers account
2. System prompts for face photo
3. Face embedding stored in database
4. User can now be identified during monitoring

### 2. Monitoring Process
1. Camera detects all faces in frame
2. System identifies registered users by name
3. Unknown users marked as "Unknown User"
4. Posture monitoring continues for identified users

### 3. Security Features
- Only registered users can access posture data
- Unknown users are flagged
- System logs unauthorized access attempts

## 🎨 Visual Indicators

### Face Detection Boxes
- **Green Box** → Registered user (shows name)
- **Red Box** → Unknown user (shows "Unknown User")
- **Confidence Score** → Display recognition confidence

### Status Indicators
- **"User: John Doe"** → Identified registered user
- **"Unknown User"** → Unregistered person detected
- **"Multiple Users Detected"** → When several people in frame

## 🔧 Implementation Steps

### Phase 1: Basic Setup
1. Install face recognition dependencies
2. Create database table for face embeddings
3. Create basic face recognition module
4. Test face detection in video feed

### Phase 2: User Registration
1. Add face capture to registration process
2. Implement face embedding storage
3. Create face registration API endpoint
4. Test face registration workflow

### Phase 3: Real-time Recognition
1. Integrate face recognition into video feed
2. Add visual indicators for identified users
3. Implement unknown user detection
4. Test multi-user scenarios

### Phase 4: Security & Polish
1. Add security logging
2. Implement confidence thresholds
3. Add face recognition settings
4. Performance optimization

## 🚀 Benefits

✅ **Multi-User Support** - Multiple people can use same camera  
✅ **Security** - Only registered users can access system  
✅ **Personalization** - Each user gets their own posture data  
✅ **Unknown Detection** - Alerts when unauthorized users detected  
✅ **Privacy** - Face data stored as encrypted embeddings  
✅ **Scalability** - Can handle multiple users simultaneously  

## 📝 Notes

- Face embeddings are mathematical representations, not actual images
- System can handle multiple faces in single frame
- Unknown users are flagged but not blocked (for demo purposes)
- Face recognition accuracy improves with multiple face samples per user
- Consider adding face recognition toggle for privacy concerns

## 🔗 Related Files to Modify

1. `app.py` - Add face recognition endpoints
2. `db.py` - Add face embedding database functions
3. `templates/profile.html` - Add face registration UI
4. `templates/dashboard.html` - Add user identification display
5. `static/js/` - Add face capture JavaScript

---
*This plan provides a complete roadmap for implementing multi-user face recognition in the PosturEase system.* 