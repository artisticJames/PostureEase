from flask import Flask, render_template, request, redirect, url_for, flash, session, Response, jsonify, send_file
import os
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time
from config import Config
from db import create_user, verify_user, save_posture_record, get_user_posture_history, get_exercises, record_completed_exercise, update_user, get_all_users, admin_create_user, delete_user, get_db_connection, test_connection, save_verification_token, verify_email_token, verify_password_change_token, clear_password_change_token, get_user_by_email, clear_user_posture_history, delete_posture_record
from datetime import datetime, timedelta
from functools import wraps
import logging
import re
from werkzeug.security import generate_password_hash, check_password_hash
import bcrypt
# Simplified to single-person posture monitoring
from email_service import email_service
import io
import csv
import zipfile
import google.generativeai as genai
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)


# Ensure the secret key is set
if not app.secret_key:
    app.secret_key = os.urandom(24)

# Pending registrations cache: token -> { data: {...}, expires_at: datetime }
PENDING_REGISTRATIONS = {}
# Pending password changes: token -> { user_id, new_password_hash, expires_at }
PENDING_PASSWORD_CHANGES = {}

# Serve favicon to avoid browser 404 requests
@app.route('/favicon.ico')
def favicon():
    icon_path = os.path.join(app.root_path, 'static', 'images', 'logo-test-2-1-10.png')
    return send_file(icon_path, mimetype='image/png')

# Configure Gemini once on startup if API key is provided
if Config.GEMINI_API_KEY:
    try:
        genai.configure(api_key=Config.GEMINI_API_KEY)
    except Exception as _e:
        logger.warning("Failed to configure Gemini API; proceeding without AI tips")

 

# Global state for real-time posture data

# Global state for real-time posture data
CURRENT_POSTURE_DATA = {
    'posture_quality': 'unknown',
    'confidence': 0.0,
    'timestamp': None,
    'last_update': None
}

# Exercise recommendation system
def generate_exercise_recommendations(current_record, history_records):
    """
    Generate personalized exercise recommendations using Gemini AI for dynamic suggestions.
    
    Args:
        current_record: Current posture record being analyzed
        history_records: List of all user's posture history records
    
    Returns:
        List of recommended exercise names
    """
    # Try Gemini AI first for dynamic recommendations
    if Config.GEMINI_API_KEY:
        try:
            # Prepare data for Gemini analysis
            posture_type = current_record.get('posture_type', 'Unknown')
            confidence_score = current_record.get('confidence_score', 0)
            session_duration = current_record.get('session_duration', 0)
            bad_time = current_record.get('bad_time', 0)
            corrections_count = current_record.get('corrections_count', 0)
            
            # Calculate session stats
            good_time = session_duration - bad_time if session_duration > bad_time else 0
            bad_percentage = (bad_time / session_duration * 100) if session_duration > 0 else 0
            
            # Create prompt for Gemini
            prompt = f"""
            You are an expert ergonomics coach. Based on this specific posture session data, suggest 3-5 personalized exercises.
            
            SESSION DATA:
            - Posture Type: {posture_type}
            - Confidence Score: {confidence_score:.2f}
            - Session Duration: {session_duration/60:.1f} minutes
            - Good Posture Time: {good_time/60:.1f} minutes
            - Bad Posture Time: {bad_time/60:.1f} minutes
            - Bad Posture Percentage: {bad_percentage:.1f}%
            - Corrections Needed: {corrections_count}
            
            Please suggest 3-5 specific, actionable exercises that address the issues shown in this session.
            Focus on exercises that can be done at a desk/office environment.
            Return ONLY the exercise names, one per line, no explanations.
            """
            
            # Get Gemini AI recommendations
            gemini_response = _gemini_generate_text('gemini-2.5-flash', prompt)
            
            if gemini_response and gemini_response.strip():
                # Parse Gemini response into exercise list
                exercises = [ex.strip() for ex in gemini_response.strip().split('\n') if ex.strip()]
                if exercises:
                    return exercises[:5]  # Limit to 5 exercises
                    
        except Exception as e:
            logger.warning(f"Gemini exercise recommendation failed: {e}")
    
    # Fallback to rule-based system if Gemini fails
    recommendations = []
    
    # Analyze current record
    posture_type = current_record.get('posture_type', '')
    confidence_score = current_record.get('confidence_score', 0)
    session_duration = current_record.get('session_duration', 0)
    bad_time = current_record.get('bad_time', 0)
    corrections_count = current_record.get('corrections_count', 0)
    
    # Analyze historical patterns
    total_bad_time = 0
    total_sessions = len(history_records)
    frequent_corrections = 0
    long_sessions = 0
    
    for record in history_records:
        if record.get('bad_time'):
            total_bad_time += record.get('bad_time', 0)
        if record.get('corrections_count', 0) > 5:
            frequent_corrections += 1
        if record.get('session_duration', 0) > 3600:  # > 1 hour
            long_sessions += 1
    
    # Calculate patterns
    avg_bad_time = total_bad_time / total_sessions if total_sessions > 0 else 0
    correction_frequency = frequent_corrections / total_sessions if total_sessions > 0 else 0
    long_session_frequency = long_sessions / total_sessions if total_sessions > 0 else 0
    
    # Recommendation logic based on patterns
    
    # 1. For poor posture confidence or high bad time
    if confidence_score < 0.7 or bad_time > session_duration * 0.3:
        recommendations.extend([
            "Chest Opener",  # Counteract rounded shoulders
            "Cat-Cow",  # Improve spine flexibility
            "Standing Cat-Cow"  # Standing version for office workers
        ])
    
    # 2. For frequent corrections (posture instability)
    if corrections_count > 3 or correction_frequency > 0.3:
        recommendations.extend([
            "High Plank",  # Core stability
            "Glute Bridge",  # Lower back stability
            "Isometric Pull Ups"  # Upper back strength
        ])
    
    # 3. For long sessions (sitting/standing for extended periods)
    if session_duration > 1800 or long_session_frequency > 0.4:  # > 30 min or frequent long sessions
        recommendations.extend([
            "Forward fold",  # Decompress spine
            "Child's pose",  # Gentle rest
            "Downward-Facing Dog"  # Full body stretch
        ])
    
    # 4. For high bad time percentage (chronic poor posture)
    if avg_bad_time > 600:  # > 10 minutes average bad time
        recommendations.extend([
            "Side Plank",  # Lateral stability
            "Pigeon Pose",  # Hip flexor release
            "Quadruped Thoracic Rotation"  # Upper back mobility
        ])
    
    # 5. For good posture, recommend maintenance exercises
    if 'Good' in posture_type and confidence_score > 0.8:
        recommendations.extend([
            "Chest Opener",  # Maintain good posture
            "Cat-Cow"  # Keep spine flexible
        ])
    
    # 6. For very long sessions (> 1 hour), prioritize decompression
    if session_duration > 3600:  # > 1 hour
        recommendations.extend([
            "Forward fold",  # Decompress spine
            "Child's pose",  # Gentle rest
            "Downward-Facing Dog"  # Full body stretch
        ])
    
    # 7. Always include at least one general recommendation if no specific issues
    if not recommendations:
        recommendations = ["Chest Opener", "Cat-Cow"]
    
    # Remove duplicates and limit to 3-4 recommendations
    unique_recommendations = list(dict.fromkeys(recommendations))  # Remove duplicates while preserving order
    return unique_recommendations[:4]

def _gemini_generate_text(_model_name_unused, prompt, *, response_mime_type=None):
    """Call Gemini 2.5 Flash and return text. No discovery or fallbacks."""
    generation_config = {}
    if response_mime_type:
        generation_config['response_mime_type'] = response_mime_type
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        resp = model.generate_content(
            prompt,
            generation_config=generation_config or None,
        )
    except Exception as e:
        raise e

    text = (getattr(resp, 'text', None) or '').strip()
    if text:
        return text
    try:
        candidates = getattr(resp, 'candidates', []) or []
        parts_text = []
        for cand in candidates:
            content = getattr(cand, 'content', None)
            parts = getattr(content, 'parts', []) if content else []
            for p in parts:
                t = getattr(p, 'text', None)
                if isinstance(t, str) and t.strip():
                    parts_text.append(t.strip())
        return "\n".join(parts_text).strip()
    except Exception:
        return ''

# Test database connection on startup
if not test_connection():
    logger.error("Failed to establish initial database connection. The application may not function correctly.")
else:
    logger.info("Initial database connection successful")

@app.before_request
def before_request():
    """
    Check database connection before each request
    """
    if not test_connection():
        logger.error("Database connection lost before request")
        return jsonify({'success': False, 'message': 'Database connection error'}), 500

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def calculate_angle(a, b, c):
    """Calculate the angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def classify_posture(landmarks, image_shape):
    """Classify posture using rule-based logic (ML classifier removed)."""
    
    if not landmarks:
        return "Unknown", 0.0
    
    # Convert landmarks to pixel coordinates
    h, w = image_shape[:2]
    points = {}
    for idx, landmark in enumerate(landmarks.landmark):
        points[idx] = (int(landmark.x * w), int(landmark.y * h))
    
    try:
        # Calculate multiple angles for better accuracy
        
        # 1. Neck angle (ear through shoulder to hip)
        neck_angle = calculate_angle(
            points[mp_pose.PoseLandmark.RIGHT_EAR.value],
            points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            points[mp_pose.PoseLandmark.RIGHT_HIP.value]
        )
        
        # 2. Upper back angle (shoulder through hip to knee)
        upper_back_angle = calculate_angle(
            points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            points[mp_pose.PoseLandmark.RIGHT_HIP.value],
            points[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        )
        
        # 3. Head tilt (ear to ear through nose)
        head_tilt = calculate_angle(
            points[mp_pose.PoseLandmark.LEFT_EAR.value],
            points[mp_pose.PoseLandmark.NOSE.value],
            points[mp_pose.PoseLandmark.RIGHT_EAR.value]
        )
        
        # 4. Shoulder alignment (left to right shoulder)
        shoulder_angle = calculate_angle(
            points[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            (points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0] + 100, 
             points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1])
        )


        # Calculate confidence scores for each metric
        neck_score = 1.0 - min(abs(170 - neck_angle) / 40.0, 1.0)
        back_score = 1.0 - min(abs(175 - upper_back_angle) / 40.0, 1.0)
        head_score = 1.0 - min(abs(180 - head_tilt) / 30.0, 1.0)
        shoulder_score = 1.0 - min(abs(180 - shoulder_angle) / 30.0, 1.0)
        
        # Weight the scores
        weights = {
            'neck': 0.40,
            'back': 0.40,
            'head': 0.10,
            'shoulder': 0.10
        }
        
        # Calculate weighted confidence score
        confidence = (
            (neck_score * weights['neck']) +
            (back_score * weights['back']) +
            (head_score * weights['head']) +
            (shoulder_score * weights['shoulder'])
        )
        
        # Clamp between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        # Simplified classification (just good/bad)
        good_thresh = 0.7  # Fixed threshold
        posture = "Good Posture" if confidence > good_thresh else "Bad Posture"
        
        return posture, confidence
            
    except KeyError:
        return "Unknown", 0.0

# Load YOLO and MediaPipe
if 'yolo_model' not in globals():
    # Prefer project classifier weights; gracefully fall back to classifier base, then detector
    posture_model = None
    USE_CLASSIFIER = False
    try:
        # 1) Try project-trained classifier (best.pt)
        _candidate = YOLO(os.path.join('Training Again', 'runs', 'classify', 'train3', 'weights', 'best.pt'))
        _dummy = np.zeros((32, 32, 3), dtype=np.uint8)
        _res = _candidate(_dummy, verbose=False)
        # Classifier outputs should expose probs on results
        _has_probs = False
        for _r in _res:
            if hasattr(_r, 'probs') and _r.probs is not None:
                _has_probs = True
                break
        if _has_probs:
            posture_model = _candidate
            USE_CLASSIFIER = True
            logger.info("Loaded posture classifier from Training Again/runs/classify/train3/weights/best.pt")
        else:
            raise RuntimeError('best.pt did not return classification probabilities')
    except Exception as _e1:
        try:
            # 2) Fallback to yolov8n-cls.pt
            _candidate2 = YOLO('yolov8n-cls.pt')
            _dummy2 = np.zeros((32, 32, 3), dtype=np.uint8)
            _res2 = _candidate2(_dummy2, verbose=False)
            _has_probs2 = False
            for _r in _res2:
                if hasattr(_r, 'probs') and _r.probs is not None:
                    _has_probs2 = True
                    break
            if _has_probs2:
                posture_model = _candidate2
                USE_CLASSIFIER = True
                logger.info("Loaded posture classifier from yolov8n-cls.pt")
            else:
                raise RuntimeError('yolov8n-cls.pt did not return classification probabilities')
        except Exception as _e2:
            # 3) Revert to detector for person detection
            yolo_model = YOLO('yolov8n.pt')
            USE_CLASSIFIER = False
            logger.info("Classifier unavailable; reverted to yolov8n.pt detector")
# Removed MyPostureDetection integration (classifier-only path)
if 'mp_pose' not in globals():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.3,  # Further lowered for better side/back detection
        min_tracking_confidence=0.3   # Further lowered for better side/back detection
    )
    mp_drawing = mp.solutions.drawing_utils

def draw_posture_status(image, prediction, conf):
    """Draw posture status label on the image."""
    text = f"Posture: {prediction} ({conf:.2f})"
    color = (0, 255, 0) if (isinstance(prediction, str) and ("Good" in prediction or "good" in prediction)) else (0, 0, 255)
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def detect_available_cameras():
    """Detect all available cameras and return their indices."""
    available_cameras = []
    for i in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def get_camera_index():
    """Get the best camera index to use."""
    # Check if specific camera index is configured
    config_camera = app.config.get('CAMERA_INDEX', -1)
    if config_camera >= 0:
        logger.info(f"Using configured camera index: {config_camera}")
        return config_camera
    
    # Auto-detect available cameras
    available_cameras = detect_available_cameras()
    logger.info(f"Available cameras: {available_cameras}")
    
    if not available_cameras:
        logger.warning("No cameras detected, using default index 0")
        return 0
    
    # Prefer external cameras (higher indices) over built-in (usually index 0)
    # Sort in descending order to prefer higher indices first
    available_cameras.sort(reverse=True)
    
    selected_camera = available_cameras[0]
    logger.info(f"Auto-selected camera index: {selected_camera}")
    return selected_camera

def gen_frames():
    camera_index = get_camera_index()
    cap = cv2.VideoCapture(camera_index)
    last_save_time = 0
    save_interval = 30  # Save posture data every 30 seconds
    current_session_data = {
        'start_time': time.time(),
        'posture_counts': {'good': 0, 'bad': 0},
        'total_frames': 0,
        'last_posture': None
    }
    
    # Initialize global alarm flag
    gen_frames.alarm_triggered = False
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        current_session_data['total_frames'] += 1

        if 'USE_CLASSIFIER' in globals() and USE_CLASSIFIER:
            # Classifier-driven path (no person detector). Use whole frame like posture_detection_simple.py
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(image_rgb)

            if results_pose.pose_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )


                # Run classifier on the RGB frame
                try:
                    cls_results = posture_model(image_rgb, verbose=False)
                    class_idx = cls_results[0].probs.top1
                    class_name = cls_results[0].names[class_idx]
                    confidence = float(cls_results[0].probs.top1conf)
                except Exception:
                    # If classifier inference fails, mark unknown
                    class_name = 'Unknown'
                    confidence = 0.0

                # Overlay posture label
                draw_posture_status(frame, class_name, confidence)

                # Update global posture data
                posture_quality = 'good' if ('Good' in str(class_name)) else 'bad'
                if posture_quality == 'good':
                    current_session_data['posture_counts']['good'] += 1
                else:
                    current_session_data['posture_counts']['bad'] += 1
                    gen_frames.alarm_triggered = True

                globals()['CURRENT_POSTURE_DATA'] = {
                    'posture_quality': posture_quality,
                    'confidence': confidence,
                    'timestamp': current_time,
                    'last_update': current_time
                }

                current_session_data['last_posture'] = class_name
            else:
                # No landmarks found
                globals()['CURRENT_POSTURE_DATA'] = {
                    'posture_quality': 'unknown',
                    'confidence': 0.0,
                    'timestamp': current_time,
                    'last_update': current_time
                }
        else:
            # Original detector + rule-based posture classification path
            results = yolo_model(frame, verbose=False)
            persons = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Only process if it's a person (class 0)
                    if int(box.cls) == 0 and box.conf[0] > 0.5:  # Lowered for better side/back detection
                        persons.append(box.xyxy[0].cpu().numpy())

            # Update posture data when no person is detected
            if not persons:
                current_time = time.time()
                globals()['CURRENT_POSTURE_DATA'] = {
                    'posture_quality': 'unknown',
                    'confidence': 0.0,
                    'timestamp': current_time,
                    'last_update': current_time
                }

            # Process only a single detected person (first match)
            if persons:
                person_box = persons[0]
                x1, y1, x2, y2 = map(int, person_box)

                # Ensure the box coordinates are within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                # Skip if box is too small
                if x2 - x1 >= 50 and y2 - y1 >= 50:
                    person_img = frame[y1:y2, x1:x2]
                    if person_img.size != 0:
                        person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                        landmark_results = pose.process(person_img_rgb)

                        if landmark_results.pose_landmarks:
                            # Draw landmarks on the person image
                            mp_drawing.draw_landmarks(
                                person_img,
                                landmark_results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                            )


                            # Rule-based posture classification
                            posture, confidence = classify_posture(landmark_results.pose_landmarks, person_img.shape)

                            # Update global posture data for real-time access
                            current_time = time.time()

                            # Determine posture quality for the indicator
                            if "Good" in posture:
                                posture_quality = 'good'
                                current_session_data['posture_counts']['good'] += 1
                            else:
                                posture_quality = 'bad'
                                current_session_data['posture_counts']['bad'] += 1
                                # Trigger alarm for bad posture
                                gen_frames.alarm_triggered = True

                            # Update global posture data
                            globals()['CURRENT_POSTURE_DATA'] = {
                                'posture_quality': posture_quality,
                                'confidence': confidence,
                                'timestamp': current_time,
                                'last_update': current_time
                            }

                            current_session_data['last_posture'] = posture

                            # Draw a compact label on the frame for visibility
                            draw_posture_status(frame, posture, float(confidence))
        
        # Save posture data periodically
        if current_time - last_save_time >= save_interval and current_session_data['total_frames'] > 0:
            # Calculate session statistics
            total_postures = current_session_data['posture_counts']['good'] + current_session_data['posture_counts']['bad']
            if total_postures > 0:
                good_percentage = (current_session_data['posture_counts']['good'] / total_postures) * 100
                session_duration = current_time - current_session_data['start_time']
                
                # Determine overall posture status for this session
                session_good_pct = app.config.get('SESSION_GOOD_THRESHOLD', 0.8) * 100
                session_fair_pct = app.config.get('SESSION_FAIR_THRESHOLD', 0.6) * 100
                if good_percentage >= session_good_pct:
                    overall_posture = "Good Posture"
                elif good_percentage >= session_fair_pct:
                    overall_posture = "Fair Posture"
                else:
                    overall_posture = "Poor Posture"
                
                # Store session data for later saving (avoid session context issues)
                # We'll save this data when the user stops monitoring
                # Calculate corrections as transitions from bad to good posture
                corrections_estimate = max(0, current_session_data['posture_counts']['bad'] // 10)  # Estimate: 1 correction per 10 bad frames
                current_session_data['pending_save'] = {
                    'posture_type': overall_posture,
                    'confidence_score': good_percentage / 100.0,
                    'session_duration': session_duration,
                    'corrections_count': corrections_estimate
                }
                
                # Reset session data
                current_session_data = {
                    'start_time': current_time,
                    'posture_counts': {'good': 0, 'bad': 0},
                    'total_frames': 0,
                    'last_posture': None
                }
                last_save_time = current_time

        # Flip frame horizontally to mirror (selfie view) similar to simple script
        if app.config.get('CAMERA_MIRROR', True):
            frame = cv2.flip(frame, 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

# Routes
@app.route('/')
def index():
    # If not logged in, redirect to login
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # If already logged in and this is a POST from fetch(), return JSON instead of redirecting HTML
    if request.method == 'POST' and 'user' in session:
        user = session['user']
        redirect_url = url_for('admin') if user.get('username') == 'admin' else url_for('index')
        return jsonify({
            'success': True,
            'user': user,
            'redirect': redirect_url
        })

    # For GET requests, keep existing redirect behavior
    if 'user' in session:
        if session['user'].get('username') == 'admin':
            return redirect(url_for('admin'))
        return redirect(url_for('index'))

    if request.method == 'POST':
        # Accept JSON or form-encoded submissions
        data = request.get_json(silent=True)
        if not data:
            data = request.form.to_dict() if request.form else None
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            })
            
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({
                'success': False,
                'message': 'Username and password are required'
            })
        
        success, result = verify_user(username, password)
        
        if success:
            # Store complete user data in session
            session['user'] = result
            session.permanent = True
            
            # Check if user is admin and set appropriate redirect
            redirect_url = url_for('admin') if username == 'admin' else url_for('index')
            
            return jsonify({
                'success': True,
                'user': result,  # Send complete user data
                'redirect': redirect_url
            })
        else:
            return jsonify({
                'success': False,
                'message': result
            })
    return render_template('login.html')

@app.route('/create-account', methods=['GET', 'POST'])
def create_account():
    # If user is already logged in, redirect to appropriate page
    if 'user' in session:
        if session['user'].get('username') == 'admin':
            return redirect(url_for('admin'))
        return redirect(url_for('index'))

    if request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            })
        
        # Format birth date
        try:
            birth_date = f"{data.get('birthYear')}-{data.get('birthMonth')}-{data.get('birthDay')}"
        except Exception:
            return jsonify({'success': False, 'message': 'Invalid birth date format'})

        # Generate OTP and cache pending registration (do not save user yet)
        from db import generate_otp, save_otp
        otp_code = generate_otp()
        expires_at = datetime.now() + timedelta(hours=24)
        PENDING_REGISTRATIONS[otp_code] = {
            'data': {
                'username': data.get('username'),
                'email': data.get('email'),
                'password': data.get('password'),
                'first_name': data.get('firstName'),
                'last_name': data.get('lastName'),
                'birth_date': birth_date,
                'gender': data.get('gender')
            },
            'expires_at': expires_at
        }

        # Save OTP to database
        otp_saved, otp_result = save_otp(data.get('email'), otp_code, 'registration')
        if not otp_saved:
            return jsonify({
                'success': False,
                'message': 'Failed to generate verification code. Please try again.'
            })

        # Send OTP verification email
        email_sent = email_service.send_otp_email(
            data.get('email'),
            otp_code,
            'registration'
        )

        if email_sent:
            return jsonify({
                'success': True,
                'message': 'We sent a verification code to your email. Enter it to complete sign up.',
                'redirect': url_for('verify_otp', type='registration', email=data.get('email')),
                'redirect_url': url_for('verify_otp', type='registration', email=data.get('email')),
                'token': otp_code,
                'username': data.get('username'),
                'email': data.get('email'),
                'base_url': Config.BASE_URL
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to send verification email. Please try again.',
                'detail': getattr(email_service, 'last_error', None),
                'token': otp_code,
                'username': data.get('username'),
                'email': data.get('email'),
                'base_url': Config.BASE_URL
            })
    return render_template('create-account.html')

@app.route('/profile')
@login_required
def profile():
    # Get complete user data from database
    user_id = session['user'].get('id')
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, username, email, first_name, last_name, 
                       birth_date, gender, profile_picture, created_at, updated_at
                FROM users 
                WHERE id = %s
            """, (user_id,))
            user_data = cursor.fetchone()
            
            if user_data:
                # Update session with fresh data from database
                session['user'].update({
                    'id': user_data['id'],
                    'username': user_data['username'],
                    'email': user_data['email'],
                    'first_name': user_data['first_name'],
                    'last_name': user_data['last_name'],
                    'birth_date': user_data['birth_date'],
                    'gender': user_data['gender'],
                    'profile_picture': user_data['profile_picture']
                })
                
                return render_template('profile.html', 
                                    user=user_data,
                                    profile_picture=user_data['profile_picture'])
            else:
                logger.error(f"User not found in database: {user_id}")
                return redirect(url_for('login'))
                
        except Exception as e:
            logger.error(f"Error fetching user profile data: {e}")
            return render_template('profile.html', 
                                user=session['user'],
                                profile_picture=None)
        finally:
            conn.close()
    else:
        logger.error("Database connection failed")
        return render_template('profile.html', 
                            user=session['user'],
                            profile_picture=None)

@app.route('/exercises')
@login_required
def exercises():
    success, exercises_list = get_exercises()
    return render_template('exercises.html', exercises=exercises_list if success else [])

@app.route('/posture-history')
@login_required
def posture_history():
    user_id = session['user']['id']
    
    # Check cache first (cache for 30 seconds)
    cache_key = f"posture_history_{user_id}"
    cached_data = getattr(app, 'history_cache', {}).get(cache_key)
    if cached_data and (time.time() - cached_data.get('timestamp', 0)) < 30:
        logger.info(f"Returning cached posture history for user {user_id}")
        return render_template('posture-history.html', 
                             history=cached_data['history'], 
                             stats=cached_data['stats'],
                             history_data=cached_data['history_data'])
    
    success, history = get_user_posture_history(user_id, limit=100)
    
    # Calculate statistics
    total_records = len(history) if success else 0
    good_posture_count = 0
    bad_posture_count = 0
    total_confidence = 0
    total_duration = 0
    total_good_time = 0
    total_bad_time = 0
    
    if success and history:
        for record in history:
            if 'Good' in record.get('posture_type', ''):
                good_posture_count += 1
            else:
                bad_posture_count += 1
            total_confidence += record.get('confidence_score', 0)
            session_duration = record.get('session_duration', 0)
            confidence_score = record.get('confidence_score', 0)
            good_time = record.get('good_time', 0)
            bad_time = record.get('bad_time', 0)
            
            total_duration += session_duration
            # Use actual tracked times from database, fallback to calculated if not available
            if good_time > 0 or bad_time > 0:
                total_good_time += good_time
                total_bad_time += bad_time
            else:
                # Fallback to old calculation for backward compatibility
                total_good_time += session_duration * confidence_score
                total_bad_time += session_duration * (1 - confidence_score)
    
    # Calculate percentages and averages
    good_posture_percentage = (good_posture_count / total_records * 100) if total_records > 0 else 0
    bad_posture_percentage = (bad_posture_count / total_records * 100) if total_records > 0 else 0
    avg_confidence = (total_confidence / total_records * 100) if total_records > 0 else 0
    total_hours = total_duration / 3600 if total_duration else 0
    good_hours = total_good_time / 3600 if total_good_time else 0
    bad_hours = total_bad_time / 3600 if total_bad_time else 0
    
    stats = {
        'total_records': total_records,
        'good_posture_percentage': round(good_posture_percentage, 1),
        'bad_posture_percentage': round(bad_posture_percentage, 1),
        'total_hours': round(total_hours, 1),
        'good_hours': round(good_hours, 1),
        'bad_hours': round(bad_hours, 1),
        'avg_confidence': round(avg_confidence, 1),
        'health_score': round((good_posture_percentage + avg_confidence) / 2, 1)
    }
    
    # Prepare history data for JavaScript with personalized exercise recommendations
    history_data = {}
    if success and history:
        for i, record in enumerate(history, 1):
            confidence_pct = (record.get('confidence_score', 0) * 100) if record.get('confidence_score') else 0
            duration_min = (record.get('session_duration', 30) / 60) if record.get('session_duration') else 30
            
            # Simplified descriptions without AI recommendations for better performance
            if 'Good' in record.get('posture_type', ''):
                description = f"Excellent posture maintenance during this session! Your alignment was optimal with a confidence score of {confidence_pct:.1f}%. Keep up the good work!"
            else:
                description = f"Your posture during this session showed room for improvement with a confidence score of {confidence_pct:.1f}%. Consider doing some posture exercises to improve your alignment."
            
            description += f"<br>Session duration: {duration_min:.1f} minutes."
            
            history_data[str(i)] = {
                'title': f"Posture Analysis - {record.get('recorded_at', 'Session')}",
                'description': description,
                'recommended_exercises': []  # Empty array - recommendations removed for performance
            }
    
    # Cache the results
    if not hasattr(app, 'history_cache'):
        app.history_cache = {}
    
    app.history_cache[cache_key] = {
        'history': history if success else [],
        'stats': stats,
        'history_data': history_data,
        'timestamp': time.time()
    }
    
    return render_template('posture-history.html', 
                         history=history if success else [], 
                         stats=stats,
                         history_data=history_data)

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route('/get-exercise-recommendations/<int:record_id>')
@login_required
def get_exercise_recommendations(record_id):
    """Get personalized exercise recommendations for a specific posture record"""
    try:
        # Get the specific record
        success, history = get_user_posture_history(session['user']['id'], limit=1000)
        if not success:
            return jsonify({'success': False, 'message': 'Failed to retrieve posture history'})
        
        # Find the specific record
        target_record = None
        for record in history:
            if record.get('id') == record_id:
                target_record = record
                break
        
        if not target_record:
            return jsonify({'success': False, 'message': 'Record not found'})
        
        # Generate recommendations
        recommendations = generate_exercise_recommendations(target_record, history)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'record_info': {
                'posture_type': target_record.get('posture_type'),
                'confidence_score': target_record.get('confidence_score'),
                'session_duration': target_record.get('session_duration'),
                'bad_time': target_record.get('bad_time'),
                'corrections_count': target_record.get('corrections_count')
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting exercise recommendations: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/admin')
@login_required
def admin():
    if session['user'].get('username') != 'admin':
        return redirect(url_for('index'))
    return render_template('admin.html')

@app.route('/change-password', methods=['GET', 'POST'])
def change_password():
    """Handle password change requests for both logged-in and non-logged-in users"""
    if request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'})

        current_password = data.get('currentPassword')
        new_password = data.get('newPassword')
        username = data.get('username')
        method = data.get('method', 'immediate')  # Default to immediate method
        
        if not current_password or not new_password:
            return jsonify({'success': False, 'message': 'Missing password data'})

        # Get user from session or try to find by username
        user_id = session.get('user', {}).get('id')

        # Connect to database
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'Database connection failed'})

        try:
            cursor = conn.cursor(dictionary=True)
            
            # If user is not logged in, try to find by username
            if not user_id and username:
                cursor.execute("SELECT id, password FROM users WHERE username = %s", (username,))
                user = cursor.fetchone()
                if user:
                    user_id = user['id']
            else:
                # Get current user data
                cursor.execute("SELECT id, password FROM users WHERE id = %s", (user_id,))
                user = cursor.fetchone()
            
            if not user:
                return jsonify({'success': False, 'message': 'User not found'})

            # Verify current password using bcrypt (supports bcrypt-stored hashes)
            stored_hash = user['password']
            if isinstance(stored_hash, bytes):
                stored_hash_str = stored_hash.decode('utf-8')
            else:
                stored_hash_str = stored_hash
            if not bcrypt.checkpw(current_password.encode('utf-8'), stored_hash_str.encode('utf-8')):
                return jsonify({'success': False, 'message': 'Current password is incorrect'})

            # Validate new password
            if len(new_password) < 8:
                return jsonify({'success': False, 'message': 'Password must be at least 8 characters long'})
            
            if not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$', new_password):
                return jsonify({'success': False, 'message': 'Password must include uppercase, lowercase, numbers, and special characters'})

            # Use OTP verification method
            from db import generate_otp, save_otp
            otp_code = generate_otp()
            expires_at = datetime.now() + timedelta(hours=1)
            
            # Pre-hash and store pending password change in memory
            new_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            PENDING_PASSWORD_CHANGES[otp_code] = {
                'user_id': user_id,
                'new_password_hash': new_hash,
                'expires_at': expires_at
            }

            # Get user email for OTP
            cursor.execute("SELECT email, username FROM users WHERE id = %s", (user_id,))
            user_info = cursor.fetchone()
            if not user_info:
                return jsonify({'success': False, 'message': 'User information not found.'})

            # Save OTP to database
            otp_saved, otp_result = save_otp(user_info['email'], otp_code, 'password_change')
            if not otp_saved:
                return jsonify({
                    'success': False,
                    'message': 'Failed to generate verification code. Please try again.'
                })

            # Send OTP verification email
            email_sent = email_service.send_otp_email(
                user_info['email'],
                otp_code,
                'password_change'
            )
            if email_sent:
                return jsonify({
                    'success': True,
                    'message': 'We sent a verification code to your email. Enter it to complete the password change.',
                    'requires_verification': True,
                    'redirect': url_for('verify_otp', type='password_change', email=user_info['email']),
                    'redirect_url': url_for('verify_otp', type='password_change', email=user_info['email'])
                })
            else:
                return jsonify({'success': False, 'message': 'Failed to send verification email. Please try again.'})

        except Exception as e:
            logger.error(f"Error changing password: {e}")
            return jsonify({'success': False, 'message': 'An error occurred'})
        finally:
            if conn:
                conn.close()

    # For GET requests, just render the template
    return render_template('change-password.html')

@app.route('/verify-email')
def verify_email():
    """Verify email address using token"""
    token = request.args.get('token')
    
    if not token:
        flash('Invalid verification link', 'error')
        return redirect(url_for('login'))
    
    success, result = verify_email_token(token)
    
    if success:
        flash('Email verified successfully! You can now log in.', 'success')
        return redirect(url_for('login'))
    else:
        flash(f'Email verification failed: {result}', 'error')
        return redirect(url_for('login'))

@app.route('/verify-email-code', methods=['POST'])
def verify_email_code():
    """Verify email using a code (token) submitted from the UI.
    If the code matches a pending registration, create the user now.
    """
    data = request.get_json(silent=True) or {}
    code = data.get('code') or data.get('token')
    email = data.get('email')
    if not code:
        return jsonify({'success': False, 'message': 'Verification code is required'})

    # First, handle pending registrations created during sign-up
    pending = PENDING_REGISTRATIONS.get(code)
    if pending:
        if pending['expires_at'] < datetime.now():
            del PENDING_REGISTRATIONS[code]
            return jsonify({'success': False, 'message': 'Verification code expired. Please sign up again.'})
        reg = pending['data']
        success, result = create_user(
            username=reg['username'],
            email=reg['email'],
            password=reg['password'],
            first_name=reg['first_name'],
            last_name=reg['last_name'],
            birth_date=reg['birth_date'],
            gender=reg['gender']
        )
        if success:
            del PENDING_REGISTRATIONS[code]
            return jsonify({'success': True, 'message': 'Email verified and account created. You can now log in.'})
        else:
            return jsonify({'success': False, 'message': result or 'Failed to create account.'})

    # Try OTP verification if email is provided
    if email:
        from db import verify_otp as db_verify_otp
        success, result = db_verify_otp(email, code, 'registration')
        if success:
            return jsonify({'success': True, 'message': 'Email verified successfully. You can now log in.'})

    # Fallback to legacy stored tokens (if any)
    success, result = verify_email_token(code)
    if success:
        return jsonify({'success': True, 'message': 'Email verified successfully. You can now log in.'})
    return jsonify({'success': False, 'message': 'Invalid or expired code.'})

@app.route('/resend-verification', methods=['POST'])
def resend_verification():
    """Resend verification email for pending registration"""
    data = request.get_json(silent=True) or {}
    email = data.get('email')
    username = data.get('username')
    token = data.get('token')
    
    if not email or not username or not token:
        return jsonify({'success': False, 'message': 'Missing required information'})
    
    # Check if the token exists in pending registrations
    pending = PENDING_REGISTRATIONS.get(token)
    if not pending:
        return jsonify({'success': False, 'message': 'Invalid or expired verification request'})
    
    if pending['expires_at'] < datetime.now():
        del PENDING_REGISTRATIONS[token]
        return jsonify({'success': False, 'message': 'Verification code expired. Please sign up again.'})
    
    # Resend the verification email
    try:
        email_sent = email_service.send_verification_email_emailjs(email, username, token)
        if email_sent:
            return jsonify({'success': True, 'message': 'Verification code resent successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to resend verification email'})
    except Exception as e:
        logger.error(f"Error resending verification email: {e}")
        return jsonify({'success': False, 'message': 'Failed to resend verification email'})

@app.route('/verify-password-code', methods=['POST'])
def verify_password_code():
    data = request.get_json(silent=True) or {}
    code = data.get('code') or data.get('token')
    email = data.get('email')
    if not code:
        return jsonify({'success': False, 'message': 'Verification code is required'})

    # First try pending password changes
    pending = PENDING_PASSWORD_CHANGES.get(code)
    if pending:
        if pending['expires_at'] < datetime.now():
            del PENDING_PASSWORD_CHANGES[code]
            return jsonify({'success': False, 'message': 'Verification code expired.'})

        # Apply the password change
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET password = %s WHERE id = %s",
                (pending['new_password_hash'], pending['user_id'])
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Error applying password change: {e}")
            return jsonify({'success': False, 'message': 'Failed to update password'})
        finally:
            try:
                cursor.close()
                conn.close()
            except Exception:
                pass

        del PENDING_PASSWORD_CHANGES[code]
        return jsonify({'success': True, 'message': 'Password changed successfully. You can now log in.'})

    # Try OTP verification if email is provided
    if email:
        from db import verify_otp as db_verify_otp
        success, result = db_verify_otp(email, code, 'password_change')
        if success:
            return jsonify({'success': True, 'message': 'Email verified successfully. Your password change has been completed.'})

    return jsonify({'success': False, 'message': 'Invalid or expired code.'})
@app.route('/verify-password-change')
def verify_password_change():
    """Verify password change using token"""
    token = request.args.get('token')
    new_password = request.args.get('password')
    
    if not token:
        flash('Invalid password change link', 'error')
        return redirect(url_for('login'))
    
    success, user = verify_password_change_token(token)
    
    if success:
        if new_password:
            # Validate new password
            if len(new_password) < 8:
                flash('Password must be at least 8 characters long', 'error')
                return redirect(url_for('login'))
            
            if not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$', new_password):
                flash('Password must include uppercase, lowercase, numbers, and special characters', 'error')
                return redirect(url_for('login'))
            
            # Update password using bcrypt for consistency
            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE users SET password = %s WHERE id = %s",
                        (hashed_password, user['id'])
                    )
                    conn.commit()
                    cursor.close()
                    conn.close()
                    
                    # Clear the token
                    clear_password_change_token(user['id'])
                    
                    # Send notification email
                    email_service.send_password_change_notification(user['email'], user['username'])
                    
                    flash('Password changed successfully!', 'success')
                    return redirect(url_for('login'))
                except Exception as e:
                    logger.error(f"Error updating password: {e}")
                    flash('An error occurred while updating password', 'error')
                    return redirect(url_for('login'))
            else:
                flash('Database connection failed', 'error')
                return redirect(url_for('login'))
        else:
            # Show password change form
            return render_template('verify-password-change.html', token=token)
    else:
        flash(f'Password change verification failed: {user}', 'error')
        return redirect(url_for('login'))

@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    """OTP verification page and handler"""
    if request.method == 'GET':
        # Show OTP verification page
        verification_type = request.args.get('type', 'registration')
        email = request.args.get('email', '')
        
        return render_template('verify-otp.html', 
                             verification_type=verification_type,
                             email=email)
    
    elif request.method == 'POST':
        # Handle OTP verification
        data = request.get_json(silent=True) or {}
        code = data.get('code')
        verification_type = data.get('type', 'registration')
        email = data.get('email')
        
        if not code or not email:
            return jsonify({'success': False, 'message': 'Missing verification code or email'})
        
        # Verify OTP
        from db import verify_otp as db_verify_otp
        success, result = db_verify_otp(email, code, verification_type)
        
        if success:
            if verification_type == 'registration':
                # Handle registration verification - create user account
                # Find pending registration by email
                pending_reg = None
                for token, data in PENDING_REGISTRATIONS.items():
                    if data['data']['email'] == email:
                        pending_reg = data
                        break
                
                if pending_reg and pending_reg['expires_at'] > datetime.now():
                    reg_data = pending_reg['data']
                    success, result = create_user(
                        username=reg_data['username'],
                        email=reg_data['email'],
                        password=reg_data['password'],
                        first_name=reg_data['first_name'],
                        last_name=reg_data['last_name'],
                        birth_date=reg_data['birth_date'],
                        gender=reg_data['gender']
                    )
                    if success:
                        # Clean up pending registration
                        for token, data in list(PENDING_REGISTRATIONS.items()):
                            if data['data']['email'] == email:
                                del PENDING_REGISTRATIONS[token]
                        return jsonify({
                            'success': True, 
                            'message': 'Email verified and account created! You can now log in.'
                        })
                    else:
                        return jsonify({
                            'success': False, 
                            'message': result or 'Failed to create account.'
                        })
                else:
                    return jsonify({
                        'success': False, 
                        'message': 'Registration data not found or expired. Please sign up again.'
                    })
            elif verification_type == 'password_change':
                # Handle password change verification
                return jsonify({
                    'success': True, 
                    'message': 'Email verified successfully! Your password has been changed.'
                })
            else:
                return jsonify({
                    'success': True, 
                    'message': 'Email verified successfully!'
                })
        else:
            return jsonify({'success': False, 'message': result})

@app.route('/resend-otp', methods=['POST'])
def resend_otp():
    """Resend OTP code"""
    data = request.get_json(silent=True) or {}
    verification_type = data.get('type', 'registration')
    email = data.get('email')
    
    if not email:
        return jsonify({'success': False, 'message': 'Email address is required'})
    
    # Generate new OTP
    from db import generate_otp, save_otp
    otp = generate_otp()
    
    # Save OTP to database
    success, result = save_otp(email, otp, verification_type)
    if not success:
        return jsonify({'success': False, 'message': 'Failed to generate verification code'})
    
    # Send OTP via email
    try:
        if verification_type == 'registration':
            email_sent = email_service.send_otp_email(email, otp, 'registration')
        elif verification_type == 'password_change':
            email_sent = email_service.send_otp_email(email, otp, 'password_change')
        else:
            email_sent = email_service.send_otp_email(email, otp, 'general')
        
        if email_sent:
            return jsonify({'success': True, 'message': 'Verification code sent successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to send verification email'})
    except Exception as e:
        logger.error(f"Error sending OTP email: {e}")
        return jsonify({'success': False, 'message': 'Failed to send verification email'})

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        # Add password reset logic here
        return redirect(url_for('login'))
    return render_template('reset-password.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
def logout():
    # Clear both session and localStorage
    session.clear()
    return render_template('logout.html')


@app.route('/save-posture', methods=['POST'])
def save_posture():
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    data = request.get_json()
    success, result = save_posture_record(
        user_id=session['user']['id'],
        posture_type=data.get('posture_type'),
        confidence_score=data.get('confidence_score')
    )
    
    return jsonify({'success': success, 'message': result})

@app.route('/save-monitoring-session', methods=['POST'])
def save_monitoring_session():
    """Save posture data when monitoring session ends"""
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'User not logged in'})
    
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'No data provided'})
    
    try:
        success, result = save_posture_record(
            user_id=session['user']['id'],
            posture_type=data.get('posture_type', 'Unknown'),
            confidence_score=data.get('confidence_score', 0.0),
            session_duration=data.get('session_duration', 0),
            corrections_count=data.get('corrections_count', 0),
            good_time=data.get('good_time', 0),
            bad_time=data.get('bad_time', 0)
        )
        
        if success:
            logger.info(f"Saved monitoring session for user {session['user']['id']}: {data.get('posture_type')}")
            return jsonify({'success': True, 'message': 'Monitoring session saved successfully'})
        else:
            return jsonify({'success': False, 'message': result})
    except Exception as e:
        logger.error(f"Error saving monitoring session: {e}")
        return jsonify({'success': False, 'message': 'Error saving monitoring session'})

@app.route('/clear-posture-history', methods=['POST'])
@login_required
def clear_posture_history():
    """Clear all posture history records for the current user"""
    try:
        user_id = session['user']['id']
        success, result = clear_user_posture_history(user_id)
        if success:
            return jsonify({'success': True, 'deleted': result})
        else:
            return jsonify({'success': False, 'message': result})
    except Exception as e:
        logger.error(f"Error clearing posture history: {e}")
        return jsonify({'success': False, 'message': 'Error clearing posture history'})

@app.route('/delete-posture-record', methods=['POST'])
@login_required
def delete_posture_record_endpoint():
    """Delete a specific posture record for the current user"""
    try:
        data = request.get_json()
        record_id = data.get('record_id')
        
        if not record_id:
            return jsonify({'success': False, 'message': 'Record ID is required'})
        
        user_id = session['user']['id']
        success, result = delete_posture_record(record_id, user_id)
        
        if success:
            return jsonify({'success': True, 'message': result})
        else:
            return jsonify({'success': False, 'message': result})
    except Exception as e:
        logger.error(f"Error deleting posture record: {e}")
        return jsonify({'success': False, 'message': 'Error deleting posture record'})


@app.route('/get-current-posture')
def get_current_posture():
    """Get current posture data for frontend monitoring"""
    try:
        global CURRENT_POSTURE_DATA
        
        # Check if we have recent posture data (within last 5 seconds)
        current_time = time.time()
        if (CURRENT_POSTURE_DATA.get('last_update') and 
            current_time - CURRENT_POSTURE_DATA['last_update'] < 5.0):
            
            # Detailed ML prediction removed; return minimal info
            detailed_info = {}
            
            return jsonify({
                'success': True,
                'posture_data': {
                    'posture_quality': CURRENT_POSTURE_DATA['posture_quality'],
                    'confidence': CURRENT_POSTURE_DATA['confidence'],
                    'timestamp': CURRENT_POSTURE_DATA['timestamp'],
                    'detailed': detailed_info
                }
            })
        else:
            # No recent data, return unknown status
            return jsonify({
                'success': True,
                'posture_data': {
                    'posture_quality': 'unknown',
                    'confidence': 0.0,
                    'timestamp': current_time
                }
            })
    except Exception as e:
        logger.error(f"Error getting current posture: {e}")
        return jsonify({'success': False, 'message': 'Error getting posture data'})

@app.route('/check-posture-alarm')
def check_posture_alarm():
    """Check if posture alarm should be triggered based on AI detection"""
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'User not logged in'})
    
    try:
        # Check if alarm was triggered in the video stream
        # We'll use a global variable to track alarm state
        if hasattr(gen_frames, 'alarm_triggered') and gen_frames.alarm_triggered:
            # Reset the alarm flag
            gen_frames.alarm_triggered = False
            return jsonify({
                'success': True,
                'alarm_triggered': True,
                'message': 'Bad posture detected! Please correct your posture.'
            })
        
        return jsonify({
            'success': True,
            'alarm_triggered': False
        })
    except Exception as e:
        logger.error(f"Error checking posture alarm: {e}")
        return jsonify({'success': False, 'message': 'Error checking alarm status'})

@app.route('/upload-profile-picture', methods=['POST'])
@login_required
def upload_profile_picture():
    """Upload and save user profile picture"""
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'User not logged in'})
    
    try:
        if 'profile_picture' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'})
        
        file = request.files['profile_picture']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        # Check file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'jfif', 'webp'}
        if not file.filename.lower().endswith(tuple('.' + ext for ext in allowed_extensions)):
            return jsonify({'success': False, 'message': 'Invalid file type. Please upload PNG, JPG, JPEG, GIF, JFIF, or WEBP'})
        
        # Check file size (max 5MB)
        if len(file.read()) > 5 * 1024 * 1024:
            file.seek(0)  # Reset file pointer
            return jsonify({'success': False, 'message': 'File too large. Maximum size is 5MB'})
        
        file.seek(0)  # Reset file pointer
        
        # Create uploads directory if it doesn't exist
        upload_folder = os.path.join(app.static_folder, 'uploads', 'profile_pictures')
        os.makedirs(upload_folder, exist_ok=True)
        
        # Generate unique filename
        user_id = session['user']['id']
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        filename = f"profile_{user_id}.{file_extension}"
        filepath = os.path.join(upload_folder, filename)
        
        # Save the file
        file.save(filepath)
        
        # Update database with profile picture path
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                profile_picture_url = f"/static/uploads/profile_pictures/{filename}"
                cursor.execute("UPDATE users SET profile_picture = %s WHERE id = %s", (profile_picture_url, user_id))
                conn.commit()
                
                # Update session with new profile picture
                session['user']['profile_picture'] = profile_picture_url
                
                return jsonify({
                    'success': True,
                    'message': 'Profile picture uploaded successfully',
                    'profile_picture_url': profile_picture_url
                })
            except Exception as e:
                logger.error(f"Database error updating profile picture: {e}")
                return jsonify({'success': False, 'message': f'Database error: {str(e)}'})
            finally:
                conn.close()
        else:
            return jsonify({'success': False, 'message': 'Database connection failed'})
            
    except Exception as e:
        logger.error(f"Error uploading profile picture: {e}")
        return jsonify({'success': False, 'message': f'Upload error: {str(e)}'})

@app.route('/complete-exercise', methods=['POST'])
def complete_exercise():
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    data = request.get_json()
    success, result = record_completed_exercise(
        user_id=session['user']['id'],
        exercise_id=data.get('exercise_id')
    )
    
    return jsonify({'success': success, 'message': result})

@app.route('/check-session')
def check_session():
    if 'user' in session:
        return jsonify({
            'logged_in': True,
            'user': session['user']
        })
    return jsonify({
        'logged_in': False
    })

@app.route('/update-profile', methods=['POST'])
@login_required
def update_profile():
    try:
        data = request.get_json()
        user_id = session['user'].get('id')
        
        if not user_id:
            logger.error("User not found in session")
            return jsonify({'success': False, 'message': 'User not found in session'})
        
        # Validate required fields
        required_fields = ['firstName', 'lastName', 'email', 'username', 'gender']
        for field in required_fields:
            if not data.get(field):
                logger.warning(f"Missing required field: {field}")
                return jsonify({'success': False, 'message': f'{field} is required'})
        
        # Validate email format
        if not re.match(r"[^@]+@[^@]+\.[^@]+", data['email']):
            logger.warning(f"Invalid email format: {data['email']}")
            return jsonify({'success': False, 'message': 'Invalid email format'})
        
        # Validate username format
        if not re.match(r"^[a-zA-Z0-9_]+$", data['username']):
            logger.warning(f"Invalid username format: {data['username']}")
            return jsonify({'success': False, 'message': 'Username can only contain letters, numbers, and underscores'})
        
        conn = get_db_connection()
        if not conn:
            logger.error("Database connection failed")
            return jsonify({'success': False, 'message': 'Database connection failed'})
        
        try:
            cursor = conn.cursor(dictionary=True)
            
            # Check if username is already taken by another user
            cursor.execute("SELECT id FROM users WHERE username = %s AND id != %s", (data['username'], user_id))
            if cursor.fetchone():
                logger.warning(f"Username already taken: {data['username']}")
                return jsonify({'success': False, 'message': 'Username already taken'})
            
            # Check if email is already taken by another user
            cursor.execute("SELECT id FROM users WHERE email = %s AND id != %s", (data['email'], user_id))
            if cursor.fetchone():
                logger.warning(f"Email already taken: {data['email']}")
                return jsonify({'success': False, 'message': 'Email already taken'})
            
            # Get current user data
            cursor.execute("SELECT birth_date, gender FROM users WHERE id = %s", (user_id,))
            current_user = cursor.fetchone()
            
            # Update user information
            update_query = """
                UPDATE users 
                SET first_name = %s,
                    last_name = %s,
                    email = %s,
                    username = %s,
                    birth_date = %s,
                    gender = %s,
                    updated_at = NOW()
            """
            # Handle birth_date properly
            birth_date = data.get('birth_date')
            if birth_date and birth_date.strip():
                birth_date_value = birth_date
            else:
                birth_date_value = current_user['birth_date']
            
            # Handle gender properly
            gender = data.get('gender')
            if gender and gender.strip():
                gender_value = gender
            else:
                gender_value = current_user['gender']
            
            update_params = [
                data['firstName'],
                data['lastName'],
                data['email'],
                data['username'],
                birth_date_value,
                gender_value
            ]
            
            # Add password update if provided
            if data.get('password') and data['password'].strip():
                update_query += ", password = %s"
                update_params.append(generate_password_hash(data['password']))
            
            update_query += " WHERE id = %s"
            update_params.append(user_id)
            
            logger.info(f"Updating user {user_id} with query: {update_query}")
            logger.info(f"Update params: {update_params}")
            
            try:
                cursor.execute(update_query, tuple(update_params))
                conn.commit()
                logger.info(f"Database update successful for user {user_id}")
            except Exception as e:
                logger.error(f"Database update failed for user {user_id}: {e}")
                conn.rollback()
                return jsonify({'success': False, 'message': f'Database update failed: {str(e)}'})
            
            # Get updated user data
            cursor.execute("""
                SELECT id, username, email, first_name, last_name, 
                       birth_date, gender, created_at, updated_at
                FROM users 
                WHERE id = %s
            """, (user_id,))
            
            updated_user = cursor.fetchone()
            
            if updated_user:
                # Update session data with all user information
                session['user'].update({
                    'username': updated_user['username'],
                    'email': updated_user['email'],
                    'first_name': updated_user['first_name'],
                    'last_name': updated_user['last_name'],
                    'birth_date': str(updated_user['birth_date']) if updated_user['birth_date'] else None,
                    'gender': updated_user['gender']
                })
                
                logger.info(f"Successfully updated profile for user_id: {user_id}")
                return jsonify({
                    'success': True,
                    'message': 'Profile updated successfully',
                    'user': session['user']
                })
            
            logger.warning(f"Failed to update profile for user_id: {user_id}")
            return jsonify({'success': False, 'message': 'Failed to update profile'})
            
        except Exception as e:
            logger.error(f"Database error while updating profile: {e}")
            return jsonify({'success': False, 'message': 'Database error occurred'})
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/admin/users', methods=['GET'])
@login_required
def get_users():
    if session['user'].get('username') != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    success, users = get_all_users()
    if success:
        return jsonify({'success': True, 'users': users})
    return jsonify({'success': False, 'message': users})

@app.route('/admin/users', methods=['POST'])
@login_required
def create_user_admin():
    if session['user'].get('username') != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'No data provided'})
    
    # Format birth date
    try:
        birth_date = f"{data.get('birthYear')}-{data.get('birthMonth')}-{data.get('birthDay')}"
    except:
        return jsonify({'success': False, 'message': 'Invalid birth date format'})
    
    success, result = admin_create_user(
        username=data.get('username'),
        email=data.get('email'),
        password=data.get('password'),
        first_name=data.get('firstName'),
        last_name=data.get('lastName'),
        birth_date=birth_date,
        gender=data.get('gender')
    )
    
    if success:
        return jsonify({'success': True, 'message': 'User created successfully'})
    return jsonify({'success': False, 'message': result})

@app.route('/admin/users/<int:user_id>', methods=['PUT', 'PATCH'])
@login_required
def update_user_admin(user_id):
	if session['user'].get('username') != 'admin':
		return jsonify({'success': False, 'message': 'Unauthorized'})

	data = request.get_json()
	if not data:
		return jsonify({'success': False, 'message': 'No data provided'})

	# Validate required fields
	required_fields = ['firstName', 'lastName', 'email', 'username']
	for field in required_fields:
		if not data.get(field):
			return jsonify({'success': False, 'message': f'{field} is required'})

	# Validate email format
	if not re.match(r"[^@]+@[^@]+\.[^@]+", data['email']):
		return jsonify({'success': False, 'message': 'Invalid email format'})

	# Validate username format
	if not re.match(r"^[a-zA-Z0-9_]+$", data['username']):
		return jsonify({'success': False, 'message': 'Username can only contain letters, numbers, and underscores'})

	# Use current values for optional fields if not provided
	birth_date = data.get('birth_date')
	gender = data.get('gender')

	# If optional fields missing, fetch current values
	if birth_date is None or gender is None:
		conn = get_db_connection()
		if not conn:
			return jsonify({'success': False, 'message': 'Database connection failed'})
		try:
			cursor = conn.cursor(dictionary=True)
			cursor.execute("SELECT birth_date, gender FROM users WHERE id = %s", (user_id,))
			current_user = cursor.fetchone()
			if not current_user:
				return jsonify({'success': False, 'message': 'User not found'})
			if birth_date is None:
				birth_date = current_user.get('birth_date')
			if gender is None:
				gender = current_user.get('gender')
		finally:
			conn.close()

	success, result = update_user(
		user_id=user_id,
		username=data['username'],
		email=data['email'],
		first_name=data['firstName'],
		last_name=data['lastName'],
		birth_date=birth_date,
		gender=gender
	)

	if success:
		return jsonify({'success': True, 'message': 'User updated successfully', 'user': result})
	else:
		return jsonify({'success': False, 'message': result})

@app.route('/admin/users/<int:user_id>', methods=['DELETE'])
@login_required
def delete_user_admin(user_id):
    if session['user'].get('username') != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    success, message = delete_user(user_id)
    
    # Removed face recognition data reload
    
    return jsonify({'success': success, 'message': message})


@app.route('/get-user-stats')
@login_required
def get_user_stats():
    """Get user statistics from posture history with caching"""
    try:
        user_id = session['user'].get('id')
        if not user_id:
            logger.error("User not found in session")
            return jsonify({'success': False, 'message': 'User not found in session'})
        
        # Check if we have cached stats for this user (cache for 10 seconds)
        cache_key = f"user_stats_{user_id}"
        cached_stats = getattr(app, 'stats_cache', {}).get(cache_key)
        if cached_stats and (time.time() - cached_stats.get('timestamp', 0)) < 10:
            logger.info(f"Returning cached stats for user {user_id}")
            return jsonify({
                'success': True,
                'stats': cached_stats['data']
            })
        
        conn = get_db_connection()
        if not conn:
            logger.error("Database connection failed")
            return jsonify({'success': False, 'message': 'Database connection failed'})
        
        try:
            cursor = conn.cursor(dictionary=True)
            
            # Optimized single query to get all stats at once with better performance
            cursor.execute("""
                SELECT 
                    COUNT(*) as sessions,
                    COALESCE(SUM(session_duration), 0) as total_duration,
                    SUM(CASE WHEN posture_type LIKE '%Good%' THEN 1 ELSE 0 END) as good_records,
                    COUNT(*) as total_records
                FROM posture_records 
                WHERE user_id = %s
            """, (user_id,))
            
            result = cursor.fetchone()
            
            if result:
                sessions = result['sessions'] or 0
                total_duration = result['total_duration'] or 0
                good_records = result['good_records'] or 0
                total_records = result['total_records'] or 0
                
                # Calculate good posture percentage
                if total_records > 0:
                    good_posture_percent = round((good_records / total_records) * 100, 1)
                else:
                    good_posture_percent = 0
                
                # Convert total duration to hours
                total_time_hours = round(total_duration / 3600, 1)
            else:
                sessions = 0
                good_posture_percent = 0
                total_time_hours = 0
            
            cursor.close()
            
            logger.info(f"Retrieved stats for user {user_id}: {sessions} sessions, {good_posture_percent}% good posture, {total_time_hours}h total time")
            
            # Cache the results
            stats_data = {
                'sessions': sessions,
                'good_posture_percent': good_posture_percent,
                'total_time_hours': total_time_hours
            }
            
            # Initialize cache if it doesn't exist
            if not hasattr(app, 'stats_cache'):
                app.stats_cache = {}
            
            # Store in cache with timestamp
            app.stats_cache[cache_key] = {
                'data': stats_data,
                'timestamp': time.time()
            }
            
            return jsonify({
                'success': True,
                'stats': stats_data
            })
            
        except Exception as e:
            logger.error(f"Database error getting user stats: {e}")
            return jsonify({'success': False, 'message': f'Database error: {str(e)}'})
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'})

@app.route('/get-user-data')
@login_required
def get_user_data():
    try:
        user_id = session['user'].get('id')
        if not user_id:
            logger.error("User not found in session")
            return jsonify({'success': False, 'message': 'User not found in session'})
        
        conn = get_db_connection()
        if not conn:
            logger.error("Database connection failed")
            return jsonify({'success': False, 'message': 'Database connection failed'})
        
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, username, email, first_name, last_name, 
                       birth_date, gender, profile_picture, created_at, updated_at
                FROM users 
                WHERE id = %s
            """, (user_id,))
            
            user = cursor.fetchone()
            cursor.close()
            
            if user:
                logger.info(f"Successfully retrieved user data for user_id: {user_id}")
                return jsonify({
                    'success': True,
                    'user': user
                })
            logger.warning(f"User not found in database for user_id: {user_id}")
            return jsonify({'success': False, 'message': 'User not found'})
        except Exception as e:
            logger.error(f"Database error while getting user data: {e}")
            return jsonify({'success': False, 'message': 'Database error occurred'})
        finally:
            if conn:
                conn.close()
    except Exception as e:
        logger.error(f"Error getting user data: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/update-camera-settings', methods=['POST'])
@login_required
def update_camera_settings():
    """Update camera mirroring settings for the current user"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'})
        
        mirror_enabled = data.get('mirror_enabled', True)
        
        # Update the global camera mirror setting
        # Note: This is a simple implementation that updates the config
        # In a production environment, you might want to store this per-user in the database
        app.config['CAMERA_MIRROR'] = mirror_enabled
        
        logger.info(f"Camera mirror setting updated to: {mirror_enabled}")
        
        return jsonify({
            'success': True, 
            'message': 'Camera settings updated successfully',
            'mirror_enabled': mirror_enabled
        })
        
    except Exception as e:
        logger.error(f"Error updating camera settings: {e}")
        return jsonify({'success': False, 'message': 'Error updating camera settings'})

@app.route('/get-camera-settings')
@login_required
def get_camera_settings():
    """Get current camera settings for the user"""
    try:
        mirror_enabled = app.config.get('CAMERA_MIRROR', True)
        
        return jsonify({
            'success': True,
            'mirror_enabled': mirror_enabled
        })
        
    except Exception as e:
        logger.error(f"Error getting camera settings: {e}")
        return jsonify({'success': False, 'message': 'Error getting camera settings'})

 

# Removed face recognition-related endpoints and admin features

@app.route('/admin/user/<int:user_id>/posture-history')
@login_required
def admin_user_posture_history(user_id):
    """Get posture history for a specific user (admin only)"""
    try:
        # Check if user is admin
        if session['user'].get('username') != 'admin':
            return jsonify({'success': False, 'message': 'Unauthorized'})
        
        # Get user information
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, username, email, first_name, last_name 
                FROM users 
                WHERE id = %s
            """, (user_id,))
            user = cursor.fetchone()
            
            if not user:
                return jsonify({'success': False, 'message': 'User not found'})
            
            # Get posture history
            success, history = get_user_posture_history(user_id, limit=100)
            
            if not success:
                return jsonify({'success': False, 'message': history})
            
            # Calculate statistics
            total_records = len(history) if history else 0
            good_posture_count = 0
            bad_posture_count = 0
            total_confidence = 0
            total_duration = 0
            
            if history:
                for record in history:
                    if 'Good' in record.get('posture_type', ''):
                        good_posture_count += 1
                    else:
                        bad_posture_count += 1
                    total_confidence += record.get('confidence_score', 0)
                    total_duration += record.get('session_duration', 0)
            
            # Calculate percentages and averages
            good_posture_percentage = (good_posture_count / total_records * 100) if total_records > 0 else 0
            bad_posture_percentage = (bad_posture_count / total_records * 100) if total_records > 0 else 0
            avg_confidence = (total_confidence / total_records * 100) if total_records > 0 else 0
            total_hours = total_duration / 3600 if total_duration else 0
            
            stats = {
                'total_records': total_records,
                'good_posture_percentage': round(good_posture_percentage, 1),
                'bad_posture_percentage': round(bad_posture_percentage, 1),
                'total_hours': round(total_hours, 1),
                'avg_confidence': round(avg_confidence, 1),
                'health_score': round((good_posture_percentage + avg_confidence) / 2, 1)
            }
            
            return jsonify({
                'success': True,
                'user': user,
                'history': history,
                'stats': stats
            })
            
        except Exception as e:
            logger.error(f"Database error getting user posture history: {e}")
            return jsonify({'success': False, 'message': 'Database error occurred'})
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        logger.error(f"Error getting user posture history: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/export-posture-history')
@login_required
def export_posture_history():
    """Export current user's posture history as PDF with personalized recommendations and graphs."""
    try:
        user_id = session['user']['id']
        
        # Get user information for the report
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'Database connection failed'}), 500
        
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT first_name, last_name, username, email 
                FROM users 
                WHERE id = %s
            """, (user_id,))
            user_info = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if not user_info:
                return jsonify({'success': False, 'message': 'User not found'}), 404
                
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return jsonify({'success': False, 'message': 'Failed to retrieve user information'}), 500
        
        success, history = get_user_posture_history(user_id, limit=1000)
        if not success:
            return jsonify({'success': False, 'message': 'Failed to retrieve posture history'}), 500

        # Build CSV in-memory
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow([
            'id', 'recorded_at', 'posture_type', 'confidence_score', 'session_duration',
            'good_time', 'bad_time', 'corrections_count'
        ])
        total_sessions = 0
        total_bad_time = 0.0
        total_good_time = 0.0
        total_duration = 0.0
        good_sessions = 0
        for record in (history or []):
            total_sessions += 1
            posture_type = record.get('posture_type', '')
            if 'Good' in posture_type:
                good_sessions += 1
            confidence = float(record.get('confidence_score', 0.0) or 0.0)
            # Removed per-session confidence trend collection
            duration = float(record.get('session_duration', 0.0) or 0.0)
            good_time = float(record.get('good_time', 0.0) or (duration * confidence))
            bad_time = float(record.get('bad_time', 0.0) or (duration * (1.0 - confidence)))
            total_duration += duration
            total_good_time += good_time
            total_bad_time += bad_time
            csv_writer.writerow([
                record.get('id'),
                record.get('recorded_at'),
                posture_type,
                f"{confidence:.3f}",
                int(duration),
                int(good_time),
                int(bad_time),
                int(record.get('corrections_count', 0) or 0)
            ])

        csv_bytes = csv_buffer.getvalue().encode('utf-8')

        # Build AI ergonomic tips using Gemini if configured
        tips_text = (
            "No posture history available to analyze." if total_sessions == 0 else ""
        )
        if total_sessions > 0:
            good_pct = (good_sessions / total_sessions) * 100.0 if total_sessions else 0.0
            avg_confidence = 0.0
            if total_duration > 0:
                avg_confidence = (total_good_time / total_duration) * 100.0
            summary_blob = (
                f"Sessions: {total_sessions}\n"
                f"Good sessions %: {good_pct:.1f}\n"
                f"Total duration (min): {total_duration/60:.1f}\n"
                f"Good time (min): {total_good_time/60:.1f}\n"
                f"Bad time (min): {total_bad_time/60:.1f}\n"
                f"Avg confidence %: {avg_confidence:.1f}\n"
            )
            if Config.GEMINI_API_KEY:
                try:
                    prompt = (
                        "You are an ergonomics coach. Given these aggregate posture stats, "
                        "write concise, actionable ergonomic recommendations (5-8 bullets) "
                        "for an office worker. Keep bullets short, concrete, and non-medical.\n\n"
                        f"STATS:\n{summary_blob}\n"
                        "Also include 3 quick desk stretches and 3 habit tips."
                    )
                    tips_text = _gemini_generate_text('gemini-2.5-flash', prompt)
                    if not tips_text:
                        raise RuntimeError('Empty response from Gemini')
                except Exception as e:
                    logger.warning(f"Gemini generation failed: {e}")
                    tips_text = "Ergonomic tips unavailable (AI error)."
            else:
                tips_text = (
                    "AI tips are disabled. Set GEMINI_API_KEY to enable ergonomic recommendations."
                )

        # Build a simple HTML report with inline SVG visualizations
        # Removed _svg_line_chart since Confidence Trend by Session is no longer used

        def _svg_stacked_bar(good_minutes, bad_minutes, width=800, height=40):
            total = good_minutes + bad_minutes
            if total <= 0:
                total = 1
            good_w = int(width * (good_minutes / total))
            bad_w = width - good_w
            return (
                f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>"
                f"<rect x='0' y='0' width='{good_w}' height='{height}' fill='#16a34a'/>"
                f"<rect x='{good_w}' y='0' width='{bad_w}' height='{height}' fill='#ef4444'/>"
                f"</svg>"
            )

        # Sanitize AI text: remove markdown headings, bullet markers, and bold markers for a cleaner look
        safe_tips_text = re.sub(r'^\s*#+\s*', '', tips_text or '', flags=re.MULTILINE)
        safe_tips_text = re.sub(r'^\s*[\*-]\s+', '', safe_tips_text, flags=re.MULTILINE)
        safe_tips_text = re.sub(r'\*\*(.*?)\*\*', r'\1', safe_tips_text)  # strip **bold**
        safe_tips_text = re.sub(r'^\s*\u2022\s+', '', safe_tips_text, flags=re.MULTILINE)  # strip bullet char

        # Build AI-driven Suggested Products (JSON) using Gemini; skip silently on errors
        products_html = ""
        if Config.GEMINI_API_KEY and total_sessions > 0:
            try:
                prod_prompt = (
                    "You are an ergonomics retail assistant. Based on these posture aggregates, "
                    "suggest practical, non-medical products a typical user can buy. Group items into 1-3 sections. "
                    "Each section must include: title, issue, and 2-4 products with name and desc. Keep names generic (no brand/links).\n\n"
                    f"STATS:\n{summary_blob}\n\n"
                    "Return ONLY JSON with this schema: {\n  \"sections\": [ { \"title\": str, \"issue\": str, \"products\": [ { \"name\": str, \"desc\": str } ] } ]\n}"
                )
                prod_text = _gemini_generate_text('gemini-2.5-flash', prod_prompt, response_mime_type='application/json')
                prod_data = json.loads(prod_text)
                sections = prod_data.get('sections') or []
                if sections:
                    parts = ["<h2>Suggested Products</h2>", "<div class='card'>"]
                    for idx, sec in enumerate(sections, 1):
                        title = sec.get('title', f'Section {idx}')
                        issue = sec.get('issue', '')
                        parts.append(f"<h3>{idx}. {title}</h3>")
                        if issue:
                            parts.append(f"<p><strong>Detected Issue:</strong> {issue}</p>")
                        prods = sec.get('products') or []
                        if prods:
                            parts.append("<p><strong>Suggested Products:</strong></p><ul style='margin-top:6px'>")
                            for p in prods:
                                name = p.get('name', 'Product')
                                desc = p.get('desc', '')
                                parts.append(f"<li><strong>{name}</strong> – {desc}</li>")
                            parts.append("</ul>")
                    parts.append("</div>")
                    products_html = "".join(parts)
            except Exception as _e:
                logger.warning(f"Gemini product JSON generation failed: {_e}")
                products_html = ""

        # Render report using template to match app theme
        from flask import render_template_string
        stacked_bar_svg = _svg_stacked_bar(int(total_good_time/60), int(total_bad_time/60))
        report_html = render_template(
            'export_report.html',
            total_sessions=total_sessions,
            good_pct=(good_sessions/total_sessions*100.0 if total_sessions else 0),
            total_duration=total_duration,
            total_good_time=total_good_time,
            total_bad_time=total_bad_time,
            stacked_bar_svg=stacked_bar_svg,
            safe_tips_text=safe_tips_text,
            products_html=products_html,
        )

        # Generate comprehensive PDF report
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        from datetime import datetime
        import os

        # Create PDF buffer
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=100, bottomMargin=50)
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Professional styles
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], 
                                    fontSize=28, spaceAfter=20, alignment=TA_CENTER, 
                                    textColor=colors.HexColor('#007AFF'), fontName='Helvetica-Bold')
        
        subtitle_style = ParagraphStyle('Subtitle', parent=styles['Heading2'], 
                                      fontSize=14, spaceAfter=15, alignment=TA_CENTER, 
                                      textColor=colors.HexColor('#666666'), fontName='Helvetica')
        
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], 
                                     fontSize=16, spaceAfter=12, spaceBefore=20, 
                                     textColor=colors.HexColor('#007AFF'), fontName='Helvetica-Bold')
        
        body_style = ParagraphStyle('Body', parent=styles['Normal'], 
                                   fontSize=11, spaceAfter=8, leading=14, 
                                   fontName='Helvetica')
        
        bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'], 
                                     fontSize=11, spaceAfter=6, leftIndent=20, 
                                     bulletIndent=10, fontName='Helvetica')
        
        # Build PDF content
        story = []
        
        # Header with Logo and Title
        try:
            # Try to load the logo
            logo_path = os.path.join('static', 'images', 'logo-test-2-1-10.png')
            if os.path.exists(logo_path):
                logo = Image(logo_path, width=1.5*inch, height=1.5*inch)
                story.append(logo)
                story.append(Spacer(1, 10))
        except Exception as logo_error:
            logger.warning(f"Could not load logo: {logo_error}")
            # Continue without logo
        
        # Company name and title
        story.append(Paragraph("<b>POSTUREASE</b>", title_style))
        story.append(Paragraph("Posture Health Analysis Report", subtitle_style))
        story.append(Spacer(1, 30))
        
        # Report info
        user_full_name = f"{user_info['first_name']} {user_info['last_name']}"
        report_info_data = [
            ['Report Generated', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
            ['User Name', user_full_name],
            ['Username', user_info['username']],
            ['Email', user_info['email']],
            ['Report Type', 'Posture Health Analysis'],
            ['Analysis Period', f"Last {total_sessions} sessions" if total_sessions > 0 else "No data available"]
        ]
        
        report_info_table = Table(report_info_data, colWidths=[2*inch, 3*inch])
        report_info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(report_info_table)
        story.append(Spacer(1, 30))
        
        # Executive Summary
        story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
        if total_sessions > 0:
            good_pct = (good_sessions / total_sessions) * 100.0
            avg_confidence = (total_good_time / total_duration) * 100.0 if total_duration > 0 else 0.0
            
            summary_data = [
                ['Metric', 'Value', 'Status'],
                ['Total Sessions', f"{total_sessions}", '✓' if total_sessions > 0 else '⚠'],
                ['Good Posture Sessions', f"{good_sessions} ({good_pct:.1f}%)", '✓' if good_pct > 60 else '⚠' if good_pct > 40 else '✗'],
                ['Total Monitoring Time', f"{total_duration/60:.1f} minutes", '✓' if total_duration > 0 else '⚠'],
                ['Good Posture Time', f"{total_good_time/60:.1f} minutes", '✓' if total_good_time > total_bad_time else '⚠'],
                ['Poor Posture Time', f"{total_bad_time/60:.1f} minutes", '⚠' if total_bad_time > total_good_time else '✓'],
                ['Average Confidence', f"{avg_confidence:.1f}%", '✓' if avg_confidence > 70 else '⚠' if avg_confidence > 50 else '✗']
            ]
            
            summary_table = Table(summary_data, colWidths=[2*inch, 2*inch, 0.5*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#007AFF')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
            ]))
            story.append(summary_table)
        else:
            story.append(Paragraph("No posture data available for analysis.", body_style))
        
        story.append(Spacer(1, 20))
        
        # Musculoskeletal Risk Assessment
        story.append(Paragraph("MUSCULOSKELETAL RISK ASSESSMENT", heading_style))
        
        # Calculate risk factors
        risk_factors = []
        risk_level = "LOW"
        risk_score = 0
        if total_sessions > 0:
            poor_posture_ratio = total_bad_time / total_duration if total_duration > 0 else 0
            avg_session_duration = total_duration / total_sessions if total_sessions > 0 else 0
            
            if poor_posture_ratio > 0.7:
                risk_factors.append("HIGH RISK: Poor posture maintained for >70% of monitoring time")
                risk_level = "HIGH"
                risk_score += 3
            elif poor_posture_ratio > 0.5:
                risk_factors.append("MODERATE RISK: Poor posture maintained for >50% of monitoring time")
                risk_level = "MODERATE"
                risk_score += 2
            elif poor_posture_ratio > 0.3:
                risk_factors.append("ELEVATED RISK: Poor posture maintained for >30% of monitoring time")
                risk_score += 1
            
            if avg_session_duration > 180:  # 3 hours
                risk_factors.append("EXTENDED SESSIONS: Average session duration exceeds 3 hours")
                risk_score += 2
                if risk_level == "LOW":
                    risk_level = "MODERATE"
            elif avg_session_duration > 120:  # 2 hours
                risk_factors.append("LONG SESSIONS: Average session duration exceeds 2 hours")
                risk_score += 1
            
            if total_bad_time > 3600:  # 1 hour total
                risk_factors.append("CUMULATIVE EXPOSURE: Total poor posture time exceeds 1 hour")
                risk_score += 2
                if risk_level == "LOW":
                    risk_level = "MODERATE"
            elif total_bad_time > 1800:  # 30 minutes total
                risk_factors.append("CUMULATIVE EXPOSURE: Total poor posture time exceeds 30 minutes")
                risk_score += 1
            
            # Additional risk factors
            if total_sessions > 50:
                risk_factors.append("FREQUENT MONITORING: High number of sessions indicates prolonged computer use")
                risk_score += 1
            
            # Update risk level based on score
            if risk_score >= 5:
                risk_level = "HIGH"
            elif risk_score >= 3:
                risk_level = "MODERATE"
            elif risk_score >= 1:
                risk_level = "ELEVATED"
        
        # Risk level indicator with detailed explanation
        risk_color = colors.HexColor('#10b981') if risk_level == "LOW" else colors.HexColor('#f59e0b') if risk_level in ["MODERATE", "ELEVATED"] else colors.HexColor('#ef4444')
        story.append(Paragraph(f"<b>Overall Risk Level: <font color='{risk_color.hexval()}'>{risk_level}</font></b>", body_style))
        story.append(Spacer(1, 10))
        
        # Risk level explanation
        risk_explanations = {
            "LOW": "Minimal risk of developing posture-related musculoskeletal disorders",
            "ELEVATED": "Some risk factors present - monitor and improve posture habits",
            "MODERATE": "Significant risk factors present - immediate attention recommended",
            "HIGH": "High risk of developing musculoskeletal disorders - professional consultation advised"
        }
        
        story.append(Paragraph(f"<b>Risk Assessment:</b> {risk_explanations.get(risk_level, 'Unable to assess')}", body_style))
        story.append(Spacer(1, 10))
        
        if risk_factors:
            story.append(Paragraph("<b>Identified Risk Factors:</b>", body_style))
            for factor in risk_factors:
                story.append(Paragraph(f"• {factor}", bullet_style))
        else:
            story.append(Paragraph("• LOW RISK: Good posture habits detected", bullet_style))
        
        story.append(Spacer(1, 20))
        
        # Potential Musculoskeletal Disorders
        story.append(Paragraph("POTENTIAL HEALTH CONCERNS", heading_style))
        
        disorders = []
        disorder_descriptions = {}
        if total_sessions > 0:
            poor_ratio = total_bad_time / total_duration if total_duration > 0 else 0
            avg_session_duration = total_duration / total_sessions if total_sessions > 0 else 0
            
            if poor_ratio > 0.6:
                disorders.extend([
                    "Forward Head Posture (Text Neck)",
                    "Upper Cross Syndrome", 
                    "Lower Back Pain (Lumbar Strain)",
                    "Cervical Spine Degeneration"
                ])
                disorder_descriptions.update({
                    "Forward Head Posture (Text Neck)": "Chronic forward positioning of the head, leading to neck strain and potential spinal misalignment",
                    "Upper Cross Syndrome": "Muscle imbalance affecting the upper back, shoulders, and neck, causing rounded shoulders and forward head posture",
                    "Lower Back Pain (Lumbar Strain)": "Strain on the lumbar spine from prolonged sitting with poor posture",
                    "Cervical Spine Degeneration": "Premature wear and tear on cervical vertebrae due to poor neck positioning"
                })
            if poor_ratio > 0.4:
                disorders.extend([
                    "Shoulder Impingement Syndrome",
                    "Carpal Tunnel Syndrome",
                    "Thoracic Outlet Syndrome"
                ])
                disorder_descriptions.update({
                    "Shoulder Impingement Syndrome": "Compression of shoulder tendons due to rounded shoulders and forward head posture",
                    "Carpal Tunnel Syndrome": "Nerve compression in the wrist from poor arm positioning and repetitive motions",
                    "Thoracic Outlet Syndrome": "Compression of nerves and blood vessels between the collarbone and first rib"
                })
            if total_bad_time > 3600:  # 1 hour total
                disorders.append("Chronic Postural Dysfunction")
                disorder_descriptions["Chronic Postural Dysfunction"] = "Long-term postural imbalances that can lead to chronic pain and reduced mobility"
            
            # Additional risk factors based on session patterns
            if avg_session_duration > 120:  # 2+ hours average
                disorders.append("Sedentary Lifestyle Effects")
                disorder_descriptions["Sedentary Lifestyle Effects"] = "Prolonged sitting without movement can lead to muscle atrophy, reduced circulation, and metabolic issues"
        
        if disorders:
            story.append(Paragraph("Based on your posture patterns and usage data, you may be at risk for the following conditions:", body_style))
            story.append(Spacer(1, 10))
            
            for disorder in disorders:
                story.append(Paragraph(f"<b>• {disorder}</b>", bullet_style))
                if disorder in disorder_descriptions:
                    story.append(Paragraph(f"  {disorder_descriptions[disorder]}", 
                                         ParagraphStyle('Description', parent=body_style, leftIndent=30, fontSize=10, textColor=colors.HexColor('#666666'))))
                story.append(Spacer(1, 5))
        else:
            story.append(Paragraph("• No significant risk factors detected based on current data", bullet_style))
            story.append(Paragraph("Continue maintaining good posture habits to prevent future issues.", body_style))
        
        story.append(Spacer(1, 20))
        
        # Personalized Ergonomic Recommendations
        story.append(Paragraph("PERSONALIZED RECOMMENDATIONS", heading_style))
        
        # Generate personalized recommendations based on risk level and data
        personalized_recommendations = []
        
        if total_sessions > 0:
            poor_ratio = total_bad_time / total_duration if total_duration > 0 else 0
            avg_session_duration = total_duration / total_sessions if total_sessions > 0 else 0
            
            # Immediate actions based on risk level
            if risk_level == "HIGH":
                personalized_recommendations.extend([
                    "URGENT: Schedule a consultation with an ergonomics specialist or physical therapist",
                    "Implement immediate posture breaks every 15-20 minutes",
                    "Consider using a posture-correcting device or app for real-time feedback",
                    "Evaluate your workstation setup with a professional"
                ])
            elif risk_level == "MODERATE":
                personalized_recommendations.extend([
                    "Set up posture reminders every 30 minutes",
                    "Invest in ergonomic equipment (adjustable chair, monitor stand, keyboard tray)",
                    "Start a daily stretching routine focusing on neck, shoulders, and back"
                ])
            elif risk_level == "ELEVATED":
                personalized_recommendations.extend([
                    "Monitor your posture more frequently during work sessions",
                    "Take micro-breaks every 45 minutes to stretch and reset posture",
                    "Consider using a standing desk or adjustable workstation"
                ])
            else:  # LOW
                personalized_recommendations.extend([
                    "Continue maintaining your good posture habits",
                    "Consider periodic posture assessments to maintain current standards"
                ])
            
            # Session-specific recommendations
            if avg_session_duration > 180:  # 3+ hours
                personalized_recommendations.append("CRITICAL: Break up long sessions into 90-minute blocks with 15-minute breaks")
            elif avg_session_duration > 120:  # 2+ hours
                personalized_recommendations.append("Take a 10-minute break every 2 hours to prevent fatigue")
            
            # Posture-specific recommendations
            if poor_ratio > 0.5:
                personalized_recommendations.extend([
                    "Focus on keeping your ears aligned with your shoulders",
                    "Use a lumbar support cushion to maintain natural spine curvature",
                    "Position your monitor at eye level to prevent forward head posture"
                ])
        
        
        # Add personalized recommendations
        if personalized_recommendations:
            story.append(Paragraph("<b>Personalized Action Items:</b>", body_style))
            for rec in personalized_recommendations:
                story.append(Paragraph(f"• {rec}", bullet_style))
        else:
            story.append(Paragraph("• Maintain proper posture during work sessions", bullet_style))
            story.append(Paragraph("• Take regular breaks every 30-45 minutes", bullet_style))
            story.append(Paragraph("• Adjust your workstation ergonomics", bullet_style))
        
        story.append(Spacer(1, 20))
        
        # Exercise Recommendations
        story.append(Paragraph("RECOMMENDED EXERCISES", heading_style))
        
        # Categorize exercises based on risk factors
        basic_exercises = [
            "Neck stretches and rotations",
            "Shoulder blade squeezes", 
            "Chest opening stretches",
            "Core strengthening exercises",
            "Lower back stretches",
            "Wrist and forearm stretches"
        ]
        
        advanced_exercises = []
        if total_sessions > 0 and risk_level in ["MODERATE", "HIGH"]:
            advanced_exercises = [
                "Upper trapezius stretches",
                "Pectoral muscle stretches",
                "Thoracic spine mobility exercises",
                "Deep neck flexor strengthening",
                "Posterior chain strengthening"
            ]
        
        story.append(Paragraph("<b>Basic Posture Exercises (Daily):</b>", body_style))
        for exercise in basic_exercises:
            story.append(Paragraph(f"• {exercise}", bullet_style))
        
        if advanced_exercises:
            story.append(Spacer(1, 10))
            story.append(Paragraph("<b>Advanced Exercises (For Higher Risk Individuals):</b>", body_style))
            for exercise in advanced_exercises:
                story.append(Paragraph(f"• {exercise}", bullet_style))
        
        story.append(Spacer(1, 20))
        
        # Add simple data visualization tables instead of complex charts
        if total_sessions > 0:
            story.append(PageBreak())  # New page for data analysis
            
            
            # Posture Quality Summary
            story.append(Paragraph("POSTURE QUALITY SUMMARY", heading_style))
            
            if total_duration > 0:
                good_pct = (total_good_time / total_duration) * 100
                bad_pct = (total_bad_time / total_duration) * 100
                
                summary_data = [
                    ['Metric', 'Value', 'Percentage'],
                    ['Total Monitoring Time', f"{total_duration/60:.1f} minutes", '100%'],
                    ['Good Posture Time', f"{total_good_time/60:.1f} minutes", f"{good_pct:.1f}%"],
                    ['Poor Posture Time', f"{total_bad_time/60:.1f} minutes", f"{bad_pct:.1f}%"],
                    ['Average Session Duration', f"{total_duration/total_sessions/60:.1f} minutes", 'N/A'],
                    ['Good Posture Sessions', f"{good_sessions} out of {total_sessions}", f"{(good_sessions/total_sessions)*100:.1f}%"]
                ]
                
                summary_table = Table(summary_data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#007AFF')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                ]))
                story.append(summary_table)
            else:
                story.append(Paragraph("No posture data available for analysis.", body_style))
        
        # Professional Footer
        story.append(Spacer(1, 40))
        story.append(Paragraph("─" * 50, ParagraphStyle('Line', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.HexColor('#dee2e6'))))
        story.append(Spacer(1, 10))
        story.append(Paragraph("<b>POSTUREASE</b> - Posture Health Analysis Report", 
                              ParagraphStyle('Footer', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER, textColor=colors.HexColor('#007AFF'))))
        story.append(Spacer(1, 15))
        
        # Comprehensive medical disclaimers
        disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.HexColor('#666666'))
        
        story.append(Paragraph("<b>IMPORTANT MEDICAL DISCLAIMER</b>", disclaimer_style))
        story.append(Spacer(1, 5))
        story.append(Paragraph("This report is generated for educational and informational purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment.", disclaimer_style))
        story.append(Spacer(1, 3))
        story.append(Paragraph("The posture analysis and recommendations provided are based on automated monitoring and should not be considered as medical diagnosis.", disclaimer_style))
        story.append(Spacer(1, 3))
        story.append(Paragraph("Always consult with a qualified healthcare professional, ergonomics specialist, or physical therapist for proper assessment and treatment of any musculoskeletal concerns.", disclaimer_style))
        story.append(Spacer(1, 3))
        story.append(Paragraph("PosturEase is not a medical device and should not be used for diagnostic or therapeutic purposes.", disclaimer_style))
        story.append(Spacer(1, 3))
        story.append(Paragraph("If you experience persistent pain, discomfort, or other symptoms, please seek immediate medical attention.", disclaimer_style))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | PosturEase v1.0", 
                              ParagraphStyle('Footer', parent=styles['Normal'], fontSize=7, alignment=TA_CENTER, textColor=colors.HexColor('#999999'))))
        
        # Build PDF
        try:
            doc.build(story)
            pdf_buffer.seek(0)
            
            filename = f"PosturEase_Health_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            return send_file(
                pdf_buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=filename
            )
        except Exception as pdf_error:
            logger.error(f"Error building PDF: {pdf_error}")
            raise pdf_error
    except Exception as e:
        logger.error(f"Error exporting posture history: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error details: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'message': f'Error exporting data: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 