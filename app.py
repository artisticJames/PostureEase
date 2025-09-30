from flask import Flask, render_template, request, redirect, url_for, flash, session, Response, jsonify
import os
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time
from config import Config
from db import create_user, verify_user, save_posture_record, get_user_posture_history, get_exercises, record_completed_exercise, update_user, get_all_users, admin_create_user, delete_user, toggle_user_status, get_db_connection, test_connection, save_verification_token, verify_email_token, verify_password_change_token, clear_password_change_token, get_user_by_email, clear_user_posture_history, delete_posture_record
from datetime import datetime, timedelta
from functools import wraps
import logging
import re
from werkzeug.security import generate_password_hash, check_password_hash
import bcrypt
# Removed face recognition and security logger imports as part of simplifying to single-person posture monitoring
from email_service import email_service
from ml_posture_classifier import initialize_ml_classifier, get_ml_classifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Removed face registration configuration

# Ensure the secret key is set
if not app.secret_key:
    app.secret_key = os.urandom(24)

# Global state for posture calibration and latest pose landmarks
# Note: This is per-process memory and resets on server restart
LATEST_POINTS = None  # dict[int, tuple[int, int]] of last detected pixel points
CALIBRATION = {
	'front_angle_deg': None,   # Baseline front angle (nose vs shoulders mid vertical)
	'timestamp': None,
	'tolerance_deg': 12.0      # Allowed deviation from baseline before marking as bad
}

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
    Generate personalized exercise recommendations based on posture history data.
    
    Args:
        current_record: Current posture record being analyzed
        history_records: List of all user's posture history records
    
    Returns:
        List of recommended exercise names
    """
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

# Test database connection on startup
if not test_connection():
    logger.error("Failed to establish initial database connection. The application may not function correctly.")
else:
    logger.info("Initial database connection successful")

# Initialize ML classifier if enabled
if app.config.get('USE_ML_CLASSIFIER', True):
    model_path = app.config.get('ML_MODEL_PATH', 'Training(Updated)')
    if initialize_ml_classifier(model_path):
        logger.info("ML posture classifier initialized successfully")
    else:
        logger.warning("Failed to initialize ML classifier, falling back to rule-based classification")
else:
    logger.info("ML classifier disabled, using rule-based classification")

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
    """Classify posture using ML model if available, otherwise fall back to rule-based classification"""
    
    # Try ML classifier first if available
    ml_classifier = get_ml_classifier()
    if ml_classifier and ml_classifier.models_loaded:
        try:
            posture, confidence = ml_classifier.classify_posture(landmarks, image_shape)
            return posture, confidence
        except Exception as e:
            logger.warning(f"ML classification failed, falling back to rule-based: {e}")
    
    # Fallback to rule-based classification
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

        # 5. Front angle (calibrated): angle between vector from shoulder midpoint to nose and vertical axis
        ls = points.get(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        rs = points.get(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        nose = points.get(mp_pose.PoseLandmark.NOSE.value)

        front_angle_deg = None
        front_score = 1.0
        if ls and rs and nose:
            shoulder_mid = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
            # Vector from shoulder midpoint to nose
            vx = nose[0] - shoulder_mid[0]
            vy = nose[1] - shoulder_mid[1]
            # Angle relative to vertical axis (0 = perfectly vertical above/below shoulders)
            # We use atan2 of x vs -y so that small lateral offsets also contribute
            front_angle_rad = np.arctan2(vx, -vy)
            front_angle_deg = float(abs(front_angle_rad * 180.0 / np.pi))

            # If calibration exists, score based on deviation from baseline
            if CALIBRATION.get('front_angle_deg') is not None:
                baseline = CALIBRATION['front_angle_deg']
                tol = CALIBRATION.get('tolerance_deg', 12.0)
                diff = abs(front_angle_deg - baseline)
                front_score = 1.0 - min(diff / max(tol, 1e-6), 1.0)

        # Calculate confidence scores for each metric
        neck_score = 1.0 - min(abs(170 - neck_angle) / 40.0, 1.0)
        back_score = 1.0 - min(abs(175 - upper_back_angle) / 40.0, 1.0)
        head_score = 1.0 - min(abs(180 - head_tilt) / 30.0, 1.0)
        shoulder_score = 1.0 - min(abs(180 - shoulder_angle) / 30.0, 1.0)
        
        # Weight the scores
        weights = {
            'neck': 0.30,
            'back': 0.30,
            'head': 0.15,
            'shoulder': 0.10,
            'front': 0.15
        }
        
        # Calculate weighted confidence score
        confidence = (
            (neck_score * weights['neck']) +
            (back_score * weights['back']) +
            (head_score * weights['head']) +
            (shoulder_score * weights['shoulder']) +
            (front_score * weights['front'])
        )
        
        # Clamp between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        # Simplified classification (just good/bad) with front-angle check emphasized if calibrated
        good_thresh = app.config.get('POSTURE_GOOD_CONFIDENCE_THRESHOLD', 0.7)
        posture = "Good Posture" if confidence > good_thresh else "Bad Posture"
        
        # Attach auxiliary details via function attributes for downstream display
        classify_posture.last_front_angle_deg = front_angle_deg
        classify_posture.last_front_score = front_score
        
        return posture, confidence
            
    except KeyError:
        return "Unknown", 0.0

# Load YOLO and MediaPipe
if 'yolo_model' not in globals():
    yolo_model = YOLO('yolov8n.pt')
if 'mp_pose' not in globals():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.3,  # Further lowered for better side/back detection
        min_tracking_confidence=0.3   # Further lowered for better side/back detection
    )
    mp_drawing = mp.solutions.drawing_utils

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
        
        # YOLO detection
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
                        
                        # Cache latest points for calibration endpoint
                        try:
                            h2, w2 = person_img.shape[:2]
                            pts_cache = {}
                            for idx, lm in enumerate(landmark_results.pose_landmarks.landmark):
                                pts_cache[idx] = (int(lm.x * w2), int(lm.y * h2))
                            globals()['LATEST_POINTS'] = pts_cache
                        except Exception:
                            pass

                        # Classify posture
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
                        
                        # Display posture classification and confidence as percentage (+ front status if available)
                        front_label = ""
                        fa = getattr(classify_posture, 'last_front_angle_deg', None)
                        fs = getattr(classify_posture, 'last_front_score', None)
                        if fa is not None:
                            front_status = "Front OK" if (fs is None or fs >= 0.6) else "Front Bad"
                            front_label = f" | {front_status}"
                        label = f"{posture} ({confidence*100:.1f}%){front_label}"
                        color = (0, 255, 0) if "Good" in posture else (0, 0, 255)
                        
                        # Calculate text position (no frame flipping needed)
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                        text_x = x1  # Use original coordinates
                        
                        # Draw text background and label
                        cv2.rectangle(frame, (text_x, y1-text_height-10), (text_x + text_width, y1), (0, 0, 0), -1)
                        cv2.putText(frame, label, (text_x, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        

        
        # Removed multi-user display and facial recognition overlays
        
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

        # Flip frame horizontally to correct camera mirroring (after all text is drawn)
        # Only flip if mirroring is enabled in config
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
        except:
            return jsonify({
                'success': False,
                'message': 'Invalid birth date format'
            })
        
        success, result = create_user(
            username=data.get('username'),
            email=data.get('email'),
            password=data.get('password'),
            first_name=data.get('firstName'),
            last_name=data.get('lastName'),
            birth_date=birth_date,
            gender=data.get('gender')
        )
        
        if success:
            # Generate verification token
            verification_token = email_service.generate_verification_token()
            expires_at = datetime.now() + timedelta(hours=24)
            
            # Save verification token
            token_success, token_result = save_verification_token(result, verification_token, expires_at, "registration")
            
            if token_success:
                # Send verification email
                email_sent = email_service.send_verification_email(
                    data.get('email'), 
                    data.get('username'), 
                    verification_token, 
                    "registration"
                )
                
                if email_sent:
                    return jsonify({
                        'success': True,
                        'message': 'Account created successfully! Please check your email to verify your account.',
                        'redirect': url_for('login')
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Account created but failed to send verification email. Please contact support.'
                    })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Account created but failed to generate verification token.'
                })
        
        return jsonify({
            'success': False,
            'message': result
        })
    return render_template('create-account.html')

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/exercises')
@login_required
def exercises():
    success, exercises_list = get_exercises()
    return render_template('exercises.html', exercises=exercises_list if success else [])

@app.route('/posture-history')
@login_required
def posture_history():
    success, history = get_user_posture_history(session['user']['id'], limit=100)
    
    # Calculate statistics
    total_records = len(history) if success else 0
    good_posture_count = 0
    total_confidence = 0
    total_duration = 0
    total_good_time = 0
    total_bad_time = 0
    
    if success and history:
        for record in history:
            if 'Good' in record.get('posture_type', ''):
                good_posture_count += 1
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
    avg_confidence = (total_confidence / total_records * 100) if total_records > 0 else 0
    total_hours = total_duration / 3600 if total_duration else 0
    good_hours = total_good_time / 3600 if total_good_time else 0
    bad_hours = total_bad_time / 3600 if total_bad_time else 0
    
    stats = {
        'total_records': total_records,
        'good_posture_percentage': round(good_posture_percentage, 1),
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
            
            # Generate personalized exercise recommendations based on posture data
            recommended_exercises = generate_exercise_recommendations(record, history)
            
            if 'Good' in record.get('posture_type', ''):
                description = f"Excellent posture maintenance during this session! Your alignment was optimal with a confidence score of {confidence_pct:.1f}%. Keep up the good work!"
            else:
                description = f"Your posture during this session showed room for improvement with a confidence score of {confidence_pct:.1f}%. Based on your posture patterns, we recommend these specific exercises:"
                description += f"<br><br><strong>Recommended Exercises:</strong><br>"
                for exercise in recommended_exercises:
                    description += f"• {exercise}<br>"
            
            description += f"<br>Session duration: {duration_min:.1f} minutes."
            
            history_data[str(i)] = {
                'title': f"Posture Analysis - {record.get('recorded_at', 'Session')}",
                'description': description,
                'recommended_exercises': recommended_exercises
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

            # Handle based on method
            if method == 'immediate':
                # Immediate password change using bcrypt for consistency
                hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                cursor.execute(
                    "UPDATE users SET password = %s WHERE id = %s",
                    (hashed_password, user_id)
                )
                conn.commit()
                
                return jsonify({'success': True, 'message': 'Password updated successfully'})
                
            else:  # email verification method
                # Generate password change verification token
                verification_token = email_service.generate_verification_token()
                expires_at = datetime.now() + timedelta(hours=1)
                
                # Save verification token
                token_success, token_result = save_verification_token(user_id, verification_token, expires_at, "password_change")
                
                if token_success:
                    # Get user email for verification
                    cursor.execute("SELECT email, username FROM users WHERE id = %s", (user_id,))
                    user_info = cursor.fetchone()
                    
                    if user_info:
                        # Send verification email
                        email_sent = email_service.send_verification_email(
                            user_info['email'], 
                            user_info['username'], 
                            verification_token, 
                            "password_change"
                        )
                        
                        if email_sent:
                            return jsonify({
                                'success': True, 
                                'message': 'Password change request sent! Please check your email to complete the process.',
                                'requires_verification': True
                            })
                        else:
                            return jsonify({
                                'success': False, 
                                'message': 'Failed to send verification email. Please try again.'
                            })
                    else:
                        return jsonify({
                            'success': False, 
                            'message': 'User information not found.'
                        })
                else:
                    return jsonify({
                        'success': False, 
                        'message': 'Failed to generate verification token.'
                    })

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

@app.route('/calibrate_posture', methods=['POST'])
@login_required
def calibrate_posture():
    """Calibrate baseline front angle using the most recent detected landmarks.
    Ask user to sit upright and look straight ahead, then invoke this.
    """
    try:
        global CALIBRATION, LATEST_POINTS
        pts = LATEST_POINTS
        if not pts:
            return jsonify({'success': False, 'message': 'No person detected. Please face the camera and try again.'}), 400

        ls = pts.get(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        rs = pts.get(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        nose = pts.get(mp_pose.PoseLandmark.NOSE.value)
        if not (ls and rs and nose):
            return jsonify({'success': False, 'message': 'Insufficient landmarks for calibration. Ensure your shoulders and face are visible.'}), 400

        shoulder_mid = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        vx = nose[0] - shoulder_mid[0]
        vy = nose[1] - shoulder_mid[1]
        angle_rad = np.arctan2(vx, -vy)
        angle_deg = float(abs(angle_rad * 180.0 / np.pi))

        CALIBRATION['front_angle_deg'] = angle_deg
        CALIBRATION['timestamp'] = time.time()

        # Optionally accept custom tolerance from client
        data = request.get_json(silent=True) or {}
        tol = data.get('tolerance_deg')
        if isinstance(tol, (int, float)) and tol > 0:
            CALIBRATION['tolerance_deg'] = float(tol)

        return jsonify({'success': True, 'message': 'Calibration saved', 'front_angle_deg': angle_deg, 'tolerance_deg': CALIBRATION['tolerance_deg']})
    except Exception as e:
        logger.error(f"Calibration error: {e}")
        return jsonify({'success': False, 'message': 'Calibration failed'}), 500

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
            
            # Get detailed ML prediction if available
            ml_classifier = get_ml_classifier()
            detailed_info = {}
            if ml_classifier:
                detailed = ml_classifier.get_detailed_prediction()
                if detailed:
                    detailed_info = {
                        'orientation': detailed.get('orientation', 'unknown'),
                        'activity': detailed.get('activity', 'unknown'),
                        'orientation_boost': detailed.get('orientation_boost', 1.0),
                        'raw_confidence': detailed.get('raw_confidence', 0.0)
                    }
            
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
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if not file.filename.lower().endswith(tuple('.' + ext for ext in allowed_extensions)):
            return jsonify({'success': False, 'message': 'Invalid file type. Please upload PNG, JPG, JPEG, or GIF'})
        
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
                return jsonify({'success': False, 'message': 'Database error'})
            finally:
                conn.close()
        else:
            return jsonify({'success': False, 'message': 'Database connection failed'})
            
    except Exception as e:
        logger.error(f"Error uploading profile picture: {e}")
        return jsonify({'success': False, 'message': 'Error uploading profile picture'})

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
        required_fields = ['firstName', 'lastName', 'email', 'username']
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
            update_params = [
                data['firstName'],
                data['lastName'],
                data['email'],
                data['username'],
                data.get('birth_date') or current_user['birth_date'],  # Use existing value if not provided
                data.get('gender') or current_user['gender']  # Use existing value if not provided
            ]
            
            # Add password update if provided
            if data.get('password'):
                update_query += ", password = %s"
                update_params.append(generate_password_hash(data['password']))
            
            update_query += " WHERE id = %s"
            update_params.append(user_id)
            
            cursor.execute(update_query, tuple(update_params))
            conn.commit()
            
            # Get updated user data
            cursor.execute("""
                SELECT id, username, email, first_name, last_name, 
                       birth_date, gender, created_at, updated_at
                FROM users 
                WHERE id = %s
            """, (user_id,))
            
            updated_user = cursor.fetchone()
            
            if updated_user:
                # Update session data
                session['user'] = {
                    'id': updated_user['id'],
                    'username': updated_user['username'],
                    'email': updated_user['email']
                }
                
                logger.info(f"Successfully updated profile for user_id: {user_id}")
                return jsonify({
                    'success': True,
                    'message': 'Profile updated successfully',
                    'user': updated_user
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

@app.route('/admin/users/<int:user_id>/status', methods=['POST'])
@login_required
def toggle_user_status_admin(user_id):
    if session['user'].get('username') != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    data = request.get_json()
    if not data or 'status' not in data:
        return jsonify({'success': False, 'message': 'Status not provided'})
    
    success, message = toggle_user_status(user_id, data['status'])
    return jsonify({'success': success, 'message': message})

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

@app.route('/get-confidence-settings')
@login_required
def get_confidence_settings():
    """Get current confidence level settings"""
    try:
        current_threshold = app.config.get('POSTURE_GOOD_CONFIDENCE_THRESHOLD', 0.7)
        confidence_levels = app.config.get('CONFIDENCE_LEVELS', {
            '50%': 0.5,
            '70%': 0.7,
            '80%': 0.8,
            '90%': 0.9
        })
        
        # Find the current level name
        current_level = '70%'  # default
        for level_name, threshold in confidence_levels.items():
            if abs(threshold - current_threshold) < 0.01:
                current_level = level_name
                break
        
        return jsonify({
            'success': True,
            'current_level': current_level,
            'current_threshold': current_threshold,
            'available_levels': confidence_levels
        })
        
    except Exception as e:
        logger.error(f"Error getting confidence settings: {e}")
        return jsonify({'success': False, 'message': 'Error getting confidence settings'})

@app.route('/update-confidence-settings', methods=['POST'])
@login_required
def update_confidence_settings():
    """Update confidence level settings"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'})
        
        confidence_level = data.get('confidence_level')
        confidence_levels = app.config.get('CONFIDENCE_LEVELS', {
            '50%': 0.5,
            '70%': 0.7,
            '80%': 0.8,
            '90%': 0.9
        })
        
        if confidence_level not in confidence_levels:
            return jsonify({'success': False, 'message': 'Invalid confidence level'})
        
        new_threshold = confidence_levels[confidence_level]
        
        # Update the app configuration
        app.config['POSTURE_GOOD_CONFIDENCE_THRESHOLD'] = new_threshold
        
        logger.info(f"Confidence threshold updated to {confidence_level} ({new_threshold})")
        
        return jsonify({
            'success': True,
            'message': f'Confidence level updated to {confidence_level}',
            'confidence_level': confidence_level,
            'threshold': new_threshold
        })
        
    except Exception as e:
        logger.error(f"Error updating confidence settings: {e}")
        return jsonify({'success': False, 'message': 'Error updating confidence settings'})

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
            total_confidence = 0
            total_duration = 0
            
            if history:
                for record in history:
                    if 'Good' in record.get('posture_type', ''):
                        good_posture_count += 1
                    total_confidence += record.get('confidence_score', 0)
                    total_duration += record.get('session_duration', 0)
            
            # Calculate percentages and averages
            good_posture_percentage = (good_posture_count / total_records * 100) if total_records > 0 else 0
            avg_confidence = (total_confidence / total_records * 100) if total_records > 0 else 0
            total_hours = total_duration / 3600 if total_duration else 0
            
            stats = {
                'total_records': total_records,
                'good_posture_percentage': round(good_posture_percentage, 1),
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

if __name__ == '__main__':
    app.run(debug=True) 