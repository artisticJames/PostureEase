from flask import Flask, render_template, request, redirect, url_for, flash, session, Response, jsonify
import os
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import math
import time
from config import Config
from db import create_user, verify_user, save_posture_record, get_user_posture_history, get_exercises, record_completed_exercise, update_user, get_all_users, admin_create_user, delete_user, toggle_user_status, get_db_connection, test_connection, save_verification_token, verify_email_token, verify_password_change_token, clear_password_change_token, get_user_by_email, clear_user_posture_history
from datetime import datetime, timedelta
from functools import wraps
import logging
import re
from werkzeug.security import generate_password_hash, check_password_hash
import bcrypt
from face_recognition_module import face_recognition_system
from security_logger import security_logger
from email_service import email_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Face registration configuration
FACE_SAMPLES_REQUIRED = 4

# Ensure the secret key is set
if not app.secret_key:
    app.secret_key = os.urandom(24)

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
    """Classify posture based on multiple body angles and measurements"""
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
            'neck': 0.35,
            'back': 0.35,
            'head': 0.15,
            'shoulder': 0.15
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
        if confidence > 0.7:
            posture = "Good Posture"
        else:
            posture = "Bad Posture"
        
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
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_drawing = mp.solutions.drawing_utils

def gen_frames():
    cap = cv2.VideoCapture(0)
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
        
        # Face Recognition - Identify users in frame
        identified_users = face_recognition_system.identify_users_in_frame(frame)
        
        # YOLO detection
        results = yolo_model(frame, verbose=False)
        persons = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Only process if it's a person (class 0)
                if int(box.cls) == 0 and box.conf[0] > 0.7:
                    persons.append(box.xyxy[0].cpu().numpy())
        
        # Process each detected person
        for person_box in persons:
            x1, y1, x2, y2 = map(int, person_box)
            
            # Ensure the box coordinates are within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Skip if box is too small
            if x2 - x1 < 50 or y2 - y1 < 50:
                continue
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                continue
                
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
                
                # Classify posture
                posture, confidence = classify_posture(landmark_results.pose_landmarks, person_img.shape)
                
                # Update session data
                if "Good" in posture:
                    current_session_data['posture_counts']['good'] += 1
                else:
                    current_session_data['posture_counts']['bad'] += 1
                    # Trigger alarm for bad posture
                    gen_frames.alarm_triggered = True
                
                current_session_data['last_posture'] = posture
                
                # Display posture classification and confidence as percentage
                label = f"{posture} ({confidence*100:.1f}%)"
                color = (0, 255, 0) if "Good" in posture else (0, 0, 255)
                
                # Calculate text position (no frame flipping needed)
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                text_x = x1  # Use original coordinates
                
                # Draw text background and label
                cv2.rectangle(frame, (text_x, y1-text_height-10), (text_x + text_width, y1), (0, 0, 0), -1)
                cv2.putText(frame, label, (text_x, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        

        
        # User count display in top-right corner
        user_count = len(identified_users)
        if user_count > 0:
            count_text = f"Users: {user_count}"
            (count_width, count_height), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            # Position in top-right corner
            count_x = frame.shape[1] - count_width - 20
            count_y = 40
            
            # Draw background rectangle
            cv2.rectangle(frame, 
                         (count_x - 10, count_y - count_height - 10), 
                         (count_x + count_width + 10, count_y + 10), 
                         (0, 0, 0), -1)
            
            # Draw count text
            cv2.putText(frame, count_text, (count_x, count_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Enhanced Face Recognition Display (integrated with AI)
        for user_name, face_location, user_id in identified_users:
            top, right, bottom, left = face_location
            
            # Use original coordinates (no flipping needed)
            box_left = left
            box_right = right
            
            # Enhanced styling based on user type
            if user_name == "Unknown User":
                color = (0, 0, 255)  # Red for unknown
                status_text = "UNREGISTERED"
                status_color = (0, 0, 255)
                # Log unknown user detection (limit frequency to avoid spam)
                if not hasattr(gen_frames, 'last_unknown_log') or time.time() - gen_frames.last_unknown_log > 30:
                    security_logger.log_unknown_user_detected()
                    gen_frames.last_unknown_log = time.time()
            else:
                color = (0, 255, 0)  # Green for known
                status_text = "REGISTERED"
                status_color = (0, 255, 0)
                # Log recognized user (limit frequency)
                if not hasattr(gen_frames, 'last_recognized_log') or time.time() - gen_frames.last_recognized_log > 60:
                    security_logger.log_face_recognized(user_id, user_name)
                    gen_frames.last_recognized_log = time.time()
            
            # Draw enhanced box around face (thicker, more prominent)
            cv2.rectangle(frame, (box_left, top), (box_right, bottom), color, 3)
            
            # Simple corner indicators for better visibility
            corner_length = 15
            thickness = 2
            # Top-left corner
            cv2.line(frame, (box_left, top), (box_left + corner_length, top), color, thickness)
            cv2.line(frame, (box_left, top), (box_left, top + corner_length), color, thickness)
            # Top-right corner
            cv2.line(frame, (box_right - corner_length, top), (box_right, top), color, thickness)
            cv2.line(frame, (box_right, top), (box_right, top + corner_length), color, thickness)
            
            # Enhanced name label with background
            name_label = f"User: {user_name}"
            (name_width, name_height), _ = cv2.getTextSize(name_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Draw background rectangle for name
            cv2.rectangle(frame, 
                         (box_left, top - name_height - 25), 
                         (box_left + name_width + 10, top), 
                         (0, 0, 0), -1)
            
            # Draw name text
            cv2.putText(frame, name_label, (box_left + 5, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Status indicator below name
            status_label = f"Status: {status_text}"
            (status_width, status_height), _ = cv2.getTextSize(status_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw background rectangle for status
            cv2.rectangle(frame, 
                         (box_left, top - name_height - status_height - 35), 
                         (box_left + status_width + 10, top - name_height - 25), 
                         (0, 0, 0), -1)
            
            # Draw status text
            cv2.putText(frame, status_label, (box_left + 5, top - name_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Save posture data periodically
        if current_time - last_save_time >= save_interval and current_session_data['total_frames'] > 0:
            # Calculate session statistics
            total_postures = current_session_data['posture_counts']['good'] + current_session_data['posture_counts']['bad']
            if total_postures > 0:
                good_percentage = (current_session_data['posture_counts']['good'] / total_postures) * 100
                session_duration = current_time - current_session_data['start_time']
                
                # Determine overall posture status for this session
                if good_percentage >= 80:
                    overall_posture = "Good Posture"
                elif good_percentage >= 60:
                    overall_posture = "Fair Posture"
                else:
                    overall_posture = "Poor Posture"
                
                # Store session data for later saving (avoid session context issues)
                # We'll save this data when the user stops monitoring
                current_session_data['pending_save'] = {
                    'posture_type': overall_posture,
                    'confidence_score': good_percentage / 100.0,
                    'session_duration': session_duration,
                    'corrections_count': current_session_data['posture_counts']['bad']
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
    
    if success and history:
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
    
    # Prepare history data for JavaScript
    history_data = {}
    if success and history:
        for i, record in enumerate(history, 1):
            confidence_pct = (record.get('confidence_score', 0) * 100) if record.get('confidence_score') else 0
            duration_min = (record.get('session_duration', 30) / 60) if record.get('session_duration') else 30
            
            if 'Good' in record.get('posture_type', ''):
                description = f"Excellent posture maintenance during this session! Your alignment was optimal with a confidence score of {confidence_pct:.1f}%. Keep up the good work!"
            else:
                description = f"Your posture during this session showed room for improvement with a confidence score of {confidence_pct:.1f}%. Recommendations: Take regular breaks, perform stretching exercises, and maintain proper ergonomic positioning."
            
            description += f" Session duration: {duration_min:.1f} minutes."
            
            history_data[str(i)] = {
                'title': f"Posture Analysis - {record.get('recorded_at', 'Session')}",
                'description': description
            }
    
    return render_template('posture-history.html', 
                         history=history if success else [], 
                         stats=stats,
                         history_data=history_data)

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

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
            corrections_count=data.get('corrections_count', 0)
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

@app.route('/get-current-posture')
def get_current_posture():
    """Get current posture data for frontend monitoring"""
    try:
        # This would ideally get data from the video processing
        # For now, we'll return a simple response
        return jsonify({
            'success': True,
            'posture_data': {
                'posture_quality': 'good',  # This would come from actual detection
                'confidence': 0.85,
                'timestamp': time.time()
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

@app.route('/admin/users/<int:user_id>', methods=['DELETE'])
@login_required
def delete_user_admin(user_id):
    if session['user'].get('username') != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    success, message = delete_user(user_id)
    
    if success:
        # Reload face recognition data to remove deleted user's face encodings
        try:
            face_recognition_system.load_known_faces()
            logger.info(f"Face recognition data reloaded after deleting user {user_id}")
        except Exception as e:
            logger.error(f"Error reloading face recognition data: {e}")
    
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

# Face Recognition Endpoints
@app.route('/register-face', methods=['POST'])
@login_required
def register_user_face():
    """Register user's face during profile setup"""
    try:
        if 'face_image' not in request.files:
            return jsonify({'success': False, 'message': 'No face image provided'})
        
        file = request.files['face_image']
        user_id = session['user']['id']
        
        # Process face image
        success = face_recognition_system.register_user_face(user_id, file)
        
        if success:
            # Log successful face registration
            user_info = face_recognition_system.get_user_by_id(user_id)
            username = user_info['username'] if user_info else 'Unknown'
            security_logger.log_face_registered(user_id, username, request.remote_addr)

            # Return progress info
            samples_collected = 0
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM user_face_embeddings WHERE user_id = %s", (user_id,))
                    row = cursor.fetchone()
                    samples_collected = int(row[0]) if row else 0
                finally:
                    conn.close()
            completed = samples_collected >= FACE_SAMPLES_REQUIRED
            
            return jsonify({
                'success': True,
                'message': 'Face registered successfully',
                'samples_collected': samples_collected,
                'samples_required': FACE_SAMPLES_REQUIRED,
                'completed': completed
            })
        else:
            # Log failed face registration
            user_info = face_recognition_system.get_user_by_id(user_id)
            username = user_info['username'] if user_info else 'Unknown'
            security_logger.log_system_error(f"Failed to register face for user {username}", request.remote_addr)
            
            error_msg = getattr(face_recognition_system, 'last_error_message', None) or 'Failed to register face'
            return jsonify({'success': False, 'message': error_msg})
            
    except Exception as e:
        logger.error(f"Error registering face: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/face-registration-status', methods=['GET'])
@login_required
def face_registration_status():
    """Get number of face samples stored for the current user."""
    try:
        user_id = session['user']['id']
        samples_collected = 0
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM user_face_embeddings WHERE user_id = %s", (user_id,))
                row = cursor.fetchone()
                samples_collected = int(row[0]) if row else 0
            finally:
                conn.close()
        return jsonify({
            'success': True,
            'samples_collected': samples_collected,
            'samples_required': FACE_SAMPLES_REQUIRED,
            'completed': samples_collected >= FACE_SAMPLES_REQUIRED
        })
    except Exception as e:
        logger.error(f"Error getting face registration status: {e}")
        return jsonify({'success': False, 'message': 'Error checking status'})

@app.route('/face-quality', methods=['POST'])
@login_required
def face_quality():
    """Real-time face quality assessment endpoint.

    Accepts raw image bytes (multipart form-data field 'frame') and returns
    brightness, blur, face size, count, and guidance.
    """
    try:
        if 'frame' not in request.files:
            return jsonify({'success': False, 'message': 'No frame provided'})
        frame_file = request.files['frame']
        data = frame_file.read()
        quality = face_recognition_system.assess_quality_from_bytes(data)
        return jsonify({'success': True, 'quality': quality})
    except Exception as e:
        logger.error(f"Error assessing face quality: {e}")
        return jsonify({'success': False, 'message': 'Error assessing quality'})

@app.route('/get-identified-users')
@login_required
def get_identified_users():
    """Get list of currently identified users in the camera feed"""
    try:
        # This would typically be called from the frontend to get real-time user identification
        # For now, return the list of known users
        known_users = []
        for i, username in enumerate(face_recognition_system.known_face_names):
            user_info = face_recognition_system.get_user_by_id(face_recognition_system.known_face_user_ids[i])
            if user_info:
                known_users.append({
                    'id': user_info['id'],
                    'username': user_info['username'],
                    'profile_picture': user_info.get('profile_picture', '')
                })
        
        return jsonify({
            'success': True,
            'known_users': known_users,
            'total_known': len(known_users)
        })
        
    except Exception as e:
        logger.error(f"Error getting identified users: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

# Security Dashboard Endpoint (Admin Only)
@app.route('/admin/security-logs')
@login_required
def get_security_logs():
    """Get security logs for admin dashboard"""
    try:
        # Check if user is admin
        if session['user'].get('username') != 'admin':
            return jsonify({'success': False, 'message': 'Unauthorized'})
        
        # Get recent security events
        events = security_logger.get_recent_security_events(limit=100)
        
        # Get security statistics
        stats = security_logger.get_security_stats(days=7)
        
        return jsonify({
            'success': True,
            'events': events,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting security logs: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/admin/reload-face-recognition', methods=['POST'])
@login_required
def reload_face_recognition():
    """Manually reload face recognition data (admin only)"""
    try:
        # Check if user is admin
        if session['user'].get('username') != 'admin':
            return jsonify({'success': False, 'message': 'Unauthorized'})
        
        # Reload face recognition data
        face_count = face_recognition_system.reload_known_faces()
        
        return jsonify({
            'success': True,
            'message': f'Face recognition data reloaded successfully. {face_count} faces loaded.',
            'face_count': face_count
        })
        
    except Exception as e:
        logger.error(f"Error reloading face recognition data: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

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