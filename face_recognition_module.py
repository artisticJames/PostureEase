import face_recognition
import cv2
import numpy as np
import mysql.connector
import pickle
from typing import List, Tuple, Optional, Deque, Dict
from collections import deque
from config import Config

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_user_ids = []
        # Recognition parameters
        self.match_threshold = 0.5  # stricter than default 0.6 for better precision
        self.min_face_size = 48  # minimum width/height in pixels to consider a face (relaxed)
        self.min_laplacian_var = 30.0  # blur threshold; lower means blurry (relaxed)
        self.min_brightness = 20  # reject very dark faces (relaxed)
        # Temporal smoothing storage keyed by quantized face location
        self._recent_votes: Dict[Tuple[int, int, int, int], Deque[int]] = {}
        self._votes_window = 5
        self._votes_required = 3
        # Error reporting
        self.last_error_message: Optional[str] = None
        self.load_known_faces()

    def assess_quality_from_rgb(self, image_rgb) -> dict:
        """Assess face capture quality from an RGB image array.

        Returns dict with keys: ok, brightness, blur, face_size, faces, suggestion.
        """
        result = {
            'ok': False,
            'brightness': None,
            'blur': None,
            'face_size': None,
            'faces': 0,
            'suggestion': 'No image'
        }

        if image_rgb is None or image_rgb.size == 0:
            result['suggestion'] = 'Invalid image'
            return result

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        brightness = float(np.mean(gray))
        result['brightness'] = brightness

        # Detect faces (use same detector as registration)
        face_locations = face_recognition.face_locations(image_rgb)
        result['faces'] = len(face_locations)
        if not face_locations:
            result['suggestion'] = 'No face detected. Center your face.'
            return result

        # Largest face metrics
        def _box_area(box: Tuple[int, int, int, int]) -> int:
            top, right, bottom, left = box
            return max(0, bottom - top) * max(0, right - left)

        face_locations.sort(key=_box_area, reverse=True)
        top, right, bottom, left = face_locations[0]
        face_width = right - left
        face_height = bottom - top
        min_side = min(face_width, face_height)
        result['face_size'] = min_side

        roi = gray[top:bottom, left:right]
        blur = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size > 0 else 0.0
        result['blur'] = blur

        # Evaluate thresholds
        if brightness < self.min_brightness:
            result['suggestion'] = 'Increase lighting.'
            return result
        if min_side < self.min_face_size:
            result['suggestion'] = 'Move closer to the camera.'
            return result

        # Lenient blur acceptance: allow slightly blurry but warn
        if blur < self.min_laplacian_var:
            if blur >= (self.min_laplacian_var * 0.6):
                result['ok'] = True
                result['suggestion'] = 'A bit blurry; try to hold still, but you can capture.'
                return result
            else:
                result['suggestion'] = 'Hold still or improve focus.'
                return result

        result['ok'] = True
        result['suggestion'] = 'Good quality. You can capture.'
        return result

    def assess_quality_from_bytes(self, image_bytes: bytes) -> dict:
        """Decode bytes to RGB and assess quality."""
        try:
            image_array = np.frombuffer(image_bytes, np.uint8)
            bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if bgr is None:
                return {'ok': False, 'suggestion': 'Invalid image'}
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return self.assess_quality_from_rgb(rgb)
        except Exception as e:
            return {'ok': False, 'suggestion': f'Quality check error: {e}'}
    
    def get_db_connection(self):
        """Get database connection"""
        db_config = {
            'host': Config.MYSQL_HOST,
            'user': Config.MYSQL_USER,
            'password': Config.MYSQL_PASSWORD,
            'database': Config.MYSQL_DB
        }
        return mysql.connector.connect(**db_config)
    
    def load_known_faces(self):
        """Load all known faces from database"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Get all face encodings with user info
            query = """
            SELECT ufe.face_encoding, u.username, u.id
            FROM user_face_embeddings ufe
            JOIN users u ON ufe.user_id = u.id
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            self.known_face_encodings = []
            self.known_face_names = []
            self.known_face_user_ids = []
            
            for face_encoding_blob, username, user_id in results:
                # Unpickle the face encoding
                face_encoding = pickle.loads(face_encoding_blob)
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(username)
                self.known_face_user_ids.append(user_id)
            
            print(f"✅ Loaded {len(self.known_face_encodings)} known faces")
            
        except Exception as e:
            print(f"❌ Error loading known faces: {e}")
            # Clear arrays on error to prevent stale data
            self.known_face_encodings = []
            self.known_face_names = []
            self.known_face_user_ids = []
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
    
    def reload_known_faces(self):
        """Manually reload known faces from database (useful after user deletion)"""
        print("🔄 Reloading known faces from database...")
        self.load_known_faces()
        return len(self.known_face_encodings)
    
    def register_user_face(self, user_id: int, face_image) -> bool:
        """Register a new user's face with quality checks.

        Accepts natural variations during capture (distance, tilt, blink) and
        stores an additional encoding per submission to improve robustness.
        """
        try:
            # Convert image to numpy array if it's a file
            if hasattr(face_image, 'read'):
                # It's a file object
                image_array = np.frombuffer(face_image.read(), np.uint8)
                face_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            elif isinstance(face_image, str):
                # It's a file path
                face_image = cv2.imread(face_image)
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Basic quality checks
            if face_image is None or face_image.size == 0:
                self.last_error_message = "Invalid image data"
                print("❌ Invalid image data")
                return False

            # Brightness check (grayscale mean)
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            brightness = float(np.mean(gray))
            if brightness < self.min_brightness:
                self.last_error_message = "Face image too dark; please move to better lighting"
                print("❌ Face image too dark; please move to better lighting")
                return False

            # Detect face in image
            face_locations = face_recognition.face_locations(face_image)
            if not face_locations:
                self.last_error_message = "No face detected; center your face and try again"
                print("❌ No face detected in image")
                return False
            
            # Choose the largest face (in case there are multiple)
            def _box_area(box: Tuple[int, int, int, int]) -> int:
                top, right, bottom, left = box
                return max(0, bottom - top) * max(0, right - left)

            face_locations.sort(key=_box_area, reverse=True)
            top, right, bottom, left = face_locations[0]

            # Size check
            if (right - left) < self.min_face_size or (bottom - top) < self.min_face_size:
                self.last_error_message = "Face too small; move closer to the camera"
                print("❌ Face too small; move closer to the camera")
                return False

            # Blur check using variance of Laplacian on the face ROI
            roi = gray[top:bottom, left:right]
            if roi.size == 0:
                self.last_error_message = "Invalid face region"
                print("❌ Invalid face region")
                return False
            lap_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())
            if lap_var < self.min_laplacian_var:
                self.last_error_message = "Image too blurry; hold still or increase lighting"
                print("❌ Image too blurry; hold still or increase lighting")
                return False

            # Extract face encoding
            face_encoding = face_recognition.face_encodings(face_image, [face_locations[0]])[0]
            
            # Store in database (append an additional encoding to improve future matches)
            success = self.save_face_encoding(user_id, face_encoding)
            if success:
                # Reload known faces
                self.load_known_faces()
            
            return success
            
        except Exception as e:
            self.last_error_message = f"Unexpected error: {e}"
            print(f"❌ Error registering face: {e}")
            return False
    
    def save_face_encoding(self, user_id: int, face_encoding) -> bool:
        """Save face encoding to database as an additional sample for the user."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Pickle the face encoding for storage
            face_encoding_blob = pickle.dumps(face_encoding)
            
            # Insert an additional encoding sample (no overwrite)
            query = """
            INSERT INTO user_face_embeddings (user_id, face_encoding)
            VALUES (%s, %s)
            """
            
            cursor.execute(query, (user_id, face_encoding_blob))
            conn.commit()
            
            print(f"✅ Face encoding saved for user {user_id}")
            return True
            
        except Exception as e:
            self.last_error_message = f"DB save error: {e}"
            print(f"❌ Error saving face encoding: {e}")
            return False
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()
    
    def identify_users_in_frame(self, frame) -> List[Tuple[str, Tuple[int, int, int, int], int]]:
        """Identify all users in a frame with distance-based matching and temporal smoothing."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find all faces in frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            identified_users = []
            
            for face_encoding, face_location in zip(face_encodings, face_locations):
                if len(self.known_face_encodings) == 0:
                    # No known faces, mark as unknown
                    identified_users.append(("Unknown User", face_location, -1))
                    continue
                
                # Distance to all known encodings
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if distances.size == 0:
                    identified_users.append(("Unknown User", face_location, -1))
                    continue

                best_index = int(np.argmin(distances))
                best_distance = float(distances[best_index])

                if best_distance <= self.match_threshold:
                    candidate_id = self.known_face_user_ids[best_index]
                    candidate_name = self.known_face_names[best_index]

                    # Temporal smoothing by quantized location key
                    key = self._quantize_location(face_location)
                    votes = self._recent_votes.setdefault(key, deque(maxlen=self._votes_window))
                    votes.append(candidate_id)
                    if self._has_majority(votes, candidate_id, self._votes_required):
                        identified_users.append((candidate_name, face_location, candidate_id))
                    else:
                        identified_users.append(("Unknown User", face_location, -1))
                else:
                    identified_users.append(("Unknown User", face_location, -1))
            
            return identified_users
            
        except Exception as e:
            print(f"❌ Error identifying users: {e}")
            return []

    def _quantize_location(self, box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Quantize face location to make a stable key for smoothing across frames."""
        top, right, bottom, left = box
        q = 16  # quantization step in pixels
        return (top // q, right // q, bottom // q, left // q)

    @staticmethod
    def _has_majority(votes: Deque[int], candidate: int, required: int) -> bool:
        """Return True if candidate appears at least 'required' times in votes."""
        count = sum(1 for v in votes if v == candidate)
        return count >= required
    
    def get_user_by_id(self, user_id: int) -> Optional[dict]:
        """Get user information by ID"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            query = "SELECT id, username, email, profile_picture FROM users WHERE id = %s"
            cursor.execute(query, (user_id,))
            user = cursor.fetchone()
            
            return user
            
        except Exception as e:
            print(f"❌ Error getting user: {e}")
            return None
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

# Global instance
face_recognition_system = FaceRecognitionSystem()
