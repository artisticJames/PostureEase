-- PostureEase Database Schema
-- This file contains the complete database structure for the PostureEase system

-- Create database
CREATE DATABASE IF NOT EXISTS posturease;
USE posturease;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    birth_date DATE,
    gender VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL,
    email_verified BOOLEAN DEFAULT FALSE,
    verification_token VARCHAR(255) NULL,
    token_expires TIMESTAMP NULL,
    password_change_token VARCHAR(255) NULL,
    password_token_expires TIMESTAMP NULL
);

-- Posture records table
CREATE TABLE IF NOT EXISTS posture_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    posture_type VARCHAR(20) NOT NULL,
    confidence_score FLOAT NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_duration INT DEFAULT 0,
    corrections_count INT DEFAULT 0,
    good_time INT DEFAULT 0,
    bad_time INT DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Exercises table
CREATE TABLE IF NOT EXISTS exercises (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    duration INT,
    difficulty VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User exercises table (for tracking completed exercises)
CREATE TABLE IF NOT EXISTS user_exercises (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    exercise_id INT NOT NULL,
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (exercise_id) REFERENCES exercises(id) ON DELETE CASCADE
);

-- User face embeddings table for face recognition encodings
CREATE TABLE IF NOT EXISTS user_face_embeddings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    face_encoding LONGBLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_face_user_id (user_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Insert default exercises
INSERT IGNORE INTO exercises (name, description, duration, difficulty) VALUES
('Neck Stretch', 'Gently stretch your neck from side to side', 5, 'Easy'),
('Shoulder Roll', 'Roll your shoulders forward and backward', 5, 'Easy'),
('Back Extension', 'Stand straight and extend your back', 5, 'Medium'),
('Core Strengthening', 'Basic core exercises for better posture', 10, 'Medium');

-- Create indexes for better performance
CREATE INDEX idx_posture_records_user_id ON posture_records(user_id);
CREATE INDEX idx_posture_records_recorded_at ON posture_records(recorded_at);
CREATE INDEX idx_user_exercises_user_id ON user_exercises(user_id);
CREATE INDEX idx_user_exercises_exercise_id ON user_exercises(exercise_id);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_email_verified ON users(email_verified);
