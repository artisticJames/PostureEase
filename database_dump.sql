-- PostureEase Database Dump
-- Generated on: 2025-10-25 23:24:52.640764
-- This file contains all data from your PostureEase database

-- Create database
CREATE DATABASE IF NOT EXISTS posturease;
USE posturease;

-- Table: ergonomic_recommendations
CREATE TABLE `ergonomic_recommendations` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `recommendation_type` enum('workstation','posture','equipment','routine','environment') NOT NULL,
  `title` varchar(255) NOT NULL,
  `description` text NOT NULL,
  `priority` enum('low','medium','high','critical') DEFAULT 'medium',
  `category` varchar(100) NOT NULL,
  `is_implemented` tinyint(1) DEFAULT '0',
  `implementation_date` timestamp NULL DEFAULT NULL,
  `effectiveness_rating` tinyint DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  CONSTRAINT `ergonomic_recommendations_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- No data in ergonomic_recommendations

-- Table: exercise_suggestion_analytics
CREATE TABLE `exercise_suggestion_analytics` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `total_suggestions` int DEFAULT '0',
  `helpful_suggestions` int DEFAULT '0',
  `completed_exercises` int DEFAULT '0',
  `skipped_exercises` int DEFAULT '0',
  `last_updated` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  CONSTRAINT `exercise_suggestion_analytics_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- No data in exercise_suggestion_analytics

-- Table: exercise_suggestions
CREATE TABLE `exercise_suggestions` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `session_id` varchar(100) DEFAULT NULL,
  `posture_quality` varchar(20) NOT NULL,
  `orientation` varchar(20) NOT NULL,
  `activity` varchar(20) NOT NULL,
  `bad_posture_duration` int DEFAULT '0',
  `confidence_score` float DEFAULT '0',
  `suggested_exercises` json DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  CONSTRAINT `exercise_suggestions_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- No data in exercise_suggestions

-- Table: exercises
CREATE TABLE `exercises` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `description` text,
  `duration` int DEFAULT NULL,
  `difficulty` varchar(20) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=41 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Data for table: exercises
INSERT INTO exercises (id, name, description, duration, difficulty, created_at) VALUES
(1, 'Neck Stretch', 'Gently stretch your neck from side to side', 5, 'Easy', '2025-06-13 23:19:37'),
(2, 'Shoulder Roll', 'Roll your shoulders forward and backward', 5, 'Easy', '2025-06-13 23:19:37'),
(3, 'Back Extension', 'Stand straight and extend your back', 5, 'Medium', '2025-06-13 23:19:37'),
(4, 'Core Strengthening', 'Basic core exercises for better posture', 10, 'Medium', '2025-06-13 23:19:37'),
(5, 'Neck Stretch', 'Gently stretch your neck from side to side', 5, 'Easy', '2025-06-13 23:23:47'),
(6, 'Shoulder Roll', 'Roll your shoulders forward and backward', 5, 'Easy', '2025-06-13 23:23:47'),
(7, 'Back Extension', 'Stand straight and extend your back', 5, 'Medium', '2025-06-13 23:23:47'),
(8, 'Core Strengthening', 'Basic core exercises for better posture', 10, 'Medium', '2025-06-13 23:23:47'),
(9, 'Neck Stretch', 'Gently stretch your neck from side to side', 5, 'Easy', '2025-06-13 23:29:14'),
(10, 'Shoulder Roll', 'Roll your shoulders forward and backward', 5, 'Easy', '2025-06-13 23:29:14'),
(11, 'Back Extension', 'Stand straight and extend your back', 5, 'Medium', '2025-06-13 23:29:14'),
(12, 'Core Strengthening', 'Basic core exercises for better posture', 10, 'Medium', '2025-06-13 23:29:14'),
(13, 'Neck Stretch', 'Gently stretch your neck from side to side', 5, 'Easy', '2025-06-13 23:37:12'),
(14, 'Shoulder Roll', 'Roll your shoulders forward and backward', 5, 'Easy', '2025-06-13 23:37:12'),
(15, 'Back Extension', 'Stand straight and extend your back', 5, 'Medium', '2025-06-13 23:37:12'),
(16, 'Core Strengthening', 'Basic core exercises for better posture', 10, 'Medium', '2025-06-13 23:37:12'),
(17, 'Neck Stretch', 'Gently stretch your neck from side to side', 5, 'Easy', '2025-08-22 17:35:18'),
(18, 'Shoulder Roll', 'Roll your shoulders forward and backward', 5, 'Easy', '2025-08-22 17:35:18'),
(19, 'Back Extension', 'Stand straight and extend your back', 5, 'Medium', '2025-08-22 17:35:18'),
(20, 'Core Strengthening', 'Basic core exercises for better posture', 10, 'Medium', '2025-08-22 17:35:18'),
(21, 'Neck Stretch', 'Gently stretch your neck from side to side', 5, 'Easy', '2025-10-03 20:05:28'),
(22, 'Shoulder Roll', 'Roll your shoulders forward and backward', 5, 'Easy', '2025-10-03 20:05:28'),
(23, 'Back Extension', 'Stand straight and extend your back', 5, 'Medium', '2025-10-03 20:05:28'),
(24, 'Core Strengthening', 'Basic core exercises for better posture', 10, 'Medium', '2025-10-03 20:05:28'),
(25, 'Neck Stretch', 'Gently stretch your neck from side to side', 5, 'Easy', '2025-10-05 19:04:01'),
(26, 'Shoulder Roll', 'Roll your shoulders forward and backward', 5, 'Easy', '2025-10-05 19:04:01'),
(27, 'Back Extension', 'Stand straight and extend your back', 5, 'Medium', '2025-10-05 19:04:01'),
(28, 'Core Strengthening', 'Basic core exercises for better posture', 10, 'Medium', '2025-10-05 19:04:01'),
(29, 'Neck Stretch', 'Gently stretch your neck from side to side', 5, 'Easy', '2025-10-13 19:10:14'),
(30, 'Shoulder Roll', 'Roll your shoulders forward and backward', 5, 'Easy', '2025-10-13 19:10:14'),
(31, 'Back Extension', 'Stand straight and extend your back', 5, 'Medium', '2025-10-13 19:10:14'),
(32, 'Core Strengthening', 'Basic core exercises for better posture', 10, 'Medium', '2025-10-13 19:10:14'),
(33, 'Neck Stretch', 'Gently stretch your neck from side to side', 5, 'Easy', '2025-10-15 19:50:04'),
(34, 'Shoulder Roll', 'Roll your shoulders forward and backward', 5, 'Easy', '2025-10-15 19:50:04'),
(35, 'Back Extension', 'Stand straight and extend your back', 5, 'Medium', '2025-10-15 19:50:04'),
(36, 'Core Strengthening', 'Basic core exercises for better posture', 10, 'Medium', '2025-10-15 19:50:04'),
(37, 'Neck Stretch', 'Gently stretch your neck from side to side', 5, 'Easy', '2025-10-16 15:44:30'),
(38, 'Shoulder Roll', 'Roll your shoulders forward and backward', 5, 'Easy', '2025-10-16 15:44:30'),
(39, 'Back Extension', 'Stand straight and extend your back', 5, 'Medium', '2025-10-16 15:44:30'),
(40, 'Core Strengthening', 'Basic core exercises for better posture', 10, 'Medium', '2025-10-16 15:44:30');

-- 40 records inserted into exercises

-- Table: posture_records
CREATE TABLE `posture_records` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `posture_type` varchar(20) NOT NULL,
  `confidence_score` float NOT NULL,
  `recorded_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `session_duration` int DEFAULT '0',
  `corrections_count` int DEFAULT '0',
  `good_time` int DEFAULT '0',
  `bad_time` int DEFAULT '0',
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  CONSTRAINT `posture_records_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=494 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Data for table: posture_records
INSERT INTO posture_records (id, user_id, posture_type, confidence_score, recorded_at, session_duration, corrections_count, good_time, bad_time) VALUES
(200, 3, 'Good Posture', 0.791437, '2025-07-11 14:03:11', 1535, 1, 0, 0),
(201, 3, 'Good Posture', 0.77382, '2025-07-25 11:12:11', 1317, 6, 0, 0),
(202, 3, 'Fair Posture', 0.634544, '2025-07-27 20:47:11', 779, 2, 0, 0),
(203, 3, 'Fair Posture', 0.71827, '2025-07-06 10:48:11', 597, 6, 0, 0),
(204, 3, 'Poor Posture', 0.255095, '2025-07-15 22:42:11', 1296, 7, 0, 0),
(205, 3, 'Fair Posture', 0.558345, '2025-07-31 11:43:11', 1475, 0, 0, 0),
(206, 3, 'Poor Posture', 0.534738, '2025-07-24 22:22:11', 2404, 7, 0, 0),
(207, 3, 'Poor Posture', 0.515358, '2025-07-31 01:54:11', 1921, 5, 0, 0),
(208, 3, 'Good Posture', 0.814155, '2025-07-16 23:49:11', 1270, 8, 0, 0),
(209, 3, 'Poor Posture', 0.530094, '2025-07-13 22:23:11', 209, 7, 0, 0),
(210, 3, 'Good Posture', 0.937867, '2025-07-23 10:36:11', 1039, 3, 0, 0),
(211, 3, 'Good Posture', 0.840969, '2025-08-02 06:52:11', 2310, 2, 0, 0),
(212, 3, 'Fair Posture', 0.576816, '2025-07-18 09:10:11', 1339, 8, 0, 0),
(213, 3, 'Fair Posture', 0.637829, '2025-07-14 10:41:11', 1119, 4, 0, 0),
(214, 3, 'Good Posture', 0.872075, '2025-07-16 02:37:11', 2577, 5, 0, 0),
(215, 3, 'Fair Posture', 0.61762, '2025-07-15 02:53:11', 174, 5, 0, 0),
(216, 3, 'Poor Posture', 0.356701, '2025-07-15 18:40:11', 2225, 1, 0, 0),
(217, 3, 'Good Posture', 0.783528, '2025-07-26 09:23:11', 380, 2, 0, 0),
(218, 3, 'Good Posture', 0.756775, '2025-07-24 19:28:11', 218, 7, 0, 0),
(219, 3, 'Fair Posture', 0.716056, '2025-07-17 18:35:11', 1176, 7, 0, 0),
(220, 3, 'Fair Posture', 0.656445, '2025-07-25 11:42:11', 1923, 5, 0, 0),
(221, 3, 'Poor Posture', 0.274954, '2025-07-30 12:47:11', 120, 5, 0, 0),
(222, 3, 'Good Posture', 0.76791, '2025-07-25 00:08:11', 1411, 5, 0, 0),
(223, 3, 'Good Posture', 0.945412, '2025-07-11 18:41:11', 1877, 0, 0, 0),
(224, 3, 'Good Posture', 0.828379, '2025-07-27 22:10:11', 2486, 0, 0, 0),
(478, 3, 'Good', 0.88, '2025-10-13 19:22:08', NULL, NULL, NULL, NULL),
(488, 39, 'Fair Posture', 0.448276, '2025-10-16 22:45:02', 29, 3, 13, 14),
(489, 39, 'Good Posture', 0.818182, '2025-10-21 14:15:49', 22, 2, 18, 2),
(490, 39, 'Good Posture', 0.781818, '2025-10-22 11:00:48', 55, 7, 43, 10),
(491, 39, 'Fair Posture', 0.0, '2025-10-22 11:09:06', 4, 0, 0, 3),
(492, 39, 'Good Posture', 0.811715, '2025-10-22 11:17:34', 478, 37, 388, 88),
(493, 39, 'Poor Posture', 0.206897, '2025-10-25 01:39:32', 29, 1, 6, 22);

-- 32 records inserted into posture_records

-- Table: recommendation_feedback
CREATE TABLE `recommendation_feedback` (
  `id` int NOT NULL AUTO_INCREMENT,
  `recommendation_id` int NOT NULL,
  `user_id` int NOT NULL,
  `was_helpful` tinyint(1) DEFAULT NULL,
  `feedback_text` text,
  `submitted_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `recommendation_id` (`recommendation_id`),
  KEY `user_id` (`user_id`),
  CONSTRAINT `recommendation_feedback_ibfk_1` FOREIGN KEY (`recommendation_id`) REFERENCES `ergonomic_recommendations` (`id`) ON DELETE CASCADE,
  CONSTRAINT `recommendation_feedback_ibfk_2` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- No data in recommendation_feedback

-- Table: recommendation_templates
CREATE TABLE `recommendation_templates` (
  `id` int NOT NULL AUTO_INCREMENT,
  `recommendation_type` enum('workstation','posture','equipment','routine','environment') NOT NULL,
  `title` varchar(255) NOT NULL,
  `description` text NOT NULL,
  `category` varchar(100) NOT NULL,
  `priority` enum('low','medium','high','critical') DEFAULT 'medium',
  `conditions` json NOT NULL,
  `is_active` tinyint(1) DEFAULT '1',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Data for table: recommendation_templates
INSERT INTO recommendation_templates (id, recommendation_type, title, description, category, priority, conditions, is_active, created_at) VALUES
(1, 'workstation', 'Adjust Monitor Height', 'Position the top of your monitor at or slightly below eye level to reduce neck strain.', 'Monitor Setup', 'high', '{"monitor_height_cm": {"lt": 50}}', 1, '2025-09-30 13:36:12'),
(2, 'workstation', 'Increase Monitor Distance', 'Position your monitor at least 50-70cm away to reduce eye strain and improve posture.', 'Monitor Setup', 'medium', '{"monitor_distance_cm": {"lt": 50}}', 1, '2025-09-30 13:36:12'),
(3, 'posture', 'Take Regular Breaks', 'Take a 5-minute break every hour to stretch and move around.', 'Movement', 'high', '{"long_sessions": {"gt": 0}}', 1, '2025-09-30 13:36:12'),
(4, 'posture', 'Improve Posture Awareness', 'Consider posture training exercises and mindfulness techniques.', 'Awareness', 'medium', '{"frequent_corrections": {"gt": 0}}', 1, '2025-09-30 13:36:12'),
(5, 'routine', 'Implement Pomodoro Technique', 'Use 25-minute work blocks with 5-minute breaks to prevent prolonged sitting.', 'Work Routine', 'high', '{"average_session_duration": {"gt": 3600}}', 1, '2025-09-30 13:36:12'),
(6, 'environment', 'Improve Workspace Lighting', 'Add task lighting and reduce screen glare to reduce eye strain.', 'Lighting', 'medium', '{"lighting_quality": {"in": ["poor", "fair"]}}', 1, '2025-09-30 13:36:12');

-- 6 records inserted into recommendation_templates

-- Table: security_logs
CREATE TABLE `security_logs` (
  `id` int NOT NULL AUTO_INCREMENT,
  `event_type` enum('face_registered','face_recognized','unknown_user_detected','unauthorized_access','system_error') NOT NULL,
  `user_id` int DEFAULT NULL,
  `username` varchar(255) DEFAULT NULL,
  `details` text,
  `ip_address` varchar(45) DEFAULT NULL,
  `timestamp` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_event_type` (`event_type`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_timestamp` (`timestamp`)
) ENGINE=InnoDB AUTO_INCREMENT=140 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Data for table: security_logs
INSERT INTO security_logs (id, event_type, user_id, username, details, ip_address, timestamp) VALUES
(1, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-07 19:01:48'),
(2, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-07 19:02:28'),
(3, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-07 19:08:11'),
(4, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-07 19:08:15'),
(5, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-07 19:11:35'),
(6, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-07 19:11:56'),
(7, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-07 19:12:27'),
(8, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-07 19:13:59'),
(9, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-07 19:14:00'),
(10, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-07 19:14:29'),
(11, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-07 19:15:14'),
(12, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-07 19:15:14'),
(13, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-07 19:16:42'),
(14, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-07 19:18:07'),
(15, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-07 19:18:24'),
(16, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-07 19:19:06'),
(17, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-07 19:19:09'),
(18, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-07 19:20:10'),
(19, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-07 19:23:33'),
(20, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-07 19:26:15'),
(21, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-07 19:28:01'),
(22, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-07 19:30:21'),
(23, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-07 19:34:27'),
(24, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-11 15:56:09'),
(25, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-11 16:11:48'),
(26, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-11 16:29:54'),
(27, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-11 16:47:06'),
(28, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-11 16:47:07'),
(29, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-11 16:53:14'),
(30, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-11 16:53:15'),
(31, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-11 16:58:49'),
(32, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-11 16:58:56'),
(33, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-11 17:01:14'),
(34, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-11 17:01:18'),
(35, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-11 17:07:37'),
(36, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-11 17:07:38'),
(37, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-11 17:10:15'),
(38, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-11 17:11:08'),
(39, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-11 17:11:15'),
(40, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-11 17:23:34'),
(41, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-11 17:24:49'),
(42, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-11 17:26:46'),
(43, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 17:07:54'),
(44, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 17:08:25'),
(45, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-22 17:10:14'),
(46, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 17:10:16'),
(47, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 17:10:51'),
(48, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-22 17:11:49'),
(49, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 17:12:01'),
(50, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-22 17:13:15'),
(51, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-22 17:15:42'),
(52, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 17:16:12'),
(53, 'face_recognized', 10, 'James21', 'Face recognized for user James21', NULL, '2025-08-22 17:16:42'),
(54, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 17:16:49'),
(55, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 17:24:21'),
(56, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:25:52'),
(57, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:26:00'),
(58, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:26:10'),
(59, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:26:27'),
(60, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:27:06'),
(61, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 17:32:53'),
(62, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:33:29'),
(63, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:33:38'),
(64, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:33:48'),
(65, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:33:54'),
(66, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:34:50'),
(67, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 17:37:04'),
(68, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:37:32'),
(69, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:37:48'),
(70, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:38:01'),
(71, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:38:25'),
(72, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 17:41:47'),
(73, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 17:46:46'),
(74, 'face_registered', 14, 'LorenceCueva', 'Face registered for user LorenceCueva', '127.0.0.1', '2025-08-22 17:47:18'),
(75, 'face_registered', 14, 'LorenceCueva', 'Face registered for user LorenceCueva', '127.0.0.1', '2025-08-22 17:47:36'),
(76, 'face_registered', 14, 'LorenceCueva', 'Face registered for user LorenceCueva', '127.0.0.1', '2025-08-22 17:47:52'),
(77, 'face_registered', 14, 'LorenceCueva', 'Face registered for user LorenceCueva', '127.0.0.1', '2025-08-22 17:48:06'),
(78, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:48:27'),
(79, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:48:34'),
(80, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:48:42'),
(81, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:48:51'),
(82, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 17:49:01'),
(83, 'face_registered', 14, 'LorenceCueva', 'Face registered for user LorenceCueva', '127.0.0.1', '2025-08-22 17:49:07'),
(84, 'face_registered', 14, 'LorenceCueva', 'Face registered for user LorenceCueva', '127.0.0.1', '2025-08-22 17:49:17'),
(85, 'face_registered', 14, 'LorenceCueva', 'Face registered for user LorenceCueva', '127.0.0.1', '2025-08-22 17:49:24'),
(86, 'face_registered', 14, 'LorenceCueva', 'Face registered for user LorenceCueva', '127.0.0.1', '2025-08-22 17:49:31'),
(87, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 17:49:48'),
(88, 'face_recognized', 14, 'LorenceCueva', 'Face recognized for user LorenceCueva', NULL, '2025-08-22 17:49:50'),
(89, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 17:50:21'),
(90, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 17:51:03'),
(91, 'face_recognized', 14, 'LorenceCueva', 'Face recognized for user LorenceCueva', NULL, '2025-08-22 17:51:06'),
(92, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 17:51:33'),
(93, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 17:52:54'),
(94, 'face_recognized', 14, 'LorenceCueva', 'Face recognized for user LorenceCueva', NULL, '2025-08-22 17:52:55'),
(95, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 17:58:25'),
(96, 'face_recognized', 14, 'LorenceCueva', 'Face recognized for user LorenceCueva', NULL, '2025-08-22 17:58:27'),
(97, 'face_registered', 14, 'LorenceCueva', 'Face registered for user LorenceCueva', '127.0.0.1', '2025-08-22 17:58:52'),
(98, 'face_recognized', 14, 'LorenceCueva', 'Face recognized for user LorenceCueva', NULL, '2025-08-22 18:00:15'),
(99, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 18:00:26'),
(100, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 18:02:21'),
(101, 'face_recognized', 14, 'LorenceCueva', 'Face recognized for user LorenceCueva', NULL, '2025-08-22 18:02:24'),
(102, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 18:11:27'),
(103, 'face_registered', 17, 'LorenceCueva', 'Face registered for user LorenceCueva', '127.0.0.1', '2025-08-22 18:11:52'),
(104, 'face_registered', 17, 'LorenceCueva', 'Face registered for user LorenceCueva', '127.0.0.1', '2025-08-22 18:11:53'),
(105, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:11:54'),
(106, 'face_registered', 17, 'LorenceCueva', 'Face registered for user LorenceCueva', '127.0.0.1', '2025-08-22 18:11:55'),
(107, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:11:55'),
(108, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:11:56'),
(109, 'face_registered', 17, 'LorenceCueva', 'Face registered for user LorenceCueva', '127.0.0.1', '2025-08-22 18:11:57'),
(110, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 18:12:10'),
(111, 'face_recognized', 17, 'LorenceCueva', 'Face recognized for user LorenceCueva', NULL, '2025-08-22 18:12:13'),
(112, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 18:12:45'),
(113, 'face_recognized', 17, 'LorenceCueva', 'Face recognized for user LorenceCueva', NULL, '2025-08-22 18:13:14'),
(114, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 18:13:17'),
(115, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 18:13:48'),
(116, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:17:51'),
(117, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:17:52'),
(118, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:17:52'),
(119, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:17:53'),
(120, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:17:53'),
(121, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:17:54'),
(122, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:17:54'),
(123, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:17:56'),
(124, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:17:57'),
(125, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:17:58'),
(126, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:17:59'),
(127, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:17:59'),
(128, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:18:00'),
(129, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:18:02'),
(130, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:18:02'),
(131, 'system_error', NULL, NULL, 'Failed to register face for user LorenceCueva', '127.0.0.1', '2025-08-22 18:18:03'),
(132, 'face_registered', 17, 'LorenceCueva', 'Face registered for user LorenceCueva', '127.0.0.1', '2025-08-22 18:18:04'),
(133, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 18:34:52'),
(134, 'face_recognized', 17, 'LorenceCueva', 'Face recognized for user LorenceCueva', NULL, '2025-08-22 18:34:57'),
(135, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 18:43:56'),
(136, 'face_recognized', 17, 'LorenceCueva', 'Face recognized for user LorenceCueva', NULL, '2025-08-22 18:43:59'),
(137, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 18:49:26'),
(138, 'face_recognized', 17, 'LorenceCueva', 'Face recognized for user LorenceCueva', NULL, '2025-08-22 18:49:29'),
(139, 'unknown_user_detected', NULL, NULL, 'Unknown user detected in camera feed', NULL, '2025-08-22 19:25:08');

-- 139 records inserted into security_logs

-- Table: user_exercise_feedback
CREATE TABLE `user_exercise_feedback` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `suggestion_id` int NOT NULL,
  `exercise_id` int NOT NULL,
  `feedback_type` enum('helpful','not_helpful','completed','skipped') NOT NULL,
  `feedback_notes` text,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  KEY `suggestion_id` (`suggestion_id`),
  KEY `exercise_id` (`exercise_id`),
  CONSTRAINT `user_exercise_feedback_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`),
  CONSTRAINT `user_exercise_feedback_ibfk_2` FOREIGN KEY (`suggestion_id`) REFERENCES `exercise_suggestions` (`id`),
  CONSTRAINT `user_exercise_feedback_ibfk_3` FOREIGN KEY (`exercise_id`) REFERENCES `exercises` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- No data in user_exercise_feedback

-- Table: user_exercises
CREATE TABLE `user_exercises` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `exercise_id` int NOT NULL,
  `completed_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  KEY `exercise_id` (`exercise_id`),
  CONSTRAINT `user_exercises_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`),
  CONSTRAINT `user_exercises_ibfk_2` FOREIGN KEY (`exercise_id`) REFERENCES `exercises` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- No data in user_exercises

-- Table: user_face_embeddings
CREATE TABLE `user_face_embeddings` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `face_encoding` blob NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  CONSTRAINT `user_face_embeddings_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=16 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- No data in user_face_embeddings

-- Table: users
CREATE TABLE `users` (
  `id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL,
  `email` varchar(100) NOT NULL,
  `password` varchar(255) NOT NULL,
  `first_name` varchar(50) NOT NULL,
  `last_name` varchar(50) NOT NULL,
  `birth_date` date DEFAULT NULL,
  `gender` varchar(10) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `profile_picture` varchar(255) DEFAULT NULL,
  `email_verified` tinyint(1) DEFAULT '0',
  `verification_token` varchar(255) DEFAULT NULL,
  `token_expires` datetime DEFAULT NULL,
  `password_change_token` varchar(255) DEFAULT NULL,
  `password_token_expires` datetime DEFAULT NULL,
  `status` tinyint(1) DEFAULT '1',
  `last_login` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`),
  UNIQUE KEY `email` (`email`)
) ENGINE=InnoDB AUTO_INCREMENT=41 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Data for table: users
INSERT INTO users (id, username, email, password, first_name, last_name, birth_date, gender, created_at, updated_at, profile_picture, email_verified, verification_token, token_expires, password_change_token, password_token_expires, status, last_login) VALUES
(3, 'admin', 'admin@posturease.com', '$2b$12$MFk5UDpWB4N6xJq77iJ.JOKjOaw0eVGMIF6d914VclIFh7G5sScvu', 'Admin', 'User', '2000-01-01', 'other', '2025-06-13 23:37:19', '2025-10-20 15:23:50', NULL, 0, NULL, NULL, NULL, NULL, 1, '2025-10-20 15:23:50'),
(39, 'xel', 'wrexeljohn1231@gmail.com', '$2b$12$KvQ8Z3SQuIzPrN6JVrDKY.xotI868Zdpa69VccvfyG1xhJvmxap3i', 'john', 'wrexel', '2004-12-17', 'male', '2025-10-16 22:43:48', '2025-10-25 22:11:03', NULL, 0, NULL, NULL, NULL, NULL, 1, '2025-10-25 22:11:03'),
(40, 'Lorenz', 'lorencecueva2522@gmail.com', '$2b$12$GZvhOPPLIX0tFSNCb7eJbe0uRWNIdjaVttNztKrLVb501goDgA4FG', 'Liza', 'Cueva', '2002-10-17', 'male', '2025-10-19 16:43:45', '2025-10-19 16:54:15', NULL, 0, NULL, NULL, NULL, NULL, 1, '2025-10-19 16:54:15');

-- 3 records inserted into users

-- Table: workstation_profiles
CREATE TABLE `workstation_profiles` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `monitor_height_cm` decimal(5,2) DEFAULT NULL,
  `monitor_distance_cm` decimal(5,2) DEFAULT NULL,
  `desk_height_cm` decimal(5,2) DEFAULT NULL,
  `chair_height_cm` decimal(5,2) DEFAULT NULL,
  `keyboard_distance_cm` decimal(5,2) DEFAULT NULL,
  `mouse_distance_cm` decimal(5,2) DEFAULT NULL,
  `lighting_quality` enum('poor','fair','good','excellent') DEFAULT 'fair',
  `noise_level` enum('quiet','moderate','loud','very_loud') DEFAULT 'moderate',
  `air_quality` enum('poor','fair','good','excellent') DEFAULT 'fair',
  `room_temperature_c` decimal(4,1) DEFAULT NULL,
  `humidity_percent` decimal(4,1) DEFAULT NULL,
  `last_updated` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  CONSTRAINT `workstation_profiles_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- No data in workstation_profiles

