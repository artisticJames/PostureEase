# PostureEase

Computer-vision powered posture assistant built with Python/Flask and YOLO/Mediapipe-based posture analysis.

## 🚀 Complete Installation & Setup Guide

### Prerequisites

Before starting, ensure you have:
- **Python 3.10+** installed
- **MySQL Server** running on your system
- **Webcam** connected to your computer
- **Git** installed (for cloning the repository)

### Step 1: Clone the Repository

```powershell
git clone https://github.com/artisticJames/PostureEase.git
cd PostureEase
```

### Step 2: Set Up Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\Activate.ps1

# If you get execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 3: Install Dependencies

```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Install additional packages that might be needed
pip install reportlab
```

### Step 4: Database Setup

#### 4.1: Install and Configure MySQL
1. **Download MySQL** from [mysql.com](https://dev.mysql.com/downloads/mysql/)
2. **Install MySQL Server** with default settings
3. **Set root password** during installation (remember this password!)
4. **Start MySQL service** (should start automatically)

#### 4.2: Create Database
```sql
-- Open MySQL Command Line Client or MySQL Workbench
-- Login with your root credentials
CREATE DATABASE posturease;
```

#### 4.3: Initialize Database Schema
```powershell
# Run database initialization scripts in order
python init_db.py
python migrate_email_verification.py
python update_db_schema.py
```

### Step 5: Configure Environment Variables (Optional but Recommended)

Create a `.env` file in the project root:

```env
# Database Configuration
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DB=posturease

# Flask Configuration
SECRET_KEY=your_secret_key_here

# Email Configuration (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=465
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
FROM_EMAIL=your_email@gmail.com
BASE_URL=http://localhost:5000

# Gemini AI Configuration (Optional)
GEMINI_API_KEY=your_gemini_api_key
```

### Step 6: Create Admin User (Optional)

```powershell
python create_admin.py
```

This creates an admin user with:
- **Username:** admin
- **Password:** admin123
- **Email:** admin@posturease.com

### Step 7: Run the Application

```powershell
python app.py
```

The application will start and you'll see output like:
```
INFO:__main__:Initial database connection successful
INFO:__main__:Loaded posture classifier from Training Again/runs/classify/train3/weights/best.pt
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

### Step 8: Access the Application

1. **Open your web browser**
2. **Navigate to:** `http://127.0.0.1:5000`
3. **Login** with your admin credentials or create a new account

## 🎯 How to Use PostureEase

### First Time Setup
1. **Create Account:** Click "Create Account" and fill in your details
2. **Login:** Use your credentials to access the dashboard
3. **Camera Setup:** Allow camera access when prompted
4. **Start Monitoring:** Click "Start Monitoring" to begin posture tracking

### Daily Usage
1. **Login** to your account
2. **Start a session** from the dashboard
3. **Position yourself** in front of the camera
4. **Maintain good posture** - the system will track your posture in real-time
5. **View your progress** in the Posture History section
6. **Export reports** to track your improvement over time

### Key Features
- **Real-time Posture Monitoring:** Live feedback on your posture
- **Posture History:** Track your progress over time
- **Exercise Recommendations:** Personalized exercises based on your posture patterns
- **Export Reports:** Generate detailed PDF reports
- **Admin Panel:** Manage users and view system statistics

## 🔧 Troubleshooting

### Common Issues

#### 1. Camera Not Working
```powershell
# Test camera access
python test_cameras.py
```

#### 2. Database Connection Issues
- Verify MySQL is running: `services.msc` → MySQL
- Check credentials in `.env` file
- Ensure database `posturease` exists

#### 3. Module Import Errors
```powershell
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

#### 4. Permission Errors (Windows)
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 5. Port Already in Use
```powershell
# Kill process using port 5000
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F
```

### Performance Optimization

#### For Better Performance:
1. **Use a good webcam** with at least 720p resolution
2. **Ensure good lighting** in your workspace
3. **Close unnecessary applications** to free up system resources
4. **Use a dedicated GPU** if available for faster AI processing

## 📊 System Requirements

### Minimum Requirements:
- **OS:** Windows 10/11, macOS 10.14+, Ubuntu 18.04+
- **Python:** 3.10 or higher
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 2GB free space
- **Camera:** Built-in or USB webcam
- **Internet:** Required for initial setup and AI features

### Recommended Requirements:
- **RAM:** 16GB or more
- **CPU:** Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **GPU:** NVIDIA GPU with CUDA support (for faster AI processing)
- **Camera:** 1080p webcam with good low-light performance

## 🆘 Getting Help

If you encounter issues:
1. **Check the logs** in the terminal output
2. **Verify all prerequisites** are installed correctly
3. **Check the troubleshooting section** above
4. **Create an issue** on the GitHub repository
5. **Check the documentation** in the `docs/` folder

## 🎉 Success!

Once everything is set up, you should see:
- ✅ Database connection successful
- ✅ Posture classifier loaded
- ✅ Flask app running on http://127.0.0.1:5000
- ✅ Camera access working
- ✅ Web interface accessible

Your PostureEase system is now ready to help you maintain better posture! 🚀

## Project structure (high level)

- `app.py` — Flask app entry point
- `templates/` and `static/` — UI templates and assets
- `TrainingModel/` — training scripts, datasets (ignored), and model artifacts (pkl)
  
- `EMAIL_SETUP.md` — email configuration guide

## Notes

- Git ignores large binaries and datasets (see `.gitignore`). Use Git LFS if you need to version model files.
- Python version: developed with Python 3.10.
- If you hit OpenCV/torch install issues, ensure you’re on a compatible Python version and have the latest `pip`:

```powershell
python -m pip install --upgrade pip
```

## Common commands

```powershell
# After making code changes
git add -A
git commit -m "Describe your change"
git push
```


## Configuration

- Copy these settings into an `.env` file in the project root (optional but recommended). The app reads values from `config.py`; using environment variables keeps secrets out of code.

Example `.env`:

```
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=yourpassword
MYSQL_DB=posturease

# Email (optional, for password reset / verification)
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=true
MAIL_USERNAME=youraddress@gmail.com
MAIL_PASSWORD=your_app_password
MAIL_DEFAULT_SENDER=youraddress@gmail.com
```

Configure the corresponding values in your local MySQL and email provider. See `EMAIL_SETUP.md` for details on Gmail/App Passwords and alternatives.

## Database

- Create the database schema by running the helper scripts in order:

```powershell
python init_db.py
python migrate_email_verification.py
python update_db_schema.py
```

These scripts establish the `users`, `posture_records`, `user_exercises`, and related tables used by the app. The code in `db.py` uses parameterized queries via `cursor.execute()` to prevent SQL injection and ensure correct typing.

## Models and datasets

- Required runtime models (not stored in Git):
  - `yolov8n.pt` in the repo root.
  - `TrainingModel/posture_model.pkl` in the `TrainingModel` folder.
- Optional training datasets (ignored by Git):
  - `TrainingModel/SittingData` and `TrainingModel/StandingData`.
- If you plan to re-train:

```powershell
cd TrainingModel
pip install -r requirements.txt
python train_model.py
```

## Troubleshooting

- OpenCV/torch install issues: upgrade pip and ensure Python 3.10.

```powershell
python -m pip install --upgrade pip
```

- MySQL connection errors: verify `.env` values and that MySQL is running and accessible; confirm user/password and database exist.
- Email sending issues: double-check SMTP host/port/TLS and use an App Password for Gmail.
- Large file warnings: models/datasets are ignored by `.gitignore`. Use Git LFS if you must version them.

## Contributing

PRs welcome. Please open an issue first for significant changes. Make sure to update tests or add minimal repro steps in the PR description.

## License

Choose a license (MIT/Apache-2.0) and add a `LICENSE` file. If unspecified, the repository defaults to “All rights reserved.”


