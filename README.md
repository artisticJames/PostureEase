# PostureEase

Computer-vision powered posture assistant built with Python/Flask and YOLO/Mediapipe-based posture analysis.

## Quick start (Windows / PowerShell)

1) Clone the repo

```powershell
git clone https://github.com/artisticJames/PostureEase.git
cd PostureEase
```

2) Create and activate a virtual environment (Python 3.10+ recommended)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

3) Install dependencies

```powershell
pip install -r requirements.txt
```

4) Provide required model files (not stored in Git)

- Download/obtain the following files and place them in these paths:
  - `yolov8n.pt` → project root: `PostureEase\yolov8n.pt`
  - `TrainingModel\posture_model.pkl` → `PostureEase\TrainingModel\posture_model.pkl`
- Large datasets in `TrainingModel\SittingData` and `TrainingModel\StandingData` are optional and ignored by Git. You only need them if you plan to retrain models.

5) (Optional) Configure email if you want password reset/verification

- See `EMAIL_SETUP.md` for SMTP setup.
- You can also create a `.env` file (not committed) to store secrets.

6) Initialize the database

```powershell
python init_db.py
# If prompted for migrations in future versions:
python migrate_email_verification.py
python update_db_schema.py
```

7) (Optional) Create an admin user

```powershell
python create_admin.py
```

8) Run the app

```powershell
python app.py
```

The app should start on `http://127.0.0.1:5000/`.

## Project structure (high level)

- `app.py` — Flask app entry point
- `templates/` and `static/` — UI templates and assets
- `TrainingModel/` — training scripts, datasets (ignored), and model artifacts (pkl)
- `face_recognition_module.py` — face recognition logic (see plan in `FACE_RECOGNITION_IMPLEMENTATION_PLAN.md`)
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


