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


