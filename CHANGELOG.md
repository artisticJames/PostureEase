Changelog
=========

This project follows a simple date-based changelog. See `README.md` for a shorter "What’s New" summary.

2025-10-08
----------
- Initialize Git repository with current working state
- Document setup in `README.md` and `EMAIL_SETUP.md`
- Add database helper scripts: `init_db.py`, `migrate_email_verification.py`, `update_db_schema.py`
- Add core Flask app `app.py` with routes for auth, profile, dashboard, posture history, and settings
- Include Jinja templates in `templates/` and static assets in `static/`
- Integrate email service (`email_service.py`) for verification and password reset
- Add YOLO/Mediapipe posture detection utilities and training assets under `Training Again/`
- Add basic tests: cameras, deletion, email, ML integration, and orientation detection
- Add `.gitignore` for environments, large datasets, models, logs, and local artifacts


