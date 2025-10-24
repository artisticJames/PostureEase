# Email Verification Setup Guide

This guide will help you set up email verification for the PosturEase system.

## Features Added

1. **Email Verification for Registration**: New users must verify their email before they can log in
2. **Email Verification for Password Changes**: Users must verify via email when changing passwords
3. **Secure Token System**: Time-limited verification tokens for security

## Setup Instructions

### 1. Database Migration

First, run the database migration to add the required columns:

```bash
python migrate_email_verification.py
```

### 2. Email Configuration

Create a `.env` file in the root directory with the following email settings:

```env
# Database Configuration
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=james-2003
MYSQL_DB=posturease

# Flask Configuration
SECRET_KEY=your-secret-key-here

# Email Configuration (Gmail Example)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=465
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
FROM_EMAIL=your-email@gmail.com
BASE_URL=http://localhost:5000
```

### 3. Gmail Setup (Recommended)

To use Gmail for sending emails:

1. **Enable 2-Factor Authentication** on your Gmail account
2. **Generate an App Password**:
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate a new app password for "Mail"
   - Use this password as `SMTP_PASSWORD` in your .env file

### 4. Alternative Email Providers

You can use other email providers by changing the SMTP settings:

**Outlook/Hotmail:**
```env
SMTP_SERVER=smtp-mail.outlook.com
SMTP_PORT=587
```

**Yahoo:**
```env
SMTP_SERVER=smtp.mail.yahoo.com
SMTP_PORT=587
```

### 5. Install Dependencies

Make sure you have the required packages:

```bash
pip install -r requirements.txt
```

## How It Works

### Registration Flow
1. User creates account
2. System generates verification token
3. Verification email is sent
4. User clicks link in email
5. Email is verified and user can log in

### Password Change Flow
1. User requests password change
2. System generates verification token
3. Verification email is sent
4. User clicks link and enters new password
5. Password is updated and confirmation email sent

## Security Features

- **Time-limited tokens**: Registration tokens expire in 24 hours, password change tokens in 1 hour
- **Secure token generation**: Uses cryptographically secure random tokens
- **Email validation**: Ensures email addresses are properly formatted
- **Password requirements**: Enforces strong password policies

## Troubleshooting

### Email Not Sending
1. Check your SMTP settings in the .env file
2. Verify your email provider's SMTP settings
3. For Gmail, ensure you're using an App Password, not your regular password
4. Check firewall settings

### Database Errors
1. Run the migration script: `python migrate_email_verification.py`
2. Check your database connection settings
3. Ensure the MySQL server is running

### Token Expired
- Registration tokens expire after 24 hours
- Password change tokens expire after 1 hour
- Users will need to request new tokens if expired

## Files Modified/Added

- `email_service.py` - Email sending functionality
- `config.py` - Added email configuration
- `db.py` - Added email verification functions
- `app.py` - Updated registration and password change routes
- `templates/verify-password-change.html` - Password change verification page
- `migrate_email_verification.py` - Database migration script
- `requirements.txt` - Updated dependencies
