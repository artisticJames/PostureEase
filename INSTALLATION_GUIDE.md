# PosturEase Update Installation Guide

## 📦 **Quick Installation Steps**

### 1. **Backup Current Files**
```bash
# Create backup directory
mkdir backup_$(date +%Y%m%d_%H%M%S)

# Backup current files
cp app.py backup_*/
cp templates/base.html backup_*/
cp templates/dashboard.html backup_*/
cp templates/exercises.html backup_*/
cp templates/posture-history.html backup_*/
cp templates/admin.html backup_*/
cp templates/login.html backup_*/
cp templates/settings.html backup_*/
```

### 2. **Replace Files**
```bash
# Extract the zip file
unzip PosturEase_Complete_Update.zip

# Copy files to your project directory
cp app.py /path/to/your/project/
cp templates/*.html /path/to/your/project/templates/
```

### 3. **Verify Installation**
```bash
# Check if files are in place
ls -la app.py
ls -la templates/

# Test the application
python app.py
```

## 🔍 **Testing Checklist**

### **PDF Export Test:**
1. Login to the application
2. Go to Settings page
3. Click "Export Data" button
4. Verify PDF downloads successfully
5. Check PDF contains all sections:
   - PosturEase branding
   - Executive summary
   - Risk assessment
   - Health concerns
   - Recommendations
   - Medical disclaimers

### **Favicon Test:**
1. Open any page in browser
2. Check browser tab shows PosturEase logo
3. Test on multiple pages:
   - Dashboard
   - Exercises
   - Settings
   - Admin (if applicable)

## 🚨 **Troubleshooting**

### **If PDF Export Fails:**
1. Check server logs for error messages
2. Verify all dependencies are installed:
   ```bash
   pip install reportlab
   pip install flask
   ```
3. Check file permissions
4. Ensure static/images/logo-test-2-1-10.png exists

### **If Favicon Doesn't Show:**
1. Clear browser cache (Ctrl+F5)
2. Check if logo file exists: `static/images/logo-test-2-1-10.png`
3. Verify file permissions
4. Try different browser

### **If Application Won't Start:**
1. Check Python syntax: `python -m py_compile app.py`
2. Verify all imports are available
3. Check Flask dependencies
4. Review server logs for specific errors

## 📞 **Support**

If you encounter any issues:
1. Check the CHANGELOG.md for detailed information
2. Review server logs for error messages
3. Verify all file paths are correct
4. Ensure all dependencies are installed

## ✅ **Success Indicators**

You'll know the update is successful when:
- ✅ PDF export works in under 10 seconds
- ✅ PDF contains professional medical report formatting
- ✅ PosturEase logo appears in browser tabs
- ✅ All pages load without errors
- ✅ Export button shows loading indicator
- ✅ No error messages in server logs

---

**Good luck with the update!** 🚀
