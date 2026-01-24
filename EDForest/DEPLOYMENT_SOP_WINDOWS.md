# 📦 VSMC Litho Platform - Windows Deployment SOP

## Standard Operating Procedure for Production Deployment

**Document Version:** 1.0  
**Last Updated:** January 24, 2026  
**Target OS:** Windows 10/11, Windows Server 2019/2022  
**Deployment Type:** Production Environment

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Deployment Steps](#deployment-steps)
4. [Post-Deployment Verification](#post-deployment-verification)
5. [Configuration](#configuration)
6. [Security Hardening](#security-hardening)
7. [Backup and Recovery](#backup-and-recovery)
8. [Troubleshooting](#troubleshooting)
9. [Rollback Procedure](#rollback-procedure)

---

## Prerequisites

### System Requirements

#### Minimum Requirements
- **OS:** Windows 10 (64-bit) or Windows Server 2019
- **CPU:** 4 cores, 2.5 GHz
- **RAM:** 8 GB
- **Storage:** 20 GB free space
- **Network:** Static IP or DHCP reservation

#### Recommended Requirements
- **OS:** Windows 11 or Windows Server 2022
- **CPU:** 8 cores, 3.0 GHz
- **RAM:** 16 GB
- **Storage:** 50 GB SSD
- **Network:** Gigabit Ethernet

### Software Requirements

#### Required Software
1. **Python 3.9 or higher**
   - Download: https://www.python.org/downloads/
   - Ensure "Add Python to PATH" is checked during installation

2. **Node.js 18.x or higher (LTS)**
   - Download: https://nodejs.org/
   - Includes npm package manager

3. **Git for Windows** (optional, for version control)
   - Download: https://git-scm.com/download/win

#### Verification Commands
```cmd
python --version
node --version
npm --version
```

### Network Requirements
- **Firewall Rules:**
  - Inbound: Port 5000 (Backend API)
  - Inbound: Port 80/443 (Frontend - if using web server)
  - Outbound: Internet access for package installation

---

## Pre-Deployment Checklist

### ☐ 1. Environment Preparation
- [ ] Windows updates installed
- [ ] Antivirus exclusions configured
- [ ] Firewall rules configured
- [ ] User accounts created (service account recommended)
- [ ] Backup of existing system (if upgrading)

### ☐ 2. Software Installation
- [ ] Python installed and in PATH
- [ ] Node.js and npm installed
- [ ] Git installed (optional)
- [ ] Text editor installed (VS Code, Notepad++, etc.)

### ☐ 3. Network Configuration
- [ ] Static IP assigned or DHCP reservation
- [ ] DNS configured
- [ ] Hostname set
- [ ] Network connectivity verified

### ☐ 4. Security Preparation
- [ ] SSL certificates obtained (if using HTTPS)
- [ ] Service account created
- [ ] Password policy reviewed
- [ ] Access control list prepared

### ☐ 5. Documentation
- [ ] Deployment plan reviewed
- [ ] Change management ticket created
- [ ] Stakeholders notified
- [ ] Rollback plan prepared

---

## Deployment Steps

### Step 1: Create Deployment Directory

```cmd
REM Create main deployment directory
mkdir C:\VSMC\LithoPlatform
cd C:\VSMC\LithoPlatform

REM Create subdirectories
mkdir logs
mkdir backups
mkdir data
```

### Step 2: Copy Application Files

**Option A: From Development Machine**
```cmd
REM Copy entire project folder
xcopy /E /I /H /Y "\\source\path\EDForest" "C:\VSMC\LithoPlatform"
```

**Option B: From Git Repository**
```cmd
cd C:\VSMC\LithoPlatform
git clone https://your-repo-url.git .
```

**Option C: From ZIP Archive**
```cmd
REM Extract ZIP to C:\VSMC\LithoPlatform
REM Use Windows Explorer or PowerShell:
Expand-Archive -Path "EDForest.zip" -DestinationPath "C:\VSMC\LithoPlatform"
```

### Step 3: Backend Deployment

#### 3.1 Install Python Dependencies
```cmd
cd C:\VSMC\LithoPlatform\backend

REM Install dependencies
pip install -r requirements.txt

REM Verify installation
pip list
```

#### 3.2 Install Authentication Dependencies
```cmd
pip install bcrypt==4.1.2 PyJWT==2.8.0
```

#### 3.3 Configure Backend
```cmd
REM Create/edit config file
notepad config.py
```

**Update config.py for production:**
```python
class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    # Set secure secret key
    SECRET_KEY = 'your-secure-random-secret-key-here'
    # Configure CORS for production domain
    CORS_ORIGINS = ['http://your-domain.com', 'https://your-domain.com']
```

#### 3.4 Initialize User Database
```cmd
REM First run will create users.json with default admin
python run.py
REM Press Ctrl+C after seeing "Running on..."
```

#### 3.5 Change Default Admin Password
```cmd
REM Edit users.json or use API to change password
REM Recommended: Change via API after first login
```

### Step 4: Frontend Deployment

#### 4.1 Install Node Dependencies
```cmd
cd C:\VSMC\LithoPlatform\frontend

REM Install dependencies
npm install

REM Verify installation
npm list --depth=0
```

#### 4.2 Configure Frontend for Production
```cmd
notepad .env.production
```

**Create .env.production:**
```env
VITE_API_URL=http://your-server-ip:5000
VITE_APP_TITLE=VSMC Litho Platform
```

#### 4.3 Build Frontend for Production
```cmd
npm run build

REM Build output will be in 'dist' folder
```

### Step 5: Configure Windows Services (Production)

#### 5.1 Create Backend Service

**Create backend-service.bat:**
```batch
@echo off
cd C:\VSMC\LithoPlatform\backend
python run.py
```

**Install as Windows Service using NSSM:**
```cmd
REM Download NSSM from https://nssm.cc/download
nssm install VSMCLithoBackend "C:\VSMC\LithoPlatform\backend\backend-service.bat"
nssm set VSMCLithoBackend AppDirectory "C:\VSMC\LithoPlatform\backend"
nssm set VSMCLithoBackend DisplayName "VSMC Litho Platform - Backend API"
nssm set VSMCLithoBackend Description "Backend API service for VSMC Litho Platform"
nssm set VSMCLithoBackend Start SERVICE_AUTO_START
nssm start VSMCLithoBackend
```

#### 5.2 Configure IIS for Frontend (Recommended)

**Install IIS:**
```powershell
# Run as Administrator
Enable-WindowsOptionalFeature -Online -FeatureName IIS-WebServerRole
Enable-WindowsOptionalFeature -Online -FeatureName IIS-WebServer
Enable-WindowsOptionalFeature -Online -FeatureName IIS-CommonHttpFeatures
Enable-WindowsOptionalFeature -Online -FeatureName IIS-HttpErrors
Enable-WindowsOptionalFeature -Online -FeatureName IIS-ApplicationDevelopment
Enable-WindowsOptionalFeature -Online -FeatureName IIS-StaticContent
Enable-WindowsOptionalFeature -Online -FeatureName IIS-DefaultDocument
```

**Configure IIS Site:**
1. Open IIS Manager
2. Right-click "Sites" → "Add Website"
3. Site name: `VSMC Litho Platform`
4. Physical path: `C:\VSMC\LithoPlatform\frontend\dist`
5. Binding: HTTP, Port 80 (or 443 for HTTPS)
6. Click OK

**Configure URL Rewrite (for React Router):**
Create `web.config` in `frontend/dist`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <system.webServer>
    <rewrite>
      <rules>
        <rule name="React Routes" stopProcessing="true">
          <match url=".*" />
          <conditions logicalGrouping="MatchAll">
            <add input="{REQUEST_FILENAME}" matchType="IsFile" negate="true" />
            <add input="{REQUEST_FILENAME}" matchType="IsDirectory" negate="true" />
          </conditions>
          <action type="Rewrite" url="/" />
        </rule>
      </rules>
    </rewrite>
  </system.webServer>
</configuration>
```

### Step 6: Configure Firewall

```cmd
REM Allow Backend API
netsh advfirewall firewall add rule name="VSMC Litho Backend" dir=in action=allow protocol=TCP localport=5000

REM Allow HTTP (if using IIS)
netsh advfirewall firewall add rule name="VSMC Litho Frontend HTTP" dir=in action=allow protocol=TCP localport=80

REM Allow HTTPS (if using SSL)
netsh advfirewall firewall add rule name="VSMC Litho Frontend HTTPS" dir=in action=allow protocol=TCP localport=443
```

### Step 7: Create Startup Scripts

**Create START_PRODUCTION.bat:**
```batch
@echo off
echo ======================================================================
echo VSMC Litho Platform - Production Startup
echo ======================================================================
echo.

REM Start Backend Service
net start VSMCLithoBackend
if errorlevel 1 (
    echo ERROR: Failed to start backend service
    pause
    exit /b 1
)

echo Backend service started successfully!
echo.

REM Start IIS (if not already running)
iisreset /start

echo.
echo ======================================================================
echo VSMC Litho Platform is now running!
echo ======================================================================
echo.
echo Backend API:  http://localhost:5000
echo Frontend:     http://localhost
echo.
echo Login: admin / [your-password]
echo.
pause
```

---

## Post-Deployment Verification

### Verification Checklist

#### ☐ 1. Backend API Verification
```cmd
REM Test health endpoint
curl http://localhost:5000/api/v1/health

REM Expected response: {"status": "healthy"}
```

#### ☐ 2. Frontend Verification
- [ ] Open browser: `http://localhost` or `http://your-server-ip`
- [ ] Verify login page loads
- [ ] Test login with admin credentials
- [ ] Verify all pages load correctly
- [ ] Test theme toggle (dark/light mode)

#### ☐ 3. Authentication Verification
- [ ] Login with admin account
- [ ] Create new user account
- [ ] Test logout functionality
- [ ] Verify token persistence
- [ ] Test protected routes

#### ☐ 4. Functionality Verification
- [ ] Upload test data to EDForest
- [ ] Run analysis
- [ ] View results and charts
- [ ] Save analysis to history
- [ ] Export data
- [ ] View history page

#### ☐ 5. Performance Verification
- [ ] Check CPU usage (should be < 50% idle)
- [ ] Check memory usage (should be < 4GB)
- [ ] Test response times (< 2 seconds)
- [ ] Verify no memory leaks

#### ☐ 6. Security Verification
- [ ] Verify HTTPS enabled (if configured)
- [ ] Test unauthorized access (should redirect to login)
- [ ] Verify password hashing
- [ ] Check file permissions
- [ ] Review firewall rules

---

## Configuration

### Environment Variables

**Backend (.env file):**
```env
# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-secure-random-secret-key-change-this

# JWT Configuration
JWT_SECRET_KEY=your-jwt-secret-key-change-this
JWT_EXPIRATION_HOURS=24

# CORS Configuration
CORS_ORIGINS=http://your-domain.com,https://your-domain.com

# Database
USERS_DB_PATH=users.json

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/backend.log
```

**Frontend (.env.production):**
```env
VITE_API_URL=http://your-server-ip:5000
VITE_APP_TITLE=VSMC Litho Platform
VITE_APP_VERSION=1.0.0
```

### Application Configuration

**Update backend/config.py:**
```python
import os

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.getenv('SECRET_KEY', 'change-this-in-production')
    
    # CORS - Update with your domain
    CORS_ORIGINS = [
        'http://your-domain.com',
        'https://your-domain.com',
        'http://your-server-ip'
    ]
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/backend.log'
```

---

## Security Hardening

### 1. Change Default Credentials
```cmd
REM After first login, change admin password via UI or API
curl -X POST http://localhost:5000/api/v1/auth/change-password ^
  -H "Authorization: Bearer YOUR_TOKEN" ^
  -H "Content-Type: application/json" ^
  -d "{\"current_password\":\"admin123\",\"new_password\":\"NewSecurePassword123!\"}"
```

### 2. Configure HTTPS (Recommended)

**Option A: Using IIS with SSL Certificate**
1. Obtain SSL certificate (Let's Encrypt, commercial CA, or self-signed)
2. Import certificate to Windows Certificate Store
3. In IIS Manager, add HTTPS binding to site
4. Select imported certificate
5. Update CORS origins in backend config

**Option B: Using Reverse Proxy (nginx)**
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        root C:/VSMC/LithoPlatform/frontend/dist;
        try_files $uri $uri/ /index.html;
    }
    
    location /api {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. File Permissions
```cmd
REM Restrict access to sensitive files
icacls "C:\VSMC\LithoPlatform\backend\users.json" /grant Administrators:F /inheritance:r
icacls "C:\VSMC\LithoPlatform\backend\config.py" /grant Administrators:F /inheritance:r
```

### 4. Disable Debug Mode
Ensure in `backend/config.py`:
```python
DEBUG = False
TESTING = False
```

### 5. Configure Windows Defender Exclusions
```powershell
# Run as Administrator
Add-MpPreference -ExclusionPath "C:\VSMC\LithoPlatform"
```

### 6. Enable Audit Logging
```cmd
REM Enable file access auditing
auditpol /set /subcategory:"File System" /success:enable /failure:enable
```

---

## Backup and Recovery

### Backup Strategy

#### Daily Backups
```batch
@echo off
REM backup-daily.bat

set BACKUP_DIR=C:\VSMC\Backups\Daily
set DATE=%date:~-4,4%%date:~-10,2%%date:~-7,2%
set BACKUP_PATH=%BACKUP_DIR%\backup_%DATE%

REM Create backup directory
mkdir "%BACKUP_PATH%"

REM Backup user database
copy "C:\VSMC\LithoPlatform\backend\users.json" "%BACKUP_PATH%\"

REM Backup history data
xcopy /E /I /Y "C:\VSMC\LithoPlatform\backend\history" "%BACKUP_PATH%\history"

REM Backup uploads
xcopy /E /I /Y "C:\VSMC\LithoPlatform\backend\uploads" "%BACKUP_PATH%\uploads"

REM Backup configuration
copy "C:\VSMC\LithoPlatform\backend\config.py" "%BACKUP_PATH%\"

REM Delete backups older than 7 days
forfiles /p "%BACKUP_DIR%" /m backup_* /d -7 /c "cmd /c rmdir /s /q @path"

echo Backup completed: %BACKUP_PATH%
```

**Schedule with Task Scheduler:**
```cmd
schtasks /create /tn "VSMC Litho Daily Backup" /tr "C:\VSMC\LithoPlatform\backup-daily.bat" /sc daily /st 02:00 /ru SYSTEM
```

#### Weekly Full Backups
```batch
@echo off
REM backup-weekly.bat

set BACKUP_DIR=C:\VSMC\Backups\Weekly
set DATE=%date:~-4,4%%date:~-10,2%%date:~-7,2%
set BACKUP_FILE=%BACKUP_DIR%\vsmc_litho_backup_%DATE%.zip

REM Create full backup
powershell Compress-Archive -Path "C:\VSMC\LithoPlatform\*" -DestinationPath "%BACKUP_FILE%" -Force

REM Delete backups older than 30 days
forfiles /p "%BACKUP_DIR%" /m *.zip /d -30 /c "cmd /c del @path"

echo Full backup completed: %BACKUP_FILE%
```

### Recovery Procedure

#### Restore from Backup
```batch
@echo off
REM restore-backup.bat

set BACKUP_PATH=%1

if "%BACKUP_PATH%"=="" (
    echo Usage: restore-backup.bat [backup_path]
    exit /b 1
)

echo ======================================================================
echo VSMC Litho Platform - Restore from Backup
echo ======================================================================
echo.
echo WARNING: This will overwrite current data!
echo Backup path: %BACKUP_PATH%
echo.
pause

REM Stop services
net stop VSMCLithoBackend
iisreset /stop

REM Restore files
copy "%BACKUP_PATH%\users.json" "C:\VSMC\LithoPlatform\backend\"
xcopy /E /I /Y "%BACKUP_PATH%\history" "C:\VSMC\LithoPlatform\backend\history"
xcopy /E /I /Y "%BACKUP_PATH%\uploads" "C:\VSMC\LithoPlatform\backend\uploads"

REM Start services
net start VSMCLithoBackend
iisreset /start

echo.
echo Restore completed successfully!
pause
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Backend Service Won't Start
**Symptoms:** Service fails to start, error in Event Viewer

**Solutions:**
```cmd
REM Check Python installation
python --version

REM Check dependencies
pip list | findstr Flask

REM Check port availability
netstat -ano | findstr :5000

REM Check logs
type C:\VSMC\LithoPlatform\backend\logs\backend.log

REM Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Issue 2: Frontend Not Loading
**Symptoms:** Blank page, 404 errors

**Solutions:**
```cmd
REM Check IIS status
iisreset /status

REM Verify build files exist
dir C:\VSMC\LithoPlatform\frontend\dist

REM Rebuild frontend
cd C:\VSMC\LithoPlatform\frontend
npm run build

REM Check IIS logs
type C:\inetpub\logs\LogFiles\W3SVC1\*.log
```

#### Issue 3: Authentication Errors
**Symptoms:** Login fails, token errors

**Solutions:**
```cmd
REM Check users.json exists
dir C:\VSMC\LithoPlatform\backend\users.json

REM Reset admin password (delete users.json, will recreate with default)
del C:\VSMC\LithoPlatform\backend\users.json
net restart VSMCLithoBackend

REM Check JWT secret key is set
type C:\VSMC\LithoPlatform\backend\config.py | findstr SECRET_KEY
```

#### Issue 4: High CPU/Memory Usage
**Symptoms:** Slow performance, system lag

**Solutions:**
```cmd
REM Check process usage
tasklist /fi "imagename eq python.exe" /v

REM Restart services
net restart VSMCLithoBackend
iisreset /restart

REM Check for memory leaks in logs
type C:\VSMC\LithoPlatform\backend\logs\backend.log | findstr ERROR
```

#### Issue 5: CORS Errors
**Symptoms:** API calls fail from frontend

**Solutions:**
```python
# Update backend/config.py
CORS_ORIGINS = [
    'http://localhost',
    'http://your-server-ip',
    'http://your-domain.com',
    'https://your-domain.com'
]
```

---

## Rollback Procedure

### Emergency Rollback Steps

#### 1. Stop Current Services
```cmd
net stop VSMCLithoBackend
iisreset /stop
```

#### 2. Restore Previous Version
```cmd
REM Restore from backup
cd C:\VSMC\Backups\Weekly
REM Find latest backup
dir /o-d *.zip

REM Extract backup
powershell Expand-Archive -Path "vsmc_litho_backup_YYYYMMDD.zip" -DestinationPath "C:\VSMC\LithoPlatform_Rollback" -Force

REM Replace current with rollback
move "C:\VSMC\LithoPlatform" "C:\VSMC\LithoPlatform_Failed"
move "C:\VSMC\LithoPlatform_Rollback" "C:\VSMC\LithoPlatform"
```

#### 3. Restart Services
```cmd
net start VSMCLithoBackend
iisreset /start
```

#### 4. Verify Rollback
```cmd
REM Test backend
curl http://localhost:5000/api/v1/health

REM Test frontend
start http://localhost
```

---

## Maintenance

### Regular Maintenance Tasks

#### Daily
- [ ] Check service status
- [ ] Review error logs
- [ ] Monitor disk space
- [ ] Verify backups completed

#### Weekly
- [ ] Review user activity
- [ ] Check for updates
- [ ] Clean old uploads/temp files
- [ ] Test backup restoration

#### Monthly
- [ ] Update dependencies
- [ ] Review security logs
- [ ] Performance optimization
- [ ] Update documentation

### Monitoring Script
```batch
@echo off
REM monitor-health.bat

echo ======================================================================
echo VSMC Litho Platform - Health Check
echo ======================================================================
echo.

REM Check backend service
sc query VSMCLithoBackend | findstr STATE
echo.

REM Check IIS
iisreset /status
echo.

REM Check disk space
wmic logicaldisk get caption,freespace,size
echo.

REM Check recent errors
echo Recent Backend Errors:
type C:\VSMC\LithoPlatform\backend\logs\backend.log | findstr /i "error" | more +10
echo.

pause
```

---

## Support and Contacts

### Technical Support
- **Internal IT:** it-support@company.com
- **Application Owner:** vsmc-admin@company.com
- **Emergency Contact:** +1-XXX-XXX-XXXX

### Documentation
- **User Manual:** `C:\VSMC\LithoPlatform\docs\USER_MANUAL.md`
- **API Documentation:** `http://localhost:5000/api/v1/docs`
- **Change Log:** `C:\VSMC\LithoPlatform\CHANGELOG.md`

---

## Appendix

### A. Required Ports
| Port | Service | Protocol | Direction |
|------|---------|----------|-----------|
| 5000 | Backend API | TCP | Inbound |
| 80 | Frontend HTTP | TCP | Inbound |
| 443 | Frontend HTTPS | TCP | Inbound |

### B. File Locations
| Component | Path |
|-----------|------|
| Application Root | `C:\VSMC\LithoPlatform` |
| Backend | `C:\VSMC\LithoPlatform\backend` |
| Frontend | `C:\VSMC\LithoPlatform\frontend` |
| User Database | `C:\VSMC\LithoPlatform\backend\users.json` |
| History Data | `C:\VSMC\LithoPlatform\backend\history` |
| Uploads | `C:\VSMC\LithoPlatform\backend\uploads` |
| Logs | `C:\VSMC\LithoPlatform\backend\logs` |
| Backups | `C:\VSMC\Backups` |

### C. Service Accounts
- **Backend Service:** Local System or dedicated service account
- **IIS Application Pool:** ApplicationPoolIdentity
- **File Permissions:** Administrators group

### D. Useful Commands
```cmd
REM Check service status
sc query VSMCLithoBackend

REM View service logs
eventvwr.msc

REM Check port usage
netstat -ano | findstr :5000

REM Test API
curl http://localhost:5000/api/v1/health

REM Restart services
net restart VSMCLithoBackend
iisreset /restart
```

---

**End of Deployment SOP**

**Document Control:**
- **Version:** 1.0
- **Date:** January 24, 2026
- **Author:** VSMC IT Team
- **Approved By:** [Name]
- **Next Review:** [Date]
