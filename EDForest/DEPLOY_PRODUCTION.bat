@echo off
setlocal enabledelayedexpansion

REM ======================================================================
REM VSMC Litho Platform - Automated Production Deployment Script
REM ======================================================================
REM Version: 1.0
REM Date: January 24, 2026
REM Description: Automated deployment script for Windows production environment
REM ======================================================================

color 0A
title VSMC Litho Platform - Production Deployment

echo.
echo ======================================================================
echo          VSMC Litho Platform - Production Deployment
echo ======================================================================
echo.
echo This script will deploy the VSMC Litho Platform to production.
echo.
echo WARNING: This will:
echo   - Install dependencies
echo   - Configure services
echo   - Update firewall rules
echo   - Modify system settings
echo.
echo Press Ctrl+C to cancel or
pause

REM ======================================================================
REM Configuration Variables
REM ======================================================================

set DEPLOY_DIR=C:\VSMC\LithoPlatform
set BACKUP_DIR=C:\VSMC\Backups
set LOG_FILE=%DEPLOY_DIR%\deployment.log
set ERROR_COUNT=0

REM ======================================================================
REM Functions
REM ======================================================================

:log
echo [%date% %time%] %~1 >> "%LOG_FILE%"
echo %~1
goto :eof

:error
set /a ERROR_COUNT+=1
echo [%date% %time%] ERROR: %~1 >> "%LOG_FILE%"
echo ERROR: %~1
goto :eof

REM ======================================================================
REM Pre-Deployment Checks
REM ======================================================================

call :log "======================================================================="
call :log "Starting Pre-Deployment Checks"
call :log "======================================================================="

REM Check if running as Administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    call :error "This script must be run as Administrator!"
    echo.
    echo Please right-click and select "Run as Administrator"
    pause
    exit /b 1
)
call :log "Administrator privileges confirmed"

REM Check Python installation
call :log "Checking Python installation..."
python --version >nul 2>&1
if %errorLevel% neq 0 (
    call :error "Python is not installed or not in PATH!"
    echo.
    echo Please install Python 3.9 or higher from https://www.python.org/
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
call :log "Python found: %PYTHON_VERSION%"

REM Check Node.js installation
call :log "Checking Node.js installation..."
node --version >nul 2>&1
if %errorLevel% neq 0 (
    call :error "Node.js is not installed or not in PATH!"
    echo.
    echo Please install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
call :log "Node.js found: %NODE_VERSION%"

REM Check if deployment directory exists
if not exist "%DEPLOY_DIR%" (
    call :error "Deployment directory not found: %DEPLOY_DIR%"
    echo.
    echo Please ensure application files are copied to %DEPLOY_DIR%
    pause
    exit /b 1
)
call :log "Deployment directory found: %DEPLOY_DIR%"

call :log "Pre-deployment checks completed successfully!"
echo.

REM ======================================================================
REM Backup Existing Installation
REM ======================================================================

call :log "======================================================================="
call :log "Creating Backup"
call :log "======================================================================="

if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

set BACKUP_TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set BACKUP_TIMESTAMP=%BACKUP_TIMESTAMP: =0%
set BACKUP_PATH=%BACKUP_DIR%\backup_%BACKUP_TIMESTAMP%

call :log "Creating backup at: %BACKUP_PATH%"

if exist "%DEPLOY_DIR%\backend\users.json" (
    mkdir "%BACKUP_PATH%" 2>nul
    copy "%DEPLOY_DIR%\backend\users.json" "%BACKUP_PATH%\" >nul 2>&1
    call :log "Backed up users.json"
)

if exist "%DEPLOY_DIR%\backend\history" (
    xcopy /E /I /Q /Y "%DEPLOY_DIR%\backend\history" "%BACKUP_PATH%\history" >nul 2>&1
    call :log "Backed up history data"
)

call :log "Backup completed successfully!"
echo.

REM ======================================================================
REM Install Backend Dependencies
REM ======================================================================

call :log "======================================================================="
call :log "Installing Backend Dependencies"
call :log "======================================================================="

cd /d "%DEPLOY_DIR%\backend"

call :log "Installing Python packages..."
pip install -r requirements.txt --quiet
if %errorLevel% neq 0 (
    call :error "Failed to install Python dependencies"
    goto :deployment_failed
)
call :log "Python packages installed successfully"

call :log "Installing authentication packages..."
pip install bcrypt==4.1.2 PyJWT==2.8.0 --quiet
if %errorLevel% neq 0 (
    call :error "Failed to install authentication packages"
    goto :deployment_failed
)
call :log "Authentication packages installed successfully"

echo.

REM ======================================================================
REM Install Frontend Dependencies
REM ======================================================================

call :log "======================================================================="
call :log "Installing Frontend Dependencies"
call :log "======================================================================="

cd /d "%DEPLOY_DIR%\frontend"

call :log "Installing Node packages..."
call npm install --silent
if %errorLevel% neq 0 (
    call :error "Failed to install Node packages"
    goto :deployment_failed
)
call :log "Node packages installed successfully"

echo.

REM ======================================================================
REM Build Frontend
REM ======================================================================

call :log "======================================================================="
call :log "Building Frontend for Production"
call :log "======================================================================="

call :log "Running production build..."
call npm run build
if %errorLevel% neq 0 (
    call :error "Failed to build frontend"
    goto :deployment_failed
)

if not exist "dist\index.html" (
    call :error "Build output not found - build may have failed"
    goto :deployment_failed
)

call :log "Frontend built successfully"
echo.

REM ======================================================================
REM Configure Firewall
REM ======================================================================

call :log "======================================================================="
call :log "Configuring Windows Firewall"
call :log "======================================================================="

call :log "Adding firewall rule for Backend API (port 5000)..."
netsh advfirewall firewall delete rule name="VSMC Litho Backend" >nul 2>&1
netsh advfirewall firewall add rule name="VSMC Litho Backend" dir=in action=allow protocol=TCP localport=5000 >nul 2>&1
if %errorLevel% equ 0 (
    call :log "Backend firewall rule added"
) else (
    call :error "Failed to add backend firewall rule"
)

call :log "Adding firewall rule for Frontend HTTP (port 80)..."
netsh advfirewall firewall delete rule name="VSMC Litho Frontend HTTP" >nul 2>&1
netsh advfirewall firewall add rule name="VSMC Litho Frontend HTTP" dir=in action=allow protocol=TCP localport=80 >nul 2>&1
if %errorLevel% equ 0 (
    call :log "Frontend HTTP firewall rule added"
) else (
    call :error "Failed to add frontend HTTP firewall rule"
)

call :log "Adding firewall rule for Frontend HTTPS (port 443)..."
netsh advfirewall firewall delete rule name="VSMC Litho Frontend HTTPS" >nul 2>&1
netsh advfirewall firewall add rule name="VSMC Litho Frontend HTTPS" dir=in action=allow protocol=TCP localport=443 >nul 2>&1
if %errorLevel% equ 0 (
    call :log "Frontend HTTPS firewall rule added"
) else (
    call :error "Failed to add frontend HTTPS firewall rule"
)

echo.

REM ======================================================================
REM Test Backend
REM ======================================================================

call :log "======================================================================="
call :log "Testing Backend API"
call :log "======================================================================="

cd /d "%DEPLOY_DIR%\backend"

call :log "Starting backend for testing..."
start /B python run.py >nul 2>&1

call :log "Waiting for backend to start..."
timeout /t 10 /nobreak >nul

call :log "Testing API health endpoint..."
curl -s http://localhost:5000/api/v1/health >nul 2>&1
if %errorLevel% equ 0 (
    call :log "Backend API is responding correctly"
) else (
    call :error "Backend API is not responding"
)

call :log "Stopping test backend..."
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *run.py*" >nul 2>&1

echo.

REM ======================================================================
REM Create Startup Scripts
REM ======================================================================

call :log "======================================================================="
call :log "Creating Startup Scripts"
call :log "======================================================================="

cd /d "%DEPLOY_DIR%"

REM Create production startup script
call :log "Creating START_PRODUCTION.bat..."
(
echo @echo off
echo title VSMC Litho Platform - Production
echo color 0B
echo.
echo ======================================================================
echo          VSMC Litho Platform - Starting Production Services
echo ======================================================================
echo.
echo Starting Backend API...
echo.
cd /d "%DEPLOY_DIR%\backend"
start "VSMC Litho - Backend API" cmd /k python run.py
echo.
echo Backend started on http://localhost:5000
echo.
echo ======================================================================
echo.
echo To access the application:
echo   1. Configure IIS to serve frontend/dist folder
echo   2. Open browser to http://localhost or your server IP
echo   3. Login with: admin / admin123
echo.
echo To stop: Close the Backend API window
echo.
pause
) > START_PRODUCTION.bat

call :log "START_PRODUCTION.bat created"

REM Create monitoring script
call :log "Creating MONITOR_HEALTH.bat..."
(
echo @echo off
echo title VSMC Litho Platform - Health Monitor
echo color 0E
echo.
echo ======================================================================
echo          VSMC Litho Platform - Health Monitor
echo ======================================================================
echo.
echo Checking Backend API...
curl -s http://localhost:5000/api/v1/health
echo.
echo.
echo Checking Disk Space...
wmic logicaldisk get caption,freespace,size
echo.
echo.
echo Recent Backend Logs:
type "%DEPLOY_DIR%\backend\logs\backend.log" 2^>nul ^| findstr /i "error" ^| more +10
echo.
echo ======================================================================
pause
) > MONITOR_HEALTH.bat

call :log "MONITOR_HEALTH.bat created"

echo.

REM ======================================================================
REM Deployment Summary
REM ======================================================================

call :log "======================================================================="
call :log "Deployment Summary"
call :log "======================================================================="

echo.
echo ======================================================================
echo                    DEPLOYMENT COMPLETED SUCCESSFULLY!
echo ======================================================================
echo.
echo Deployment Details:
echo   - Deployment Directory: %DEPLOY_DIR%
echo   - Backup Location: %BACKUP_PATH%
echo   - Log File: %LOG_FILE%
echo   - Errors Encountered: %ERROR_COUNT%
echo.
echo ======================================================================
echo                         NEXT STEPS
echo ======================================================================
echo.
echo 1. Configure IIS:
echo    - Create new website
echo    - Point to: %DEPLOY_DIR%\frontend\dist
echo    - Configure bindings (HTTP/HTTPS)
echo    - Add web.config for URL rewriting
echo.
echo 2. Start Backend Service:
echo    - Run: START_PRODUCTION.bat
echo    - Or install as Windows Service using NSSM
echo.
echo 3. Change Default Password:
echo    - Login with: admin / admin123
echo    - Change password immediately!
echo.
echo 4. Configure Production Settings:
echo    - Update backend\config.py
echo    - Set SECRET_KEY and JWT_SECRET_KEY
echo    - Configure CORS origins
echo.
echo 5. Test Application:
echo    - Open browser to http://localhost
echo    - Login and test all features
echo    - Monitor logs for errors
echo.
echo ======================================================================
echo.
echo For detailed instructions, see: DEPLOYMENT_SOP_WINDOWS.md
echo.

call :log "Deployment completed successfully at %date% %time%"

pause
exit /b 0

REM ======================================================================
REM Error Handler
REM ======================================================================

:deployment_failed
echo.
echo ======================================================================
echo                    DEPLOYMENT FAILED!
echo ======================================================================
echo.
echo Errors encountered: %ERROR_COUNT%
echo.
echo Check the log file for details: %LOG_FILE%
echo.
echo To rollback:
echo   1. Stop any running services
echo   2. Restore from backup: %BACKUP_PATH%
echo   3. Contact support if needed
echo.
echo ======================================================================
pause
exit /b 1
