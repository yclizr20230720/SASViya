# ✅ VSMC Litho Platform - Deployment Checklist

## Quick Reference Checklist for Windows Deployment

---

## Pre-Deployment (1-2 hours before)

### System Preparation
- [ ] Windows updates installed and system rebooted
- [ ] Python 3.9+ installed and verified (`python --version`)
- [ ] Node.js 18+ installed and verified (`node --version`)
- [ ] Git installed (optional)
- [ ] Antivirus exclusions configured for `C:\VSMC\LithoPlatform`
- [ ] Firewall rules configured (ports 5000, 80, 443)
- [ ] Static IP or DHCP reservation configured
- [ ] DNS records updated (if applicable)

### Backup Current System
- [ ] Backup existing application (if upgrading)
- [ ] Backup user database
- [ ] Backup configuration files
- [ ] Document current version number
- [ ] Create system restore point

### Access and Permissions
- [ ] Administrator access confirmed
- [ ] Service account created (if using)
- [ ] File permissions reviewed
- [ ] Network access verified

---

## Deployment (30-60 minutes)

### Step 1: Copy Files
- [ ] Create directory: `C:\VSMC\LithoPlatform`
- [ ] Copy application files to deployment directory
- [ ] Verify all files copied successfully
- [ ] Check file integrity (compare file counts)

### Step 2: Backend Setup
- [ ] Navigate to `backend` folder
- [ ] Install Python dependencies: `pip install -r requirements.txt`
- [ ] Install auth dependencies: `pip install bcrypt==4.1.2 PyJWT==2.8.0`
- [ ] Verify all packages installed: `pip list`
- [ ] Update `config.py` for production
- [ ] Set SECRET_KEY and JWT_SECRET_KEY
- [ ] Configure CORS origins
- [ ] Test backend startup: `python run.py`
- [ ] Verify API responds: `curl http://localhost:5000/api/v1/health`
- [ ] Stop test instance (Ctrl+C)

### Step 3: Frontend Setup
- [ ] Navigate to `frontend` folder
- [ ] Install Node dependencies: `npm install`
- [ ] Create `.env.production` file
- [ ] Set VITE_API_URL to backend address
- [ ] Build production bundle: `npm run build`
- [ ] Verify `dist` folder created
- [ ] Check build output for errors

### Step 4: Service Configuration
- [ ] Install NSSM (if using Windows Service)
- [ ] Create backend Windows Service
- [ ] Configure service to auto-start
- [ ] Start backend service: `net start VSMCLithoBackend`
- [ ] Verify service running: `sc query VSMCLithoBackend`

### Step 5: Web Server Setup
- [ ] Install IIS (if not installed)
- [ ] Create new IIS website
- [ ] Point to `frontend/dist` folder
- [ ] Configure bindings (HTTP/HTTPS)
- [ ] Add `web.config` for URL rewriting
- [ ] Start IIS site
- [ ] Verify site accessible

### Step 6: Firewall Configuration
- [ ] Add rule for port 5000 (Backend)
- [ ] Add rule for port 80 (HTTP)
- [ ] Add rule for port 443 (HTTPS, if applicable)
- [ ] Test external access
- [ ] Verify firewall logs

---

## Post-Deployment Verification (15-30 minutes)

### Functional Testing
- [ ] Open browser to application URL
- [ ] Verify login page loads
- [ ] Login with admin credentials (admin/admin123)
- [ ] Change admin password immediately
- [ ] Create test user account
- [ ] Test logout and re-login
- [ ] Navigate to all pages (Home, EDForest, History, About)
- [ ] Upload test data file
- [ ] Run analysis
- [ ] View results and charts
- [ ] Save analysis to history
- [ ] Export data
- [ ] View history page
- [ ] Test theme toggle (dark/light mode)
- [ ] Test on different browsers (Chrome, Edge, Firefox)

### Performance Testing
- [ ] Check CPU usage (Task Manager)
- [ ] Check memory usage
- [ ] Test response times (< 2 seconds)
- [ ] Monitor for 10 minutes
- [ ] Check for memory leaks
- [ ] Review error logs

### Security Verification
- [ ] Verify HTTPS enabled (if configured)
- [ ] Test unauthorized access (should redirect to login)
- [ ] Verify password is hashed in users.json
- [ ] Check file permissions on sensitive files
- [ ] Review Windows Event Viewer for errors
- [ ] Test from external network (if applicable)

### Backup Verification
- [ ] Run backup script
- [ ] Verify backup files created
- [ ] Test restore procedure (on test system if possible)
- [ ] Schedule automatic backups
- [ ] Document backup location

---

## Configuration (15 minutes)

### Security Hardening
- [ ] Change default admin password
- [ ] Update SECRET_KEY in config.py
- [ ] Update JWT_SECRET_KEY
- [ ] Configure HTTPS (if not done)
- [ ] Set file permissions
- [ ] Enable audit logging
- [ ] Configure Windows Defender exclusions
- [ ] Review and update CORS origins

### Monitoring Setup
- [ ] Configure Windows Event Logging
- [ ] Set up log rotation
- [ ] Create monitoring script
- [ ] Schedule health checks
- [ ] Configure email alerts (if applicable)

### Documentation
- [ ] Update deployment documentation
- [ ] Document server IP/hostname
- [ ] Document admin credentials (secure location)
- [ ] Create user guide
- [ ] Update network diagram
- [ ] Document any custom configurations

---

## Handover (15 minutes)

### Knowledge Transfer
- [ ] Demonstrate application to users
- [ ] Provide login credentials
- [ ] Show how to access application
- [ ] Explain basic functionality
- [ ] Provide user documentation
- [ ] Share support contact information

### Operations Handover
- [ ] Provide admin credentials to IT team
- [ ] Share deployment documentation
- [ ] Explain backup/restore procedures
- [ ] Show how to restart services
- [ ] Demonstrate monitoring tools
- [ ] Provide troubleshooting guide

---

## Post-Deployment Tasks (Within 24 hours)

### Monitoring
- [ ] Monitor application for 24 hours
- [ ] Check error logs regularly
- [ ] Monitor system resources
- [ ] Verify backups running
- [ ] Check user feedback

### Communication
- [ ] Send deployment completion email
- [ ] Update change management ticket
- [ ] Notify stakeholders
- [ ] Schedule follow-up meeting
- [ ] Document lessons learned

### Cleanup
- [ ] Remove temporary files
- [ ] Clean up old backups
- [ ] Archive deployment files
- [ ] Update asset inventory
- [ ] Close deployment ticket

---

## Rollback Criteria

### When to Rollback
- [ ] Critical functionality not working
- [ ] Security vulnerability discovered
- [ ] Performance degradation > 50%
- [ ] Data corruption detected
- [ ] Multiple user complaints
- [ ] Service unavailable > 15 minutes

### Rollback Procedure
1. [ ] Stop current services
2. [ ] Restore from backup
3. [ ] Restart services
4. [ ] Verify functionality
5. [ ] Notify stakeholders
6. [ ] Document issues

---

## Success Criteria

### Deployment Successful If:
- ✅ Application accessible from network
- ✅ All users can login
- ✅ All features working correctly
- ✅ No critical errors in logs
- ✅ Performance meets requirements
- ✅ Backups configured and working
- ✅ Security measures in place
- ✅ Documentation complete

---

## Quick Reference

### Important URLs
- **Frontend:** http://[server-ip] or http://[domain]
- **Backend API:** http://[server-ip]:5000
- **API Health:** http://[server-ip]:5000/api/v1/health

### Default Credentials
- **Username:** admin
- **Password:** admin123 (CHANGE IMMEDIATELY)

### Important Paths
- **Application:** `C:\VSMC\LithoPlatform`
- **Backend:** `C:\VSMC\LithoPlatform\backend`
- **Frontend:** `C:\VSMC\LithoPlatform\frontend`
- **Users DB:** `C:\VSMC\LithoPlatform\backend\users.json`
- **Logs:** `C:\VSMC\LithoPlatform\backend\logs`
- **Backups:** `C:\VSMC\Backups`

### Service Commands
```cmd
REM Start backend service
net start VSMCLithoBackend

REM Stop backend service
net stop VSMCLithoBackend

REM Restart IIS
iisreset /restart

REM Check service status
sc query VSMCLithoBackend
```

### Emergency Contacts
- **IT Support:** [phone/email]
- **Application Owner:** [phone/email]
- **On-Call Engineer:** [phone/email]

---

## Sign-Off

### Deployment Team
- **Deployed By:** _________________ Date: _______
- **Verified By:** _________________ Date: _______
- **Approved By:** _________________ Date: _______

### Notes
```
[Add any deployment-specific notes here]
```

---

**Checklist Version:** 1.0  
**Last Updated:** January 24, 2026  
**Next Review:** [Date]
