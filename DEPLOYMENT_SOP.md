# SAS Viya Python Package Deployment SOP

## Document Information
- **Document Title**: SAS Viya Python Package Deployment Standard Operating Procedure
- **Version**: 1.0
- **Date**: 2025-01-09
- **Author**: System Administrator
- **Approved By**: [Approval Required]

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Pre-Deployment Checklist](#pre-deployment-checklist)
4. [Deployment Steps](#deployment-steps)
5. [Post-Deployment Verification](#post-deployment-verification)
6. [Configuration](#configuration)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)
9. [Rollback Procedures](#rollback-procedures)
10. [Maintenance](#maintenance)

## Overview

This SOP provides step-by-step instructions for deploying the SAS Viya Python Development Framework to a Virtual Machine (VM) environment. The package includes connection management, model implementations, deployment utilities, and security frameworks for SAS Viya integration.

### Package Components
- Connection management for SASPy, SWAT, and sasctl
- Model implementations (churn analysis, fraud detection, time series forecasting)
- Deployment and monitoring utilities
- Security and governance frameworks
- Example scripts and test suites

## Prerequisites

### System Requirements
- **Operating System**: Linux (RHEL 7+, Ubuntu 18.04+, CentOS 7+) or Windows Server 2016+
- **Python Version**: 3.7 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: Minimum 10GB free space
- **Network**: Access to SAS Viya server and internet for package downloads

### Software Dependencies
- Python 3.7+
- pip (Python package manager)
- Git (for version control)
- Virtual environment tools (venv or conda)

### Access Requirements
- **VM Access**: SSH access (Linux) or RDP access (Windows) with administrative privileges
- **SAS Viya Access**: Valid credentials and network connectivity to SAS Viya server
- **Network Ports**: Ensure ports 5570 (CAS), 443/80 (HTTPS/HTTP) are accessible
- **Firewall**: Configure firewall rules for SAS Viya communication

### Credentials Required
- VM administrator credentials
- SAS Viya username and password
- SAS Viya server URL and connection details

## Pre-Deployment Checklist

### ☐ Infrastructure Verification
- [ ] VM is provisioned and accessible
- [ ] Required ports are open (5570 for CAS, 443/80 for web services)
- [ ] Network connectivity to SAS Viya server confirmed
- [ ] Sufficient disk space available (minimum 10GB)
- [ ] Python 3.7+ is installed and accessible

### ☐ Access Verification
- [ ] Administrative access to target VM confirmed
- [ ] SAS Viya credentials validated
- [ ] SAS Viya server accessibility tested
- [ ] Required network permissions obtained

### ☐ Backup and Safety
- [ ] Current system state documented
- [ ] Backup of existing Python environment (if applicable)
- [ ] Rollback plan prepared
- [ ] Change management approval obtained

## Deployment Steps

### Step 1: Environment Preparation

#### 1.1 Connect to VM
```bash
# For Linux VM
ssh username@vm-hostname

# For Windows VM - use RDP client
# Connect to VM IP address with administrator credentials
```

#### 1.2 Update System (Linux)
```bash
# RHEL/CentOS
sudo yum update -y

# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y
```

#### 1.3 Install Python and Dependencies
```bash
# RHEL/CentOS
sudo yum install -y python3 python3-pip python3-venv git

# Ubuntu/Debian
sudo apt install -y python3 python3-pip python3-venv git

# Windows (using PowerShell as Administrator)
# Download and install Python from python.org
# Install Git from git-scm.com
```

#### 1.4 Verify Python Installation
```bash
python3 --version
pip3 --version
git --version
```

### Step 2: Create Deployment Directory

#### 2.1 Create Application Directory
```bash
# Linux
sudo mkdir -p /opt/sas-viya-python
sudo chown $USER:$USER /opt/sas-viya-python
cd /opt/sas-viya-python

# Windows
mkdir C:\opt\sas-viya-python
cd C:\opt\sas-viya-python
```

#### 2.2 Create Virtual Environment
```bash
# Linux/Windows
python3 -m venv venv

# Activate virtual environment
# Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3: Package Deployment

#### 3.1 Transfer Package Files
```bash
# Option 1: Using SCP (if package is on local machine)
scp -r sas_viya_project/ username@vm-hostname:/opt/sas-viya-python/

# Option 2: Using Git (if package is in repository)
git clone https://github.com/your-org/sas-viya-python.git .

# Option 3: Direct copy (if files are already on VM)
cp -r /path/to/sas_viya_project/* /opt/sas-viya-python/
```

#### 3.2 Set Proper Permissions (Linux)
```bash
chmod -R 755 /opt/sas-viya-python
chown -R $USER:$USER /opt/sas-viya-python
```

### Step 4: Install Python Dependencies

#### 4.1 Upgrade pip
```bash
pip install --upgrade pip
```

#### 4.2 Install Package Dependencies
```bash
# Install from requirements file
pip install -r requirements.txt

# Alternative: Install package in development mode
pip install -e .
```

#### 4.3 Verify Installation
```bash
pip list | grep -E "(saspy|swat|sasctl|pandas|numpy|scikit-learn)"
```

### Step 5: Configuration Setup

#### 5.1 Configure SAS Connection
```bash
# Edit the SAS configuration file
nano config/sascfg_personal.py

# Update the following values:
# - SAS Viya server URL
# - Authentication method
# - Username/password or OAuth token
```

#### 5.2 Configure Viya Settings
```bash
# Edit YAML configuration
nano config/viya_config.yaml

# Update:
# - Server URL and ports
# - Authentication details
# - Default caslibs and settings
```

#### 5.3 Set Environment Variables
```bash
# Linux - Add to ~/.bashrc or ~/.profile
export PYTHONPATH="/opt/sas-viya-python/src:$PYTHONPATH"
export SAS_VIYA_HOME="/opt/sas-viya-python"

# Windows - Set system environment variables
setx PYTHONPATH "C:\opt\sas-viya-python\src;%PYTHONPATH%"
setx SAS_VIYA_HOME "C:\opt\sas-viya-python"
```

## Post-Deployment Verification

### Step 6: Basic Functionality Tests

#### 6.1 Test Python Import
```bash
cd /opt/sas-viya-python
python3 -c "
import sys
sys.path.append('src')
from connections.viya_connection import ViyaConnection
print('✓ Package imports successfully')
"
```

#### 6.2 Test SAS Viya Connection
```bash
python3 -c "
import sys
sys.path.append('src')
from connections.viya_connection import ViyaConnection
conn = ViyaConnection()
# Test will depend on your specific configuration
print('✓ Connection module loaded successfully')
"
```

#### 6.3 Run Unit Tests
```bash
python3 -m pytest tests/ -v
```

### Step 7: Integration Testing

#### 7.1 Test Example Scripts
```bash
# Test churn analysis example (modify connection details first)
python3 examples/churn_example.py

# Test fraud detection example
python3 examples/fraud_example.py

# Test forecasting example
python3 examples/forecasting_example.py
```

#### 7.2 Test Complete Workflow
```bash
# Run the complete workflow example
python3 examples/complete_workflow.py
```

## Configuration

### SAS Configuration File (config/sascfg_personal.py)
```python
# Update these values for your environment
SAS_config_names = ['viya']

viya = {
    'url': 'https://your-viya-server.company.com',
    'context': 'SAS Studio compute context',
    'authkey': 'viya_user-pw',
    'options': ["fullstimer", "memsize=4G"]
}

viya_user_pw = {
    'url': 'https://your-viya-server.company.com',
    'user': 'your_username',
    'pw': 'your_password'
}
```

### Viya Configuration File (config/viya_config.yaml)
```yaml
viya:
  server:
    url: "https://your-viya-server.company.com"
    cas_port: 5570
    protocol: "cas"
  
  authentication:
    method: "user_password"
    username: "your_username"
    password: "your_password"
  
  settings:
    timeout: 300
    memory: "4G"
    log_level: "INFO"
```

## Testing

### Functional Tests
1. **Connection Tests**: Verify all three connection types (SASPy, SWAT, sasctl)
2. **Model Tests**: Test each model implementation
3. **Deployment Tests**: Verify deployment pipeline functionality
4. **Security Tests**: Validate security and governance features

### Performance Tests
1. **Load Testing**: Test with large datasets
2. **Memory Usage**: Monitor memory consumption
3. **Network Performance**: Test connection stability

### Integration Tests
1. **End-to-End Workflow**: Complete model development to deployment
2. **Error Handling**: Test error scenarios and recovery
3. **Monitoring**: Verify monitoring and alerting functionality

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Python Import Errors
**Symptoms**: `ModuleNotFoundError` when importing package modules
**Solution**:
```bash
# Ensure PYTHONPATH is set correctly
export PYTHONPATH="/opt/sas-viya-python/src:$PYTHONPATH"

# Or install package in development mode
pip install -e .
```

#### Issue 2: SAS Connection Failures
**Symptoms**: Connection timeout or authentication errors
**Solution**:
1. Verify network connectivity: `telnet your-viya-server.com 443`
2. Check credentials in configuration files
3. Verify firewall settings
4. Test with SAS Studio web interface first

#### Issue 3: Permission Errors (Linux)
**Symptoms**: Permission denied errors when running scripts
**Solution**:
```bash
# Fix file permissions
chmod -R 755 /opt/sas-viya-python
chown -R $USER:$USER /opt/sas-viya-python
```

#### Issue 4: Memory Issues
**Symptoms**: Out of memory errors during model training
**Solution**:
1. Increase VM memory allocation
2. Adjust memory settings in SAS configuration
3. Use data chunking for large datasets

#### Issue 5: Package Dependencies
**Symptoms**: Missing or incompatible package versions
**Solution**:
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Check for conflicts
pip check
```

### Log Files and Debugging
- **Application Logs**: `/opt/sas-viya-python/viya_app.log`
- **Python Logs**: Check console output and error messages
- **SAS Logs**: Available through SAS Viya interface
- **System Logs**: `/var/log/` (Linux) or Event Viewer (Windows)

## Rollback Procedures

### Emergency Rollback Steps

#### Step 1: Stop Application Services
```bash
# Stop any running Python processes
pkill -f "python.*sas_viya"
```

#### Step 2: Restore Previous Environment
```bash
# If backup was created
rm -rf /opt/sas-viya-python
mv /opt/sas-viya-python.backup /opt/sas-viya-python
```

#### Step 3: Restore System Configuration
```bash
# Restore environment variables
# Remove or comment out SAS Viya related exports from ~/.bashrc
```

#### Step 4: Verify Rollback
```bash
# Test system functionality
python3 --version
pip list
```

### Rollback Validation
- [ ] System returns to previous state
- [ ] No residual configuration remains
- [ ] Other applications unaffected
- [ ] System performance normal

## Maintenance

### Regular Maintenance Tasks

#### Weekly Tasks
- [ ] Check application logs for errors
- [ ] Monitor system resource usage
- [ ] Verify SAS Viya connectivity
- [ ] Review security logs

#### Monthly Tasks
- [ ] Update Python packages (test in development first)
- [ ] Review and rotate log files
- [ ] Performance monitoring and optimization
- [ ] Security patch assessment

#### Quarterly Tasks
- [ ] Full system backup
- [ ] Disaster recovery testing
- [ ] Security audit
- [ ] Documentation updates

### Monitoring and Alerting
1. **System Monitoring**: CPU, memory, disk usage
2. **Application Monitoring**: Connection health, error rates
3. **Security Monitoring**: Failed authentication attempts
4. **Performance Monitoring**: Response times, throughput

### Backup Strategy
1. **Configuration Backup**: Daily backup of config files
2. **Code Backup**: Version control with Git
3. **Data Backup**: Regular backup of processed data
4. **System Backup**: Weekly VM snapshots

## Security Considerations

### Access Control
- Use principle of least privilege
- Regular password rotation
- Multi-factor authentication where possible
- Network segmentation

### Data Protection
- Encrypt sensitive configuration data
- Secure credential storage
- Data masking for non-production environments
- Regular security assessments

### Compliance
- Follow organizational security policies
- Maintain audit trails
- Regular compliance reviews
- Document security procedures

## Support and Escalation

### Level 1 Support
- Basic connectivity issues
- Configuration problems
- User access issues

### Level 2 Support
- Complex technical issues
- Performance problems
- Integration issues

### Level 3 Support
- Architecture changes
- Security incidents
- Disaster recovery

### Contact Information
- **Primary Support**: [support-email@company.com]
- **Emergency Contact**: [emergency-contact@company.com]
- **SAS Viya Admin**: [sas-admin@company.com]

## Document Control

### Version History
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-09 | System Admin | Initial version |

### Review Schedule
- **Next Review Date**: 2025-04-09
- **Review Frequency**: Quarterly
- **Document Owner**: IT Operations Team

### Approval
- **Technical Review**: [Name, Date]
- **Security Review**: [Name, Date]
- **Management Approval**: [Name, Date]

---

**Note**: This SOP should be customized based on your specific environment, security requirements, and organizational policies. Always test deployment procedures in a development environment before applying to production systems.