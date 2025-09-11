# SAS Viya Python Development Project

This project contains a comprehensive Python development framework for SAS Viya integration, including real-world scenarios, best practices, and deployment strategies.

## Project Structure

```
sas_viya_project/
├── config/                     # Configuration files
│   ├── __init__.py
│   ├── sascfg_personal.py     # SAS configuration
│   └── viya_config.yaml       # Viya settings
├── src/                       # Source code
│   ├── __init__.py
│   ├── connections/           # Connection management
│   │   ├── __init__.py
│   │   └── viya_connection.py
│   ├── data_processing/       # Data processing modules
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── feature_engineering.py
│   ├── models/               # Model implementations
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── churn_analysis.py
│   │   ├── fraud_detection.py
│   │   └── time_series_forecasting.py
│   ├── deployment/           # Deployment utilities
│   │   ├── __init__.py
│   │   ├── model_deployment.py
│   │   └── monitoring.py
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   ├── error_handling.py
│   │   └── performance_optimizer.py
│   └── security/            # Security and governance
│       ├── __init__.py
│       └── governance.py
├── examples/                # Example usage scripts
│   ├── __init__.py
│   ├── churn_example.py
│   ├── fraud_example.py
│   └── forecasting_example.py
├── tests/                   # Test files
├── notebooks/              # Jupyter notebooks
├── requirements.txt        # Python dependencies
└── setup.py               # Package setup
```

## Prerequisites

- SAS Viya 3.5+ or SAS Viya 4
- Python 3.7+
- Required Python packages (see requirements.txt)

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Configure SAS Viya connection in `config/sascfg_personal.py`

3. Update Viya settings in `config/viya_config.yaml`

## Usage

See the `examples/` directory for complete usage examples of each component.

## Features

- **Connection Management**: Unified connection handling for SASPy, SWAT, and sasctl
- **Real-World Scenarios**: Customer churn prediction, fraud detection, time series forecasting
- **Best Practices**: Error handling, logging, performance optimization
- **Deployment**: Model deployment pipelines and monitoring
- **Security**: Data security and governance frameworks


📁 Configuration (config/)
sascfg_personal.py - SAS Viya connection configuration
viya_config.yaml - YAML-based configuration settings
📁 Source Code (src/)
connections/ - Connection management for SASPy, SWAT, and sasctl
models/ - All model implementations:
base_model.py - Base class for consistent model structure
churn_analysis.py - Customer churn prediction pipeline
fraud_detection.py - Real-time fraud detection system
time_series_forecasting.py - Advanced forecasting models
deployment/ - Model deployment and monitoring utilities
utils/ - Error handling and performance optimization
data_processing/ - Data loading and feature engineering
security/ - Security and governance frameworks
📁 Examples (examples/)
Complete working examples for each scenario
complete_workflow.py - End-to-end deployment example
📁 Tests (tests/)
Unit tests for connection management
Framework for testing other components
📁 Documentation
Comprehensive README with setup instructions
Requirements file with all dependencies
Setup.py for package installation
Key Features Implemented
✅ Connection Management - Unified handling of all SAS Viya connections
✅ Real-World Scenarios - Customer churn, fraud detection, time series forecasting
✅ Best Practices - Error handling, logging, performance optimization
✅ Deployment Pipeline - Complete model deployment and monitoring
✅ Security & Governance - Data security and compliance frameworks
✅ Modular Design - Clean separation of concerns and reusable components

Next Steps
Configure your environment:

Update config/sascfg_personal.py with your SAS Viya server details
Install dependencies: pip install -r requirements.txt
Run examples:

Start with examples/churn_example.py to test your setup
Try the complete workflow in examples/complete_workflow.py
Customize for your needs:

Modify the models in src/models/ for your specific use cases
Add your own data processing logic in src/data_processing/
The project is now ready for immediate use and provides a solid foundation for SAS Viya Python development with enterprise-grade practices built in!