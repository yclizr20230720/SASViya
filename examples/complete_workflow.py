"""
Complete workflow example showing all components working together
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from connections.viya_connection import ViyaConnection
from models.churn_analysis import CustomerChurnAnalysis
from deployment.model_deployment import ModelDeploymentPipeline
from deployment.monitoring import ModelMonitoring
from utils.error_handling import logger

def complete_deployment_example():
    """Complete example showing all components working together"""
    
    # Initialize connections
    viya_conn = ViyaConnection()
    
    try:
        # Establish connections
        sas_session = viya_conn.connect_saspy()
        cas_session = viya_conn.connect_swat('your-cas-server.com')
        sasctl_session = viya_conn.connect_sasctl('your-viya-server.com', 'username', 'password')
        
        if not all([sas_session, cas_session, sasctl_session]):
            raise Exception("Failed to establish all required connections")
        
        logger.info("All connections established successfully")
        
        # 1. Model Development
        logger.info("Starting model development...")
        churn_analysis = CustomerChurnAnalysis(viya_conn)
        
        # Load and process data
        df = churn_analysis.load_data_from_viya('customer_processed_data')
        df_processed = churn_analysis.feature_engineering(df)
        
        # Build and evaluate model
        model, X_test, y_test, y_pred, y_prob = churn_analysis.build_churn_model(df_processed)
        logger.info("Model training completed successfully")
        
        # 2. Model Deployment
        logger.info("Starting model deployment...")
        deployment_pipeline = ModelDeploymentPipeline(sasctl_session)
        
        model_obj, image, publication = deployment_pipeline.deploy_python_model(
            model, 'customer_churn_v2', 'Customer Analytics'
        )
        logger.info("Model deployed successfully")
        
        # 3. Model Monitoring Setup
        logger.info("Setting up model monitoring...")
        monitoring = ModelMonitoring(sasctl_session)
        
        monitoring_config = {
            'drift_threshold': 0.2,
            'performance_threshold': 0.05,
            'baseline_auc': 0.85,
            'target_event': '1',
            'bias_threshold': 0.1,
            'sensitive_variables': ['gender', 'age_group']
        }
        
        monitors = monitoring.setup_model_monitoring('customer_churn_v2', monitoring_config)
        logger.info("Model monitoring configured successfully")
        
        # 4. Generate deployment report
        deployment_report = {
            'model_name': 'customer_churn_v2',
            'model_id': model_obj.id if hasattr(model_obj, 'id') else 'N/A',
            'deployment_status': 'SUCCESS',
            'monitoring_enabled': True,
            'performance_metrics': {
                'auc_score': f"{roc_auc_score(y_test, y_prob):.4f}",
                'accuracy': f"{(y_pred == y_test).mean():.4f}"
            }
        }
        
        logger.info("Deployment completed successfully!")
        logger.info(f"Deployment report: {deployment_report}")
        
        return deployment_report
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise
        
    finally:
        # Always close connections
        viya_conn.close_connections()
        logger.info("All connections closed")

if __name__ == "__main__":
    try:
        report = complete_deployment_example()
        print("Deployment completed successfully!")
        print(f"Report: {report}")
    except Exception as e:
        print(f"Deployment failed: {e}")
        sys.exit(1)