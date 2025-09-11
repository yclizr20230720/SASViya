from sasctl import Session
from sasctl.services import model_repository as mr
from sasctl.services import model_management as mm

class ModelDeploymentPipeline:
    def __init__(self, session):
        self.session = session
    
    def deploy_python_model(self, model, model_name, project_name):
        """Deploy Python model to SAS Viya"""
        
        # Step 1: Register model
        model_obj = mr.create_model(
            model=model,
            name=model_name,
            project=project_name,
            description=f"Python model: {model_name}"
        )
        
        # Step 2: Create model image
        image = mm.create_model_version(
            model_obj,
            version_name="1.0",
            files=[
                "model.pkl",  # Your serialized model
                "score.py",   # Scoring script
                "requirements.txt"
            ]
        )
        
        # Step 3: Deploy to destination
        destination = mm.create_destination(
            name=f"{model_name}_destination",
            destination_type="microAnalyticService",
            properties={
                "baseRepoUrl": "your-docker-registry.com",
                "kubernetesCluster": "your-k8s-cluster"
            }
        )
        
        # Step 4: Publish model
        publication = mm.publish_model(
            model_obj,
            destination,
            name=f"{model_name}_publication"
        )
        
        return model_obj, image, publication
    
    def create_scoring_script(self, model_path, feature_names):
        """Create scoring script for deployment"""
        scoring_script = f"""
import pickle
import pandas as pd
import numpy as np

def score_model(input_data):
    '''
    Score function for deployed model
    '''
    # Load model
    with open('{model_path}', 'rb') as f:
        model = pickle.load(f)
    
    # Prepare features
    features = input_data[{feature_names}]
    
    # Generate predictions
    predictions = model.predict_proba(features)[:, 1]
    
    # Return results
    return pd.DataFrame({{
        'prediction': predictions,
        'risk_level': ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' 
                      for p in predictions]
    }})

# Model metadata
MODEL_NAME = "{model_path}"
INPUT_VARIABLES = {feature_names}
OUTPUT_VARIABLES = ['prediction', 'risk_level']
        """
        
        with open('score.py', 'w') as f:
            f.write(scoring_script)
        
        return 'score.py'