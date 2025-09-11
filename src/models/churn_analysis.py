import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

class CustomerChurnAnalysis:
    def __init__(self, viya_conn):
        self.viya = viya_conn
        self.model = None
        self.feature_importance = None
        
    def load_data_from_viya(self, table_name, caslib='PUBLIC'):
        """Load customer data from SAS Viya"""
        # Load data using SWAT
        cas_table = self.viya.cas_session.load_table(
            path=table_name, 
            caslib=caslib
        )
        
        # Convert to pandas DataFrame
        df = cas_table.to_frame()
        print(f"Loaded {len(df)} records from {table_name}")
        return df
    
    def feature_engineering(self, df):
        """Create features for churn prediction"""
        # Calculate customer lifetime value
        df['clv'] = df['monthly_charges'] * df['tenure']
        
        # Create interaction features
        df['charges_per_tenure'] = df['total_charges'] / (df['tenure'] + 1)
        df['service_count'] = (
            df['phone_service'].astype(int) +
            df['internet_service'].notna().astype(int) +
            df['streaming_tv'].astype(int) +
            df['streaming_movies'].astype(int)
        )
        
        # Encode categorical variables
        categorical_cols = ['gender', 'partner', 'dependents', 'internet_service', 'contract']
        df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
        
        return df_encoded
    
    def build_churn_model(self, df, target_col='churn'):
        """Build and train churn prediction model"""
        # Prepare features
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        print("Model Performance:")
        print(classification_report(y_test, y_pred))
        print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model, X_test, y_test, y_pred, y_prob
    
    def deploy_to_viya(self, model_name='customer_churn_model'):
        """Deploy model to SAS Viya for scoring"""
        # Save model locally first
        model_path = f'{model_name}.pkl'
        joblib.dump(self.model, model_path)
        
        # Deploy using sasctl
        from sasctl.services import model_repository
        from sasctl import publish_model
        
        # Register model in SAS Model Manager
        model_obj = publish_model(
            model=self.model,
            name=model_name,
            description="Customer churn prediction model",
            input_data=self.X_sample,  # Sample input data
            project_name="Customer Analytics"
        )
        
        print(f"Model {model_name} deployed successfully to SAS Viya")
        return model_obj