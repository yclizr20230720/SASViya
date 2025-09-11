import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import swat

class FraudDetectionPipeline:
    def __init__(self, cas_session):
        self.cas = cas_session
        self.fraud_rules = {}
        self.model_table = None
        
    def create_fraud_features(self, transaction_data):
        """Create features for fraud detection"""
        cas_table = self.cas.upload(transaction_data, casout='transactions_temp')
        
        # CAS-based feature engineering
        self.cas.datastep.runcode(f"""
        data work.fraud_features;
            set {cas_table.name};
            
            /* Time-based features */
            transaction_hour = hour(transaction_timestamp);
            is_weekend = (weekday(transaction_timestamp) in (1,7));
            is_night = (transaction_hour < 6 or transaction_hour > 22);
            
            /* Amount-based features */
            log_amount = log(amount + 1);
            amount_zscore = (amount - 50) / 100;  /* Assuming mean=50, std=100 */
            
            /* Velocity features (requires historical data) */
            /* This would typically involve more complex CAS programming */
            
            /* Merchant category risk score */
            select (merchant_category);
                when ('ATM') risk_score = 0.1;
                when ('GAS_STATION') risk_score = 0.2;
                when ('GROCERY') risk_score = 0.1;
                when ('RESTAURANT') risk_score = 0.3;
                when ('ONLINE') risk_score = 0.8;
                otherwise risk_score = 0.5;
            end;
        run;
        """)
        
        return self.cas.CASTable('fraud_features', caslib='work')
    
    def build_fraud_model(self, training_data):
        """Build fraud detection model using SAS Viya ML"""
        # Load training data
        train_table = self.cas.upload(training_data, casout='fraud_training')
        
        # Feature selection
        self.cas.loadactionset('featuremachine')
        feature_result = self.cas.featuremachine.featureMachine(
            table=train_table,
            target='is_fraud',
            inputs=['transaction_hour', 'amount', 'merchant_category', 'is_weekend', 
                   'is_night', 'log_amount', 'risk_score'],
            maxFeatures=20
        )
        
        # Train gradient boosting model
        self.cas.loadactionset('gradboost')
        model_result = self.cas.gradboost.gbtreeTrain(
            table=train_table,
            target='is_fraud',
            inputs=feature_result.SelectedVars.Variable.tolist(),
            nTrees=100,
            learningRate=0.1,
            maxDepth=6,
            casOut={'name': 'fraud_model', 'replace': True}
        )
        
        self.model_table = self.cas.CASTable('fraud_model')
        return model_result
    
    def create_real_time_scoring(self):
        """Create real-time scoring pipeline"""
        # Create scoring code
        scoring_code = """
        data _null_;
            if _n_ = 1 then do;
                declare hash fraud_model(dataset: 'work.fraud_model');
                fraud_model.defineKey('_PartID_');
                fraud_model.defineData('_PBM0_', '_PBM1_');
                fraud_model.defineDone();
            end;
            
            set work.new_transactions;
            
            /* Feature engineering */
            transaction_hour = hour(transaction_timestamp);
            is_weekend = (weekday(transaction_timestamp) in (1,7));
            is_night = (transaction_hour < 6 or transaction_hour > 22);
            log_amount = log(amount + 1);
            
            /* Score transaction */
            if fraud_model.find() = 0 then do;
                fraud_probability = _PBM1_;
                fraud_flag = (fraud_probability > 0.5);
            end;
            else do;
                fraud_probability = .;
                fraud_flag = .;
            end;
            
            output;
        run;
        """
        
        return scoring_code
    
    def process_streaming_data(self, stream_table_name):
        """Process streaming transactions for real-time fraud detection"""
        # Set up event stream processing
        self.cas.loadactionset('eventStreamProcessing')
        
        # Create ESP project for real-time scoring
        esp_project = f"""
        <project name="fraud_detection" pubsub="auto" threads="4">
            <contqueries>
                <contquery name="cq1">
                    <windows>
                        <window-source name="transactions" insert-only="true">
                            <schema>
                                <fields>
                                    <field name="transaction_id" type="string" key="true"/>
                                    <field name="amount" type="double"/>
                                    <field name="merchant_category" type="string"/>
                                    <field name="transaction_timestamp" type="stamp"/>
                                    <field name="customer_id" type="string"/>
                                </fields>
                            </schema>
                        </window-source>
                        
                        <window-calculate name="features">
                            <schema>
                                <fields>
                                    <field name="transaction_id" type="string" key="true"/>
                                    <field name="amount" type="double"/>
                                    <field name="log_amount" type="double"/>
                                    <field name="transaction_hour" type="int32"/>
                                    <field name="is_weekend" type="int32"/>
                                    <field name="risk_score" type="double"/>
                                </fields>
                            </schema>
                            <expr>
                                log_amount = log(amount + 1);
                                transaction_hour = hour(transaction_timestamp);
                                is_weekend = (weekday(transaction_timestamp) in (1,7)) ? 1 : 0;
                                risk_score = (merchant_category == "ONLINE") ? 0.8 : 0.2;
                            </expr>
                        </window-calculate>
                        
                        <window-score name="fraud_scores" model="fraud_model">
                            <schema>
                                <fields>
                                    <field name="transaction_id" type="string" key="true"/>
                                    <field name="fraud_probability" type="double"/>
                                    <field name="fraud_flag" type="int32"/>
                                </fields>
                            </schema>
                        </window-score>
                    </windows>
                    
                    <edges>
                        <edge source="transactions" target="features"/>
                        <edge source="features" target="fraud_scores"/>
                    </edges>
                </contquery>
            </contqueries>
        </project>
        """
        
        return esp_project