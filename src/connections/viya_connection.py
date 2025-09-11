import saspy
import swat
import pandas as pd
import numpy as np
from sasctl import Session
import warnings
warnings.filterwarnings('ignore')

class ViyaConnection:
    def __init__(self, config_name='viya'):
        self.sas_session = None
        self.cas_session = None
        self.sasctl_session = None
        self.config_name = config_name
        
    def connect_saspy(self):
        """Connect using SASPy for traditional SAS programming"""
        try:
            self.sas_session = saspy.SASsession(cfgname=self.config_name)
            print("SASPy connection established successfully")
            return self.sas_session
        except Exception as e:
            print(f"SASPy connection failed: {e}")
            return None
    
    def connect_swat(self, host, port=5570, protocol='cas'):
        """Connect using SWAT for CAS operations"""
        try:
            self.cas_session = swat.CAS(host, port, protocol=protocol)
            print("SWAT connection established successfully")
            return self.cas_session
        except Exception as e:
            print(f"SWAT connection failed: {e}")
            return None
    
    def connect_sasctl(self, host, username, password):
        """Connect using sasctl for model management"""
        try:
            self.sasctl_session = Session(host, username, password)
            print("sasctl connection established successfully")
            return self.sasctl_session
        except Exception as e:
            print(f"sasctl connection failed: {e}")
            return None
    
    def close_connections(self):
        """Close all connections"""
        if self.sas_session:
            self.sas_session.endsas()
        if self.cas_session:
            self.cas_session.close()
        if self.sasctl_session:
            self.sasctl_session.close()