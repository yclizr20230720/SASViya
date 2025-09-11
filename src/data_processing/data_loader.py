"""
Data loading utilities for SAS Viya
"""

import pandas as pd
import swat

class DataLoader:
    """Utility class for loading data from various sources into SAS Viya"""
    
    def __init__(self, cas_session):
        self.cas = cas_session
    
    def load_from_csv(self, file_path, casout_name, **kwargs):
        """Load CSV file into CAS table"""
        df = pd.read_csv(file_path, **kwargs)
        cas_table = self.cas.upload(df, casout=casout_name)
        return cas_table
    
    def load_from_database(self, connection_string, query, casout_name):
        """Load data from database into CAS table"""
        # Implementation would depend on specific database type
        pass
    
    def load_from_api(self, api_url, headers, casout_name):
        """Load data from REST API into CAS table"""
        import requests
        response = requests.get(api_url, headers=headers)
        data = response.json()
        df = pd.DataFrame(data)
        cas_table = self.cas.upload(df, casout=casout_name)
        return cas_table