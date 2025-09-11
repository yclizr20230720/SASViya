# Base classes for consistent structure
class BaseViyaModel:
    """Base class for all Viya models"""
    
    def __init__(self, viya_connection):
        self.viya = viya_connection
        self.model = None
        self.model_metadata = {}
    
    def preprocess_data(self, data):
        """Override in subclasses"""
        raise NotImplementedError
    
    def train_model(self, training_data):
        """Override in subclasses"""
        raise NotImplementedError
    
    def validate_model(self, validation_data):
        """Override in subclasses"""
        raise NotImplementedError
    
    def deploy_model(self, model_name):
        """Common deployment logic"""
        # Implementation here
        pass