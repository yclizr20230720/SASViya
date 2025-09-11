"""
Tests for connection management
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from connections.viya_connection import ViyaConnection

class TestViyaConnection(unittest.TestCase):
    
    def setUp(self):
        self.viya_conn = ViyaConnection()
    
    @patch('connections.viya_connection.saspy.SASsession')
    def test_connect_saspy_success(self, mock_saspy):
        """Test successful SASPy connection"""
        mock_session = Mock()
        mock_saspy.return_value = mock_session
        
        result = self.viya_conn.connect_saspy()
        
        self.assertEqual(result, mock_session)
        self.assertEqual(self.viya_conn.sas_session, mock_session)
    
    @patch('connections.viya_connection.swat.CAS')
    def test_connect_swat_success(self, mock_swat):
        """Test successful SWAT connection"""
        mock_session = Mock()
        mock_swat.return_value = mock_session
        
        result = self.viya_conn.connect_swat('test-server.com')
        
        self.assertEqual(result, mock_session)
        self.assertEqual(self.viya_conn.cas_session, mock_session)
    
    def test_close_connections(self):
        """Test closing all connections"""
        # Mock sessions
        self.viya_conn.sas_session = Mock()
        self.viya_conn.cas_session = Mock()
        self.viya_conn.sasctl_session = Mock()
        
        self.viya_conn.close_connections()
        
        self.viya_conn.sas_session.endsas.assert_called_once()
        self.viya_conn.cas_session.close.assert_called_once()
        self.viya_conn.sasctl_session.close.assert_called_once()

if __name__ == '__main__':
    unittest.main()