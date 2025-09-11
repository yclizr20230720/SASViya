import os

# SAS Viya Configuration
SAS_config_names = ['viya']

viya = {
    'url': 'https://your-viya-server.com',
    'context': 'SAS Studio compute context',
    'authkey': 'viya_user-pw',
    'options': ["fullstimer", "memsize=4G"]
}

# Authentication (choose one method)
viya_user_pw = {
    'url': 'https://your-viya-server.com',
    'user': 'your_username',
    'pw': 'your_password'
}

# Or use OAuth token
viya_oauth = {
    'url': 'https://your-viya-server.com',
    'token': 'your_oauth_token'
}