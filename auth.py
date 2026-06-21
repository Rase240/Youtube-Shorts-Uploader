import os
import pickle
import json
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Resolve all paths relative to THIS file's directory (the project root)
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(_PROJECT_DIR, ".env"))

CLIENT_SECRETS_PATH = os.path.join(
    _PROJECT_DIR, os.getenv("CLIENT_SECRETS_PATH", "client_secrets.json")
)
TOKEN_PATH = os.path.join(_PROJECT_DIR, "token.pickle")

# YouTube access for upload, delete, update, and read
SCOPES = [
    "https://www.googleapis.com/auth/youtube",
]

def get_credentials(token_file=None):
    """
    Handles Google OAuth flow and returns the credentials.
    Reads from the specified token_file.
    """
    creds = None

    if token_file and os.path.exists(token_file):
        with open(token_file, "rb") as f:
            creds = pickle.load(f)

    # Force re-authentication if cached credentials do not contain all required scopes
    if creds and hasattr(creds, 'scopes'):
        if not all(scope in creds.scopes for scope in SCOPES):
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            if token_file:
                with open(token_file, "wb") as f:
                    pickle.dump(creds, f)
        else:
            raise Exception("Credentials invalid or not found. Please authenticate first using oauth.")

    return creds

def get_authorization_url():
    """
    Returns the Google OAuth authorization URL for out-of-band flow.
    """
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_PATH, SCOPES)
    flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
    auth_url, _ = flow.authorization_url(prompt='consent')
    
    # Save the code verifier state for PKCE
    state_file = os.path.join(_PROJECT_DIR, "oauth_state.json")
    with open(state_file, "w") as f:
        json.dump({"code_verifier": getattr(flow, 'code_verifier', None)}, f)
        
    return auth_url

def get_credentials_from_code(code, token_file):
    """
    Exchanges the authorization code for credentials and saves it to token_file.
    """
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_PATH, SCOPES)
    flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
    
    # Load the code verifier state for PKCE
    state_file = os.path.join(_PROJECT_DIR, "oauth_state.json")
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            data = json.load(f)
            if data.get("code_verifier"):
                flow.code_verifier = data["code_verifier"]
        os.remove(state_file)
        
    flow.fetch_token(code=code)
    creds = flow.credentials
    
    with open(token_file, "wb") as f:
        pickle.dump(creds, f)
        
    return creds
