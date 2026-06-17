import os
import pickle
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

# Only YouTube upload — Drive downloads are public, no auth needed
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
]

def get_credentials():
    """
    Handles Google OAuth flow and returns the credentials.
    Caches the credentials in token.pickle inside the project directory.
    """
    creds = None

    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_PATH, SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "wb") as f:
            pickle.dump(creds, f)

    return creds
