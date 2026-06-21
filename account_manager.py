import os
import json
import shutil

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ACCOUNTS_FILE = os.path.join(_PROJECT_DIR, "accounts.json")
LEGACY_TOKEN = os.path.join(_PROJECT_DIR, "token.pickle")

def load_accounts():
    if not os.path.exists(ACCOUNTS_FILE):
        return {"current_account": None, "accounts": {}}
    try:
        with open(ACCOUNTS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"current_account": None, "accounts": {}}

def save_accounts(data):
    with open(ACCOUNTS_FILE, "w") as f:
        json.dump(data, f, indent=4)

def migrate_legacy_token():
    data = load_accounts()
    
    if os.path.exists(LEGACY_TOKEN):
        if not data["accounts"]:
            # Move legacy token to token_1.pickle
            new_token_name = "token_1.pickle"
            shutil.move(LEGACY_TOKEN, os.path.join(_PROJECT_DIR, new_token_name))
            
            # Note: We won't have the channel name until we make an API call,
            # so we'll set it as "Legacy Account".
            data["accounts"]["1"] = {
                "channel_name": "Legacy Account (Run a command to update name)",
                "token_file": new_token_name
            }
            data["current_account"] = "1"
            save_accounts(data)

def get_current_account_id():
    migrate_legacy_token()
    data = load_accounts()
    return data.get("current_account")

def set_current_account(acc_id):
    data = load_accounts()
    if str(acc_id) not in data.get("accounts", {}):
        raise ValueError(f"Account ID {acc_id} does not exist.")
    data["current_account"] = str(acc_id)
    save_accounts(data)

def add_account(channel_name, new_token_file=None):
    migrate_legacy_token()
    data = load_accounts()
    accounts = data.get("accounts", {})
    
    # Determine next ID
    existing_ids = [int(k) for k in accounts.keys() if k.isdigit()]
    next_id = str(max(existing_ids) + 1) if existing_ids else "1"
    
    if new_token_file is None:
        new_token_file = f"token_{next_id}.pickle"
        
    accounts[next_id] = {
        "channel_name": channel_name,
        "token_file": new_token_file
    }
    
    data["accounts"] = accounts
    
    # If it's the only account, set it as current
    if not data.get("current_account"):
        data["current_account"] = next_id
        
    save_accounts(data)
    return next_id, new_token_file

def update_account_name(acc_id, channel_name):
    data = load_accounts()
    if str(acc_id) in data.get("accounts", {}):
        data["accounts"][str(acc_id)]["channel_name"] = channel_name
        save_accounts(data)

def get_token_file_for_account(acc_id=None):
    migrate_legacy_token()
    data = load_accounts()
    
    if acc_id is None:
        acc_id = data.get("current_account")
        
    if not acc_id:
        return None
        
    acc = data.get("accounts", {}).get(str(acc_id))
    if not acc:
        return None
        
    return os.path.join(_PROJECT_DIR, acc["token_file"])

def get_account_name(acc_id=None):
    data = load_accounts()
    if acc_id is None:
        acc_id = data.get("current_account")
        
    if not acc_id:
        return None
        
    acc = data.get("accounts", {}).get(str(acc_id))
    if not acc:
        return None
        
    return acc.get("channel_name", "Unknown Channel")
