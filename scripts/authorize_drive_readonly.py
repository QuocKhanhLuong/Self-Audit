"""One local OAuth consent flow; only credentials, never MRI, pass through this Mac."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile

READONLY = "https://www.googleapis.com/auth/drive.readonly"


def authorize(client_json, output_dir, timeout=600):
    import modal
    from google_auth_oauthlib.flow import InstalledAppFlow
    client_json = Path(client_json).expanduser().resolve()
    config = json.loads(client_json.read_text())
    if not config.get("installed", {}).get("client_id") or not config["installed"].get("client_secret"):
        raise ValueError("A Google OAuth Desktop client JSON is required; do not supply a token in chat")
    for logger in ("google_auth_oauthlib", "google.auth", "urllib3", "requests_oauthlib", "oauthlib"):
        logging.getLogger(logger).setLevel(logging.CRITICAL)
    flow = InstalledAppFlow.from_client_config(config, scopes=[READONLY], autogenerate_code_verifier=True)
    # The browser visit is the sole consent operation; the callback and upload are automatic.
    credentials = flow.run_local_server(host="127.0.0.1", port=0, open_browser=False,
        timeout_seconds=timeout, authorization_prompt_message="Open this Google consent link:\n{url}",
        success_message="Drive read-only consent received. You can close this tab.",
        access_type="offline", prompt="consent")
    if not credentials.refresh_token:
        raise ValueError("Google did not return a refresh token; no Modal secret created")
    granted = set(credentials.granted_scopes or credentials.scopes or [])
    if READONLY not in granted or any(s in granted for s in ("https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/drive.file")):
        raise ValueError("Expected a read-only Drive grant")
    private = Path(output_dir).expanduser().resolve()
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(private, 0o700)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token_path = private / f"drive-readonly-{stamp}.json"
    fd, temporary = tempfile.mkstemp(dir=private)
    try:
        os.fchmod(fd, 0o600)
        serialized = credentials.to_json()
        with os.fdopen(fd, "w") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, token_path)
        secret_name = "selfaudit-drive-readonly-" + stamp.lower()
        modal.Secret.objects.create(secret_name, {"SELFAUDIT_DRIVE_AUTHORIZED_USER_JSON": serialized})
    finally:
        Path(temporary).unlink(missing_ok=True)
    receipt = {"status": "AUTHORIZED", "scope": READONLY, "modal_secret": secret_name,
               "credential_file": str(token_path), "credential_file_mode": "0600", "dataset_downloaded_to_mac": False}
    (private / "LATEST_AUTH.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--client-json", required=True)
    p.add_argument("--output-dir", default="~/.config/selfaudit/private")
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()
    try:
        print(json.dumps(authorize(args.client_json, args.output_dir, args.timeout), indent=2))
    except Exception as exc:
        # OAuth library exceptions may include codes or response bodies; never echo them.
        print(json.dumps({"status": "BLOCKED_AUTHORIZATION", "error_type": type(exc).__name__, "details_withheld": True}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
