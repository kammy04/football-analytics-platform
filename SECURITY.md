# Security notes

This project started from the "Data Mining & Information Security" plan, so the data layer is built
with a few basic protections, even though the data itself is public.

| Risk | What the code does | Where |
|---|---|---|
| SQL injection | Every query that takes user input uses `?` placeholders; values are bound by SQLite, never formatted into the SQL string. A test sends `A' OR '1'='1` and checks it matches nothing. | `src/football/db.py` → `team_matches()`, `tests/test_features.py` |
| Accidental writes | The app opens SQLite with `mode=ro`, so even a bug in the UI can't modify or delete data. Tested. | `db.connect_readonly()` |
| Untrusted input | Team and player names from the UI are checked against the values loaded from the database before use. | `app/streamlit_app.py` |
| Secrets in Git | No credentials are needed. `.streamlit/secrets.toml`, the database and model files are git-ignored. | `.gitignore` |
| Personal data | Only public match results and video-game player ratings; no user data is collected or stored. | - |
| Untrusted model files | `joblib` files can execute code when loaded, so the app only loads the model that `run_pipeline.py` writes locally, never one downloaded from elsewhere. | `models/` |

**If deployed publicly** (e.g. Streamlit Community Cloud), serve over HTTPS (the default there). If user
accounts were added, store passwords only as salted hashes (e.g. `hashlib.scrypt` or bcrypt) and use the
platform's auth rather than rolling your own.
