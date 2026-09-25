# Render Deployment Guide

1. Push this folder to a GitHub repository.
2. Sign in to Render.
3. Choose **New → Web Service**.
4. Connect the GitHub repository.
5. Runtime: Python.
6. Build command: `pip install -r requirements.txt`
7. Start command: `gunicorn app:app`
8. Add environment variable `SECRET_KEY`.
9. Deploy.
10. Verify `/health` returns `status: ok`.

`render.yaml` is also included for Blueprint-style deployment.

> A live public URL cannot be created from this package alone because deployment requires authorization to the owner's GitHub/Render account.
