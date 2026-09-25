# Final Deployment Checklist

## Local test
1. `py -m venv venv`
2. `venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. `python app.py`
5. Open `http://127.0.0.1:5000`
6. Test the prediction form on mobile-width and desktop-width browser windows.
7. Check `http://127.0.0.1:5000/health` returns `status: ok`.

## Render
1. Push this project folder to GitHub.
2. Render → New → Web Service.
3. Connect the repository.
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn --workers 1 --threads 4 --timeout 120 app:app`
6. Add `SECRET_KEY` or use the included `render.yaml` Blueprint.
7. Deploy.
8. Open `/health` after deployment.
9. Run one prediction to verify the model and loading overlay.

## Important
- The raw dataset is stored as `data/house_prices.csv.gz`, so it stays below GitHub's normal per-file upload limit.
- Pandas reads the compressed CSV directly for retraining/notebook work.
- The deployed Flask app does not load the 187k-row dataset at runtime; it loads only the saved model and metadata.
