# Live Deployment

## Local
```bash
pip install -r requirements.txt
streamlit run app.py
```

Create `.streamlit/secrets.toml` locally:
```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-5.6-luna"
```
Never commit this file.

## Streamlit Community Cloud
1. Push the project to GitHub.
2. Create an app at Streamlit Community Cloud.
3. Select your repo, `main`, and `app.py`.
4. Open Advanced settings.
5. Paste the same TOML into **Secrets**.
6. Deploy.
7. Set a memorable custom `streamlit.app` subdomain.
8. Test the URL in an incognito window.

The app should show **Mode: Live LLM**.
