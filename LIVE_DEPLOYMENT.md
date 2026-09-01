# Live Deployment

1. Push this project to a public GitHub repository.
2. Create a Streamlit Community Cloud app.
3. Select the `main` branch and `app.py`.
4. In Advanced settings → Secrets add:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-5.6-luna"
```

5. Choose a professional `streamlit.app` subdomain.
6. Deploy.
7. Test the public URL in an incognito browser.
8. Confirm the sidebar says **Live LLM connected**.

Never commit `.streamlit/secrets.toml`.
