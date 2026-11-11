# Hugging Face Space (Docker SDK) for the Streamlit app. HF no longer offers a built-in Streamlit SDK.
FROM python:3.12-slim

# Spaces run containers as a non-root user (uid 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
USER user
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Hugging Face Spaces route traffic to port 8501 (set as app_port in README.md)
EXPOSE 8501
# XSRF/CORS protection is off so CSV uploads work inside the Space's iframe (demo app, no login or secrets)
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false", "--browser.gatherUsageStats=false"]
