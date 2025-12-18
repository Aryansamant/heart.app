FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit /app/streamlit

EXPOSE 8501

CMD ["streamlit", "run", "streamlit/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
