# Dockerfile for normal use with docker. This exposes the streamlit app on port 8501.
FROM python:3.8
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD [ "streamlit", "run", "src/app/app.py"]