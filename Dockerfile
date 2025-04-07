FROM python:3.12-slim

WORKDIR /app

COPY ./models /app

RUN pip install --no-cache-dir -r requirements_st.txt

EXPOSE 8501

CMD ["streamlit", "run", "./src/streamlit/app.py"]