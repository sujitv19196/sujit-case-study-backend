FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/cpu
COPY . .
# COPY ../scripts/gpt4querygen.py /app
# COPY ../faiss/faiss_index_v7_10d /app/faiss
EXPOSE 5001
CMD ["flask", "--app", "./", "run", "--host", "0.0.0.0", "--port", "5001"]