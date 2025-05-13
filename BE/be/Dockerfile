FROM python:3.10.11

WORKDIR /app/be

COPY . /app/be

COPY requirements.txt /app/requirements.txt

ENV PYTHONPATH=/app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "be.main:app", "--host", "0.0.0.0", "--port", "8000"]

