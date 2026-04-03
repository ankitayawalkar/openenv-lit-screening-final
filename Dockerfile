FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install openai

CMD ["python", "inference.py"]