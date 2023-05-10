FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install -r /app/requirements.txt

EXPOSE 80

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]