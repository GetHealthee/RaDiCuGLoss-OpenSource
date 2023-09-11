FROM python:3.10-slim-buster

COPY . ./src

WORKDIR ./src

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5678

ENV PYTHONPATH "${PYTHONPATH}:/src"

CMD ["python", "app/app.py"]