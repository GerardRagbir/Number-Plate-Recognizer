FROM tiangolo/uvicorn-gunicorn:python3.10

LABEL maintainer="Gerard Ragbir <gerard.ragbir@gmail.com>"
WORKDIR /
COPY .detector/src .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000