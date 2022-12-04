FROM python:3.10
WORKDIR /detector/src/
COPY ./detector/src/requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000