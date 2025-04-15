FROM python:3

WORKDIR /mlflow_test

COPY . /mlflow_test/

RUN pip install --upgrade pip
RUN pip install mlflow

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]