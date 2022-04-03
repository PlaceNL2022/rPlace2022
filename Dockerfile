FROM python:latest

COPY . /opt/PlaceNL
RUN pip install --force /opt/PlaceNL

ENTRYPOINT ["PlaceNL"]
