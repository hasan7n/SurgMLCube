FROM temp/mycont:0.0.3

LABEL org.opencontainers.image.authors="MLPerf MLBox Working Group"

COPY ./requirements.txt project/requirements.txt 

RUN pip3 install --no-cache-dir -r project/requirements.txt

COPY . /project

WORKDIR /project

ENTRYPOINT ["python3", "mlcube.py"]
