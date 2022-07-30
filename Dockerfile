FROM python:3.9

WORKDIR /oxus

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . . 

CMD ["uvicorn", "inference:app"]


