FROM python:3.9

WORKDIR /Deep-learning-model-to-identify-brain-tumors-using-MIR-images

COPY . /Deep-learning-model-to-identify-brain-tumors-using-MIR-images

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "sever_test.py"]
