FROM python:3.11.7

WORKDIR /app
COPY flask_app/ /app/

COPY model/model_transform.pkl /app/model/model_transform.pkl
COPY reports/experiment_info.json /app/reports/experiment_info.json
RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python","app.py"]
# CMD ["gunicorn","-b","0.0.0.0:5000","app:app"]
