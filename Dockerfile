FROM python:3.6
WORKDIR /app
COPY . /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 80 5000
ENV FLASK_APP src/app.py
ENV FLASK_ENV development
CMD ["flask", "run", "--host", "0.0.0.0"]
