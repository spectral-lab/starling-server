FROM python:3.6
WORKDIR /app
COPY . /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 80 5000
ENV FLASK_APP src/app.py

# #env
# ENV FLASK_ENV development
#prod
ENV FLASK_ENV production

CMD ["flask", "run", "--host", "0.0.0.0"]