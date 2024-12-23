FROM python:3.13

WORKDIR /app

ENV WWWUSER=1000
ENV WWWGROUP=1000

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/awscliv2.zip"
RUN	unzip /awscliv2.zip -d /

RUN	/aws/install

COPY requirements.txt .

RUN pip install -r requirements.txt

# ENTRYPOINT [ "python3" ]

CMD [ "bash" ]