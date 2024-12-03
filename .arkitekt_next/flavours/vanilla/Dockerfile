FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
RUN pip install cellpose




RUN mkdir /app
WORKDIR /app
COPY .arkitekt_next /app/.arkitekt_next
COPY app.py /app/app.py
COPY test.py /app/test.py
RUN python test.py


RUN pip install "arkitekt-next[all]>=0.8.48"