FROM python:3.9-slim
RUN pip install --no-cache-dir "setuptools<58" wheel && \
    pip install --no-cache-dir ryu==4.34 eventlet==0.30.2 "dnspython<2" \
        numpy scipy networkx pyyaml
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
WORKDIR /app
COPY pyproject.toml /app/
COPY src /app/src
COPY configs /app/configs
ENV PYTHONPATH=/app/src
