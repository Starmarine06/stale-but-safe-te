FROM python:3.9-slim
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy scipy networkx pyyaml
# Ryu 4.34 needs an old setuptools/eventlet pair and a non-isolated build (pbr hook);
# install it last so the pin does not affect torch.
RUN pip install --no-cache-dir "setuptools<58" wheel pbr && \
    pip install --no-cache-dir --no-build-isolation ryu==4.34 eventlet==0.30.2 "dnspython<2"
WORKDIR /app
COPY pyproject.toml /app/
COPY src /app/src
COPY configs /app/configs
ENV PYTHONPATH=/app/src
