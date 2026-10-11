FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    mininet openvswitch-switch openvswitch-testcontroller iperf3 iproute2 iputils-ping net-tools \
    python3 python3-pip python3-numpy python3-scipy python3-networkx python3-yaml procps \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml /app/
COPY src /app/src
COPY configs /app/configs
ENV PYTHONPATH=/app/src
COPY docker/entry_mininet.sh /entry.sh
RUN chmod +x /entry.sh
ENTRYPOINT ["/entry.sh"]
