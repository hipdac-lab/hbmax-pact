FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        python3.6 \
        python3-dev \
        python3-pip \
        python3-setuptools && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir conan==1.51.0

RUN ln -s /usr/bin/python3.6 /usr/bin/python

ENV HBMAX_ROOT=/opt
WORKDIR $HBMAX_ROOT
RUN mkdir -p $HBMAX_ROOT/test-data
RUN mkdir -p $HBMAX_ROOT/experiments
ENV DATADIR=$HBMAX_ROOT/test-data
COPY experiments/compare.sh $HBMAX_ROOT/experiments
# Build apex
RUN cd $HBMAX_ROOT \
    && git clone https://github.com/hipdac-lab/hbmax-pact \
    && cd $HBMAX_ROOT/hbmax-pact \
    && conan create conan/waf-generator user/stable \
    && conan create conan/trng user/stable 

RUN cd $HBMAX_ROOT/hbmax-pact \
    && conan install . \
    && ./waf configure build_release 
RUN chmod +x $HBMAX_ROOT/experiments/compare.sh
CMD ["/bin/bash"]

