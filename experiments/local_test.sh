#!/bin/bash
sudo docker run --rm -v $(pwd)/test-data/:/opt/test-data hipdac/hbmax:0.1 /opt/experiments/compare.sh dblp 8