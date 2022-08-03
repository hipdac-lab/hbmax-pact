# HBMax: Optimizing Memory Efficiency for Parallel Influence Maximization on Multicore Architectures

By [Xinyu Chen](xinyu.chen1@wsu.edu), [Marco Minutoli](marco.minutoli@pnnl.gov), [Jiannan Tian](jiannan.tian@wsu.edu), [Mahantesh Halappanavar](mahantesh.halappanavar@pnnl.gov), [Ananth Kalyanaraman](ananth@wsu.edu), [Dingwen Tao](dingwen.tao@wsu.edu).

HBMax is a memory-efficient optimization approach based on [Ripples](https://github.com/pnnl/ripples) for multi-threaded parallel influence maximization solution over large real-world social networks.

This code is to demonstrate HBMax on [DBLP co-authorship network data](https://snap.stanford.edu/data/com-DBLP.html) and compare with the original solution Ripples. 

*******

## Method 1: Use Docker Image (Recommended)

To ease the use of HBMAX, we provide a docker image with the essential environment.

### Step 1: Download docker image

Assuming [docker](https://docs.docker.com/get-docker/) has been installed, please run the following command to pull our prepared [docker image](https://hub.docker.com/r/xinyu2/hbmax) from DockerHub:
```
docker pull xinyu2/hbmax
```

### Step 2: Test HBMAX and compare with Ripples

```
sudo docker run --rm xinyu2/hbmax /opt/experiments/compare.sh dblp 8
```
The output should be shown like below. 

<img width="352" alt="example" src="https://user-images.githubusercontent.com/5705572/182407194-92d2c1c9-77c5-4bd3-8451-e954216a629a.png">

## Method 2: Build From Source
### System Environment
- OS: Linux Ubuntu (>= 18.04)
- Compiler: GCC (>=7.4.0) with OpenMP (>=4.5)

Note that according to [OpenMP website](https://www.openmp.org/resources/openmp-compilers-tools/), OpenMP 4.5 is fully supported for C and C++ since GCC 6. 

### Step 1: Install dependencies
1. Install Python3, CMake, Git, Wget
```
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
        build-essential cmake git wget python3.6 python3-dev \
        python3-pip python3-setuptools gcc
```

2. Install Conan
```
pip3 install --no-cache-dir --upgrade pip
pip3 install --no-cache-dir conan==1.51.0
export PATH=$PATH:~/.local/bin
```

### Step 2: Download HBMax code and example data
```
HBMAX_ROOT=$(pwd)
git clone https://github.com/hipdac-lab/hbmax-pact
cd $HBMAX_ROOT/hbmax-pact/test-data
wget https://eecs.wsu.edu/~dtao/data/dblp.txt
```

### Step 3: Build HBMax
```
cd $HBMAX_ROOT/hbmax-pact   
conan create conan/waf-generator user/stable
conan create conan/trng user/stable
conan install .
./waf configure build_release
```

### Step 4: Test HBMax and compare with Ripples
```
cd $HBMAX_ROOT/hbmax-pact/
export DATADIR=$HBMAX_ROOT/hbmax-pact/test-data
export EXECDIR=$HBMAX_ROOT/hbmax-pact/build/release/tools
export task=dblp
export ncpus=8
export OMP_NUM_THREADS=$ncpus
$EXECDIR/imm -i $DATADIR/$task.txt -p -k 100 -d IC -e 0.2 -q 6  >> new_${task}_${ncpus}.txt
$EXECDIR/oimm -i $DATADIR/$task.txt -p -k 100 -d IC -e 0.2 -q 1  >> old_${task}_${ncpus}.txt
echo "======== begin "$task"-hbmax with "$ncpus" threads =============" >> /tmp/result
grep 'IMM Parallel :' new_${task}_${ncpus}.txt | awk '{print "hbmax   using:" $8}' >> /tmp/result
grep 'IMM Parallel :' old_${task}_${ncpus}.txt | awk '{print "ripples using:" $8}' >> /tmp/result
echo "======== finish "$task"-hbmax ================" >> /tmp/result
cat /tmp/result
rm new_${task}_${ncpus}.txt old_${task}_${ncpus}.txt  /tmp/result
```

The output should be shown like below.

<img width="352" alt="example" src="https://user-images.githubusercontent.com/5705572/182407194-92d2c1c9-77c5-4bd3-8451-e954216a629a.png">
