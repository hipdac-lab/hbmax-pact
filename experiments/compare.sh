#!/bin/bash
datadir=$HBMAX_ROOT/hbmax-pact/test-data
bindir=$HBMAX_ROOT/hbmax-pact/build/release/tools
task=$1
ncpus=$2
echo "======== begin "$task"-hbmax with "$ncpus" threads ============="
export  OMP_NUM_THREADS=$ncpus
$bindir/imm -i $datadir/$task.txt -p -k 100 -d IC -e 0.2 -q 6  >> new_$task_$ncpus.txt
$bindir/oimm -i $datadir/$task.txt -p -k 100 -d IC -e 0.2 -q 1  >> old_$task_$ncpus.txt
grep 'IMM Parallel :' new_$task_$ncpus.txt | awk '{print "hbmax   using:" $8}'
grep 'IMM Parallel :' old_$task_$ncpus.txt | awk '{print "ripples using:" $8}'
echo "======== finish "$task"-hbmax ================"
