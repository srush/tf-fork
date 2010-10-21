#!/bin/bash

# Assumes that a Hadoop cluster exists and HADOOP_CONF_DIR is set
# This doesn't necessarily have to be run on a node inside the cluster

export HADOOP_HOME=/home/nlg-01/chiangd/pkg/hadoop
DISCHADOOP=/home/nlg-02/ar_009/disc-hadoop
export PATH=$HADOOP_HOME/bin:$PATH
export PYTHON=/home/nlg-03/riesa/tools/python26/bin/python
TRANS=/home/rcf-40/ar_009/Projects/transforest/
TRANSEXAMPLE=/home/rcf-40/ar_009/Projects/transforest/example
TRANSSCRIPT=/home/rcf-40/ar_009/Projects/transforest/scripts
TRANSTRAIN=/home/rcf-40/ar_009/Projects/transforest/Features
TRANSHADOOP=/home/rcf-40/ar_009/Projects/transforest/hadoop/
TRANSDATA=/home/nlg-02/ar_009/
TRANSDATAIN=/home/nlg-02/ar_009/transdata/core_data/
TRANSDATAOUT=/home/nlg-02/ar_009/transdata/processed_data/

export LD_LIBRARY_PATH=/lib:/usr/lib:/home/rcf-78/liangh/local/lib:/home/rcf-40/ar_009/lib:/usr/usc/gnu/gcc/3.4.0/lib:/usr/usc/gnu/gcc/3.3.3/lib:/home/rcf-12/graehl/isd/linux/lib:/auto/hpc-22/dmarcu/nlg/contrib/local/pcre/lib:/auto/hpc-22/dmarcu/nlg/contrib/local/db-4.2/lib:/home/rcf-12/graehl/dev/boost/stage/lib:/auto/hpc-22/dmarcu/nlg/dragos/lib/qt-3.1/lib:/auto/hpc-22/dmarcu/nlg/dragos/lib:/home/rcf-12/liangh/liangh/graehl/lib:/afs/csail.mit.edu/u/s/srush/Lib/gurobi301/linux64//lib

export PYTHONPATH=$PYTHONPATH:$TRANS:$TRANSTRAIN
export PYTHONPATH=/home/nlg-03/riesa/tools/python26/bin:/home/nlg-03/riesa/tools/pysrc:/home/nlg-03/riesa/tools/lib:/home/nlg-01/chiangd/svector/lib:/home/rcf-40/ar_009/.python/lib:/home/rcf-40/ar_009/.python/lib/python2.6/site-packages:$PYTHONPATH


HADOOPSTREAM="hadoop jar $HADOOP_HOME/contrib/streaming/hadoop-streaming.jar -cmdenv PYTHONPATH=$TRANS:$TRANSTRAIN -cmdenv LD_LIBRARY_PATH=/lib:/usr/lib:/home/rcf-78/liangh/local/lib:/home/rcf-40/ar_009/lib:/usr/usc/gnu/gcc/3.4.0/lib:/usr/usc/gnu/gcc/3.3.3/lib:/home/rcf-12/graehl/isd/linux/lib:/auto/hpc-22/dmarcu/nlg/contrib/local/pcre/lib:/auto/hpc-22/dmarcu/nlg/contrib/local/db-4.2/lib:/home/rcf-12/graehl/dev/boost/stage/lib:/auto/hpc-22/dmarcu/nlg/dragos/lib/qt-3.1/lib:/auto/hpc-22/dmarcu/nlg/dragos/lib:/home/rcf-12/liangh/liangh/graehl/lib:/afs/csail.mit.edu/u/s/srush/Lib/gurobi301/linux64//lib -cmdenv PYTHON=/home/nlg-03/riesa/tools/python26/bin/python"


WORKDIR=.
hadoop fs -mkdir $WORKDIR
TMPDIR=${TMPDIR-/tmp}

NODES=`wc -l < $HADOOP_CONF_DIR/slaves`
MAPS_PER_NODE=10

OUTDIR=$1

CHINESE=$TRANSDATAIN/core.f.parse_10
ENGLISH=$TRANSDATAIN/core.e_10
RULES=$TRANSDATAIN/minRules.txt.2000.count2.rhs50


$PYTHON $TRANSHADOOP/zip.py $CHINESE $ENGLISH > $TMPDIR/combined.txt

hadoop fs -put $TMPDIR/combined.txt $WORKDIR/combined.txt
hadoop fs -put $RULES $WORKDIR/rules.txt

$HADOOPSTREAM \
  -input $WORKDIR/combined.txt \
  -output $WORKDIR/tforest \
  -mapper "$PYTHON $TRANSSCRIPT/process_parses.py rules" \
  -cacheFile "$WORKDIR/rules.txt#rules" \
  -jobconf mapred.map.tasks=500 \
  -jobconf mapred.reduce.tasks=0 \
  -jobconf mapred.output.compress=true \
  -jobconf mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec

echo "GETTING"

rm -fr $PBS_O_WORKDIR/tforest
hadoop fs -get $WORKDIR/tforest $PBS_O_WORKDIR/tforest
