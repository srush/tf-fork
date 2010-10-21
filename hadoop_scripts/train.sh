#!/bin/bash

# Assumes that a Hadoop cluster exists and HADOOP_CONF_DIR is set
# This doesn't necessarily have to be run on a node inside the cluster

export HADOOP_HOME=/home/nlg-01/chiangd/pkg/hadoop
DISCHADOOP=/home/nlg-02/ar_009/disc-hadoop
export PATH=$HADOOP_HOME/bin:$PATH
PYTHON=/home/nlg-03/riesa/tools/python26/bin/python
TRANS=/home/rcf-40/ar_009/Projects/transforest/
TRANSEXAMPLE=/home/rcf-40/ar_009/Projects/transforest/example
TRANSTRAIN=/home/rcf-40/ar_009/Projects/transforest/Features
TRANSHADOOP=/home/rcf-40/ar_009/Projects/transforest/hadoop/
TRANSDATA=/home/nlg-02/ar_009/

export LD_LIBRARY_PATH=/lib:/usr/lib:/home/rcf-78/liangh/local/lib:/home/rcf-40/ar_009/lib:/usr/usc/gnu/gcc/3.4.0/lib:/usr/usc/gnu/gcc/3.3.3/lib:/home/rcf-12/graehl/isd/linux/lib:/auto/hpc-22/dmarcu/nlg/contrib/local/pcre/lib:/auto/hpc-22/dmarcu/nlg/contrib/local/db-4.2/lib:/home/rcf-12/graehl/dev/boost/stage/lib:/auto/hpc-22/dmarcu/nlg/dragos/lib/qt-3.1/lib:/auto/hpc-22/dmarcu/nlg/dragos/lib:/home/rcf-12/liangh/liangh/graehl/lib:/afs/csail.mit.edu/u/s/srush/Lib/gurobi301/linux64//lib

export PYTHONPATH=$PYTHONPATH:$TRANS:$TRANSTRAIN
export PYTHONPATH=/home/nlg-03/riesa/tools/python26/bin:/home/nlg-03/riesa/tools/pysrc:/home/nlg-03/riesa/tools/lib:/home/nlg-01/chiangd/svector/lib:/home/rcf-40/ar_009/.python/lib:/home/rcf-40/ar_009/.python/lib/python2.6/site-packages:$PYTHONPATH


HADOOPSTREAM="hadoop jar $HADOOP_HOME/contrib/streaming/hadoop-streaming.jar -cmdenv PYTHONPATH=$TRANS:$TRANSTRAIN -cmdenv LD_LIBRARY_PATH=/lib:/usr/lib:/home/rcf-78/liangh/local/lib:/home/rcf-40/ar_009/lib:/usr/usc/gnu/gcc/3.4.0/lib:/usr/usc/gnu/gcc/3.3.3/lib:/home/rcf-12/graehl/isd/linux/lib:/auto/hpc-22/dmarcu/nlg/contrib/local/pcre/lib:/auto/hpc-22/dmarcu/nlg/contrib/local/db-4.2/lib:/home/rcf-12/graehl/dev/boost/stage/lib:/auto/hpc-22/dmarcu/nlg/dragos/lib/qt-3.1/lib:/auto/hpc-22/dmarcu/nlg/dragos/lib:/home/rcf-12/liangh/liangh/graehl/lib:/afs/csail.mit.edu/u/s/srush/Lib/gurobi301/linux64//lib"


WORKDIR=.
hadoop fs -mkdir $WORKDIR
TMPDIR=${TMPDIR-/tmp}

NODES=`wc -l < $HADOOP_CONF_DIR/slaves`
MAPS_PER_NODE=10

OUTDIR=$1


$PYTHON $TRANSHADOOP/make_jar.py $TRANSDATA/core.f.parse_2000_newfeat.tforest
hadoop fs -put $TMPDIR/files/files.jar $WORKDIR/files.jar
hadoop fs -put files.txt $WORKDIR/files.txt

cp $TRANSEXAMPLE/config.blank $TMPDIR/full_params0
hadoop fs -put $TRANSEXAMPLE/config.blank $WORKDIR/read_full_params0

  
for i in {1..20}
do
LAST=$(($i-1)) 
$HADOOPSTREAM \
  -input $WORKDIR/files.txt \
  -output $WORKDIR/params$i.txt \
  -mapper "$PYTHON $TRANSTRAIN/train_manager.py -w newparams --prefix=file_archive/  --lm $TRANSEXAMPLE/lm.3.sri --order 3  -b 100 --dist train" \
  -reducer "$PYTHON $TRANSTRAIN/distributed_trainer.py reduce" \
  -cacheArchive $WORKDIR/files.jar#file_archive \
  -cacheFile "$WORKDIR/read_full_params$LAST#newparams" \
  -jobconf mapred.map.tasks=50 \
  -jobconf mapred.reduce.tasks=20 


rm $TMPDIR/params$i.txt
hadoop fs -getmerge $WORKDIR/params$i.txt $TMPDIR/params$i.txt

# params is just updates, so merge it with our current full params 
#$PYTHON $TRANSTRAIN/distributed_trainer.py file $TMPDIR/full_params$LAST $TMPDIR/params$i.txt > $TMPDIR/full_params$i

$PYTHON $TRANSHADOOP/untab.py < $TMPDIR/params$i.txt > $TMPDIR/read_full_params

hadoop fs -put $TMPDIR/read_full_params $WORKDIR/read_full_params$i


rm $PBS_O_WORKDIR/params$i.txt
hadoop fs -getmerge $WORKDIR/params$i.txt $PBS_O_WORKDIR/params$i.txt
rm $PBS_O_WORKDIR/read_full_params$i
hadoop fs -get  $WORKDIR/read_full_params$i $PBS_O_WORKDIR/read_full_params$i
done

