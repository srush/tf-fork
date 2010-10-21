#!/bin/bash
#PBS -l nodes=20

export HADOOP_HOME=/home/nlg-01/chiangd/pkg/hadoop
DISCHADOOP=/home/nlg-02/ar_009/disc-hadoop
export PATH=$DISCHADOOP:$HADOOP_HOME/bin:$PATH
PYTHON=/home/nlg-03/riesa/tools/python26/bin/python
TRANSHADOOP=/home/rcf-40/ar_009/Projects/transforest/hadoop/
NAME=v55.noselect

$PYTHON -V

# The directory where the Hadoop cluster will be created
CLUSTERDIR=$PBS_O_WORKDIR/cluster

# Create the cluster
$PYTHON /home/nlg-01/chiangd/hadoop/pbs_hadoop.py $CLUSTERDIR || exit 1
export HADOOP_CONF_DIR=$CLUSTERDIR/conf
hadoop fs -mkdir .

$TRANSHADOOP/train.sh || exit 1
#extract.zh-en.sh || exit 1
