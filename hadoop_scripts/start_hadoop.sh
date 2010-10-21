export HADOOP_HOME=/home/nlg-03/mt-apps/hadoop/0.20.1+169.89/
DISCHADOOP=/home/nlg-02/ar_009/disc-hadoop
export PATH=$DISCHADOOP:$HADOOP_HOME/bin:$PATH
PYTHON=/home/nlg-03/riesa/tools/python26/bin/python
TRANSHADOOP=/home/rcf-40/ar_009/Projects/transforest/hadoop/


export LD_LIBRARY_PATH=/lib:/usr/lib:/home/rcf-78/liangh/local/lib:/home/rcf-40/ar_009/lib:/usr/usc/gnu/gcc/3.4.0/lib:/usr/usc/gnu/gcc/3.3.3/lib:/home/rcf-12/graehl/isd/linux/lib:/auto/hpc-22/dmarcu/nlg/contrib/local/pcre/lib:/auto/hpc-22/dmarcu/nlg/contrib/local/db-4.2/lib:/home/rcf-12/graehl/dev/boost/stage/lib:/auto/hpc-22/dmarcu/nlg/dragos/lib/qt-3.1/lib:/auto/hpc-22/dmarcu/nlg/dragos/lib:/home/rcf-12/liangh/liangh/graehl/lib:/afs/csail.mit.edu/u/s/srush/Lib/gurobi301/linux64//lib

# The directory where the Hadoop cluster will be created
CLUSTERDIR=$PWD/cluster

# Create the cluster
$PYTHON $PWD/pbs_hadoop.py $CLUSTERDIR || exit 1
export HADOOP_CONF_DIR=$CLUSTERDIR/conf
hadoop fs -mkdir .
