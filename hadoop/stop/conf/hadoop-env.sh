export JAVA_HOME=/usr/lib/jvm/jre-1.6.0-openjdk.x86_64/
export HADOOP_LOG_DIR=/auto/nlg-02/ar_009/Projects/transforest/hadoop/stop/logs
export HADOOP_PID_DIR=/tmp/
export PATH=/home/nlg-02/ar_009/lib/bin:/usr/usc/globus/4.0.1/bin:/usr/usc/globus/4.0.1/sbin:/usr/usc/matlab/2009b/bin:/home/nlg-03/riesa/tools/mpich2/bin:/home/nlg-03/riesa/tools/python26/bin:/home/rcf-78/riesa/bin:/home/rcf-78/riesa/ar/code/:/home/rcf-78/riesa/tools/ArabicSVMTools/Package/bin:/home/rcf-78/riesa/tools/bin:/bin:/usr/bin:/usr/X11R6/bin:/usr/java/jdk1.5.0_13/bin:/home/rcf-40/ar_009/.cabal/bin/:/home/rcf-40/ar_009/.xmonad/:/var/lib/gems/1.8/bin/:/home/rcf-40/ar_009/Lib/ampl/bin/:/home/rcf-40/ar_009/Lib/graclus1.2/:/afs/csail.mit.edu/u/s/srush/Lib/gurobi301/linux64//bin:/usr/usc/gnu/gcc/default/bin:/home/nlg-03/wang11/sbmt-bin/v3.0/decoder/new_decoder/mini_v11.5/x86_64/bin:/home/nlg-01/contrib/local/jdk1.6.0/bin:/home/rcf-78/sdeneefe/bin:/home/nlg-03/mt-apps/grid/0.2/bin:/home/nlg-05/voeckler/perl/x86_64/bin:/home/rcf-12/chiangd/hadoop-tutorial:/home/nlg-03/riesa/subtrees/code:/home/nlg-03/riesa/tools/bin:/home/nlg-03/riesa/tools/vim72/build/bin
# Set Hadoop-specific environment variables here.

# The only required environment variable is JAVA_HOME.  All others are
# optional.  When running a distributed configuration it is best to
# set JAVA_HOME in this file, so that it is correctly defined on
# remote nodes.

# The java implementation to use.  Required.
export JAVA_HOME=/usr/usc/jdk/1.6.0

# Extra Java CLASSPATH elements.  Optional.
# export HADOOP_CLASSPATH=

# The maximum amount of heap to use, in MB. Default is 1000.
# export HADOOP_HEAPSIZE=2000

# Extra Java runtime options.  Empty by default.
# export HADOOP_OPTS=-server

# Command specific options appended to HADOOP_OPTS when specified
export HADOOP_NAMENODE_OPTS="-Dcom.sun.management.jmxremote $HADOOP_NAMENODE_OPTS"
export HADOOP_SECONDARYNAMENODE_OPTS="-Dcom.sun.management.jmxremote $HADOOP_SECONDARYNAMENODE_OPTS"
export HADOOP_DATANODE_OPTS="-Dcom.sun.management.jmxremote $HADOOP_DATANODE_OPTS"
export HADOOP_BALANCER_OPTS="-Dcom.sun.management.jmxremote $HADOOP_BALANCER_OPTS"
export HADOOP_JOBTRACKER_OPTS="-Dcom.sun.management.jmxremote $HADOOP_JOBTRACKER_OPTS"
# export HADOOP_TASKTRACKER_OPTS=
# The following applies to multiple commands (fs, dfs, fsck, distcp etc)
# export HADOOP_CLIENT_OPTS

# Extra ssh options.  Empty by default.
# export HADOOP_SSH_OPTS="-o ConnectTimeout=1 -o SendEnv=HADOOP_CONF_DIR"

# Where log files are stored.  $HADOOP_HOME/logs by default.
# export HADOOP_LOG_DIR=${HADOOP_HOME}/logs

# File naming remote slave hosts.  $HADOOP_HOME/conf/slaves by default.
# export HADOOP_SLAVES=${HADOOP_HOME}/conf/slaves

# host:path where hadoop code should be rsync'd from.  Unset by default.
# export HADOOP_MASTER=master:/home/$USER/src/hadoop

# Seconds to sleep between slave commands.  Unset by default.  This
# can be useful in large clusters, where, e.g., slave rsyncs can
# otherwise arrive faster than the master can service them.
# export HADOOP_SLAVE_SLEEP=0.1

# The directory where pid files are stored. /tmp by default.
# export HADOOP_PID_DIR=/var/hadoop/pids

# A string representing this instance of hadoop. $USER by default.
# export HADOOP_IDENT_STRING=$USER

# The scheduling priority for daemon processes.  See 'man nice'.
# export HADOOP_NICENESS=10
