from svector import Vector
from utility import *
from crf import LBFGSCRF
import sys, os
import gflags as flags
from itertools import *

code_dir = os.getenv("TRANSFOREST")
sys.path.append(code_dir+'/scripts/')
from shell_helpers import *

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_maps", 8, "number of mappers")
flags.DEFINE_integer("num_reds", 5, "number of reducers")
flags.DEFINE_integer("min_split", 100, "smallest mr split")

#env = env.get


class DistributedCRF(LBFGSCRF):

  def set_combined_training(self, training):
    self.trainfiles = training

  def write_weights(self, weights, weight_file):
    output = getfile(weight_file,1)
    for feat in weights:
      if abs(weights[feat]) > 1e-10:
        print >>output, "%s\t%s"%(feat, weights[feat])
    output.close()

  def read_weights(self, weight_file):
    weights = Vector()
    print >>sys.stderr, weight_file
    input = getfile(weight_file,0)
    for l in input:
      feat, weight = l.strip().split("\t")
      weights[feat] = float(weight)
    return weights

#   def start_hadoop(self, weight_file):
#     command = string.Template(
#     """
#     ${HADOOPSTREAM} \
#                   -input ${WORKDIR}/files.txt \
#                   -output ${WORKDIR}/params${i}.txt \
#                   -mapper "${PYTHON} $TRANSTRAIN/train_manager.py -w newparams --prefix=file_archive/  --lm ${TRANSEXAMPLE}/lm.3.sri --order 3  -b 100 --dist train" \
#                   -reducer "${PYTHON} $TRANSTRAIN/distributed_trainer.py reduce" \
#                   -cacheArchive ${WORKDIR}/files.jar#file_archive \
#                   -cacheFile "${WORKDIR}/read_full_params${LAST}\#newparams" \
#                   -jobconf mapred.map.tasks=50 \
#                   -jobconf mapred.reduce.tasks=20 
#     """
#     )
#     command.fill({i: i,
#                   LAST: i-1,
#                   WORKDIR:,
#                   TRANSEXAMPLE:
#                   TRANSTRAIN
#                   HADOOPSTREAMING
#                   PYTHON
#                   })
#     os.system(command)

#     rm $TMPDIR/params$i.txt
#     hadoop fs -getmerge $WORKDIR/params$i.txt $TMPDIR/params$i.txt


#     $PYTHON $TRANSHADOOP/untab.py < $TMPDIR/params$i.txt > $TMPDIR/read_full_params

#     hadoop fs -put $TMPDIR/read_full_params $WORKDIR/read_full_params$i

#   def start_local(self, weight_file, marginal_file):
#     "run a round of crf to get marginals"
#     command = string.Template(
#     """
#     cat ${WORKDIR}/files.txt | ${PYTHON} $TRANSTRAIN/train_manager.py -w ${weight_file} --prefix file_archive--lm ${TRANSEXAMPLE}/lm.3.sri --order 3  -b 100 --dist crf > ${marginal_file}
#     """
#     os.system()


  def start_hadoop(self, weight_file):
    shell = ShellRunner({})
    shell.subs["TRANSFOREST"] = os.getenv("TRANSFOREST")
    shell.subs["TRAINING"] = "file://" + self.trainfiles[0] #"train_input" #
    shell.subs["ROUND"] = self.round
    shell.subs["WEIGHTS"] = weight_file
    self.marginal_file = shell.subs["MARGINALS"] = "/tmp/marg"
    
    shell.call("~/.python/bin/dumbo rm /tmp/marg$ROUND -hadoop /home/nlg-03/mt-apps/hadoop/0.20.1+169.89/")
    shell.call("cd $TRANSFOREST/Features/;~/.python/bin/dumbo start $TRANSFOREST/Features/compute_marginals.py -inputformat text -input $TRAINING -output /tmp/marg$ROUND -weights $WEIGHTS -memlimit 2500m -jobconf mapred.map.tasks=%s -jobconf mapred.map.tasks.maximum=4 -jobconf mapred.min.split.size=%s -jobconf mapred.reduce.tasks=%s -hadoop /home/nlg-03/mt-apps/hadoop/0.20.1+169.89/"%(FLAGS.num_maps, FLAGS.min_split, FLAGS.num_reds))
    shell.call("~/.python/bin/dumbo rm /tmp/marg$ROUND/_logs -hadoop /home/nlg-03/mt-apps/hadoop/0.20.1+169.89/")
    shell.call("~/.python/bin/dumbo cat /tmp/marg$ROUND -hadoop /home/nlg-03/mt-apps/hadoop/0.20.1+169.89/ > $MARGINALS")

  def compute_marginals(self):
    self.decoder.set_model_weights(self.weights)

    train_fore = self.decoder.load(self.trainfiles)
    oracle_fore = self.decoder.load_oracle(self.oraclefiles)
    
    update = Vector()
    cum_log_likelihood = 0.0
    for i, (example, oracle) in enumerate(izip(train_fore, oracle_fore), 1):
      marginals, oracle_marginals, oracle_log_prob = self.decoder.compute_marginals(example, oracle)
      cum_log_likelihood += oracle_log_prob      
      update += (oracle_marginals - marginals)
    update["log_likelihood"] = -cum_log_likelihood 
    self.write_weights("-", -update)

    
  def one_pass_on_train(self, weights):
    
    
    # 1) write out weights to a file

    self.round += 1
    self.decoder.set_model_weights(weights)    
    #if self.round <> 1:
    #  self.eval.tune()    
    prec = self.eval.eval()
    print "-----------------------"
    print "Final %s"%prec.compute_score()
    print "Num feat %s"%len(self.weights)
    print "-----------------------"

    weight_filename = os.getenv("TRANSFOREST") + "/tmp/weights.round."+str(self.round)+"."+str(self.name)

    weight_file = open(weight_filename, 'w')
    self.write_weights(weights, weight_file)
    print weight_filename
    # 2) Start hadoop/local job to process weights
    #if self.use_hadoop:
    print >>sys.stderr, "calling hadoop"
    self.start_hadoop(weight_filename)
    #else:
    #  self.start_local(self.weight_file)

    # 3) Read in hadoop marginals to memory
    marginals = self.read_weights(self.marginal_file)

    # 4) Read in log-likelihood to memory
    # HACK, store likelihood in marginals
    log_likelihood = marginals["log_likelihood"]
    del marginals["log_likelihood"]
    print >>sys.stderr, log_likelihood
    # return
    return marginals, log_likelihood

  @staticmethod
  def cmdline_distibuted_crf(decoder):
    crf = DistributedCRF(decoder, train = FLAGS.train, dev = FLAGS.dev,
                   output = FLAGS.out, iter = FLAGS.iter)            
    return crf
    

    
