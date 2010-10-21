#!/usr/bin/env python

'''crf sgd trainer.'''


from __future__ import division

from itertools import *
from math import *
import trainer, sys, os, traceback
import time
from  evaluator import IDecoder, Evaluator
import gflags as flags
from svector import Vector
import lbfgs
from random import random
logs = sys.stderr
FLAGS = flags.FLAGS

DEBUG=True
SMALL = 1e-20

sigma = 1.0
eta_0 = 0.1
alpha = 0.9
# C
regularization_constant = 0.50

#USE_L1 = True
USE_L1 = True
USE_L2 = False
assert not (USE_L1 and USE_L2)

flags.DEFINE_integer("train_size", 2000000, "train size")

class LBFGSCRF(trainer.BaseTrainer):

    def __init__(self, decoder, train, dev, output, iter = 1):
        trainer.BaseTrainer.__init__(self, decoder, [train], output=output)
        self.iter = iter
        self.eval = Evaluator(decoder, [dev])
        self.just_basic = False
        
    def set_oracle_files(self, oraclefiles):
      self.oraclefiles = oraclefiles

    def rm_features(self, remove_set):
      self.remove_set = remove_set

    def enforce_just_basic(self):
        self.just_basic = True

    def do_update(self, full_update, k):      
      print >>logs, "Doing update %s, Eta %s" % (k, eta)
      
      if DEBUG:
        all_feat = []
        for feat in full_update:
          all_feat.append((abs(full_update[feat]), full_update[feat], feat))
        all_feat.sort(reverse=True)
        self.dump(dict([(f,v2) for (v,v2,f) in all_feat[0:10]]))

      update = full_update
      self.weights += update

      if USE_L2:
        for feat in update:
          prev = self.weights[feat]
          former = (prev - eta*update[feat]) ** 2 
          self.u -= (regularization_constant/self.N) * (1/(sigma*sigma)) * eta * self.weights[feat]
          self.weight_norm -= former 
          self.weight_norm += (self.weights[feat]) ** 2
          
        #self.dump(self.weights)
      print >>logs, "Weight Norm %s" % (self.weight_norm)
      #self.dump(self.weights)
      self.decoder.set_model_weights(self.weights)


    def one_pass_on_train(self, weights):
      self.weights = weights
      for feat in weights:
          if feat.startswith("Basic"):
              print feat, weights[feat]
      self.decoder.set_model_weights(self.weights)
      self.round +=1 
      weight_file = open("tmp/weights.round."+str(self.round)+"."+str(self.name), 'w')
      for feat in self.weights:
        if abs(self.weights[feat]) > 1e-10:
          print >>weight_file, feat + "\t" + str(self.weights[feat])
      weight_file.close()
      #show train score
      #if self.round <> 1:
          #self.eval.tune()    
      prec = self.eval.eval()
      print "-----------------------"
      print "Final %s"%prec.compute_score()
      print "Num feat %s"%len(self.weights)
      print "-----------------------"

      try:
          num_updates = 0
          train_fore = self.decoder.load(self.trainfiles)
          oracle_fore = self.decoder.load_oracle(self.oraclefiles)

          update = Vector()
          cum_log_likelihood = 0.0
          start = time.time()
          for i, (example, oracle) in enumerate(izip(train_fore, oracle_fore), 1):
              if i == FLAGS.train_size:
                  break
              self.c += 1
              marginals, oracle_marginals, oracle_log_prob = self.decoder.compute_marginals(example, oracle)
              cum_log_likelihood += oracle_log_prob

              #little_update = (oracle_marginals - marginals)

              #print "Text Length", oracle_marginals["Basic/text-length"], marginals["Basic/text-length"], little_update["Basic/text-length"]
              #print "GT Prob", little_update["Basic/gt_prob"]

              update += oracle_marginals
              update -= marginals


              if i % 100 == 0:
                  end = time.time() 
                  print >> logs, "... example %d (len %d)...%s sec" % (i, len(example), (end-start)),
                   #print >> logs, "Oracle log prob %s"%oracle_log_prob
                  start = time.time()
      except:
          print "Unexpected error:", sys.exc_info()[0], traceback.print_exc(file=sys.stdout)  
          os._exit(0)
      #for i in range(len(self.ind2feature)):
      #  print i, update[self.ind2feature[i]], "\"" +self.ind2feature[i]+"\""

      #for f in update:
      #  print update[f], "\"" +f+"\""
      print "\n\n\n"
      print cum_log_likelihood
      print
      print
      
      print "Text Length", update["Basic/text-length"]
      print "Text length weight", self.weights["Basic/text-length"]

      return -update, -cum_log_likelihood

    def set_feature_mappers(self, (feature2ind, ind2feature)):      
      self.feature2ind = feature2ind
      self.ind2feature = ind2feature
      
    def train(self):
        # q is the penalty assessed so far
        self.name = random()
        starttime = time.time()
        self.round = 0
        print >> logs, "starting lbfgs at", time.ctime()

        best_prec = 0
        all_round = []
        #for it in xrange(1, self.iter+1):

        print >> logs, "iteration %d starts..............%s" % (1, time.ctime())
        
        iterstarttime = time.time()
        
        def wrap(w):
          print "Weights", w.norm()
          print >>sys.stderr, self
          (marginals, log_likelihood) = self.one_pass_on_train(w)

          for feat in marginals:
            for rm_feat in self.remove_set:
              if feat.startswith(rm_feat):
                marginals[feat] = 0.0
                del marginals[feat]
                
          if self.just_basic:
            for feat in marginals:
              if not feat.startswith("Basic"):
                marginals[feat] = 0.0
              else:
                print feat, marginals[feat]
          print "Log Likelihood", log_likelihood
          print "Marginals", marginals.norm()
          return (marginals, log_likelihood)
        print len(self.feature2ind), len(self.ind2feature)
            
        lbfgs.run(wrap, None, self.feature2ind, self.ind2feature, USE_L1, regularization_constant, self.weights)
        
        print >> logs, "iteration %d training finished at %s. now evaluating on dev..." % (1, time.ctime())
            
            

        
        logs.flush()



    @staticmethod
    def cmdline_crf(decoder):
        crf = LBFGSCRF(decoder, train = FLAGS.train, dev = FLAGS.dev,
                          output = FLAGS.out, iter = FLAGS.iter)
        #crf.set_configuration(FLAGS.corpus_size, FLAGS.batch_size)
        
        return crf
