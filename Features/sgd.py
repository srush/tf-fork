#!/usr/bin/env python

'''crf sgd trainer.'''


from __future__ import division

from itertools import *
from math import *
import trainer, sys
import time
from  evaluator import IDecoder, Evaluator
import gflags as flags
from svector import Vector
logs = sys.stderr
FLAGS = flags.FLAGS

DEBUG=True
SMALL = 1e-20

flags.DEFINE_integer("corpus_size", 2000, "number of sentences in the corpus")
flags.DEFINE_integer("batch_size", 50, "number of sentences per update")

sigma = 1.0
eta_0 = 0.1
alpha = 0.9
# C
regularization_constant = 0.5

#USE_L1 = True
USE_L1 = False
USE_L2 = False
assert not (USE_L1 and USE_L2) 
# SGD with L1 penalty - http://www.aclweb.org/anthology/P/P09/P09-1054.pdf
class BaseCRF(trainer.BaseTrainer):

    def __init__(self, decoder, train, dev, output, iter = 1):
        trainer.BaseTrainer.__init__(self, decoder, [train], output=output)
        self.iter = iter
        self.eval = Evaluator(decoder, [dev])
        self.u = 0
        #self.N = 10
        self.weight_norm = 0.0
        
    def set_oracle_files(self, oraclefiles):
      self.oraclefiles = oraclefiles

    def set_configuration(self, corpus_size, batch_size):
      self.N = corpus_size
      self.B = batch_size

    def get_eta(self, k):
      # exponential eta rate
      return eta_0 * (alpha ** (k/ self.N))

    @staticmethod
    def clip_weights(fv, range):
      """keep weights within a fixed update range"""
      for f in fv:
        fv[f] = min(fv[f],range) if fv[f] >0 else max(fv[f],-range)
      return fv


    def do_update(self, full_update, k):
      
      eta = self.get_eta(k)
      print >>logs, "Doing batch update %s, Eta %s" % (k, eta)
      if DEBUG:
        all_feat = []
        for feat in full_update:
          all_feat.append((abs(full_update[feat]), full_update[feat], feat))
        all_feat.sort(reverse=True)
        self.dump(dict([(f,v2) for (v,v2,f) in all_feat[0:10]]))

      #update = self.clip_weights(full_update, 1.0)
      update = full_update
      self.weights += eta * update
      if USE_L1:
        self.u += (regularization_constant/self.N) * eta
        for feat in update:
          prev = self.weights[feat]
          if self.weights[feat] > 0 :
            self.weights[feat] = max(0, self.weights[feat] - (self.u + self.q[feat]))
          else:
            self.weights[feat] = min(0, self.weights[feat] + (self.u - self.q[feat]))
          
          self.q[feat] += self.weights[feat] - prev
          

          # keep \sum |w| up to date
          assert not isnan(update[feat]), feat
          assert not isnan(self.weights[feat]), feat
          assert not isnan(prev)
          
          self.weight_norm -= abs(prev - eta*update[feat]) 
          self.weight_norm += abs(self.weights[feat])
          
          if self.weights[feat] == 0: del self.weights[feat]
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

    def compute_objective(self, log_likelihood):
      # loglikelihood = log p(y* | x ; w)
      objective = log_likelihood
      regular = 0.0
      if USE_L1 or USE_L2:
        regular = (regularization_constant / self.N) * self.weight_norm
        print >>logs, "Regularization = %s" % regular
        objective -= regular
      print >>logs, "Log-likelihood = %s" % log_likelihood

      print >>logs, "Estimate of objective = %s" % objective
      return objective, regular
    
    def one_pass_on_train(self):
      num_updates = 0
      train_fore = self.decoder.load(self.trainfiles)
      oracle_fore = self.decoder.load_oracle(self.oraclefiles)
      round_obj = []
      round_regular = []

      def collect_batch():
        update = Vector()
        cum_log_likelihood = 0.0
        start = time.time()
        for i, (example, oracle) in enumerate(izip(train_fore, oracle_fore), 1):
          
          self.c += 1

          marginals, oracle_marginals, oracle_log_prob = self.decoder.compute_marginals(example, oracle)
          cum_log_likelihood += oracle_log_prob
          
          update += (oracle_marginals - marginals)

          if i % self.B == 0:
            end = time.time()
            yield update, cum_log_likelihood
            update = Vector()
            cum_log_likelihood =0.0

            
          if i % 10 == 0:
            end = time.time() if i % self.B <> 0 else end
            print >> logs, "... example %d (len %d)..." % (i, len(example)),
            print >> logs, "oracle forest %s"%((end - start))
            #print >> logs, "Oracle log prob %s"%oracle_log_prob
            start = time.time()
      
      for self.k, (batch, log_likelihood) in enumerate(collect_batch(), self.k):
        obj, regular = self.compute_objective(log_likelihood)
        round_regular.append(regular)
        round_obj.append(obj)
        self.do_update(batch, self.k)
        #if i == 1: break
        
      return num_updates, round_regular, round_obj

    def train(self):
        # q is the penalty assessed so far
        self.q = Vector()
        self.u = 0.0
        self.k = 0

        starttime = time.time()

        print >> logs, "starting perceptron at", time.ctime()

        best_prec = 0
        all_round = []
        for it in xrange(1, self.iter+1):

            print >> logs, "iteration %d starts..............%s" % (it, time.ctime())

            iterstarttime = time.time()
            num_updates, round_regular, round_obj = self.one_pass_on_train()
            all_round.append((round_regular, round_obj))
            print >> logs, "iteration %d training finished at %s. now evaluating on dev..." % (it, time.ctime())
            
            
            self.decoder.set_model_weights(self.weights)
            prec = self.eval.eval().compute_score()
            print "Final %s"%prec
            print "Num feat %s"%len(self.weights)
            print "Round Regular %s"%sum(round_regular)
            print "Round Obj %s"%sum(round_obj)
            logs.flush()



    @staticmethod
    def cmdline_crf(decoder):
        crf = BaseCRF(decoder, train = FLAGS.train, dev = FLAGS.dev,
                          output = FLAGS.out, iter = FLAGS.iter)
        crf.set_configuration(FLAGS.corpus_size, FLAGS.batch_size)
        
        return crf
