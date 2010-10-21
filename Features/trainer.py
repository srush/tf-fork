#!/usr/bin/env python

'''generic averaged perceptron trainer.'''

from __future__ import division

import math
import sys
logs = sys.stderr

from collections import defaultdict

from svector import Vector

import time
from  evaluator import IDecoder, Evaluator

import gflags as flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("iter", 1, "number of passes over the whole training data", short_name="i")
flags.DEFINE_boolean("avg", True, "averaging parameters")
flags.DEFINE_string ("train", None, "training corpus")
flags.DEFINE_string ("dev", None, "dev corpus")
flags.DEFINE_string ("out", None, "output file (for weights)")
flags.DEFINE_string ("clip", 1.0, "clip updates (for weights)")

max_learning_rate = 1.0

class BaseTrainer(object):
    def __init__(self, decoder, train, output):
        self.trainfiles = train
        
        self.outfile = output
        self.decoder = decoder # a class, providing functions: load(), decode(), get_feats()

        self.updates = Vector()
        self.weights = decoder.get_model_weights() 
        self.allweights = Vector()
        self.c = 0. # counter: how many examples have i seen so far? = it * |train| + i

    def dump(self, weights):
        # calling model's write
        self.decoder.write_model(filename=self.outfile, weights=weights)


class BasePerceptron(BaseTrainer):
    def __init__(self, decoder, train, output):
        BaseTrainer.__init__(self, decoder, train, output)
        self.updates2 = Vector()
        self.num_updates = 0
        self.scales = Vector()

    def update_feature_scales(self):
        #print "Feature scales"
        for feat in self.updates2:
            self.scales[feat] = (self.updates2[feat] +1.0) / (self.num_updates + 1.0)
         #   print feat, self.scales[feat]

    def one_pass_on_train(self):
        num_updates = 0    
        early_updates = 0
        for i, example in enumerate(self.decoder.load(self.trainfiles), 1):
            print >> logs, "... example %d (len %d)..." % (i, len(example)),
            self.c += 1

            similarity, deltafeats = self.decoder.decode(example, early_stop=True)


            self.updates2 += deltafeats*deltafeats
            self.num_updates += 1
            self.update_feature_scales()
                
            for feat in deltafeats:
                if self.scales[feat]:
                    deltafeats[feat] = deltafeats[feat]/self.scales[feat]

            
            #if FLAGS.clip <> None:
            #    deltafeats = self.clip_weights(deltafeats, FLAGS.clip)

                

            if similarity.compute_score() < 1 - 1e-8: #update                

                num_updates += 1              

                print >> logs, "sim={0}, |delta|={1}".format(similarity, len(deltafeats))
                if FLAGS.debuglevel >=2:
                    print >> logs, "deltafv=", deltafeats            
                
                self.weights += (deltafeats )
                self.updates += deltafeats
                if FLAGS.avg:
                    self.allweights += (deltafeats) * self.c

                if similarity < 1e-8: # early-update happened
                    early_updates += 1
                
            else:
                print >> logs, "PASSED! :)"

        return num_updates, early_updates
    

    @staticmethod
    def clip_weights(fv, range):
        """keep weights within a fixed update range"""
        for f in fv:
            fv[f] = min(fv[f],range) if fv[f] >0 else max(fv[f],-range)
        return fv


class Perceptron(BasePerceptron):

    def __init__(self, decoder, train, dev, output, iter = 1 , avg = True):
        BasePerceptron.__init__(self, decoder, [train], output=output)
        self.iter = iter
        self.avg = avg
        self.eval = Evaluator(decoder, [dev])


    @staticmethod
    def cmdline_perc(decoder):
        return Perceptron(decoder, train = FLAGS.train, dev = FLAGS.dev,
                          output = FLAGS.out, iter = FLAGS.iter, avg = FLAGS.avg )
        
    def avg_weights(self):
        return self.weights - self.allweights * (1/self.c)
        
    def train(self):

        starttime = time.time()

        print >> logs, "starting perceptron at", time.ctime()

        best_prec = 0
        for it in xrange(1, self.iter+1):

            print >> logs, "iteration %d starts..............%s" % (it, time.ctime())

            iterstarttime = time.time()
            num_updates, early_updates = self.one_pass_on_train()

            print >> logs, "iteration %d training finished at %s. now evaluating on dev..." % (it, time.ctime())
            avgweights = self.avg_weights() if self.avg else self.weights
            if FLAGS.debuglevel >= 2:
                print >> logs, "avg w=", avgweights
            self.decoder.set_model_weights(avgweights)
            prec = self.eval.eval().compute_score()
            
            print >> logs, "at iteration {0}, updates= {1} (early {4}), dev= {2}, |w|= {3}, time= {5:.3f}h acctime= {6:.3f}h"\
                  .format(it, num_updates, prec, len(avgweights), early_updates, \
                          (time.time() - iterstarttime)/3600, (time.time() - starttime)/3600.)
            logs.flush()

            if prec > best_prec:
                best_prec = prec
                best_it = it
                best_wlen = len(avgweights)
                print >> logs, "new high at iteration {0}: {1}. Dumping Weights...".format(it, prec)
                self.dump(avgweights)

            self.decoder.set_model_weights(self.weights) # restore non-avg

        print >> logs, "peaked at iteration {0}: {1}, |bestw|= {2}.".format(best_it, best_prec, best_wlen)
        print >> logs, "perceptron training of %d iterations finished on %s (took %.2f hours)"  % \
              (it, time.ctime(), (time.time() - starttime)/3600.)


