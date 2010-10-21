#!/usr/bin/env python

from __future__ import division

import sys
import time
from collections import defaultdict

logs = sys.stderr

import gflags as flags
FLAGS=flags.FLAGS

from lmstate import LMState, TaroState
from bleu import Bleu
from svector import Vector

flags.DEFINE_boolean("newbeam", True, "new beaming")
flags.DEFINE_boolean("taro", True, "Taro-style state")

class Decoder(object):

    def __init__(self):
        self.firstpassscore = 0
        self.firstpassbleus = Bleu()
    
    def add_state(self, new):
        ''' adding a new state to the appropriate beam, and checking finality. '''

        beam = self.beams[new.step]

        if new.step > self.max_step:
            self.max_step = new.step

        self.num_edges += 1

        if not FLAGS.newbeam:
            old = beam.get(new, None)
            if old is None or new < old:
                if old is not None: # in-beam
                    new.merge_with(old)

                beam[new] = new

                if FLAGS.debuglevel >= 2:
                    print >> logs, "adding to beam %d: %s" % (new.step, new)
        else:
            # new beam: simply hold it here, uniq later
            beam.append(new)

    def beam_search(self, forest, b=1):

        if FLAGS.futurecost:
            forest.bestparse(LMState.weights + Vector("lm1=%s" % (LMState.weights["lm"] * FLAGS.lmratio))) 
            sc, tr, fv = forest.root.bestres
            forest.bleu.rescore(tr)
            print >> logs, "1-best score: %.3f, bleu: %s" % (sc, forest.bleu.score_ratio_str())
            self.firstpassscore += sc
            self.firstpassbleus += forest.bleu

        self.num_states = self.num_edges = 0
        self.num_stacks = 0
        self.final_items = []
        self.best = None
        
        beams = defaultdict(dict if not FLAGS.newbeam else list) # +inf
        self.beams = beams
        
        self.max_step = -1
        self.add_state(LMState.start_state(forest.root) if not FLAGS.taro else TaroState.start_state(forest.root)) # initial state

        self.nstates = 0  # space complexity
        self.nedges = 0 # time complexity

        i = 0
        while i <= self.max_step:

            if not FLAGS.newbeam:
                # N.B.: values, not keys! (keys may not be updated)
                curr_beam = sorted(beams[i].values())[:b]  # beam pruning, already uniq
            else:
                buf = sorted(beams[i])[:b]  # beam pruning, not uniq
                curr_beam = []
                uniq = {}
                uniq_stack = {}
                for item in buf:
                    if item not in uniq:
                        uniq[item] = item
                        curr_beam.append(item)
                        
                    uniq_stack[item.stack] = item
                    
                self.num_stacks += len(uniq_stack)
                
            self.num_states += len(curr_beam)
            
            if FLAGS.debuglevel >= 1:
                print >> logs, "beam %d, %d states" % (i, len(curr_beam))
                print >> logs, "\n".join([str(x) for x in curr_beam])
                print >> logs
                
            for old in curr_beam:
                if old.is_final():
                    self.final_items.append(old)        

                else:
                    for new in old.predict():
                        self.add_state(new)

                    if FLAGS.complete:
                        for new in old.complete():
                            self.add_state(new)

            i += 1

        self.final_items.sort()
        
        return self.final_items[0], self.final_items[:b]

### main ###

def main():
    
    weights = Model.cmdline_model()
    lm = Ngram.cmdline_ngram()
    
    LMState.init(lm, weights)

    decoder = Decoder()

    tot_bleu = Bleu()
    tot_score = 0.
    tot_time = 0.
    tot_len = tot_fnodes = tot_fedges = 0
    tot_steps = tot_states = tot_edges = tot_stacks = 0

    for i, forest in enumerate(Forest.load("-", is_tforest=True, lm=lm), 1):

        t = time.time()
        
        best, final_items = decoder.beam_search(forest, b=FLAGS.beam)
        score, trans, fv = best.score, best.trans(), best.get_fvector()

        t = time.time() - t
        tot_time += t

        tot_score += score
        forest.bleu.rescore(trans)
        tot_bleu += forest.bleu

        fnodes, fedges = forest.size()

        tot_len += len(forest.sent)
        tot_fnodes += fnodes
        tot_fedges += fedges
        tot_steps += decoder.max_step
        tot_states += decoder.num_states
        tot_edges += decoder.num_edges
        tot_stacks += decoder.num_stacks

        print >> logs, ("sent %d, b %d\tscore %.4f\tbleu+1 %s" + \
              "\ttime %.3f\tsentlen %-3d fnodes %-4d fedges %-5d\tstep %d  states %d  edges %d stacks %d") % \
              (i, FLAGS.beam, score, 
               forest.bleu.score_ratio_str(), t, len(forest.sent), fnodes, fedges,
               decoder.max_step, decoder.num_states, decoder.num_edges, decoder.num_stacks)

        if FLAGS.k > 1 or FLAGS.forest:
           lmforest = best.toforest(forest)

        if FLAGS.forest:
            lmforest.dump()

        if FLAGS.k > 1:
           lmforest.lazykbest(FLAGS.k, weights=weights)
           klist = lmforest.root.klist

           if not FLAGS.mert:
               for j, (sc, tr, fv) in enumerate(klist, 1):
                   print >> logs, "k=%d score=%.4f fv=%s\n%s" % (j, sc, fv, tr)

        else:
            klist = [(best.score, best.trans(), best.get_fvector())]
        
        if FLAGS.mert: # <score>... <hyp> ...
            print >> logs, '<sent No="%d">' % i
            print >> logs, "<Chinese>%s</Chinese>" % " ".join(forest.cased_sent)

            for sc, tr, fv in klist:
                print >> logs, "<score>%.3lf</score>" % sc
                print >> logs, "<hyp>%s</hyp>" % tr
                print >> logs, "<cost>%s</cost>" % fv

            print >> logs, "</sent>"

        if not FLAGS.forest:
            print trans

    print >> logs, "avg %d sentences, first pass score: %.4f, bleu: %s" % \
          (i, decoder.firstpassscore/i, decoder.firstpassbleus.score_ratio_str())
                                                                            
    print >> logs, ("avg %d sentences, b %d\tscore %.4lf\tbleu %s\ttime %.3f" + \
          "\tsentlen %.1f fnodes %.1f fedges %.1f\tstep %.1f states %.1f edges %.1f stacks %.1f") % \
          (i, FLAGS.beam, tot_score/i, tot_bleu.score_ratio_str(), tot_time/i,
           tot_len/i, tot_fnodes/i, tot_fedges/i,
           tot_steps/i, tot_states/i, tot_edges/i, tot_stacks/i)

    print >> logs, LMState.cachehits, LMState.cachemiss

if __name__ == "__main__":

    from ngram import Ngram
    from model import Model
    from forest import Forest

    flags.DEFINE_integer("beam", 1, "beam size", short_name="b")
    flags.DEFINE_integer("debuglevel", 0, "debug level")
    flags.DEFINE_boolean("mert", True, "output mert-friendly info (<hyp><cost>)")
    flags.DEFINE_boolean("profile", False, "profiling")
    flags.DEFINE_integer("kbest", 1, "kbest output", short_name="k")
    flags.DEFINE_boolean("forest", False, "dump +LM forest")
    flags.DEFINE_boolean("futurecost", True, "precompute future cost")

    argv = FLAGS(sys.argv)

    if FLAGS.profile:
        import cProfile as profile
        profile.run('main()', '/tmp/a')
        import pstats
        p = pstats.Stats('/tmp/a')
        p.sort_stats('time').print_stats(50)

    else:
        main()

