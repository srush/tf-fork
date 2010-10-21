import openfst, fsa, sys
sys.path.append('..')
from ngram import Ngram
import gflags as flags
FLAGS = flags.FLAGS


argv = FLAGS(sys.argv)
lm = Ngram.cmdline_ngram()
s_table = openfst.SymbolTable.ReadText(argv[1])

lm_fsa = openfst.Read(argv[2])

score = 0.0
#score += lm.word_prob_bystr("doubts", "<s>")
score += lm.word_prob_bystr("mostly", "<s> doubts")
score += lm.word_prob_bystr("</s>", "doubts mostly")
#score += lm.word_prob_bystr("</s>", "mostly")
print "Score is ", score * -0.141

words = "xinhua news agency , hong kong , february 23 electricity"
sc = lm.word_prob("xinhua news agency , hong kong , february 23 electricity")
print "Long Score is ", sc * 0.141


def make_fsa(sent):
  sent_fst = openfst.StdVectorFst()
  ssent = sent.split()
  sent_fst.AddState()
  for w in ssent:
    sent_fst.AddState()
  for i,w in enumerate(ssent):
    sym = s_table.Find(w)
    sent_fst.AddArc(i, openfst.StdArc(sym, sym, 0.0, i+1))
  sent_fst.SetStart(0)
  sent_fst.SetFinal(len(ssent), 0.0)
  return sent_fst

sent_fst1 = make_fsa(words)


t = openfst.ShortestDistance(sent_fst1, False)
print t
 
# sent_fst1 = openfst.StdVectorFst()
# sent_fst1.AddState()
# sent_fst1.AddState()
# sent_fst1.AddState()
# sent_fst1.AddState()
# sent_fst1.AddState()

# sent_fst1.SetStart(0)
# sent_fst1.SetFinal(4, 0.0)




# s1 = s_table.Find("doubts")
# s_new1 = s_table.Find("*SRC*")
# s_new2 = s_table.AddSymbol("BLAHBLAHBLAH")
# s2 = s_table.Find("mostly")
# sent_fst1.AddArc(0, openfst.StdArc(s1, s1, 0.0, 1))
# sent_fst1.AddArc(1, openfst.StdArc(s_new1, s_new1, 0.0, 2))
# sent_fst1.AddArc(2, openfst.StdArc(s_new2, s_new2, 0.0, 3))
# sent_fst1.AddArc(3, openfst.StdArc(s2, s2, 0.0, 4))




sent_fst1.SetInputSymbols(s_table)
sent_fst1.SetOutputSymbols(s_table)

lm_fsa.SetInputSymbols(s_table)
lm_fsa.SetOutputSymbols(s_table)


openfst.ArcSortInput(lm_fsa)

inter_fst = openfst.StdVectorFst()
short_fst = openfst.StdVectorFst()
inter_det_fst = openfst.StdVectorFst()


#openfst.Intersect(fst1, fst2, inter_fst)
inter_fst = fsa.rho_compose(sent_fst1, False, lm_fsa, True, True)
openfst.Determinize(inter_fst, inter_det_fst)
print inter_det_fst.NumStates()
fsa.print_fst(inter_det_fst)

openfst.ShortestPath(inter_det_fst, short_fst, 1)
print short_fst.NumStates()
openfst.TopSort(short_fst)
fsa.print_fst(short_fst)


#-----


fst1 = openfst.StdVectorFst()
fst2 = openfst.StdVectorFst()
fst1.AddState()
fst1.AddState()
fst1.AddState()
fst1.AddState()

fst2.AddState()
fst2.AddState()
fst2.AddState()

fst1.SetStart(0)
fst2.SetStart(0)

fst1.SetFinal(2, 0.0)
fst2.SetFinal(2, 0.0)

#fst1.AddArc(0, openfst.StdArc(1, 1, 10.0, 1))
fst1.AddArc(0, openfst.StdArc(0, 0, 10.0, 1))
fst1.AddArc(1, openfst.StdArc(0, 0, 10.0, 1))
fst1.AddArc(0, openfst.StdArc(fsa.RHO, fsa.RHO, 10.0, 0))
fst1.AddArc(1, openfst.StdArc(fsa.RHO, fsa.RHO, 10.0, 2))

#fst1.AddArc(2, openfst.StdArc(fsa.RHO, fsa.RHO, 10.0, 3))
#fst1.AddArc(0, openfst.StdArc(fsa.RHO, fsa.RHO, 10.0, 1))
#fst1.AddArc(1, openfst.StdArc(fsa.RHO, fsa.RHO, 10.0, 2))

fst2.AddArc(0, openfst.StdArc(1, 1, 10.0, 1))
fst2.AddArc(1, openfst.StdArc(2, 2, 10.0, 2))


inter_fst = openfst.StdVectorFst()
short_fst = openfst.StdVectorFst()
inter_det_fst = openfst.StdVectorFst()


#openfst.Intersect(fst1, fst2, inter_fst)
inter_fst = fsa.rho_compose(fst2, False, fst1, True, True)
openfst.Determinize(inter_fst, inter_det_fst)
print inter_det_fst.NumStates()
fsa.print_fst(inter_det_fst)

openfst.ShortestPath(inter_det_fst, short_fst, 1)
print short_fst.NumStates()
openfst.TopSort(short_fst)
fsa.print_fst(short_fst)

