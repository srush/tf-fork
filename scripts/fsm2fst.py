import sys

states = set()
for l in sys.stdin:
  tmp = l.strip().split()
  states.add(int(tmp[0]))
  if len(tmp) >=3:
    tmp.insert(2, tmp[2])
  if len >= 5:
    tmp[-1] = float(tmp[-1]) * (1 / (10000.5 * 2.30258509)) * 0.141221 
    tmp[-1] = str(tmp[-1])
  print " ".join(tmp)

last = max(states)


for s in states:
  last +=1
  print " ".join([str(s),str(last),"*SRC*", "*SRC*", "0.0"])
  print " ".join([str(last),str(s),"*RHO*", "*RHO*", "0.0"])
  last +=1
  print " ".join([str(s),str(last),"*PRE*", "*PRE*", "0.0"])
  print " ".join([str(last),str(s),"*RHO*", "*RHO*", "0.0"])
