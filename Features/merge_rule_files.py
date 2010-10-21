import sys

lex_align, standard_rules = map(open,sys.argv[1:])


d = {}

for l2 in lex_align:
  features = ""
  lrule, lfeatures = l2.strip().split("###")
  
  for feat in lfeatures.split():
    if ":" in feat:
      features += " " + feat +" "
    d[lrule] = features
    
for l in standard_rules:
  rule, features = l.strip().split("###")
  if d.has_key(rule):
    features += d[rule]
  print rule + "###" + features
