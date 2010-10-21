UP = "UP"
DOWN = "DOWN"

def is_lex(word):
  return word[0]=="\""
def is_lex(word):
  return word[0]=="\""

def strip_lex(word):
  if word == "\"": return word
  elif word == "\"\\\"\"": return "\""
  return word.strip("\"")

def super_strip(word):
  return strip_lex(word.split("+++")[0])

def get_sym_pos(word):
  assert word[0] == "x"
  return int(word[1:])
