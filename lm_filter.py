#!/usr/bin/env python

'''filter srilm file using a dictionary of target words.
cat <srilm-file> | ~/newcode/lm-filter.py <words-file> > <filtered-srilm-file>

note: previous step: building the target dictionary.
several options:
1. from a phrase/rule table. see targetwords.py (implemented for 3 systems: pharaoh, hiero, and ghkm-style)
2. from a kbest translation list. use cut and awk (easy).
   cat 93.5000-best | cut -d "|" -f 4 | awk \'{for (i=1;i<=NF;i++) {print $i}}\' | sort | uniq
   note: -f 4: cut takes only a single char as delimiter while the real thing is \"|||\"
   to split: haven\'t found better way to do it than awk

3. from a forest. can be back via 1 (see forest2rules.py, then targetwords.py) or directly (see targetwords.py -f)
'''

import sys
import re

logs = sys.stderr

# \data\
unigram = "\\1-grams:"
bigram = "\\2-grams:"
trigram = "\\3-grams:"

## warning: i don't care the counts -- they are of no use. but the API will print a warning.

def read_dict(filename):
	d = set(["<s>", "</s>", "<unk>"])

	for line in open(filename, "r"):
		d.add(line.strip())

	print >> logs, "%s: %d words read" % (filename, len(d))
	return d

if __name__ == "__main__":
	def usage():
		print >> logs, "cat <srilm-file> | lm_filter.py <words-file> > <filtered-srilm-file>"		
		print >> logs, "or (batched usage):"
		print >> logs, "\t cat <srilm-file> | lm_filter.py -r <range> -s <input_suffix> [-o <output_suffix>]"
		print >> logs, "\t e.g., cat noUN.srilm.unk | lm_filter.py -r 1-141 -s targetwords -o trigram"
		print >> logs, "\t it will store filtered lms to 1.trigram, 2.trigram, ..."
		
		sys.exit(1)

	import getopt
	rang, suffix = None, None
	outfix = "trigram"
	try:
		opts, args = getopt.getopt(sys.argv[1:], "r:s:o:")
	except:
		usage()
	for o, a in opts:
		if o == "-r":
			f, t = map(int, a.split("-"))
			rang = range(f, t+1)
		elif o == "-s":
			suffix = a
		elif o == "-o":
			outfix = a
		else:
			usage()

	dicts = {}
	outfiles = {}
	if not rang:
		rang = [1]
		dicts[1] = read_dict(args[0])
		outfiles[1] = sys.stdout		 
	else:		  
		for i in rang:			
			dicts[i] = read_dict("%d.%s" % (i, suffix))
			outfiles[i] = open("%d.%s" % (i, outfix), "w")

	tot = all_bad = 0
	bad = dict([(i, 0) for i in rang])	
	for line in sys.stdin:
		tot += 1
		if tot % 100000 == 0:
			print >> logs, "%d lines read, %d (%.2lf%%) bad for all" % (tot, all_bad, all_bad * 100.0 / tot)
		parts = line.split("\t")
		if len(parts) >= 2:
			words = parts[1].split()
			very_bad = True
			for i in rang:
				d = dicts[i]
				o = outfiles[i]
				for word in words:
					if word not in d:
						bad[i] += 1
						break
				else:
					print >> o, line,
					very_bad = False
			if very_bad:
				all_bad += 1
		else:
			## keep all non-ngram lines
			for o in outfiles.values():
				print >> o, line,

	for i in rang:
		outfiles[i].close()
		good = tot - bad[i]
		print >> logs, "%d.%s: %d out of %d saved, ratio = %.3lf" % (i, outfix, good, tot, float(good) / tot)

	good = tot - all_bad
	print >> logs, "%d out of %d useful, ratio = %.3lf" % (good, tot, float(good) / tot)
	
