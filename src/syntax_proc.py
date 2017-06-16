__author__ = 'swabha'

import sys,codecs
from itertools import izip

def join_fnconll_parseyconll(conllfile, synfile, outfile):
    with codecs.open(outfile, "w", "utf-8") as outf:
        with codecs.open(conllfile, "r", "utf-8") as cf:
            with codecs.open(synfile, "r", "utf-8") as sf:
                for l,sl in izip(cf,sf):

                    cfields = l.strip().split("\t")
                    if len(cfields) == 1:
                        outf.write("\n")
                        continue

                    if len(cfields) != 15:
                        raise Exception("incorrect CoNLL 2009 format", l, cfields)

                    sfields = sl.strip().split("\t")
                    if len(sfields) != 10:
                        raise Exception("incorrect parsey CoNLL format")

                    newfields = cfields[:4] # ID FORM LEMMA PLEMMA = 0,1,2,3
                    newfields += sfields[3:6:2] # syntaxnetPOS fnPOS = 4,5  ~ replacing POS PPOS
                    newfields += cfields[6:9] # sent_num PFEAT HEAD = 6,7,8 ~ replacing FEAT PFEAT HEAD
                    newfields += sfields[6:7] # syntaxnetHEAD = 9           ~ replacing PHEAD
                    newfields += cfields[10:11] # DEPREL = 10
                    newfields += sfields[7:8] # syntaxnetDEPREL = 11        ~ replacing PDEPREL
                    newfields += cfields[12:] # FILLPRED PRED APREDS = 12,13,14
                    if len(newfields) != len(cfields):
                        raise Exception("didn't join properly", len(newfields), len(cfields), newfields)
                    outf.write("\t".join(newfields) + "\n")
                sf.close()
            cf.close()
        outf.close()

join_fnconll_parseyconll(sys.argv[1], sys.argv[2], sys.argv[3])