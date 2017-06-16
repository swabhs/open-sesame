import numpy as np
import sys

exfs = {}
ensemblename = sys.argv[1]

sys.stderr.write("reading " + ensemblename + " factors... ")
for en in xrange(1, 6):
    sys.stderr.write(str(en) + " ")
    enf = open(ensemblename + str(en), "rb")

    for l in enf:
        fields = l.split("\t")
        fac = (fields[0], fields[1], fields[2], fields[3])
        if en == 1:
            exfs[fac] = float(fields[4])
        else:
            exfs[fac] = np.add(exfs[fac], float(fields[4]))

    enf.close()
sys.stderr.write(" done!\n")

outf = open("full_" + ensemblename, "w")
for fac in exfs:
    outf.write(fac[0] + "\t" + fac[1] + "\t" + fac[2] + "\t" + fac[3] + "\t" + str(exfs[fac]) + "\n")
outf.close()
