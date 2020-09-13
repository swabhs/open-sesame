# coding=utf-8
# Copyright 2018 Swabha Swayamdipta. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import sys

exfs = {}
ensemblename = sys.argv[1]

sys.stderr.write("reading " + ensemblename + " factors... ")
for en in range(1, 6):
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
