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
import codecs
import sys

import globalconfig as gc

def read_sents(sent_file):
    sentences = []
    with codecs.open(sent_file, "r", "utf-8") as sentf:
        for line in sentf:
            words = line.strip().split()
            sentences.append(words)
        sentf.close()
    print(f"number of sentences: {len(sentences)}")
    return sentences

def read_fe_file(fe_file):
    # format:1   0.0 4   Measure_mass    pound.n 15  pounds  742 Count   14  Unit    15  Stuff   16:17
    #        crap   crap    numframesfes    frame   lu  targetpos target  sentnum fename  fepos
    frames = {}
    tfdict1 = {}
    sent1 = None
    with codecs.open(fe_file, "r", "utf-8") as fef:
        numframes = 0
        for line in fef:
            fields = line.strip().split("\t")
            targetpos = fields[5].split("_")
            sentnum = int(fields[7])

            tfdict = {}
            if sentnum in frames:
                tfdict = frames[sentnum]
            for tp in targetpos:
                if int(tp) in tfdict:
                    print(f"repetition!!! {tfdict, sentnum,} repeated target = {tp, (fields[3],fields[4])}")
                    tfdict1[int(tp)] = (fields[3],fields[4])
                    sent1 = sentnum
                    continue
                tfdict[int(tp)] = (fields[3],fields[4])
            frames[sentnum]= tfdict
            numframes += int(len(targetpos) > 0)
        print(f"number of annotated sentences: {len(frames)}")
        print(f"number of FSPs: {numframes}")
        fef.close()
    return frames, tfdict1, sent1

def join_google_fe_test_conll(conllfile, frames, tfdict1, sent1, outfile):
    with codecs.open(outfile, "w", "utf-8") as outf:
        with codecs.open(conllfile, "r", "utf-8") as cf:
            for l in cf:
                cfields = l.strip().split("\t")
                if len(cfields) == 1:
                    outf.write("\n")
                    continue

                if len(cfields) != 15:
                    raise Exception("incorrect CoNLL 2009 format", l, cfields)
                newfields = cfields[:12]

                sent_num = int(cfields[6])
                position = int(cfields[0])-1
                if sent_num in frames and position in frames[sent_num] and cfields[12] != gc.EMPTY_LABEL:
                    newfields.append(cfields[12]) # keep our LUs but use their frames.
                    newfields.append(frames[sent_num][position][0])
                    newfields.append(gc.EMPTY_FE)
                elif sent_num == sent1 and position in tfdict1:
                    newfields.append(cfields[12]) # keep our LUs but use their frames.
                    newfields.append(tfdict1[position][0])
                    newfields.append(gc.EMPTY_FE)
                else:
                    newfields += [gc.EMPTY_LABEL, gc.EMPTY_LABEL, gc.EMPTY_FE] # FILLPRED PRED APREDS = 12,13,14
                if len(newfields) != len(cfields):
                    raise Exception("didn't join properly", len(newfields), len(cfields), newfields)
                outf.write("\t".join(newfields) + "\n")
            cf.close()
        outf.close()

def write_tab_separated(sents, frames, tfdict1, sent1):
    '''
    this one is for Michael Roth to carry out his experiments
    :param sents:
    :param frames:
    :param tfdict1:
    :param sent1:
    :return:
    '''
    with codecs.open("roth"+str(len(sents)), "w", "utf-8") as outf:

        for i in range(len(sents)):
            for j in range(len(sents[i])):
                outf.write(str(j+1) + "\t")
                outf.write(sents[i][j] + "\t")
                if i in frames and j in frames[i]:
                    outf.write(frames[i][j][0] + "\t" + frames[i][j][1])
                else:
                    outf.write("_\t_")
                outf.write("\n")
            outf.write("\n")
            if i == sent1:
                for j in range(len(sents[i])):
                    outf.write(str(j+1) + "\t")
                    outf.write(sents[i][j] + "\t")
                    if j in tfdict1:
                        outf.write(tfdict1[j][0] + "\t" + tfdict1[j][1])
                    else:
                        outf.write("_\t_")
                    outf.write("\n")
                outf.write("\n")
        outf.close()

s = read_sents(sys.argv[1])
f, t1, s1 = read_fe_file(sys.argv[2])
join_google_fe_test_conll(sys.argv[3], f, t1, s1, "dip.predictedframes.conll")
