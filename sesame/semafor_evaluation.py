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

from __future__ import division

import codecs
import sys

from optparse import OptionParser

from . import conll09 as c9
from .dataio import read_conll
from .globalconfig import EMPTY_FE


def convert_conll_to_frame_elements(conllfile, fefile):
    """
    SEMAFOR style FE file format:
    1   0.0 4   Measure_mass    pound.n 15  pounds  742 Count   14  Unit    15  Stuff   16:17
    """
    examples, _, _ = read_conll(conllfile)

    notanfe = c9.FEDICT.getid(EMPTY_FE)
    with codecs.open(fefile, "w", "utf-8") as outf:

        for ex in examples:
            numfes = sum([len(ex.invertedfes[fi]) for fi in ex.invertedfes if fi != notanfe]) + 1  # num(FEs + frame)
            frame = c9.FRAMEDICT.getstr(ex.frame.id)
            lu = c9.LUDICT.getstr(ex.lu.id) + "." + c9.LUPOSDICT.getstr(ex.lu.posid)
            tfkeys = sorted(ex.targetframedict.keys())
            tfpos = str(tfkeys[0])
            target = c9.VOCDICT.getstr(ex.tokens[tfkeys[0]])

            # multi-token targets
            if len(tfkeys) > 1:
                tfpos += "_" + str(tfkeys[-1])
            for tpos in tfkeys[1:]:
                target += " " + c9.VOCDICT.getstr(ex.tokens[tpos])

            outf.write("1\t0.0\t"
                       + str(numfes) + "\t"
                       + frame + "\t"
                       + lu + "\t"
                       + tfpos + "\t"
                       + target + "\t"
                       + str(ex.sent_num) + "\t")

            for fe in ex.invertedfes:
                festr = c9.FEDICT.getstr(fe)
                if festr == EMPTY_FE:
                    continue

                # SEMAFOR doesn't predict, but does evaluate against multiple spans,
                # so the following is good
                for span in ex.invertedfes[fe]:
                    outf.write(festr + "\t")
                    if span[0] == span[1]:
                        outf.write(str(span[0]) + "\t")
                    else:
                        outf.write(str(span[0]) + ":" + str(span[1]) + "\t")
            outf.write("\n")

        outf.close()


def count_frame_elements(fefile):
    haslongerspans = False
    with codecs.open(fefile, "r", "utf-8") as fef:
        numfes = 0
        for line in fef:
            fields = line.strip().split("\t")
            if len(fields) < 8:
                raise Exception('what is this?', line)
            numfes += len(fields) - 8
            for span in fields[9::2]:
                ele = span.split(":")
                if len(ele) == 1:
                    spanlen = 1
                else:
                    spanlen = int(ele[1]) - int(ele[0]) + 1
                if spanlen > 20:
                    haslongerspans = True
        fef.close()
    print(f"#FEs = {numfes / 2}")
    print(f"contains longer spans? {haslongerspans}")


def detail_read_fe_file(fefile):
    exwithdiscontfe = 0
    frames = {}
    with codecs.open(fefile, "r", "utf-8") as fef:
        numframes = 0
        for line in fef:
            fields = line.strip().split("\t")
            # tpositions = fields[5].split("_")
            sentnum = int(fields[7])

            tfdict = {}
            if sentnum in frames:
                tfdict = frames[sentnum]

            if (fields[5], fields[3]) in tfdict:
                raise Exception("frame already present!!!", fields[3], tfdict[(fields[5], fields[3])])
            else:
                fes = {}
                for x in range(8, len(fields), 2):
                    fefield, fespan = fields[x:x + 2]
                    if fefield in fes:
                        print("discontinous FEs found in ", fields[2:])
                        exwithdiscontfe += 1
                    else:
                        fes[fefield] = []
                    spanpos = fespan.split(":")
                    if len(spanpos) == 1:
                        spanbeg = spanend = int(spanpos[0])
                    else:
                        spanbeg = int(spanpos[0])
                        spanend = int(spanpos[-1])
                    fes[fefield].append((spanbeg, spanend))
                tfdict[(fields[5], fields[3])] = fes
            frames[sentnum] = tfdict
        for sent in frames:
            numframes += len(frames[sent])
        sys.stderr.write("# annotated sentences in %s: %d\n" % (fefile, len(frames)))
        sys.stderr.write("# FSPs: %d\n" % numframes)
        sys.stderr.write("# FSPs with discontinuous arguments: %d\n" % exwithdiscontfe)
        fef.close()
    return frames


def compare_fefiles(fefile1, fefile2):
    framel1 = detail_read_fe_file(fefile1)
    framel2 = detail_read_fe_file(fefile2)
    if len(framel1) != len(framel2):
        raise Exception("unequal!")

    for sent in framel1:
        if sent not in framel2:
            raise Exception("where is this sentence?", framel1[sent], sent)

    for sent in framel2:
        if sent not in framel1:
            raise Exception("where is this sentence?", framel2[sent], sent)

    # they have the same sentences
    for sent in framel1:
        tf1 = framel1[sent]
        tf2 = framel2[sent]
        if len(tf1) != len(tf2):
            raise Exception("different frames in sent ", sent, framel1[sent], framel2[sent])
        for key in tf1:
            if key not in tf2:
                print("where is this frame in " + fefile2 + " ", sent, key, tf1[key])
        for key in tf2:
            if key not in tf1:
                print("where is this frame in " + fefile1 + " ", sent, key, tf2[key])

        # they have same frames
        for key in tf1:
            if key not in tf2: continue
            if len(tf1[key]) != len(tf2[key]):
                print("different number of FEs!", sent, key, tf1[key], tf2[key])
            for fe in tf1[key]:
                if fe not in tf2[key]:
                    print("missing FE in " + fefile2 + " ", sent, fe, tf1[key])

            for fe in tf2[key]:
                if fe not in tf1[key]:
                    print("missing FE in " + fefile1 + " ", sent, fe, tf2[key])

            # they have same fes
            for fe in tf2[key]:
                if fe in tf1[key] and set(tf2[key][fe]) != set(tf1[key][fe]):
                    raise Exception("mismatching spans", key, fe, sent)

def main():
    e_parser = OptionParser()
    e_parser.add_option("--e_mode",
                        dest="e_mode",
                        type="choice",
                        choices=["convert_conll_to_fe", "count_frame_elements", "compare_fefiles"],
                        default="convert_conll_to_fe")
    e_parser.add_option("--conll_file", type="str", metavar="FILE")
    e_parser.add_option("--fe_file", type="str", metavar="FILE")
    e_parser.add_option("--fe_file_other", type="str", metavar="FILE")
    e_options, _ = e_parser.parse_args()

    if e_options.e_mode == "convert_conll_to_fe":
        assert e_options.conll_file and e_options.fe_file
        convert_conll_to_frame_elements(e_options.conll_file, e_options.fe_file)
    elif e_options.e_mode == "count_frame_elements":
        assert e_options.fe_file
        count_frame_elements(e_options.fe_file)
    elif e_options.e_mode == "compare_fefiles":
        assert e_options.fe_file and e_options.fe_file_other
        compare_fefiles(e_options.fe_file, e_options.fe_file_other)

if __name__ == "__main__":
    main()
