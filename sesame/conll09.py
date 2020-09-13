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
from copy import deepcopy

from .frame_semantic_graph import LexicalUnit, Frame, FrameElement, FrameSemParse
from .globalconfig import EMPTY_LABEL, EMPTY_FE, BIO_INDEX_DICT, INDEX_BIO_DICT, DEBUG_MODE, SINGULAR, BEGINNING, INSIDE
from .housekeeping import FspDict, extract_spans

VOCDICT = FspDict()
LEMDICT = FspDict()
POSDICT = FspDict()
FRAMEDICT = FspDict()
LUDICT = FspDict()
LUPOSDICT = FspDict()
FEDICT = FspDict()
DEPRELDICT = FspDict()
CLABELDICT = FspDict()


class CoNLL09Element:
    """
    All the elements in a single line of a CoNLL 2009-like file.
    """

    def __init__(self, conll_line, read_depsyn=None):
        ele = conll_line.split("\t")
        lufields = ['_', '_']
        self.id = int(ele[0])
        self.form = VOCDICT.addstr(ele[1].lower())
        self.nltk_lemma = LEMDICT.addstr(ele[3])
        self.fn_pos = ele[4]  # Not a gold POS tag, provided by taggers used in FrameNet, ignore.
        self.nltk_pos = POSDICT.addstr(ele[5])
        self.sent_num = int(ele[6])

        self.dephead = EMPTY_LABEL
        self.deprel = EMPTY_LABEL
        if read_depsyn:
            self.dephead = int(ele[9])
            self.deprel = DEPRELDICT.addstr(ele[11])

        self.is_pred = (ele[12] != EMPTY_LABEL)
        if self.is_pred:
            lufields = ele[12].split(".")
        self.lu = LUDICT.addstr(lufields[0])
        self.lupos = LUPOSDICT.addstr(lufields[1])
        self.frame = FRAMEDICT.addstr(ele[13])

        # BIOS scheme
        self.is_arg = (ele[14] != EMPTY_FE)
        self.argtype = BIO_INDEX_DICT[ele[14][0]]
        if self.is_arg:
            self.role = FEDICT.addstr(ele[14][2:])
        else:
            self.role = FEDICT.addstr(ele[14])

    def get_str(self, rolelabel=None, no_args=False):
        idstr = str(self.id)
        form = VOCDICT.getstr(self.form)
        predicted_lemma = LEMDICT.getstr(self.nltk_lemma)
        nltkpos = POSDICT.getstr(self.nltk_pos)

        dephead = "_"
        deprel = "_"
        if self.dephead != EMPTY_LABEL:
            dephead = str(self.dephead)
            deprel = DEPRELDICT.getstr(self.deprel)

        if self.is_pred:
            lu = LUDICT.getstr(self.lu) + "." + LUPOSDICT.getstr(self.lupos)
        else:
            lu = LUDICT.getstr(self.lu)
        frame = FRAMEDICT.getstr(self.frame)

        if rolelabel is None:
            if self.is_arg:
                rolelabel = INDEX_BIO_DICT[self.argtype] + "-" + FEDICT.getstr(self.role)
            else:
                rolelabel = INDEX_BIO_DICT[self.argtype]

        if no_args:  # For Target ID / Frame ID predictions
            rolelabel = "O"

        if DEBUG_MODE:
            return idstr + form + lu + frame + rolelabel
        else:
            # ID    FORM    LEMMA   PLEMMA  POS PPOS    SENT#   PFEAT   HEAD    PHEAD   DEPREL  PDEPREL LU  FRAME ROLE
            # 0     1       2       3       4   5       6       7       8       9       10      11      12  13    14
            return f"{self.id}\t{form}\t_\t{predicted_lemma}\t{self.fn_pos}\t{nltkpos}\t{self.sent_num}\t_\t_\t{dephead}\t_\t{deprel}\t{lu}\t{frame}\t{rolelabel}\n"

class CoNLL09Example(FrameSemParse):
    """a single example in CoNLL 09 format which corresponds to a single frame-semantic parse structure"""

    def __init__(self, sentence, elements):
        FrameSemParse.__init__(self, sentence)
        # not in parent class
        self._elements = elements
        self.sent_num = elements[0].sent_num

        notfes = []
        self.invertedfes = {}
        for e in elements:
            if e.is_pred:
                self.add_target((e.id - 1), e.lu, e.lupos, e.frame)

            if e.role not in self.invertedfes:
                self.invertedfes[e.role] = []
            if e.argtype == SINGULAR:
                self.invertedfes[e.role].append((e.id - 1, e.id - 1))
                self.numargs += 1
            elif e.argtype == BEGINNING:
                self.invertedfes[e.role].append((e.id - 1, None))
                self.numargs += 1
            elif e.argtype == INSIDE:
                argspan = self.invertedfes[e.role].pop()
                self.invertedfes[e.role].append((argspan[0], e.id - 1))
            else:
                notfes.append(e.id - 1)

        if FEDICT.getid(EMPTY_FE) in self.invertedfes:
            self.invertedfes[FEDICT.getid(EMPTY_FE)] = extract_spans(notfes)

        self.modifiable = False  # true cz generally gold.

    def _get_inverted_femap(self):
        tmp = {}
        for e in self._elements:
            if e.role not in tmp:
                tmp[e.role] = []
            tmp[e.role].append(e.id - 1)

        inverted = {}
        for felabel in tmp:
            argindices = sorted(tmp[felabel])
            argranges = extract_spans(argindices)
            inverted[felabel] = argranges

        return inverted

    def get_str(self, predictedfes=None):
        mystr = ""
        if predictedfes is None:
            for e in self._elements:
                mystr += e.get_str()
        else:
            rolelabels = [EMPTY_FE for _ in self._elements]
            for feid in predictedfes:
                felabel = FEDICT.getstr(feid)
                if felabel == EMPTY_FE:
                    continue
                for argspan in predictedfes[feid]:
                    if argspan[0] == argspan[1]:
                        rolelabels[argspan[0]] = INDEX_BIO_DICT[SINGULAR] + "-" + felabel
                    else:
                        rolelabels[argspan[0]] = INDEX_BIO_DICT[BEGINNING] + "-" + felabel
                    for position in range(argspan[0] + 1, argspan[1] + 1):
                        rolelabels[position] = INDEX_BIO_DICT[INSIDE] + "-" + felabel

            for e, role in zip(self._elements, rolelabels):
                mystr += e.get_str(rolelabel=role)

        return mystr

    def get_predicted_frame_conll(self, predicted_frame):
        """
        Get new CoNLL string, after substituting predicted frame.
        """
        new_conll_str = ""
        for e in range(len(self._elements)):
            field = deepcopy(self._elements[e])
            if (field.id - 1) in predicted_frame:
                field.is_pred = True
                field.lu = predicted_frame[field.id - 1][0].id
                field.lupos = predicted_frame[field.id - 1][0].posid
                field.frame = predicted_frame[field.id - 1][1].id
            else:
                field.is_pred = False
                field.lu = LUDICT.getid(EMPTY_LABEL)
                field.lupos = LUPOSDICT.getid(EMPTY_LABEL)
                field.frame = FRAMEDICT.getid(EMPTY_LABEL)
            new_conll_str += field.get_str()
        return new_conll_str

    def get_predicted_target_conll(self, predicted_target, predicted_lu):
        """
        Get new CoNLL string, after substituting predicted target.
        """
        new_conll_str = ""
        for e in range(len(self._elements)):
            field = deepcopy(self._elements[e])
            if (field.id - 1) == predicted_target:
                field.is_pred = True
                field.lu = predicted_lu.id
                field.lupos = predicted_lu.posid
            else:
                field.is_pred = False
                field.lu = LUDICT.getid(EMPTY_LABEL)
                field.lupos = LUPOSDICT.getid(EMPTY_LABEL)
            field.frame = FRAMEDICT.getid(EMPTY_LABEL)
            new_conll_str += field.get_str(no_args=True)
        return new_conll_str

    def print_internal(self, logger):
        self.print_internal_sent(logger)
        self.print_internal_frame(logger)
        self.print_internal_args(logger)

    def print_internal_sent(self, logger):
        logger.write("tokens and depparse:\n")
        for x in range(len(self.tokens)):
            logger.write(VOCDICT.getstr(self.tokens[x]) + " ")
        logger.write("\n")

    def print_internal_frame(self, logger):
        logger.write("LU and frame: ")
        for tfpos in self.targetframedict:
            t, f = self.targetframedict[tfpos]
            logger.write(VOCDICT.getstr(self.tokens[tfpos]) + ":" + \
                LUDICT.getstr(t.id) + "." + LUPOSDICT.getstr(t.posid) + \
                FRAMEDICT.getstr(f.id) + "\n")

    def print_external_frame(self, predtf, logger):
        logger.write("LU and frame: ")
        for tfpos in predtf:
            t, f = predtf[tfpos]
            logger.write(VOCDICT.getstr(self.tokens[tfpos]) + ":" + \
                LUDICT.getstr(t.id) + "." + LUPOSDICT.getstr(t.posid) + \
                FRAMEDICT.getstr(f.id) + "\n")

    def print_internal_args(self, logger):
        logger.write("frame:" + FRAMEDICT.getstr(self.frame.id).upper() + "\n")
        for fepos in self.invertedfes:
            if fepos == FEDICT.getid(EMPTY_FE):
                continue
            for span in self.invertedfes[fepos]:
                logger.write(FEDICT.getstr(fepos) + "\t")
                for s in range(span[0], span[1] + 1):
                    logger.write(VOCDICT.getstr(self.tokens[s]) + " ")
                logger.write("\n")
        logger.write("\n")

    def print_external_parse(self, parse, logger):
        for fepos in parse:
            if fepos == FEDICT.getid(EMPTY_FE):
                continue
            for span in parse[fepos]:
                logger.write(FEDICT.getstr(fepos) + "\t")
                for s in range(span[0], span[1] + 1):
                    logger.write(VOCDICT.getstr(self.tokens[s]) + " ")
                logger.write("\n")
        logger.write("\n")


def lock_dicts():
    VOCDICT.lock()
    LEMDICT.lock()
    POSDICT.lock()
    FRAMEDICT.lock()
    LUDICT.lock()
    LUPOSDICT.lock()
    FEDICT.lock()
    DEPRELDICT.lock()
    CLABELDICT.lock()


def post_train_lock_dicts():
    VOCDICT.post_train_lock()
    LEMDICT.post_train_lock()
    POSDICT.post_train_lock()
    FRAMEDICT.post_train_lock()
    LUDICT.post_train_lock()
    LUPOSDICT.post_train_lock()
    FEDICT.post_train_lock()
    DEPRELDICT.post_train_lock()
    CLABELDICT.post_train_lock()
