# -*- coding: utf-8 -*-
from framesemparse import *
from copy import deepcopy
from housekeeping import *

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
    """ all the elements in a single line of a CoNLL 2009 file"""

    def __init__(self, conll_line, read_depsyn):
        ele = conll_line.split("\t")
        lufields = ['_', '_']
        self.id = int(ele[0])
        self.form = VOCDICT.addstr(ele[1].lower())
        self.nltk_lemma = LEMDICT.addstr(ele[3])
        self.fn_pos = ele[4]  # not really a gold POS tag, just one provided by FrameNet, ignore
        self.nltk_pos = POSDICT.addstr(ele[5])
        self.sent_num = int(ele[6])

        self.dephead = NOTALABEL
        self.deprel = NOTALABEL
        if read_depsyn:
            self.dephead = int(ele[9])
            self.deprel = DEPRELDICT.addstr(ele[11])

        self.is_pred = (ele[12] != NOTALABEL)
        if self.is_pred:
            lufields = ele[12].split(".")
        self.lu = LUDICT.addstr(lufields[0])
        self.lupos = LUPOSDICT.addstr(lufields[1])
        self.frame = FRAMEDICT.addstr(ele[13])

        # BIOS scheme
        self.is_arg = (ele[14] != NOTANFE)
        self.argtype = ARGTYPES[ele[14][0]]
        if self.is_arg:
            self.role = FEDICT.addstr(ele[14][2:])
        else:
            self.role = FEDICT.addstr(ele[14])

    def get_str(self, rolelabel=None):
        idstr = str(self.id) + "\t"
        form = VOCDICT.getstr(self.form) + "\t"
        lem = LEMDICT.getstr(self.nltk_lemma) + "\t"
        nltkpos = POSDICT.getstr(self.nltk_pos) + "\t"

        dephead = "_\t"
        deprel = "_\t"
        if self.dephead != NOTALABEL:
            dephead = str(self.dephead) + "\t"
            deprel = DEPRELDICT.getstr(self.deprel) + "\t"

        if self.is_pred:
            lu = LUDICT.getstr(self.lu) + "." + LUPOSDICT.getstr(self.lupos) + "\t"
        else:
            lu = LUDICT.getstr(self.lu) + "\t"
        frame = FRAMEDICT.getstr(self.frame) + "\t"

        if rolelabel is None:
            if self.is_arg:
                rolelabel = INV_ARGTYPES[self.argtype] + "-" + FEDICT.getstr(self.role) + "\t"
            else:
                rolelabel = INV_ARGTYPES[self.argtype] + "\t"

        if DEBUGMODE:
            return idstr + form + lu + frame + rolelabel
        else:
            return (idstr  # ID = 0
                    + form  # FORM = 1
                    + "_\t" + lem  # LEMMA PLEMMA = 2,3
                    + self.fn_pos + "\t" + nltkpos  # POS PPOS = 4,5
                    + str(self.sent_num) + "\t_\t"  # FEAT PFEAT = 6,7 ~ replacing FEAT with sentence number
                    + "_\t" + dephead  # HEAD PHEAD = 8,9
                    + "_\t" + deprel  # DEPREL PDEPREL = 10,11
                    + lu + frame  # FILLPRED PRED = 12,13
                    + rolelabel + "\n")  # APREDS = 14


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

        if FEDICT.getid(NOTANFE) in self.invertedfes:
            self.invertedfes[FEDICT.getid(NOTANFE)] = extract_spans(notfes)

        # self.invertedfes = self._get_inverted_femap()
        # for felabel in self.invertedfes:
        #     if felabel == FEDICT.getid(NOTANFE): continue
        #     argranges = self.invertedfes[felabel]
        #     for arng in argranges:
        #         if arng[0] == arng[1]:
        #             self.add_fe(arng[0], SINGLE, felabel)
        #         else:
        #             self.add_fe(arng[0], BEG, felabel)
        #             self.add_fe(arng[1], END, felabel)

        self.modifiable = False  # true cz generally gold.

    def _get_inverted_femap(self):
        tmp = {}
        for e in self._elements:
            # if not e.is_arg:
            #     continue
            # elif e.role not in tmp:
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
            rolelabels = [NOTANFE for _ in self._elements]
            for feid in predictedfes:
                felabel = FEDICT.getstr(feid)
                if felabel == NOTANFE:
                    continue
                for argspan in predictedfes[feid]:
                    if argspan[0] == argspan[1]:
                        rolelabels[argspan[0]] = INV_ARGTYPES[SINGULAR] + "-" + felabel
                    else:
                        rolelabels[argspan[0]] = INV_ARGTYPES[BEGINNING] + "-" + felabel
                    for position in xrange(argspan[0] + 1, argspan[1] + 1):
                        rolelabels[position] = INV_ARGTYPES[INSIDE] + "-" + felabel

            for e, role in zip(self._elements, rolelabels):
                mystr += e.get_str(rolelabel=role)

        return mystr

    def get_newstr_frame(self, targetpred):  # after substituting predicted frame
        mystr = ""
        for e in xrange(len(self._elements)):
            tmp = deepcopy(self._elements[e])
            if (tmp.id - 1) in targetpred:
                tmp.is_pred = True
                tmp.lu = targetpred[tmp.id - 1][0].id
                tmp.lupos = targetpred[tmp.id - 1][0].posid
                tmp.frame = targetpred[tmp.id - 1][1].id
            else:
                tmp.is_pred = False
                tmp.lu = LUDICT.getid(NOTALABEL)
                tmp.lupos = LUPOSDICT.getid(NOTALABEL)
                tmp.frame = FRAMEDICT.getid(NOTALABEL)
            mystr += tmp.get_str()
        return mystr

    # def get_newstr_fe(self, targetfes): # after substituting predicted FEs
    #     mystr = ""
    #     insidespan = False
    #     felabel = None
    #     for e in xrange(len(self._elements)):
    #         tmp = deepcopy(self._elements[e])
    #         tmp.is_arg = None
    #         tmp.argtype = None
    #         tmp.role = None
    #         if (tmp.id - 1) in targetfes:
    #
    #             fetype = targetfes[tmp.id - 1].keys()[0] # there is only one item in the map fes[fepos]
    #             felabel = targetfes[tmp.id - 1].values()[0].id
    #
    #             tmp.is_arg = True
    #             if fetype == SINGLE:  # TODO: CHANGE!!!
    #                 tmp.argtype = SINGULAR
    #             elif fetype == BEG:
    #                 tmp.argtype = BEGINNING
    #             else:
    #                 tmp.argtype = INSIDE
    #             tmp.role = felabel
    #             if felabel is None: tmp.role = FEDICT.getid(UNK)
    #
    #             if fetype == BEG:
    #                 tmp.argtype = BEGINNING
    #                 insidespan = True
    #             else:
    #                 insidespan = False
    #
    #         elif insidespan:
    #             tmp.is_arg = True
    #             tmp.argtype = INSIDE
    #             tmp.role = felabel
    #             if felabel is None: # TODO: how is it inside a span and UNK?
    #                 tmp.role = FEDICT.getid(UNK)
    #
    #         else:
    #             tmp.is_arg = False
    #             tmp.argtype = OUTSIDE
    #             tmp.role = FEDICT.getid(NOTANFE)
    #
    #         mystr += tmp.get_str()
    #     return mystr

    def print_internal(self):
        self.print_internal_sent()
        self.print_internal_frame()
        self.print_internal_args()

    def print_internal_sent(self):
        print "tokens and depparse:\n",
        for x in xrange(len(self.tokens)):
            print VOCDICT.getstr(self.tokens[x]),
        print

    def print_internal_frame(self):
        print "LU and frame:",
        for tfpos in self.targetframedict:
            t, f = self.targetframedict[tfpos]
            print VOCDICT.getstr(self.tokens[tfpos]), ":", \
                LUDICT.getstr(t.id) + "." + LUPOSDICT.getstr(t.posid), \
                FRAMEDICT.getstr(f.id)

    def print_external_frame(self, predtf):
        print "LU and frame:",
        for tfpos in predtf:
            t, f = predtf[tfpos]
            print VOCDICT.getstr(self.tokens[tfpos]), ":", \
                LUDICT.getstr(t.id) + "." + LUPOSDICT.getstr(t.posid), \
                FRAMEDICT.getstr(f.id)

    def print_internal_args(self):
        print "frame:", FRAMEDICT.getstr(self.frame.id).upper()
        for fepos in self.invertedfes:
            if fepos == FEDICT.getid(NOTANFE):
                continue
            for span in self.invertedfes[fepos]:
                print FEDICT.getstr(fepos), "\t",
                for s in xrange(span[0], span[1] + 1):
                    print VOCDICT.getstr(self.tokens[s]),
                print
        print

    def print_external_parse(self, parse):
        for fepos in parse:
            if fepos == FEDICT.getid(NOTANFE):
                continue
            for span in parse[fepos]:
                print FEDICT.getstr(fepos), "\t",
                for s in xrange(span[0], span[1] + 1):
                    print VOCDICT.getstr(self.tokens[s]),
                print
        print


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
