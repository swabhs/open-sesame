from globalconfig import *
# import sys, codecs
#
# UTF8Writer = codecs.getwriter('utf8')
# sys.stdout = UTF8Writer(sys.stdout)

class LexicalUnit(object):

    def __init__(self, id, posid):
        self.id = id
        self.posid = posid

    def get_str(self, ludict, luposdict):
        return ludict.getstr(self.id) + " " + luposdict.getstr(posid)

    def __hash__(self):
        return hash((self.id, self.posid))

    def __eq__(self, other):
        return self.id == other.id and self.posid == other.posid

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)


class Frame(object):

    def __init__(self, id):
        self.id = id

    def get_str(self, framedict):
        return framedict.getstr(self.id)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)


class FrameElement(object):

    def __init__(self, id):
        self.id = id

    def get_str(self, fedict):
        return fedict.getstr(self.id)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)


class FrameSemParse(object):
    """frame-semantic parse structure, contain a target LU, frame it evokes, arguments and corresponding frame elements all in the context of a sentence"""

    def __init__(self, sentence):
        self.tokens = sentence.tokens
        self.postags = sentence.postags
        self.sentence = sentence # TODO: clunky, there should be some inheritance, etc.
        self.targetframedict = {} # map of target position and frame-id
        self.frame = None
        self.lu = None
        # self.fes = {} # map of FE position to a map between FE-type(BIOS) and the label
        self.numargs = 0
        self.modifiable = True # to differentiate between gold and predicted

    def print_example(self, vocdict, ludict, fedict, framedict):
        if not DEBUGMODE:
            return
        for t in self.tokens:
            print vocdict.getstr(t),
        print
        for tf in self.targetframedict:
            t, f = self.targetframedict[tf]
            print "LexUnit: ", ludict.getstr(t.id)
            print "Frame: ", f.get_str(framedict),
        print
        for fe in self.fes:
            print "FE:", fedict.getstr(fe.id),
            for span in self.fes[fe]:
                for pos in span:
                    print vocdict.getstr(self.tokens[pos])
        print 

    def add_target(self, targetpos, luid, lupos, frameid):
        if not self.modifiable:
            raise Exception('attempt to add target and frame to unmodifiable example')
        if targetpos in self.targetframedict:
            raise Exception('target already in parse', targetpos, frameid)

        if self.frame is not None and frameid != self.frame.id:
            raise Exception("two different frames in a single parse, illegal", frameid, self.frame.id)
        self.frame = Frame(frameid)

        if self.lu is not None and luid != self.lu.id:
            raise Exception("different LU ID than original", self.lu.id, luid)
        self.lu = LexicalUnit(luid, lupos)
        self.targetframedict[targetpos] = (self.lu, self.frame)
        
    def add_fe(self, felabelid, festartpos, feendpos):
        if not self.modifiable:
            raise Exception('attempt to add argument to unmodifiable example')
        fe = FrameElement(felabelid)
        if fe not in self.fes:
            self.fes[fe] = []
        self.fes[fe].append((festartpos,feendpos))
        self.numargs += 1

    def get_only_targets(self):
        if self.modifiable:
            raise Exception('still modifying the example, incomplete...')
        tdict = {}
        for luidx in self.targetframedict:
            tdict[luidx] = self.targetframedict[luidx][0]
        return tdict
