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
from .globalconfig import DEBUG_MODE

class LexicalUnit(object):

    def __init__(self, id, posid):
        self.id = id
        self.posid = posid

    def get_str(self, ludict, luposdict):
        return ludict.getstr(self.id) + "." + luposdict.getstr(self.posid)

    def __hash__(self):
        return hash((self.id, self.posid))

    def __eq__(self, other):
        return self.id == other.id and self.posid == other.posid

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)


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
        return not (self == other)


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
        return not (self == other)


class FrameSemParse(object):
    """frame-semantic parse structure, contain a target LU, frame it evokes, arguments and corresponding frame elements all in the context of a sentence"""

    def __init__(self, sentence):
        self.tokens = sentence.tokens
        self.postags = sentence.postags
        self.lemmas = sentence.lemmas
        # TODO(Swabha): add some inheritance, etc.
        self.sentence = sentence
        self.targetframedict = {}  # map of target position and frame-id
        self.frame = None
        self.lu = None
        # self.fes = {} # map of FE position to a map between FE-type(BIOS) and the label
        self.numargs = 0
        self.modifiable = True  # to differentiate between gold and predicted

    def add_target(self, targetpos, luid, lupos, frameid):
        if not self.modifiable:
            raise Exception(
                'attempt to add target and frame to unmodifiable example')
        if targetpos in self.targetframedict:
            raise Exception('target already in parse', targetpos, frameid)

        if self.frame is not None and frameid != self.frame.id:
            raise Exception(
                "two different frames in a single parse, illegal", frameid, self.frame.id)
        self.frame = Frame(frameid)

        if self.lu is not None and luid != self.lu.id:
            raise Exception("different LU ID than original", self.lu.id, luid)
        self.lu = LexicalUnit(luid, lupos)
        self.targetframedict[targetpos] = (self.lu, self.frame)

    def get_only_targets(self):
        if self.modifiable:
            raise Exception('still modifying the example, incomplete...')
        tdict = {}
        for luidx in self.targetframedict:
            tdict[luidx] = self.targetframedict[luidx][0]
        return tdict
