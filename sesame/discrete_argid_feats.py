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


class ArgPosition(object):
    BEFORE, AFTER, OVERLAP, WITHIN = list(range(4))

    @staticmethod
    def size():
        return 4

    @staticmethod
    def whereisarg(span, targetspan):
        if span[1] < targetspan[0]:
            return ArgPosition.BEFORE
        elif span[0] > targetspan[1]:
            return ArgPosition.AFTER
        elif span[0] >= targetspan[0] and span[1] <= targetspan[1]:
            return ArgPosition.WITHIN
        return ArgPosition.OVERLAP


class SpanWidth(object):
    one, two, threefive, fiveten, tenfifteen, fifteentwenty, evenmore = list(range(7))

    @staticmethod
    def size():
        return 7

    @staticmethod
    def howlongisspan(beg, end):
        slen = end - beg + 1
        if beg == end:
            return SpanWidth.one
        elif slen == 2:
            return SpanWidth.two
        elif 3 <= slen <= 5:
            return SpanWidth.threefive
        elif 6 <= slen <= 10:
            return SpanWidth.fiveten
        elif 11 <= slen <= 15:
            return SpanWidth.tenfifteen
        elif 16 <= slen <= 20:
            return SpanWidth.fifteentwenty
        return SpanWidth.evenmore


class OutHeads(object):
    zero, one, two, three, four, five, btwn610, btwn1115, morethan15 = list(range(9))

    @staticmethod
    def size():
        return 9

    @staticmethod
    def getnumouts(beg, end, outheads):
        if (beg, end) not in outheads:
            raise Exception("error in outhead calculation", beg, end, outheads)
        numheads = outheads[(beg, end)]
        if numheads == 0:
            return OutHeads.zero
        elif numheads == 1:
            return OutHeads.one
        elif numheads == 2:
            return OutHeads.two
        elif numheads == 3:
            return OutHeads.three
        elif numheads == 4:
            return OutHeads.four
        elif numheads == 5:
            return OutHeads.five
        elif 5 < numheads <= 10:
            return OutHeads.btwn610
        elif 10 < numheads <= 15:
            return OutHeads.btwn1115
        return OutHeads.morethan15