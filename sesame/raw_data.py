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

import nltk
lemmatizer = nltk.stem.WordNetLemmatizer()

from .conll09 import CoNLL09Element, CoNLL09Example
from .sentence import Sentence


def make_data_instance(text, index):
    """
    Takes a line of text and creates a CoNLL09Example instance from it.
    """
    tokenized = nltk.tokenize.word_tokenize(text.lstrip().rstrip())
    pos_tagged = [p[1] for p in nltk.pos_tag(tokenized)]

    lemmatized = [lemmatizer.lemmatize(tokenized[i])
                    if not pos_tagged[i].startswith("V") else lemmatizer.lemmatize(tokenized[i], pos='v')
                    for i in range(len(tokenized))]

    conll_lines = ["{}\t{}\t_\t{}\t_\t{}\t{}\t_\t_\t_\t_\t_\t_\t_\tO\n".format(
        i+1, tokenized[i], lemmatized[i], pos_tagged[i], index) for i in range(len(tokenized))]
    elements = [CoNLL09Element(conll_line) for conll_line in conll_lines]

    sentence = Sentence(syn_type=None, elements=elements)
    instance = CoNLL09Example(sentence, elements)

    return instance
