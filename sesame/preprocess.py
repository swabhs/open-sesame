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

'''
Reads XML files containing FrameNet 1.$VERSION annotations, and converts them to a CoNLL 2009-like format.
'''
import codecs
import os.path
import sys

import importlib
importlib.reload(sys)

import tqdm
import xml.etree.ElementTree as et
from optparse import OptionParser

from .globalconfig import (VERSION, TRAIN_EXEMPLAR, TRAIN_FTE, DEV_CONLL, TEST_CONLL,
                          FULLTEXT_DIR, PARSER_DATA_DIR, TEST_FILES, DEV_FILES, LU_DIR, EMBEDDINGS_FILE)
from .xml_annotations import FrameAnnotation, SentenceAnnotation


optpr = OptionParser()
optpr.add_option("--filter_embeddings", action="store_true", default=False)
optpr.add_option("--exemplar", action="store_true", default=False)
(options, args) = optpr.parse_args()

logger = open("preprocess-fn{}.log".format(VERSION), "w")

trainf = TRAIN_EXEMPLAR
ftetrainf = TRAIN_FTE
devf = DEV_CONLL
testf = TEST_CONLL

trainsentf = TRAIN_EXEMPLAR + ".sents"
ftetrainsentf = TRAIN_FTE + ".sents"
devsentf = DEV_CONLL + ".sents"
testsentf = TEST_CONLL + ".sents"

relevantfelayers = ["Target", "FE"]
relevantposlayers = ["BNC", "PENN"]
ns = {'fn' : 'http://framenet.icsi.berkeley.edu'}

firsts = {trainf: True,
          devf: True,
          testf: True,
          ftetrainf: True}
sizes = {trainf: 0,
         devf: 0,
         testf: 0,
         ftetrainf: 0}
totsents = numsentsreused = fspno = numlus = 0.0
isfirst = isfirstsent = True


def write_to_conll(outf, fsp, firstex, sentid):
    mode = "a"
    if firstex:
        mode = "w"

    with codecs.open(outf, mode, "utf-8") as outf:
        for i in range(fsp.sent.size()):
            token, postag, nltkpostag, nltklemma, lu, frm, role = fsp.info_at_idx(i)

            outf.write(str(i + 1) + "\t")  # ID = 0
            outf.write(str(token.encode('utf-8')) + "\t")  # FORM = 1
            outf.write("_\t" + nltklemma + "\t")  # LEMMA PLEMMA = 2,3
            outf.write(postag + "\t" + nltkpostag + "\t")  # POS PPOS = 4,5
            outf.write(str(sentid - 1) + "\t_\t")  # FEAT PFEAT = 6,7 ~ replacing FEAT with sentence number
            outf.write("_\t_\t")  # HEAD PHEAD = 8,9
            outf.write("_\t_\t")  # DEPREL PDEPREL = 10,11
            outf.write(lu + "\t" + frm + "\t")  # FILLPRED PRED = 12,13
            outf.write(role + "\n")  #APREDS = 14

        outf.write("\n")  # end of sentence
        outf.close()


def write_to_sent_file(outsentf, sentence, isfirstsent):
    mode = "a"
    if isfirstsent: mode = "w"

    with codecs.open(outsentf, mode, "utf-8") as outf:
        outf.write(sentence + "\n")  # end of sentence
        outf.close()


def process_xml_labels(label, layertype):
    try:
        st = int(label.attrib["start"])
        en = int(label.attrib["end"])
    except KeyError:
        logger.write("\t\tIssue: start and/or end labels missing in " + layertype + "\n")
        return
    return (st, en)


def process_sent(sent, outsentf, isfirstsent):
    senttext = ""
    for t in sent.findall('fn:text', ns):  # not a real loop
        senttext = t.text

    write_to_sent_file(outsentf, senttext, isfirstsent)
    sentann = SentenceAnnotation(senttext)

    for anno in sent.findall('fn:annotationSet', ns):
        for layer in anno.findall('fn:layer', ns):
            layertype = layer.attrib["name"]
            if layertype in relevantposlayers:
                for label in layer.findall('fn:label', ns):
                    startend = process_xml_labels(label, layertype)
                    sentann.add_token(startend)
                    sentann.add_postag(label.attrib["name"])
                if sentann.normalize_tokens(logger) is None:
                    logger.write("\t\tSkipping: incorrect tokenization\n")
                    return
                break
        if sentann.foundpos:
            break

    if not sentann.foundpos:
        # TODO do some manual tokenization
        logger.write("\t\tSkipping: missing POS tags and hence tokenization\n")
        return
    return sentann


def get_all_fsps_in_sent(sent, sentann, fspno, lex_unit, frame, isfulltextann, corpus):
    numannosets = 0
    fsps = {}
    fspset = set([])

    # get all the FSP annotations for the sentece : it might have multiple targets and hence multiple FSPs
    for anno in sent.findall('fn:annotationSet', ns):
        annotation_id = anno.attrib["ID"]
        if annotation_id == "2019791" and VERSION == "1.5":
            # Hack to skip an erroneous annotation of Cathedral as raise.v with frame "Growing_food".
            continue
        numannosets += 1
        if numannosets == 1:
            continue
        anno_id = anno.attrib["ID"]
        if isfulltextann: # happens only for fulltext annotations
            if "luName" in anno.attrib:
                if anno.attrib["status"] == "UNANN" and "test" not in corpus: # keep the unannotated frame-elements only for test, to enable comparison
                    continue
                lex_unit = anno.attrib["luName"]
                frame = anno.attrib["frameName"]
                if frame == "Test35": continue # bogus frame
            else:
                continue
            logger.write("\tannotation: " + str(anno_id) + "\t" + frame + "\t" + lex_unit + "\n")
        fsp = FrameAnnotation(lex_unit, frame, sentann)

        for layer in anno.findall('fn:layer', ns):  # not a real loop
            layertype = layer.attrib["name"]
            if layertype not in relevantfelayers:
                continue
            if layertype == "Target" :
                for label in layer.findall('fn:label', ns):  # can be a real loop
                    startend = process_xml_labels(label, layertype)
                    if startend is None:
                        break
                    fsp.add_target(startend, logger)
            elif layer.attrib["name"] == "FE" and layer.attrib["rank"] == "1":
                for label in layer.findall('fn:label', ns):
                    startend = process_xml_labels(label, layertype)
                    if startend is None:
                        if "itype" in label.attrib:
                            logger.write("\t\tIssue: itype = " + label.attrib["itype"] + "\n")
                            continue
                        else:
                            break
                    fsp.add_fe(startend, label.attrib["name"], logger)

        if not fsp.foundtarget:
            logger.write("\t\tSkipping: missing target\n")
            continue
        if not fsp.foundfes:
            logger.write("\t\tIssue: missing FSP annotations\n")
        if fsp not in fspset:
            fspno += 1
            fsps[anno_id] = fsp
            fspset.add(fsp)
        else:
            logger.write("\t\tRepeated frames encountered for same sentence\n")

    return numannosets, fspno, fsps


def get_annoids(filelist, outf, outsentf):
    annos = []
    isfirstex = True
    fspno = 0
    numsents = 0
    invalidsents = 0
    repeated = 0
    totfsps = 0
    sents = set([])
    isfirstsentex = True

    for tfname in tqdm.tqdm(filelist):
        tfname = os.path.join(FULLTEXT_DIR, tfname)
        logger.write("\n" + tfname + "\n")
        if not os.path.isfile(tfname):
            continue
        with codecs.open(tfname, 'rb', 'utf-8') as xml_file:
            tree = et.parse(xml_file)

        root = tree.getroot()
        for sentence in root.iter('{http://framenet.icsi.berkeley.edu}sentence'):
            numsents += 1
            logger.write("sentence:\t" + str(sentence.attrib["ID"]) + "\n")
            for annotation in sentence.iter('{http://framenet.icsi.berkeley.edu}annotationSet'):
                annotation_id = annotation.attrib["ID"]
                if annotation_id == "2019791" and VERSION == "1.5":
                    # Hack to skip an erroneous annotation of Cathedral as raise.v with frame "Growing_food".
                    continue
                if "luName" in annotation.attrib and "frameName" in annotation.attrib:
                    annos.append(annotation.attrib["ID"])
            # get the tokenization and pos tags for a sentence
            sentann = process_sent(sentence, outsentf, isfirstsentex)
            isfirstsentex = False
            if sentann is None:
                invalidsents += 1
                logger.write("\t\tIssue: Token-level annotations not found\n")
                continue

            # get all the different FSP annotations in the sentence
            x, fspno, fsps = get_all_fsps_in_sent(sentence, sentann, fspno, None, None, True, outf)
            totfsps += len(fsps)
            if len(fsps) == 0: invalidsents += 1
            if sentann.text in sents:
                repeated += 1
            for fsp in list(fsps.values()):
                sents.add(sentann.text)
                write_to_conll(outf, fsp, isfirstex, numsents)
                sizes[outf] += 1
                isfirstex = False
        xml_file.close()
    sys.stderr.write("# total sents processed = %d\n" % numsents)
    sys.stderr.write("# repeated sents        = %d\n" % repeated)
    sys.stderr.write("# invalid sents         = %d\n" % invalidsents)
    sys.stderr.write("# sents in set          = %d\n" % len(sents))
    sys.stderr.write("# annotations           = %d\n" % totfsps)
    return annos


def process_fulltext():
    sys.stderr.write("\nReading {} fulltext data ...\n".format(VERSION))

    # read and write all the test examples in conll
    logger.write("\n\nTEST\n\n")
    sys.stderr.write("TEST\n")
    test_annos = get_annoids(TEST_FILES, testf, testsentf)

    # read and write all the dev examples in conll
    logger.write("\n\nDEV\n\n")
    sys.stderr.write("DEV\n")
    dev_annos = get_annoids(DEV_FILES, devf, devsentf)

    # read all the full-text train examples in conll
    train_fte_files = []
    for f in os.listdir(FULLTEXT_DIR):
        if f not in TEST_FILES and f not in DEV_FILES and not f.endswith("xsl"):
            train_fte_files.append(f)
    logger.write("\n\nFULLTEXT TRAIN\n\n")
    sys.stderr.write("FULLTEXT TRAIN\n")
    get_annoids(train_fte_files, ftetrainf, ftetrainsentf)

    return dev_annos, test_annos


def process_lu_xml(lufname, all_exemplars, dev_annos, test_annos):
    global totsents, numsentsreused, fspno, numlus, isfirst, isfirstsent
    with codecs.open(lufname, 'rb', 'utf-8') as xml_file:
        tree = et.parse(xml_file)
    root = tree.getroot()
    ns = {'fn': 'http://framenet.icsi.berkeley.edu'}

    frame = root.attrib["frame"]
    lex_unit = root.attrib["name"]
    logger.write("\n" + lufname + "\t" + frame + "\t" + lex_unit + "\n")

    sentno = 0
    for sent in root.iter('{http://framenet.icsi.berkeley.edu}sentence'):
        sentno += 1
        # get the tokenization and pos tags for a sentence
        sent_id = int(sent.attrib["ID"])
        logger.write("sentence:\t" + str(sent_id) + "\n")

        sentann = process_sent(sent, trainsentf, isfirstsent)
        isfirstsent = False

        if sentann is None:
            logger.write("\t\tIssue: Token-level annotations not found\n")
            continue

        # get all the different FSP annotations in the sentence
        numannosets, fspno, fsps = get_all_fsps_in_sent(sent, sentann, fspno, lex_unit, frame, False, "exemplartrain")
        for anno_id in fsps:
            if anno_id in test_annos or anno_id in dev_annos:
                continue
            else:
                if sent_id in all_exemplars:
                    all_exemplars[sent_id].append(fsps[anno_id])
                else:
                    all_exemplars[sent_id] = [fsps[anno_id]]
                sizes[trainf] += 1

        if numannosets > 2:
            numsentsreused += (numannosets - 2)
    numlus += 1
    xml_file.close()

    logger.write(lufname + ": total sents = " + str(sentno) + "\n")
    totsents += sentno

    return all_exemplars

def process_exemplars(dev_annos, test_annos):
    global totsents, numsentsreused, fspno, numlus, isfirst
    # get the names of all LU xml files
    all_lus = []
    for f in os.listdir(LU_DIR):
        luf = os.path.join(LU_DIR, f)
        if luf.endswith("xsl"):
            continue
        all_lus.append(luf)
    sys.stderr.write("\nReading exemplar data from " + str(len(all_lus)) + " LU files...\n")

    logger.write("\n\nTRAIN EXEMPLAR\n\n")

    all_exemplars = {}
    for luname in tqdm.tqdm(sorted(all_lus)):
        if not os.path.isfile(luname):
            logger.write("\t\tIssue: Couldn't find %s - strange, terminating!\n" % (luname))
            break
        all_exemplars = process_lu_xml(luname, all_exemplars, dev_annos, test_annos)

    total_exemplars = sum([len(x) for x in all_exemplars.values()])
    sys.stderr.write("\nWriting %d exemplars to %s ...\n" % (total_exemplars, trainf))
    isfirst = True
    for write_id, sentid in enumerate(sorted(all_exemplars), 1):
        for fsp_ in all_exemplars[sentid]:
            write_to_conll(trainf, fsp_, isfirst, sentid=write_id)
            isfirst = False

    sys.stderr.write("\n\n# total LU sents = %d \n" % (totsents))
    sys.stderr.write("# total LU FSPs = %d \n"  % (fspno))
    sys.stderr.write("# total LU files = %d \n" % (numlus))
    sys.stderr.write("average # FSPs per LU = %.3f \n" % (fspno / numlus))
    sys.stderr.write("# LU sents reused for multiple annotations = %d \n" % (numsentsreused))
    sys.stderr.write("\noutput file sizes:\n")
    for s in sizes:
        sys.stderr.write("%s :\t %d \n" % (s, sizes[s]))
    sys.stderr.write("\n")


def filter_embeddings(embedding_files):
    """
    Filters the embeddings file to retain only the vocabulary in the train, dev and test files.
    """
    sys.stderr.write("\nReading FrameNet {} vocabulary...\n".format(VERSION))
    vocab = set([])
    corpora = [DEV_CONLL, TRAIN_FTE, TRAIN_EXEMPLAR, TEST_CONLL]
    for corpus in corpora:
        with codecs.open(corpus, "r", "utf-8") as cf:
            tokens = [line.split("\t")[1].lower() for line in cf if line != "\n"]
            cf.close()
        vocab.update(tokens)
    sys.stderr.write("\nTotal (train + dev + test) vocabulary size = {}\nFiltering out the word vectors ...".format(len(vocab)))

    for emb_file in embedding_files:
        embeddings_file = open(DATA_DIR + emb_file, 'r')
        new_embeddings_file = DATA_DIR.split(".txt")[0] + VERSION + ".framevocab.txt"
        filtered_embeddings = open(new_embeddings_file, 'w')
        num_embeddings = 0
        for l in embeddings_file:
            fields = l.strip().split(' ')
            wd = fields[0].lower()
            if wd in vocab:
                filtered_embeddings.write(l)
                num_embeddings += 1
        embeddings_file.close()
        filtered_embeddings.close()
        sys.stderr.write("\nTotal embeddings in {} = {}\n".format(new_embeddings_file, num_embeddings))


if __name__ == "__main__":
    if not os.path.exists(PARSER_DATA_DIR):
        os.makedirs(PARSER_DATA_DIR)

    dev, test = process_fulltext()

    if options.exemplar:
        process_exemplars(dev, test)

    if options.filter_embeddings:
        filter_embeddings([EMBEDDINGS_FILE])

    logger.close()
