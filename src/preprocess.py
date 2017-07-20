#!/usr/bin/python

'''
Reads xml files containing FrameNet 1.x LUs, and converts them to CoNLL format
'''
import os.path
import codecs
import xml.etree.ElementTree as et
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from globalconfig import *
from xmlannotations import *

trainf = TRAIN_EXEMPLAR
ftetrainf = TRAIN_FTE
devf = DEV_CONLL
testf = TEST_CONLL

trainsentf = TRAIN_EXEMPLAR + ".sents"
ftetrainsentf = TRAIN_FTE + ".sents"
devsentf = DEV_CONLL + ".sents"
testsentf = TEST_CONLL + ".sents"

DEVSELECTPROB = 0.0
relevantfelayers = ["Target", "FE"]
relevantposlayers = ["BNC", "PENN"]
ns = {'fn' : 'http://framenet.icsi.berkeley.edu'}


firsts = {trainf:True, devf:True, testf:True, ftetrainf:True}
sizes = {trainf:0, devf:0, testf:0, ftetrainf:0}

def write_to_conll(outf, fsp, firstex, sentid):
    mode = "a"
    if firstex:
        mode = "w"

    with codecs.open(outf, mode, "utf-8") as outf:
        for i in xrange(fsp.sent.size()):
            token, postag, nltkpostag, nltklemma, lu, frm, role = fsp.info_at_idx(i)

            outf.write(str(i+1) + "\t") # ID = 0
            outf.write(token.encode('utf-8') + "\t") # FORM = 1
            outf.write("_\t" + nltklemma + "\t") # LEMMA PLEMMA = 2,3
            outf.write(postag + "\t" + nltkpostag + "\t") # POS PPOS = 4,5
            outf.write(str(sentid-1) + "\t_\t") # FEAT PFEAT = 6,7 ~ replacing FEAT with sentence number
            outf.write("_\t_\t") # HEAD PHEAD = 8,9
            outf.write("_\t_\t") # DEPREL PDEPREL = 10,11
            outf.write(lu + "\t" + frm + "\t") # FILLPRED PRED = 12,13
            outf.write(role + "\n") #APREDS = 14

        outf.write("\n") # end of sentence
        outf.close()

def write_to_sent_file(outsentf, sentence, isfirstsent):
    mode = "a"
    if isfirstsent: mode = "w"

    with codecs.open(outsentf, mode, "utf-8") as outf:
        outf.write(sentence + "\n") # end of sentence
        outf.close()

def process_xml_labels(label, layertype):
    try:
        st = int(label.attrib["start"])
        en = int(label.attrib["end"])
    except KeyError:
        sys.stderr.write("\t\tIssue: start and/or end labels missing in " + layertype + "\n")
        return
    return (st, en)

def process_sent(sent, outsentf, isfirstsent):
    senttext = ""
    for t in sent.findall('fn:text', ns): # not a real loop
        senttext = t.text

    write_to_sent_file(outsentf, senttext, isfirstsent)
    sentann = SentAnno(senttext)

    for anno in sent.findall('fn:annotationSet', ns):
        for layer in anno.findall('fn:layer', ns):
            layertype = layer.attrib["name"]
            if layertype in relevantposlayers:
                for label in layer.findall('fn:label', ns):
                    startend = process_xml_labels(label, layertype)
                    sentann.add_token(startend)
                    sentann.add_postag(label.attrib["name"])
                if sentann.normalize_tokens() is None:
                    sys.stderr.write("\t\tSkipping: incorrect tokenization\n")
                    return
                break
        if sentann.foundpos:
            break

    if not sentann.foundpos:
        # TODO do some manual tokenization
        sys.stderr.write("\t\tSkipping: missing POS tags and hence tokenization\n")
        return
    return sentann

def get_all_fsps_in_sent(sent, sentann, fspno, lex_unit, frame, isfulltextann, corpus):
    numannosets = 0
    fsps = {}
    fspset = set([])

    # get all the FSP annotations for the sentece : it might have multiple targets and hence multiple FSPs
    for anno in sent.findall('fn:annotationSet', ns):
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
            sys.stderr.write("\tannotation: " + str(anno_id) + "\t" + frame + "\t" + lex_unit + "\n")
        fsp = FrameAnno(lex_unit, frame, sentann)

        for layer in anno.findall('fn:layer', ns): # not a real loop
            layertype = layer.attrib["name"]
            if layertype not in relevantfelayers:
                continue
            if layertype == "Target" :
                for label in layer.findall('fn:label', ns): # can be a real loop
                    startend = process_xml_labels(label, layertype)
                    if startend is None:
                        break
                    fsp.add_target(startend)
            elif layer.attrib["name"] == "FE" and layer.attrib["rank"] == "1":
                for label in layer.findall('fn:label', ns):
                    startend = process_xml_labels(label, layertype)
                    if startend is None:
                        if "itype" in label.attrib:
                            sys.stderr.write("\t\tIssue: itype = " + label.attrib["itype"] + "\n")
                            continue
                        else:
                            break
                    fsp.add_fe(startend, label.attrib["name"])

        if not fsp.foundtarget:
            sys.stderr.write("\t\tSkipping: missing target\n")
            continue
        if not fsp.foundfes:
            sys.stderr.write("\t\tIssue: missing FSP annotations\n")
        if fsp not in fspset:
            fspno += 1
            fsps[anno_id] = fsp
            fspset.add(fsp)
        else:
            sys.stderr.write("\t\tRepeated frames encountered for same sentence\n")

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

    for tfname in filelist:
        tfname = FTEDIR + tfname
        sys.stderr.write("\n" + tfname + "\n")
        if not os.path.isfile(tfname):
            continue
        with codecs.open(tfname, 'rb', 'utf-8') as xml_file:
            tree = et.parse(xml_file)

        root = tree.getroot()
        for sentence in root.iter('{http://framenet.icsi.berkeley.edu}sentence'):
            numsents += 1
            sys.stderr.write("sentence:\t" + str(sentence.attrib["ID"]) + "\n")
            for annotation in sentence.iter('{http://framenet.icsi.berkeley.edu}annotationSet'):
                if "luName" in annotation.attrib and "frameName" in annotation.attrib:
                    annos.append(annotation.attrib["ID"])
            # get the tokenization and pos tags for a sentence
            sentann = process_sent(sentence, outsentf, isfirstsentex)
            isfirstsentex = False
            if sentann is None:
                invalidsents += 1
                sys.stderr.write("\t\tIssue: Token-level annotations not found\n")
                continue

            # get all the different FSP annotations in the sentence
            x, fspno, fsps = get_all_fsps_in_sent(sentence, sentann, fspno, None, None, True, outf)
            totfsps += len(fsps)
            if len(fsps) == 0: invalidsents += 1
            if sentann.text in sents:
                repeated += 1
            for fsp in fsps.values():
                sents.add(sentann.text)
                write_to_conll(outf, fsp, isfirstex, numsents)
                sizes[outf] += 1
                isfirstex = False
        xml_file.close()
    sys.stdout.write("# total sents processed = %d\n" %numsents)
    sys.stdout.write("# repeated sents        = %d\n" %repeated)
    sys.stdout.write("# invalid sents         = %d\n" %invalidsents)
    sys.stdout.write("# sents in set          = %d\n" %len(sents))
    sys.stdout.write("# annotations           = %d\n" %totfsps)
    return annos

def process_fulltext():
    sys.stdout.write("\nReading fulltext data...\n")

    # read and write all the test examples in conll
    sys.stderr.write("\n\nTEST\n\n")
    sys.stdout.write("TEST\n")
    test_annos = get_annoids(TESTFILES, testf, testsentf)

    # read and write all the dev examples in conll
    sys.stderr.write("\n\nDEV\n\n")
    sys.stdout.write("DEV\n")
    dev_annos = get_annoids(DEVFILES, devf, devsentf)

    # read all the full-text train examples in conll
    train_fte_files = []
    for f in os.listdir(FTEDIR):
        if f not in TESTFILES and f not in DEVFILES and not f.endswith("xsl"):
            train_fte_files.append(f)
    sys.stderr.write("\n\nFULLTEXT TRAIN\n\n")
    sys.stdout.write("FULLTEXT TRAIN\n")
    get_annoids(train_fte_files, ftetrainf, ftetrainsentf)

    return dev_annos, test_annos

def process_lu_xml(lufname, dev_annos, test_annos):
    global totsents, numsentsreused, fspno, numlus, isfirst, isfirstsent
    with codecs.open(lufname, 'rb', 'utf-8') as xml_file:
        tree = et.parse(xml_file)
    root = tree.getroot()
    ns ={'fn' : 'http://framenet.icsi.berkeley.edu'}

    frame = root.attrib["frame"]
    lex_unit = root.attrib["name"]
    sys.stderr.write("\n" + lufname + "\t" + frame + "\t" + lex_unit + "\n")

    sentno = 0
    for sent in root.iter('{http://framenet.icsi.berkeley.edu}sentence'):
        sentno += 1
        # get the tokenization and pos tags for a sentence
        sent_id = sent.attrib["ID"]
        sys.stderr.write("sentence:\t" + str(sent_id) + "\n")

        sentann = process_sent(sent, trainsentf, isfirstsent)
        isfirstsent = False

        if sentann is None:
            sys.stderr.write("\t\tIssue: Token-level annotations not found\n")
            continue

        # get all the different FSP annotations in the sentence
        numannosets, fspno, fsps = get_all_fsps_in_sent(sent, sentann, fspno, lex_unit, frame, False, "exemplartrain")
        for anno_id in fsps:
            if anno_id in test_annos or anno_id in dev_annos:
                continue
            else:
                write_to_conll(trainf, fsps[anno_id], isfirst, sentno)
                sizes[trainf] += 1
                isfirst = False

        if numannosets > 2:
            numsentsreused += (numannosets - 2)
    numlus += 1
    xml_file.close()

    sys.stderr.write(lufname + ": total sents = " + str(sentno) + "\n")
    totsents += sentno

def process_exemplars(dev_annos, test_annos):
    global totsents, numsentsreused, fspno, numlus, isfirst
    # get the names of all LU xml files
    all_lus = []
    for f in os.listdir(LUDIR):
        luf = os.path.join(LUDIR, f)
        if luf.endswith("xsl"):
            continue
        all_lus.append(luf)
    sys.stdout.write("\nReading exemplar data from " + str(len(all_lus)) + " LU files...\n")

    sys.stderr.write("\n\nTRAIN EXEMPLAR\n\n")
    for i, luname in enumerate(sorted(all_lus), 1):
        if i % 1000 == 0:
            sys.stdout.write(str(i) + "...")
        if not os.path.isfile(luname):
            sys.stderr.write("\t\tIssue: Couldn't find " + luname + " - strange, terminating!\n")
            break
        process_lu_xml(luname, dev_annos, test_annos)

    sys.stdout.write("\n\n# total LU sents = " + str(totsents) + "\n")
    sys.stdout.write("# total LU FSPs = " + str(fspno) + "\n")
    sys.stdout.write("# total LU files = " + str(numlus) + "\n")
    sys.stdout.write("average # FSPs per LU = " + str(fspno / numlus) + "\n")
    sys.stdout.write("# LU sents reused for multiple annotations = " + str(numsentsreused) + "\n")
    sys.stdout.write("\noutput file sizes:\n")
    for s in sizes:
        sys.stdout.write(s + ":\t" + str(sizes[s]) + "\n")
    sys.stdout.write("\n")

def preprocess_wvf(ws):
    """
    read the train, dev and test files, along with the word vectors to find which words to retain vectors for.
    :return:
    """
    sys.stdout.write("Reading FrameNet vocabulary...\n")
    reqdtoks = set([])
    corpora = [DEV_CONLL, TRAIN_FTE, TEST_CONLL]
    for c in corpora:
        with codecs.open(c, "r", "utf-8") as cf:
            ctoks = [line.split("\t")[1].lower() for line in cf  if line != "\n"]
            cf.close()
        reqdtoks.update(ctoks)
    sys.stdout.write("\ntotal(train+dev+test) vocabulary size = " + str(len(reqdtoks)) + "\nfiltering out the word vectors...")

    #ws = ["glove.6B.100d.txt", "glove.6B.50d.txt",  "glove.840B.300d.txt", "glove.6B.200d.txt"]
    for w in ws:
        wvf = open(DATADIR + w, 'r')
        newwvf = DATADIR + w[:-3] + "framevocab.txt"
        filtered_wvf = open(newwvf, 'w')
        numwv = 0
        for l in wvf:
            fields = l.strip().split(' ')
            wd = fields[0].lower()
            if wd in reqdtoks:
                filtered_wvf.write(l)
                numwv +=1
        wvf.close()
        filtered_wvf.close()
        sys.stdout.write("\ntotal num word vectors in file " + newwvf + " = " + str(numwv) + "\n")

dev, test = process_fulltext()

totsents = numsentsreused = fspno = numlus = 0.0
isfirst = isfirstsent = True
process_exemplars(dev, test)

if len(sys.argv) >= 2:
   preprocess_wvf([sys.argv[1]])
