import sys, codecs, os
import xml.etree.ElementTree as et
from conll09 import *
from sentence import *
from nltk.corpus import BracketParseCorpusReader
from globalconfig import *

def read_conll(conllfile, syn_type=None):
    sys.stderr.write("reading " + conllfile + "...\n")

    read_depsyn = read_constits = False
    if syn_type == "dep":
        read_depsyn = True
    elif syn_type == "constit":
        read_constits = True
        cparses = read_brackets(CONSTIT_MAP[conllfile])


    examples = []
    elements = []
    missingargs = 0.0
    totalexamples = 0.0

    # outf = open(conllfile+".sentswithargs", "wb")
    next = 0
    with codecs.open(conllfile, "r", "utf-8") as cf:
        snum = -1
        for l in cf:
            l = l.strip()
            if l == "":
                if elements[0].sent_num != snum:
                    sentence = Sentence(syn_type,elements=elements)
                    if read_constits:
                        sentence.get_all_parts_of_ctree(cparses[next], CLABELDICT, True)
                    next += 1
                    snum = elements[0].sent_num
                e = CoNLL09Example(sentence, elements)
                examples.append(e)
                if read_depsyn:
                    sentence.get_all_paths_to(sorted(e.targetframedict.keys())[0])
                elif read_constits:
                    sentence.get_cpath_to_target(sorted(e.targetframedict.keys())[0])

                if e.numargs == 0:
                    missingargs += 1

                # outf.write(str(snum)+'\n')
                totalexamples += 1

                elements = []
                continue
            elements.append(CoNLL09Element(l, read_depsyn))
        cf.close()
        # outf.close()
    sys.stderr.write("# examples in %s : %d in %d sents\n" %(conllfile, len(examples), next))
    sys.stderr.write("# examples with missing arguments : %d\n\n" %missingargs)
    if read_constits: analyze_constits_fes(examples)
    return examples, missingargs, totalexamples

def analyze_constits_fes(examples):
    matchspan = 0.0
    notmatch = 0.0
    matchph = {}
    for ex in examples:
        for fe in ex.invertedfes:
            if fe == FEDICT.getid(NOTALABEL): continue
            for span in ex.invertedfes[fe]:
                if span in ex.sentence.constitspans:
                    matchspan += 1
                    phrases = ex.sentence.constitspans[span]
                    for phrase in phrases:
                        if phrase not in matchph:
                            matchph[phrase] = 0
                        matchph[phrase] += 1
                else:
                    notmatch += 1
    tot = matchspan + notmatch
    sys.stderr.write("matches = %d %.2f%%\n"
                     "non-matches = %d %.2f%%\n"
                     "total = %d\n"
                     % (matchspan, matchspan*100/tot, notmatch, notmatch*100/tot, tot))
    # sorted_mp = sorted(matchph.items(), key=operator.itemgetter(1), reverse=True)
    # for phrase,v in sorted_mp:
    #     sys.stderr.write(CLABELDICT.getstr(phrase) + ":\t" + str(v) + "\n")
    sys.stderr.write("phrases which are constits = %d\n" %(len(matchph)))

def create_target_frame_map(luIndex_file,  tf_map):
    sys.stderr.write("reading the frame - lexunit map from " + LU_INDEX + "...\n")

    f = open(luIndex_file, "rb")
    #    with codecs.open(luIndex_file, "r", "utf-8") as xml_file: # TODO: why won't this right way of reading work?
    tree = et.parse(f)
    root = tree.getroot()

    multiplicity = 0
    repeated = 0
    tot = 0
    for lu in root.iter('{http://framenet.icsi.berkeley.edu}lu'):
        lu_name = lu.attrib["name"]
        frame = lu.attrib["frameName"]
        if lu_name not in tf_map:
            tf_map[lu_name] = []
        else:
            repeated += 1
        tf_map[lu_name].append(frame)
        if len(tf_map[lu_name]) > multiplicity:
            multiplicity = len(tf_map[lu_name])
        tot += 1
    f.close()

    sys.stderr.write("# unique targets = " + str(len(tf_map)) + "\n")
    sys.stderr.write("# total targets = " + str(tot) + "\n")
    sys.stderr.write("# targets with multiple frames = " + str(repeated) + "\n")
    sys.stderr.write("# max frames per target = " + str(multiplicity) + "\n\n")

def read_fes_lus(frame_file):
    f = open(frame_file, "rb")
    #    with codecs.open(luIndex_file, "r", "utf-8") as xml_file: # TODO: why won't this right way of reading work?
    tree = et.parse(f)
    root = tree.getroot()

    frcount = 0
    for frame in root.iter('{http://framenet.icsi.berkeley.edu}frame'):
        framename = frame.attrib["name"]
        frid = FRAMEDICT.addstr(framename)
        frcount += 1

    if frcount > 1:
        raise Exception("More than one frame?", frame_file, framename)

    fes = []
    corefes = []
    for fe in root.iter('{http://framenet.icsi.berkeley.edu}FE'):
        fename = fe.attrib["name"]
        feid = FEDICT.addstr(fename)
        fes.append(feid)
        if fe.attrib["coreType"] == "Core": corefes.append(feid)

    lus = []
    for lu in root.iter('{http://framenet.icsi.berkeley.edu}lexUnit'):
        lu_fields = lu.attrib["name"].split(".")
        luid = LUDICT.addstr(lu_fields[0])
        LUPOSDICT.addstr(lu_fields[1])
        lus.append(luid)
    f.close()

    return frid, fes, corefes, lus

def read_frame_maps():
    sys.stderr.write("reading the frame-element - frame map from " + FRAME_DIR + "...\n")

    frmfemap = {}
    corefrmfemap = {}
    lufrmmap = {}
    maxfesforframe = 0
    longestframe = None

    for f in os.listdir(FRAME_DIR):
        framef = os.path.join(FRAME_DIR, f)
        if framef.endswith("xsl"):
            continue
        frm, fes, corefes, lus = read_fes_lus(framef)
        frmfemap[frm] = fes
        corefrmfemap[frm] = corefes
        if len(frmfemap[frm]) > maxfesforframe:
            maxfesforframe = len(frmfemap[frm])
            longestframe = frm
        for l in lus:
            if l not in lufrmmap:
                lufrmmap[l] = []
            lufrmmap[l].append(frm)

    sys.stderr.write("# max FEs for frame: "  + str(maxfesforframe)
                     + " in Frame(" +FRAMEDICT.getstr(longestframe) + ")\n\n")
    return frmfemap, corefrmfemap, lufrmmap

def read_related_lus():
    sys.stderr.write("reading the frame-element - frame map from " + FRAME_DIR + "...\n")

    lufrmmap = {}
    maxframes = 0
    longestlu = None

    frmlumap = {}
    maxlus = 0
    longestfrm = None

    for f in os.listdir(FRAME_DIR):
        framef = os.path.join(FRAME_DIR, f)
        if framef.endswith("xsl"):
            continue
        frm, fes, corefes, lus = read_fes_lus(framef)

        for l in lus:
            if l not in lufrmmap:
                lufrmmap[l] = set([])
            lufrmmap[l].add(frm)
            if len(lufrmmap[l]) > maxframes:
                maxframes = len(lufrmmap[l])
                longestlu = l


            if frm not in frmlumap:
                frmlumap[frm] = set([])
            frmlumap[frm].add(l)
            if len(frmlumap[frm]) > maxlus:
                maxlus = len(frmlumap[frm])
                longestfrm = frm

    related_lus = {}
    for l in lufrmmap:
        for frm in lufrmmap[l]:
            if frm in frmlumap:
                if l not in related_lus:
                    related_lus[l] = set([])
                related_lus[l].update(frmlumap[frm])

    # print "lu-frame", LUDICT.getstr(lufrmmap.items()[0][0])
    # for x in lufrmmap.items()[0][1]:
    #     print FRAMEDICT.getstr(x),
    # print
    #
    # print "frame-lu", FRAMEDICT.getstr(frmlumap.items()[0][0])
    # for x in frmlumap.items()[0][1]:
    #     print LUDICT.getstr(x),
    # print
    #
    # print "lu-lu", LUDICT.getstr(related_lus.items()[0][0])
    # for x in related_lus.items()[0][1]:
    #     print LUDICT.getstr(x),
    # print

    sys.stderr.write("# max frames for LU: %d in LU(%s)\n"
                     "# max LUs for frame: %d in Frame(%s)\n"
                     % (maxframes, LUDICT.getstr(longestlu),
                        maxlus, FRAMEDICT.getstr(longestfrm)))

    return lufrmmap, related_lus

def get_wvec_map():
    if not os.path.exists(FILTERED_WVECS_FILE):
        raise Exception("word vector file not found!", FILTERED_WVECS_FILE)
    sys.stderr.write("reading the word vectors file from " + FILTERED_WVECS_FILE + "...\n")
    wvf = open(FILTERED_WVECS_FILE,'r')
    wvf.readline()
    wd_vecs = {VOCDICT.addstr(line.split(' ')[0]) : [float(f) for f in line.strip().split(' ')[1:]] for line in wvf}
    return wd_vecs

def get_chains(node, inherit_map, path):
    if node in inherit_map:
        for par in inherit_map[node]:
            path = get_chains(par, inherit_map, path+[par])
    return path

def read_frame_relations():
    sys.stderr.write("reading inheritance relationships from " + FRAME_REL_FILE + "...\n")

    f = open(FRAME_REL_FILE, "rb")
    #  with codecs.open(luIndex_file, "r", "utf-8") as xml_file:
    #  TODO: why won't this right way of reading work?
    tree = et.parse(f)
    root = tree.getroot()

    relations = {}
    commonest_frame_child = None
    max_num_parents = 0
    paths = {}

    fe_relations = {}
    commonest_fe_child = None
    max_parent_fes = 0
    fepaths = {}

    for reltype in root.iter('{http://framenet.icsi.berkeley.edu}frameRelationType'):
        #if reltype.attrib["name"] in ["ReFraming_Mapping", "Precedes"]:
        if reltype.attrib["name"] != "Inheritance":
            continue


        for relation in reltype.findall('{http://framenet.icsi.berkeley.edu}frameRelation'):
            sub_frame = FRAMEDICT.addstr(relation.attrib["subFrameName"])
            super_frame = FRAMEDICT.addstr(relation.attrib["superFrameName"])
            if sub_frame not in relations:
                relations[sub_frame] = []
            relations[sub_frame].append(super_frame)
            if len(relations[sub_frame]) > max_num_parents:
                max_num_parents = len(relations[sub_frame])
                commonest_frame_child = sub_frame

            for ferelation in relation.findall('{http://framenet.icsi.berkeley.edu}FERelation'):
                sub_fe = FEDICT.addstr(ferelation.attrib["subFEName"])
                super_fe = FEDICT.addstr(ferelation.attrib["superFEName"])
                if sub_fe != super_fe:
                    if sub_fe not in fe_relations:
                        fe_relations[sub_fe] = []
                    fe_relations[sub_fe].append(super_fe)
                    if len(fe_relations[sub_fe]) > max_parent_fes:
                        max_parent_fes = len(fe_relations[sub_fe])
                        commonest_fe_child = sub_fe

    f.close()

    for leaf in relations.keys():
        if leaf not in paths:
            paths[leaf] = []
        paths[leaf] += get_chains(leaf, relations, [])
    xpaths = {p:set(paths[p]) for p in paths}

    # TODO: not sure why there is a problem with getting the entire path for FE relations
    # TODO: for now, it's only one hop
    # for feleaf in fe_relations.keys():
    #     fe_relations[feleaf] = list(set(fe_relations[feleaf]))
    #     if feleaf not in fepaths:
    #         fepaths[feleaf] = []
    #     fepaths[feleaf] += get_chains(feleaf, fe_relations, [])
    # xfepaths = {p:set(fepaths[p]) for p in fepaths}

    sys.stderr.write("# descendant frames: %d commonest descendant = %s (%d parents)\n"
                     %(len(xpaths), FRAMEDICT.getstr(commonest_frame_child), max_num_parents))
    sys.stderr.write("# descendant FEs: %d commonest descendant = %s (%d parents)\n\n"
                     %(len(fe_relations), FEDICT.getstr(commonest_fe_child), max_parent_fes))

    return xpaths, fe_relations

def read_brackets(constitfile):
    sys.stderr.write("reading constituents from " + constitfile + "...\n")
    reader = BracketParseCorpusReader(PARSERDATADIR + "rnng/", constitfile)
    parses = reader.parsed_sents()
    return parses

def read_ptb():
    sys.stderr.write("reading PTB data from " + PTBDATADIR + "...\n")
    sentences = []
    senno = 0
    with codecs.open("ptb.sents", "w", "utf-8") as ptbsf:
        for constitfile in os.listdir(PTBDATADIR):
            reader = BracketParseCorpusReader(PTBDATADIR, constitfile)
            parses = reader.parsed_sents()
            # todo map from parses to sentences
            for p in parses:
                ptbsf.write(" ".join(p.leaves()) + "\n")
                tokpos = p.pos()
                tokens = [VOCDICT.addstr(tok) for tok,pos in tokpos]
                postags = [POSDICT.addstr(pos) for tok,pos in tokpos]
                s = Sentence("constit",sentnum=senno,tokens=tokens,postags=postags,)
                s.get_all_parts_of_ctree(p, CLABELDICT, False)
                sentences.append(s)
                senno += 1
            # if senno >= 100: break
        sys.stderr.write("# PTB sentences: %d\n" %len(sentences))
        ptbsf.close()
    return sentences

