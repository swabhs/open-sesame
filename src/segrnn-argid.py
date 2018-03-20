# -*- coding: utf-8 -*-
from evaluation import *
from discreteargidfeats import *
from dynet import *
from arksemaforeval import *
from optparse import OptionParser

import os
import sys
import time
import math

MODELSYMLINK = "model.segrnn-argid." + VERSION
modelfname = "tmp/" + VERSION + "model.sra-" + str(time.time())

optpr = OptionParser()
optpr.add_option("--testf", dest="test_conll", help="Annotated CoNLL test file", metavar="FILE", default=TEST_CONLL)
optpr.add_option("--mode", dest="mode", type='choice', choices=['train', 'test', 'refresh', 'ensemble'],
                 default='train')
optpr.add_option("--saveensemble", action="store_true", default=True)
optpr.add_option('--model', dest="modelfile", help="Saved model file", metavar="FILE", default=MODELSYMLINK)
optpr.add_option("--exemplar", action="store_true", default=False)
optpr.add_option("--spanlen", type='choice', choices=['clip', 'filter'], default='clip')
optpr.add_option('--loss', type='choice', choices=['log', 'softmaxm', 'hinge'], default='softmaxm')
optpr.add_option('--cost', type='choice', choices=['hamming', 'recall'], default='recall')
optpr.add_option('--roc', type='int', default=2)
optpr.add_option("--dropout", action="store_true", default=True)
optpr.add_option("--wordvec", action="store_true", default=True)
optpr.add_option("--hier", action="store_true", default=False)
optpr.add_option("--syn", type='choice', choices=['dep', 'constit', 'none'], default='none')
optpr.add_option("--ptb", action="store_true", default=False)
optpr.add_option("--fefile", help="output frame element file for semafor eval", metavar="FILE",
                 default="my.test.predict.sentences.frame.elements")

(options, args) = optpr.parse_args()
if options.exemplar:
    train_conll = TRAIN_EXEMPLAR
    # TODO: we still don't have exemplar constituent parses
else:
    train_conll = TRAIN_FTE
    train_constits = TRAIN_FTE_CONSTITS

USE_SPAN_CLIP = (options.spanlen == 'clip')
ALLOWED_SPANLEN = 20

USE_DROPOUT = options.dropout
DROPOUT_RATE = 0.1

USE_WV = options.wordvec
USE_HIER = options.hier

USE_DEPS = USE_CONSTITS = False
if options.syn == "dep":
    USE_DEPS = True
elif options.syn == "constit":
    USE_CONSTITS = True

USE_PTB_CONSTITS = options.ptb

SAVE_FOR_ENSEMBLE = (options.mode == "test") and options.saveensemble

UNK_PROB = 0.1
RECALL_ORIENTED_COST = options.roc

sys.stderr.write("\nCOMMAND: " + ' '.join(sys.argv) + "\n")
sys.stderr.write("\nPARSER SETTINGS\n_____________________\n")
sys.stderr.write("PARSING MODE:   \t" + options.mode + "\n")
sys.stderr.write("USING EXEMPLAR? \t" + str(options.exemplar) + "\n")
sys.stderr.write("USING SPAN CLIP?\t" + str(USE_SPAN_CLIP) + "\n")
sys.stderr.write("LOSS TYPE:      \t" + options.loss + "\n")
sys.stderr.write("COST TYPE:      \t" + options.cost + "\n")
sys.stderr.write("R-O COST VALUE: \t" + str(options.roc) + "\n")
sys.stderr.write("USING DROPOUT?  \t" + str(USE_DROPOUT) + "\n")
sys.stderr.write("USING WORDVECS? \t" + str(USE_WV) + "\n")
sys.stderr.write("USING HIERARCHY?\t" + str(USE_HIER) + "\n")
sys.stderr.write("USING D-SYNTAX? \t" + str(USE_DEPS) + "\n")
sys.stderr.write("USING C-SYNTAX? \t" + str(USE_CONSTITS) + "\n")
sys.stderr.write("USING PTB-CLOSS?\t" + str(USE_PTB_CONSTITS) + "\n")

if options.mode in ["train", "refresh"]:
    sys.stderr.write("MODEL WILL BE SAVED TO\t%s\n" % modelfname)
if options.mode == "test":
    sys.stderr.write("SAVING ENSEMBLE?\t" + str(SAVE_FOR_ENSEMBLE) + "\n")
sys.stderr.write("_____________________\n")

if USE_PTB_CONSTITS:
    ptbexamples = read_ptb()

trainexamples, _, _ = read_conll(train_conll, options.syn)
post_train_lock_dicts()

frmfemap, corefrmfemap, _ = read_frame_maps()
# to handle FE in annotation (sigh)
frmfemap[FRAMEDICT.getid("Measurable_attributes")].append(FEDICT.getid("Dimension"))
frmfemap[FRAMEDICT.getid("Removing")].append(FEDICT.getid("Frequency"))

if USE_WV:
    wvs = get_wvec_map()
    sys.stderr.write("using pretrained embeddings of dimension " + str(len(wvs.values()[0])) + "\n")

if USE_HIER:
    frmrelmap, feparents = read_frame_relations()

lock_dicts()
UNKTOKEN = VOCDICT.getid(UNK)

# default labels - in conll format these correspond to _
NOTANLU = LUDICT.getid(NOTALABEL)
NOTANFEID = FEDICT.getid(NOTANFE)

sys.stderr.write("# words in vocab:       " + str(VOCDICT.size()) + "\n")
sys.stderr.write("# POS tags:             " + str(POSDICT.size()) + "\n")
sys.stderr.write("# lexical units:        " + str(LUDICT.size()) + "\n")
sys.stderr.write("# LU POS tags:          " + str(LUPOSDICT.size()) + "\n")
sys.stderr.write("# frames:               " + str(FRAMEDICT.size()) + "\n")
sys.stderr.write("# FEs:                  " + str(FEDICT.size()) + "\n")
sys.stderr.write("# dependency relations: " + str(DEPRELDICT.size()) + "\n")
sys.stderr.write("# constituency labels:  " + str(CLABELDICT.size()) + "\n")

trainexamples = filter_long_ex(trainexamples, USE_SPAN_CLIP, ALLOWED_SPANLEN, NOTANFEID)

if options.mode in ['train', 'refresh']:
    devexamples, _, _ = read_conll(DEV_CONLL, options.syn)
    sys.stderr.write("unknowns in dev\n\n_____________________\n")
    out_conll_file = "argid.predicted.fn" + VERSION + ".dev.conll"
else:
    devexamples, _, _ = read_conll(options.test_conll, options.syn)
    sys.stderr.write("unknowns in test\n\n_____________________\n")
    out_conll_file = "argid.predicted." + options.test_conll.split("/")[-1]
    if SAVE_FOR_ENSEMBLE:
        out_ens_file = "ensemble." + out_conll_file[:-11]
    if options.mode == "ensemble":
        in_ens_file = "full_ensemble." +  out_conll_file[:-11]

sys.stderr.write("# unseen, unlearnt test words in vocab: " + str(VOCDICT.num_unks()) + "\n")
sys.stderr.write("# unseen, unlearnt test POS tags:       " + str(POSDICT.num_unks()) + "\n")
sys.stderr.write("# unseen, unlearnt test lexical units:  " + str(LUDICT.num_unks()) + "\n")
sys.stderr.write("# unseen, unlearnt test LU pos tags:    " + str(LUPOSDICT.num_unks()) + "\n")
sys.stderr.write("# unseen, unlearnt test frames:         " + str(FRAMEDICT.num_unks()) + "\n")
sys.stderr.write("# unseen, unlearnt test FEs:            " + str(FEDICT.num_unks()) + "\n")
sys.stderr.write("# unseen, unlearnt test deprels:        " + str(DEPRELDICT.num_unks()) + "\n")
sys.stderr.write("# unseen, unlearnt test constit labels: " + str(CLABELDICT.num_unks()) + "\n\n")


# TODO: like the Google papers try different versions of these out in parallel
TOKDIM = 60
POSDIM = 4
if USE_WV:
    PRETDIM = len(wvs.values()[0])
LUDIM = 64
LUPOSDIM = 2
FRMDIM = 100
FEDIM = 50
INPDIM = TOKDIM + POSDIM + 1

if USE_CONSTITS:
    PHRASEDIM = 16
    PATHLSTMDIM = 64
    PATHDIM = 64

LSTMINPDIM = 64
LSTMDIM = 64
LSTMDEPTH = 1
HIDDENDIM = 64

ARGPOSDIM = ArgPosition.size()
SPANDIM = SpanWidth.size()

ALL_FEATS_DIM = 2 * LSTMDIM \
                + LUDIM \
                + LUPOSDIM \
                + FRMDIM \
                + LSTMINPDIM \
                + LSTMDIM \
                + FEDIM \
                + ARGPOSDIM \
                + SPANDIM \
                + 2  # spanlen and log spanlen features and is a constitspan

if USE_DEPS:
    DEPHEADDIM = LSTMINPDIM + POSDIM
    DEPRELDIM = 8
    OUTHEADDIM = OutHeads.size()

    PATHLSTMINPDIM = DEPHEADDIM + DEPRELDIM
    PATHLSTMDIM = 64
    PATHDIM = 64
    ALL_FEATS_DIM += OUTHEADDIM + PATHDIM

if USE_CONSTITS:
    ALL_FEATS_DIM += 1 + PHRASEDIM  # is a constit and what is it
    ALL_FEATS_DIM += PATHDIM

model = Model()
adam = AdamTrainer(model, 0.0005, 0.01, 0.9999, 1e-8)

v_x = model.add_lookup_parameters((VOCDICT.size(), TOKDIM))
p_x = model.add_lookup_parameters((POSDICT.size(), POSDIM))

lu_x = model.add_lookup_parameters((LUDICT.size(), LUDIM))
lp_x = model.add_lookup_parameters((LUPOSDICT.size(), LUPOSDIM))
frm_x = model.add_lookup_parameters((FRAMEDICT.size(), FRMDIM))
ap_x = model.add_lookup_parameters((ArgPosition.size(), ARGPOSDIM))
sp_x = model.add_lookup_parameters((SpanWidth.size(), SPANDIM))

if USE_DEPS:
    dr_x = model.add_lookup_parameters((DEPRELDICT.size(), DEPRELDIM))
    oh_s = model.add_lookup_parameters((OutHeads.size(), OUTHEADDIM))

if USE_CONSTITS:
    ct_x = model.add_lookup_parameters((CLABELDICT.size(), PHRASEDIM))

fe_x = model.add_lookup_parameters((FEDICT.size(), FEDIM))

if USE_WV:
    e_x = model.add_lookup_parameters((VOCDICT.size(), PRETDIM))
    for wordid in wvs:
        e_x.init_row(wordid, wvs[wordid])
    w_e = model.add_parameters((LSTMINPDIM, PRETDIM))
    b_e = model.add_parameters((LSTMINPDIM, 1))

w_i = model.add_parameters((LSTMINPDIM, INPDIM))
b_i = model.add_parameters((LSTMINPDIM, 1))

builders = [
    LSTMBuilder(LSTMDEPTH, LSTMINPDIM, LSTMDIM, model),
    LSTMBuilder(LSTMDEPTH, LSTMINPDIM, LSTMDIM, model),
]

basefwdlstm = LSTMBuilder(LSTMDEPTH, LSTMINPDIM, LSTMINPDIM, model)
baserevlstm = LSTMBuilder(LSTMDEPTH, LSTMINPDIM, LSTMINPDIM, model)

w_bi = model.add_parameters((LSTMINPDIM, 2 * LSTMINPDIM))
b_bi = model.add_parameters((LSTMINPDIM, 1))

tgtlstm = LSTMBuilder(LSTMDEPTH, LSTMINPDIM, LSTMDIM, model)
ctxtlstm = LSTMBuilder(LSTMDEPTH, LSTMINPDIM, LSTMDIM, model)

if USE_DEPS:
    w_di = model.add_parameters((LSTMINPDIM, LSTMINPDIM + DEPHEADDIM + DEPRELDIM))
    b_di = model.add_parameters((LSTMINPDIM, 1))

    pathfwdlstm = LSTMBuilder(LSTMDEPTH, LSTMINPDIM, PATHLSTMDIM, model)
    pathrevlstm = LSTMBuilder(LSTMDEPTH, LSTMINPDIM, PATHLSTMDIM, model)

    w_p = model.add_parameters((PATHDIM, 2 * PATHLSTMDIM))
    b_p = model.add_parameters((PATHDIM, 1))
elif USE_CONSTITS:
    cpathfwdlstm = LSTMBuilder(LSTMDEPTH, PHRASEDIM, PATHLSTMDIM, model)
    cpathrevlstm = LSTMBuilder(LSTMDEPTH, PHRASEDIM, PATHLSTMDIM, model)

    w_cp = model.add_parameters((PATHDIM, 2 * PATHLSTMDIM))
    b_cp = model.add_parameters((PATHDIM, 1))

w_z = model.add_parameters((HIDDENDIM, ALL_FEATS_DIM))
b_z = model.add_parameters((HIDDENDIM, 1))
w_f = model.add_parameters((1, HIDDENDIM))
b_f = model.add_parameters((1, 1))

if USE_PTB_CONSTITS:
    w_c = model.add_parameters((2, LSTMDIM))
    b_c = model.add_parameters((2, 1))
    w_fb = model.add_parameters((LSTMDIM, 2 * LSTMDIM))
    b_fb = model.add_parameters((LSTMDIM, 1))
    DELTA = len(trainexamples) * 1.0 / len(ptbexamples)
    sys.stderr.write("weighing PTB down by %f\n" % DELTA)


def get_base_embeddings(trainmode, unkdtokens, tg_start, sentence):
    pw_i = parameter(w_i)
    pb_i = parameter(b_i)

    pw_bi = parameter(w_bi)
    pb_bi = parameter(b_bi)

    sentlen = len(unkdtokens)
    # tfkeys = sorted(tfdict)
    # tg_start = tfkeys[0]

    if trainmode:
        emb_x = [noise(v_x[tok], 0.1) for tok in unkdtokens]
    else:
        emb_x = [v_x[tok] for tok in unkdtokens]
    pos_x = [p_x[pos] for pos in sentence.postags]
    dist_x = [scalarInput(i - tg_start + 1) for i in xrange(sentlen)]

    baseinp_x = [(pw_i * concatenate([emb_x[j], pos_x[j], dist_x[j]]) + pb_i) for j in xrange(sentlen)]

    if USE_WV:
        pw_e = parameter(w_e)
        pb_e = parameter(b_e)
        for j in xrange(sentlen):
            if unkdtokens[j] in wvs:
                nonupdatedwv = nobackprop(e_x[unkdtokens[j]])
                baseinp_x[j] = baseinp_x[j] + pw_e * nonupdatedwv + pb_e

    embposdist_x = [rectify(baseinp_x[j]) for j in xrange(sentlen)]

    if USE_DROPOUT:
        basefwdlstm.set_dropout(DROPOUT_RATE)
        baserevlstm.set_dropout(DROPOUT_RATE)
    bfinit = basefwdlstm.initial_state()
    basefwd = bfinit.transduce(embposdist_x)
    brinit = baserevlstm.initial_state()
    baserev = brinit.transduce(reversed(embposdist_x))
    basebi_x = [rectify(pw_bi * concatenate([basefwd[eidx], baserev[sentlen - eidx - 1]]) + pb_bi) for eidx in
                xrange(sentlen)]

    if USE_DEPS:
        pw_di = parameter(w_di)
        pb_di = parameter(b_di)

        dhead_x = [embposdist_x[dephead] for dephead in sentence.depheads]
        dheadp_x = [pos_x[dephead] for dephead in sentence.depheads]
        drel_x = [dr_x[deprel] for deprel in sentence.deprels]
        baseinp_x = [rectify(pw_di * concatenate([dhead_x[j], dheadp_x[j], drel_x[j], basebi_x[j]]) + pb_di) for j in
                     xrange(sentlen)]
        basebi_x = baseinp_x

    return basebi_x


def get_target_frame_embeddings(embposdist_x, tfdict):
    tfkeys = sorted(tfdict)
    tg_start = tfkeys[0]
    sentlen = len(embposdist_x)

    # adding target word feature
    lu, frame = tfdict[tg_start]
    tginit = tgtlstm.initial_state()
    target_x = tginit.transduce(embposdist_x[tg_start: tg_start + len(tfkeys) + 1])[-1]

    # adding context features
    ctxt = range(tg_start - 1, tfkeys[-1] + 2)
    if ctxt[0] < 0: ctxt = ctxt[1:]
    if ctxt[-1] > sentlen: ctxt = ctxt[:-1]
    c_init = ctxtlstm.initial_state()
    ctxt_x = c_init.transduce(embposdist_x[ctxt[0]:ctxt[-1]])[-1]

    # adding features specific to LU and frame
    lu_v = lu_x[lu.id]
    lp_v = lp_x[lu.posid]

    if USE_HIER and frame.id in frmrelmap:
        frame_v = esum([frm_x[frame.id]] + [frm_x[par] for par in frmrelmap[frame.id]])
    else:
        frame_v = frm_x[frame.id]
    tfemb = concatenate([lu_v, lp_v, frame_v, target_x, ctxt_x])

    return tfemb, frame


def get_span_embeddings(embpos_x):
    sentlen = len(embpos_x)
    fws = [[None for _ in xrange(sentlen)] for _ in xrange(sentlen)]
    bws = [[None for _ in xrange(sentlen)] for _ in xrange(sentlen)]

    if USE_DROPOUT:
        builders[0].set_dropout(DROPOUT_RATE)
        builders[1].set_dropout(DROPOUT_RATE)

    for i in xrange(sentlen):
        fw_init = builders[0].initial_state()
        tmpfws = fw_init.transduce(embpos_x[i:])
        if len(tmpfws) != sentlen - i:
            raise Exception("incorrect number of forwards", len(tmpfws), i, sentlen)

        spanend = sentlen
        if USE_SPAN_CLIP: spanend = min(sentlen, i + ALLOWED_SPANLEN + 1)
        for j in xrange(i, spanend):
            # for j in xrange(i, sentlen):
            fws[i][j] = tmpfws[j - i]

        bw_init = builders[1].initial_state()
        tmpbws = bw_init.transduce(reversed(embpos_x[:i + 1]))
        if len(tmpbws) != i + 1:
            raise Exception("incorrect number of backwards", i, len(tmpbws))
        spansize = i + 1
        if USE_SPAN_CLIP and spansize - 1 > ALLOWED_SPANLEN: spansize = ALLOWED_SPANLEN + 1
        for k in xrange(spansize):
            bws[i - k][i] = tmpbws[k]

    return fws, bws


def get_deppath_embeddings(sentence, embpos_x):
    spaths = {}
    for spath in set(sentence.shortest_paths.values()):
        shp = [embpos_x[node] for node in spath]
        if USE_DROPOUT:
            pathfwdlstm.set_dropout(DROPOUT_RATE)
            pathrevlstm.set_dropout(DROPOUT_RATE)
        pfinit = pathfwdlstm.initial_state()
        pathfwd = pfinit.transduce(shp)
        prinit = pathrevlstm.initial_state()
        pathrev = prinit.transduce(reversed(shp))

        pw_p = parameter(w_p)
        pb_p = parameter(b_p)
        pathlstm = rectify(pw_p * concatenate([pathfwd[-1], pathrev[-1]]) + pb_p)

        spaths[spath] = pathlstm
    return spaths


def get_cpath_embeddings(sentence):
    phrpaths = {}
    for phrpath in set(sentence.cpaths.values()):
        shp = [ct_x[node] for node in phrpath]
        if USE_DROPOUT:
            cpathfwdlstm.set_dropout(DROPOUT_RATE)
            cpathrevlstm.set_dropout(DROPOUT_RATE)
        cpfinit = cpathfwdlstm.initial_state()
        cpathfwd = cpfinit.transduce(shp)
        cprinit = cpathrevlstm.initial_state()
        cpathrev = cprinit.transduce(reversed(shp))

        pw_cp = parameter(w_cp)
        pb_cp = parameter(b_cp)
        cpathlstm = rectify(pw_cp * concatenate([cpathfwd[-1], cpathrev[-1]]) + pb_cp)

        phrpaths[phrpath] = cpathlstm
    return phrpaths


def get_factor_expressions(fws, bws, tfemb, tfdict, valid_fes, sentence, spaths_x=None, cpaths_x=None):
    pw_z = parameter(w_z)
    pb_z = parameter(b_z)
    pw_f = parameter(w_f)
    pb_f = parameter(b_f)

    factexprs = {}
    sentlen = len(fws)

    sortedtfd = sorted(tfdict.keys())
    targetspan = (sortedtfd[0], sortedtfd[-1])

    for j in xrange(sentlen):
        istart = 0
        if USE_SPAN_CLIP and j > ALLOWED_SPANLEN: istart = max(0, j - ALLOWED_SPANLEN)
        for i in xrange(istart, j + 1):

            spanlen = scalarInput(j - i + 1)
            logspanlen = scalarInput(math.log(j - i + 1))
            spanwidth = sp_x[SpanWidth.howlongisspan(i, j)]
            spanpos = ap_x[ArgPosition.whereisarg((i, j), targetspan)]

            fbemb_ij_basic = concatenate([fws[i][j], bws[i][j], tfemb, spanlen, logspanlen, spanwidth, spanpos])
            if USE_DEPS:
                outs = oh_s[OutHeads.getnumouts(i, j, sentence.outheads)]
                shp = spaths_x[sentence.shortest_paths[(i, j, targetspan[0])]]
                fbemb_ij = concatenate([fbemb_ij_basic, outs, shp])
            elif USE_CONSTITS:
                isconstit = scalarInput((i, j) in sentence.constitspans)
                lca = ct_x[sentence.lca[(i, j)][1]]
                phrp = cpaths_x[sentence.cpaths[(i, j, targetspan[0])]]
                fbemb_ij = concatenate([fbemb_ij_basic, isconstit, lca, phrp])
            else:
                fbemb_ij = fbemb_ij_basic

            for y in valid_fes:
                fctr = Factor(i, j, y)
                if USE_HIER and y in feparents:
                    fefixed = esum([fe_x[y]] + [fe_x[par] for par in feparents[y]])
                else:
                    fefixed = fe_x[y]
                # fefixed = nobackprop(fe_x[y])
                fbemb_ijy = concatenate([fefixed, fbemb_ij])
                factexprs[fctr] = pw_f * rectify(pw_z * fbemb_ijy + pb_z) + pb_f
                # if USE_DROPOUT:
                #     factexprs[fctr] = dropout(factexprs[fctr], DROPOUT_RATE)
    return factexprs


def denominator_check(n, k):
    ssum = [k]
    for _ in xrange(1, n):
        ssum.append(sum(ssum) * k + k)
    return ssum


def hamming_cost(factor, goldfactors):
    # print "hamming-cost"
    if factor in goldfactors:
        return scalarInput(0)
    return scalarInput(1)


def recall_oriented_cost(factor, goldfactors):
    alpha = RECALL_ORIENTED_COST
    beta = 1

    if factor in goldfactors:
        return scalarInput(0)
    i = factor.begin
    j = factor.end
    alphabetacost = 0
    if factor.label != NOTANFEID:
        alphabetacost += beta
    # find number of good gold factors it kicks out
    for gf in goldfactors:
        if i <= gf.begin <= j and gf.label != NOTANFEID:
            alphabetacost += alpha

    return scalarInput(alphabetacost)


def cost(factor, goldfactors):
    if options.cost == 'hamming':
        return hamming_cost(factor, goldfactors)
    elif options.cost == 'recall':
        return recall_oriented_cost(factor, goldfactors)
    else:
        raise Exception("undefined cost type", options.cost)


def get_logloss_partition(factorexprs, valid_fes, sentlen):
    logalpha = [None for _ in xrange(sentlen)]
    # ssum = lossformula(sentlen, len(valid_fes))
    for j in xrange(sentlen):
        # full length spans
        spanscores = []
        if not USE_SPAN_CLIP or j <= ALLOWED_SPANLEN:
            spanscores = [factorexprs[Factor(0, j, y)] for y in valid_fes]

        # recursive case
        istart = 0
        if USE_SPAN_CLIP and j > ALLOWED_SPANLEN: istart = max(0, j - ALLOWED_SPANLEN - 1)
        for i in xrange(istart, j):
            facscores = [logalpha[i] + factorexprs[Factor(i + 1, j, y)] for y in valid_fes]
            spanscores.extend(facscores)

        if not USE_SPAN_CLIP and len(spanscores) != len(valid_fes) * (j + 1):
            raise Exception("counting errors")
        logalpha[j] = logsumexp(spanscores)

    return logalpha[sentlen - 1]


def get_softmax_margin_partition(factorexprs, goldfactors, valid_fes, sentlen):
    logalpha = [None for _ in xrange(sentlen)]
    for j in xrange(sentlen):
        # full length spans
        spanscores = []
        if not USE_SPAN_CLIP or j <= ALLOWED_SPANLEN:
            spanscores = [factorexprs[Factor(0, j, y)]
                          + cost(Factor(0, j, y), goldfactors) for y in valid_fes]

        # recursive case
        istart = 0
        if USE_SPAN_CLIP and j > ALLOWED_SPANLEN: istart = max(0, j - ALLOWED_SPANLEN - 1)
        for i in xrange(istart, j):
            facscores = [logalpha[i]
                         + factorexprs[Factor(i + 1, j, y)]
                         + cost(Factor(i + 1, j, y), goldfactors) for y in valid_fes]
            spanscores.extend(facscores)

        if not USE_SPAN_CLIP and len(spanscores) != len(valid_fes) * (j + 1):
            raise Exception("counting errors")
        logalpha[j] = logsumexp(spanscores)

    return logalpha[sentlen - 1]


def get_hinge_partition(factorexprs, goldfacs, valid_fes, sentlen):
    alpha = [None for _ in xrange(sentlen)]
    backpointers = [None for _ in xrange(sentlen)]

    for j in xrange(sentlen):
        # full length spans
        bestscore = float("-inf")
        if not USE_SPAN_CLIP or j <= ALLOWED_SPANLEN:
            for y in valid_fes:
                factor = Factor(0, j, y)
                facscore = factorexprs[factor] + cost(factor, goldfacs)
                if facscore.scalar_value() > bestscore:
                    bestscore = facscore.scalar_value()
                    alpha[j] = facscore
                    backpointers[j] = (0, y)

        # recursive case
        istart = 0
        if USE_SPAN_CLIP and j > ALLOWED_SPANLEN: istart = max(0, j - ALLOWED_SPANLEN - 1)
        for i in xrange(istart, j):
            for y in valid_fes:
                factor = Factor(i + 1, j, y)
                facscore = alpha[i] + factorexprs[factor] + cost(factor, goldfacs)
                if facscore.scalar_value() > bestscore:
                    bestscore = facscore.scalar_value()
                    alpha[j] = facscore
                    backpointers[j] = (i + 1, y)

    predfactors = []
    j = sentlen - 1
    i = backpointers[j][0]
    while i >= 0:
        fe = backpointers[j][1]
        predfactors.append(Factor(i, j, fe))
        if i == 0:
            break
        j = i - 1
        i = backpointers[j][0]
    return alpha[sentlen - 1], predfactors


def get_hinge_loss(factorexprs, gold_fes, valid_fes, sentlen):
    goldfactors = [Factor(span[0], span[1], feid) for feid in gold_fes for span in gold_fes[feid]]
    numeratorexprs = [factorexprs[gf] for gf in goldfactors]
    numerator = esum(numeratorexprs)

    denominator, predfactors = get_hinge_partition(factorexprs, goldfactors, valid_fes, sentlen)

    if set(predfactors) == set(goldfactors):
        return None

    hingeloss = denominator - numerator
    if denominator.scalar_value() < numerator.scalar_value():
        raise Exception("ERROR: predicted cost less than gold!",
                        denominator.scalar_value(),
                        numerator.scalar_value(),
                        hingeloss.scalar_value())
    return hingeloss


def get_constit_loss(fws, bws, goldspans):
    if not USE_PTB_CONSTITS:
        raise Exception("should not be using the constit loss now!", USE_PTB_CONSTITS)

    if len(goldspans) == 0:
        return None, 0

    pw_fb = parameter(w_fb)
    pb_fb = parameter(b_fb)
    pw_c = parameter(w_c)
    pb_c = parameter(b_c)

    losses = []
    sentlen = len(fws)

    for j in xrange(sentlen):
        istart = 0
        if USE_SPAN_CLIP and j > ALLOWED_SPANLEN: istart = max(0, j - ALLOWED_SPANLEN)
        for i in xrange(istart, j + 1):
            constit_ij = pw_c * rectify(pw_fb * concatenate([fws[i][j], bws[i][j]]) + pb_fb) + pb_c
            logloss = log_softmax(constit_ij)

            isconstit = int((i, j) in goldspans)
            losses.append(pick(logloss, isconstit))

    ptbconstitloss = scalarInput(DELTA) * -esum(losses)
    numspanstagged = len(losses)
    return ptbconstitloss, numspanstagged


def get_loss(factorexprs, gold_fes, valid_fes, sentlen):
    if options.loss == 'hinge':
        return get_hinge_loss(factorexprs, gold_fes, valid_fes, sentlen)

    goldfactors = [Factor(span[0], span[1], feid) for feid in gold_fes for span in gold_fes[feid]]
    numeratorexprs = [factorexprs[gf] for gf in goldfactors]
    numerator = esum(numeratorexprs)

    if options.loss == 'log':
        partition = get_logloss_partition(factorexprs, valid_fes, sentlen)
    elif options.loss == 'softmaxm':
        partition = get_softmax_margin_partition(factorexprs, goldfactors, valid_fes, sentlen)
    else:
        raise Exception("undefined loss function", options.loss)

    lossexp = partition - numerator
    if partition.scalar_value() < numerator.scalar_value():
        sys.stderr.write("WARNING: partition ~~ numerator! possibly overfitting difference = %f\n"
                         % lossexp.scalar_value())
        return None
        # raise Exception("ERROR: partition shouldn't exceeed numerator!",
        #                 partition.scalar_value(),
        #                 numerator.scalar_value(),
        #                 lossexp.scalar_value())

    if lossexp.scalar_value() < 0.0:
        sys.stderr.write(str(gold_fes) + "\ngolds\n")
        gsum = 0
        for fac in goldfactors:
            gsum += factorexprs[fac].scalar_value()
            sys.stderr.write(fac.to_str(FEDICT) + " " + str(factorexprs[fac].scalar_value()) + "\n")
        sys.stderr.write("my calculation = " + str(gsum) + " vs " + str(numerator.scalar_value()) + "\n")
        for j in xrange(sentlen):
            sys.stderr.write(":" + str(j) + "\t")
            if not USE_SPAN_CLIP or j <= ALLOWED_SPANLEN:
                sys.stderr.write("0 ")
            istart = 0
            if USE_SPAN_CLIP and j > ALLOWED_SPANLEN: istart = max(0, j - ALLOWED_SPANLEN - 1)
            for i in xrange(istart, j):
                sys.stderr.write(str(i + 1) + " ")
            sys.stderr.write("\n")
        raise Exception("negative probability! probably overcounting spans?",
                        numerator.scalar_value(),
                        partition.scalar_value(),
                        lossexp.scalar_value())
    return lossexp


def decode(factexprscalars, sentlen, valid_fes):
    alpha = [None for _ in xrange(sentlen)]
    backpointers = [None for _ in xrange(sentlen)]
    # predfactors = []
    if USE_DROPOUT:
        raise Exception("incorrect usage of dropout, turn off!\n")

    for j in xrange(sentlen):
        if USE_SPAN_CLIP and j > ALLOWED_SPANLEN: continue
        bestscore = float("-inf")
        bestlabel = None
        for y in valid_fes:
            fac = Factor(0, j, y)
            facscore = math.exp(factexprscalars[fac])
            # facscore = exp(factexprscalars[fac]).scalar_value()
            if facscore > bestscore:
                bestscore = facscore
                bestlabel = y
        alpha[j] = bestscore
        backpointers[j] = (0, bestlabel)

    for j in xrange(sentlen):
        bestscore = float("-inf")
        bestbeg = bestlabel = None
        if alpha[j] is not None:
            bestscore = alpha[j]
            bestbeg, bestlabel = backpointers[j]

        istart = 0
        if USE_SPAN_CLIP and j > ALLOWED_SPANLEN: istart = max(0, j - ALLOWED_SPANLEN - 1)
        for i in xrange(istart, j):
            for y in valid_fes:
                fac = Factor(i + 1, j, y)
                # facscore = exp(factexprscalars[fac]).scalar_value()
                facscore = math.exp(factexprscalars[fac])
                if facscore * alpha[i] > bestscore:
                    bestscore = facscore * alpha[i]
                    bestlabel = y
                    bestbeg = i + 1
        alpha[j] = bestscore
        backpointers[j] = (bestbeg, bestlabel)

    j = sentlen - 1
    i = backpointers[j][0]
    argmax = {}
    while i >= 0:
        fe = backpointers[j][1]  # FrameElement(backpointers[j][1])
        # predfactors.append(Factor(i, j, fe))
        if fe in argmax:
            argmax[fe].append((i, j))
        else:
            argmax[fe] = [(i, j)]
        if i == 0:
            break
        j = i - 1
        i = backpointers[j][0]

    # merging neighboring spans in prediction (to combat spurious ambiguity)
    mergedargmax = {}
    for fe in argmax:
        mergedargmax[fe] = []
        if fe == NOTANFEID:
            mergedargmax[fe].extend(argmax[fe])
            continue

        argmax[fe].sort()
        mergedspans = [argmax[fe][0]]
        for span in argmax[fe][1:]:
            prevsp = mergedspans[-1]
            if span[0] == prevsp[1] + 1:
                prevsp = mergedspans.pop()
                mergedspans.append((prevsp[0], span[1]))
            else:
                mergedspans.append(span)
        mergedargmax[fe] = mergedspans
    return mergedargmax


def identify_fes(unkdtoks, sentence, tfdict, goldfes=None, testidx=None):
    renew_cg()
    trainmode = (goldfes is not None)

    global USE_DROPOUT
    if options.dropout: USE_DROPOUT = trainmode

    # pw_f = zeroes((1, HIDDENDIM))
    # pb_f = zeroes((1, 1))

    sentlen = len(unkdtoks)
    tfkeys = sorted(tfdict)
    tg_start = tfkeys[0]

    embpos_x = get_base_embeddings(trainmode, unkdtoks, tg_start, sentence)
    tfemb, frame = get_target_frame_embeddings(embpos_x, tfdict)

    fws, bws = get_span_embeddings(embpos_x)
    valid_fes = frmfemap[frame.id] + [NOTANFEID]
    if USE_DEPS:
        spaths_x = get_deppath_embeddings(sentence, embpos_x)
        factor_exprs = get_factor_expressions(fws, bws, tfemb, tfdict, valid_fes, sentence, spaths_x=spaths_x)
    elif USE_CONSTITS:
        cpaths_x = get_cpath_embeddings(sentence)
        factor_exprs = get_factor_expressions(fws, bws, tfemb, tfdict, valid_fes, sentence, cpaths_x=cpaths_x)
    else:
        factor_exprs = get_factor_expressions(fws, bws, tfemb, tfdict, valid_fes, sentence)

    if trainmode:
        segrnnloss = get_loss(factor_exprs, goldfes, valid_fes, sentlen)
        if USE_PTB_CONSTITS:
            goldspans = []
            for feid in goldfes:
                if feid == NOTANFEID: continue
                goldspans.extend(goldfes[feid])

            constitloss, numspans = get_constit_loss(fws, bws, goldspans)
            if segrnnloss is not None and constitloss is not None:
                # segrnnloss of 1 segmentation vs all, globally normalized
                return segrnnloss + constitloss, 1 + numspans
            elif segrnnloss is None:
                return constitloss, numspans
        return segrnnloss, 1  # segrnnloss of 1 segmentation vs all, globally normalized
    else:
        if SAVE_FOR_ENSEMBLE:
            outensapf = open(out_ens_file, "a")
            for fact in factor_exprs:
                outensapf.write(
                    str(testidx) + "\t"
                    + fact.to_str(FEDICT) + "\t"
                    + str((factor_exprs[fact]).scalar_value())
                    + "\n")
            outensapf.close()
        facexprscalars = {fact: factor_exprs[fact].scalar_value() for fact in factor_exprs}
        argmax = decode(facexprscalars, sentlen, valid_fes)
        return argmax


def identify_spans(unkdtoks, sentence, goldspans):
    renew_cg()

    embpos_x = get_base_embeddings(True, unkdtoks, 0, sentence)
    fws, bws = get_span_embeddings(embpos_x)

    return get_constit_loss(fws, bws, goldspans)


def print_result(golds, pred_targmaps):
    with codecs.open(out_conll_file, "w", "utf-8") as f:
        for gold, pred in zip(golds, pred_targmaps):
            result = gold.get_str(predictedfes=pred)
            f.write(result + "\n")
        f.close()


def print_eval_result(examples, expredictions):
    evalstarttime = time.time()
    corp_up, corp_ur, corp_uf, \
    corp_lp, corp_lr, corp_lf, \
    corp_wp, corp_wr, corp_wf, \
    corp_ures, corp_labldres, corp_tokres = evaluate_corpus_argid(
        examples, expredictions, corefrmfemap, NOTANFEID)

    sys.stderr.write("\n[test] wpr = %.5f (%.1f/%.1f) wre = %.5f (%.1f/%.1f)\n"
                     "[test] upr = %.5f (%.1f/%.1f) ure = %.5f (%.1f/%.1f)\n"
                     "[test] lpr = %.5f (%.1f/%.1f) lre = %.5f (%.1f/%.1f)\n"
                     "[test] wf1 = %.5f uf1 = %.5f lf1 = %.5f [took %.3f s]\n"
                     % (corp_wp, corp_tokres[0], corp_tokres[1] + corp_tokres[0],
                        corp_wr, corp_tokres[0], corp_tokres[-1] + corp_tokres[0],
                        corp_up, corp_ures[0], corp_ures[1] + corp_ures[0],
                        corp_ur, corp_ures[0], corp_ures[-1] + corp_ures[0],
                        corp_lp, corp_labldres[0], corp_labldres[1] + corp_labldres[0],
                        corp_lr, corp_labldres[0], corp_labldres[-1] + corp_labldres[0],
                        corp_wf, corp_uf, corp_lf,
                        time.time() - evalstarttime))


# main
NUMEPOCHS = 1000
LOSS_EVAL_EPOCH = 100
DEV_EVAL_EPOCHS = 10 * LOSS_EVAL_EPOCH

if options.mode in ['test', 'refresh']:
    sys.stderr.write("reusing " + options.modelfile + "...\n")
    model.populate(options.modelfile)

if options.mode in ['train', 'refresh']:
    tagged = loss = 0.0
    bestdevf = 0.0

    if USE_PTB_CONSTITS:
        trainexamples = trainexamples + ptbexamples

    starttime = time.time()
    for epoch in xrange(NUMEPOCHS):
        random.shuffle(trainexamples)

        for idx, trex in enumerate(trainexamples, 1):
            if (idx - 1) % LOSS_EVAL_EPOCH == 0 and tagged > 0:
                adam.status()
                sys.stderr.write("%d loss = %f [took %.3f s]\n" % (idx - 1, (loss / tagged), time.time() - starttime))
                starttime = time.time()
                tagged = 0.0
                loss = 0.0

            unkedtoks = []
            unk_replace_tokens(trex.tokens, unkedtoks, VOCDICT, UNK_PROB, UNKTOKEN)

            if USE_PTB_CONSTITS and type(trex) == Sentence:  # a PTB example
                trexloss, taggedinex = identify_spans(unkedtoks,
                                                      trex,
                                                      trex.constitspans.keys())
            else:  # an FN example
                trexloss, taggedinex = identify_fes(unkedtoks,
                                                    trex.sentence,
                                                    trex.targetframedict,
                                                    goldfes=trex.invertedfes)
                # totnumspans = sum([len(trex.invertedfes[fi]) for fi in trex.invertedfes])
            tagged += taggedinex

            if trexloss is not None:
                loss += trexloss.scalar_value()
                trexloss.backward()
                adam.update()

            if (idx - 1) % DEV_EVAL_EPOCHS == 0 and idx > 1:
                devstarttime = time.time()
                ures = labldres = tokenwise = [0.0, 0.0, 0.0]
                predictions = []

                for devex in devexamples:

                    dargmax = identify_fes(devex.tokens,
                                           devex.sentence,
                                           devex.targetframedict)
                    if devex.frame.id in corefrmfemap:
                        corefes = corefrmfemap[devex.frame.id]
                    else:
                        corefes = {}
                    u, l, t = evaluate_example_argid(devex.invertedfes, dargmax, corefes, len(devex.tokens), NOTANFEID)
                    ures = np.add(ures, u)
                    labldres = np.add(labldres, l)
                    tokenwise = np.add(tokenwise, t)

                    predictions.append(dargmax)

                up, ur, uf = calc_f(ures)
                lp, lr, lf = calc_f(labldres)
                wp, wr, wf = calc_f(tokenwise)
                sys.stderr.write("[dev epoch=%d after=%d] wprec = %.5f wrec = %.5f wf1 = %.5f\n"
                                 "[dev epoch=%d after=%d] uprec = %.5f urec = %.5f uf1 = %.5f\n"
                                 "[dev epoch=%d after=%d] lprec = %.5f lrec = %.5f lf1 = %.5f"
                                 % (epoch, idx, wp, wr, wf, epoch, idx, up, ur, uf, epoch, idx, lp, lr, lf))

                if lf > bestdevf:
                    bestdevf = lf
                    print_result(devexamples, predictions)
                    sys.stderr.write(" -- saving")

                    model.save(modelfname)
                    os.symlink(modelfname, "tmp.link")
                    os.rename("tmp.link", MODELSYMLINK)
                sys.stderr.write(" [took %.3f s]\n" % (time.time() - devstarttime))
                starttime = time.time()
        adam.update_epoch(1.0)

elif options.mode == "ensemble":
    exfs = {x: {} for x in xrange(len(devexamples))}
    USE_DROPOUT = False

    sys.stderr.write("reading ensemble factors...")
    enf = open(in_ens_file, "rb")
    for l in enf:
        fields = l.split("\t")
        fac = Factor(int(fields[1]), int(fields[2]), FEDICT.getid(fields[3]))
        exfs[int(fields[0])][fac] = float(fields[4])
    enf.close()

    sys.stderr.write("done!\n")
    teststarttime = time.time()
    sys.stderr.write("testing " + str(len(devexamples)) + " examples ...\n")

    testpredictions = []
    for tidx, testex in enumerate(devexamples, 1):
        if tidx % 100 == 0:
            sys.stderr.write(str(tidx) + "...")
        valid_fes_for_frame = frmfemap[testex.frame.id] + [NOTANFEID]
        testargmax = decode(exfs[tidx - 1], len(testex.tokens), valid_fes_for_frame)
        testpredictions.append(testargmax)

    sys.stderr.write(" [took %.3f s]\n" % (time.time() - teststarttime))
    sys.stderr.write("printing output conll to " + out_conll_file + " ... ")
    print_result(devexamples, testpredictions)
    sys.stderr.write("done!\n")
    print_eval_result(devexamples, testpredictions)
    sys.stderr.write("printing frame-elements to " + options.fefile + " ...\n")
    convert_conll_to_frame_elements(out_conll_file, options.fefile)
    sys.stderr.write("done!\n")

elif options.mode == "test":
    if SAVE_FOR_ENSEMBLE:
        outensf = open(out_ens_file, "w")
        outensf.close()

    sys.stderr.write("testing " + str(len(devexamples)) + " examples ...\n")
    teststarttime = time.time()

    testpredictions = []
    for tidx, testex in enumerate(devexamples, 1):
        if tidx % 100 == 0:
            sys.stderr.write(str(tidx) + "...")
        testargmax = identify_fes(testex.tokens,
                                  testex.sentence,
                                  testex.targetframedict,
                                  testidx=tidx - 1)
        testpredictions.append(testargmax)

    sys.stderr.write(" [took %.3f s]\n" % (time.time() - teststarttime))
    sys.stderr.write("printing output conll to " + out_conll_file + " ... ")
    print_result(devexamples, testpredictions)
    sys.stderr.write("done!\n")
    print_eval_result(devexamples, testpredictions)
    sys.stderr.write("printing frame-elements to " + options.fefile + " ...\n")
    convert_conll_to_frame_elements(out_conll_file, options.fefile)
    sys.stderr.write("done!\n")
