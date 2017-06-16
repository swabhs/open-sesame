# -*- coding: utf-8 -*-
from evaluation import *
from dynet import *
from arksemaforeval import *

import os
import sys
import time
from optparse import OptionParser

MODELSYMLINK = "model.frameid." + VERSION
modelfname = "tmp/" + VERSION  + "model-" + str(time.time())

# TODO use optparse
optpr = OptionParser()
optpr.add_option("--mode", dest="mode", type='choice', choices=['train', 'test', 'refresh'], default='train')
optpr.add_option('--model', dest="modelfile", help="Saved model file", metavar="FILE", default=MODELSYMLINK)
optpr.add_option("--nodrop", action="store_true", default=False)
optpr.add_option("--nowordvec", action="store_true", default=False)
optpr.add_option("--hier", action="store_true", default=False)
optpr.add_option("--exemplar", action="store_true", default=False)

(options, args) = optpr.parse_args()

if options.exemplar:
    train_conll = TRAIN_EXEMPLAR
else:
    train_conll = TRAIN_FTE

USE_DROPOUT = not options.nodrop
USE_WV = not options.nowordvec
USE_HIER = options.hier

sys.stderr.write("\nCOMMAND: " + ' '.join(sys.argv) + "\n")
sys.stderr.write("\nPARSER SETTINGS\n_____________________\n")
sys.stderr.write("PARSING MODE:   \t" + options.mode + "\n")
sys.stderr.write("USING EXEMPLAR? \t" + str(options.exemplar) + "\n")
sys.stderr.write("USING DROPOUT?  \t" + str(USE_DROPOUT) + "\n")
sys.stderr.write("USING WORDVECS? \t" + str(USE_WV) + "\n")
sys.stderr.write("USING HIERARCHY?\t" + str(USE_HIER) + "\n")
if options.mode in ["train", "refresh"]:
    sys.stderr.write("MODEL WILL BE SAVED TO\t%s\n" %modelfname)
sys.stderr.write("_____________________\n")

UNK_PROB = 0.1
DROPOUT_RATE = 0.01

TOKDIM = 60
POSDIM = 4
LUDIM = 64
LPDIM = 5
INPDIM = TOKDIM + POSDIM

LSTMINPDIM = 64
LSTMDIM = 64
LSTMDEPTH = 2
HIDDENDIM = 64


def find_multitokentargets(examples, split):
    multitoktargs = tottargs = 0.0
    for tr in examples:
        tottargs += 1
        if len(tr.targetframedict) > 1:
            # tr.print_internal()
            multitoktargs += 1
            tfs = set(tr.targetframedict.values())
            if len(tfs) > 1:
                raise Exception("different frames for neighboring targets!", tr.targetframedict)
    sys.stderr.write("multi-token targets in %s: %.3f%% [%d / %d]\n"
                     %(split, multitoktargs*100/tottargs, multitoktargs, tottargs))

trainexamples, m, t = read_conll(train_conll)

post_train_lock_dicts()
lufrmmap, relatedlus = read_related_lus()
if USE_WV:
    wvs = get_wvec_map()
    PRETDIM = len(wvs.values()[0])
    sys.stderr.write("using pretrained embeddings of dimension " + str(PRETDIM) + "\n")


lock_dicts() 
UNKTOKEN = VOCDICT.getid(UNK)

sys.stderr.write("# words in vocab: " + str(VOCDICT.size()) + "\n")
sys.stderr.write("# POS tags: " + str(POSDICT.size()) + "\n")
sys.stderr.write("# lexical units: " + str(LUDICT.size()) + "\n")
sys.stderr.write("# LU POS tags: " + str(LUPOSDICT.size()) + "\n")
sys.stderr.write("# frames: " + str(FRAMEDICT.size()) + "\n")

if options.mode in ["train", "refresh"]:
    devexamples, m, t = read_conll(DEV_CONLL)
    sys.stderr.write("unknowns in dev\n\n_____________________\n")
    out_conll_file = "predicted." + VERSION + ".frameid.dev.out"
elif options.mode  == "test":
    devexamples, m, t = read_conll(TEST_CONLL)
    sys.stderr.write("unknowns in test\n\n_____________________\n")
    out_conll_file = "predicted." + VERSION + ".frameid.test.out"
    fefile = "my.predict.test.frame.elements"
else:
    raise Exception("invalid parser mode", options.mode)

find_multitokentargets(trainexamples, "train")
find_multitokentargets(devexamples, "dev/test")

sys.stderr.write("# unseen, unlearnt test words in vocab: " + str(VOCDICT.num_unks()) + "\n")
sys.stderr.write("# unseen, unlearnt test POS tags: " + str(POSDICT.num_unks()) + "\n")
sys.stderr.write("# unseen, unlearnt test lexical units: " + str(LUDICT.num_unks()) + "\n")
sys.stderr.write("# unseen, unlearnt test LU pos tags: " + str(LUPOSDICT.num_unks()) + "\n")
sys.stderr.write("# unseen, unlearnt test frames: " + str(FRAMEDICT.num_unks()) + "\n\n")

# sys.exit()

model = Model()
adam = SimpleSGDTrainer(model)
# adam = AdamTrainer(model, 0.0001, 0.01, 0.9999, 1e-8)

v_x = model.add_lookup_parameters((VOCDICT.size(), TOKDIM))
p_x = model.add_lookup_parameters((POSDICT.size(), POSDIM))
lu_x = model.add_lookup_parameters((LUDICT.size(), LUDIM))
lp_x = model.add_lookup_parameters((LUPOSDICT.size(), LPDIM))
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

tlstm = LSTMBuilder(LSTMDEPTH, 2*LSTMDIM, LSTMDIM, model)

w_z = model.add_parameters((HIDDENDIM, LSTMDIM + LUDIM + LPDIM))
b_z = model.add_parameters((HIDDENDIM, 1))
w_f = model.add_parameters((FRAMEDICT.size(), HIDDENDIM))
b_f = model.add_parameters((FRAMEDICT.size(), 1))

def identify_frames(builders, tokens, postags, lexunit, targetpositions, goldframe=None):
    renew_cg()
    trainmode = (goldframe is not None)

    sentlen = len(tokens) - 1
    emb_x = [v_x[tok] for tok in tokens]
    pos_x = [p_x[pos] for pos in postags]

    pw_i = parameter(w_i)
    pb_i = parameter(b_i)

    emb2_xi = [(pw_i * concatenate([emb_x[i], pos_x[i]])  + pb_i) for i in xrange(sentlen+1)]
    if USE_WV:
        pw_e = parameter(w_e)
        pb_e = parameter(b_e)
        for i in xrange(sentlen+1):
            if tokens[i] in wvs:
                nonupdatedwv = e_x[tokens[i]]  # prevent the wvecs from being updated
                emb2_xi[i] = emb2_xi[i] + pw_e * nonupdatedwv + pb_e

    emb2_x = [rectify(emb2_xi[i]) for i in xrange(sentlen+1)]

    pw_z = parameter(w_z)
    pb_z = parameter(b_z)
    pw_f = parameter(w_f)
    pb_f = parameter(b_f)

    # initializing the two LSTMs
    if USE_DROPOUT and trainmode:
        builders[0].set_dropout(DROPOUT_RATE)
        builders[1].set_dropout(DROPOUT_RATE)
    f_init, b_init = [i.initial_state() for i in builders]

    fw_x = f_init.transduce(emb2_x)
    bw_x = b_init.transduce(reversed(emb2_x))

    # only using the first target position - summing them hurts :(
    targetembs = [concatenate([fw_x[targetidx], bw_x[sentlen - targetidx - 1]]) for targetidx in targetpositions]
    targinit = tlstm.initial_state()
    target_vec = targinit.transduce(targetembs)[-1]

    valid_frames = list(lufrmmap[lexunit.id])
    chosenframe = valid_frames[0]
    logloss = None
    if len(valid_frames) > 1:
        if USE_HIER and lexunit.id in relatedlus:
            lu_vec = esum([lu_x[luid] for luid in relatedlus[lexunit.id]])
        else:
            lu_vec = lu_x[lexunit.id]
        fbemb_i = concatenate([target_vec, lu_vec, lp_x[lexunit.posid]])
        # TODO add more Baidu-style features here
        f_i = pw_f * rectify(pw_z * fbemb_i + pb_z) + pb_f
        if trainmode and USE_DROPOUT:
            f_i = dropout(f_i, DROPOUT_RATE)

        logloss = log_softmax(f_i, valid_frames)

        if not trainmode:
            chosenframe = np.argmax(logloss.npvalue())

    if trainmode: chosenframe = goldframe.id

    losses = []
    if logloss is not None:
        losses.append(pick(logloss, chosenframe))

    prediction = {tidx: (lexunit, Frame(chosenframe)) for tidx in targetpositions}

    objective = -esum(losses) if losses else None
    return objective, prediction

def print_result(goldexamples, pred_targmaps):
    with codecs.open(out_conll_file, "w", "utf-8") as f:
        for g,p in zip(goldexamples, pred_targmaps):
            result = g.get_newstr_frame(p) + "\n"
            f.write(result)
        f.close()


# main
NUMEPOCHS = 10
if options.exemplar: NUMEPOCHS = 25
EVAL_EVERY_EPOCH = 100
DEV_EVAL_EPOCH = 5 * EVAL_EVERY_EPOCH

if options.mode in ["train", "refresh"]:
    tagged = loss = 0.0
    bestdevf = 0.0

    for epoch in xrange(NUMEPOCHS):
        random.shuffle(trainexamples)
        for idx, trex in enumerate(trainexamples, 1):
            if idx % EVAL_EVERY_EPOCH == 0:
                adam.status()
                sys.stderr.write("%d loss = %.6f\n" %(idx, loss/tagged))
                tagged = loss = 0.0
            inptoks = []
            unk_replace_tokens(trex.tokens, inptoks, VOCDICT, UNK_PROB, UNKTOKEN)

            trexloss,_ = identify_frames(
                builders, inptoks, trex.postags, trex.lu, trex.targetframedict.keys(), trex.frame)

            if trexloss is not None:
                loss += trexloss.scalar_value()
                trexloss.backward()
                adam.update()
            tagged += 1

            if idx % DEV_EVAL_EPOCH == 0:
                corpus_result = [0.0, 0.0, 0.0]
                devtagged = devloss = 0.0
                predictions = []
                for devex in devexamples:
                    devludict = devex.get_only_targets()
                    dl, predicted = identify_frames(
                        builders, devex.tokens, devex.postags, devex.lu, devex.targetframedict.keys())
                    if dl is not None:
                        devloss += dl.scalar_value()
                    predictions.append(predicted)

                    devex_result = evaluate_example_frameid(devex.frame, predicted)
                    corpus_result = np.add(corpus_result, devex_result)
                    devtagged += 1

                devp, devr, devf = calc_f(corpus_result)
                devtp, devfp, devfn = corpus_result
                sys.stderr.write("[dev epoch=%d] loss = %.6f "
                                 "p = %.4f (%.1f/%.1f) r = %.4f (%.1f/%.1f) f1 = %.4f"
                                 % (epoch, devloss/devtagged,
                                    devp, devtp, devtp + devfp,
                                    devr, devtp, devtp + devfn,
                                    devf))
                if devf > bestdevf:
                    bestdevf = devf
                    print_result(devexamples, predictions)
                    sys.stderr.write(" -- saving")

                    model.save(modelfname)
                    os.symlink(modelfname, "tmp.link")
                    os.rename("tmp.link", MODELSYMLINK)
                sys.stderr.write("\n")
        adam.update_epoch(1.0)

elif options.mode == "test":
    model.load(options.modelfile)
    corpus_tpfpfn = [0.0, 0.0, 0.0]

    testpredictions = []

    sn = devexamples[0].sent_num
    sl = [0.0,0.0,0.0]
    print("Sent#%d :" % sn)
    devexamples[0].print_internal_sent()

    for testex in devexamples:
        _, predicted = identify_frames(builders, testex.tokens, testex.postags, testex.lu, testex.targetframedict.keys())

        tpfpfn = evaluate_example_frameid(testex.frame, predicted)
        corpus_tpfpfn = np.add(corpus_tpfpfn, tpfpfn)

        testpredictions.append(predicted)

        sentnum = testex.sent_num
        if sentnum != sn:
            lp, lr, lf = calc_f(sl)
            print("\t\t\t\t\t\t\t\t\tTotal: %.1f / %.1f / %.1f\n"
                  "Sentence ID=%d: Recall=%.5f (%.1f/%.1f) Precision=%.5f (%.1f/%.1f) Fscore=%.5f"
                  "\n-----------------------------\n"
                  % (sl[0], sl[0]+sl[1], sl[0]+sl[-1],
                     sn,
                     lr, sl[0], sl[-1] + sl[0],
                     lp, sl[0], sl[1] + sl[0],
                     lf))
            sl = [0.0,0.0,0.0]
            sn = sentnum
            print("Sent#%d :" % sentnum)
            testex.print_internal_sent()

        print "gold:"
        testex.print_internal_frame()
        print "prediction:"
        testex.print_external_frame(predicted)

        sl = np.add(sl, tpfpfn)
        print tpfpfn[0], "/", tpfpfn[0]+tpfpfn[1], "/", tpfpfn[0]+tpfpfn[-1]

    # last sentence
    lp, lr, lf = calc_f(sl)
    print("\t\t\t\t\t\t\t\t\tTotal: %.1f / %.1f / %.1f\n"
          "Sentence ID=%d: Recall=%.5f (%.1f/%.1f) Precision=%.5f (%.1f/%.1f) Fscore=%.5f"
          "\n-----------------------------\n"
          % (sl[0], sl[0]+sl[1], sl[0]+sl[-1],
             sentnum,
             lp, sl[0], sl[1] + sl[0],
             lr, sl[0], sl[-1] + sl[0],
             lf))

    testp, testr, testf = calc_f(corpus_tpfpfn)
    testtp, testfp, testfn = corpus_tpfpfn
    sys.stderr.write("[test] p = %.4f (%.1f/%.1f) r = %.4f (%.1f/%.1f) f1 = %.4f\n" %(
        testp, testtp, testtp + testfp,
        testr, testtp, testtp + testfp,
        testf))

    sys.stderr.write("printing output conll to " + out_conll_file + " ... ")
    print_result(devexamples, testpredictions)
    sys.stderr.write("done!\n")

    sys.stderr.write("printing frame-elements to " + fefile + " ...\n")
    convert_conll_to_frame_elements(out_conll_file, fefile)
    sys.stderr.write("done!\n")
