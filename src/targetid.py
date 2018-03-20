# -*- coding: utf-8 -*-
from evaluation import *
from dynet import *
from arksemaforeval import *

import os
import sys
import time
from optparse import OptionParser

MODELSYMLINK = "model.targetid." + VERSION
modelfname = "tmp/" + VERSION  + "model-" + str(time.time())

# TODO(swabha): use optparse
optpr = OptionParser()
optpr.add_option("--mode", dest="mode", type='choice', choices=['train', 'test', 'refresh'], default='train')
optpr.add_option('--model', dest="modelfile", help="Saved model file", metavar="FILE", default=MODELSYMLINK)
optpr.add_option("--nodrop", action="store_true", default=False)
optpr.add_option("--nowordvec", action="store_true", default=False)

(options, args) = optpr.parse_args()

train_conll = TRAIN_FTE

USE_DROPOUT = not options.nodrop
USE_WV = not options.nowordvec

sys.stderr.write("\nCOMMAND: " + ' '.join(sys.argv) + "\n")
sys.stderr.write("\nPARSER SETTINGS\n_____________________\n")
sys.stderr.write("PARSING MODE:   \t" + options.mode + "\n")
sys.stderr.write("USING DROPOUT?  \t" + str(USE_DROPOUT) + "\n")
sys.stderr.write("USING WORDVECS? \t" + str(USE_WV) + "\n")
if options.mode in ["train", "refresh"]:
    sys.stderr.write("MODEL WILL BE SAVED TO\t%s\n" %modelfname)
sys.stderr.write("_____________________\n")

UNK_PROB = 0.1
DROPOUT_RATE = 0.01

TOKDIM = 60
POSDIM = 4
LEMDIM = 64

INPDIM = TOKDIM + POSDIM + LEMDIM

LSTMINPDIM = 64
LSTMDIM = 64
LSTMDEPTH = 2
HIDDENDIM = 64


def combine_examples(corpus_ex):
    combined_ex = [corpus_ex[0]]
    for ex in corpus_ex[1:]:
        if ex.sent_num == combined_ex[-1].sent_num:
            current_sent = combined_ex.pop()
            target_frame_dict = current_sent.targetframedict.copy()   
            target_frame_dict.update(ex.targetframedict)
            current_sent.targetframedict = target_frame_dict
            combined_ex.append(current_sent)
            continue
        combined_ex.append(ex)
    return combined_ex

trainexamples, m, t = read_conll(train_conll)
train = combine_examples(trainexamples)
print(len(trainexamples), "reduced to ", len(train))

post_train_lock_dicts()

target_lu_map = create_target_lu_map()

if USE_WV:
    wvs = get_wvec_map()
    PRETDIM = len(wvs.values()[0])
    sys.stderr.write("using pretrained embeddings of dimension " + str(PRETDIM) + "\n")

lock_dicts() 
UNKTOKEN = VOCDICT.getid(UNK)

sys.stderr.write("# words in vocab: " + str(VOCDICT.size()) + "\n")
sys.stderr.write("# POS tags: " + str(POSDICT.size()) + "\n")
sys.stderr.write("# lemmas: " + str(LEMDICT.size()) + "\n")

if options.mode in ["train", "refresh"]:
    devexamples, m, t = read_conll(DEV_CONLL)
    dev = combine_examples(devexamples)
    sys.stderr.write("unknowns in dev\n\n_____________________\n")
    out_conll_file = "predicted." + VERSION + ".targetid.dev.out"
elif options.mode  == "test":
    devexamples, m, t = read_conll(TEST_CONLL)
    dev = combine_examples(devexamples)
    sys.stderr.write("unknowns in test\n\n_____________________\n")
    out_conll_file = "predicted." + VERSION + ".targetid.test.out"
    fefile = "my.predict.test.frame.elements"
else:
    raise Exception("invalid parser mode", options.mode)

sys.stderr.write("# unseen, unlearnt test words in vocab: " + str(VOCDICT.num_unks()) + "\n")
sys.stderr.write("# unseen, unlearnt test POS tags: " + str(POSDICT.num_unks()) + "\n")
sys.stderr.write("# unseen, unlearnt test lemmas: " + str(LEMDICT.num_unks()) + "\n")

# def check_if_in_index(data_split):
#     problematic = []
#     total = 0
#     pos_lupos_map = {}

#     for ex in data_split:
#         for target in ex.targetframedict:
#             token = VOCDICT.getstr(ex.tokens[target]).lower()
#             lemma = LEMDICT.getstr(ex.lemmas[target]).lower()
#             total += 1
#             pos = POSDICT.getstr(ex.postags[target]).lower()
#             if token not in target_lu_map and lemma not in target_lu_map:
#                 # sys.stderr.write(token + " ")
#                 problematic.append((token, lemma))
#             else:
#                 if pos not in pos_lupos_map:
#                     pos_lupos_map[pos] = set([])
                
#                 if token in target_lu_map:
#                     for lu in target_lu_map[token]:
#                         pos_lupos_map[pos].add(lu.split('.')[1])
#                 else:
#                     for lu in target_lu_map[lemma]:
#                         pos_lupos_map[pos].add(lu.split('.')[1])
    
#     sys.stderr.write("\n{} total, {} unique problematic out of {}".format(len(problematic), len(set(problematic)), total))
#     import ipdb; ipdb.set_trace()

# check_if_in_index(train)
# check_if_in_index(dev)
# sys.exit()

model = Model()
adam = SimpleSGDTrainer(model, 0.01)
# adam = AdamTrainer(model, 0.0001, 0.01, 0.9999, 1e-8)

v_x = model.add_lookup_parameters((VOCDICT.size(), TOKDIM))
p_x = model.add_lookup_parameters((POSDICT.size(), POSDIM))
l_x = model.add_lookup_parameters((LEMDICT.size(), LEMDIM))

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

w_z = model.add_parameters((HIDDENDIM, 2*LSTMDIM))
b_z = model.add_parameters((HIDDENDIM, 1))
w_f = model.add_parameters((2, HIDDENDIM))  # prediction: is a target or not.
b_f = model.add_parameters((2, 1))

def identify_targets(builders, tokens, postags, lemmas, goldtargets=None):
    renew_cg()
    trainmode = (goldtargets is not None)

    sentlen = len(tokens)
    emb_x = [v_x[tok] for tok in tokens]
    pos_x = [p_x[pos] for pos in postags]
    lem_x = [l_x[lem] for lem in lemmas]

    pw_i = parameter(w_i)
    pb_i = parameter(b_i)

    emb2_xi = [(pw_i * concatenate([emb_x[i], pos_x[i], lem_x[i]])  + pb_i) for i in xrange(sentlen)]
    if USE_WV:
        pw_e = parameter(w_e)
        pb_e = parameter(b_e)
        for i in xrange(sentlen):
            if tokens[i] in wvs:
                nonupdatedwv = e_x[tokens[i]]  # prevent the wvecs from being updated
                emb2_xi[i] = emb2_xi[i] + pw_e * nonupdatedwv + pb_e

    emb2_x = [rectify(emb2_xi[i]) for i in xrange(sentlen)]

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

    losses = []
    predicted_targets = []
    for i in xrange(sentlen):
        h_i = concatenate([fw_x[i], bw_x[sentlen - i - 1]])
        score_i = pw_f * rectify(pw_z * h_i + pb_z) + pb_f
        if trainmode and USE_DROPOUT:
            score_i = dropout(score_i, DROPOUT_RATE)

        logloss = log_softmax(score_i, [0, 1])
        if not trainmode:
            isitatarget = np.argmax(logloss.npvalue())
        else:
            isitatarget = int(i in goldtargets)
        
        if int(np.argmax(logloss.npvalue())) != 0:
            predicted_targets.append(i)
        
        losses.append(pick(logloss, isitatarget))
    
    objective = -esum(losses) if losses else None
    return objective, predicted_targets

# def print_result(goldexamples, pred_targmaps):
#     with codecs.open(out_conll_file, "w", "utf-8") as f:
#         for g,p in zip(goldexamples, pred_targmaps):
#             result = g.get_newstr_frame(p) + "\n"
#             f.write(result)
#         f.close()


# main
NUMEPOCHS = 100
EVAL_EVERY_EPOCH = 100
DEV_EVAL_EPOCH = 10 * EVAL_EVERY_EPOCH

if options.mode in ["train", "refresh"]:
    tagged = loss = 0.0
    bestdevf = 0.0
    train_result = [0.0, 0.0, 0.0]

    for epoch in xrange(NUMEPOCHS):
        random.shuffle(train)
        for idx, trex in enumerate(train, 1):
            if idx % EVAL_EVERY_EPOCH == 0:
                adam.status()
                _, _, trainf = calc_f(train_result)
                sys.stderr.write("%d loss = %.6f train f1 = %.4f\n" %(idx, loss/tagged, trainf))
                tagged = loss = 0.0
                train_result = [0.0, 0.0, 0.0]
            inptoks = []
            unk_replace_tokens(trex.tokens, inptoks, VOCDICT, UNK_PROB, UNKTOKEN)

            trexloss, trexpred = identify_targets(
                builders, inptoks, trex.postags, trex.lemmas, goldtargets=trex.targetframedict.keys())
            trainex_result = evaluate_example_targetid(trex.targetframedict.keys(), trexpred)
            train_result = np.add(train_result, trainex_result)

            if trexloss is not None:
                loss += trexloss.scalar_value()
                trexloss.backward()
                adam.update()
            tagged += 1

            if idx % DEV_EVAL_EPOCH == 0:
                corpus_result = [0.0, 0.0, 0.0]
                devtagged = devloss = 0.0
                predictions = []
                for devex in dev:
                    devludict = devex.get_only_targets()
                    dl, predicted = identify_targets(
                        builders, devex.tokens, devex.postags, devex.lemmas)
                    if dl is not None:
                        devloss += dl.scalar_value()
                    predictions.append(predicted)

                    devex_result = evaluate_example_targetid(devex.targetframedict.keys(), predicted)
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
                    # TODO(swabha) : instead of printing the result
                    # print_result(devexamples, predictions)
                    sys.stderr.write(" -- saving")

                    model.save(modelfname)
                    os.symlink(modelfname, "tmp.link")
                    os.rename("tmp.link", MODELSYMLINK)
                sys.stderr.write("\n")

elif options.mode == "test":
    model.populate(options.modelfile)
    corpus_tpfpfn = [0.0, 0.0, 0.0]

    testpredictions = []

    # sn = devexamples[0].sent_num
    # sl = [0.0,0.0,0.0]
    # print("Sent#%d :" % sn)
    # devexamples[0].print_internal_sent()

    for testex in dev:
        _, predicted = identify_targets(builders, testex.tokens, testex.postags, testex.lemmas)

        tpfpfn = evaluate_example_targetid(testex.targetframedict.keys(), predicted)
        corpus_tpfpfn = np.add(corpus_tpfpfn, tpfpfn)

        testpredictions.append(predicted)

        # sentnum = testex.sent_num
        # if sentnum != sn:
        #     lp, lr, lf = calc_f(sl)
        #     print("\t\t\t\t\t\t\t\t\tTotal: %.1f / %.1f / %.1f\n"
        #           "Sentence ID=%d: Recall=%.5f (%.1f/%.1f) Precision=%.5f (%.1f/%.1f) Fscore=%.5f"
        #           "\n-----------------------------\n"
        #           % (sl[0], sl[0]+sl[1], sl[0]+sl[-1],
        #              sn,
        #              lr, sl[0], sl[-1] + sl[0],
        #              lp, sl[0], sl[1] + sl[0],
        #              lf))
        #     sl = [0.0,0.0,0.0]
        #     sn = sentnum
        #     print("Sent#%d :" % sentnum)
        #     testex.print_internal_sent()

        # print "gold:"
        # testex.print_internal_frame()
        # print "prediction:"
        # testex.print_external_frame(predicted)

        # sl = np.add(sl, tpfpfn)
        # print tpfpfn[0], "/", tpfpfn[0]+tpfpfn[1], "/", tpfpfn[0]+tpfpfn[-1]

    # last sentence
    # lp, lr, lf = calc_f(sl)
    # print("\t\t\t\t\t\t\t\t\tTotal: %.1f / %.1f / %.1f\n"
    #       "Sentence ID=%d: Recall=%.5f (%.1f/%.1f) Precision=%.5f (%.1f/%.1f) Fscore=%.5f"
    #       "\n-----------------------------\n"
    #       % (sl[0], sl[0]+sl[1], sl[0]+sl[-1],
    #          sentnum,
    #          lp, sl[0], sl[1] + sl[0],
    #          lr, sl[0], sl[-1] + sl[0],
    #          lf))

    testtp, testfp, testfn = corpus_tpfpfn
    testp, testr, testf = calc_f(corpus_tpfpfn)
    sys.stderr.write("[test] p = %.4f (%.1f/%.1f) r = %.4f (%.1f/%.1f) f1 = %.4f\n" %(
        testp, testtp, testtp + testfp,
        testr, testtp, testtp + testfp,
        testf))

    # sys.stderr.write("printing output conll to " + out_conll_file + " ... ")
    # print_result(devexamples, testpredictions)
    sys.stderr.write("done!\n")

    # sys.stderr.write("printing frame-elements to " + fefile + " ...\n")
    # convert_conll_to_frame_elements(out_conll_file, fefile)
    # sys.stderr.write("done!\n")
