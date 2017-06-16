__author__ = 'swabha'
import numpy as np
import time
from dataio import *
from itertools import izip

def calc_f(scores):
    tp, fp, fn = scores
    if tp == 0.0 and fp == 0.0:
        pr = 0.0
    else:
        pr = tp / (tp + fp)
    if tp == 0.0 and fn == 0.0:
        re = 0.0
    else:
        re = tp / (tp + fn)
    if pr == 0.0 and re == 0.0:
        f = 0.0
    else:
        f = 2.0 * pr * re / (pr + re)
    return pr, re, f

def evaluate_example_frameid(goldframe, prediction):
    tp = fp = fn = 0.0
    predframe = prediction.items()[0][1][1]

    if predframe == goldframe:
        tp += 1
    else:
        fp += 1
        fn += 1
    return [tp, fp, fn]

def unlabeled_eval(goldfes, predargmax, notanfeid):
    utp = ufp = ufn = 0.0
    goldspans = []
    for feid in goldfes:
        if feid == notanfeid: continue
        goldspans.append(sorted(goldfes[feid]))

    predspans = []
    for fe in predargmax:
        if fe == notanfeid: continue
        predspans.append(sorted(predargmax[fe]))

    # unlabeled spans
    for s in goldspans:
        if s in predspans:
            utp += 1
        else:
            ufn += 1
    for s in predspans:
        if s not in goldspans:
            ufp += 1

    return utp,ufp,ufn

def labeled_eval(corefes, goldfes, predargmax, notanfeid):
    def score(fe_id):
        if corefes == {}:
            return 1.0
        if fe_id in corefes:
            return 1.0
        return 0.5

    ltp = lfp = lfn = 0.0
    # labeled spans
    for goldfe in goldfes:
        if goldfe == notanfeid: continue
        if goldfe in predargmax and set(predargmax[goldfe]) == set(goldfes[goldfe]):
            ltp += score(goldfe)
        else:
            lfn += score(goldfe)

    # the condition below handles frames which are not annotated. ideally these should be ignored in the data.
    # if len(goldfes) == 1 and notanfeid in goldfes: return ltp, lfp, lfn

    for predfe in predargmax:
        if predfe == notanfeid: continue
        if predfe not in goldfes or set(goldfes[predfe]) != set(predargmax[predfe]):
            lfp += score(predfe)

    return ltp, lfp, lfn

def token_level_eval(sentlen, goldfes, predargmax, notanfeid):
    goldtoks = [0 for _ in xrange(sentlen)]
    for feid in goldfes:
        for span in goldfes[feid]:
            for s in xrange(span[0],span[1]+1):
                goldtoks[s] = feid

    predtoks = [0 for _ in xrange(sentlen)]
    for feid in predargmax:
        for span in predargmax[feid]:
            for s in xrange(span[0], span[1]+1):
                predtoks[s] = feid

    # token-level F1
    wtp = wfp = wfn = 0.0
    for i in xrange(sentlen):
        if goldtoks[i] == predtoks[i]:
            if goldtoks[i] != notanfeid:
                wtp += 1
        elif goldtoks[i] == notanfeid and predtoks[i] != notanfeid:
            wfp += 1
        elif predtoks[i] == notanfeid and goldtoks[i] != notanfeid:
            wfn += 1
        else:
            wfp += 1
            wfn += 1

    return wtp, wfp, wfn

def evaluate_example_argid(goldfes, predargmax, corefes, sentlen, notanfeid=None):
    utp, ufp, ufn = unlabeled_eval(goldfes, predargmax, notanfeid)
    ltp, lfp, lfn = labeled_eval(corefes, goldfes, predargmax, notanfeid)
    wtp, wfp, wfn = token_level_eval(sentlen, goldfes, predargmax, notanfeid)

    return [utp,ufp,ufn], [ltp,lfp,lfn], [wtp, wfp, wfn]

def evaluate_corpus_argid(goldex, predictions, corefrmfemap, notanfeid):
    ures = labldres = tokres = [0.0, 0.0, 0.0]

    # first sentence
    sl = [0.0,0.0,0.0]
    sn = goldex[0].sent_num
    print("Sent#%d :" % sn)
    goldex[0].print_internal_sent()

    for testex, tpred in izip(goldex, predictions):
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
        testex.print_internal_args()

        print "prediction:"
        testex.print_external_parse(tpred)

        if testex.frame.id in corefrmfemap:
            corefes = corefrmfemap[testex.frame.id]
        else:
            corefes = {}

        u, l, t = evaluate_example_argid(testex.invertedfes, tpred, corefes, len(testex.tokens), notanfeid)

        ures = np.add(ures, u)
        labldres = np.add(labldres, l)
        tokres = np.add(tokres, t)

        sl = np.add(sl, l)
        print l[0],"/",l[0]+l[1], "/", l[0]+l[-1]

    # last sentence
    lp, lr, lf = calc_f(sl)
    print("\t\t\t\t\t\t\t\t\tTotal: %.1f / %.1f / %.1f\n"
          "Sentence ID=%d: Recall=%.5f (%.1f/%.1f) Precision=%.5f (%.1f/%.1f) Fscore=%.5f"
          "\n-----------------------------\n"
          % (sl[0], sl[0]+sl[1], sl[0]+sl[-1],
             sentnum,
             lr, sl[0], sl[-1] + sl[0],
             lp, sl[0], sl[1] + sl[0],
             lf))

    up, ur, uf = calc_f(ures)
    lp, lr, lf = calc_f(labldres)
    wp, wr, wf = calc_f(tokres)
    return up, ur, uf, lp, lr, lf, wp, wr, wf, ures, labldres, tokres


if __name__ == "__main__":

    goldfile = TEST_CONLL
    goldexamples, _, _ = read_conll(goldfile)

    predfile = sys.argv[1]
    predexamples, _, _ = read_conll(predfile)
    predfes = [ex.invertedfes for ex in predexamples]

    _, coreffmap, _ = read_frame_maps()

    evalstarttime = time.time()
    corp_up, corp_ur, corp_uf, \
    corp_lp, corp_lr, corp_lf, \
    corp_wp, corp_wr, corp_wf, \
    corp_ures, corp_labldres, corp_tokres = evaluate_corpus_argid(
        goldexamples, predfes, coreffmap, FEDICT.getid(NOTANFE))

    sys.stderr.write("\n[test] wpr = %.5f (%6.1f/%6.1f) wre = %.5f (%6.1f/%6.1f) wf1 = %.5f\n"
                     "[test] upr = %.5f (%6.1f/%6.1f) ure = %.5f (%6.1f/%6.1f) uf1 = %.5f\n"
                     "[test] lpr = %.5f (%6.1f/%6.1f) lre = %.5f (%6.1f/%6.1f) lf1 = %.5f [took %.3f s]\n"
                 % (corp_wp, corp_tokres[0], corp_tokres[1] + corp_tokres[0],
                    corp_wr, corp_tokres[0], corp_tokres[-1] + corp_tokres[0],
                    corp_wf,
                    corp_up, corp_ures[0], corp_ures[1] + corp_ures[0],
                    corp_ur, corp_ures[0], corp_ures[-1] + corp_ures[0],
                    corp_uf,
                    corp_lp, corp_labldres[0], corp_labldres[1] + corp_labldres[0],
                    corp_lr, corp_labldres[0], corp_labldres[-1] + corp_labldres[0],
                    corp_lf,
                    time.time()-evalstarttime))
