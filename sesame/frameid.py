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

import codecs
import json
import numpy as np
import os
import random
import sys
import time
import tqdm
from optparse import OptionParser

from dynet import Model, LSTMBuilder, SimpleSGDTrainer, lookup, concatenate, rectify, renew_cg, dropout, log_softmax, esum, pick

from .conll09 import lock_dicts, post_train_lock_dicts, VOCDICT, POSDICT, FRAMEDICT, LUDICT, LUPOSDICT
from .dataio import get_wvec_map, read_conll, read_related_lus
from .evaluation import calc_f, evaluate_example_frameid
from .frame_semantic_graph import Frame
from .globalconfig import VERSION, TRAIN_FTE, UNK, DEV_CONLL, TEST_CONLL, TRAIN_EXEMPLAR
from .housekeeping import unk_replace_tokens
from .raw_data import make_data_instance
from .semafor_evaluation import convert_conll_to_frame_elements


optpr = OptionParser()
optpr.add_option("--mode", dest="mode", type="choice", choices=["train", "test", "refresh", "predict"], default="train")
optpr.add_option("-n", "--model_name", help="Name of model directory to save model to.")
optpr.add_option("--hier", action="store_true", default=False)
optpr.add_option("--exemplar", action="store_true", default=False)
optpr.add_option("--raw_input", type="str", metavar="FILE")
optpr.add_option("--config", type="str", metavar="FILE")
(options, args) = optpr.parse_args()

model_dir = "logs/{}/".format(options.model_name)
model_file_name = "{}best-frameid-{}-model".format(model_dir, VERSION)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if options.exemplar:
    train_conll = TRAIN_EXEMPLAR
else:
    train_conll = TRAIN_FTE

USE_DROPOUT = True
if options.mode in ["test", "predict"]:
    USE_DROPOUT = False
USE_WV = True
USE_HIER = options.hier

sys.stderr.write("_____________________\n")
sys.stderr.write("COMMAND: {}\n".format(" ".join(sys.argv)))
if options.mode in ["train", "refresh"]:
    sys.stderr.write("VALIDATED MODEL SAVED TO:\t{}\n".format(model_file_name))
else:
    sys.stderr.write("MODEL FOR TEST / PREDICTION:\t{}\n".format(model_file_name))
sys.stderr.write("PARSING MODE:\t{}\n".format(options.mode))
sys.stderr.write("_____________________\n\n")


def find_multitokentargets(examples, split):
    multitoktargs = tottargs = 0.0
    for tr in examples:
        tottargs += 1
        if len(tr.targetframedict) > 1:
            multitoktargs += 1
            tfs = set(tr.targetframedict.values())
            if len(tfs) > 1:
                raise Exception("different frames for neighboring targets!", tr.targetframedict)
    sys.stderr.write("multi-token targets in %s: %.3f%% [%d / %d]\n"
                     % (split, multitoktargs*100/tottargs, multitoktargs, tottargs))

trainexamples, m, t = read_conll(train_conll)
find_multitokentargets(trainexamples, "train")

post_train_lock_dicts()
lufrmmap, relatedlus = read_related_lus()
if USE_WV:
    pretrained_embeddings_map = get_wvec_map()
    PRETRAINED_DIM = len(list(pretrained_embeddings_map.values())[0])

lock_dicts()
UNKTOKEN = VOCDICT.getid(UNK)


if options.mode in ["train", "refresh"]:
    devexamples, m, t = read_conll(DEV_CONLL)
    find_multitokentargets(devexamples, "dev/test")
    out_conll_file = "{}predicted-{}-frameid-dev.conll".format(model_dir, VERSION)
elif options.mode == "test":
    devexamples, m, t = read_conll(TEST_CONLL)
    find_multitokentargets(devexamples, "dev/test")
    out_conll_file = "{}predicted-{}-frameid-test.conll".format(model_dir, VERSION)
    fefile = "{}predicted-{}-frameid-test.fes".format(model_dir, VERSION)
elif options.mode == "predict":
    assert options.raw_input is not None
    instances, _, _ = read_conll(options.raw_input)
    out_conll_file = "{}predicted-frames.conll".format(model_dir)
else:
    raise Exception("Invalid parser mode", options.mode)

# Default configurations.
configuration = {'train': train_conll,
                 'use_exemplar': options.exemplar,
                 'use_hierarchy': USE_HIER,
                 'unk_prob': 0.1,
                 'dropout_rate': 0.01,
                 'token_dim': 100,
                 'pos_dim': 100,
                 'lu_dim': 100,
                 'lu_pos_dim': 100,
                 'lstm_input_dim': 100,
                 'lstm_dim': 100,
                 'lstm_depth': 2,
                 'hidden_dim': 100,
                 'use_dropout': USE_DROPOUT,
                 'pretrained_embedding_dim': PRETRAINED_DIM,
                 'num_epochs': 100 if not options.exemplar else 25,
                 'patience': 25,
                 'eval_after_every_epochs': 100,
                 'dev_eval_epoch_frequency': 50 if options.exemplar else 5}
configuration_file = os.path.join(model_dir, 'configuration.json')
if options.mode == "train":
    if options.config:
        config_json = open(options.config, "r")
        configuration = json.load(config_json)
    with open(configuration_file, 'w') as fout:
        fout.write(json.dumps(configuration))
        fout.close()
else:
    json_file = open(configuration_file, "r")
    configuration = json.load(json_file)

UNK_PROB = configuration['unk_prob']
DROPOUT_RATE = configuration['dropout_rate']

TOKDIM = configuration['token_dim']
POSDIM = configuration['pos_dim']
LUDIM = configuration['lu_dim']
LPDIM = configuration['lu_pos_dim']
INPDIM = TOKDIM + POSDIM

LSTMINPDIM = configuration['lstm_input_dim']
LSTMDIM = configuration['lstm_dim']
LSTMDEPTH = configuration['lstm_depth']
HIDDENDIM = configuration['hidden_dim']

NUM_EPOCHS = configuration['num_epochs']
PATIENCE = configuration['patience']
EVAL_EVERY_EPOCH = configuration['eval_after_every_epochs']
DEV_EVAL_EPOCH = configuration['dev_eval_epoch_frequency'] * EVAL_EVERY_EPOCH

sys.stderr.write("\nPARSER SETTINGS (see {})\n_____________________\n".format(configuration_file))
for key in sorted(configuration):
    sys.stderr.write("{}:\t{}\n".format(key.upper(), configuration[key]))

sys.stderr.write("\n")

def print_data_status(fsp_dict, vocab_str):
    sys.stderr.write("# {} = {}\n\tUnseen in dev/test = {}\n\tUnlearnt in dev/test = {}\n".format(
        vocab_str, fsp_dict.size(), fsp_dict.num_unks()[0], fsp_dict.num_unks()[1]))

print_data_status(VOCDICT, "Tokens")
print_data_status(POSDICT, "POS tags")
print_data_status(LUDICT, "LUs")
print_data_status(LUPOSDICT, "LU POS tags")
print_data_status(FRAMEDICT, "Frames")
sys.stderr.write("\n_____________________\n\n")

model = Model()
trainer = SimpleSGDTrainer(model)
# trainer = AdamTrainer(model, 0.0001, 0.01, 0.9999, 1e-8)

v_x = model.add_lookup_parameters((VOCDICT.size(), TOKDIM))
p_x = model.add_lookup_parameters((POSDICT.size(), POSDIM))
lu_x = model.add_lookup_parameters((LUDICT.size(), LUDIM))
lp_x = model.add_lookup_parameters((LUPOSDICT.size(), LPDIM))
if USE_WV:
    e_x = model.add_lookup_parameters((VOCDICT.size(), PRETRAINED_DIM))
    for wordid in pretrained_embeddings_map:
        e_x.init_row(wordid, pretrained_embeddings_map[wordid])

    # Embedding for unknown pretrained embedding.
    u_x = model.add_lookup_parameters((1, PRETRAINED_DIM), init='glorot')

    w_e = model.add_parameters((LSTMINPDIM, PRETRAINED_DIM+INPDIM))
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

    emb2_xi = []
    for i in range(sentlen + 1):
        if tokens[i] in pretrained_embeddings_map:
            # If update set to False, prevents pretrained embeddings from being updated.
            emb_without_backprop = lookup(e_x, tokens[i], update=True)
            features_at_i = concatenate([emb_x[i], pos_x[i], emb_without_backprop])
        else:
            features_at_i = concatenate([emb_x[i], pos_x[i], u_x])
        emb2_xi.append(w_e * features_at_i + b_e)

    emb2_x = [rectify(emb2_xi[i]) for i in range(sentlen+1)]

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
        # TODO(swabha): Add more Baidu-style features here.
        f_i = w_f * rectify(w_z * fbemb_i + b_z) + b_f
        if trainmode and USE_DROPOUT:
            f_i = dropout(f_i, DROPOUT_RATE)

        logloss = log_softmax(f_i, valid_frames)

        if not trainmode:
            chosenframe = np.argmax(logloss.npvalue())

    if trainmode:
        chosenframe = goldframe.id

    losses = []
    if logloss is not None:
        losses.append(pick(logloss, chosenframe))

    prediction = {tidx: (lexunit, Frame(chosenframe)) for tidx in targetpositions}

    objective = -esum(losses) if losses else None
    return objective, prediction

def print_as_conll(goldexamples, pred_targmaps):
    with codecs.open(out_conll_file, "w", "utf-8") as f:
        for g,p in zip(goldexamples, pred_targmaps):
            result = g.get_predicted_frame_conll(p) + "\n"
            f.write(result)
        f.close()


best_dev_f1 = 0.0
if options.mode in ["refresh"]:
    sys.stderr.write("Reloading model from {} ...\n".format(model_file_name))
    model.populate(model_file_name)
    with open(os.path.join(model_dir, "best-dev-f1.txt"), "r") as fin:
        for line in fin:
            best_dev_f1 = float(line.strip())
    fin.close()
    sys.stderr.write("Best dev F1 so far = %.4f\n" % best_dev_f1)

if options.mode in ["train", "refresh"]:
    loss = 0.0
    devf = best_dev_f1 = 0.0
    devtp = devfp = 0
    last_updated_epoch = 0

    epochs_trained = 0
    epoch_iterator = tqdm.trange(epochs_trained,
                                 NUM_EPOCHS,
                                 desc="FrameID Epoch")

    for epoch, _ in enumerate(epoch_iterator):
        random.shuffle(trainexamples)
        train_iterator = tqdm.tqdm(trainexamples,
                              desc="Iteration")
        trainer.status()
        for idx, trex in enumerate(train_iterator, 1):
            train_iterator.set_description(
                "epoch = %d loss = %.6f val_f1 = %.4f (%d/%d) best_val_f1 = %.4f" % (
                    epoch, loss/idx, devf, devtp, devtp + devfp, best_dev_f1))

            inptoks = []
            unk_replace_tokens(trex.tokens, inptoks, VOCDICT, UNK_PROB, UNKTOKEN)

            trexloss,_ = identify_frames(
                builders, inptoks, trex.postags, trex.lu, list(trex.targetframedict.keys()), trex.frame)

            if trexloss is not None:
                loss += trexloss.scalar_value()
                trexloss.backward()
                trainer.update()

            if idx % DEV_EVAL_EPOCH == 0:
                corpus_result = [0.0, 0.0, 0.0]
                devtagged = devloss = 0.0
                predictions = []
                for devex in devexamples:
                    devludict = devex.get_only_targets()
                    dl, predicted = identify_frames(
                        builders, devex.tokens, devex.postags, devex.lu, list(devex.targetframedict.keys()))
                    if dl is not None:
                        devloss += dl.scalar_value()
                    predictions.append(predicted)

                    devex_result = evaluate_example_frameid(devex.frame, predicted)
                    corpus_result = np.add(corpus_result, devex_result)
                    devtagged += 1

                devp, devr, devf = calc_f(corpus_result)
                devtp, devfp, devfn = corpus_result
                if devf > best_dev_f1:
                    best_dev_f1 = devf
                    with open(os.path.join(model_dir, "best-dev-f1.txt"), "w") as fout:
                        fout.write("{}\n".format(best_dev_f1))
                        fout.close()

                    print_as_conll(devexamples, predictions)
                    model.save(model_file_name)
                    last_updated_epoch = epoch
        if epoch - last_updated_epoch > PATIENCE:
            sys.stderr.write("Ran out of patience, ending training.\n")
            sys.stderr.write("Best model with F1 = {} saved to {}\n".format(best_dev_f1, model_file_name))
            break
        loss = 0.0

elif options.mode == "test":
    sys.stderr.write("Loading model from {} ...\n".format(model_file_name))
    model.populate(model_file_name)
    corpus_tpfpfn = [0.0, 0.0, 0.0]

    testpredictions = []

    sn = devexamples[0].sent_num
    sl = [0.0,0.0,0.0]
    logger = open("{}/frameid-prediction-analysis.log".format(model_dir), "w")
    logger.write("Sent#%d :\n" % sn)
    devexamples[0].print_internal_sent(logger)

    for testex in devexamples:
        _, predicted = identify_frames(builders, testex.tokens, testex.postags, testex.lu, list(testex.targetframedict.keys()))

        tpfpfn = evaluate_example_frameid(testex.frame, predicted)
        corpus_tpfpfn = np.add(corpus_tpfpfn, tpfpfn)

        testpredictions.append(predicted)

        sentnum = testex.sent_num
        if sentnum != sn:
            lp, lr, lf = calc_f(sl)
            logger.write("\t\t\t\t\t\t\t\t\tTotal: %.1f / %.1f / %.1f\n"
                  "Sentence ID=%d: Recall=%.5f (%.1f/%.1f) Precision=%.5f (%.1f/%.1f) Fscore=%.5f"
                  "\n-----------------------------\n"
                  % (sl[0], sl[0]+sl[1], sl[0]+sl[-1],
                     sn,
                     lr, sl[0], sl[-1] + sl[0],
                     lp, sl[0], sl[1] + sl[0],
                     lf))
            sl = [0.0,0.0,0.0]
            sn = sentnum
            logger.write("Sent#%d :\n" % sentnum)
            testex.print_internal_sent(logger)

        logger.write("gold:\n")
        testex.print_internal_frame(logger)
        logger.write("prediction:\n")
        testex.print_external_frame(predicted, logger)

        sl = np.add(sl, tpfpfn)
        logger.write("{} / {} / {}\n".format(tpfpfn[0], tpfpfn[0]+tpfpfn[1], tpfpfn[0]+tpfpfn[-1]))

    # last sentence
    lp, lr, lf = calc_f(sl)
    logger.write("\t\t\t\t\t\t\t\t\tTotal: %.1f / %.1f / %.1f\n"
          "Sentence ID=%d: Recall=%.5f (%.1f/%.1f) Precision=%.5f (%.1f/%.1f) Fscore=%.5f"
          "\n-----------------------------\n"
          % (sl[0], sl[0]+sl[1], sl[0]+sl[-1],
             sentnum,
             lp, sl[0], sl[1] + sl[0],
             lr, sl[0], sl[-1] + sl[0],
             lf))

    testp, testr, testf = calc_f(corpus_tpfpfn)
    testtp, testfp, testfn = corpus_tpfpfn
    sys.stderr.write("[test] p = %.4f (%.1f/%.1f) r = %.4f (%.1f/%.1f) f1 = %.4f\n" % (
        testp, testtp, testtp + testfp,
        testr, testtp, testtp + testfp,
        testf))

    sys.stderr.write("Printing output conll to " + out_conll_file + " ... ")
    print_as_conll(devexamples, testpredictions)
    sys.stderr.write("Done!\n")

    sys.stderr.write("Printing frame-elements to " + fefile + " ...\n")
    convert_conll_to_frame_elements(out_conll_file, fefile)
    sys.stderr.write("Done!\n")
    logger.close()

elif options.mode == "predict":
    sys.stderr.write("Loading model from {} ...\n".format(model_file_name))
    model.populate(model_file_name)

    predictions = []
    for instance in instances:
        _, prediction = identify_frames(builders, instance.tokens, instance.postags, instance.lu, list(instance.targetframedict.keys()))
        predictions.append(prediction)
    sys.stderr.write("Printing output in CoNLL format to {}\n".format(out_conll_file))
    print_as_conll(instances, predictions)
    sys.stderr.write("Done!\n")
