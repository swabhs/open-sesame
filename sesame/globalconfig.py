# -*- coding: utf-8 -*-
import json
import sys

config_json = open("configurations/global_config.json", "r")
configuration = json.load(config_json)
for key in sorted(configuration):
    sys.stderr.write("{}:\t{}\n".format(key.upper(), configuration[key]))

VERSION = str(configuration["version"])
DATA_DIR = configuration["data_directory"]
EMBEDDINGS_FILE = configuration["embeddings_file"]
DEBUG_MODE = configuration["debug_mode"]

# The following variables are held constant throughout the repository. Change at your own peril!

PARSER_DATA_DIR = DATA_DIR + "neural/fn" + VERSION + "/"
TRAIN_FTE = PARSER_DATA_DIR + "fn" + VERSION + ".fulltext.train.syntaxnet.conll"
TRAIN_EXEMPLAR = PARSER_DATA_DIR + "fn" + VERSION + ".exemplar.train.syntaxnet.conll"
DEV_CONLL = PARSER_DATA_DIR + "fn" + VERSION + ".dev.syntaxnet.conll"
TEST_CONLL = PARSER_DATA_DIR + "fn" + VERSION + ".test.syntaxnet.conll"

FN_DATA_DIR = DATA_DIR + "fndata-" + VERSION + "/"
LU_INDEX = FN_DATA_DIR + "luIndex.xml"
LU_DIR = FN_DATA_DIR + "lu/"
FULLTEXT_DIR = FN_DATA_DIR + "fulltext/"
FRAME_DIR = FN_DATA_DIR + "frame/"
FRAME_REL_FILE = FN_DATA_DIR + "frRelation.xml"

TEST_FILES = [
        "ANC__110CYL067.xml",
        "ANC__110CYL069.xml",
        "ANC__112C-L013.xml",
        "ANC__IntroHongKong.xml",
        "ANC__StephanopoulosCrimes.xml",
        "ANC__WhereToHongKong.xml",
        "KBEval__atm.xml",
        "KBEval__Brandeis.xml",
        "KBEval__cycorp.xml",
        "KBEval__parc.xml",
        "KBEval__Stanford.xml",
        "KBEval__utd-icsi.xml",
        "LUCorpus-v0.3__20000410_nyt-NEW.xml",
        "LUCorpus-v0.3__AFGP-2002-602187-Trans.xml",
        "LUCorpus-v0.3__enron-thread-159550.xml",
        "LUCorpus-v0.3__IZ-060316-01-Trans-1.xml",
        "LUCorpus-v0.3__SNO-525.xml",
        "LUCorpus-v0.3__sw2025-ms98-a-trans.ascii-1-NEW.xml",
        "Miscellaneous__Hound-Ch14.xml",
        "Miscellaneous__SadatAssassination.xml",
        "NTI__NorthKorea_Introduction.xml",
        "NTI__Syria_NuclearOverview.xml",
        "PropBank__AetnaLifeAndCasualty.xml",
        ]

DEV_FILES = [
        "ANC__110CYL072.xml",
        "KBEval__MIT.xml",
        "LUCorpus-v0.3__20000415_apw_eng-NEW.xml",
        "LUCorpus-v0.3__ENRON-pearson-email-25jul02.xml",
        "Miscellaneous__Hijack.xml",
        "NTI__NorthKorea_NuclearOverview.xml",
        "NTI__WMDNews_062606.xml",
        "PropBank__TicketSplitting.xml",
        ]

PTB_DATA_DIR = DATA_DIR + "ptb/"

TRAIN_FTE_CONSTITS = "fn" + VERSION + ".fulltext.train.rnng.brackets"
DEV_CONSTITS = "fn" + VERSION + ".dev.rnng.brackets"
TEST_CONSTITS = "fn" + VERSION + ".test.rnng.brackets"

CONSTIT_MAP = {
        TRAIN_FTE : TRAIN_FTE_CONSTITS,
        DEV_CONLL : DEV_CONSTITS,
        TEST_CONLL : TEST_CONSTITS
        }

# Label settings
UNK = "UNK"
EMPTY_LABEL = "_"
EMPTY_FE = "O"

# BIOS scheme settings
BEGINNING = 0
INSIDE = 1
OUTSIDE = 2
SINGULAR = 3

BIO_INDEX_DICT = {
        "B": BEGINNING,
        "I": INSIDE,
        EMPTY_FE: OUTSIDE,
        "S": SINGULAR
}

INDEX_BIO_DICT = {index: tag for tag, index in BIO_INDEX_DICT.iteritems()}