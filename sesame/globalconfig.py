# -*- coding: utf-8 -*-
UNK = "UNK"
NOTALABEL = "_"
NOTANFE = "O"

# BIOS SCHEME
BEGINNING = 0
INSIDE = 1
OUTSIDE = 2
SINGULAR = 3

ARGTYPES={"B": BEGINNING,
          "I": INSIDE,
          NOTANFE: OUTSIDE,
          "S": SINGULAR}

INV_ARGTYPES={v:k for k,v in ARGTYPES.iteritems()}

DEBUGMODE = False

VERSION="1.5"
DATADIR = "data/"
PARSERDATADIR = DATADIR + "neural/fn" + VERSION + "/"

TRAIN_FTE = PARSERDATADIR + "fn" + VERSION + ".fulltext.train.syntaxnet.conll"
TRAIN_EXEMPLAR = PARSERDATADIR + "fn" + VERSION + ".exemplar.train.syntaxnet.conll"
DEV_CONLL   = PARSERDATADIR + "fn" + VERSION + ".dev.syntaxnet.conll"
TEST_CONLL  = PARSERDATADIR + "fn" + VERSION + ".test.syntaxnet.conll"

FNDATADIR = DATADIR + "fndata-" + VERSION + "/"
LU_INDEX = FNDATADIR + "luIndex.xml"
LUDIR = FNDATADIR + "lu/"
FTEDIR = FNDATADIR + "fulltext/"
FRAME_DIR = FNDATADIR + "frame/"
FRAME_REL_FILE = FNDATADIR + "frRelation.xml"

TESTFILES = [
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

DEVFILES = [
        "ANC__110CYL072.xml",
#        "ANC__EntrepreneurAsMadonna.xml",
#        "C-4__C-4Text.xml",
        "KBEval__MIT.xml",
#        "LUCorpus-v0.3__20000420_xin_eng-NEW.xml",
#        "LUCorpus-v0.3__20000424_nyt-NEW.xml",
        "LUCorpus-v0.3__20000415_apw_eng-NEW.xml",
        "LUCorpus-v0.3__ENRON-pearson-email-25jul02.xml",
#        "LUCorpus-v0.3__AFGP-2002-600045-Trans.xml",
#        "LUCorpus-v0.3__CNN_AARONBROWN_ENG_20051101_215800.partial-NEW.xml",
#        "LUCorpus-v0.3__CNN_ENG_20030614_173123.4-NEW-1.xml",
#        "LUCorpus-v0.3__artb_004_A1_E1_NEW.xml",
        "Miscellaneous__Hijack.xml",
#        "NTI__LibyaCountry1.xml",
        "NTI__NorthKorea_NuclearOverview.xml",
#        "NTI__SouthAfrica_Introduction.xml",
#        "NTI__WMDNews_042106.xml",
        "NTI__WMDNews_062606.xml",
        "PropBank__TicketSplitting.xml",
        ]


FILTERED_WVECS_FILE = DATADIR + "glove.6B.100d.framevocab.txt"

PTBDATADIR = DATADIR + "ptb/"

TRAIN_FTE_CONSTITS = "fn" + VERSION + ".fulltext.train.rnng.brackets"
DEV_CONSTITS = "fn" + VERSION + ".dev.rnng.brackets"
TEST_CONSTITS = "fn" + VERSION + ".test.rnng.brackets"

CONSTIT_MAP = {TRAIN_FTE:TRAIN_FTE_CONSTITS, DEV_CONLL:DEV_CONSTITS, TEST_CONLL:TEST_CONSTITS}

# # arg span types
# BEG = "b"
# END = "e"
# SINGLE = "s"



