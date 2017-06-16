import sys
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

class SentAnno:

    def __init__(self, text):
        self.text = text
        self.tokens = []
        self.postags = []
        self.nltkpostags = []
        self.nltklemmas = []
        self.foundpos = False # either BNC or PENN annotations
        self.stindices = {}
        self.enindices = {}

    def add_token(self, startend):
        st, en = startend
        st = int(st)
        en = int(en)
        self.stindices[st] = len(self.tokens)
        self.enindices[en] = len(self.tokens)

    def normalize_tokens(self):
        if len(self.stindices) != len(self.enindices):
            sys.stderr.write("\t\tIssue: overlapping tokenization for multiple tokens\n")
            return
        start = {}
        idx = 0
        for s in sorted(self.stindices):
            self.stindices[s] = idx
            start[idx] = s
            idx += 1
        end = {}
        idx = 0
        for t in sorted(self.enindices):
            self.enindices[t] = idx
            end[idx] = t
            if idx > 0 and end[idx - 1] > start[idx]:
                sys.stderr.write("\t\tIssue: overlapping tokenization of neighboring tokens\n")
                return
            token = self.text[start[idx] : t + 1].strip()
            if " " in token:
                sys.stderr.write("\t\tIssue: incorrect tokenization "  + token + "\n")
                return
            if token == "": continue
            self.tokens.append(token)
            idx += 1
        try:
            self.nltkpostags = [ele[1] for ele in pos_tag(self.tokens)]
            for idx in xrange(len(self.tokens)):
                tok = self.tokens[idx]
                if self.nltkpostags[idx].startswith("V"):
                    self.nltklemmas.append(lemmatizer.lemmatize(tok, pos='v'))
                else:
                    self.nltklemmas.append(lemmatizer.lemmatize(tok))
        except IndexError:
            print self.tokens
            print pos_tag(self.tokens)
        return True

    def get_tokens_by_offset(self, startend):
        st, en = startend
        st = int(st)
        en = int(en)
        if st not in self.stindices or en not in self.enindices:
            raise Exception("\t\tBug: broken tokenization", st, en)
        return self.stindices[st], self.enindices[en]

    def add_postag(self, postag):
        self.foundpos = True
        self.postags.append(postag)
        
    def size(self):
        return len(self.tokens)

    def info_at_idx(self, idx):
        if len(self.tokens) <= idx :
            raise Exception("\t\tBug: invalid index", idx)
        if len(self.postags) <= idx:
            postag = NOTALABEL
        else:
            postag = self.postags[idx]
        return self.tokens[idx], postag, self.nltkpostags[idx], self.nltklemmas[idx]
    

class FrameAnno:

    def __init__(self, lu, frame, sent):
        self.lu = lu
        self.frame = frame
        self.sent = sent
        self.target = set([])
        self.foundtarget = False
        self.fe = {}
        self.foundfes = False

    def add_fe(self, offset, arglabel):
        try:
            st, en = self.sent.get_tokens_by_offset(offset)
        except Exception:
            sys.stderr.write("\t\tIssue: broken tokenization for FE\n")
            return
        self.foundfes = True
        for idx in xrange(st, en + 1):
            if idx in self.fe:
                raise Exception("\t\tIssue: duplicate FE at ", idx, self.fe)

        # BIOS tagging
        if st == en:
            self.fe[st] = "S-" + arglabel
        else:
            self.fe[st] = "B-" + arglabel
            for idx in xrange(st+1, en+1):
                if idx in self.fe:
                    raise Exception("duplicate FE at ", idx, offset, arglabel)
                self.fe[idx] = "I-" + arglabel


    def add_target(self, offset):
        try:
            st, en = self.sent.get_tokens_by_offset(offset)
        except Exception:
            sys.stderr.write("\t\tIssue: broken tokenization for target\n")
            return
        self.foundtarget = True
        for idx in xrange(st, en + 1):
            if idx in self.target:
                sys.stderr.write("\t\tIssue: duplicate target at " + str(idx) + "\n")
            self.target.add(idx)

    def info_at_idx(self, idx):
        token, postag, nltkpostag, nltklemma = self.sent.info_at_idx(idx)
        lexunit = frm = "_"
        role = "O"

        if idx in self.target:
            lexunit = self.lu
            frm = self.frame

        if idx in self.fe:
            role = self.fe[idx]

        return token, postag, nltkpostag, nltklemma, lexunit, frm, role

    def __hash__(self):
        return hash((self.lu, self.frame, frozenset(self.target)))

    def __eq__(self, other):
        return self.lu == other.lu and self.frame == other.frame and self.target == other.target

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # true at the same time
        return not(self == other)
