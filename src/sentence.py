from itertools import izip
from nltk.tree import ParentedTree


class Sentence(object):
    """tokens and pos tags for each example sentence.
    The same sentence can be associated with multiple frame-semantic parses"""

    # TODO add inheritance for constit sentence vs dep sentence
    def __init__(self, syn_type, elements=None, tokens=None, postags=None, sentnum=None):
        if elements:
            self.sent_num = elements[0].sent_num
            self.tokens = [e.form for e in elements]
            self.postags = [e.nltk_pos for e in elements]
        if sentnum:
            self.sent_num = sentnum
        if tokens:
            self.tokens = tokens
        if postags:
            self.postags = postags

        if syn_type == "dep":
            self.depheads = [e.dephead - 1 for e in elements]
            self.root = None
            for i in xrange(len(self.depheads)):
                if self.depheads[i] == -1:
                    self.depheads[i] = i  # head of ROOT is itself
                    self.root = i
            if self.root is None:
                raise Exception("root not found!")
            self.deprels = [e.deprel for e in elements]

            self.rootpath = [self.get_path_to_root(i) for i in xrange(len(self.tokens))]
            self.outheads = self.get_heads_outside()
            self.paths = {}
            self.shortest_paths = {}
        elif syn_type == "constit":
            self.cparse = None
            self.constitspans = {}
            self.crootpaths = {}
            self.leafnodes = []
            self.idxlabelmap = {}
            self.lca = {}
            self.cpaths = {}

    def get_path_to_root(self, node):
        par = self.depheads[node]
        path = [par]
        while par != self.root:
            par = self.depheads[par]
            path.append(par)
        return path

    def get_heads_outside(self):
        outheads = {}
        for j in xrange(len(self.tokens)):
            for i in xrange(j + 1):
                outheads[(i, j)] = sum([1 for s in xrange(i, j + 1) if not i <= self.depheads[s] <= j])
        return outheads

    def get_common_path(self, src, dest):
        """
        :param src: source node in tree
        :param dest: destination node
        :return: undirected path from src to dest
        """
        if dest == self.depheads[src] or src == self.depheads[dest]:
            return []
        if dest in self.rootpath[src]:
            return self.rootpath[src][:-len(self.rootpath[dest]) - 1]
        if src in self.rootpath[dest]:
            return self.rootpath[dest][:-len(self.rootpath[src]) - 1]

        pathfrom = self.rootpath[src][::-1]
        pathto = self.rootpath[dest][::-1]
        i = 0
        for n1, n2 in izip(pathfrom, pathto):
            if n1 == n2:
                i += 1
                continue
            if n1 == dest:
                return pathfrom[:i + 1]
            return pathfrom[i:][::-1] + pathto[i:]

        if i == len(pathfrom):
            return pathto[i - 1:]
        return pathfrom[i - 1:][::-1]

    def get_all_paths_to(self, node):
        if node in self.paths:
            return
        for n in xrange(len(self.tokens)):
            if n != node and (n, node) not in self.paths:
                self.paths[(n, node)] = self.get_common_path(n, node)
        self.get_all_shortest_paths(node)

    def get_all_shortest_paths(self, target):
        for j in xrange(len(self.tokens)):
            for i in xrange(j + 1):
                self.shortest_paths[(i, j, target)] = frozenset(self.get_shortest_path_in_span(target, (i, j)))
                # print "num shortest paths:", len(set(self.shortest_paths.values())),
                # "num spans:", len(self.shortest_paths)

    def get_shortest_path_in_span(self, target, span):
        splen = len(self.tokens) + 1
        nodewithsp = span[0]
        for node in span:
            if node == target:
                return [node]
            if (node, target) not in self.paths:
                raise Exception("never considered this path", node, span, target)
            if len(self.paths[(node, target)]) < splen:
                splen = len(self.paths[(node, target)])
                nodewithsp = node
        return [nodewithsp] + self.paths[(nodewithsp, target)] + [target]

    def get_all_parts_of_ctree(self, cparse, clabeldict, learn_features):
        self.cparse = ParentedTree.fromstring(str(cparse))
        if len(cparse.leaves()) != len(self.tokens):
            raise Exception("sentences do not line up!")

        # replacing leaves with node-ids
        idx = 0
        for pos in self.cparse.treepositions('leaves'):
            self.cparse[pos] = idx
            idx += 1
        # replacing internal nodes with node-ids
        for st in self.cparse.subtrees():
            # if x[0] in parentedp.leaves(): continue
            self.idxlabelmap[idx] = clabeldict.addstr(st.label())
            st.set_label(idx)
            idx += 1
        self.get_all_constit_spans()

        if not learn_features:
            return
        # get stuff for constit features
        self.leafnodes = [k for k in self.cparse.subtrees(lambda t: t.height() == 2)]
        for a in xrange(len(self.leafnodes)):
            if self.leafnodes[a][0] != a:
                raise Exception("order mixup!")
        self.get_cpath_to_root()

        # get all lowest common ancestors
        for j in xrange(len(self.leafnodes)):
            for k in xrange(j, len(self.leafnodes)):
                lca, lcaid = self.get_lca(self.leafnodes[j], self.leafnodes[k])
                self.lca[(j, k)] = (lca, lcaid)

    def get_all_constit_spans(self):
        for st in self.cparse.subtrees():
            x = st.flatten()
            span = (x[0], x[-1])
            if span not in self.constitspans:
                self.constitspans[span] = []
            self.constitspans[span].append(self.idxlabelmap[x.label()])

    def get_cpath_to_root(self):
        for st in self.cparse.subtrees():

            leaf = st.label()
            self.crootpaths[leaf] = [st]

            if st == self.cparse.root():
                continue
            par = st.parent()
            while par != self.cparse.root():
                self.crootpaths[leaf].append(par)
                par = par.parent()

            self.crootpaths[leaf].append(par)

    def get_lca(self, src, dest):
        if src == dest:
            return src, self.idxlabelmap[src.label()]
        #
        # print self.crootpaths.keys()
        # print src.label()
        # print self.cparse.root.label()
        pathfrom = self.crootpaths[src.label()][::-1]
        pathto = self.crootpaths[dest.label()][::-1]
        common = 0
        for n1, n2 in izip(pathfrom, pathto):
            if n1 == n2:
                common += 1
                continue
            return pathfrom[common - 1], self.idxlabelmap[pathfrom[common - 1].label()]

    def get_common_cpath(self, src, dest):
        if src == dest:
            return [src]

        pathfrom = self.crootpaths[src.label()][::-1]
        pathto = self.crootpaths[dest.label()][::-1]
        common = 0
        for n1, n2 in izip(pathfrom, pathto):
            if n1 == n2:
                common += 1
                continue
            break
        return pathfrom[common - 1:][::-1] + pathto[common:]

    def get_cpath_to_target(self, target):
        for j in xrange(len(self.leafnodes)):
            for k in xrange(j, len(self.leafnodes)):
                lca, _ = self.lca[(j, k)]
                path = self.get_common_cpath(lca, self.leafnodes[target])
                self.cpaths[(j, k, target)] = frozenset([self.idxlabelmap[p.label()] for p in path])
