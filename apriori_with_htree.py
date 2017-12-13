import csv, itertools, parameters


def load_data(filename):
    """
    Loads transactions from given file
    :param filename:
    :return:
    """
    reader = csv.reader(open(filename, 'r'), delimiter=',')
    trans = [map(int, row[1:]) for row in reader]
    return trans


def find_frequent_one(data_set, support):
    """
    Find frequent one itemsets within data set
    :param data_set:
    :param support: Provided support value
    :return:
    """
    candidate_one = {}
    total = len(data_set)
    for row in data_set:
        for val in row:
            if val in candidate_one:
                candidate_one[val] += 1
            else:
                candidate_one[val] = 1

    frequent_1 = []
    for key, cnt in candidate_one.items():
        # check if given item has sufficient count.
        if cnt >= (support * total / 100):
            frequent_1.append(([key], cnt))
    return frequent_1


class HNode:
    """
    Class which represents node in a hash tree.
    """

    def __init__(self):
        self.children = {}
        self.isLeaf = True
        self.bucket = {}


class HTree:
    """
    Wrapper class for HTree instance
    """

    def __init__(self, max_leaf_cnt, max_child_cnt):
        self.root = HNode()
        self.max_leaf_cnt = max_leaf_cnt
        self.max_child_cnt = max_child_cnt
        self.frequent_itemsets = []

    def recur_insert(self, node, itemset, index, cnt):
        # TO-DO
        """
        Recursively adds nodes inside the tree and if required splits leaf node and
        redistributes itemsets among child converting itself into intermediate node.
        :param node:
        :param itemset:
        :param index:
        :return:
        """
        if index == len(itemset):
            # last bucket so just insert
            if itemset in node.bucket:
                node.bucket[itemset] += cnt
            else:
                node.bucket[itemset] = cnt
            return

        if node.isLeaf:

            if itemset in node.bucket:
                node.bucket[itemset] += cnt
            else:
                node.bucket[itemset] = cnt
            if len(node.bucket) == self.max_leaf_cnt:
                # bucket has reached its maximum capacity and its intermediate node so
                # split and redistribute entries.
                for old_itemset, old_cnt in node.bucket.iteritems():

                    hash_key = self.hash(old_itemset[index])
                    if hash_key not in node.children:
                        node.children[hash_key] = HNode()
                    self.recur_insert(node.children[hash_key], old_itemset, index + 1, old_cnt)
                # there is no point in having this node's bucket
                # so just delete it
                del node.bucket
                node.isLeaf = False
        else:
            hash_key = self.hash(itemset[index])
            if hash_key not in node.children:
                node.children[hash_key] = HNode()
            self.recur_insert(node.children[hash_key], itemset, index + 1, cnt)

    def insert(self, itemset):
        # as list can't be hashed we need to convert this into tuple
        # which can be easily hashed in leaf node buckets
        itemset = tuple(itemset)
        self.recur_insert(self.root, itemset, 0, 0)

    def add_support(self, itemset):
        runner = self.root
        itemset = tuple(itemset)
        index = 0
        while True:
            if runner.isLeaf:
                if itemset in runner.bucket:
                    runner.bucket[itemset] += 1
                break
            hash_key = self.hash(itemset[index])
            if hash_key in runner.children:
                runner = runner.children[hash_key]
            else:
                break
            index += 1

    def dfs(self, node, support_cnt):
        if node.isLeaf:
            for key, value in node.bucket.iteritems():
                if value >= support_cnt:
                    self.frequent_itemsets.append((list(key), value))
                    # print key, value, support_cnt
            return

        for child in node.children.values():
            self.dfs(child, support_cnt)

    def get_frequent_itemsets(self, support_cnt):
        """
        Returns all frequent itemsets which can be considered for next level
        :param support_cnt: Minimum cnt required for itemset to be considered as frequent
        :return:
        """
        self.frequent_itemsets = []
        self.dfs(self.root, support_cnt)
        return self.frequent_itemsets

    def hash(self, val):
        return val % self.max_child_cnt


def generate_hash_tree(candidate_itemsets, length, max_leaf_cnt=4, max_child_cnt=5):
    """
    This function generates hash tree of itemsets with each node having no more than child_max_length
    childs and each leaf node having no more than max_leaf_length.
    :param candidate_itemsets: Itemsets
    :param length: Length if each itemset
    :param max_leaf_length:
    :param child_max_length:
    :return:
    """
    htree = HTree(max_child_cnt, max_leaf_cnt)
    for itemset in candidate_itemsets:
        # add this itemset to hashtree
        htree.insert(itemset)
    return htree


def generate_k_subsets(dataset, length):
    subsets = []
    for itemset in dataset:
        subsets.extend(map(list, itertools.combinations(itemset, length)))
    return subsets


def is_prefix(list_1, list_2):
    for i in range(len(list_1) - 1):
        if list_1[i] != list_2[i]:
            return False
    return True


def apriori_generate_frequent_itemsets(dataset, support):
    """
    Generates frequent itemsets
    :param dataset:
    :param support:
    :return: List of f-itemsets with their respective count in
            form of list of tuples.
    """
    support_cnt = int(support / 100.0 * len(dataset))
    all_frequent_itemsets = find_frequent_one(dataset, support)
    prev_frequent = [x[0] for x in all_frequent_itemsets]
    length = 2
    while len(prev_frequent) > 1:
        new_candidates = []
        for i in range(len(prev_frequent)):
            j = i + 1
            while j < len(prev_frequent) and is_prefix(prev_frequent[i], prev_frequent[j]):
                # this part makes sure that all of the items remain lexicographically sorted.
                new_candidates.append(prev_frequent[i][:-1] +
                                      [prev_frequent[i][-1]] +
                                      [prev_frequent[j][-1]]
                                      )
                j += 1

        # generate hash tree and find frequent itemsets
        h_tree = generate_hash_tree(new_candidates, length)
        # for each transaction, find all possible subsets of size "length"
        k_subsets = generate_k_subsets(dataset, length)

        # support counting and finding frequent itemsets
        for subset in k_subsets:
            h_tree.add_support(subset)

        # find frequent itemsets
        new_frequent = h_tree.get_frequent_itemsets(support_cnt)
        all_frequent_itemsets.extend(new_frequent)
        prev_frequent = [tup[0] for tup in new_frequent]
        prev_frequent.sort()
        length += 1

    return all_frequent_itemsets


def generate_association_rules(f_itemsets, confidence):
    """
    This method generates association rules with confidence greater than threshold
    confidence. For finding confidence we don't need to traverse dataset again as we
    already have support of frequent itemsets.
    Remember Anti-monotone property ?
    I've done pruning in this step also, which reduced its complexity significantly:
    Say X -> Y is AR which don't have enough confidence then any other rule X' -> Y'
    where (X' subset of X) is not possible as sup(X') >= sup(X).

    :param f_itemsets: Frequent itemset with their support values
    :param confidence:
    :return: Returns association rules with associated confidence
    """

    hash_map = {}
    for itemset in f_itemsets:
        hash_map[tuple(itemset[0])] = itemset[1]

    a_rules = []
    for itemset in f_itemsets:
        length = len(itemset[0])
        if length == 1:
            continue

        union_support = hash_map[tuple(itemset[0])]
        for i in range(1, length):

            lefts = map(list, itertools.combinations(itemset[0], i))
            for left in lefts:
                conf = 100.0 * union_support / hash_map[tuple(left)]
                if conf >= confidence:
                    a_rules.append([left,list(set(itemset[0]) - set(left)), conf])
    return a_rules


def print_rules(rules):

    for item in rules:
        left = ','.join(map(str, item[0]))
        right = ','.join(map(str, item[1]))
        print (' ==> '.join([left, right]))
    print('Total Rules Generated: ', len(rules))

if __name__ == '__main__':
    transactions = load_data('1000-out1.csv')
    # print find_frequent_one(transactions, 5)
    frequent = apriori_generate_frequent_itemsets(transactions, parameters.SUPPORT)
    # for item in frequent:
    #     if len(item[0]) > 1:
    #         print item

    a_rules = generate_association_rules(frequent, parameters.CONFIDENCE)
    print_rules(a_rules)