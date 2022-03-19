import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
    $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    def H(p):
        return 1 - p**2 - (1 - p)**2

    sfv = np.sort(feature_vector)
    stv = target_vector[feature_vector.argsort()].copy()

    if len(np.unique(sfv)) <= 1:
        return None, None, None, None

    thresholds = (np.unique(sfv)[:-1] + np.unique(sfv)[1:]) / 2 
    pos = stv.cumsum()[:-1]
    props = np.arange(1, len(sfv))

    pl = pos / props
    pr = (np.sum(target_vector) - pos) / (props[::-1]) 
    ginis = (-(props / len(sfv)) * H(pl) - (1 - props / len(sfv)) * H(pr))[sfv[1:] - sfv[:-1] != 0]
    threshold_best = thresholds[np.argmax(ginis)]
    gini_best = np.max(ginis)

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self.depth = 0

    def _fit_node(self, sub_X, sub_y, node):
        ### my code
        self.depth += 1
        if ((self._max_depth != None) and (self.depth == self._max_depth)) or ((self._min_samples_split != None) and (len(sub_y) < self._min_samples_split)):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            self.depth -= 1
            return 
        ###
        
        if np.all(sub_y == sub_y[0]): #here
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            self.depth -= 1 # my code
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]): #here
            feature_type = self._feature_types[feature]
            categories_map = {}
            feature_vector = None #here
            
            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count #here
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]))) #here
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature]))) #here
            else:
                raise ValueError

            if len(np.unique(feature_vector)) == 1: #here
                continue

            ths, gns, threshold, gini = find_best_split(feature_vector, sub_y)
            ### my code
            spl = feature_vector < threshold
            if self._min_samples_leaf != None:
                def is_ok_split(t):
                    s = feature_vector < t
                    return (np.sum(s) >= self._min_samples_leaf) and (np.sum(np.logical_not(s)) >= self._min_samples_leaf)
                if np.sum(spl) < self._min_samples_leaf or np.sum(np.logical_not(spl)) < self._min_samples_leaf:
                    is_ok_t = np.array(list(map(is_ok_split, ths)))
                    if np.sum(is_ok_t) == 0:
                        continue
                    threshold = ths[is_ok_t][np.argmax(gns[is_ok_t])]
                    gini = np.max(gns[is_ok_t])
            ###
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold    

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical": #here (советую в следующий раз заменить английскую c на русскую с)
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] #here
            self.depth -= 1 # my code
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"]) #here
        self.depth -= 1 # my code
        

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        if node['type'] == 'terminal':
            return node['class']
        elif node['type'] == 'nonterminal' and self._feature_types[node["feature_split"]] == "real":
            if x[node['feature_split']] < node['threshold']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        elif node['type'] == 'nonterminal' and self._feature_types[node["feature_split"]] == "categorical":
            if x[node['feature_split']] in node['categories_split']: 
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        else:
            raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)
        return self

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
    ### my code
    def get_tree(self):
        return self._tree

    def get_depth_of_node(self, node):
        if node['type'] == 'terminal':
            return 0
        return max(self.get_depth_of_node(node['left_child']), self.get_depth_of_node(node['right_child'])) + 1

    def get_depth(self):
        return self.get_depth_of_node(self._tree) + 1

    def get_params(self, deep=True):
        return {"feature_types": self._feature_types, "max_depth": self._max_depth, "min_samples_split": self._min_samples_split, "min_samples_leaf": self._min_samples_leaf}

