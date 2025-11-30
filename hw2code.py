import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-п_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака
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

    x = feature_vector
    y = target_vector
    n = len(x)

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    mask = x_sorted[1:] != x_sorted[:-1]
    if not np.any(mask):
        return np.array([]), np.array([]), None, None

    thresholds = (x_sorted[1:][mask] + x_sorted[:-1][mask]) / 2

    y_cumsum = np.cumsum(y_sorted)
    total_ones = y_cumsum[-1]

    left_sizes = np.arange(1, n)[mask]
    right_sizes = n - left_sizes

    left_ones = y_cumsum[left_sizes - 1]
    right_ones = total_ones - left_ones

    left_p1 = left_ones / left_sizes
    left_p0 = 1 - left_p1
    right_p1 = right_ones / right_sizes
    right_p0 = 1 - right_p1

    g_left = 1 - left_p1**2 - left_p0**2
    g_right = 1 - right_p1**2 - right_p0**2

    ginis = -(left_sizes / n) * g_left - (right_sizes / n) * g_right

    best_idx = np.argmax(ginis)
    gini_best = ginis[best_idx]
    threshold_best = thresholds[best_idx]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split if min_samples_split is not None else 2
        self._min_samples_leaf = min_samples_leaf if min_samples_leaf is not None else 1

    def _fit_node(self, sub_X, sub_y, node, depth=0):

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, split_best, gini_best = None, None, None, -np.inf

        for feature in range(sub_X.shape[1]):

            feature_type = self._feature_types[feature]
            feature_vector = sub_X[:, feature]

            if feature_type == "categorical":
                categories = np.unique(feature_vector)
                means = [np.mean(sub_y[feature_vector == c]) for c in categories]
                sorted_vals = [c for _, c in sorted(zip(means, categories))]
                mapping = {v: i for i, v in enumerate(sorted_vals)}
                feature_vector = np.array([mapping[v] for v in feature_vector])

            thresholds, ginis, thr_best, gini_local_best = find_best_split(feature_vector, sub_y)

            if thr_best is None:
                continue

            if gini_local_best > gini_best:
                gini_best = gini_local_best
                feature_best = feature
                threshold_best = thr_best
                split_best = feature_vector < thr_best
                best_feature_type = feature_type
                if feature_type == "categorical":
                    best_categories = [v for v in mapping if mapping[v] < thr_best]

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if split_best.sum() < self._min_samples_leaf or (~split_best).sum() < self._min_samples_leaf:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if best_feature_type == "real":
            node["threshold"] = threshold_best
        else:
            node["categories_split"] = best_categories

        node["left_child"] = {}
        node["right_child"] = {}

        self._fit_node(sub_X[split_best], sub_y[split_best], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split_best], sub_y[~split_best], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]

        if "threshold" in node:
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        else:
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
