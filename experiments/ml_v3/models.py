"""Classical estimators with fold-local transforms and explicit angle structure."""
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def configurations():
    result = {}
    def add(name, feature, model, primary=False, **kw):
        result[name] = {"feature": feature, "model": model, "primary": primary,
                        "normalization": "standard", "selection": None, "structure": "joint", **kw}
    features = ["legacy", "robust", "spectrum", "relations", "fusion", "relative"]
    for feature in features:
        add(f"{feature}_svm1", feature, "svm", True, C=1)
        add(f"{feature}_et3", feature, "et", True, leaf=3)
        add(f"{feature}_lda", feature, "lda", True)
    add("legacy_svm10", "legacy", "svm", True, C=10)
    add("fusion_rf3", "fusion", "rf", True, leaf=3)
    add("fusion_hgb31", "fusion", "hgb", True, leaves=31)
    add("fusion_et3_independent", "fusion", "et", True, leaf=3, structure="independent")
    add("fusion_et3_hierarchy", "fusion", "et", True, leaf=3, structure="hierarchy")
    for feature in features:
        for c in [.1, 10]:
            name = f"{feature}_svm{c}"
            if name not in result:
                add(name, feature, "svm", C=c)
    for feature in ["legacy", "spectrum", "fusion"]:
        for leaf in [1, 6]:
            add(f"{feature}_rf{leaf}", feature, "rf", leaf=leaf)
    for feature in ["legacy", "fusion"]:
        for leaf in [1, 6]:
            add(f"{feature}_et{leaf}", feature, "et", leaf=leaf)
    add("fusion_hgb7", "fusion", "hgb", leaves=7)
    add("fusion_robustscale_svm1", "fusion", "svm", C=1, normalization="robust")
    add("fusion_pca_svm1", "fusion", "svm", C=1, selection="pca95")
    for k in [32, 64]:
        add(f"fusion_f{k}_svm1", "fusion", "svm", C=1, selection=k)
        add(f"fusion_f{k}_et3", "fusion", "et", leaf=3, selection=k)
    for feature in ["upper", "forearm"]:
        add(f"{feature}_svm1", feature, "svm", C=1)
        add(f"{feature}_et3", feature, "et", leaf=3)
    add("legacy_linear1", "legacy", "linear", C=1)
    add("fusion_et3_unweighted", "fusion", "et", leaf=3, balanced=False)
    assert sum(c["primary"] for c in result.values()) == 23
    return result


def estimator(config):
    model = config["model"]
    if model == "svm":
        return SVC(C=config["C"], gamma="scale", cache_size=256, probability=False)
    if model == "linear":
        return LinearSVC(C=config["C"], dual="auto", max_iter=5000, random_state=42)
    if model == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if model in ("rf", "et"):
        cls = RandomForestClassifier if model == "rf" else ExtraTreesClassifier
        return cls(n_estimators=300, min_samples_leaf=config["leaf"], max_features=.5,
                   random_state=42, n_jobs=2,
                   class_weight="balanced" if config.get("balanced", True) else None)
    if model == "hgb":
        return HistGradientBoostingClassifier(max_iter=100, learning_rate=.05,
                   max_leaf_nodes=config["leaves"], min_samples_leaf=15,
                   l2_regularization=1, early_stopping=False, random_state=42)
    raise ValueError(model)


class Model:
    def __init__(self, config):
        self.config = config

    def fit(self, x, y, records):
        self.scaler = RobustScaler() if self.config["normalization"] == "robust" else StandardScaler()
        z = self.scaler.fit_transform(x)
        self.columns = np.arange(x.shape[1])
        selection = self.config["selection"]
        self.pca = None
        if isinstance(selection, int):
            unique = np.unique(records)
            means = np.array([np.median(z[records == r], axis=0) for r in unique])
            targets = np.array([y[records == r][0] for r in unique])
            values = f_classif(means, targets)[0]
            values = np.nan_to_num(values, nan=-np.inf, posinf=np.finfo(float).max)
            self.columns = np.sort(np.argsort(-values, kind="stable")[:selection])
            z = z[:, self.columns]
        elif selection == "pca95":
            self.pca = PCA(n_components=.95, svd_solver="full")
            z = self.pca.fit_transform(z)
        structure = self.config["structure"]
        if structure == "joint":
            self.model = estimator(self.config).fit(z, y)
        else:
            self.gesture = estimator(self.config).fit(z, y//2)
            if structure == "independent":
                self.angles = [estimator(self.config).fit(z, y%2)]
            else:
                self.angles = [estimator(self.config).fit(z[y//2 == g], (y[y//2 == g]%2)) for g in range(4)]
        return self

    def transform(self, x):
        z = self.scaler.transform(x)
        return self.pca.transform(z) if self.pca is not None else z[:, self.columns]

    def predict_proba(self, x):
        z = self.transform(x)
        if self.config["structure"] == "joint":
            if self.config["model"] in ("svm", "linear"):
                # Never used in probability-ensemble selection.
                margin = self.model.decision_function(z)
                q = np.exp(margin-margin.max(1, keepdims=True))
                return q/q.sum(1, keepdims=True)
            return self.model.predict_proba(z)
        gp = self.gesture.predict_proba(z)
        ap = ([self.angles[0].predict_proba(z)]*4 if len(self.angles) == 1 else
              [a.predict_proba(z) for a in self.angles])
        return np.stack([gp[:, g, None]*ap[g] for g in range(4)], 1).reshape(-1, 8)

    def predict(self, x):
        if self.config["structure"] == "joint":
            return self.model.predict(self.transform(x))
        return self.predict_proba(x).argmax(1)
