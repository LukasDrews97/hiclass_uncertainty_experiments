from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from hiclass._calibration.Calibrator import _Calibrator
from hiclass.HierarchicalClassifier import make_leveled
import numpy as np
import scipy

class FlatClassifier(BaseEstimator):
    def __init__(self, local_classifier: BaseEstimator, n_jobs: int = 1, calibration_method: str = None):
        self.local_classifier = local_classifier
        self.n_jobs = n_jobs
        self.calibration_method = calibration_method
        self.calibrator = None
        self.separator_ = "::HiClass::Separator::"

    def _disambiguate(self, y):
        if y.ndim == 2:
            new_y = []
            for i in range(y.shape[0]):
                row = [str(y[i, 0])]
                for j in range(1, y.shape[1]):
                    parent = str(row[-1])
                    child = str(y[i, j])
                    row.append(parent + self.separator_ + child)
                new_y.append(np.asarray(row, dtype=np.str_))
            return np.array(new_y)
        return y
    
    def _transform_labels(self, y, create_classes_ = False):
        #y = y[:, -1] # only keep leaf nodes as labels
        assert y.ndim == 2
        
        y = make_leveled(y)
        y = self._disambiguate(y)
        
        if create_classes_:
            self.max_levels_ = y.shape[1]
            self.max_level_dimensions_ = np.array([len(np.unique(y[:, level])) for level in range(y.shape[1])])
            self.global_classes_ = [np.unique(y[:, level]).astype("str") for level in range(y.shape[1])]
            self.global_class_to_index_mapping_ = [
                {self.global_classes_[level][index]: index for index in range(len(self.global_classes_[level]))}
                for level in range(y.shape[1])
            ]
            
            classes_ = [self.global_classes_[0]]
            for level in range(1, len(self.max_level_dimensions_)):
                classes_.append(
                    np.sort(
                        np.unique(
                            [
                                label.split(self.separator_)[level]
                                for label in self.global_classes_[level]
                            ]
                        )
                    )
                )
                
            self.classes_ = classes_
            self.class_to_index_mapping_ = [
                {local_labels[index]: index for index in range(len(local_labels))}
                for local_labels in classes_
            ]

        return y[:, -1]

    def _combine_and_reorder(self, proba):
        res_proba = np.zeros(
            shape=(proba.shape[0], len(self.classes_[-1]))
        )

        for old_label in self.global_classes_[-1]:
            old_idx = self.global_class_to_index_mapping_[-1][old_label]
            local_label = old_label.split(self.separator_)[-1]
            new_idx = self.class_to_index_mapping_[-1][local_label]
            res_proba[:, new_idx] += proba[:, old_idx]
        return res_proba
        
    def fit(self, X, y):
        self.y_ = self._transform_labels(y, create_classes_ = True)
        self.X_ = check_array(X, accept_sparse="csr", allow_nd=True, ensure_2d=False)
        self.local_classifier.fit(self.X_, self.y_)
        return self

    def calibrate(self, X, y):
        y = self._transform_labels(y)
        X = check_array(X, accept_sparse="csr", allow_nd=True, ensure_2d=False)

        if self.calibration_method is None:
            self.calibrator = self.local_classifier
        else:

            if self.calibration_method == "cvap":
                if isinstance(self.X_, scipy.sparse._csr.csr_matrix):
                    self.X_cal = scipy.sparse.vstack([self.X_, X])
                else:
                    self.X_cal = np.vstack([self.X_, X])
                self.y_cal = np.hstack([self.y_, y])
            else:
                self.X_cal = X
                self.y_cal = y
                
            self.calibrator = _Calibrator(estimator=self.local_classifier, method=self.calibration_method)
            self.calibrator.fit(self.X_cal, self.y_cal)
        return self

    def predict(self, X, from_proba=False):
        if from_proba:
            return self._predict_from_proba(X)
        
        X = check_array(X, accept_sparse="csr", allow_nd=True, ensure_2d=False)
        preds = self.local_classifier.predict(X)
        preds = np.array([row.split(self.separator_) for row in preds])
        return preds

    def _predict_from_proba(self, X):
        # get predictions for all levels except last
        preds = self.predict(X)[:, :-1]

        # get probabilities for leaf nodes
        proba_last_level = self.predict_proba(X)
        # create predictions from proba
        preds_last_level = self.classes_[-1][np.argmax(proba_last_level, axis=1)]
        
        return np.column_stack([preds, preds_last_level])

    def predict_proba(self, X):
        if not self.calibrator:
            self.calibrator = self.local_classifier
        X = check_array(X, accept_sparse="csr", allow_nd=True, ensure_2d=False)
        proba = self.calibrator.predict_proba(X)

        # reorder probabilities using local labels
        return self._combine_and_reorder(proba)
    
    def _clean_up(self):
        if hasattr(self, "X_"):
            del self.X_
        if hasattr(self, "y_"):
            del self.y_
        if hasattr(self, "X_cal"):
            del self.X_cal
        if hasattr(self, "y_cal"):
            del self.y_cal
