from catboost import CatBoostClassifier
import joblib

class ModelsLoader:
    def __init__(self):
        pass

    def load_dtc(self):
        return joblib.load('models/dtc_model.pkl')

    def load_rfc(self):
        return joblib.load('models/rfc_model.pkl')

    def load_gbc(self):
        return joblib.load('models/gbc_model.pkl')

    def load_stacking(self):
        return joblib.load('models/stacking_model.pkl')

    def load_cbc(self):
        return CatBoostClassifier().load_model('models/cbc_model.cbm')

    def load_fcnn(self):
        return joblib.load('models/fcnn_model.pkl')
