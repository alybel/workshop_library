import hashlib
import inspect
from . import settings
import os
import pickle


def do_hash(s):
    s = s.encode('utf-8')
    h = hashlib.md5(s)
    return h.hexdigest()


def data_fingerprint(df):
    s = '%s_%s_%s_%s_%s' % (df.columns, df.shape, df.index, df.head(), df.tail())
    return do_hash(s)


def get_model_hash(df, *args):
    from . import training
    data_hash = data_fingerprint(df)
    args = [str(arg) for arg in args]
    s = '_'.join(args)
    version_hash = do_hash(inspect.getsource(training))
    return '%s_%s_%s' % (data_hash, do_hash(s), version_hash)


class ModelAdmin(object):
    def __init__(self, model_store='models.db'):
        self.model_store = model_store

    def load_model_if_exists(self, model_hash):

        if os.path.exists(self.model_store):
            keyval_store = pickle.load(open(self.model_store, 'rb'))
            if model_hash in keyval_store:
                print('Load Model from Store')
                return keyval_store[model_hash]
        else:
            return None

    def write_to_store(self, model_hash, model):
        if os.path.exists(self.model_store):
            keyval_store = pickle.load(open(self.model_store, 'wb'))
        else:
            keyval_store = {}
        if model_hash in keyval_store:
            print('this exact model is already stored')
        else:
            keyval_store[model_hash] = model
            pickle.dump(keyval_store, open(self.model_store, 'wb'))

    def clear_store(self):
        os.remove(self.model_store)

    def list(self):
        if os.path.exists(self.model_store):
            keyval_store = pickle.load(open(self.model_store, 'rb'))
            return keyval_store.keys()
        else:
            return None

