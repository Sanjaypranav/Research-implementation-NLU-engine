from pathlib import Path

from ruth.nlu.featurizers.sparse_featurizers.count_vector_featurizer import CountVectorFeaturizer

cv = CountVectorFeaturizer.load({
    "analyzer": "char_wb",
    "stop_words": None,
    "min_df": 1,
    "max_df": 1.0,
    "min_ngram": 1,
    "max_ngram": 1,
    "lowercase": True,
    "max_features": None,
    "use_lemma": True,
    "name": "CountVectorFeaturizer",
    "index": 0,
    "unique": "element_0_CountVectorFeaturizer",
    "file_name": "element_0_CountVectorFeaturizer.pkl",
    "class": "ruth.nlu.featurizers.sparse_featurizers.count_vector_featurizer.CountVectorFeaturizer"
}, Path("C:\Users\vinit\PycharmProjects\Research-implementation-NLU-engine\models\ruth_20220328-225708"))
