import numpy as np
import script.config as cfg
from sklearn.pipeline import Pipeline
from skpdspipe.pipeline import DFFeatureUnion, DFColumnsSelector, DataFrameUnion
from skpdspipe.apply import DFApplyFn, DFStringify, DFUseApplyFn, DFSetDType
from skpdspipe.impute import DFSimpleImputer
from skpdspipe.encode import DFKBinsDiscretizer, DFOrdinalEncoder

# This pipelines will preprocess raw data
numerical_pipe = Pipeline([
    ('slctr', DFColumnsSelector(cfg.NUMERIC_FEATURES)),
    ('imptr', DFSimpleImputer(strategy='constant',
                              fill_value=-9)),
    ('dtype', DFSetDType(dtype='float'))
])

categorical_pipe = Pipeline([
    ('slctr', DFColumnsSelector(cfg.CATEGORICAL_FEATURES)),
    ('encdr', DFOrdinalEncoder()),
    ('string', DFStringify()),
    ('dtype', DFSetDType(dtype='category'))
])

ordinal_pipe = Pipeline([
    ('slctr', DFColumnsSelector(cfg.ORDINAL_FEATURES)),
    ('dtype', DFSetDType(dtype='int'))
])

bined_pipe = Pipeline([
    ('scltr', DFColumnsSelector(cfg.BIN_FEATURES)),
    ('bin', DFKBinsDiscretizer(n_bins=10,
                               encode='ordinal',
                               dtype='float',
                               fill_value=np.nan)),
    ('string', DFStringify()),
    ('dtype', DFSetDType(dtype='category'))
])

# Union of cleaned feature
pp_union = DFFeatureUnion([
    ('numeric', numerical_pipe),
    ('categorical', categorical_pipe),
    ('ordinal', ordinal_pipe),
    ('bined', bined_pipe)
])


def fn_fare_per_mile(X): return X.fare / (X.trip_miles+1)
def fn_fare_per_sec(X): return X.fare / (X.trip_seconds+1)
def fn_mile_per_sec(X): return X.trip_miles / (X.trip_seconds+1)


def make_numeric_feature(name, subset, fn, use=True):
    return Pipeline([
        ('slct', DFColumnsSelector(subset)),
        ('func', DFApplyFn(name=name, fn=fn)),
        ('imptr', DFSimpleImputer(strategy='constant',
                                  fill_value=-9)),
        ('use', DFUseApplyFn(use=use))
    ])


# Union of engineered feature
fe_union = DFFeatureUnion([
    ('fare_per_mile', make_numeric_feature(
        name='fare_per_mile',
        subset=['fare', 'trip_miles'],
        fn=fn_fare_per_mile,
        use=True
    )),
    ('fare_per_sec', make_numeric_feature(
        name='fare_per_sec',
        subset=['fare', 'trip_seconds'],
        fn=fn_fare_per_sec,
        use=True
    )),
    ('mile_per_sec', make_numeric_feature(
        name='mile_per_sec',
        subset=['trip_miles', 'trip_seconds'],
        fn=fn_mile_per_sec,
        use=True
    ))
])

# Union of all features
final_union = DFFeatureUnion([
    ('cleaned', pp_union),
    ('feateng', fe_union)
])
