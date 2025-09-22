import pandas as pd
from churn_ml.config import TrainConfig
from churn_ml.model import train_model


def test_train_model(tmp_path):
    df = pd.DataFrame({
        'feat_num': [1,2,3,4,5,6,7,8],
        'feat_cat': ['a','b','a','b','a','b','a','b'],
        'Churn': [0,1,0,1,0,1,0,1],
    })
    config = TrainConfig(target='Churn', test_size=0.25, random_state=0)
    model = train_model(df, config, out_path=tmp_path / 'model.joblib')
    assert model is not None
