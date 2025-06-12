import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# 数据预处理
try:
    # 加载数据
    df = pd.read_csv('data/train.csv')
    
    # 编码分类特征
    label_encoders = {}
    categorical_cols = ['Soil Type', 'Crop Type', 'Fertilizer Name']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # 划分特征和标签
    X = df.drop(['id', 'Fertilizer Name'], axis=1)
    y = df['Fertilizer Name']
    
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 转换为DMatrix格式（XGBoost优化数据结构）
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 模型参数
    params = {
        'objective': 'multi:softprob',
        'num_class': len(y.unique()),
        'max_depth': 6,
        'eta': 0.3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss'
    }
    
    # 模型训练
    model = xgb.train(params, dtrain, num_boost_round=100,
                     evals=[(dtrain, 'train'), (dtest, 'test')],
                     early_stopping_rounds=10)
    
    # 预测与评估
    pred_proba = model.predict(dtest)
    top3_preds = [(-prob).argsort()[:3] for prob in pred_proba]
    
    # 计算准确率
    top1_acc = sum(y_test.values[i] in preds for i, preds in enumerate(top3_preds)) / len(y_test)
    
    print(f'\nTop1准确率: {top1_acc:.2%}')
    print('\n分类报告:\n', classification_report(y_test, pred_proba.argmax(axis=1)))
    
    # 保存模型和编码器
    model.save_model('xgboost_model.json')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    print('\n模型和编码器已保存！')

except Exception as e:
    print(f'\n训练过程出错: {str(e)}')