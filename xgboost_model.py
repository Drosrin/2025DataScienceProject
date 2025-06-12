import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.preprocessing import LabelEncoder 
def train():
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

def test():
    # 加载测试数据
    test_df = pd.read_csv('data/test.csv')
    test_df.drop(columns=['id'], inplace=True, errors='ignore')
    # 加载标签编码器
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = joblib.load(f)
    
    
    # 加载预处理对象和模型
    model = xgb.Booster()
    model.load_model('xgboost_model.json')

    # 特征预处理
    # 应用标签编码
    category_columns = ['Crop Type', 'Soil Type']
    for col in category_columns:
        test_df[col] = label_encoders[col].transform(test_df[col])
    
    dtest = xgb.DMatrix(test_df)
    
    # 生成预测结果
    proba = model.predict(dtest)
    top3_indices = [(-prob).argsort()[:3] for prob in proba]
    fertilizer_encoder = label_encoders['Fertilizer Name']
    top3_fertilizers = [
        ' '.join(fertilizer_encoder.inverse_transform(indices))
        for indices in top3_indices
    ]

    # 创建提交文件
    submission = pd.DataFrame({
        'id': range(750000, 750000 + len(test_df)),
        'Fertilizer Name': top3_fertilizers
    })
    submission.to_csv('data/submission.csv', index=False)
    print('预测结果已保存至 data/submission.csv')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='XGBoost模型训练和测试')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--test', action='store_true', help='测试模型')
    args = parser.parse_args()
    if args.train:
        train()
    if args.test:
        test()
