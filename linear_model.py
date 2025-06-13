import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

def train():
    try:
        df = pd.read_csv('data/train.csv')

        label_encoders = {}
        categorical_cols = ['Soil Type', 'Crop Type', 'Fertilizer Name']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        X = df.drop(['id', 'Fertilizer Name'], axis=1)
        y = df['Fertilizer Name']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print(f'\nTop1准确率: {model.score(X_test, y_test):.2%}')
        print('\n分类报告:\n', classification_report(y_test, y_pred))

        joblib.dump(model, 'linear_model.pkl')
        joblib.dump(label_encoders, 'linear_label_encoders.pkl')
        print('\n模型和编码器已保存！')

    except Exception as e:
        print(f'\n训练过程出错: {str(e)}')

def test():
    test_df = pd.read_csv('data/test.csv')
    test_df.drop(columns=['id'], inplace=True, errors='ignore')

    model = joblib.load('linear_model.pkl')
    label_encoders = joblib.load('linear_label_encoders.pkl')

    for col in ['Crop Type', 'Soil Type']:
        test_df[col] = label_encoders[col].transform(test_df[col])

    proba = model.predict_proba(test_df)
    top3_indices = [(-prob).argsort()[:3] for prob in proba]
    fertilizer_encoder = label_encoders['Fertilizer Name']
    
    top3_fertilizers = [
        ' '.join(fertilizer_encoder.inverse_transform(indices))
        for indices in top3_indices
    ]

    submission = pd.DataFrame({
        'id': range(750000, 750000 + len(test_df)),
        'Fertilizer Name': top3_fertilizers
    })
    submission.to_csv('data/submission_linear.csv', index=False)
    print('预测结果已保存至 data/submission_linear.csv')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='线性模型训练和测试')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--test', action='store_true', help='测试模型')
    args = parser.parse_args()
    if args.train:
        train()
    if args.test:
        test()