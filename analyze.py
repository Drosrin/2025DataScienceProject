import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei'] 

try:
    df = pd.read_csv('data/train.csv')
    df.drop(columns=['id'], inplace=True, errors='ignore')  # 新增id列移除
    print(f'成功加载{len(df)}条数据')

    # 数值特征分析
    print('\n数值特征统计:')
    print(df.describe())

    # 类别特征分布
    plt.figure(figsize=(10,6))
    df['Fertilizer Name'].value_counts().plot(kind='bar')
    plt.title('肥料类型分布')
    plt.xticks(rotation=45)
    plt.savefig('fertilizer_distribution.png')

    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title('特征相关性热力图')
    plt.savefig('correlation_heatmap.png')
    # 结论 类别均衡；没有多重共线性

    # 各特征分布子图
    numeric_features = df.select_dtypes(include=['float64','int64']).columns.difference(['id'])  # 排除可能残留的id列
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(numeric_features, 1):
        plt.subplot(2, 3, i)  # 改为2x3布局
        sns.histplot(df[feature], kde=True)
        plt.title(f'{feature}分布')
    plt.tight_layout()
    plt.savefig('feature_distributions.png')

    # 按肥料类型分组分布
    plt.figure(figsize=(15, 8))  # 优化图形尺寸
    for i, feature in enumerate(numeric_features, 1):
        plt.subplot(2, 3, i)  # 改为2x3布局
        sns.boxplot(x='Fertilizer Name', y=feature, data=df)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_distributions_by_fertilizer.png')
    
except Exception as e:
    print(f'发生错误: {e}')