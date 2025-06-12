import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei'] 

try:
    df = pd.read_csv('data/train.csv')
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

except Exception as e:
    print(f'发生错误: {e}')