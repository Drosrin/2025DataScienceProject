# DataSciProj
analyze - 简单的可视化

启动模型：
```shell
python launcher.py --model [model_name, e.g. xgboost] [--train] [--test]
```
模型要求：必须以[model_name]_model.py的形式存在，且其中必须有train和test函数。两个函数都不存在入参。
模型需要读取data/train.csv作为训练数据，data/test.csv作为测试数据。测试数据的结果保存在data/{model_name}_submission.csv中。