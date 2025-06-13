import argparse
import importlib
import sys
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='模型训练测试启动器')
    parser.add_argument('--model', type=str, required=True, help='模型名称（对应xxx_model.py）')
    parser.add_argument('--train', action='store_true', help='是否执行训练')
    parser.add_argument('--test', action='store_true', help='是否执行测试')
    args = parser.parse_args()

    # 动态导入模型模块（支持当前目录下的普通Python文件）
    model_module_name = f'{args.model}_model'
    model_file_path = os.path.join(os.getcwd(), f'{model_module_name}.py')
    if not os.path.exists(model_file_path):
        raise ValueError(f'未找到模型文件{model_file_path}，请检查文件名是否符合xxx_model.py格式')
    
    # 将当前目录添加到模块搜索路径
    sys.path.insert(0, os.getcwd())
    try:
        model_module = importlib.import_module(model_module_name)
    except ImportError as e:
        raise ValueError(f'导入模型模块{model_module_name}失败: {str(e)}')
    finally:
        # 移除临时添加的路径，避免影响后续导入
        sys.path.pop(0)

    # 检查并调用train函数
    if args.train:
        if hasattr(model_module, 'train'):
            model_module.train()
        else:
            print(f'警告：模型{args.model}缺少train()函数，跳过训练')

    # 检查并调用test函数
    if args.test:
        if hasattr(model_module, 'test'):
            model_module.test()
        else:
            print(f'警告：模型{args.model}缺少test()函数，跳过测试')


if __name__ == '__main__':
    main()