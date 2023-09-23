import os
import argparse
from model import *
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(probability=True),  # Enable probability estimates
}
methods = ['baseline', 'SelectK', 'PCA']
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def config():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--data_name', type=str, required=True, help='Dataset chosen')
    parser.add_argument('--df_path', type=str, required=True, help='Path to the main dataframe')
    parser.add_argument('--column_path', type=str, required=True, help='Path to the column dataframe')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--methods', type=list, required=False, help='List of methods to use for evaluation')
    parser.add_argument('--api_key', type=str, default=None, help='openai api key')
    parser.add_argument('--openai_key_txt_file', type=str, default='../api_keys.txt', help='The txt file that contains openai api key.')

    args = parser.parse_args()

    args = vars(args)
    
    openai_key_config(args['api_key'], args['openai_key_txt_file'])
    
    return args


def main(args):
    evaluator = ModelEvaluator(**args)
    evaluator.evaluate_models(models=models,methods=methods)
    return

if __name__ == '__main__':
    args = config()
    main(args)