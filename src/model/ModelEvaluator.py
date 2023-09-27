import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_val_score, KFold
from .utils import *
import openai
import os 
import json
import time
from sklearn.metrics import accuracy_score, roc_auc_score
from Fine_tune import base
import datetime
import json
    
class ModelEvaluator:
    def __init__(self, data_name, df_path, column_path, target, methods, **kwargs):
        self.data_name = data_name
        if self.data_name == "circor" or self.data_name == "heart_disease":
            self.df = clean_csv(df_path, data_name).reset_index(drop=True)
        else:
            self.df = arff_to_dataframe(df_path, data_name).reset_index(drop=True)
        self.column = pd.read_csv(column_path).reset_index(drop=True)
        self.original_columnlist = self.df.columns.drop(target).tolist()
        self.df['response'] = self.column.iloc[:, 0]
        self.target = target
        self.methods = methods if methods is not None else []
        self.data_name =data_name
        
    @staticmethod
    def get_matching_cols(df, regex):
        r = re.compile(regex)
        return list(filter(r.match, df.columns))

    @staticmethod
    def get_embedding_cols(df):
        return ModelEvaluator.get_matching_cols(df, "(vec_\d+)")

    @staticmethod
    def get_embedding(text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

    @staticmethod
    def explode(col, prefix):
        n_cols = len(col[0])
        col_names = [prefix + str(i) for i in range(n_cols)]
        return pd.DataFrame(col.to_list(), columns=col_names)

    def method_baseline(self, df, scaler=None):
        X = df[self.original_columnlist]
        y = df[self.target]
        if scaler is None:
            scaler = StandardScaler()
            X_final = scaler.fit_transform(X)
        else:
            X_final = scaler.transform(X)
        
        return X_final, y, scaler


    def fit_PCA(self,df):
        def explode(col, prefix):
            n_cols = len(col[0])
            col_names = [prefix + str(i) for i in range(n_cols)]
            return pd.DataFrame(col.to_list(), columns=col_names)

        # Explode text_vector
        exploded = explode(df['text_vector'], 'vec_')
        df.loc[:, exploded.columns] = exploded

        y = df[self.target]
        X_cat = df[self.original_columnlist]
        embed_cols = ModelEvaluator.get_embedding_cols(df)
        X_text = df[embed_cols]
        X_comb = pd.concat([X_cat, X_text], axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_comb)

        best_n_components = None
        best_score = float('-inf')
        for n_components in range(1, 50):
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)

            model = LogisticRegression()
            score = cross_val_score(model, X_pca, y, cv=5, scoring='roc_auc').mean()

            if score > best_score:
                best_score = score
                best_n_components = n_components

        pca = PCA(n_components=best_n_components)
        pca.fit(X_scaled)
        return pca, best_n_components, scaler

    def transform_with_PCA(self, pca, scaler, df):
        def explode(col, prefix):
            n_cols = len(col[0])
            col_names = [prefix + str(i) for i in range(n_cols)]
            return pd.DataFrame(col.to_list(), columns=col_names)

        # Explode text_vector
        exploded = explode(df['text_vector'], 'vec_')
        df.loc[:, exploded.columns] = exploded

        y = df[self.target]
        X_cat = df[self.original_columnlist]
        embed_cols = ModelEvaluator.get_embedding_cols(df)
        X_text = df[embed_cols]
        X_comb = pd.concat([X_cat, X_text], axis=1)
        X_scaled = scaler.transform(X_comb)

        X_pca = pca.transform(X_scaled)
        X_final = pd.DataFrame(X_pca)
        X_final.columns = [f'PC{i+1}' for i in range(X_final.shape[1])]

        return X_final, y

    def method_SelectK(self,df, scaler=None, selector=None):
        def explode(col, prefix):
            n_cols = len(col[0])
            col_names = [prefix + str(i) for i in range(n_cols)]
            return pd.DataFrame(col.to_list(), columns=col_names)

        # Explode text_vector
        exploded = explode(df['text_vector'], 'vec_')
        df.loc[:, exploded.columns] = exploded

        y = df[self.target]
        X_cat = df[self.original_columnlist]
        embed_cols = ModelEvaluator.get_embedding_cols(df)
        X_text = df[embed_cols]
        X_comb = pd.concat([X_cat, X_text], axis=1)

        # Scale the data
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_comb)
        else:
            X_scaled = scaler.transform(X_comb)

        # Feature selection
        if selector is None:
            selector = SelectKBest(mutual_info_classif, k=20)
            X_selected = selector.fit_transform(X_scaled, y)
        else:
            X_selected = selector.transform(X_scaled)

        X_final = pd.concat([X_cat, pd.DataFrame(X_selected)], axis=1)
        X_final.columns = X_final.columns.astype(str)

        return X_final, y, scaler, selector


    def evaluate_models(self, train_df, test_df, models, methods):
        method_results = {}
        
        for method in methods:
            method_results[method] = {}

            if method == 'baseline':
                X_train, y_train, scaler = self.method_baseline(train_df)
                X_test, y_test, _ = self.method_baseline(test_df, scaler)
            elif method == 'PCA':
                pca, best_n_components, scaler = self.fit_PCA(train_df)
                X_train, y_train = self.transform_with_PCA(pca, scaler, train_df)
                X_test, y_test = self.transform_with_PCA(pca, scaler, test_df)
            elif method == 'SelectK':
                X_train, y_train, train_scaler, train_selector = self.method_SelectK(train_df)
                X_test, y_test, _, _ = self.method_SelectK(test_df, train_scaler, train_selector)

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                for metric in ['accuracy', 'roc_auc']:
                    if metric == 'accuracy':
                        score = accuracy_score(y_test, y_pred)
                    elif metric == 'roc_auc':
                        y_prob = model.predict_proba(X_test)[:, 1]  # assuming binary classification
                        score = roc_auc_score(y_test, y_prob)

                    if metric not in method_results[method]:
                        method_results[method][metric] = {}
                    if model_name not in method_results[method][metric]:
                        method_results[method][metric][model_name] = []

                    method_results[method][metric][model_name].append(score)

                    print(f'Method: {method} | Model: {model_name} | {metric}: {score}')

        return method_results
    
    def run(self,data_name, models, methods,seeds,metrics_list,colors):
        all_results = {}

        for seed in seeds:
            print(f"Processing seed {seed}...")
            
            # Adjust the paths to load the data based on the current seed
            train_data = pd.read_csv(f'/data/chenxi/llm-feature-engeneering/src/Fine_tune/{data_name}/data_seed_{seed}/train.csv')
            test_data = pd.read_csv(f'/data/chenxi/llm-feature-engeneering/src/Fine_tune/{data_name}/data_seed_{seed}/test.csv')
            
            generator = base.EmbeddingGeneratorForNLPSequenceClassification.from_use_case(
        use_case="NLP.SequenceClassification",
        model_name="distilbert-base-uncased",
        tokenizer_max_length=512
    )
            
            train_data['text_vector'] = generator.generate_embeddings(text_col=train_data['response'])
            test_data['text_vector'] = generator.generate_embeddings(text_col=test_data['response'])
            
            # Evaluate the models and store the results
            seed_results = self.evaluate_models(train_data, test_data, models, methods)
            all_results[seed] = seed_results
        output_json = {}

        for method in methods:
            output_json[method] = {}
            for model_name in models.keys():
                scores = [all_results[seed][method]['roc_auc'][model_name][0] for seed in seeds]
                
                # Calculate median, mean, and standard deviation
                median_score = np.median(scores)
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                # Create performance strings for both mean and median
                mean_performance_str = f"{mean_score:.4f} ± {std_score:.2f}"
                median_performance_str = f"{median_score:.4f} ± {std_score:.2f}"
                
                output_json[method][model_name] = {
                    "mean_performance": mean_performance_str,
                    "median_performance": median_performance_str,
                    "mean": mean_score,
                    "median": median_score,
                    "std": std_score
                }
        # Save the JSON structure to a file
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_path = f"/data/chenxi/llm-feature-engeneering/log/{data_name}/new/{current_time}.json"
        with open(file_path, 'w') as file:
            json.dump(output_json, file, indent=4)

        # Now, plot the combined results
        for metric in metrics_list:
            plt.figure(figsize=(15, 10))
            
            x_ticks_positions = np.arange(len(models))
            for i, method in enumerate(methods):
                scores_for_all_models = []
                for j, model_name in enumerate(models.keys()):
                    scores = [all_results[seed][method][metric][model_name][0] for seed in seeds]
                    scores_for_all_models.append(scores)
                
                # Plot boxplots for all models for the current method
                bp = plt.boxplot(scores_for_all_models, positions=x_ticks_positions + i * 0.2, widths=0.15,
                                patch_artist=True, boxprops=dict(facecolor=colors[i], alpha=0.6))
                for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                    plt.setp(bp[element], color=colors[i])
                plt.setp(bp["boxes"], facecolor=colors[i])
                plt.setp(bp["fliers"], markeredgecolor=colors[i])
            
            plt.xticks(ticks=x_ticks_positions, labels=models.keys(), rotation=45)
            plt.legend(handles=[mpatches.Patch(color=colors[i], label=method) for i, method in enumerate(methods)], loc='upper right')
            plt.title(f"Model performance ({metric}) across seeds")
            plt.ylabel(metric)
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(f"/data/chenxi/llm-feature-engeneering/log/{data_name}/{metric}_{current_time}.png")
            plt.show()
            
        # def evaluate_models(self, models, methods):
    #     colors = ['black', 'green', 'blue', 'red']

    #     # Generate a unique timestamp
    #     timestamp = time.strftime("%Y%m%d-%H%M%S")
    #     log_dir = f'log/{self.data_name}/{timestamp}'

    #     # Ensure the log directory exists
    #     if not os.path.exists(log_dir):
    #         os.makedirs(log_dir)

    #     results = {}  # Dictionary to store mean and median values

    #     for metric in ['accuracy', 'roc_auc']:
    #         plt.figure(figsize=(15, 10))

    #         for i, method in enumerate(methods):
    #             if method == 'baseline':
    #                 self.df = self.prepare_data()
    #                 X_final, y = self.method_baseline()
    #             elif method == 'PCA':
    #                 self.df = self.prepare_data()
    #                 X_final, y = self.method_PCA()
    #             elif method == 'SelectK':
    #                 self.df = self.prepare_data()
    #                 X_final, y, best_k = self.method_SelectK()
    #                 print(best_k)
    #             else:
    #                 raise ValueError(f"Unknown method: {method}")

    #             kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    #             performance_metrics = {metric: {model_name: cross_val_score(model, X_final, y, cv=kfold, scoring=metric) for model_name, model in models.items()}}

    #             for name, scores in performance_metrics[metric].items():
    #                 print(f'Method: {method}, Model: {name}, {metric}: {scores.mean()} ± {scores.std()}')
                    
    #                 # Update the results dictionary
    #                 if method not in results:
    #                     results[method] = {}
    #                 results[method][name] = {
    #                     'performance': f"{scores.mean():.4f} ± {scores.std():.2f}",  # Format the string as mean ± std
    #                     'median': np.median(scores)
    #                 }

    #             x_ticks_positions = np.arange(len(models)) + i * 0.2
    #             plt.boxplot([performance_metrics[metric][model_name] for model_name in models.keys()], positions=x_ticks_positions, widths=0.2, patch_artist=True,
    #                         boxprops=dict(facecolor=colors[i], color=colors[i], alpha=0.6),
    #                         capprops=dict(color=colors[i]),
    #                         whiskerprops=dict(color=colors[i]),
    #                         flierprops=dict(color=colors[i], markeredgecolor=colors[i]),
    #                         medianprops=dict(color='black'))

    #         plt.legend(handles=[mpatches.Patch(color=colors[i], label=methods[i]) for i in range(len(methods))], loc='upper right')
    #         plt.title(f"Model performance ({metric})")
    #         plt.ylabel(metric)
    #         plt.xticks(ticks=np.arange(len(models)), labels=models.keys())

    #         # Save the plot to the unique log directory
    #         plt.savefig(f'{log_dir}/{metric}_performance.png')
    #         plt.show()

    #     # Save the results to a JSON file in the unique log directory
    #     with open(f'{log_dir}/results.json', 'w') as json_file:
    #         json.dump(results, json_file, indent=4)