import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_val_score, KFold
from .utils import *
import openai
import os 
import json
import time
    
class ModelEvaluator:
    def __init__(self, data_name, df_path, column_path, target, methods, **kwargs):
        if data_name == r"circor|heart_disease":
            self.df = clean_csv(df_path, data_name).reset_index(drop=True)
        else:
            self.df = arff_to_dataframe(df_path, data_name).reset_index(drop=True)
        self.column = pd.read_csv(column_path).reset_index(drop=True)
        self.original_columnlist = self.df.columns.drop(target).tolist()
        self.df['response'] = self.column.iloc[:, 0]
        self.target = target
        self.prepare_data()
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

    def prepare_data(self):
        self.df['text_vector'] = self.df['response'].apply(lambda x: self.get_embedding(x, model='text-embedding-ada-002'))
        
        tab_vec_name = 'text_vector'
        prefix = "vec_"
        exploded = self.explode(self.df[tab_vec_name], prefix)
        self.df.loc[:, exploded.columns] = exploded

    def method_baseline(self):
        X = self.df[self.original_columnlist]
        y = self.df[self.target]
        scaler = StandardScaler()
        X_final = scaler.fit_transform(X)
        return X_final, y

    def method_PCA(self):
        X = self.df.drop(self.target, axis=1)
        y = self.df[self.target]

        X_cat = self.df[self.original_columnlist]
        embed_cols = ModelEvaluator.get_embedding_cols(self.df)
        X_text = self.df[embed_cols]

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
        X_pca = pca.fit_transform(X_scaled)

        X_final = pd.concat([X_cat, pd.DataFrame(X_pca)], axis=1)
        X_final.columns = X_final.columns.astype(str)

        return X_final, y

    def method_SelectK(self):
        y = self.df[self.target]
        X_cat = self.df[self.original_columnlist]
        embed_cols = ModelEvaluator.get_embedding_cols(self.df)
        X_text = self.df[embed_cols]
        X_comb = pd.concat([X_cat, X_text], axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_comb)
        
        possible_k_values = list(range(1, 50))
        
        best_score = -np.inf
        best_k = None
        best_features = None

        model = SVC(probability=True)

        for k in possible_k_values:
            selector = SelectKBest(mutual_info_classif, k=k)
            X_selected = selector.fit_transform(X_scaled, y)

            score = cross_val_score(model, X_selected, y, cv=5, scoring='roc_auc').mean()            

            if score > best_score:
                best_score = score
                best_k = k
                best_features = X_selected

        X_final = pd.concat([X_cat, pd.DataFrame(best_features)], axis=1)
        X_final.columns = X_final.columns.astype(str)
        return X_final, y, best_k

    def evaluate_models(self, models, methods):
        colors = ['black', 'green', 'blue', 'red']

        # Generate a unique timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = f'log/{self.data_name}/{timestamp}'

        # Ensure the log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        results = {}  # Dictionary to store mean and median values

        for metric in ['accuracy', 'roc_auc']:
            plt.figure(figsize=(15, 10))

            for i, method in enumerate(methods):
                if method == 'baseline':
                    X_final, y = self.method_baseline()
                elif method == 'PCA':
                    X_final, y = self.method_PCA()
                elif method == 'SelectK':
                    X_final, y, best_k = self.method_SelectK()
                    print(best_k)
                else:
                    raise ValueError(f"Unknown method: {method}")

                kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                performance_metrics = {metric: {model_name: cross_val_score(model, X_final, y, cv=kfold, scoring=metric) for model_name, model in models.items()}}

                for name, scores in performance_metrics[metric].items():
                    print(f'Method: {method}, Model: {name}, {metric}: {scores.mean()} ± {scores.std()}')
                    
                    # Update the results dictionary
                    if method not in results:
                        results[method] = {}
                    results[method][name] = {
                        'mean': scores.mean(),
                        'median': np.median(scores)
                    }

                x_ticks_positions = np.arange(len(models)) + i * 0.2
                plt.boxplot([performance_metrics[metric][model_name] for model_name in models.keys()], positions=x_ticks_positions, widths=0.2, patch_artist=True,
                            boxprops=dict(facecolor=colors[i], color=colors[i], alpha=0.6),
                            capprops=dict(color=colors[i]),
                            whiskerprops=dict(color=colors[i]),
                            flierprops=dict(color=colors[i], markeredgecolor=colors[i]),
                            medianprops=dict(color='black'))

            plt.legend(handles=[mpatches.Patch(color=colors[i], label=methods[i]) for i in range(len(methods))], loc='upper right')
            plt.title(f"Model performance ({metric})")
            plt.ylabel(metric)
            plt.xticks(ticks=np.arange(len(models)), labels=models.keys())

            # Save the plot to the unique log directory
            plt.savefig(f'{log_dir}/{metric}_performance.png')
            plt.show()

        # Save the results to a JSON file in the unique log directory
        with open(f'{log_dir}/results.json', 'w') as json_file:
            json.dump(results, json_file, indent=4)