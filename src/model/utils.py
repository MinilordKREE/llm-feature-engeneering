# utils.py
import pandas as pd
from io import StringIO
import openai
import time
from sklearn.preprocessing import LabelEncoder
import csv
import numpy as np

def arff_to_dataframe(file_path):

    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract attribute names
    attributes = []
    for line in lines:
        if line.startswith("@attribute"):
            attributes.append(line.split()[1])
    
    # Extract data
    data_start_index = lines.index("@data\n") + 1
    data_lines = "\n".join(lines[data_start_index:])
    
    # Convert data lines to DataFrame
    df = pd.read_csv(StringIO(data_lines), header=None, names=attributes, na_values="?")
    
    # Replace missing values with -1
    df.fillna(-1, inplace=True)
    
    # Convert categorical string data into numbers
    for column in df.columns:
        if df[column].dtype == 'object':  # Check if the column is of object type (string)
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
    
    return df

def clean_csv(file_path, data_name=None):
    df = pd.read_csv(file_path)
    
    if data_name == "heart_disease":
        attribute_names = [
            "age", "sex", "cp", "trestbps", "chol",
            "fbs", "restecg", "thalach", "exang",
            "oldpeak", "slope", "ca", "thal", "num"
        ]
        processed_data = []
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            for record in csv_reader:
                record_dict = {}
                for i in range(len(attribute_names)):
                    if record[i] == "?":
                        record_dict[attribute_names[i]] = None
                    else:
                        record_dict[attribute_names[i]] = float(record[i])
                processed_data.append(record_dict)
        df = pd.DataFrame(processed_data).dropna()
        df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
        return df
    elif data_name == "circor":
        df=df.drop(columns=['Patient ID','Recording locations:','Additional ID'])
        df_clean = df.copy()
        df['Murmur locations'] = df['Murmur locations'].str.split('+')
        locations = ['PV', 'TV', 'AV', 'MV']
        for location in locations:
            df[location] = df['Murmur locations'].apply(lambda x: 1 if x is not np.nan and location in x else 0)
        df.drop('Murmur locations', axis=1, inplace=True)
        age_mapping = {'Neonate': 1, 'Infant': 2, 'Child': 3, 'Adolescent': 4, 'Young adult': 5}
        df_clean['Age'] = df_clean['Age'].map(age_mapping)
        df_clean['Age'].fillna(-1, inplace=True)
        le = LabelEncoder()
        df_clean['Sex'] = le.fit_transform(df_clean['Sex'])
        df_clean['Pregnancy status'] = df_clean['Pregnancy status'].map({False: 0, True: 1})

        df_clean['Height'].fillna((df_clean['Height'].mean()), inplace=True)
        df_clean['Weight'].fillna((df_clean['Weight'].mean()), inplace=True)
        df_clean['Murmur'] = df_clean['Murmur'].map({'Present': 1, 'Absent': 0, 'Unknown': 2})
        df_clean['Murmur locations'] = df_clean['Murmur locations'].str.split('+')
        locations = ['PV', 'TV', 'AV', 'MV']
        for location in locations:
            df_clean[location] = df_clean['Murmur locations'].apply(lambda x: 1 if x is not np.nan and location in x else 0)
        df_clean.drop('Murmur locations', axis=1, inplace=True)

        df_clean['Most audible location'] = df_clean['Most audible location'].map({np.nan: 0, 'PV': 1, 'TV': 2, 'AV': 3, 'MV': 4})

        df_clean['Outcome'] = df_clean['Outcome'].map({'Normal': 0, 'Abnormal': 1})
        df_clean['Campaign'] = df_clean['Campaign'].map({'CC2014': 0, 'CC2015': 1})
        string_features = ['Systolic murmur timing', 'Systolic murmur shape', 'Systolic murmur grading', 'Systolic murmur pitch', 'Systolic murmur quality', 
                        'Diastolic murmur timing', 'Diastolic murmur shape', 'Diastolic murmur grading', 'Diastolic murmur pitch', 'Diastolic murmur quality']
        for feature in string_features:
            df_clean[feature] = df_clean[feature].astype('category')
            df_clean[feature] = df_clean[feature].cat.codes
            df_clean[feature].fillna(-1, inplace=True)
        return df_clean
        
        
        
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def decoder_for_gpt3(input, max_length):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo-16k-0613',
                # model='gpt-4',
                messages=[
                    {"role": "system", "content": "Given a hypothetical patient profile, explore and articulate possible hypotheses, correlations, or insights that may arise based on common medical knowledge and assumptions. Consider the following attributes in your analysis: Age, Sex, Pregnancy status, Height, Weight, Presence of murmur, Most audible location of the murmur, Systolic and Diastolic murmur characteristics, Auscultation locations, and Campaign data. Note: The data is theoretical and should be treated as a fictional case study. Please provide your response in a consistent paragraph format."},
                    {"role": "user", "content": input}
                ],
                max_tokens=max_length,
                temperature=1,
            )
            # return response["choices"][0]['message']['content']
            content = response["choices"][0]['message']['content']
            return content
                
        except openai.error.RateLimitError as e:
            retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
            print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            
        except openai.error.ServiceUnavailableError as e:
            retry_time = 10  # Adjust the retry time as needed
            print(f"Service is unavailable. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            
        except openai.error.APIError as e:
            retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
            print(f"API error occurred. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)

        except OSError as e:
            retry_time = 5  # Adjust the retry time as needed
            print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
        except TimeoutError as e:
            retry_time = 60  # Adjust the retry time as needed
            print(f"Timeout error occurred: {e}. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
        except BaseException as e:
            retry_time = 60  # Adjust the retry time as needed
            print(f"Timeout error occurred: {e}. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)

        except openai.error.OpenAIError as e:
            raise e
def openai_key_config(api_key=None, key_file = '../api_keys.txt'):
    if api_key is not None:
        print(f'api_key: {api_key}')
        openai.api_key = api_key.strip()
        return
    if not os.path.exists(key_file):
        raise FileNotFoundError(f"Please enter your API key in {key_file}")
    api_key = open(key_file).readlines()[0].strip()
    print(f'api_key: {api_key}')
    openai.api_key = api_key