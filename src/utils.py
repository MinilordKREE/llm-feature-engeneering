# utils.py
import pandas as pd
from io import StringIO
import openai
import time
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
    df = pd.read_csv(StringIO(data_lines), header=None, names=attributes)
    
    return df

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
