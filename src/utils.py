# list of models @ https://platform.openai.com/docs/models/continuous-model-upgrades
# cost @ https://openai.com/pricing and https://platform.openai.com/docs/deprecations/ for older models
import os

import pandas as pd


def load_with_conditions(filename, overwrite_res=False):
    if not os.path.exists(filename) or overwrite_res:
        print("File not found or overwrite requested. Creating new dataframe.")
        df = pd.DataFrame()
    elif filename.split(".")[-1] == "csv":
        print("Loading existing dataframe.")
        df = pd.read_csv(filename)
    elif filename.split(".")[-1] == "pkl":
        print("Loading existing dataframe.")
        df = pd.read_pickle(filename)
    else:
        raise ValueError("File format not recognized. Please use .csv or .pkl.")

    return df


def save_dataframe(filename, res_df):
    if filename.endswith(".csv"):
        res_df.to_csv(filename, index=False)
    elif filename.endswith(".pkl"):
        res_df.to_pickle(filename)
    else:
        raise ValueError("filename not recognized")


llms_dict = {'01-ai/yi-34b': '01-ai/yi-34b',
             '01-ai/yi-6b': '01-ai/yi-6b',
             'AlephAlpha/luminous-base': 'AlephAlpha/luminous-base',
             'AlephAlpha/luminous-extended': 'AlephAlpha/luminous-extended',
             'AlephAlpha/luminous-supreme': 'AlephAlpha/luminous-supreme',
             'ai21/j2-grande': 'ai21/j2-grande',
             'ai21/j2-jumbo': 'ai21/j2-jumbo',
             'anthropic/claude-2.0': 'anthropic/claude-2.0',
             'anthropic/claude-2.1': 'anthropic/claude-2.1',
             'anthropic/claude-instant-1.2': 'anthropic/claude-instant-1.2',
             'anthropic/claude-v1.3': 'anthropic/claude-v1.3',
             'cohere/command': 'cohere/command',
             'cohere/command-light': 'cohere/command-light',
             'google/text-bison@001': 'google/text-bison@001',
             'google/text-unicorn@001': 'google/text-unicorn@001',
             'meta/llama-2-13b': 'meta/llama-2-13b',
             'meta/llama-2-70b': 'meta/llama-2-70b',
             'meta/llama-2-7b': 'meta/llama-2-7b',
             'meta/llama-65b': 'meta/llama-65b',
             'mistralai/mistral-7b-v0.1': 'mistralai/mistral-7b-v0.1',
             'mistralai/mixtral-8x7b-32kseqlen': 'mistralai/mixtral-8x7b-32kseqlen',
             'gpt-3.5-turbo-0613': 'openai/gpt-3.5-turbo-0613',
             'gpt-4-0613': 'openai/gpt-4-0613',
             'gpt-4-1106-preview': 'openai/gpt-4-1106-preview',
             'text-davinci-002': 'openai/text-davinci-002',
             'text-davinci-003': 'openai/text-davinci-003',
             'tiiuae/falcon-40b': 'tiiuae/falcon-40b',
             'tiiuae/falcon-7b': 'tiiuae/falcon-7b',
             'writer/palmyra-x-v2': 'writer/palmyra-x-v2',
             'writer/palmyra-x-v3': 'writer/palmyra-x-v3'}
llms_helm = list(llms_dict.keys())

# keep two families of models (anthropic and meta's llama) in the test set
train_llms_helm = [
    '01-ai/yi-6b',
    '01-ai/yi-34b',
    'AlephAlpha/luminous-base',
    'AlephAlpha/luminous-supreme',
    'ai21/j2-grande',
    'ai21/j2-jumbo',
    'cohere/command',
    'google/text-bison@001',
    'google/text-unicorn@001',
    'mistralai/mixtral-8x7b-32kseqlen',
    'mistralai/mistral-7b-v0.1',
    'gpt-3.5-turbo-0613',
    'gpt-4-1106-preview',
    'text-davinci-002',
    'text-davinci-003',
    'tiiuae/falcon-7b',
    'writer/palmyra-x-v3',
    'writer/palmyra-x-v2',
]

validation_llms_helm = [
    'tiiuae/falcon-40b',
    'gpt-4-0613',
    'AlephAlpha/luminous-extended',
    'cohere/command-light',
]

test_llms_helm = [
    'anthropic/claude-2.1',
    'anthropic/claude-2.0',
    'anthropic/claude-instant-1.2',
    'anthropic/claude-v1.3',
    'meta/llama-2-70b',
    'meta/llama-2-13b',
    'meta/llama-2-7b',
    'meta/llama-65b',
]

llms_reasoning = [
    'text-ada-001',
    'text-babbage-001',
    'text-curie-001',
    'text-davinci-001',
    'text-davinci-002',
    'text-davinci-003',
    'gpt-3.5-turbo-0301',
    'gpt-3.5-turbo-0613',
    'gpt-3.5-turbo-1106',
    'gpt-3.5-turbo-0125',
    'gpt-4-0314',
    'gpt-4-0613',
    'gpt-4-1106-preview',
    'gpt-4-0125-preview',
]
train_llms_reasoning = [
    'text-ada-001',
    'text-babbage-001',
    'text-curie-001',
    'text-davinci-001',
    'text-davinci-002',
    'gpt-3.5-turbo-0301',
    'gpt-3.5-turbo-0613',
    'gpt-3.5-turbo-1106',
]

validation_llms_reasoning = [
    'text-davinci-003',
    'gpt-3.5-turbo-0125',
]

test_llms_reasoning = [
    'gpt-4-0125-preview',
    'gpt-4-0314',
    'gpt-4-0613',
    'gpt-4-1106-preview',
]
