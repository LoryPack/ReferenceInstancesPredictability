import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from time import sleep
from typing import List

import numpy as np
import openai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from tqdm import tqdm

from src.embedding import Embedder

tqdm.pandas()

sort_models_order = ['text-ada-001',
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


def ngram_vectorize_new(train_texts, val_texts, test_texts, ngram_range=(1, 2), token_mode='word',
                        min_document_frequency=2, system_prompt=None):
    """Vectorizes texts as n-gram vectors.

    # Arguments
        train_texts: list, training text strings.
        val_texts: list, validation text strings.
        test_texts: list, test text strings.
        ngram_range: tuple (min, max). The range of n-gram values to consider.
        token_mode: Whether text should be split into word or character n-grams. 'word' or 'char'.
        min_document_frequency: The minimum document frequency below which a token will be discarded.
        system_prompt: str. The system prompt to add to the beginning of each text.

    # Returns
        x_train, x_val, x_test: vectorized training, validation and test texts
        vectorizer: the fitted vectorizer
    """

    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
        'ngram_range': ngram_range,  # Use 1-grams + 2-grams.
        'dtype': 'int64',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': token_mode,  # Split text into word tokens.
        'min_df': min_document_frequency,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # add the system prompt
    if system_prompt is not None:
        train_texts = [f"{system_prompt} {text}" for text in train_texts]
        val_texts = [f"{system_prompt} {text}" for text in val_texts]
        test_texts = [f"{system_prompt} {text}" for text in test_texts]

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts).toarray()

    # Vectorize val texts.
    x_val = vectorizer.transform(val_texts).toarray()

    # Vectorize test texts.
    x_test = vectorizer.transform(test_texts).toarray()

    return x_train, x_val, x_test, vectorizer


def select_features(x_train, train_labels, x_val, x_test, top_k=20000):
    """Selects top 'k' of the vectorized features.

    # Arguments
        x_train: np.ndarray, vectorized training texts.
        train_labels: np.ndarray, training labels.
        x_val: np.ndarray, vectorized val texts.
        x_test: np.ndarray, vectorized test texts.
        top_k: int. The number of top features to select.

    # Returns
        x_train, x_val, x_test: training, validation and test texts with selected features
        selector: the fitted selector
    """

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(top_k, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float64')
    x_val = selector.transform(x_val).astype('float64')
    x_test = selector.transform(x_test).astype('float64')

    return x_train, x_val, x_test, selector


class ResultsLoader(ABC):
    default_llms: List[str]

    def __init__(self, llms=None, base_path_raw=None, processed_filename=None, verbose=False, load_results_kwargs={},
                 load_processed_kwargs={}):
        self.verbose = verbose

        if llms is None:
            self.llms = self.default_llms
        else:
            self.llms = llms

        if processed_filename is not None:
            self.results_df = self.load_processed(processed_filename, **load_processed_kwargs)
            # discard all rows for which no success has been recorded for any llm
            self.results_df = self.results_df[
                ~self.results_df[[f"Success_{llm}" for llm in self.llms]].isna().all(axis=1)]
        else:
            if not base_path_raw is None:
                # add it to the kwargs
                load_results_kwargs["base_path"] = base_path_raw

            self.results_df = self.load_results(**load_results_kwargs)
            # shuffle the dataframe
            self.results_df = self.results_df.sample(frac=1, random_state=42)
            # reset index
            self.results_df = self.results_df.reset_index(drop=True)

    @abstractmethod
    def load_results(self, base_path):
        """This should return a single dataframe which includes the prompt, a column for each feature and a success
         column for each llm"""
        pass

    @staticmethod
    def load_processed(filename, compression=False, compression_kwargs=None):
        """
        Load the data from the processed json file and return a DataFrame with the data.
        """
        if not compression:
            compression_kwargs = None
        elif compression and compression_kwargs is None:
            compression_kwargs = {'method': 'gzip', 'compresslevel': 1, 'mtime': 1}

        return pd.read_json(filename, orient="columns", compression=compression_kwargs)

    def save_processed(self, filename, compression=False, compression_kwargs=None):
        """
        Save the processed DataFrame to a json file.
        """
        if not compression:
            compression_kwargs = None
        elif compression and compression_kwargs is None:
            compression_kwargs = {'method': 'gzip', 'compresslevel': 1, 'mtime': 1}

        self.results_df.to_json(filename, orient="columns", indent=1, compression=compression_kwargs)

    def train_test_split(self, train_size=0.8, rng=np.random.RandomState(42), discard_na_rows=False):
        """

        :param train_size: Fraction of the data to use for training
        :param rng: RandomState object to use for reproducibility
        :param discard_na_rows: If True, discard all rows where there is at least one nan in the columns
        starting with "Success"
        :return:
        Train and test dataframes
        """
        results_df = self.discard_if_one_na_in_success_per_row() if discard_na_rows else self.results_df
        # split the results into train and test
        train_df = results_df.sample(frac=train_size, random_state=rng)
        test_df = results_df.drop(train_df.index)

        return train_df, test_df

    def train_val_test_split(self, train_size=0.8, val_size=0.1, rng=np.random.RandomState(42), discard_na_rows=False):
        """

        :param train_size: Fraction of the data to use for training
        :param val_size: Fraction of the data to use for validation
        :param rng: RandomState object to use for reproducibility
        :param discard_na_rows: If True, discard all rows where there is at least one nan in the columns
        :return:
        Train, validation and test dataframes
        """
        results_df = self.discard_if_one_na_in_success_per_row() if discard_na_rows else self.results_df
        # split the results into train, validation and test
        train_df = results_df.sample(frac=train_size, random_state=rng)
        val_test_df = results_df.drop(train_df.index)
        val_df = val_test_df.sample(frac=val_size / (1 - train_size), random_state=rng)
        test_df = val_test_df.drop(val_df.index)

        return train_df, val_df, test_df

    def multiple_way_split(self, sizes=[0.4, 0.1, 0.4, 0.1], rng=np.random.RandomState(42), discard_na_rows=False):
        """
        :param sizes: List of fractions of the data to use for the 4 splits
        :param rng: RandomState object to use for reproducibility
        :param discard_na_rows: If True, discard all rows where there is at least one nan in the columns
        :return:
        Four dataframes with sizes corresponding to those in sizes
        """
        assert sum(sizes) == 1, "The sum of the sizes must be 1"
        results_df = self.discard_if_one_na_in_success_per_row() if discard_na_rows else self.results_df
        n_samples = len(results_df)
        # create size of splits
        splits = [int(size * n_samples) for size in sizes]
        if sum(splits) != n_samples:
            splits[-1] += n_samples - sum(splits)
        # create an array of 0,1,2,3 with sizes corresponding to the splits
        split_array = np.concatenate([np.full(size, i) for i, size in enumerate(splits)])
        # shuffle the array
        rng.shuffle(split_array)
        split_dataframes = []
        for i in range(4):
            split_dataframes.append(results_df.iloc[split_array == i])
        return split_dataframes

    def discard_if_one_na_in_success_per_row(self, inplace=False, print_n_discarded=False):
        # discard all rows where there is at least one nan in the "Success" columns for the considered llms
        length_before = len(self.results_df)
        res_df_after_discard = self.results_df[
            self.results_df[[f"Success_{llm}" for llm in self.llms]].notna().all(axis=1)]
        if print_n_discarded:
            print(f"Discarded {length_before - len(res_df_after_discard)} rows")
        if inplace:
            self.results_df = res_df_after_discard
            return self
        else:
            return res_df_after_discard

    def shuffle_data(self, inplace=True, rng=np.random.RandomState(42)):
        if inplace:
            self.results_df = self.results_df.sample(frac=1, random_state=rng)
            return self
        else:
            return self.results_df.sample(frac=1, random_state=rng)

    def get_average_success(self, llm):
        # check that column exists in the first place
        if f"Success_{llm}" not in self.results_df.columns:
            raise ValueError(f"Column Success_{llm} does not exist in the dataframe")

        # discard rows where that llm has not been tested when computing the mean
        return self.results_df[self.results_df[f"Success_{llm}"].notna()][f"Success_{llm}"].mean()

    @staticmethod
    def _setup_openai(env_file="../.env"):
        from dotenv import load_dotenv
        load_dotenv(env_file)
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def _extract_prompts_df(self, features_to_add=None, max_n_samples=None, skip_na_rows=False,
                            add_system_prompt=False):
        if features_to_add is None:
            features_to_add = []
        if max_n_samples is None:
            max_n_samples = len(self.results_df)

        if skip_na_rows:
            prompts_df = self.discard_if_one_na_in_success_per_row()[["prompt"] + features_to_add]
        else:
            prompts_df = self.results_df[["prompt"] + features_to_add]

        if len(prompts_df) == 0:
            raise ValueError("No rows to annotate as there are no prompts in the dataframe shared for all llms.")

        prompts_df = prompts_df.iloc[:max_n_samples]

        # if add_system_prompt, add the system prompt to the dataframe
        if add_system_prompt:
            prompts_df["prompt_sys_prompt"] = prompts_df["prompt"].apply(lambda x: f"{self.system_prompt} {x}")

        return prompts_df

    def _check_system_prompt_exists(self, add_system_prompt):
        if add_system_prompt:
            if not hasattr(self, "system_prompt"):
                raise ValueError("System prompt has not been defined. Please define it before running this method.")

    def extract_openai_embeddings(self, max_n_samples=None, skip_na_rows=False, inplace=True,
                                  add_system_prompt=False,
                                  env_file="../.env", embedding_model="text-embedding-3-large", batch_size=2048,
                                  sleep_time_if_fail=0.1, ):
        """
        This function extracts the embeddings using the OpenAI embeddings API and stores them in a new column in the
        dataframe. Notice that the embedding computation via API has some intrinsic randomness.
        :param max_n_samples: The maximum number of samples to extract embeddings for. If None, extract embeddings for
        all samples
        :param skip_na_rows: If True, discard all rows where there is at least one nan in the columns starting with
        "Success"
        :param inplace: If True, replace the original dataframe with the one with the embeddings. If False, return the
        new dataframe
        :param env_file: Path to the .env file containing the OpenAI API key
        :param embedding_model: The embedding model to use. See https://beta.openai.com/docs/api-reference/embeddings
        :param batch_size: The size of the batch to use for the API calls. It can be at max 2048
        :param sleep_time_if_fail: Time to sleep if the API call fails

        """
        if batch_size > 2048:
            raise ValueError("The batch size cannot be larger than 2048")

        self._check_system_prompt_exists(add_system_prompt)

        self._setup_openai(env_file)

        # now I need to embedding for each prompt; if skip_na_rows, I will skip the rows where there is at least one
        # nan in the columns starting with "Success"
        # Moreover, if max_n_samples is not None, I will only extract the embeddings for the first max_n_samples
        # todo implement check and add embeddings for that row only if they do not yet exist

        prompts_df = self._extract_prompts_df(max_n_samples=max_n_samples, skip_na_rows=skip_na_rows,
                                              add_system_prompt=add_system_prompt)

        def _get_openai_embedding(texts, model, sleep_time):
            # texts here is a list
            texts = [text.replace("\n", " ") for text in texts]
            while True:
                # try:
                outs = openai.Embedding.create(input=texts, model=model)
                return [np.array(out['embedding']) for out in outs['data']]
            # except:
            #     sleep(sleep_time)

        # prompts_df["openai_embeddings_large" + ("_sys_prompt" if add_system_prompt else "")] = np.nan
        embeddings_list = []
        for i in tqdm(range(0, len(prompts_df), batch_size)):
            # extract the prompts in the batch
            batch_prompts = prompts_df["prompt" + ("_sys_prompt" if add_system_prompt else "")].iloc[
                            i:i + batch_size].tolist()
            # extract the embeddings for the batch
            embeddings = _get_openai_embedding(batch_prompts, embedding_model, sleep_time_if_fail)
            embeddings_list.extend(embeddings)

        prompts_df["openai_embeddings_large" + ("_sys_prompt" if add_system_prompt else "")] = embeddings_list

        # now merge with the original df; if inplace, I will replace the original df
        if inplace:
            self.results_df = self.results_df.merge(prompts_df, on="prompt", how="left")
            return self
        else:
            return self.results_df.merge(prompts_df, on="prompt", how="left")

    def extract_simple_embeddings(self, max_n_samples=None, skip_na_rows=False, inplace=True, add_system_prompt=False,
                                  embedding_model="word2vec", loaded_embedder=None,
                                  return_mean=True, return_std=False, normalise=False,
                                  filter_stopwords=False, overwrite_existing=False):
        """
        Which embedding model should I use as default? fasttext or word2vec?
        word2vec is faster but cannot handle words that it has not seen during training, while fasttext can as it also
        considers sub-word information. However, fasttext is slower.

        :param max_n_samples: The maximum number of samples to extract embeddings for. If None, extract embeddings for
        all samples
        :param skip_na_rows: If True, discard all rows where there is at least one nan in the columns starting with
        "Success" for the considered llms
        :param inplace: If True, replace the original dataframe with the one with the embeddings. If False, return the
        new dataframe
        :param embedding_model: The embedding model to use, either "fasttext" or "word2vec". Default is "word2vec".
        :param loaded_embedder: If not None, the Embedder object to use for the embeddings. If None, a new Embedder
        object will be created.
        :param return_mean : bool, default=True
            Whether to return the mean of the embeddings of the words in the transcript or all embeddings.
        :param return_std : bool, default=True
            Whether to return the standard deviation of the embeddings of the words in the transcript, besides the mean
            or full embeddings.
        :param normalise : bool, default=False
            Whether to normalise the embeddings. If return_mean is True, this will normalise the mean vector. If
            return_mean is False, this will normalise each embedding.
        :param filter_stopwords : bool, default=False
            Whether to filter out stopwords.
        :param overwrite_existing : bool, default=False
            Whether to overwrite the existing embeddings if they exist.

        :return:
        """

        self._check_system_prompt_exists(add_system_prompt)

        prompts_df = self._extract_prompts_df(max_n_samples=max_n_samples, skip_na_rows=skip_na_rows,
                                              add_system_prompt=add_system_prompt)

        # check if the embeddings have already been computed; if not overwrite_existing, discard the rows where the
        # embeddings are already present
        if not overwrite_existing:
            if f"{embedding_model}_embeddings" in self.results_df.columns:
                previous_length = len(prompts_df)
                prompts_df = prompts_df[~prompts_df["prompt"].isin(
                    self.results_df[self.results_df[f"{embedding_model}_embeddings"].notna()]["prompt"])]
                self._print_if_verbose(
                    f"Discarding {previous_length - len(prompts_df)} rows as the embeddings have already been computed")

        if len(prompts_df) == 0:
            self._print_if_verbose("All embeddings have already been computed")
            if inplace:
                return self
            else:
                return self.results_df

        if loaded_embedder is None:
            self._print_if_verbose(f"Instantiating the Embedder with {embedding_model} model...")
            embedder = Embedder(embedding_model, return_mean=return_mean, return_std=return_std, normalise=normalise,
                                verbose=self.verbose, filter_stopwords=filter_stopwords)
            self._print_if_verbose(f"Done")
        else:
            embedder = loaded_embedder
            embedder.update(return_mean=return_mean, return_std=return_std, normalise=normalise,
                            verbose=self.verbose, filter_stopwords=filter_stopwords)

        self._print_if_verbose(f"Extracting embeddings for {max_n_samples} samples...")
        prompts_df[f"{embedding_model}_embeddings" + ("_sys_prompt" if add_system_prompt else "")] = prompts_df[
            "prompt" + ("_sys_prompt" if add_system_prompt else "")].apply(embedder)
        self._print_if_verbose(f"Done")

        # now merge with the original df; if inplace, I will replace the original df
        self._print_if_verbose("Merging with the original dataframe and return.")
        if inplace:
            self.results_df = self.results_df.merge(prompts_df, on="prompt", how="left")
            return self
        else:
            return self.results_df.merge(prompts_df, on="prompt", how="left")

    def _print_if_verbose(self, msg):
        if self.verbose:
            print(msg)

    def _print_df_size(self, df, llm, type: str = "raw"):
        self._print_if_verbose(
            f"{'Loaded raw' if type == 'raw' else 'Created merged'} data for {llm}; shape {df.shape}")


class EvalsResultsLoader(ResultsLoader):
    """Generic class that loads the results of a dataset evaluated with evals. Notice that some of the datasets loaded
    by the subclasses below (Arithmetic, NeuBAROCO, HotpotQA) were also evaluated with evals;
    the classes above differ from this one as they merge the
    evaluation results with the original data and with the default/native features contained in the prompt. This class
    is instead more generic.
    Notice, that, in contrast to the DatasetPreparer classes, here the task has to be the full {task}_{subtask} string,
    if there is a subtask"""

    # some may not have been tested with the base models

    base_path_raw_default = "../results/"

    def __init__(self, task, llms=None, base_path_raw=None, processed_filename=None, verbose=False,
                 load_results_kwargs={}, load_processed_kwargs={}):
        self.task = task
        if base_path_raw is None:
            base_path_raw = self.base_path_raw_default
        # determine self.default_llms from the files in the f"{base_path}/{self.task}" directory
        if not hasattr(self, "default_llms") and llms is None:
            self.default_llms = self._get_default_llms(base_path_raw)
            # print(f"Default llms for {self.task}: {self.default_llms}")

        super().__init__(llms=llms, base_path_raw=base_path_raw, processed_filename=processed_filename, verbose=verbose,
                         load_results_kwargs=load_results_kwargs, load_processed_kwargs=load_processed_kwargs)

        # extract the system prompt by finding the first element where "sampling" is not None
        first_sampling_not_none = self.results_df["sampling"].notna().idxmax()
        sampling = self.results_df["sampling"].loc[first_sampling_not_none]

        # NOTICE the following thing works because all of the "evals" datasets contain a system prompt, otherwise it
        # may give errors.
        if isinstance(sampling["prompt"], list):
            self.system_prompt = sampling["prompt"][0]["content"]
            # print("prompt is a list")
        else:
            self.system_prompt = sampling["prompt"].split("\nUser: ")[
                0]

        # now can drop the "match" columns
        if "match" in self.results_df.columns:
            self.results_df = self.results_df.drop(columns=["match"])
        # do not drop sampling as this may be needed later on (for instance if you save the df and reload)
        # if "sampling" in self.results_df.columns:
        #     self.results_df = self.results_df.drop(columns=["sampling"])

    def load_results(self, base_path="../results/"):
        llms = self.llms

        results_dict = {}
        for llm in llms:
            results_dict[llm] = {}
            raw_data = pd.read_json(f"{base_path}/{self.task}/{llm}.jsonl", lines=True)

            results_df = self.reformat_results_df(raw_data)
            # check if there are duplicated entries:
            duplicated = results_df.duplicated(subset=["prompt"])
            if duplicated.any():
                self._print_if_verbose(f"Warning: {duplicated.sum()} duplicated entries for {llm}")
                # drop the duplicates?
                results_df = results_df.drop_duplicates(subset=["prompt"])

            results_dict[llm] = results_df
            # print("llm", llm)
            # print("accuracy rate ", results_df["Success"].sum() / len(results_df), len(results_df))
            # print("conform answers ", results_df["conform_answer"].sum() / len(results_df))
            self._print_df_size(results_dict[llm], llm, type="raw")
            # rename the "Success" column to "Success_{llm}"
            results_dict[llm] = results_dict[llm].rename(columns={"Success": f"Success_{llm}"})

        # now I want to create a single dataframe with all the results. In practice, start from the first and append
        # the "success" columns of the other llms, making sure that they go with the same prompt
        results_df = results_dict[llms[0]]
        self._print_df_size(results_df, llms[0], type="merged")
        for llm in llms[1:]:
            results_df = results_df.merge(results_dict[llm][["prompt", f"Success_{llm}"]],
                                          on="prompt", how="outer")
            self._print_df_size(results_df, llm, type="merged")

        return results_df

    def reformat_results_df(self, results_df):
        # discard all things where run_id is nan
        results_df = results_df[~results_df["run_id"].isna()]
        # drop the "spec", "final_report" and "created_by" columns
        results_df = results_df.drop(columns=["spec", "final_report", "created_by"], errors="ignore")

        # There are two events for each "sample_id" (one for the sampling and one for the evaluation, called "match").
        # Use pivot to put them in a proper format
        # results dataframe in a different format: one row per sample_id, where I will store the "data" entry for the "sampling" and "metrics" events
        results_df = results_df.pivot(index="sample_id", columns="type", values="data")
        # define "Success"
        # results_df["Success"] = results_df.apply(lambda x: 1 if x["match"]["correct"] else 0, axis=1)
        # IMPORTANT I redefine here success as the naive match does not strip nor lower the strings
        results_df["Success"] = results_df.apply(self.check_answer_correct, axis=1)
        # now extract the original prompt; this depends on whether the prompt is a list or not
        # the way the prompt is stored is different for chat and completion models
        if isinstance(results_df.iloc[0]["sampling"]["prompt"], list):
            results_df["prompt"] = results_df.apply(lambda x: x["sampling"]["prompt"][1]["content"], axis=1)
            # print("prompt is a list")
        else:
            results_df["prompt"] = results_df.apply(lambda x: x["sampling"]["prompt"], axis=1)
            # notice this also discards the few-shot prompt
            results_df["prompt"] = results_df["prompt"].apply(
                lambda x: x.split("\nUser: ")[-1].split("\nAssistant:")[0])
            # print("prompt is a string")
        return results_df

    @staticmethod
    def check_answer_correct(row):
        # strip spaces, new lines and "."; also remove non-breaking spaces
        if isinstance(row["match"]["expected"], str):
            expected = row["match"]["expected"].strip().lower().replace("\u00A0", " ")
        elif isinstance(row["match"]["expected"], int):
            expected = str(row["match"]["expected"])
        else:
            raise ValueError(f"Expected is neither a string nor an integer: {row['match']['expected']}")
        sampled = row["match"]["sampled"].strip().lower().replace("\u00A0", " ")

        # The overall regular expression is constructed to find the expected string in a case-insensitive manner
        # within the sampled string, considering word boundaries and whitespace. The re.search function returns a
        # match object if a match is found, and None otherwise.

        match = re.search(
            r"(^|\b)" + re.escape(expected) + r"(\b|$|\s)",
            sampled,
        )

        correctness_regex = 1 if match else 0

        # correctness_trivial = 1 if expected in sampled else 0
        #
        # # print the two if they are different
        # if correctness_regex != correctness_trivial:
        #     print("==== Correctness: regex=", correctness_regex, "trivial=", correctness_trivial)
        #     print(f'expected>{row["match"]["expected"]}---')
        #     print(f'sampled>{row["match"]["sampled"]}---')

        return correctness_regex

        # if expected in sampled:
        #     return 1
        # else:
        #     return 0

    def _get_default_llms(self, base_path_raw):
        # check in the directory of the task for all *jsonl files and extract the llms from the filenames
        # list of all files
        all_files = os.listdir(f"{base_path_raw}/{self.task}")
        # list of all jsonl files
        jsonl_files = [f for f in all_files if f.endswith(".jsonl")]
        # list of all llms
        llms = [f.replace(".jsonl", "") for f in jsonl_files]
        # now sort the list of llms using sort_models_order
        llms = [llm for llm in sort_models_order if llm in llms]
        return llms


class HelmResultsLoader(ResultsLoader):
    base_path_raw_default = "../results_helm/helm_lite_v1.0.0/"

    # the (sub)-scenarios that are commented out do not have identical prompts across LLMs, most likely due to
    # incoherent number of few-shot prompts.
    possible_scenarios = {'commonsense': ['commonsense'],
                          'gsm': ['gsm'],
                          'med_qa': ['med_qa'],
                          'legalbench': ['abercrombie,',
                                         # 'corporate_lobbying,', # incoherent number of few-shots across llms
                                         'function_of_decision_section,',
                                         'proa,',
                                         'international_citizenship_questions,'],
                          'math': [  # 'algebra',  # incoherent number of few-shots across llms
                              'counting_and_probability',
                              # 'geometry',  # incoherent number of few-shots across llms
                              # 'intermediate_algebra', # incoherent number of few-shots across llms
                              'number_theory',
                              'prealgebra',
                              'precalculus'],
                          'mmlu': ['abstract_algebra',
                                   'college_chemistry',
                                   'computer_security',
                                   'econometrics',
                                   'us_foreign_policy'],
                          # 'narrative_qa': ['narrative_qa'], # discarded as non-binary metric (f1_score)
                          # 'natural_qa': ['closedbook',  # discarded as non-binary metric (f1_score)
                          # 'openbook_longans'],
                          # 'wmt_14': ['cs-en',  # discarded as non-binary metric (bleu)
                          #            'de-en',
                          #            'fr-en',
                          #            'hi-en',
                          #            'ru-en']
                          }

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
    default_llms = list(llms_dict.keys())
    # reverse the dictionary
    llms_dict_reversed = {v: k for k, v in llms_dict.items()}

    def __init__(self, scenario, subscenario=None, llms=None, base_path_raw=None, processed_filename=None,
                 verbose=False, load_results_kwargs={}, load_processed_kwargs={}):

        if scenario not in self.possible_scenarios:
            raise ValueError(f"Scenario {scenario} not in the possible scenarios: {self.possible_scenarios.keys()}")
        if len(self.possible_scenarios[scenario]) > 1 and subscenario is None:
            raise ValueError(
                f"Subscenario must be provided for scenario {scenario}; choose one of {self.possible_scenarios[scenario]}")

        self.scenario = scenario
        self.subscenario = subscenario

        if base_path_raw is None:
            base_path_raw = self.base_path_raw_default

        super().__init__(llms=llms, base_path_raw=base_path_raw, processed_filename=processed_filename, verbose=verbose,
                         load_results_kwargs=load_results_kwargs, load_processed_kwargs=load_processed_kwargs)

    def load_results(self, base_path="../results_helm/helm_lite_v1.0.0/"):
        llms = self.llms

        directory = Path(base_path)
        runs = [item.name for item in directory.iterdir() if item.is_dir()]

        # filter runs to exclude the ones that are not relevant for the selected scenario:
        runs = [run for run in runs if self.scenario in run]
        # filter using sub-scenario if provided
        if self.subscenario is not None:
            runs = [run for run in runs if self.subscenario in run]
        # filter runs to exclude the ones that are not relevant for the selected llms:
        runs = [[run for run in runs if llm.replace("/", "_") in run] for llm in self.llms]
        runs = [item for sublist in runs for item in sublist]

        results_dict = {}
        for run in runs:
            with open(base_path + f'/{run}/scenario_state.json') as f:
                data = json.load(f)
            with open(base_path + f'/{run}/display_predictions.json') as f:
                data2 = json.load(f)
                # this is needed as it contains "the success metric"

            # looks like the useful metric is always the last one
            metric = list(data2[0]['stats'].keys())[-1]
            # print("Metric used:", metric)

            df = pd.DataFrame(data["request_states"])
            df = self._extract_info(df)

            df2 = pd.DataFrame(data2)
            df2[metric] = df2["stats"].apply(lambda x: x[metric])
            # print(df2[metric].value_counts())
            # notice df_2 also contains other info that could be useful (eg the inference time, the number of tokens in prompt and response)

            # set the index to be the id
            df = df.set_index("id")
            df2 = df2.set_index("instance_id")

            len_before = len(df)

            # join
            df = df.join(df2[[metric]], how="inner")
            # check if the length is the same
            if len(df) != len_before:
                raise ValueError("Length of the two dataframes is different after joining")

            df["prompt"] = df["prompt"].apply(self._cleanup_prompt)

            llm = self.llms_dict_reversed[df["model"].iloc[0]]

            # discard all other columns
            df = df[["split", "question", "prompt", "temperature", metric, "completion"]]

            # check if there are duplicated entries:
            duplicated = df.duplicated(subset=["prompt"])
            if duplicated.any():
                self._print_if_verbose(f"Warning: {duplicated.sum()} duplicated entries for {llm}")
                # drop the duplicates?
                df = df.drop_duplicates(subset=["prompt"])

            # rename "success" column to "Success_llm"
            df = df.rename(columns={metric: f"Success_{llm}"})
            # same for temperature
            df = df.rename(columns={"temperature": f"Temperature_{llm}"})
            # same for completion
            df = df.rename(columns={"completion": f"Completion_{llm}"})

            results_dict[llm] = df

            # self._print_df_size(df, llm, type="raw")

        # now I want to create a single dataframe with all the results. In practice, start from the first and append
        # the "success", "temperature" and "completion" columns of the other llms, making sure that they go with
        # the same prompt
        results_df = results_dict[llms[0]]
        self._print_df_size(results_df, llms[0], type="merged")
        for llm in llms[1:]:
            results_df = results_df.merge(results_dict[llm], on=["prompt", "split", "question"], how="outer")
            self._print_df_size(results_df, llm, type="merged")

        # keep only the rows where "split" == "test"
        results_df = results_df[results_df["split"] == "test"]

        return results_df

    @staticmethod
    def _extract_info(df):
        df["split"] = df["instance"].apply(lambda x: x["split"])
        df["question"] = df["instance"].apply(lambda x: x["input"]["text"])
        df["id"] = df["instance"].apply(lambda x: x["id"])
        df["prompt"] = df["request"].apply(lambda x: x["prompt"])
        df["model"] = df["request"].apply(lambda x: x["model"])
        df["temperature"] = df["request"].apply(lambda x: x["temperature"])
        df["completion"] = df["result"].apply(lambda x: x["completions"][0]["text"])

        return df

    @staticmethod
    def _cleanup_prompt(prompt):
        """This is to remove the wrapping that was applied to the assistant models (claude, GPT4), in order to make all
        of the prompts identical with all models. The wrapping is as follows:

        Here are some input-output examples. Read the examples carefully to figure out the mapping. The output of the
        last example is not given, and your job is to figure out what it is.
        [prompt with in-context examples]
        Please provide the output to this last example. It is critical to follow the format of the preceding outputs!
        this removes the two parts of text (before and after the actual prompt)

        There are actually some small variations on this, ie in some cases there are additional new lines at the start,
        while in some other cases the postfix is followed by either "Assistant:" or "A:"
        """
        # prompt = prompt.replace(
        #     "\n\nHuman: Here are some input-output examples. Read the examples carefully to figure out the mapping. The output of the last example is not given, and your job is to figure out what it is.\n\n",
        #     "")
        # prompt = prompt.replace(
        #     "\n\nPlease provide the output to this last example. It is critical to follow the format of the preceding outputs!\n\nAssistant: Answer:",
        #     "")
        # # notice this is slightly different from the one above! Quite messy!
        # prompt = prompt.replace(
        #     "\n\nPlease provide the output to this last example. It is critical to follow the format of the preceding outputs!\n\nAssistant: A:",
        #     "")

        # use split instead, as the one above does not work well
        prompt = prompt.split(
            "Here are some input-output examples. Read the examples carefully to figure out the mapping. The output of the last example is not given, and your job is to figure out what it is.\n\n")[
            -1]
        prompt = prompt.split(
            "\n\nPlease provide the output to this last example. It is critical to follow the format of the preceding outputs!")[
            0]
        return prompt


def _finalize_train_validation_test_dfs(train_df, test_df, validation_size, subsampled_n_train,
                                        subsampled_n_test, random_state):
    # subsample
    if subsampled_n_train is not None:
        train_df = train_df.sample(n=subsampled_n_train, random_state=random_state)
    if subsampled_n_test is not None:
        test_df = test_df.sample(n=subsampled_n_test, random_state=random_state)

    # split the training set into train and validation
    validation_df = train_df.sample(frac=validation_size, random_state=random_state)
    train_df = train_df.drop(validation_df.index)

    # reset all indices
    train_df.reset_index(drop=True, inplace=True)
    validation_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return train_df, validation_df, test_df


def load_reasoning(llms, features, base_path="../results/kindsofreasoning/", ood_split=False, validation_size=0.2,
                   subsampled_n_train=None, subsampled_n_test=None, random_state=42):
    """
    This loads the reasoning evals run with the instruct-only OpenAI LLMs.

    The n-gram computation needs to be done externally from this function.

    Test Splits:
    Random. Random shuffle split (in-distribution)
    OOD_1. Arithmetic left out
    OOD_2. Causal reasoning left out
    OOD_3. all the other kinds of reasoning left out
    OOD_4. world knowledge and common sense left out

    Notice that splits 1, 2 and 3 are a covering of the full dataset; the final split is instead complementary.

    :param llms: list of llms to consider
    :param features: list of features to consider (among "openai_embeddings", "word2vec", "fasttext")
    :param base_path:
    :param ood_split: If False, the test split will be drawn randomly from all the results; otherwise, "OOD_{split}" for
    split in {1, 2, 3, 4} can be used.
    :param validation_size: dedicate that fraction of the train split to validation. This is done after subsampling
    the training set (see below)
    :param subsampled_n_train: If not None, the train split will be subsampled to this number of rows
    :param subsampled_n_test: If not None, the test split will be subsampled to this number of rows
    :param random_state: random state for the subsampling

    :return:
    List of three dataframes: train_df, validation_df, test_df
    """

    openai_embeddings = "openai_embeddings" in features

    evals_dict = {
        'formal_fallacies_syllogisms_negation': ["logical_reasoning"],
        'logical_args': ["logical_reasoning", "common_sense"],
        'babi_task_16': ["inductive_reasoning"],
        'logiqav2-logical-reasoning-plus': ["deductive_reasoning"],
        'wanli': ["deductive_reasoning"],
        'alpha_nli': ["abductive_reasoning"],
        'reclor-logical-reasoning-plus': ["abductive_reasoning", "deductive_reasoning", "inductive_reasoning"],
        'crass_ai': ["counterfactual_reasoning"],
        'cause_and_effect_one_sentence': ["causal_reasoning"],
        'cause_and_effect_two_sentences': ["causal_reasoning"],
        'fantasy_reasoning': ["causal_reasoning"],
        'goal_step_wikihow_goal_inference': ["causal_reasoning"],
        'goal_step_wikihow_step_inference': ["causal_reasoning"],
        'goal_step_wikihow_step_ordering': ["causal_reasoning"],
        "copa": ["causal_reasoning", "world_knowledge"],
        "cosmos_qa": ["causal_reasoning", "world_knowledge"],
        "ropes": ["causal_reasoning", "world_knowledge"],
        "anli": ["causal_reasoning", "world_knowledge"],
        "emoji_movie": ["analogical_reasoning", "world_knowledge"],
        'abstract_narrative_understanding_9_distractors': ["analogical_reasoning"],
        'abstract_narrative_understanding_99_distractors': ["analogical_reasoning"],
        'odd_one_out': ["analogical_reasoning"],
        'metaphor_boolean': ["analogical_reasoning"],
        'geometric_shapes': ["spatial_reasoning"],
        'space_nli': ["spatial_reasoning"],
    }

    # add the arithmetic results:
    for n_digits in range(1, 6):
        for operation in ["addition", "subtraction", "multiplication", "division"]:
            eval_name = f"arithmetic_{n_digits}_digit_{operation}"
            evals_dict[eval_name] = ["arithmetic"]

    splits_dict = {
        "OOD_1": ["arithmetic"],
        "OOD_2": ["causal_reasoning"],
        "OOD_3": ["logical_reasoning", "deductive_reasoning", "inductive_reasoning", "spatial_reasoning",
                  "abductive_reasoning", "counterfactual_reasoning", "analogical_reasoning"],
        "OOD_4": ["world_knowledge", "common_sense"],
    }

    if ood_split and ood_split not in splits_dict.keys():
        raise ValueError(f"ood_split must be one of {list(splits_dict.keys())}")

    if ood_split:
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
    else:
        pooled_results_df = pd.DataFrame()

    loaded_embedders_dict = {}
    for embedding_type in ["word2vec", "fasttext"]:
        if embedding_type in features:
            loaded_embedders_dict[embedding_type] = Embedder(embedding_type)

    for i, eval_name in tqdm(enumerate(evals_dict.keys())):
        # load with the embeddings:
        if openai_embeddings:
            instance = EvalsResultsLoader(task=eval_name, llms=llms,
                                          processed_filename=f"{base_path}/{eval_name}_with_embeddings.gz",
                                          load_processed_kwargs={"compression": True})
        else:
            instance = EvalsResultsLoader(task=eval_name, llms=llms, base_path_raw=base_path)
        instance = instance.discard_if_one_na_in_success_per_row(
            inplace=True)  # this discards all rows where the considered llm has not been tested

        has_system_prompt = hasattr(instance, "system_prompt")

        if openai_embeddings and has_system_prompt:
            instance.results_df["openai_embeddings_large"] = instance.results_df["openai_embeddings_large_sys_prompt"]

        for embedding_type in ["word2vec", "fasttext"]:
            if embedding_type in features:
                # compute the word2vec and fasttext embeddings
                instance.extract_simple_embeddings(skip_na_rows=True, embedding_model=embedding_type,
                                                   loaded_embedder=loaded_embedders_dict[embedding_type],
                                                   filter_stopwords=True,
                                                   add_system_prompt=has_system_prompt)

                if has_system_prompt:
                    instance.results_df[f"{embedding_type}_embeddings"] = instance.results_df[
                        f"{embedding_type}_embeddings_sys_prompt"]

        if ood_split:
            if any([kind in splits_dict[ood_split] for kind in evals_dict[eval_name]]):
                test_df = pd.concat([test_df, instance.results_df], ignore_index=True)
            else:
                train_df = pd.concat([train_df, instance.results_df], ignore_index=True)
        else:
            # add this to the pooled results
            pooled_results_df = pd.concat([pooled_results_df, instance.results_df], ignore_index=True)

    if not ood_split:
        # now do a train-test split
        train_df = pooled_results_df.sample(frac=0.7, random_state=random_state)
        test_df = pooled_results_df.drop(train_df.index)

    train_df, validation_df, test_df = _finalize_train_validation_test_dfs(
        train_df, test_df, validation_size, subsampled_n_train, subsampled_n_test, random_state)

    return train_df, validation_df, test_df


def load_helm_lite(llms, features, base_path="../results/helm_lite_v1.0.0", ood_split=False, validation_size=0.2,
                   subsampled_n_train=None, subsampled_n_test=None, random_state=42):
    """

    :param llms: list of llms to consider
    :param features: list of features to consider (among "openai_embeddings", "word2vec", "fasttext")
    :param base_path:
    :param ood_split: If False, the train-test split will be done randomly;
    if OOD_1, the test split will be math, gsm, and mmlu abstract algebra;
    if OOD_2, the test split will be legalbench;
    if OOD_3, the test split will be commonsense, med_qa, and mmlu (except abstract algebra)
    :param validation_size: dedicate that fraction of the train split to validation. This is done after subsampling
    the training set (see below)
    :param subsampled_n_train: If not None, the train split will be subsampled to this number of rows
    :param subsampled_n_test: If not None, the test split will be subsampled to this number of rows
    :param random_state: random state for the subsampling

    :return:
    List of three dataframes: train_df, validation_df, test_df
    """
    openai_embeddings = "openai_embeddings" in features

    possible_scenarios = HelmResultsLoader.possible_scenarios

    if ood_split:
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
    else:
        pooled_results_df = pd.DataFrame()

    for scenario in tqdm(possible_scenarios.keys()):
        for subscenario in possible_scenarios[scenario]:
            eval = f"{scenario}_{subscenario}"
            # load with the embeddings:
            if openai_embeddings:
                print(f"{base_path}/{eval}'_with_embeddings.gz")
                instance = HelmResultsLoader(scenario=scenario, subscenario=subscenario, llms=llms,
                                             processed_filename=f"{base_path}/{eval}_with_embeddings.gz",
                                             load_processed_kwargs={"compression": True} if openai_embeddings else {})
            else:
                instance = HelmResultsLoader(scenario=scenario, subscenario=subscenario, llms=llms,
                                             base_path_raw=base_path)
            instance = instance.discard_if_one_na_in_success_per_row(
                inplace=True)  # this discards all rows where the considered llm has not been tested

            has_system_prompt = hasattr(instance, "system_prompt")

            if openai_embeddings and has_system_prompt:
                instance.results_df["openai_embeddings_large"] = instance.results_df[
                    "openai_embeddings_large_sys_prompt"]

            for embedding_type in ["word2vec", "fasttext"]:
                if embedding_type in features:
                    # compute the word2vec and fasttext embeddings
                    instance.extract_simple_embeddings(skip_na_rows=True, embedding_model=embedding_type,
                                                       filter_stopwords=True,
                                                       add_system_prompt=has_system_prompt)

                    if has_system_prompt:
                        instance.results_df[f"{embedding_type}_embeddings"] = instance.results_df[
                            f"{embedding_type}_embeddings_sys_prompt"]

            if ood_split == "OOD_1":
                # add to test split if "math", "gsm", "mmlu_abstract_algebra" evals
                if scenario in ['math', 'gsm'] or (scenario == 'mmlu' and subscenario == 'abstract_algebra'):
                    test_df = pd.concat([test_df, instance.results_df], ignore_index=True)
                else:
                    train_df = pd.concat([train_df, instance.results_df], ignore_index=True)
            elif ood_split == "OOD_2":
                # add to test split if "legalbench" evals
                if scenario == 'legalbench':
                    test_df = pd.concat([test_df, instance.results_df], ignore_index=True)
                else:
                    train_df = pd.concat([train_df, instance.results_df], ignore_index=True)
            elif ood_split == "OOD_3":
                # add to test split if "commonsense", "med_qa", "mmlu" evals (except abstract_algebra)
                if scenario in ['commonsense', 'med_qa'] or (scenario == 'mmlu' and subscenario != 'abstract_algebra'):
                    test_df = pd.concat([test_df, instance.results_df], ignore_index=True)
                else:
                    train_df = pd.concat([train_df, instance.results_df], ignore_index=True)
            else:
                # add this to the pooled results
                pooled_results_df = pd.concat([pooled_results_df, instance.results_df], ignore_index=True)

    if not ood_split:
        # now do a train-test split
        train_df = pooled_results_df.sample(frac=0.7, random_state=random_state)
        test_df = pooled_results_df.drop(train_df.index)

    train_df, validation_df, test_df = _finalize_train_validation_test_dfs(
        train_df, test_df, validation_size, subsampled_n_train, subsampled_n_test, random_state)

    return train_df, validation_df, test_df


if __name__ == "__main__":
    # test loading the kindsofreasoning dataset with the embeddings:
    train_df, validation_df, test_df = load_reasoning(llms=sort_models_order, features=["openai_embeddings"],
                                                      ood_split=False,
                                                      base_path="../results/kindsofreasoning_embeddings/")

    # # test loading the kindsofreasoning dataset:
    train_df, validation_df, test_df = load_reasoning(llms=sort_models_order, features=[""], ood_split=False)

    # test loading the helm_lite dataset with the embeddings:
    train_df, validation_df, test_df = load_helm_lite(
        llms=['tiiuae/falcon-40b', 'gpt-4-0613', 'AlephAlpha/luminous-extended', 'cohere/command-light', ],
        features=["openai_embeddings"], ood_split=False, base_path="../results/helm_lite_v1.0.0_embeddings/")

    # test loading the helm_lite dataset:
    train_df, validation_df, test_df = load_helm_lite(
        llms=['tiiuae/falcon-40b', 'gpt-4-0613', 'AlephAlpha/luminous-extended', 'cohere/command-light', ],
        features=[""], ood_split=False)
