import os

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as pairwise
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from scipy.spatial import distance
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.irt import load_irt_parameters, create_irt_dataset, train_irt_model
from src.classification_utils import brierDecomp
from src.results_loaders import load_evals


class SampleSelector:
    methods = ["random", "random_best_of", "clustering_embeddings", "clustering_LLM_success", "clustering_IRT_values",
               "factor_analysis_embeddings", "factor_analysis_LLM_success", "factor_analysis_IRT_values"]

    def __init__(self, df, feature, train_llms):
        self.df = df
        self.feature = feature
        self.train_llms = train_llms

    def _train_val_split(self, generator, train_size=0.7):
        # split the df into train and val
        train_df = self.df.sample(frac=train_size, random_state=generator)
        val_df = self.df.drop(train_df.index)
        return train_df, val_df

    def select_random(self, n_selected, random_state=42, force_label_diverseness=True, max_tries=100):
        """Simply draw a random set of rows; this does not use the successes of the train LLMs neither the feature"""

        # define a numpy generator
        generator = np.random.default_rng(random_state)

        diverse = False
        trial = 0
        while not diverse and trial < max_tries:
            selected_df = self.df.sample(n=n_selected, random_state=generator)
            if force_label_diverseness:
                skip_this = False
                for llm in self.train_llms:
                    # check that there at least 2 different values in the 'Success' column
                    if len(selected_df[f'Success_{llm}'].unique()) < 2:
                        skip_this = True
                        continue
                if not skip_this:
                    diverse = True
            else:
                diverse = True
            trial += 1

        if not diverse:
            print(f"Could not find a diverse set of {n_selected} samples after {max_tries} tries; returning the last "
                  f"selection anyway, but expect issues.")
            return selected_df
        else:
            return selected_df

    def select_random_best_of(self, n_selected, random_state=42, best_of=20, train_size=0.7,
                              classifier=LogisticRegression, classifier_kwargs={}):
        """Select a random subset from the train split of the df; repeat that `best_of` times and keep the best on
         according to the performance on the validation split. As the classifier needs one different value for each
         label, it re-tries generating the subset until they have a different value per label, for all train_llms.
         As such, this may take a long time (and it may not even be possible with some selections of train_llms)."""

        # define a numpy generator
        generator = np.random.default_rng(random_state)

        train_df, val_df = self._train_val_split(generator, train_size=train_size)

        best_score = 0
        best_subset = None

        # Initialize a classifier
        clf = classifier(**classifier_kwargs)

        trials = 0

        while trials < best_of:
            # Select a random subset from train_df
            subset = train_df.sample(n=n_selected, random_state=generator)

            skip_this = False
            scores = []
            for llm in self.train_llms:
                # check that there at least 2 different values in the 'Success' column
                if len(subset[f'Success_{llm}'].unique()) < 2:
                    skip_this = True
                    continue

                # Train the classifier on the subset and evaluate it on val_df
                clf.fit(np.array(list(subset[self.feature].apply(lambda x: np.array(x)).values)),
                        subset[f'Success_{llm}'])
                score = roc_auc_score(val_df[f'Success_{llm}'], clf.predict(
                    np.array(list(val_df[self.feature].apply(lambda x: np.array(x)).values))))
                scores.append(score)
            if skip_this:
                continue

            # Calculate the average score
            avg_score = sum(scores) / len(scores)

            # If the score is better than the current best score, update the best score and the best subset
            if avg_score > best_score:
                best_score = avg_score
                best_subset = subset

            trials += 1
            # print(f"Trial {trials} done")

        # Return the best subset
        return best_subset

    def select_greedy(self, n_selected, random_state=42, train_size=0.7):
        """Greedy selection method: add elements one by one to the selected set, always selecting the one that maximizes the performance of a given classifier.
        NB: this does not work properly as the classifiers need at least one sample for each of the labels to work, which is not guaranteed here -> you may encounter issues for this reason."""
        # define a numpy generator
        generator = np.random.default_rng(random_state)

        train_df, val_df = self._train_val_split(generator, train_size=train_size)

        # initialize the selected set with 2 random elements from df, on
        selected_df = train_df.sample(n=1, random_state=generator)

        # initialize a classifier
        clf = LogisticRegression()

        while len(selected_df) < n_selected:
            # print(len(selected_df))
            max_score = 0
            best_sample = None

            for _, sample in train_df.iterrows():
                # print(f"\t{_}")
                if sample.name in selected_df.index:
                    continue

                temp_df = pd.concat([selected_df, sample.to_frame().T], ignore_index=True)
                scores = []
                for llm in self.train_llms:
                    clf.fit(np.array(list(temp_df[self.feature].apply(lambda x: np.array(x)).values)),
                            temp_df[f'Success_{llm}'])
                    score = roc_auc_score(val_df[f'Success_{llm}'], clf.predict(val_df[self.feature]))
                    scores.append(score)

                avg_score = sum(scores) / len(scores)

                if avg_score > max_score:
                    max_score = avg_score
                    best_sample = sample

            best_sample_df = pd.DataFrame(best_sample)
            selected_df = pd.concat([selected_df, best_sample_df])

        return selected_df

    def select_clustering_embeddings(self, n_selected, clustering_method=KMeans, method_kwargs={},
                                     random_state=42):
        """Consider self.feature and perform clustering with n_selected clusters, then return the element closest to
        the centroid for each cluster; notice this does not use the successes of the train LLMs"""
        # Perform clustering

        df_feats = self.df[self.feature]

        return self._clustering(df_feats, clustering_method=clustering_method, n_clusters=n_selected,
                                method_kwargs=method_kwargs, random_state=random_state)

    def select_clustering_LLM_success(self, n_selected, clustering_method=KMeans, method_kwargs={},
                                      random_state=42):
        """Consider the successes of the train LLMs and perform clustering with n_selected clusters,
        then return the element closest to the centroid for each cluster"""

        df_feats = self.df[[f'Success_{llm}' for llm in self.train_llms]]

        return self._clustering(df_feats, clustering_method=clustering_method, n_clusters=n_selected,
                                method_kwargs=method_kwargs, random_state=random_state)

    def select_clustering_IRT_values(self, n_selected, clustering_method=KMeans, method_kwargs={},
                                     random_state=42, irt_path='data_irt/irt_model/'):
        """Consider the IRT values and perform clustering with n_selected clusters, then return the element closest to
        the centroid for each cluster"""
        X_irt = self._load_IRT_tiny_benchmarks(path=irt_path)

        # convert that to a pandas series
        df_feats = pd.Series(X_irt.tolist())

        return self._clustering(df_feats, clustering_method=clustering_method, n_clusters=n_selected,
                                method_kwargs=method_kwargs, random_state=random_state)

    def select_factor_analysis_LLM_success_features(self, n_selected=None, n_selected_per_factor=None,
                                                    max_n_factors=30, random_state=42, **kwargs):
        """Perform factor analysis on the successes of the train LLMs (considering the success on each instance as
         features) and select the n_selected_per_factor elements with the largest loadings."""
        # This does not seem to work as soon as there is one instance where all successes are either 0 or 1; that is
        # handled by removing those columns and repeated ones

        X_feats = self.df[[f'Success_{llm}' for llm in self.train_llms]].values

        # transpose the matrix
        X_feats = X_feats.T

        # compute the standard deviation wrt 0th axis
        # std = np.std(X_feats, axis=0)
        # count how many 0s and print
        # print(f"Number of 0s: {np.sum(std == 0)}")

        return self._factor_analysis_select_features(X_feats, n_selected=n_selected,
                                                     n_selected_per_factor=n_selected_per_factor,
                                                     max_n_factors=max_n_factors,
                                                     random_state=random_state, **kwargs)

    def select_factor_analysis_LLM_success_samples(self, n_selected=None, n_selected_per_factor=None, max_n_factors=30,
                                                   random_state=42, **kwargs):
        """Perform factor analysis on the successes of the train LLMs (by considering instances as subjects and the
        success on each LLM as a feature and select the n_selected_per_factor elements with the values in the H matrix
        loadings."""
        # This does not seem to work as soon as there is one LLM which either always failed or succeeded!

        X_feats = self.df[[f'Success_{llm}' for llm in self.train_llms]].values

        # compute the standard deviation wrt 0th axis
        # std = np.std(X_feats, axis=0)
        # count how many 0s and print
        # print(f"Number of 0s: {np.sum(std == 0)}")

        return self._factor_analysis_select_samples(X_feats, n_selected=n_selected,
                                                    n_selected_per_factor=n_selected_per_factor,
                                                    max_n_factors=max_n_factors,
                                                    random_state=random_state, **kwargs)

    def select_factor_analysis_embeddings(self, n_selected=None, n_selected_per_factor=None, max_n_factors=30,
                                          random_state=42, **kwargs):
        """Perform factor analysis on the samples and select the n_selected_per_factor elements with the absolute
        largest values of the H matrix."""
        X_feats = np.array(self.df[self.feature].tolist())

        return self._factor_analysis_select_samples(X_feats, n_selected=n_selected,
                                                    n_selected_per_factor=n_selected_per_factor,
                                                    max_n_factors=max_n_factors,
                                                    random_state=random_state, **kwargs)

    def select_factor_analysis_IRT_values(self, n_selected=None, n_selected_per_factor=None, max_n_factors=30,
                                          random_state=42, irt_path='data_irt/irt_model/', **kwargs):
        """Perform factor analysis on the samples and select the n_selected_per_factor elements with the absolute
        largest values of the H matrix."""
        X = self._load_IRT_tiny_benchmarks(path=irt_path)

        return self._factor_analysis_select_samples(X, n_selected=n_selected,
                                                    n_selected_per_factor=n_selected_per_factor,
                                                    max_n_factors=max_n_factors,
                                                    random_state=random_state, **kwargs)

    def _clustering(self, df_feats, clustering_method=KMeans, n_clusters=None, method_kwargs={},
                    random_state=42):
        # the tiny_benchmarks clustering also includes balance_weights to make sure that the different sub-scenarios are
        # weighted evenly in defining the clusters and computing the performance. It may make sense there as they are
        # interested in estimating the average performance, but I think it is pointless for our setup.

        if isinstance(df_feats.iloc[0], list):
            X_feats = np.array(df_feats.apply(lambda x: np.array(x)).tolist())
        else:
            X_feats = df_feats.values

        if "n_init" not in method_kwargs:
            method_kwargs["n_init"] = "auto"
        clustering = clustering_method(n_clusters=n_clusters, random_state=random_state, **method_kwargs)

        clustering.fit(X_feats)

        # check it has the cluster_centers_ attribute
        if not hasattr(clustering, "cluster_centers_"):
            raise ValueError(
                f"The clustering method {clustering_method} does not have the 'cluster_centers_' attribute")

        # Get cluster assignments for each data point
        cluster_assignments = clustering.labels_

        # For each cluster, find the data point closest to the centroid
        selected_indices = []
        for cluster_id in range(n_clusters):
            # get the indices of the points in the current cluster
            cluster_indices = np.where(cluster_assignments == cluster_id)[0]
            # get the points in the current cluster
            cluster_points = X_feats[cluster_indices]
            centroid = clustering.cluster_centers_[cluster_id]
            # compute the distance of each point to the centroid
            distances = distance.cdist(cluster_points, [centroid], 'euclidean').squeeze()
            # find the index of the closest point within the cluster
            closest_point_index_cluster = distances.argmin()
            # find the index of the closest point within the original dataframe
            closest_point_index_X_feats = cluster_indices[closest_point_index_cluster]
            selected_indices.append(closest_point_index_X_feats)

        return self.df.iloc[sorted(selected_indices)]

    def _factor_analysis(self, X_feats, n_selected=None, n_selected_per_factor=None, max_n_factors=30,
                         random_state=42, remove_duplicate_columns=False, **kwargs):
        np.random.seed(random_state)

        # one and only one of n_selected and n_selected_per_factor should be provided
        if (n_selected is None) == (n_selected_per_factor is None):
            raise ValueError("Exactly one of n_selected and n_selected_per_factor should be provided")

        # remove duplicates in the columns and columns where all elements are identical:
        # this flag as it does not make sense to do that when I am selecting using the samples rather than the features.
        if remove_duplicate_columns:
            X_feats, original_to_new_indices = self._remove_duplicate_columns(X_feats)
        else:
            original_to_new_indices = {i: i for i in range(X_feats.shape[1])}

        # The factor analysis step seems to perform the scaling automatically

        # adequacy tests
        # Bartlett’s Test of Sphericity
        chi_square_value, p_value = calculate_bartlett_sphericity(X_feats)
        if p_value > 0.05 or np.isnan(p_value):
            print("According to Bartlett's test, "
                  "factor analysis is not appropriate; the method will continue nevertheless, but expect "
                  "unreliable results.")
            print(f"Bartlett's test: Chi-square value: {chi_square_value}, p-value: {p_value}")

        # Kaiser-Meyer-Olkin (KMO) Test
        kmo_all, kmo_model = calculate_kmo(X_feats)
        if kmo_model < 0.6 or np.isnan(kmo_model):
            print("According to the KMO test, "
                  "factor analysis is not appropriate; the method will continue nevertheless, but expect "
                  "unreliable results.")
            print(f"KMO: {kmo_model} (should be > 0.6)")

        # Create factor analysis object and perform factor analysis
        fa = FactorAnalyzer(n_factors=max_n_factors, rotation=None, **kwargs)
        fa.fit(X_feats)

        # check the eigenvalues and see how many are larger than 1
        ev, v = fa.get_eigenvalues()
        # print(f"Eigenvalues: {ev}")
        # count how many are larger than 1
        n_factors = sum(ev > 1)

        n_factors = min(n_factors, max_n_factors)

        # repeat that with those number of factors and rotation
        fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax", **kwargs)
        fa.fit(X_feats)

        # list with the number of elements to select per each factor
        if n_selected_per_factor is not None:
            n_selected_per_factor_list = [n_selected_per_factor] * n_factors
        else:
            # use n_selected
            n_selected_per_factor_list = [n_selected // n_factors] * n_factors
            # correct rounding errors
            n_missing = n_selected - sum(n_selected_per_factor_list)
            for i in range(n_missing):
                n_selected_per_factor_list[i] += 1
            assert sum(n_selected_per_factor_list) == n_selected

        return fa, n_selected_per_factor_list, original_to_new_indices

    def _factor_analysis_select_features(self, X_feats, n_selected=None, n_selected_per_factor=None, max_n_factors=30,
                                         random_state=42, **kwargs):
        """This is my original implementation, where I use factor analysis to select the most informative features based
        on the loadings matrix. In particular, I take the features which have the largest loading for each factor."""
        # partially following https://www.datacamp.com/tutorial/introduction-factor-analysis
        # set numpy random seed

        fa, n_selected_per_factor_list, original_to_new_indices = self._factor_analysis(
            X_feats, n_selected=n_selected, n_selected_per_factor=n_selected_per_factor, max_n_factors=max_n_factors,
            random_state=random_state, remove_duplicate_columns=True, **kwargs
        )

        # now need to select the instances that have larger loadings
        loadings = fa.loadings_
        n_factors = loadings.shape[1]

        selected_indices = []
        for factor in range(n_factors):
            factor_loadings = np.abs(loadings[:, factor])
            # remove those that have already been selected
            factor_loadings[selected_indices] = -np.inf
            # select n_selected_per_factor instances with the largest loadings
            selected_indices.extend(factor_loadings.argsort()[-n_selected_per_factor_list[factor]:][::-1])

        corrected_indices = sorted([original_to_new_indices[i] for i in selected_indices])

        return self.df.iloc[corrected_indices]

    def _factor_analysis_select_samples(self, X_feats, n_selected=None, n_selected_per_factor=None, max_n_factors=30,
                                        random_state=42, **kwargs):
        """Here I instead use factor analysis to select the most informative instances based
        on the H matrix (e the matrix of latent factor values for each sample). In particular, I select the instances
        which have the largest values of each factor."""

        fa, n_selected_per_factor_list, original_to_new_indices = self._factor_analysis(
            X_feats, n_selected=n_selected, n_selected_per_factor=n_selected_per_factor, max_n_factors=max_n_factors,
            random_state=random_state, remove_duplicate_columns=False, **kwargs
        )

        H = fa.transform(X_feats)
        n_factors = H.shape[1]

        # now need to select the samples (ie the instances) that have larger values of the H matrix for each factor
        selected_indices = []
        for factor in range(n_factors):
            factor_values = np.abs(H[:, factor])
            # remove those that have already been selected
            factor_values[selected_indices] = -np.inf
            # select n_selected_per_factor instances with the largest loadings
            selected_indices.extend(factor_values.argsort()[-n_selected_per_factor_list[factor]:][::-1])

        # this is not needed now as I am not selecting columns but rather rows
        # corrected_indices = sorted([original_to_new_indices[i] for i in selected_indices])

        return self.df.iloc[selected_indices]

    def _remove_duplicate_columns(self, X_feats):
        # Convert X_feats to a DataFrame for easier duplicate removal
        X_feats_df = pd.DataFrame(X_feats.T)

        # Get a boolean mask for duplicate columns
        mask_duplicates = X_feats_df.duplicated()

        # Get a boolean mask for columns where all elements are identical
        mask_identical = X_feats_df.nunique(axis=1) == 1

        # Combine the two masks
        mask = mask_duplicates | mask_identical

        # Create a mapping from original to new indices
        original_to_new_indices = {i: j for i, j in enumerate(np.where(~mask)[0])}

        # Remove duplicate and identical columns
        X_feats = X_feats_df.loc[~mask].values.T
        return X_feats, original_to_new_indices

    def compute_IRT_tiny_benchmarks(self, dim_irt, random_state=42, device='cpu', epochs=2000, lr=.1,
                                    dataset_name='data_irt/irt_dataset.jsonlines',
                                    model_name='data_irt/irt_model/'):
        """
        This is a separate method that computes the IRT values using the code from tinyBenchmarks. Notice that their
        original code also included a heuristics for selecting the dimension of the IRT model. This stores the results
        in some files that are then called by the _load_IRT_tiny_benchmarks method, which is employed both in the
        factor analysis and the clustering IRT methods.

        :param dim_irt: The number of dimensions for the IRT model.
        :param random_state: The random state for the numpy generator
        :param device: Either 'cuda' or 'cpu'
        :param epochs: Number of epochs for IRT model training (py-irt default is 2000)
        :param lr: Learning rate for IRT model training (py-irt default is .1)
        :param dataset_name: The name of the file where the dataset will be saved.
        :param model_name: The desired name for the output model.

        :return:
        """

        # set the seed as well:
        np.random.seed(random_state)

        Y_bin_train = self.df[[f'Success_{llm}' for llm in self.train_llms]].values.T

        # create needed folder
        os.makedirs(dataset_name.split("/")[0], exist_ok=True)

        # notice tinyBenchmarks also uses a validation heuristic to choose the best dimension of IRT, but I will
        # ignore this for now

        create_irt_dataset(Y_bin_train, dataset_name)

        train_irt_model(dataset_name=dataset_name,
                        model_name=model_name,
                        D=dim_irt, lr=lr, epochs=epochs, device=device)

    def _load_IRT_tiny_benchmarks(self, path='data_irt/irt_model/'):
        # load the results; I could actually do this in the places where I need this, as it is likely to be quite 
        # expensive 
        A, B, _ = load_irt_parameters(path)
        X = np.vstack((A.squeeze(), B.squeeze().reshape((1, -1)))).T

        # X is of shape (n_instances, n_irt_params)
        return X

    def select_with_monotonicity(self, random_state=42):
        """Implement, not sure how!"""
        pass

    # make a single method that calls the other
    def select(self, method, *args, **kwargs):
        selector_methods_dict = {
            "random": self.select_random,
            "random_best_of": self.select_random_best_of,
            "clustering_embeddings": self.select_clustering_embeddings,
            "clustering_LLM_success": self.select_clustering_LLM_success,
            "clustering_IRT_values": self.select_clustering_IRT_values,
            "factor_analysis_embeddings": self.select_factor_analysis_embeddings,
            "factor_analysis_IRT_values": self.select_factor_analysis_IRT_values,
            "factor_analysis_LLM_success_samples": self.select_factor_analysis_LLM_success_samples,
            "factor_analysis_LLM_success_features": self.select_factor_analysis_LLM_success_features,
        }

        selector_methods = list(selector_methods_dict.keys())

        if method not in selector_methods:
            raise ValueError(f"Method {method} not recognized")

        return selector_methods_dict[method](*args, **kwargs)


class AssessorFromReference:
    methods = ["upper_bound_train", "reference_only", "concatenate_ref_success", "calibrate_general_classifier"]

    def __init__(self, reference_df, train_df, validation_df, test_df, feature, train_llms, validation_llms, test_llms):
        """
        Methods to predict the success of the test llms on the test_df using the reference_df and the feature.

        :param reference_df: the selected reference_df, on which we assume we can access the 'Success' columns for train_llms and test_llms
        :param train_df: this is the train set where we assume we can access the 'Success' columns for train_llms; it is likely the one where reference_df was selected from
        :param validation_df: this is the validation set
        :param test_df: this is the test set
        :param feature: the feature to use for the prediction
        :param train_llms: the llms for which we have the 'Success' columns in the reference_df and train_df
        :param validation_llms: the llms for which we want to predict the 'Success' columns in the validation_df
        :param test_llms: the llms for which we want to predict the 'Success' columns in the test_df
        """

        self.reference_df = reference_df
        self.test_df = test_df
        self.train_df = train_df
        self.validation_df = validation_df
        self.feature = feature
        self.train_llms = train_llms
        self.validation_llms = validation_llms
        self.test_llms = test_llms

        # check if train and test llms have no intersection
        if len(set(train_llms).intersection(set(test_llms))) > 0:
            raise ValueError("The train and test llms should not have any intersection")
        # check if train and validation llms have no intersection
        if len(set(train_llms).intersection(set(validation_llms))) > 0:
            raise ValueError("The train and validation llms should not have any intersection")

        # check the number of unique values in the 'Success' columns for all train and test llms
        for llm in train_llms + validation_llms + test_llms:
            if len(test_df[f"Success_{llm}"].unique()) < 2:
                raise ValueError(
                    f"The 'Success' column for {llm} in the test_df should have at least 2 different values")
            if len(validation_df[f"Success_{llm}"].unique()) < 2:
                raise ValueError(
                    f"The 'Success' column for {llm} in the test_df should have at least 2 different values")

    def _predict_independently_on_test_llms(self, df, classifier=LogisticRegression, **kwargs):
        """Private method to handle the common logic of training the classifier and making predictions."""

        features_train = self._process_feature(df)
        features_validation = self._process_feature(self.validation_df)
        features_test = self._process_feature(self.test_df)

        results_per_llm_dict_val = {}
        results_per_llm_dict_test = {}

        for llms, results_dict, features in [(self.validation_llms, results_per_llm_dict_val, features_validation),
                                             (self.test_llms, results_per_llm_dict_test, features_test)]:
            for llm in llms:
                if len(df[f"Success_{llm}"].unique()) < 2:
                    print(
                        f"The 'Success' column for {llm} in the reference_df should have at least 2 different values for the "
                        f"_predict_independently_on_test_llms method to work; skipping this llm.")
                    continue

                clf = classifier(**kwargs)
                clf.fit(features_train, df[f'Success_{llm}'])
                results_dict[llm] = clf.predict_proba(features)[:, -1]

        return results_per_llm_dict_val, results_per_llm_dict_test

    def _process_feature(self, df):
        return np.array(list(df[self.feature].apply(lambda x: np.array(x)).values))

    def _extract_features_similarities_with_reference_df(self, df, features):

        if not isinstance(features, list):
            features = [features]
        if len(features) == 0:
            raise ValueError("At least one feature should be provided")

        # extract the various features
        features_array = None
        for feature in features:
            if feature not in ["embeddings"] + list(pairwise.distance_metrics().keys()):
                raise ValueError(f"Feature {feature} not recognized; must be ‘cityblock’,  ‘cosine’,  ‘euclidean’,  "
                                 f"‘haversine’,  ‘l1’,  ‘l2’,  ‘manhattan’,  ‘nan_euclidean’ or 'embeddings'")
            elif feature == "embeddings":
                features_array_new = self._process_feature(df)
            else:
                features_array_new = pairwise.pairwise_distances(
                    self._process_feature(df),
                    self._process_feature(self.reference_df),
                    metric=feature)
            if features_array is None:
                features_array = features_array_new
            else:
                features_array = np.concatenate((features_array, features_array_new), axis=1)

            # print(features_array.shape, features_test.shape)

        return features_array

    @staticmethod
    def _interaction_terms_features(X):

        # Expand dimensions of X along the third axis
        X_expanded = np.expand_dims(X, axis=2)

        # Compute the interaction terms using broadcasting
        X_interactions = X_expanded * X_expanded.transpose((0, 2, 1))

        # Take the upper triangular part of the interaction matrix (excluding the diagonal)
        i_upper = np.triu_indices(X.shape[1], k=1)
        X_interactions = X_interactions[:, i_upper[0], i_upper[1]]

        # concatenate original with new
        X = np.concatenate((X, X_interactions), axis=1)

        return X

    @staticmethod
    def _partial_interaction_terms_features(X):
        p = X.shape[1] // 2

        X_interactions = X[:, :p] * X[:, p:]

        # concatenate original with new
        X = np.concatenate((X, X_interactions), axis=1)

        return X

    def predict_upper_bound_train(self, classifier=LogisticRegression, **kwargs):
        """This is an upper bound for the performance that you can get by assuming we have access to the
         'Success' columns for the test_llms in the train_df, which is NOT realistic in practice. This is identical to
         the experiments done for each LLM independently in notebook 4."""
        return self._predict_independently_on_test_llms(self.train_df, classifier, **kwargs)

    def predict_reference_only(self, classifier=LogisticRegression, **kwargs):
        """This is a baseline where you only consider the reference benchmark to predict the performance on the
        new instances, not pooling together info across LLMs."""
        return self._predict_independently_on_test_llms(self.reference_df, classifier, **kwargs)

    def predict_concatenate_ref_success(self, classifier=LogisticRegression, features=["embeddings"],
                                        interaction_terms=None, **kwargs):
        """This is a method that concatenates the Success of each of the train_llms on all elements of the reference_df
        to the feature of each of the train_df, to build a single dataset that is used to train the classifier.
        Then, it predicts the success of the test_llms on the test_df using the feature of the test_df and the
        success on the reference_df.

        :classifier: the classifier to use
        :features: the features to use for the classifier, alongside the success on the reference dataset. "embeddings"
        means `self.feature` will be used; you can then add a similarity from sklearn.metrics.pairwise (such as
        cosine_similarity), in which case the similarity will be computed between the embeddings of each sample (train
        and test) and all the reference samples. Each element of the list must be either "embeddings" or one of
        ‘cityblock’,  ‘cosine’,  ‘euclidean’,  ‘haversine’,  ‘l1’,  ‘l2’,  ‘manhattan’,  ‘nan_euclidean’
        :interaction_terms: whether to add interaction terms to the features. Can be "full" or "partial". If "full", all
        pairwise interaction terms are added; if "partial", only the interaction terms between the first half of the
        features and the second half are added. That is meaningful when the considered features are one form of
        similarity, as this computes the interaction between the similarity and the correctness for each element of
        the reference dataset.
        """
        # notice here the train df was used both to select the reference dataset and to train the classifier.
        # should I at least remove the reference dataset from the training set? it probably does not matter much

        features_train = self._extract_features_similarities_with_reference_df(self.train_df, features)
        features_val = self._extract_features_similarities_with_reference_df(self.validation_df, features)
        features_test = self._extract_features_similarities_with_reference_df(self.test_df, features)

        # empty arrays
        features_train_total = None
        y_train_total = None
        for llm in self.train_llms:
            # append the results on the reference_df of this llm in new columns to the features_train array;
            # need to duplicate the success column to match the number of rows in features_train
            features_train_llm = np.concatenate(
                (features_train, np.tile(self.reference_df[f'Success_{llm}'], (features_train.shape[0], 1))), axis=1)

            # concatenate the features_train_llm to the features_train_total, in new rows
            if features_train_total is None:
                features_train_total = features_train_llm
            else:
                features_train_total = np.concatenate((features_train_total, features_train_llm), axis=0)

            # print(llm, features_train.shape, features_train_llm.shape, features_train_total.shape)

            # concatenate the success column of the train_df to the y_train_total
            if y_train_total is None:
                y_train_total = self.train_df[f'Success_{llm}']
            else:
                y_train_total = np.concatenate((y_train_total, self.train_df[f'Success_{llm}']))

        if interaction_terms == "full":
            features_train_total = self._interaction_terms_features(features_train_total)
        elif interaction_terms == "partial":
            features_train_total = self._partial_interaction_terms_features(features_train_total)

        # train the classifier
        clf = classifier(**kwargs)
        clf.fit(features_train_total, y_train_total)

        results_per_llm_dict_val = {}
        results_per_llm_dict_test = {}

        # now use the trained classifier to predict the success of the test_llms on the test_df and of the
        # validation_llms on validation_df
        for llms, results_dict, features in [(self.validation_llms, results_per_llm_dict_val, features_val),
                                             (self.test_llms, results_per_llm_dict_test, features_test)]:
            for llm in llms:
                # append the results on the reference_df of this llm in new columns to the features_test array;
                #  need to duplicate the success column to match the number of rows in features_test
                features_llm = np.concatenate(
                    (features, np.tile(self.reference_df[f'Success_{llm}'], (features.shape[0], 1))), axis=1)

                if interaction_terms == "full":
                    features_llm = self._interaction_terms_features(features_llm)
                elif interaction_terms == "partial":
                    features_llm = self._partial_interaction_terms_features(features_llm)

                # print(llm, features_test.shape, features_llm.shape)

                # predict the success of the test_llm on the test_df
                predictions = clf.predict_proba(features_llm)[:, -1]
                results_dict[llm] = predictions

        return results_per_llm_dict_val, results_per_llm_dict_test

    def predict_calibrate_general_classifier(self, classifier=LogisticRegression, calibration_step=True, **kwargs):
        """
        First Train a classifier on the successes of all train_llms on the training dataset (by creating a tall
        dataset). Then use the reference_df to calibrate the classifier fit before for the considered test_llms.

        This relies on the CalibratedClassifierCV from sklearn, used with `cv="prefit"`.

        :param classifier: the classifier to use
        :param calibration_step: whether to calibrate the classifier using the reference_df. If False, the resulting
        predictions will be a baseline, as the classifier will not be calibrated for the test llm
        :param kwargs: additional arguments to pass to the classifier
        :return:
        A dictionary with the predictions for each llm in test_llms
        """

        features_train = self._process_feature(self.train_df)

        features_train_total = np.concatenate([features_train for _ in self.train_llms])
        y_train_total = np.concatenate([self.train_df[f'Success_{llm}'] for llm in self.train_llms])

        # train the classifier
        clf = classifier(**kwargs)
        clf.fit(features_train_total, y_train_total)

        # now, for each test_llm, calibrate the classifier using the reference_df, and evaluate its performance on the
        # test_df. Same for the validation_df
        results_per_llm_dict_val = {}
        results_per_llm_dict_test = {}

        features_reference = self._process_feature(self.reference_df)
        features_validation = self._process_feature(self.validation_df)
        features_test = self._process_feature(self.test_df)

        for llms, results_dict, features in [(self.validation_llms, results_per_llm_dict_val, features_validation),
                                             (self.test_llms, results_per_llm_dict_test, features_test)]:
            for llm in llms:
                if calibration_step:
                    if len(self.reference_df[f"Success_{llm}"].unique()) < 2:
                        print(
                            f"The 'Success' column for {llm} in the reference_df should have at least 2 different values "
                            f"for the predict_calibrate_general_classifier method to work; skipping this llm.")
                        continue

                    # calibrate the classifier using the reference_df
                    y_reference = self.reference_df[f'Success_{llm}']
                    clf_calibrated = CalibratedClassifierCV(clf, cv="prefit")
                    clf_calibrated.fit(features_reference, y_reference)

                    # predict the success of the test_llm on the test_df
                    predictions = clf_calibrated.predict_proba(features)[:, -1]
                else:
                    # predict the success of the test_llm on the test_df
                    predictions = clf.predict_proba(features)[:, -1]

                results_dict[llm] = predictions

        return results_per_llm_dict_val, results_per_llm_dict_test

    def evaluate_predictions(self, results_per_llm_dict, subset="test"):
        """Evaluate a set of predictions by computing the average Brier score and the AUC on the test_df, for each llm in test_llms"""

        performances = []

        for llm in results_per_llm_dict:
            if subset == "test":
                labels_test = self.test_df[f'Success_{llm}']
            elif subset == "validation":
                labels_test = self.validation_df[f'Success_{llm}']
            else:
                raise ValueError(f"subset {subset} not recognized")
            y_pred = results_per_llm_dict[llm]

            # compute the Brier score
            BrierScore, Calibration, Refinement = brierDecomp(y_pred, labels_test)

            # compute the AUC
            if not (sum(labels_test) == 0 or sum(labels_test) == len(labels_test)):
                roc_auc = roc_auc_score(labels_test, y_pred)
            else:
                roc_auc = np.nan

            # compute accuracy by thresholding at 0.5
            y_pred_binary = y_pred > 0.5
            accuracy = np.mean(y_pred_binary == labels_test)

            performances.append({
                "llm": llm,
                "BrierScore": BrierScore,
                "Calibration": Calibration,
                "Refinement": Refinement,
                "AUROC": roc_auc,
                "Accuracy": accuracy,
                "Predictions": y_pred,
                "subset": subset
            })

        return performances

    # make a single method that calls the other

    def predict(self, method, classifier=LogisticRegression, **kwargs):
        assessor_methods_dict = {
            "upper_bound_train": self.predict_upper_bound_train,
            "reference_only": self.predict_reference_only,
            "concatenate_ref_success": self.predict_concatenate_ref_success,
            "calibrate_general_classifier": self.predict_calibrate_general_classifier
        }

        assessor_methods = list(assessor_methods_dict.keys())

        if method not in assessor_methods:
            raise ValueError(f"Method {method} not recognized")

        results_per_llm_dict = assessor_methods_dict[method](classifier=classifier, **kwargs)

        return results_per_llm_dict


if __name__ == "__main__":
    llms = ["text-davinci-001", "text-davinci-002",
            "text-davinci-003", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview"]
    train_llms = llms[0:4]
    test_llms = llms[4:]

    for split, arithmetic_only in [(False, False)]:
        train_df, test_df = load_evals(llms, ["openai_embeddings"], ood_split=split, arithmetic_only=arithmetic_only,
                                       base_path="feature_predictiveness/results", subsampled_n_train=300)
    # print(train_df.shape, test_df.shape)

    train_df["openai_embeddings_subsampled"] = train_df["openai_embeddings_large"].apply(lambda x: x[:1000])
    test_df["openai_embeddings_subsampled"] = test_df["openai_embeddings_large"].apply(lambda x: x[:1000])

    # define the selector
    selector = SampleSelector(train_df, "openai_embeddings_subsampled", train_llms)

    # compute IRT with d=5
    selector.compute_IRT_tiny_benchmarks(5, epochs=2000)

    # try all possible selection methods
    for method in selector.methods:
        print(f"Trying method {method}")
        selected_df = selector.select(method, 2)
        print(selected_df.shape)

    # now try the assessors
    # first select a subset of the train_df using random
    selected_df = selector.select("random", 50)

    assessor = AssessorFromReference(selected_df, train_df, test_df, "openai_embeddings_subsampled",
                                     train_llms, test_llms)

    # try all possible prediction methods
    for method in assessor.methods:
        print(f"Trying method {method}")
        results_per_llm_dict = assessor.predict(method)
        print(assessor.evaluate_predictions(results_per_llm_dict))

    # try the concatenate_ref_success method with cosine_similarity and interaction_terms
    results_per_llm_dict = assessor.predict("concatenate_ref_success", features=["cosine"],
                                            interaction_terms="full")
