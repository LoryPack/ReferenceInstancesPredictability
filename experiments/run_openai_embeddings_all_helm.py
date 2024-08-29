import os

from src.results_loaders import HelmResultsLoader

if __name__ == '__main__':

    save = True
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

    for scenario in possible_scenarios.keys():
        for subscenario in possible_scenarios[scenario]:
            print(f"Running {scenario}, {subscenario}")
            processed_filename = f"../results/helm_lite_v1.0.0_embeddings/{scenario}_{subscenario}_with_embeddings.gz"
            # create the folder if it does not exist
            os.makedirs(os.path.dirname("../results/helm_lite_v1.0.0_embeddings/"), exist_ok=True)

            instance = HelmResultsLoader(scenario=scenario, subscenario=subscenario,
                                         base_path_raw="../results/helm_lite_v1.0.0/",
                                         verbose=False, )

            instance.discard_if_one_na_in_success_per_row(inplace=True,print_n_discarded=True)

            print("Shape before embeddings are computed:", instance.results_df.shape)

            instance.extract_openai_embeddings(skip_na_rows=False,
                                               env_file="../.env", add_system_prompt=False)
            if save:
                instance.save_processed(processed_filename, compression=True)
            print("Shape after embeddings are computed:", instance.results_df.shape)
