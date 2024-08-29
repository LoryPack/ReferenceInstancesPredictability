import os

from src.results_loaders import EvalsResultsLoader

if __name__ == '__main__':

    all_evals = ["odd_one_out", "cause_and_effect_one_sentence", "cause_and_effect_two_sentences", "crass_ai",
                 "logical_args", "emoji_movie", "fantasy_reasoning", "metaphor_boolean", "geometric_shapes",
                 "space_nli", "abstract_narrative_understanding_4_distractors", "arithmetic_1_digit_division",
                 "arithmetic_1_digit_subtraction", "arithmetic_1_digit_addition", "arithmetic_1_digit_multiplication",
                 "arithmetic_2_digit_division", "arithmetic_3_digit_division", "arithmetic_2_digit_multiplication",
                 "arithmetic_2_digit_addition", "arithmetic_2_digit_subtraction", "arithmetic_3_digit_multiplication",
                 "arithmetic_3_digit_addition", "arithmetic_3_digit_subtraction", "arithmetic_4_digit_multiplication",
                 "arithmetic_4_digit_addition", "arithmetic_4_digit_subtraction", "arithmetic_4_digit_division",
                 "arithmetic_5_digit_multiplication", "arithmetic_5_digit_addition", "arithmetic_5_digit_subtraction",
                 "arithmetic_5_digit_division", "copa", "anli", "cosmos_qa", "ropes",
                 "goal_step_wikihow_goal_inference", "alpha_nli", "goal_step_wikihow_step_inference",
                 "abstract_narrative_understanding_9_distractors", "goal_step_wikihow_step_ordering", "wanli",
                 "babi_task_16", "abstract_narrative_understanding_99_distractors",
                 "formal_fallacies_syllogisms_negation"]

    # add the aritmetic results:
    for n_digits in range(1, 6):
        for operation in ["addition", "subtraction", "multiplication", "division"]:
            eval_name = f"arithmetic_{n_digits}_digit_{operation}"
            all_evals.append(eval_name)

    # 'dyck_languages',  # exclude as there is some issue there.
    expensive_evals = [
        'social_iqa', 'goal_step_wikihow_goal_inference', 'alpha_nli',
        'goal_step_wikihow_step_inference', 'abstract_narrative_understanding_9_distractors',
        'goal_step_wikihow_step_ordering', 'wanli', 'babi_task_16', 'roc_stories',
        'formal_fallacies_syllogisms_negation', 'intersect_geometry_shapes_2',
        'intersect_geometry_shapes_3', 'intersect_geometry_shapes_4',
        'intersect_geometry_shapes_5', 'intersect_geometry_shapes_6', 'abstract_narrative_understanding_99_distractors',
        'mnist_ascii']

    save = True
    max_n_samples_list = [
        10000,
        1000
    ]

    evals_list = [
        all_evals,
        expensive_evals
    ]

    for evals_to_use, max_n_samples in zip(evals_list, max_n_samples_list):
        for eval in evals_to_use:
            print(f"Running {eval}, number {evals_to_use.index(eval) + 1}/{len(evals_to_use)}")
            processed_filename = f"../results/kindsofreasoning_embeddings/{eval}_with_embeddings.gz"
            # create the folder if it does not exist
            os.makedirs(os.path.dirname("../results/kindsofreasoning_embeddings/"), exist_ok=True)

            instance = EvalsResultsLoader(task=eval, base_path_raw="../results/kindsofreasoning/",
                                          verbose=False, )
            print("Shape before embeddings are computed:", instance.results_df.shape)

            instance.extract_openai_embeddings(max_n_samples=max_n_samples, skip_na_rows=False,
                                               env_file="../.env", add_system_prompt=False)
            if save:
                instance.save_processed(processed_filename, compression=True)
            print("Shape after embeddings are computed:", instance.results_df.shape)

            # now do with the system prompt if the dataset has that
            if hasattr(instance, "system_prompt"):
                instance.extract_openai_embeddings(max_n_samples=max_n_samples, skip_na_rows=False,
                                                   env_file="../.env", add_system_prompt=True)
                if save:
                    instance.save_processed(processed_filename, compression=True)

                print("Shape after embeddings are computed with system prompt:", instance.results_df.shape)
