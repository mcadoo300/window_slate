import numpy as np
import pandas as pd
# import open bandit pipeline (obp)
import obp
import pdb
from obp.ope import SlateStandardIPS, SlateIndependentIPS, SlateRewardInteractionIPS, SlateOffPolicyEvaluation
from obp.dataset import (
    logistic_reward_function,
    logistic_sparse_reward_function,
    SyntheticSlateBanditDataset,
    linear_behavior_policy_logit,
    linear_reward_function,
    polynomial_reward_function
)

from itertools import product
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

# obp version
print(obp.__version__)

import warnings
warnings.filterwarnings('ignore')

#new imports
from obp.ope.estimators_slate import SlateIndependentWindow, SlateIndependentWindowStandard, SlateIndependentWindow2
from obp.utils import softmax


# new functions
def variedPolicy(logit, lmbda, similar=False):
    if similar:
            evaluation_policy_logit_ = (
                1 - epsilon
            ) * logit + epsilon * np.ones(logit.shape)
    else:  # "dissimilar"
        evaluation_policy_logit_ = (
            1 - epsilon
        ) * -logit + epsilon * np.ones(
            logit.shape
        )
    return evaluation_policy_logit_

# generate a synthetic bandit dataset with 10 actions
# we use `logistic_reward_function` as the reward function and `linear_behavior_policy_logit` as the behavior policy.
# one can define their own reward function and behavior policy such as nonlinear ones.
rand_sips_se=0
rand_iips_se=0
rand_rips_se=0
rand_wips_se=0

opt_sips_se=0
opt_iips_se=0
opt_rips_se=0
opt_wips_se=0

anti_sips_se=0
anti_iips_se=0
anti_rips_se=0
anti_wips_se=0

num_seed=10
seed_count = 10
settings = [4000]
for round_test in settings:
    sips_data = np.empty((0, 3), dtype=object)
    iips_data = np.empty((0, 3), dtype=object)
    rips_data = np.empty((0, 3), dtype=object)
    wiips_data = np.empty((0, 3), dtype=object)
    wipss_data = np.empty((0, 3), dtype=object)
    wiipsfull_data = np.empty((0, 3), dtype=object)
    for seed in range(seed_count):
        logging=False
        epsilon=0.5
        n_unique_action=10
        len_list = 4

        dim_context = 5
        reward_type = "continuous"
        reward_structure="window_additive"
        click_model=None
        random_state=seed
        base_reward_function=linear_reward_function

        # obtain  test sets of synthetic logged bandit data
        n_rounds_test = round_test


        # define Uniform Random Policy as a baseline behavior policy
        dataset_with_random_behavior = SyntheticSlateBanditDataset(
            n_unique_action=n_unique_action,
            len_list=len_list,
            dim_context=dim_context,
            reward_type=reward_type,
            reward_structure=reward_structure,
            click_model=click_model,
            random_state=random_state,
            behavior_policy_function=linear_behavior_policy_logit,
            base_reward_function=base_reward_function,
        )

        # compute the factual action choice probabililties for the test set of the synthetic logged bandit data
        bandit_feedback_with_random_behavior = dataset_with_random_behavior.obtain_batch_bandit_feedback(
            n_rounds=n_rounds_test,
            return_pscore_item_position=True,
        )

        
        #print(bandit_feedback_with_random_behavior)
        if logging:
            pdb.set_trace()
            print(bandit_feedback_with_random_behavior)
            behavior_pscores=[dataset_with_random_behavior.behavior_logit,
                    bandit_feedback_with_random_behavior["pscore_cascade"],
                    bandit_feedback_with_random_behavior["pscore"],
                    bandit_feedback_with_random_behavior["pscore_item_position"],
                    bandit_feedback_with_random_behavior["pscore_idp_window"]]
            print(behavior_pscores)
            rows=['behavior logit','Cascade','standard','independent','independent window']
            df = pd.DataFrame(behavior_pscores,index=rows)
            df.to_excel('behavior_pscores.xlsx')
        # print policy value
        random_policy_value = dataset_with_random_behavior.calc_on_policy_policy_value(
            reward=bandit_feedback_with_random_behavior["reward"],
            slate_id=bandit_feedback_with_random_behavior["slate_id"],
        )


        random_policy_logit_ = np.zeros((n_rounds_test, n_unique_action))
        similar_policy_logit = variedPolicy(dataset_with_random_behavior.behavior_logit,epsilon,similar=True)
        dissimilar_policy_logit_ = variedPolicy(dataset_with_random_behavior.behavior_logit,epsilon,similar=False)


        random_policy_pscores = dataset_with_random_behavior.obtain_pscore_given_evaluation_policy_logit(
            action=bandit_feedback_with_random_behavior["action"],
            evaluation_policy_logit_=random_policy_logit_
        )

        optimal_policy_pscores = dataset_with_random_behavior.obtain_pscore_given_evaluation_policy_logit(
            action=bandit_feedback_with_random_behavior["action"],
            evaluation_policy_logit_=similar_policy_logit
        )


        anti_optimal_policy_pscores = dataset_with_random_behavior.obtain_pscore_given_evaluation_policy_logit(
            action=bandit_feedback_with_random_behavior["action"],
            evaluation_policy_logit_=dissimilar_policy_logit_
        )


        sips = SlateStandardIPS(len_list=len_list)
        iips = SlateIndependentIPS(len_list=len_list)
        rips = SlateRewardInteractionIPS(len_list=len_list)
        wiips2 = SlateIndependentWindow2(len_list=len_list)


        ope = SlateOffPolicyEvaluation(
            bandit_feedback=bandit_feedback_with_random_behavior,
            ope_estimators=[sips,wiips2, iips, rips]
        )
        
        _, estimated_interval_random = ope.summarize_off_policy_estimates(
            evaluation_policy_pscore=random_policy_pscores[0],
            evaluation_policy_pscore_item_position=random_policy_pscores[1],
            evaluation_policy_pscore_cascade=random_policy_pscores[2],
            alpha=0.05,
            n_bootstrap_samples=1000,
            random_state=dataset_with_random_behavior.random_state,
            evaluation_policy_pscore_idp_window=random_policy_pscores[3]
        )

        estimated_interval_random["policy_name"] = "random"
        ope.visualize_off_policy_estimates(
            evaluation_policy_pscore=random_policy_pscores[0],
            evaluation_policy_pscore_item_position=random_policy_pscores[1],
            evaluation_policy_pscore_cascade=random_policy_pscores[2],
            alpha=0.05,
            n_bootstrap_samples=1000, # number of resampling performed in bootstrap sampling
            random_state=dataset_with_random_behavior.random_state,
            evaluation_policy_pscore_idp_window=random_policy_pscores[3],
        )
        #pdb.set_trace()

        _, estimated_interval_similar = ope.summarize_off_policy_estimates(
            evaluation_policy_pscore=optimal_policy_pscores[0],
            evaluation_policy_pscore_item_position=optimal_policy_pscores[1],
            evaluation_policy_pscore_cascade=optimal_policy_pscores[2],
            alpha=0.05,
            n_bootstrap_samples=1000,
            random_state=dataset_with_random_behavior.random_state,
            evaluation_policy_pscore_idp_window=optimal_policy_pscores[3],
        )

        estimated_interval_similar["policy_name"] = "optimal"
        ope.visualize_off_policy_estimates(
            evaluation_policy_pscore=optimal_policy_pscores[0],
            evaluation_policy_pscore_item_position=optimal_policy_pscores[1],
            evaluation_policy_pscore_cascade=optimal_policy_pscores[2],
            alpha=0.05,
            n_bootstrap_samples=1000, # number of resampling performed in bootstrap sampling
            random_state=dataset_with_random_behavior.random_state,
            evaluation_policy_pscore_idp_window=optimal_policy_pscores[3],
        )

        _, estimated_interval_dissimilar = ope.summarize_off_policy_estimates(
            evaluation_policy_pscore=anti_optimal_policy_pscores[0],
            evaluation_policy_pscore_item_position=anti_optimal_policy_pscores[1],
            evaluation_policy_pscore_cascade=anti_optimal_policy_pscores[2],
            alpha=0.05,
            n_bootstrap_samples=1000,
            random_state=dataset_with_random_behavior.random_state,
            evaluation_policy_pscore_idp_window=anti_optimal_policy_pscores[3],
        )
        estimated_interval_dissimilar["policy_name"] = "anti-optimal"
        ope.visualize_off_policy_estimates(
            evaluation_policy_pscore=anti_optimal_policy_pscores[0],
            evaluation_policy_pscore_item_position=anti_optimal_policy_pscores[1],
            evaluation_policy_pscore_cascade=anti_optimal_policy_pscores[2],
            alpha=0.05,
            n_bootstrap_samples=1000, # number of resampling performed in bootstrap sampling
            random_state=dataset_with_random_behavior.random_state,
            evaluation_policy_pscore_idp_window=anti_optimal_policy_pscores[3],
        )

        ground_truth_policy_value_random = dataset_with_random_behavior.calc_ground_truth_policy_value(
            context=bandit_feedback_with_random_behavior["context"],
            evaluation_policy_logit_=random_policy_logit_
        )

        ground_truth_policy_value_optimal = dataset_with_random_behavior.calc_ground_truth_policy_value(
            context=bandit_feedback_with_random_behavior["context"],
            evaluation_policy_logit_=similar_policy_logit
        )
        ground_truth_policy_value_optimal

        ground_truth_policy_value_anti_optimal = dataset_with_random_behavior.calc_ground_truth_policy_value(
            context=bandit_feedback_with_random_behavior["context"],
            evaluation_policy_logit_=dissimilar_policy_logit_
        )
        ground_truth_policy_value_anti_optimal

        #pdb.set_trace()
        estimated_interval_random["ground_truth"] = ground_truth_policy_value_random
        estimated_interval_similar["ground_truth"] = ground_truth_policy_value_optimal
        estimated_interval_dissimilar["ground_truth"] = ground_truth_policy_value_anti_optimal
        #pdb.set_trace()
        estimated_intervals = pd.concat(
            [
                estimated_interval_random,
                estimated_interval_similar,
                estimated_interval_dissimilar
            ]
        )
        #pdb.set_trace()
        #estimated_intervals.to_excel(f"intervals_{reward_structure}_{epsilon}_{len_list}_{n_unique_action}_{n_rounds_test}_total_error.xlsx")

        # evaluate the estimation performances of OPE estimators 
        # by comparing the estimated policy values and its ground-truth.
        # `summarize_estimators_comparison` returns a pandas dataframe containing estimation performances of given estimators 
        #pdb.set_trace()
        relative_ee_for_random_evaluation_policy = ope.summarize_estimators_comparison(
            ground_truth_policy_value=ground_truth_policy_value_random,
            evaluation_policy_pscore=random_policy_pscores[0],
            evaluation_policy_pscore_item_position=random_policy_pscores[1],
            evaluation_policy_pscore_cascade=random_policy_pscores[2],
            evaluation_policy_pscore_idp_window=random_policy_pscores[3],
        )
        #pdb.set_trace()

        # evaluate the estimation performances of OPE estimators 
        # by comparing the estimated policy values and its ground-truth.
        # `summarize_estimators_comparison` returns a pandas dataframe containing estimation performances of given estimators 

        relative_ee_for_similar_evaluation_policy = ope.summarize_estimators_comparison(
            ground_truth_policy_value=ground_truth_policy_value_optimal,
            evaluation_policy_pscore=optimal_policy_pscores[0],
            evaluation_policy_pscore_item_position=optimal_policy_pscores[1],
            evaluation_policy_pscore_cascade=optimal_policy_pscores[2],
            evaluation_policy_pscore_idp_window=optimal_policy_pscores[3],
        )

        # evaluate the estimation performances of OPE estimators 
        # by comparing the estimated policy values and its ground-truth.
        # `summarize_estimators_comparison` returns a pandas dataframe containing estimation performances of given estimators 

        relative_ee_for_dissimilar_evaluation_policy = ope.summarize_estimators_comparison(
            ground_truth_policy_value=ground_truth_policy_value_anti_optimal,
            evaluation_policy_pscore=anti_optimal_policy_pscores[0],
            evaluation_policy_pscore_item_position=anti_optimal_policy_pscores[1],
            evaluation_policy_pscore_cascade=anti_optimal_policy_pscores[2],
            evaluation_policy_pscore_idp_window=anti_optimal_policy_pscores[3],
        
        )
        print(relative_ee_for_random_evaluation_policy)
        print(relative_ee_for_similar_evaluation_policy)
        print(relative_ee_for_dissimilar_evaluation_policy)
        #pdb.set_trace()
        #print(relative_ee_for_random_evaluation_policy["se"]["sips"])



        row = np.array([[relative_ee_for_random_evaluation_policy["se"]["sips"],
                        relative_ee_for_similar_evaluation_policy["se"]["sips"],
                        relative_ee_for_dissimilar_evaluation_policy["se"]["sips"]]])
        sips_data = np.append(sips_data, row, axis=0)

        row = np.array([[relative_ee_for_random_evaluation_policy["se"]["iips"],
                        relative_ee_for_similar_evaluation_policy["se"]["iips"],
                        relative_ee_for_dissimilar_evaluation_policy["se"]["iips"]]])
        iips_data = np.append(iips_data, row, axis=0)

        row = np.array([[relative_ee_for_random_evaluation_policy["se"]["rips"],
                        relative_ee_for_similar_evaluation_policy["se"]["rips"],
                        relative_ee_for_dissimilar_evaluation_policy["se"]["rips"]]])
        rips_data = np.append(rips_data, row, axis=0)

        wiips_data = np.append(wiips_data, row, axis=0)

        row = np.array([[relative_ee_for_random_evaluation_policy["se"]["wiipsfull"],
                        relative_ee_for_similar_evaluation_policy["se"]["wiipsfull"],
                        relative_ee_for_dissimilar_evaluation_policy["se"]["wiipsfull"]]])
        wiipsfull_data = np.append(wiipsfull_data, row, axis=0)

        wipss_data = np.append(wipss_data, row, axis=0)

    df = pd.DataFrame(sips_data, columns=["random", "similar", "dissimilar"])
    # Save the DataFrame to a file (e.g., CSV)
    #pdb.set_trace()
    df.to_excel(f"slate_{len_list}/sips_{reward_structure}_linear_reward_function_{epsilon}_{len_list}_{n_rounds_test}_error_{seed_count}_verify.xlsx", index=False)

    df = pd.DataFrame(iips_data, columns=["random", "similar", "dissimilar"])
    # Save the DataFrame to a file (e.g., CSV)
    df.to_excel(f"slate_{len_list}/iips_{reward_structure}_linear_reward_function_{epsilon}_{len_list}_{n_rounds_test}_error_{seed_count}_verify.xlsx", index=False)

    df = pd.DataFrame(rips_data, columns=["random", "similar", "dissimilar"])
    # Save the DataFrame to a file (e.g., CSV)
    df.to_excel(f"slate_{len_list}/rips_{reward_structure}_linear_reward_function_{epsilon}_{len_list}_{n_rounds_test}_error_{seed_count}_verify.xlsx", index=False)

    df = pd.DataFrame(wiipsfull_data, columns=["random", "similar", "dissimilar"])
    # Save the DataFrame to a file (e.g., CSV)
    df.to_excel(f"slate_{len_list}/wiipsfull_{reward_structure}_linear_reward_function_{epsilon}_{len_list}_{n_rounds_test}_error_{seed_count}_verify.xlsx", index=False)
 