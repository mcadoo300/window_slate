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
    linear_behavior_policy_logit
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

settings=[1000,2000,3000,4000,5000]

num_seed=10



for s in settings:
    sips_data = np.empty((0, 3), dtype=object)
    iips_data = np.empty((0, 3), dtype=object)
    rips_data = np.empty((0, 3), dtype=object)
    wiips_data = np.empty((0, 3), dtype=object)
    wipss_data = np.empty((0, 3), dtype=object)
    wiipsfull_data = np.empty((0, 3), dtype=object)
    for seed in range(10):
        logging=False
        epsilon=0.5
        n_unique_action=10
        len_list = 5

        dim_context = 2
        reward_type = "binary"
        reward_structure="window_additive"
        click_model=None
        random_state=seed
        base_reward_function=logistic_reward_function

        # obtain  test sets of synthetic logged bandit data
        n_rounds_test = s



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
        #pdb.set_trace()
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
        #print(random_policy_value)
        #pdb.set_trace()
        #print(bandit_feedback_with_random_behavior)

        random_policy_logit_ = np.zeros((n_rounds_test, n_unique_action))

        """base_expected_reward = dataset_with_random_behavior.base_reward_function(
            context=bandit_feedback_with_random_behavior["context"],
            action_context=dataset_with_random_behavior.action_context,
            random_state=dataset_with_random_behavior.random_state,
        )"""

        similar_policy_logit = variedPolicy(dataset_with_random_behavior.behavior_logit,epsilon,similar=True)
        dissimilar_policy_logit_ = variedPolicy(dataset_with_random_behavior.behavior_logit,epsilon,similar=False)
        if logging:
            pdb.set_trace()
            evaluation_logits=[random_policy_logit_[0],similar_policy_logit[0],dissimilar_policy_logit_[0]]
            #print(evaluation_logits)
            rows=['random','optimal','anti-optimal']
            df = pd.DataFrame(evaluation_logits,index=rows)
            df.to_excel('evaluation_logits.xlsx')

        #pdb.set_trace()
        random_policy_pscores = dataset_with_random_behavior.obtain_pscore_given_evaluation_policy_logit(
            action=bandit_feedback_with_random_behavior["action"],
            evaluation_policy_logit_=random_policy_logit_
        )
        #pdb.set_trace()
        #print(random_policy_pscores)
        optimal_policy_pscores = dataset_with_random_behavior.obtain_pscore_given_evaluation_policy_logit(
            action=bandit_feedback_with_random_behavior["action"],
            evaluation_policy_logit_=similar_policy_logit
        )
        #pdb.set_trace()
        #print(optimal_policy_pscores)

        anti_optimal_policy_pscores = dataset_with_random_behavior.obtain_pscore_given_evaluation_policy_logit(
            action=bandit_feedback_with_random_behavior["action"],
            evaluation_policy_logit_=dissimilar_policy_logit_
        )
        #pdb.set_trace()
        #print(anti_optimal_policy_pscores)
        if logging:
            pdb.set_trace()
            #print(random_policy_pscores)
            evaluation_pscores=[random_policy_pscores,optimal_policy_pscores,anti_optimal_policy_pscores]
            #print(evaluation_pscores)
            rows=['random','optimal','anti-optimal']
            df = pd.DataFrame(evaluation_pscores,index=rows,columns=["standard","independent","cascade", "window"])
            df.to_excel('evaluation_pscores.xlsx')
        # estimate the policy value of the evaluation policies based on their action choice probabilities
        # it is possible to set multiple OPE estimators to the `ope_estimators` argument

        sips = SlateStandardIPS(len_list=len_list)
        iips = SlateIndependentIPS(len_list=len_list)
        rips = SlateRewardInteractionIPS(len_list=len_list)
        wiips = SlateIndependentWindow(len_list=len_list)
        wipss = SlateIndependentWindowStandard(len_list=len_list)
        wiips2 = SlateIndependentWindow2(len_list=len_list)


        ope = SlateOffPolicyEvaluation(
            bandit_feedback=bandit_feedback_with_random_behavior,
            ope_estimators=[wipss, sips,wiips,wiips2, iips, rips]
        )

        #pdb.set_trace()
        #print(random_policy_pscores)
        #print(random_policy_pscores[5])
        #print(random_policy_pscores[4])
        #print(random_policy_pscores[1])
        
        _, estimated_interval_random = ope.summarize_off_policy_estimates(
            evaluation_policy_pscore=random_policy_pscores[0],
            evaluation_policy_pscore_item_position=random_policy_pscores[1],
            evaluation_policy_pscore_cascade=random_policy_pscores[2],
            alpha=0.05,
            n_bootstrap_samples=1000,
            random_state=dataset_with_random_behavior.random_state,
            evaluation_policy_pscore_idp_window=random_policy_pscores[3],
            evaluation_policy_pscore_idp_window_normal=random_policy_pscores[4],
            evaluation_policy_pscore_idp_window2=random_policy_pscores[5],
        )

        estimated_interval_random["policy_name"] = "random"
        #pdb.set_trace()
        #print(estimated_interval_random)
        #pdb.set_trace()
        #print(estimated_interval_random, '\n')
        # visualize estimated policy values of Uniform Random by the three OPE estimators
        # and their 95% confidence intervals (estimated by nonparametric bootstrap method)
        ope.visualize_off_policy_estimates(
            evaluation_policy_pscore=random_policy_pscores[0],
            evaluation_policy_pscore_item_position=random_policy_pscores[1],
            evaluation_policy_pscore_cascade=random_policy_pscores[2],
            alpha=0.05,
            n_bootstrap_samples=1000, # number of resampling performed in bootstrap sampling
            random_state=dataset_with_random_behavior.random_state,
            evaluation_policy_pscore_idp_window=random_policy_pscores[3],
            evaluation_policy_pscore_idp_window_normal=random_policy_pscores[4],
            evaluation_policy_pscore_idp_window2=random_policy_pscores[5],
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
            evaluation_policy_pscore_idp_window_normal=optimal_policy_pscores[4],
            evaluation_policy_pscore_idp_window2=optimal_policy_pscores[5],
        )

        estimated_interval_similar["policy_name"] = "optimal"
        #pdb.set_trace()
        print(estimated_interval_similar, '\n')
        # visualize estimated policy values of Optimal by the three OPE estimators
        # and their 95% confidence intervals (estimated by nonparametric bootstrap method)
        ope.visualize_off_policy_estimates(
            evaluation_policy_pscore=optimal_policy_pscores[0],
            evaluation_policy_pscore_item_position=optimal_policy_pscores[1],
            evaluation_policy_pscore_cascade=optimal_policy_pscores[2],
            alpha=0.05,
            n_bootstrap_samples=1000, # number of resampling performed in bootstrap sampling
            random_state=dataset_with_random_behavior.random_state,
            evaluation_policy_pscore_idp_window=optimal_policy_pscores[3],
            evaluation_policy_pscore_idp_window_normal=optimal_policy_pscores[4],
            evaluation_policy_pscore_idp_window2=optimal_policy_pscores[5],
        )

        _, estimated_interval_dissimilar = ope.summarize_off_policy_estimates(
            evaluation_policy_pscore=anti_optimal_policy_pscores[0],
            evaluation_policy_pscore_item_position=anti_optimal_policy_pscores[1],
            evaluation_policy_pscore_cascade=anti_optimal_policy_pscores[2],
            alpha=0.05,
            n_bootstrap_samples=1000,
            random_state=dataset_with_random_behavior.random_state,
            evaluation_policy_pscore_idp_window=anti_optimal_policy_pscores[3],
            evaluation_policy_pscore_idp_window_normal=anti_optimal_policy_pscores[4],
            evaluation_policy_pscore_idp_window2=anti_optimal_policy_pscores[5],
        )
        estimated_interval_dissimilar["policy_name"] = "anti-optimal"

        print(estimated_interval_dissimilar, '\n')
        # visualize estimated policy values of Anti-optimal by the three OPE estimators
        # and their 95% confidence intervals (estimated by nonparametric bootstrap method)
        ope.visualize_off_policy_estimates(
            evaluation_policy_pscore=anti_optimal_policy_pscores[0],
            evaluation_policy_pscore_item_position=anti_optimal_policy_pscores[1],
            evaluation_policy_pscore_cascade=anti_optimal_policy_pscores[2],
            alpha=0.05,
            n_bootstrap_samples=1000, # number of resampling performed in bootstrap sampling
            random_state=dataset_with_random_behavior.random_state,
            evaluation_policy_pscore_idp_window=anti_optimal_policy_pscores[3],
            evaluation_policy_pscore_idp_window_normal=anti_optimal_policy_pscores[4],
            evaluation_policy_pscore_idp_window2=anti_optimal_policy_pscores[5],
        )

        ground_truth_policy_value_random = dataset_with_random_behavior.calc_ground_truth_policy_value(
            context=bandit_feedback_with_random_behavior["context"],
            evaluation_policy_logit_=random_policy_logit_
        )
        #pdb.set_trace()
        print(ground_truth_policy_value_random)

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
        estimated_intervals.to_excel(f"intervals_{reward_structure}_{epsilon}_{len_list}_{n_unique_action}_{n_rounds_test}_total_error.xlsx")

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
            evaluation_policy_pscore_idp_window_normal=random_policy_pscores[4],
            evaluation_policy_pscore_idp_window2=random_policy_pscores[5],
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
            evaluation_policy_pscore_idp_window_normal=optimal_policy_pscores[4],
            evaluation_policy_pscore_idp_window2=optimal_policy_pscores[5],
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
            evaluation_policy_pscore_idp_window_normal=anti_optimal_policy_pscores[4],
            evaluation_policy_pscore_idp_window2=anti_optimal_policy_pscores[5],
        
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

        row = np.array([[relative_ee_for_random_evaluation_policy["se"]["wiips"],
                        relative_ee_for_similar_evaluation_policy["se"]["wiips"],
                        relative_ee_for_dissimilar_evaluation_policy["se"]["wiips"]]])
        wiips_data = np.append(wiips_data, row, axis=0)

        row = np.array([[relative_ee_for_random_evaluation_policy["se"]["wiipsfull"],
                        relative_ee_for_similar_evaluation_policy["se"]["wiipsfull"],
                        relative_ee_for_dissimilar_evaluation_policy["se"]["wiipsfull"]]])
        wiipsfull_data = np.append(wiipsfull_data, row, axis=0)

        row = np.array([[relative_ee_for_random_evaluation_policy["se"]["wipss"],
                        relative_ee_for_similar_evaluation_policy["se"]["wipss"],
                        relative_ee_for_dissimilar_evaluation_policy["se"]["wipss"]]])
        wipss_data = np.append(wipss_data, row, axis=0)

    df = pd.DataFrame(sips_data, columns=["random", "similar", "dissimilar"])
    # Save the DataFrame to a file (e.g., CSV)
    df.to_excel(f"slate_5/sips_{reward_structure}_{epsilon}_{len_list}_{n_rounds_test}_error.xlsx", index=False)

    df = pd.DataFrame(iips_data, columns=["random", "similar", "dissimilar"])
    # Save the DataFrame to a file (e.g., CSV)
    df.to_excel(f"slate_5/iips_{reward_structure}_{epsilon}_{len_list}_{n_rounds_test}_error.xlsx", index=False)

    df = pd.DataFrame(rips_data, columns=["random", "similar", "dissimilar"])
    # Save the DataFrame to a file (e.g., CSV)
    df.to_excel(f"slate_5/rips_{reward_structure}_{epsilon}_{len_list}_{n_rounds_test}_error.xlsx", index=False)

    df = pd.DataFrame(wipss_data, columns=["random", "similar", "dissimilar"])
    # Save the DataFrame to a file (e.g., CSV)
    df.to_excel(f"slate_5/wipss_{reward_structure}_{epsilon}_{len_list}_{n_rounds_test}_error.xlsx", index=False)

    df = pd.DataFrame(wiips_data, columns=["random", "similar", "dissimilar"])
    # Save the DataFrame to a file (e.g., CSV)
    df.to_excel(f"slate_5/wiips_{reward_structure}_{epsilon}_{len_list}_{n_rounds_test}_error.xlsx", index=False)

    df = pd.DataFrame(wiipsfull_data, columns=["random", "similar", "dissimilar"])
    # Save the DataFrame to a file (e.g., CSV)
    df.to_excel(f"slate_5/wiipsfull_{reward_structure}_{epsilon}_{len_list}_{n_rounds_test}_error.xlsx", index=False)
