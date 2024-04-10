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
from obp.ope.estimators_slate import SlateIndependentWindow, SlateIndependentWindowNormal
from obp.utils import softmax


# new functions
def variedPolicy(logit, lmbda):
    #pdb.set_trace()
    #print(logit)
    mtrx_dif = np.ones((n_rounds_test, n_unique_action))* (1-abs(lmbda))
    #print(mtrx_dif)
    new_logit = logit * lmbda
    #print(new_logit)
    new_logit+= mtrx_dif
    #print(new_logit)
    new_policy = softmax(new_logit)
    #print(new_policy)
    return new_policy

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

settings=[250,500,1000,2000,5000]

for setting in range(1):
    logging=False
    epsilon=0.5
    n_unique_action=10
    len_list = 7
    dim_context = 2
    reward_type = "binary"
    reward_structure="window_additive"
    click_model=None
    random_state=12345
    base_reward_function=logistic_reward_function

    # obtain  test sets of synthetic logged bandit data
    n_rounds_test = 1000



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

    optimal_policy_logit_ = variedPolicy(dataset_with_random_behavior.behavior_logit,epsilon)
    anti_optimal_policy_logit_ = variedPolicy(dataset_with_random_behavior.behavior_logit,-epsilon)
    if logging:
        pdb.set_trace()
        evaluation_logits=[random_policy_logit_[0],optimal_policy_logit_[0],anti_optimal_policy_logit_[0]]
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
        evaluation_policy_logit_=optimal_policy_logit_
    )
    #pdb.set_trace()
    #print(optimal_policy_pscores)

    anti_optimal_policy_pscores = dataset_with_random_behavior.obtain_pscore_given_evaluation_policy_logit(
        action=bandit_feedback_with_random_behavior["action"],
        evaluation_policy_logit_=anti_optimal_policy_logit_
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
    wips = SlateIndependentWindow(len_list=len_list)
    wipsn = SlateIndependentWindowNormal(len_list=len_list)


    ope = SlateOffPolicyEvaluation(
        bandit_feedback=bandit_feedback_with_random_behavior,
        ope_estimators=[wipsn, sips,wips, iips, rips]
    )

    #pdb.set_trace()
    #print(random_policy_pscores)
    #print(random_policy_pscores[0])
    #print(random_policy_pscores[4])
    
    _, estimated_interval_random = ope.summarize_off_policy_estimates(
        evaluation_policy_pscore=random_policy_pscores[0],
        evaluation_policy_pscore_item_position=random_policy_pscores[1],
        evaluation_policy_pscore_cascade=random_policy_pscores[2],
        alpha=0.05,
        n_bootstrap_samples=1000,
        random_state=dataset_with_random_behavior.random_state,
        evaluation_policy_pscore_idp_window=random_policy_pscores[3],
        evaluation_policy_pscore_idp_window_normal=random_policy_pscores[4],
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
    )
    #pdb.set_trace()

    _, estimated_interval_optimal = ope.summarize_off_policy_estimates(
        evaluation_policy_pscore=optimal_policy_pscores[0],
        evaluation_policy_pscore_item_position=optimal_policy_pscores[1],
        evaluation_policy_pscore_cascade=optimal_policy_pscores[2],
        alpha=0.05,
        n_bootstrap_samples=1000,
        random_state=dataset_with_random_behavior.random_state,
        evaluation_policy_pscore_idp_window=optimal_policy_pscores[3],
        evaluation_policy_pscore_idp_window_normal=optimal_policy_pscores[4],
    )

    estimated_interval_optimal["policy_name"] = "optimal"
    #pdb.set_trace()
    print(estimated_interval_optimal, '\n')
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
    )

    _, estimated_interval_anti_optimal = ope.summarize_off_policy_estimates(
        evaluation_policy_pscore=anti_optimal_policy_pscores[0],
        evaluation_policy_pscore_item_position=anti_optimal_policy_pscores[1],
        evaluation_policy_pscore_cascade=anti_optimal_policy_pscores[2],
        alpha=0.05,
        n_bootstrap_samples=1000,
        random_state=dataset_with_random_behavior.random_state,
        evaluation_policy_pscore_idp_window=anti_optimal_policy_pscores[3],
        evaluation_policy_pscore_idp_window_normal=anti_optimal_policy_pscores[4],
    )
    estimated_interval_anti_optimal["policy_name"] = "anti-optimal"

    print(estimated_interval_anti_optimal, '\n')
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
    )

    ground_truth_policy_value_random = dataset_with_random_behavior.calc_ground_truth_policy_value(
        context=bandit_feedback_with_random_behavior["context"],
        evaluation_policy_logit_=random_policy_logit_
    )
    #pdb.set_trace()
    print(ground_truth_policy_value_random)

    ground_truth_policy_value_optimal = dataset_with_random_behavior.calc_ground_truth_policy_value(
        context=bandit_feedback_with_random_behavior["context"],
        evaluation_policy_logit_=optimal_policy_logit_
    )
    ground_truth_policy_value_optimal

    ground_truth_policy_value_anti_optimal = dataset_with_random_behavior.calc_ground_truth_policy_value(
        context=bandit_feedback_with_random_behavior["context"],
        evaluation_policy_logit_=anti_optimal_policy_logit_
    )
    ground_truth_policy_value_anti_optimal

    #pdb.set_trace()
    estimated_interval_random["ground_truth"] = ground_truth_policy_value_random
    estimated_interval_optimal["ground_truth"] = ground_truth_policy_value_optimal
    estimated_interval_anti_optimal["ground_truth"] = ground_truth_policy_value_anti_optimal
    #pdb.set_trace()
    estimated_intervals = pd.concat(
        [
            estimated_interval_random,
            estimated_interval_optimal,
            estimated_interval_anti_optimal
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
    )
    #pdb.set_trace()

    # evaluate the estimation performances of OPE estimators 
    # by comparing the estimated policy values and its ground-truth.
    # `summarize_estimators_comparison` returns a pandas dataframe containing estimation performances of given estimators 

    relative_ee_for_optimal_evaluation_policy = ope.summarize_estimators_comparison(
        ground_truth_policy_value=ground_truth_policy_value_optimal,
        evaluation_policy_pscore=optimal_policy_pscores[0],
        evaluation_policy_pscore_item_position=optimal_policy_pscores[1],
        evaluation_policy_pscore_cascade=optimal_policy_pscores[2],
        evaluation_policy_pscore_idp_window=optimal_policy_pscores[3],
        evaluation_policy_pscore_idp_window_normal=optimal_policy_pscores[4],
    )

    # evaluate the estimation performances of OPE estimators 
    # by comparing the estimated policy values and its ground-truth.
    # `summarize_estimators_comparison` returns a pandas dataframe containing estimation performances of given estimators 

    relative_ee_for_anti_optimal_evaluation_policy = ope.summarize_estimators_comparison(
        ground_truth_policy_value=ground_truth_policy_value_anti_optimal,
        evaluation_policy_pscore=anti_optimal_policy_pscores[0],
        evaluation_policy_pscore_item_position=anti_optimal_policy_pscores[1],
        evaluation_policy_pscore_cascade=anti_optimal_policy_pscores[2],
        evaluation_policy_pscore_idp_window=anti_optimal_policy_pscores[3],
        evaluation_policy_pscore_idp_window_normal=anti_optimal_policy_pscores[4],
    
    )
    print(relative_ee_for_random_evaluation_policy)
    print(relative_ee_for_optimal_evaluation_policy)
    print(relative_ee_for_anti_optimal_evaluation_policy)
    """rand_sips_se+=relative_ee_for_random_evaluation_policy.loc["sips"]["se"]
    rand_iips_se+=relative_ee_for_random_evaluation_policy.loc["iips"]["se"]
    rand_rips_se+=relative_ee_for_random_evaluation_policy.loc["rips"]["se"]
    rand_wips_se+=relative_ee_for_random_evaluation_policy.loc["wips"]["se"]
    
    opt_sips_se+=relative_ee_for_optimal_evaluation_policy.loc["sips"]["se"]
    opt_iips_se+=relative_ee_for_optimal_evaluation_policy.loc["iips"]["se"]
    opt_rips_se+=relative_ee_for_optimal_evaluation_policy.loc["rips"]["se"]
    opt_wips_se+=relative_ee_for_optimal_evaluation_policy.loc["wips"]["se"]

    anti_sips_se+=relative_ee_for_anti_optimal_evaluation_policy.loc["sips"]["se"]
    anti_iips_se+=relative_ee_for_anti_optimal_evaluation_policy.loc["iips"]["se"]
    anti_rips_se+=relative_ee_for_anti_optimal_evaluation_policy.loc["rips"]["se"]
    anti_wips_se+=relative_ee_for_anti_optimal_evaluation_policy.loc["wips"]["se"]"""

"""rand_sips_se/=num_seed
rand_iips_se/=num_seed
rand_rips_se/=num_seed
rand_wips_se/=num_seed

opt_sips_se/=num_seed
opt_iips_se/=num_seed
opt_rips_se/=num_seed
opt_wips_se/=num_seed

anti_sips_se/=num_seed
anti_iips_se/=num_seed
anti_rips_se/=num_seed
anti_wips_se/=num_seed"""

"""results = np.array(
    [
    [
        rand_sips_se,
        rand_iips_se,
        rand_rips_se,
        rand_wips_se
    ],
    [
        opt_sips_se,
        opt_iips_se,
        opt_rips_se,
        opt_wips_se
    ],
    [
        anti_sips_se,
        anti_iips_se,
        anti_rips_se,
        anti_wips_se,
    ]
    ]
)

row_labels = ['rand', 'opt', 'anti']
column_labels = ['sips', 'iips', 'rips', 'wips']
df = pd.DataFrame(data=results, index=row_labels, columns=column_labels)"""

#df.to_excel(f"{reward_structure}_{epsilon}_{len_list}_{n_rounds_test}_total_{num_seed}_error.xlsx")