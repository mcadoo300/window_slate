import numpy as np
import pandas as pd
# import open bandit pipeline (obp)
import obp
import pdb
from obp.ope import SlateStandardIPS, SlateIndependentIPS, SlateRewardInteractionIPS, SlateOffPolicyEvaluation
from obp.dataset import (
    logistic_reward_function,
    SyntheticSlateBanditDataset,
)

from itertools import product
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

# obp version
print(obp.__version__)

import warnings
warnings.filterwarnings('ignore')

# generate a synthetic bandit dataset with 10 actions
# we use `logistic_reward_function` as the reward function and `linear_behavior_policy_logit` as the behavior policy.
# one can define their own reward function and behavior policy such as nonlinear ones. 

n_unique_action=10
len_list = 3
dim_context = 2
reward_type = "binary"
reward_structure="cascade_additive"
click_model=None
random_state=12345
base_reward_function=logistic_reward_function

# obtain  test sets of synthetic logged bandit data
n_rounds_test = 10000



# define Uniform Random Policy as a baseline behavior policy
dataset_with_random_behavior = SyntheticSlateBanditDataset(
    n_unique_action=n_unique_action,
    len_list=len_list,
    dim_context=dim_context,
    reward_type=reward_type,
    reward_structure=reward_structure,
    click_model=click_model,
    random_state=random_state,
    behavior_policy_function=None,  # set to uniform random
    base_reward_function=base_reward_function,
)

# compute the factual action choice probabililties for the test set of the synthetic logged bandit data
bandit_feedback_with_random_behavior = dataset_with_random_behavior.obtain_batch_bandit_feedback(
    n_rounds=n_rounds_test,
    return_pscore_item_position=True,
)

# print policy value
random_policy_value = dataset_with_random_behavior.calc_on_policy_policy_value(
    reward=bandit_feedback_with_random_behavior["reward"],
    slate_id=bandit_feedback_with_random_behavior["slate_id"],
)
print(random_policy_value)

random_policy_logit_ = np.zeros((n_rounds_test, n_unique_action))

base_expected_reward = dataset_with_random_behavior.base_reward_function(
    context=bandit_feedback_with_random_behavior["context"],
    action_context=dataset_with_random_behavior.action_context,
    random_state=dataset_with_random_behavior.random_state,
)

optimal_policy_logit_ = base_expected_reward * 3
anti_optimal_policy_logit_ = -3 * base_expected_reward

random_policy_pscores = dataset_with_random_behavior.obtain_pscore_given_evaluation_policy_logit(
    action=bandit_feedback_with_random_behavior["action"],
    evaluation_policy_logit_=random_policy_logit_
)

optimal_policy_pscores = dataset_with_random_behavior.obtain_pscore_given_evaluation_policy_logit(
    action=bandit_feedback_with_random_behavior["action"],
    evaluation_policy_logit_=optimal_policy_logit_
)

anti_optimal_policy_pscores = dataset_with_random_behavior.obtain_pscore_given_evaluation_policy_logit(
    action=bandit_feedback_with_random_behavior["action"],
    evaluation_policy_logit_=anti_optimal_policy_logit_
)

# estimate the policy value of the evaluation policies based on their action choice probabilities
# it is possible to set multiple OPE estimators to the `ope_estimators` argument

sips = SlateStandardIPS(len_list=len_list)
iips = SlateIndependentIPS(len_list=len_list)
rips = SlateRewardInteractionIPS(len_list=len_list)

ope = SlateOffPolicyEvaluation(
    bandit_feedback=bandit_feedback_with_random_behavior,
    ope_estimators=[sips, iips, rips]
)

#pdb.set_trace()
_, estimated_interval_random = ope.summarize_off_policy_estimates(
    evaluation_policy_pscore=random_policy_pscores[0],
    evaluation_policy_pscore_item_position=random_policy_pscores[1],
    evaluation_policy_pscore_cascade=random_policy_pscores[2],
    alpha=0.05,
    n_bootstrap_samples=1000,
    random_state=dataset_with_random_behavior.random_state
)
estimated_interval_random["policy_name"] = "random"

print(estimated_interval_random, '\n')
# visualize estimated policy values of Uniform Random by the three OPE estimators
# and their 95% confidence intervals (estimated by nonparametric bootstrap method)
ope.visualize_off_policy_estimates(
    evaluation_policy_pscore=random_policy_pscores[0],
    evaluation_policy_pscore_item_position=random_policy_pscores[1],
    evaluation_policy_pscore_cascade=random_policy_pscores[2],
    alpha=0.05,
    n_bootstrap_samples=1000, # number of resampling performed in bootstrap sampling
    random_state=dataset_with_random_behavior.random_state
)
#pdb.set_trace()

_, estimated_interval_optimal = ope.summarize_off_policy_estimates(
    evaluation_policy_pscore=optimal_policy_pscores[0],
    evaluation_policy_pscore_item_position=optimal_policy_pscores[1],
    evaluation_policy_pscore_cascade=optimal_policy_pscores[2],
    alpha=0.05,
    n_bootstrap_samples=1000,
    random_state=dataset_with_random_behavior.random_state
)

estimated_interval_optimal["policy_name"] = "optimal"

print(estimated_interval_optimal, '\n')
# visualize estimated policy values of Optimal by the three OPE estimators
# and their 95% confidence intervals (estimated by nonparametric bootstrap method)
ope.visualize_off_policy_estimates(
    evaluation_policy_pscore=optimal_policy_pscores[0],
    evaluation_policy_pscore_item_position=optimal_policy_pscores[1],
    evaluation_policy_pscore_cascade=optimal_policy_pscores[2],
    alpha=0.05,
    n_bootstrap_samples=1000, # number of resampling performed in bootstrap sampling
    random_state=dataset_with_random_behavior.random_state
)

_, estimated_interval_anti_optimal = ope.summarize_off_policy_estimates(
    evaluation_policy_pscore=anti_optimal_policy_pscores[0],
    evaluation_policy_pscore_item_position=anti_optimal_policy_pscores[1],
    evaluation_policy_pscore_cascade=anti_optimal_policy_pscores[2],
    alpha=0.05,
    n_bootstrap_samples=1000,
    random_state=dataset_with_random_behavior.random_state
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
    random_state=dataset_with_random_behavior.random_state
)

ground_truth_policy_value_random = dataset_with_random_behavior.calc_ground_truth_policy_value(
    context=bandit_feedback_with_random_behavior["context"],
    evaluation_policy_logit_=random_policy_logit_
)
ground_truth_policy_value_random

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
estimated_intervals

# evaluate the estimation performances of OPE estimators 
# by comparing the estimated policy values and its ground-truth.
# `summarize_estimators_comparison` returns a pandas dataframe containing estimation performances of given estimators 
#pdb.set_trace()
relative_ee_for_random_evaluation_policy = ope.summarize_estimators_comparison(
    ground_truth_policy_value=ground_truth_policy_value_random,
    evaluation_policy_pscore=random_policy_pscores[0],
    evaluation_policy_pscore_item_position=random_policy_pscores[1],
    evaluation_policy_pscore_cascade=random_policy_pscores[2]
)
#pdb.set_trace()
print(relative_ee_for_random_evaluation_policy)

# evaluate the estimation performances of OPE estimators 
# by comparing the estimated policy values and its ground-truth.
# `summarize_estimators_comparison` returns a pandas dataframe containing estimation performances of given estimators 

relative_ee_for_optimal_evaluation_policy = ope.summarize_estimators_comparison(
    ground_truth_policy_value=ground_truth_policy_value_optimal,
    evaluation_policy_pscore=optimal_policy_pscores[0],
    evaluation_policy_pscore_item_position=optimal_policy_pscores[1],
    evaluation_policy_pscore_cascade=optimal_policy_pscores[2]
)
print(relative_ee_for_optimal_evaluation_policy)

# evaluate the estimation performances of OPE estimators 
# by comparing the estimated policy values and its ground-truth.
# `summarize_estimators_comparison` returns a pandas dataframe containing estimation performances of given estimators 

relative_ee_for_anti_optimal_evaluation_policy = ope.summarize_estimators_comparison(
    ground_truth_policy_value=ground_truth_policy_value_anti_optimal,
    evaluation_policy_pscore=anti_optimal_policy_pscores[0],
    evaluation_policy_pscore_item_position=anti_optimal_policy_pscores[1],
    evaluation_policy_pscore_cascade=anti_optimal_policy_pscores[2]
)
print(relative_ee_for_anti_optimal_evaluation_policy)

#pdb.set_trace()
estimated_intervals["errbar_length"] = (
    estimated_intervals.drop(["mean", "policy_name", "ground_truth"], axis=1).diff(axis=1).iloc[:, -1].abs()
)
alpha = 0.05
plt.style.use("ggplot")
def errplot(x, y, yerr, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    data.plot(x=x, y=y, yerr=yerr, kind="bar", ax=ax, **kwargs)
    ax.hlines(data["ground_truth"].iloc[0], -1, len(x)+1)
    
g = sns.FacetGrid(
    estimated_intervals.reset_index().rename(columns={"index": "OPE estimator", "mean": "Policy value"}),
    col="policy_name"
)
g.map_dataframe(errplot, "OPE estimator", "Policy value", "errbar_length")
plt.show()