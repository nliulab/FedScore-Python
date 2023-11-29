import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from confidenceinterval import roc_auc_score as auc_score_ci, accuracy_score, tpr_score, tnr_score, ppv_score, npv_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from AutoScore import utils
import warnings
warnings.filterwarnings('ignore')


def AutoScore_rank(train_set, validation_set=None, method="rf", ntree=100):
    """
    AutoScore STEP (i): Rank variables with machine learning (AutoScore Module 1)
    :param train_set: Training set
    :param validation_set: Validation set
    :param method: Method to calculate variable importance ranking: random forest importance or AUC from LR
    :param ntree: Number of trees for random forest based ranking
    :return: Variable importance ranking
    :raise ValueError ('method' must be either 'rf' or 'auc')
    :raise ValueError ('validation_set' must be specified if method is 'auc')
    """
    X_train, y_train = train_set.drop('label', axis=1), train_set['label']
    variables = list(X_train.columns)
    X_validation, y_validation = validation_set.drop('label', axis=1), validation_set['label']
    if method == 'rf':
        X_train_imputed = utils.impute_data(X_train)
        model = RandomForestClassifier(n_estimators=ntree)
        model.fit(X_train_imputed, y_train)
        importance_df = pd.DataFrame(columns=['var', 'importance'])
        importance_df['var'], importance_df['importance'] = np.array(variables), model.feature_importances_
        importance_df = importance_df.sort_values('importance', ascending=False, ignore_index=True)
        print("The ranking based on variable importance was shown below for each variable: \n")
        print(importance_df.to_string(index=False))
        importance_df_asending = importance_df.sort_values('importance')
        utils.plot_importance(importance_df_asending)
        return importance_df
    elif method == 'auc':
        if validation_set is None or validation_set.empty:
            raise ValueError('ERROR: Please specify the validation set for variable ranking by AUC.')
        X_train_imputed, X_validation_imputed = utils.impute_data(X_train, X_validation)
        AUC = []
        for var in variables:
            if X_train_imputed[var].nunique() > 1:
                model = LogisticRegression()
                model.fit(pd.DataFrame(X_train_imputed[var]), y_train)
                y_validation_pred = model.predict_proba(pd.DataFrame(X_validation_imputed[var]))[:, 1]
                AUC.append(roc_auc_score(y_validation, y_validation_pred))
            else:
                AUC.append(0)
        AUC_df = pd.DataFrame()
        AUC_df['var'], AUC_df['AUC'] = np.array(variables), np.array(AUC)
        AUC_df = AUC_df.sort_values('AUC', ascending=False, ignore_index=True)
        print("The ranking based on variable importance was shown below for each variable: \n")
        print(AUC_df.to_string(index=False))
        AUC_df_ascending = AUC_df.sort_values('AUC')
        utils.plot_importance(AUC_df_ascending)
        return AUC_df
    else:
        raise ValueError("ERROR: Please specify methods among available options: 'rf' or 'auc'.")


def AutoScore_parsimony(train_set, validation_set, rank, max_score=100, n_min=1, n_max=20, cross_validation=False,
                        fold=10, categorize="quantile", quantiles=(0, 0.05, 0.2, 0.8, 0.95, 1), max_cluster=5,
                        do_trace=False, auc_lim_min=0.5, auc_lim_max="adaptive", save_filename='AUC_plot.png'):
    """
    AutoScore STEP (ii): Select the best model with parsimony plot (AutoScore Modules 2+3+4)
    :param train_set: Training set
    :param validation_set: Validation set
    :param rank: Variable importance ranking from AutoScore_rank
    :param max_score: Maximum possible total score for each data instance
    :param n_min: Minimum number of variables to include in the parsimony plot
    :param n_max: Maximum number of variables to include in the parsimony plot
    :param cross_validation: Whether to use cross validation for computing parsimony plot
    :param fold: Number of folds for K-Fold cross validation
    :param categorize: Method to compute cut vector for numerical variables
    :param quantiles: If categorize by quantile, the quantiles to cut numerical variables
    :param max_cluster: If categorize by KMeans clustering, the maximum number of clusters used
    :param do_trace: If using cross validation, whether to plot parsimony plot for each individual fold
    :param auc_lim_min: Minimum y-axis value for parsimony plot
    :param auc_lim_max: Maximum y-axis value for parsimony plot
    :return: Dataframe with all AUC values for parsimony plot
    """
    if n_max > rank.shape[0]:
        print(f"WARNING: the n_max ({n_max}) is larger the number of all variables ({rank.shape[0]}). "
              f"We Automatically revise the n_max to {rank.shape[0]}")
        n_max = rank.shape[0]

    if cross_validation:
        kf = KFold(n_splits=fold)
        AUC_df = pd.DataFrame(columns=['fold', 'num_var', 'AUC'])
        for i, (train_index, validation_index) in enumerate(kf.split(train_set)):
            train_set_temp = train_set.iloc[train_index]
            validation_set_temp = train_set.iloc[validation_index]
            AUC = []
            for j in range(n_min, n_max + 1):
                variable_list = list(rank['var'][:j])
                train_set_1, validation_set_1 = (train_set_temp[variable_list + ['label']],
                                                 validation_set_temp[variable_list + ['label']])
                model_roc = compute_auc_val(train_set_1, validation_set_1, variable_list,
                                            categorize, quantiles, max_cluster, max_score)
                AUC.append(model_roc)
            AUC_df_fold = pd.DataFrame()
            AUC_df_fold['fold'], AUC_df_fold['num_var'], AUC_df_fold['AUC'] = [i] * len(AUC), list(range(n_min, n_max + 1)), AUC
            AUC_df = AUC_df.append(AUC_df_fold)
            if do_trace:
                print(f"List of AUC values for fold {i + 1}")
                print(AUC_df_fold[['num_var', 'AUC']].to_string(index=False))
                plt.bar(AUC_df_fold['num_var'], AUC_df_fold['AUC'])
                plt.title(f"Parsimony plot (cross validation) for fold {i + 1}")
                plt.xlabel("Number of variables")
                plt.ylabel("Area Under Curve")
                plt.show()

        final_AUC = AUC_df.groupby('num_var')['AUC'].mean()
        final_AUC = final_AUC.reset_index(drop=True)
        print("***List of final mean AUC values through cross-validation are shown below \n")
        for n in range(final_AUC.shape[0]):
            print("Select", final_AUC.index[n] + n_min, "Variable(s): Area Under the Curve", final_AUC[n])
        utils.plot_AUC(final_AUC, rank['var'][(n_min-1):n_max], auc_lim_min, auc_lim_max, ylab="Area Under the Curve",
                       title=f"Final Parsimony plot based on {fold}-fold Cross Validation", save_filename=save_filename)
        return AUC_df
    else:
        AUC = []
        for i in range(n_min, n_max + 1):
            variable_list = list(rank['var'][:i])
            train_set_1, validation_set_1 = train_set[variable_list + ['label']], validation_set[variable_list + ['label']]
            model_roc = compute_auc_val(train_set_1, validation_set_1, variable_list, categorize, quantiles, max_cluster, max_score)
            AUC.append(model_roc)
            print("Select", i, "Variable(s): Area Under the Curve", model_roc)
        utils.plot_AUC(AUC, rank['var'][(n_min-1):n_max], auc_lim_min, auc_lim_max, ylab="Area Under the Curve",
                       title="Parsimony plot on validation set", save_filename=save_filename)
        AUC_df = pd.DataFrame()
        AUC_df['num_variable'], AUC_df['AUC'] = list(range(n_min, n_max + 1)), AUC
        return AUC_df


def AutoScore_weighting(train_set, validation_set, final_variables, max_score=100, categorize="quantile",
                        max_cluster=5, quantiles=(0, 0.05, 0.2, 0.8, 0.95, 1)):
    """
    AutoScore STEP (iii): Generate the initial score with the final list of variables (Re-run AutoScore Modules 2+3)
    :param train_set: Training set
    :param validation_set: Validation set
    :param final_variables: Final list of variables to include in the model
    :param max_score: Maximum possible total score for each data instance
    :param categorize: Method to compute cut vector for numerical variables
    :param max_cluster: If categorize by KMeans clustering, the maximum number of clusters used
    :param quantiles: If categorize by quantile, the quantiles to cut numerical variables
    :return: Cut vector for included numerical variables
    """
    print("****Included Variables: \n")
    print(pd.DataFrame(data=list(final_variables), columns=['variable_name'], index=range(1, len(final_variables) + 1)).to_string())
    final_variables_with_categories = utils.get_final_variables_with_categories(train_set, final_variables)
    train_set_1, validation_set_1 = train_set[final_variables_with_categories], validation_set[final_variables_with_categories]
    train_set_1['label'], validation_set['label'] = train_set['label'], validation_set['label']

    # AutoScore Module 2 : cut numeric and transfer categories and generate "cut_vec"
    cut_vec = utils.get_cut_vec(train_set_1, quantiles, max_cluster, categorize)
    train_set_2, validation_set_2 = (utils.transform_df_fixed(train_set_1, cut_vec),
                                     utils.transform_df_fixed(validation_set_1, cut_vec))

    # AutoScore Module 3 : Score weighting
    score_table = compute_score_table(train_set_2, max_score, final_variables_with_categories)
    print("****Initial Scores: \n")
    utils.print_scoring_table(score_table, final_variables_with_categories)
    validation_set_2['label'] = validation_set['label']

    # Using "assign_score" to generate score based on new dataset and Scoring table "score_table"
    validation_set_3 = utils.assign_score(validation_set_2, score_table)
    X_validation_3, y_validation_3 = validation_set_3.drop('label', axis=1), validation_set_3['label']
    total_score = np.sum(X_validation_3, axis=1)
    plot_roc_curve(total_score, y_validation_3)
    print("***Performance (based on validation set):\n")
    print_roc_performance(total_score, y_validation_3)
    print("***The cutoffs of each variable generated by AutoScore are saved in cut_vec. "
          "You can decide whether to revise or fine-tune them.\n")
    return cut_vec


def AutoScore_fine_tuning(train_set, validation_set, final_variables, cut_vec, max_score=100):
    """
    AutoScore STEP (iv): Fine-tune the score by revising cut_vec with domain knowledge (AutoScore Module 5)
    :param train_set: Training set
    :param validation_set: Validation set
    :param final_variables: Final list of variables to include in the model
    :param cut_vec: Final cut vector after fine-tuning for included numerical variables
    :param max_score: Maximum possible total score for each data instance
    :return: Score table for final variables after fine-tuning cut vectors
    """
    final_variables_with_categories = utils.get_final_variables_with_categories(train_set, final_variables)
    train_set_1, validation_set_1 = train_set[final_variables_with_categories], validation_set[final_variables_with_categories]
    train_set_1['label'], validation_set_1['label'] = train_set['label'], validation_set['label']
    train_set_2, validation_set_2 = (utils.transform_df_fixed(train_set_1, cut_vec),
                                     utils.transform_df_fixed(validation_set_1, cut_vec))
    score_table = compute_score_table(train_set_2, max_score, final_variables_with_categories)
    print("***Fine-tuned Scores: \n")
    utils.print_scoring_table(score_table, final_variables_with_categories)
    validation_set_2['label'] = validation_set['label']
    validation_set_3 = utils.assign_score(validation_set_2, score_table)
    X_validation_3, y_validation_3 = validation_set_3.drop('label', axis=1), validation_set_3['label']
    total_score = np.sum(X_validation_3, axis=1)
    plot_roc_curve(total_score, y_validation_3)
    print("***Performance (based on validation set, after fine-tuning):")
    print_roc_performance(total_score, y_validation_3, threshold='best')
    return score_table


def AutoScore_testing(test_set, final_variables, cut_vec, scoring_table, threshold="best", with_label=True):
    """
    AutoScore STEP (v): Evaluate the final score with ROC analysis (AutoScore Module 6)
    :param test_set: Test set
    :param final_variables: Final list of variables to include in the model
    :param cut_vec: Final cut vector after fine-tuning for included numerical variables
    :param scoring_table: Final score table for score assignment
    :param threshold: Class (0/1) threshold for binary classification
    :param with_label: Whether the data includes a binary label
    :return: Predicted total scores and ground truth labels for test set
    """
    if with_label:
        final_variables_with_categories = utils.get_final_variables_with_categories(test_set, final_variables)
        test_set_1 = test_set[final_variables_with_categories]
        test_set_1['label'] = test_set['label']
        test_set_2 = utils.transform_df_fixed(test_set_1, cut_vec)
        test_set_3 = utils.assign_score(test_set_2, scoring_table).fillna(0)
        X_test, y_test = test_set_3.drop('label', axis=1), test_set_3['label']
        total_score = np.sum(X_test, axis=1)
        plot_roc_curve(total_score, y_test, setting='Test')
        print("***Performance using AutoScore:")
        print_roc_performance(total_score, y_test, threshold)
        pred_score = pd.DataFrame(columns=['pred_score', 'label'])
        pred_score['pred_score'], pred_score['label'] = total_score, y_test
        return pred_score
    else:
        final_variables_with_categories = utils.get_final_variables_with_categories(test_set, final_variables)
        test_set_1 = test_set[final_variables_with_categories]
        test_set_2 = utils.transform_df_fixed(test_set_1, cut_vec)
        test_set_3 = utils.assign_score(test_set_2, scoring_table).fillna(0)
        total_score = np.sum(test_set_3, axis=1)
        pred_score = pd.DataFrame(columns=['pred_score', 'label'])
        pred_score['pred_score'], pred_score['label'] = total_score, None
        return pred_score


# Internal functions

def compute_score_table(train_set, max_score, variable_list):
    """
    Compute score table to convert each feature value of to an integer score
    :param train_set: Training set for computing score table
    :param max_score: Maximum possible total score for each data instance
    :param variable_list: Final list of variables to include in the model
    :return: Score table for variables in the variable_list
    """
    # First step logistic regression
    X_train, y_train = train_set.drop('label', axis=1), train_set['label']
    design_matrix, initial_ref_cols = utils.generate_design_matrix(X_train)
    model = LogisticRegression(penalty=None, max_iter=1000, solver='newton-cg')
    model.fit(design_matrix, y_train)
    coef = pd.Series(data=model.coef_[0], index=design_matrix.columns)

    # Second step logistic regression
    train_set_2, unchanged_vars = utils.change_reference(design_matrix, coef)
    for v in initial_ref_cols.columns:
        for x in unchanged_vars:
            if x == '_'.join(v.split('_')[:-1]):
                initial_ref_cols = initial_ref_cols.drop(v, axis=1)
    train_set_2 = pd.concat([train_set_2, initial_ref_cols], axis=1)
    X_train_2, y_train_2 = train_set_2, train_set['label']
    model2 = LogisticRegression(penalty=None, max_iter=1000, solver='newton-cg')
    model2.fit(X_train_2, y_train_2)
    coef2 = pd.Series(data=model2.coef_[0], index=X_train_2.columns)

    # Rounding for final score table
    coef_vec_tmp = pd.Series(data=[x / min(coef2) for x in coef2], index=X_train_2.columns)
    score_table = utils.add_baseline(X_train, coef_vec_tmp)

    # Normalization according to "max_score" and regenerate score_table
    total_max, total = max_score, 0
    for var in variable_list:
        indicator_vars = [x for x in score_table.index if var == '_'.join(x.split('_')[:-1])]
        total += max([score_table[x] for x in indicator_vars])
    score_table = score_table / (total / total_max)
    score_table = score_table.round(0).astype(int)
    return score_table

def compute_auc_val(train_set, validation_set, variable_list, categorize, quantiles, max_cluster, max_score):
    """
    Compute AUC value on validation set for including variables in the variable_list
    :param train_set: Training set (Generate score table)
    :param validation_set: Validation set (Compute the AUC value)
    :param variable_list: List of variables to include in the model
    :param categorize: Method to compute cut vector for numerical variables
    :param quantiles: If categorize by quantile, the quantiles to cut numerical variables
    :param max_cluster: If categorize by KMeans clustering, the maximum number of clusters used
    :param max_score: Maximum possible total score for each data instance
    :return: AUC value based on validation set
    """
    cut_vec = utils.get_cut_vec(train_set, quantiles, max_cluster, categorize)
    train_set_1, validation_set_1 = (utils.transform_df_fixed(train_set, cut_vec),
                                     utils.transform_df_fixed(validation_set, cut_vec))
    if train_set_1.isnull().values.any():
        print("WARNING: NA in training set 1: ", train_set_1.isnull().sum().sum())
    if validation_set_1.isnull().values.any():
        print("WARNING: NA in validation set 1: ", validation_set_1.isnull().sum().sum())

    score_table = compute_score_table(train_set_1, max_score, variable_list)
    if score_table.isnull().values.any():
        print("WARNING: NA in the score table: ", score_table.isnull().sum().sum())

    validation_set_2 = utils.assign_score(validation_set_1, score_table)
    if validation_set_2.isnull().values.any():
        print("WARNING: NA in validation set 2: ", validation_set_2.isnull().sum().sum())
    X_validation_2, y_validation_2 = validation_set_2.drop('label', axis=1), validation_set_2['label']
    total_scores = np.sum(X_validation_2, axis=1)
    auc = roc_auc_score(y_validation_2, total_scores)
    return auc


def plot_roc_curve(score, labels, setting='Validation'):
    """
    Plot ROC curve based on total scores and ground truth labels
    :param score: Total scores for each data sample
    :param labels: Ground truth labels for each data sample
    :param setting: Whether the ROC curve is based on validation set or test set
    :return: None (plot the ROC curve)
    """
    fpr, tpr, _ = roc_curve(labels, score)
    auc = roc_auc_score(labels, score)
    plt.plot(fpr, tpr, label="AUC = " + str(auc))
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.plot([0, 1], [0, 1], linestyle='dashed')
    plt.title('ROC Curve for ' + setting)
    plt.xlabel('1 - Specificity (FPR)')
    plt.ylabel('Sensitivity (TPR)')
    plt.legend(loc=4)
    plt.show()


def print_roc_performance(score, labels, threshold='best', print_metrics=True):
    """
    Print binary classification metrics
    :param score: Total score for each data sample
    :param labels: Ground truth labels for each data sample
    :param threshold: Class (0/1) threshold for binary classification
    :param print_metrics: Whether to print evaluation metrics
    :return: Specificity, Sensitivity, NPV, PPV, with their 95% confidence intervals
    """
    zero_one_labels = [int(x) for x in labels]
    fpr, tpr, thresholds = roc_curve(labels, score)
    auc, auc_ci = auc_score_ci(zero_one_labels, score, confidence_level=0.95)
    auprc = average_precision_score(labels, score)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    if threshold == 'best':
        pred = score >= int(optimal_threshold)
    else:
        pred = score >= int(threshold)
    sensitivity, sensitivity_ci = tpr_score(zero_one_labels, pred, confidence_level=0.95)
    specificity, specificity_ci = tnr_score(zero_one_labels, pred, confidence_level=0.95)
    PPV, PPV_ci = ppv_score(zero_one_labels, pred, confidence_level=0.95)
    NPV, NPV_ci = npv_score(zero_one_labels, pred, confidence_level=0.95)

    auc, auc_ci = round(auc, 4), [round(x, 4) for x in auc_ci]
    sensitivity, sensitivity_ci = round(sensitivity, 4), [round(x, 4) for x in sensitivity_ci]
    specificity, specificity_ci = round(specificity, 4), [round(x, 4) for x in specificity_ci]
    PPV, PPV_ci = round(PPV, 4), [round(x, 4) for x in PPV_ci]
    NPV, NPV_ci = round(NPV, 4), [round(x, 4) for x in NPV_ci]

    if print_metrics:
        print('AUROC:', auc, '95% CI:', auc_ci)
        print('AUPRC:', round(auprc, 4))
        print('Best score threshold: >=', int(optimal_threshold))
        if threshold == 'best':
            print('Your score threshold: >=', int(optimal_threshold))
        else:
            print('Your score threshold: >=', int(threshold))
        print('Other performance indicators based on this score threshold:')
        print('Sensitivity:', sensitivity, '95% CI:', sensitivity_ci)
        print('Specificity:', specificity, '95% CI:', specificity_ci)
        print('PPV:', PPV, '95% CI:', PPV_ci)
        print('NPV:', NPV, '95% CI:', NPV_ci)
    return specificity, sensitivity, PPV, NPV, specificity_ci, sensitivity_ci, PPV_ci, NPV_ci

def conversion_table(pred_score, by='risk', values=(0.01, 0.05, 0.1, 0.2, 0.5)):
    """
    Convert from predicted total scores to risk, or vice versa
    :param pred_score: Predicted total scores from test set
    :param by: Whether to convert from risk to scores, or vice versa
    :param values: The values of risk or score to include in the table
    :return: None (print the conversion table)
    :raise ValueError ('by' must be either 'risk' or 'score')
    """
    score_model = LogisticRegression(penalty=None)
    score_model.fit(X=np.array(pred_score['pred_score']).reshape(-1, 1), y=pred_score['label'])
    pred_score['pred_risk'] = score_model.predict_proba(np.array(pred_score['pred_score']).reshape(-1, 1))[:, 1]

    if by == 'risk':
        result_dict = {"Predicted Risk [>=]": [], "Score cut-off [>=]": [], "Percentage of patients (%)": [],
                       "Accuracy": [], "Sensitivity": [], "Specificity": [], "PPV": [], "NPV": []}
        for val in values:
            try:
                score_cutoff = min(pred_score.loc[pred_score['pred_risk'] > val, 'pred_score'])
            except ValueError:
                print(f'WARNING: No samples in the test set have predicted risk >= {int(val * 100)}%. This value is ignored.')
                continue
            result_dict['Predicted Risk [>=]'].append(f'{int(val * 100)}%')
            result_dict['Score cut-off [>=]'].append(score_cutoff)
            result_dict['Percentage of patients (%)'].append(
                round(pred_score.loc[pred_score['pred_risk'] > val].shape[0] / pred_score.shape[0] * 100, 1))
            accuracy, accuracy_ci = accuracy_score(pred_score['label'], pred_score['pred_risk'] > val)
            accuracy, accuracy_ci = round(accuracy, 4), [round(x, 4) for x in accuracy_ci]
            accuracy_ci = [str(round(100 * x, 1)) + '%' for x in accuracy_ci]
            result_dict['Accuracy'].append(f'{round(accuracy * 100, 1)}% ({accuracy_ci[0]}-{accuracy_ci[1]})')
            specificity, sensitivity, PPV, NPV, specificity_ci, sensitivity_ci, PPV_ci, NPV_ci = (
                print_roc_performance(pred_score['pred_score'], pred_score['label'], threshold=score_cutoff, print_metrics=False))
            specificity_ci = [str(round(100 * x, 1)) + '%' for x in specificity_ci]
            sensitivity_ci = [str(round(100 * x, 1)) + '%' for x in sensitivity_ci]
            PPV_ci = [str(round(100 * x, 1)) + '%' for x in PPV_ci]
            NPV_ci = [str(round(100 * x, 1)) + '%' for x in NPV_ci]
            result_dict['Sensitivity'].append(f'{round(sensitivity * 100, 1)}% ({sensitivity_ci[0]}-{sensitivity_ci[1]})')
            result_dict['Specificity'].append(f'{round(specificity * 100, 1)}% ({specificity_ci[0]}-{specificity_ci[1]})')
            result_dict['PPV'].append(f'{round(PPV * 100, 1)}% ({PPV_ci[0]}-{PPV_ci[1]})')
            result_dict['NPV'].append(f'{round(NPV * 100, 1)}% ({NPV_ci[0]}-{NPV_ci[1]})')
    elif by == 'score':
        result_dict = {"Score cut-off [>=]": [], "Predicted Risk [>=]": [], "Percentage of patients (%)": [],
                       "Accuracy": [], "Sensitivity": [], "Specificity": [], "PPV": [], "NPV": []}
        for val in values:
            try:
                predicted_risk = min(pred_score.loc[pred_score['pred_score'] > val, 'pred_risk'])
            except ValueError:
                print(f'WARNING: No samples in the test set have predicted score >= {int(val)}. This value is ignored.')
                continue
            result_dict['Score cut-off [>=]'].append(str(val))
            result_dict['Predicted Risk [>=]'].append(f'{round(predicted_risk * 100, 1)}%')
            result_dict['Percentage of patients (%)'].append(
                round(pred_score.loc[pred_score['pred_score'] > val].shape[0] / pred_score.shape[0] * 100, 1))
            accuracy, accuracy_ci = accuracy_score(pred_score['label'], pred_score['pred_score'] > val)
            accuracy, accuracy_ci = round(accuracy, 4), [round(x, 4) for x in accuracy_ci]
            accuracy_ci = [str(round(100 * x, 1)) + '%' for x in accuracy_ci]
            result_dict['Accuracy'].append(f'{round(accuracy * 100, 1)}% ({accuracy_ci[0]}-{accuracy_ci[1]})')
            specificity, sensitivity, PPV, NPV, specificity_ci, sensitivity_ci, PPV_ci, NPV_ci = (
                print_roc_performance(pred_score['pred_score'], pred_score['label'], threshold=val, print_metrics=False))
            specificity_ci = [str(round(100 * x, 1)) + '%' for x in specificity_ci]
            sensitivity_ci = [str(round(100 * x, 1)) + '%' for x in sensitivity_ci]
            PPV_ci = [str(round(100 * x, 1)) + '%' for x in PPV_ci]
            NPV_ci = [str(round(100 * x, 1)) + '%' for x in NPV_ci]
            result_dict['Sensitivity'].append(f'{round(sensitivity * 100, 1)}% ({sensitivity_ci[0]}-{sensitivity_ci[1]})')
            result_dict['Specificity'].append(f'{round(specificity * 100, 1)}% ({specificity_ci[0]}-{specificity_ci[1]})')
            result_dict['PPV'].append(f'{round(PPV * 100, 1)}% ({PPV_ci[0]}-{PPV_ci[1]})')
            result_dict['NPV'].append(f'{round(NPV * 100, 1)}% ({NPV_ci[0]}-{NPV_ci[1]})')
    else:
        raise ValueError('ERROR: Please specify correct method for categorizing threshold: by "risk" or "score".')
    result_df = pd.DataFrame.from_dict(result_dict)
    print(result_df.to_string(index=False))

def compute_uni_variable_table(df):
    """
    Compute uni-variable odds ratios (with 95% CI) and p values for all variables
    :param df: Dataframe for odds ratio computation
    :return: None (print the uni-variable odds ratio table)
    """
    df_copy = df.copy()
    df_copy['label'] = [int(x) for x in df_copy['label']]
    result_dict = {'Variable': [], 'OR (95% CI)': [], 'p value': []}
    for var in df_copy.columns:
        if var == 'label':
            continue
        result_dict['Variable'].append(var)
        model = sm.Logit(df_copy['label'], sm.add_constant(df_copy[[var]]), missing='drop').fit(disp=0)
        curr_result = utils.model_summary_to_dataframe(model)
        coef, low_ci, high_ci = (round(np.exp(curr_result['coef'][0]), 3),
                                 round(np.exp(curr_result['conf_lower'][0]), 3),
                                 round(np.exp(curr_result['conf_higher'][0]), 3))
        result_dict['OR (95% CI)'].append(f'{coef} ({low_ci}-{high_ci})')
        p_val = round(curr_result['pvals'][0], 3)
        p_val = str(p_val) if p_val >= 0.001 else '<0.001'
        result_dict['p value'].append(p_val)
    result_df = pd.DataFrame.from_dict(result_dict)
    print(result_df.to_string(index=False))

def compute_multi_variable_table(df):
    """
    Compute multi-variable odds ratios (with 95% CI) and p values for all variables
    :param df: Dataframe for odds ratio computation
    :return: None (print the multi-variable odds ratio table)
    """
    df_copy = df.copy().dropna()
    df_copy['label'] = [int(x) for x in df_copy['label']]
    X, y = sm.add_constant(df_copy.drop('label', axis=1)), df_copy['label']
    result_dict = {'Variable': [], 'OR (95% CI)': [], 'p value': []}
    model = sm.Logit(y, X).fit(disp=0)
    model_result = utils.model_summary_to_dataframe(model)
    for i in range(len(model_result.index)):
        result_dict['Variable'].append(model_result.index[i])
        coef, low_ci, high_ci = (round(np.exp(model_result['coef'][i]), 3),
                                 round(np.exp(model_result['conf_lower'][i]), 3),
                                 round(np.exp(model_result['conf_higher'][i]), 3))
        result_dict['OR (95% CI)'].append(f'{coef} ({low_ci}-{high_ci})')
        p_val = round(model_result['pvals'][i], 3)
        p_val = str(p_val) if p_val >= 0.001 else '<0.001'
        result_dict['p value'].append(p_val)
    result_df = pd.DataFrame.from_dict(result_dict)
    print(result_df.to_string(index=False))
