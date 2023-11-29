import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from collections import Counter
from tableone import TableOne
import warnings
warnings.filterwarnings('ignore')


def load_sample_data(dataset):
    """
    Load sample dataset from data directory
    :param dataset: The sample dataset to be loaded (binary, small, missing)
    :return: The selected sample dataset as a Dataframe
    :raise ValueError ('dataset' must be 'binary', 'small', or 'missing')
    """
    if dataset == 'binary':
        data = pd.read_csv('../data/sample_data.csv', index_col=0)
    elif dataset == 'small':
        data = pd.read_csv('../data/sample_data_small.csv', index_col=0)
    elif dataset == 'missing':
        data = pd.read_csv('../data/sample_data_with_missing.csv', index_col=0)
    else:
        raise ValueError("ERROR: Please choose a sample dataset from 'binary', 'small', or 'missing'.")
    return data


def plot_importance(df):
    """
    Plot variable importance
    :param df: Variable ranking dataframe (from AutoScore_rank)
    :return: None (plot variable importance)
    """
    values = df['importance'] if 'importance' in df.columns else df['AUC']
    plt.barh(df['var'], values)
    plt.xlabel('Importance')
    plt.title('Importance Ranking')
    plt.show()


def plot_AUC(auc_values, variables, auc_lim_min, auc_lim_max,
             ylab="Mean Area Under the Curve", title="Parsimony plot on the validation set",
             save_filename='AUC_plot.png'):
    """
    Create parsimony plot for AutoScore_parsimony
    :param auc_values: List of AUC values for different number of variables
    :param variables: List of variables to be included in the parsimony plot
    :param auc_lim_min: Minimum y-axis value for parsimony plot
    :param auc_lim_max: Maximum y-axis value for parsimony plot
    :param ylab: Label for y-axis (e.g. Area Under the Curve)
    :param title: Title for parsimony plot (e.g. Parsimony plot on the validation set)
    :return: None (create parsimony plot for AutoScore_parsimony)
    """
    variables = variables.reset_index(drop=True)
    if auc_lim_max == 'adaptive':
        auc_lim_max = 0.1 * math.ceil(max(auc_values) / 0.1)
    plt.bar(list(variables), list(auc_values))
    plt.xticks(rotation=90)
    plt.ylim(auc_lim_min, auc_lim_max)
    plt.title(title)
    plt.ylabel(ylab)
    plt.savefig(save_filename)
    plt.show()


def get_cut_vec(df, quantiles=(0, 0.05, 0.2, 0.8, 0.95, 1), max_cluster=5, categorize="quantile"):
    """
    Compute cut vector for numerical variables
    :param df: Dataframe for computing cut vector
    :param quantiles: If categorize by quantile, the quantiles to cut numerical variables
    :param max_cluster: If categorize by KMeans clustering, the maximum number of clusters used
    :param categorize: Method to compute cut vector for numerical variables
    :return: Dict of cut vectors for all numerical variables in df
    :raise ValueError ('categorize' must be 'quantile' or 'kmeans')
    """
    all_cut_vec = {}
    if categorize == 'quantile':
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 2:
                curr_cutoff = np.unique(np.nanquantile(df[col], quantiles))
                all_cut_vec[col] = curr_cutoff
    elif categorize == 'kmeans':
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 2:
                col_label = df[[col, 'label']].copy().dropna()
                kmeans = KMeans(n_clusters=max_cluster)
                labels = kmeans.fit_predict(col_label[col].values.reshape(-1, 1))
                temp_df = pd.DataFrame(columns=['data', 'label'])
                temp_df['data'], temp_df['label'] = col_label[col], labels
                curr_cutoff = list(temp_df.groupby('label')['data'].min())
                curr_cutoff.append(max(df[col]))
                curr_cutoff = np.unique(sorted(curr_cutoff))
                all_cut_vec[col] = curr_cutoff
    else:
        raise ValueError("ERROR: Please specify correct method for categorizing numerical variables: 'quantile' or 'kmeans'.")
    for var in all_cut_vec:
        if len(all_cut_vec[var]) > 2:
            all_cut_vec[var] = all_cut_vec[var][1:-1]
            if (df[var] % 1 == 0).all():
                all_cut_vec[var] = all_cut_vec[var].astype(int)
            elif is_numeric_dtype(df[var]):
                all_cut_vec[var] = all_cut_vec[var].round(1)
        else:
            all_cut_vec[var] = 'binary'
    return all_cut_vec


def transform_df_fixed(df, cut_vec):
    """
    Transform the numerical variables to categorical variables based on cut vector
    :param df: Dataframe to be transformed
    :param cut_vec: Cut vectors of numerical variables
    :return: Transformed dataframe
    """
    for col in cut_vec:
        if cut_vec[col] == 'binary' or len(cut_vec[col]) == 0:
            cnt = Counter(df.loc[df[col].notnull(), col])
            mode = max(cnt, key=cnt.get)
            df[col] = [str(mode) if x == mode else 'not_' + str(mode) for x in df[col]]
        else:
            cut_vec_tmp = cut_vec[col].copy()
            if np.nanmin(df[col]) <= cut_vec_tmp[0]:
                cut_vec_tmp = np.insert(cut_vec_tmp, 0, np.nanmin(df[col]) - 100)
            if np.nanmax(df[col]) >= cut_vec_tmp[-1]:
                cut_vec_tmp = np.append(cut_vec_tmp, np.nanmax(df[col]) + 100)
            df[col] = pd.cut(df[col], cut_vec_tmp, include_lowest=False, right=False, duplicates='drop', precision=3)
            df[col] = [x if x == x else 'Unknown' for x in df[col]]
    return df

def generate_design_matrix(df):
    """
    Generate the one-hot encoded design matrix with only binary columns
    :param df: Original dataframe
    :return: one-hot encoded design_matrix with only binary columns and dropped columns during one-hot encoding
    """
    initial_ref_cols = pd.DataFrame()
    design_matrix = pd.DataFrame()
    for col in df.columns:
        one_hot = pd.get_dummies(df[col])
        one_hot.columns = [col + '_' + str(x) for x in one_hot.columns]
        initial_ref_cols = pd.concat([initial_ref_cols, one_hot.iloc[:, 0]], axis=1)
        one_hot = one_hot.drop(one_hot.columns[0], axis=1)
        design_matrix = pd.concat([design_matrix, one_hot], axis=1)
    return design_matrix, initial_ref_cols


def change_reference(df, coef_vec):
    """
    To ensure all scores are positive, we change reference categories for variables with negative coefficients from
    first run of logistic regression
    :param df: Original design matrix for the first run of logistic regression
    :param coef_vec: Coefficients from the first run of logistic regression
    :return: New design matrix for the second run of logistic regression and a list of variables that did not change
    """
    unchanged_vars = []
    variables = np.unique(['_'.join(x.split('_')[:-1]) for x in df.columns])
    for var in variables:
        var_levels = [x for x in df.columns if var == '_'.join(x.split('_')[:-1])]
        if min(coef_vec[var_levels]) < 0:
            ref_category = coef_vec[var_levels].idxmin()
            df = df.drop(ref_category, axis=1)
        else:
            unchanged_vars.append(var)
    return df, unchanged_vars


def add_baseline(df, coef_vec):
    """
    Add the previously dropped reference categories back to the coefficient vector with coefficient as zero
    :param df: Training set without labels
    :param coef_vec: Coefficient vector from the second run of logistic regression
    :return: Coefficient vector with reference categories
    """
    coef_names_all = []
    for col in df.columns:
        coef_names_all += [col + '_' + str(x) for x in set(df[col])]
    coef_vec_all = pd.Series(data=[coef_vec[i] if i in coef_vec.index else 0 for i in coef_names_all],
                             index=coef_names_all)
    return coef_vec_all


def assign_score(df, score_table):
    """
    Assign score to each data sample based on score table
    :param df: Dataframe with all features categorized
    :param score_table: Score table for all variables
    :return: Dataframe with scores for each entry
    """
    if 'label' in df.columns:
        df_new = df.drop('label', axis=1)
    else:
        df_new = df.copy()
    for col in df_new.columns:
        column = df_new[col].apply(str)
        score_table_tmp = score_table[[col in x for x in score_table.index]]
        score_table_tmp = score_table_tmp[['nan' not in idx.split('_')[-1] for idx in score_table_tmp.index]]
        if any(['[' in x for x in score_table_tmp.index]):
            unknown_score = score_table_tmp[col + '_Unknown'] if col + '_Unknown' in score_table_tmp else None
            if unknown_score is not None:
                score_table_tmp = score_table_tmp.drop(col + '_Unknown')
            score_table_tmp = score_table_tmp.sort_index(key=sort_by_interval)
            if unknown_score is not None:
                score_table_tmp[col + '_Unknown'] = unknown_score
        else:
            score_table_tmp = score_table_tmp.sort_index()
        df_new[col] = column.apply(get_score_for_one_entry, args=(col, score_table_tmp))
    if 'label' in df.columns:
        df_new['label'] = df['label']
    return df_new

def get_score_for_one_entry(entry, col, score_table):
    """
    Get score for one dataframe entry
    :param entry: An entry from dataframe (i.e. a category for a variable)
    :param col: Name of the variable
    :param score_table: Score table for all variables
    :return: The score for that entry based on score table
    """
    idx = col + '_' + entry
    if idx in score_table:
        return score_table[idx]
    score_table_no_unknown = score_table.drop(col + '_Unknown') if col + '_Unknown' in score_table else score_table
    first_val = float(score_table_no_unknown.index[0].split('_')[-1].split(',')[-1].strip('[]()'))
    last_val = float(score_table_no_unknown.index[-1].split('_')[-1].split(',')[0].strip('[]()'))
    if float(entry.split('_')[-1].split(',')[-1].strip('[]()')) == first_val:
        return score_table[score_table_no_unknown.index[0]]
    elif float(entry.split('_')[-1].split(',')[0].strip('[]()')) == last_val:
        return score_table[score_table_no_unknown.index[-1]]
    else:
        return 0

def sort_by_interval(series):
    return [float(x.split('_')[-1].split(',')[0].strip('[')) for x in series]

def print_scoring_table(score_table, final_variables):
    """
    Print score table for all included variables
    :param score_table: Score table for all variables
    :param final_variables: List of variables included in the model
    :return: None (Print the score table for each variable)
    """
    for col in final_variables:
        print('Variable:', col)
        score_table_tmp = score_table[[col == '_'.join(x.split('_')[:-1]) for x in score_table.index]]
        score_table_tmp = score_table_tmp[['nan' not in idx.split('_')[-1] for idx in score_table_tmp.index]]
        if any(['[' in x for x in score_table_tmp.index]):
            unknown_score = score_table_tmp[col + '_Unknown'] if col + '_Unknown' in score_table_tmp else None
            if unknown_score is not None:
                score_table_tmp = score_table_tmp.drop(col + '_Unknown')
            score_table_tmp = score_table_tmp.sort_index(key=sort_by_interval)
            if unknown_score is not None:
                score_table_tmp[col + '_Unknown'] = unknown_score
        else:
            score_table_tmp = score_table_tmp.sort_index()
        for col_int in score_table_tmp.index:
            interval = col_int.split('_')[-1]
            if '[' in interval or ')' in interval:
                score_table_no_unknown = score_table_tmp.drop(
                    col + '_Unknown') if col + '_Unknown' in score_table_tmp else score_table_tmp
                if col_int == score_table_no_unknown.index[0]:
                    first_val = interval.split(',')[-1].strip(')').strip()
                    print(f'<{first_val} \t {score_table_no_unknown[col_int]}')
                elif col_int == score_table_no_unknown.index[-1]:
                    last_val = interval.split(',')[0].strip('[').strip()
                    print(f'>={last_val} \t {score_table_no_unknown[col_int]}')
                else:
                    print(interval, '\t', score_table_tmp[col_int])
            else:
                print(interval, '\t', score_table_tmp[col_int])
        print('==========================')

def convert_categorical_vars(df):
    """
    Convert categorical variables to one-hot encoded columns
    :param df: Original dataframe
    :return: New dataframe with categorical variables one-hot encoded
    """
    categorical_features = []
    df_new = df.copy()
    for col in df_new.columns:
        if col != 'label' and (is_object_dtype(df_new[col]) or is_bool_dtype(df_new[col]) or
                               (is_numeric_dtype(df_new[col]) and df_new[col].nunique() <= 5)):
            df_new[col].fillna('Unknown', inplace=True)
            categorical_features.append(col)
    encoder = OneHotEncoder(drop='first')
    one_hot_cols = pd.DataFrame(encoder.fit_transform(df_new[categorical_features]).toarray(),
                                columns=encoder.get_feature_names_out())
    df_new = pd.concat([df_new, one_hot_cols], axis=1)
    df_new = df_new.drop(categorical_features, axis=1)
    return df_new


def check_data(df):
    """
    Check validity of data
    :param df: Dataframe to be checked
    :return: None (Print useful information in case any issue is found)
    :raise ValueError (Column 'label' must be present in df)
    """
    if 'label' not in df.columns:
        raise ValueError("ERROR: These is no dependent variable label to indicate the outcome. Please add label first.")
    if df['label'].nunique() != 2:
        print("WARNING: Please keep outcome variable BINARY.")
    check_predictor(df)


def check_predictor(df):
    """
    Check validity of all variables in the dataframe
    :param df: Dataframe to be checked
    :return: None (Print useful information in case any issue is found)
    """
    df_new = df.drop('label', axis=1)
    # 1. Check for special tokens
    has_special_symbol = False
    special_symbols = [",", "[", "]", "(", ")"]
    for var in df_new.columns:
        for s in special_symbols:
            if s in var:
                has_special_symbol = True
                print(f"WARNING: Variable name {var} has special character {s}. Please double check all feature names.")
    if has_special_symbol:
        print("SUGGESTED ACTION: For each variable name above, consider replacing special characters by '_'.")

    # 2. Check for duplicate names
    has_duplicate_names = False
    cnt = Counter(df_new.columns)
    for var in cnt:
        if cnt[var] > 1:
            has_duplicate_names = True
            print(f"WARNING: Variable name {var} is duplicated. Please double check all feature names.")
    if has_duplicate_names:
        print("SUGGESTED ACTION: For each variable above, please rename them before using AutoScore. "
              "Consider appending '_1', '_2', etc, to variable names.")

    # 3. Check for categorical variables
    too_many_cats = False
    has_cat_var_with_special_symbol = False
    for var in df_new.columns:
        if is_object_dtype(df_new[var]) or is_bool_dtype(df_new[var]):
            cats = df_new[var].astype(str).unique()
            if len(cats) > 10:
                too_many_cats = True
                print(f"WARNING: Too many categories (> 10) in categorical variable {var}")
            for c in cats:
                for s in special_symbols:
                    if s in str(c):
                        has_cat_var_with_special_symbol = True
                        print(f"WARNING: Special character {s} is detected in category {c} of categorical variable {var}")
    if has_cat_var_with_special_symbol:
        print("SUGGESTED ACTION: For each variable above, please change name of categories before using AutoScore."
              "Consider replacing ',' with '_'.")

    if has_special_symbol or has_duplicate_names or too_many_cats or has_cat_var_with_special_symbol:
        print("Please check data again for other potential issues after handling all issues reported above.")
    else:
        print("Data type check passed.")

    # Check for missing values
    na_cnt = df_new.isnull().sum().sum()
    if na_cnt == 0:
        print("No NA in data.")
    else:
        missing_info = pd.DataFrame(columns=['variable', 'num_missing', '%missing'])
        missing_info['variable'] = df_new.columns
        missing_info['num_missing'] = df_new.isnull().sum().reset_index(drop=True)
        missing_info['%missing'] = missing_info['num_missing'] / df_new.shape[0] * 100
        missing_info = missing_info.loc[missing_info['num_missing'] > 0]
        print("WARNING: NA detected in data:")
        print(missing_info.to_string())
        print("SUGGESTED ACTION:\n * Consider imputation and supply AutoScore with complete data.")
        print("* Alternatively, AutoScore can handle missing values as a separate 'Unknown' category, IF:\n",
              "- You believe the missingness in your dataset is informative, AND\n"
              "- Missingness is prevalent enough that you prefer to preserve them as NA rather than removing or doing imputation, AND\n",
              "- Missingness is not too prevalent, which may make results unstable.\n")


def impute_data(train_set, validation_set=None):
    """
    Perform missing value imputation for dataframe
    :param train_set: Training set with potentially missing values
    :param validation_set: Validation set with potentially missing values
    :return: New dataframe with missing values filled
    :raise ValueError (There must not be any missing value in 'label' column)
    """
    n_train = train_set.shape[0]
    if validation_set is not None:
        df = pd.concat([train_set, validation_set], axis=0)
    else:
        df = train_set.copy()
    for col in df.columns:
        if col == 'label' and df[col].isnull().sum() > 0:
            raise ValueError("ERROR: There are missing values in the outcome label. Please fix it and try again.")
        if (is_object_dtype(df[col]) or is_bool_dtype(df[col])) and df[col].isnull().sum() > 0:
            cnt = Counter(df.loc[df[col].notnull(), col])
            mode = max(cnt, key=cnt.get)
            df[col] = df[col].fillna(mode)
        elif df[col].isnull().sum() > 0:
            median = np.nanmedian(df[col])
            df[col] = df[col].fillna(median)
    if validation_set is None:
        return df
    else:
        train_set, validation_set = df.iloc[:n_train], df.iloc[n_train:]
        return train_set, validation_set


def split_data(data, ratio, cross_validation=False, strat_by_label=False):
    """
    Split dataset into train, validation and test set
    :param data: Full dataset
    :param ratio: Ratio of train, validation and test set (e.g. (0.7, 0.1, 0.2))
    :param cross_validation: Whether to use cross validation
    :param strat_by_label: Whether to use stratified sampling by label
    :return: Train set, validation set, test set
    """
    ratio = [x / sum(ratio) for x in ratio]
    n = data.shape[0]
    validation_ratio, test_ratio = ratio[1], ratio[2]
    if not strat_by_label:
        test_index = np.random.choice(n, int(test_ratio * n), replace=False)
        train_validation_index = [x for x in range(n) if x not in test_index]
        validation_index = np.random.choice(train_validation_index, int(validation_ratio * n), replace=False)
        train_index = [x for x in train_validation_index if x not in validation_index]
    else:
        train_index, validation_index, test_index, train_validation_index = [], [], [], []
        for label in data['label'].unique():
            index_for_label = data[data['label'] == label].index
            test_index_for_label = np.random.choice(index_for_label, int(test_ratio * len(index_for_label)), replace=False)
            train_validation_index_for_label = [x for x in index_for_label if x not in test_index_for_label]
            validation_index_for_label = np.random.choice(train_validation_index_for_label,
                                                          int(validation_ratio * len(index_for_label)), replace=False)
            train_index_for_label = [x for x in train_validation_index_for_label if x not in validation_index_for_label]
            train_index.extend(train_index_for_label)
            validation_index.extend(validation_index_for_label)
            test_index.extend(test_index_for_label)
            train_validation_index.extend(train_validation_index_for_label)
    if cross_validation:
        train_set = data.iloc[train_validation_index]
        validation_set = data.iloc[train_validation_index]
        test_set = data.iloc[test_index]
    else:
        train_set = data.iloc[train_index]
        validation_set = data.iloc[validation_index]
        test_set = data.iloc[test_index]
    return train_set, validation_set, test_set

def plot_predicted_risk(pred_score, max_score=100):
    """
    Plot predicted risks (score vs. risk)
    :param pred_score: Predicted total scores from test set
    :param max_score: Maximum possible total score for each data instance
    :return: None (Create two interactive plots for risk visualization)
    """
    risk_df = pd.DataFrame(columns=['score', 'pred_risk'])
    risk_df['score'] = np.arange(0, max_score + 1)
    score_model = LogisticRegression(penalty=None)
    score_model.fit(X=[[x] for x in pred_score['pred_score']], y=pred_score['label'])
    risk_df['pred_risk'] = score_model.predict_proba(np.array(risk_df['score']).reshape(-1, 1))[:, 1]
    score_vs_risk_fig = px.line(risk_df, x='score', y='pred_risk',
                                labels={'score': 'Score', 'pred_risk': 'Predicted risk'})
    score_vs_risk_fig.update_layout(xaxis_title='Score', yaxis_title='Predicted risk')
    score_vs_risk_fig.show()

    risk_df['proportion'] = [pred_score[pred_score['pred_score'] == score].shape[0] /
                             pred_score.shape[0] for score in risk_df['score']]
    pred_score_hist = px.histogram(risk_df, x='score', y='proportion', nbins=len(risk_df['score']),
                                   labels={'score': 'Score', 'proportion': 'Proportion'})
    pred_score_hist.update_layout(xaxis_title='Score', yaxis_title='Proportion of subjects')
    pred_score_hist.show()

def compute_descriptive_table(df):
    """
    Compute a descriptive statistics table for all variables
    :param df: Original dataset
    :return: None (Print the descriptive statistics table)
    """
    categorical = [col for col in df.columns if is_object_dtype(df[col]) or is_bool_dtype(df[col])
                   or (is_numeric_dtype(df[col]) and df[col].nunique() <= 5)]
    table = TableOne(df, categorical=categorical, groupby='label', pval=True, missing=False)
    print(table.tabulate(tablefmt="fancy_grid"))

def model_summary_to_dataframe(model):
    """
    Convert statsmodels result to dataframe
    :param model: Fitted model from statsmodels
    :return: Dataframe with extracted information from the model
    """
    results_df = pd.DataFrame({"pvals": model.pvalues, "coef": model.params,
                               "conf_lower": model.conf_int()[0], "conf_higher": model.conf_int()[1]})
    results_df = results_df.drop('const', axis=0)
    return results_df

def get_final_variables_with_categories(df, final_variables):
    """
    Convert user-defined variable list to column names of the design matrix
    :param df: Design matrix with only binary columns
    :param final_variables: User-defined list of variables
    :return: Transformed list of column names
    """
    final_variables_with_categories = []
    for var in final_variables:
        for col in df.columns:
            variable_name = '_'.join(col.split('_')[:-1])
            if var == variable_name or var == col:
                final_variables_with_categories.append(col)
    result = sorted(list(set(final_variables_with_categories)))
    return result
