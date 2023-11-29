import math
import os
import shutil
import socket
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from context import AutoScore_binary, utils


class Client:
    def __init__(self, name, host, port):
        self.name = name
        self.BUFFER_SIZE = 4096
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def send_to_server(self, load_path):
        filename, file_size = os.path.basename(load_path), os.path.getsize(load_path)
        msg = f"{filename}@{file_size}"
        self.sock.send(msg.encode())
        with open(load_path, "r") as f:
            data = f.read(self.BUFFER_SIZE)
            while data:
                self.sock.send(data.encode())
                data = f.read(self.BUFFER_SIZE)

    def receive_from_server(self, save_path):
        filename, file_size = self.sock.recv(self.BUFFER_SIZE).decode().split("@")
        msg = f"READY TO RECEIVE {filename} of size {file_size}."
        self.sock.send(msg.encode())
        n_recv = math.ceil(int(file_size) / self.BUFFER_SIZE)
        with open(os.path.join(save_path, filename), "w") as f:
            for _ in range(n_recv):
                data = self.sock.recv(self.BUFFER_SIZE).decode()
                if data:
                    f.write(data)
                else:
                    break

    def get_local_ranking(self, train_set, validation_set=None, method="rf", ntree=100):
        self.local_rank = AutoScore_binary.AutoScore_rank(train_set, validation_set, method, ntree)
        self.local_rank.drop(self.local_rank.columns[1], axis=1, inplace=True)
        self.local_rank['rank'] = range(1, self.local_rank.shape[0] + 1)
        save_path = f"../output/{self.name}_data/{self.name}_rank.csv"
        self.local_rank.to_csv(save_path, index=False)
        self.send_to_server(save_path)
        self.receive_from_server(f"../output/{self.name}_data")

    def get_local_cut_vec(self, df, quantiles=(0, 0.05, 0.2, 0.8, 0.95, 1), max_cluster=5, categorize='quantile'):
        df_new = df.drop('label', axis=1)
        numerical_cut_vec = utils.get_cut_vec(df_new, quantiles, max_cluster, categorize)
        for v in numerical_cut_vec:
            numerical_cut_vec[v] = numerical_cut_vec[v].tolist()
        print(numerical_cut_vec)
        categorical_vars = {}
        for col in df_new.columns:
            if not col in numerical_cut_vec:
                categorical_vars[col] = [int(x) for x in df_new[col].unique()]
        self.local_cut_vec = {'numerical': numerical_cut_vec, 'categorical': categorical_vars}
        print(self.local_cut_vec)
        save_path = f"../output/{self.name}_data/{self.name}_cut_vec.json"
        with open(save_path, 'w') as f:
            json.dump(self.local_cut_vec, f, indent=4)
        self.send_to_server(save_path)
        self.receive_from_server(f"../output/{self.name}_data")

    def generate_design_matrix(self, train_set, validation_set):
        time.sleep(5)
        with open(f'../output/{self.name}_data/global_cut_vector.json') as f:
            global_cut_vec = json.loads(f.read())
        train_set_cat, validation_set_cat = (utils.transform_df_fixed(train_set, global_cut_vec),
                                             utils.transform_df_fixed(validation_set, global_cut_vec))
        self.X_train_cat, self.y_train = train_set_cat.drop('label', axis=1), train_set_cat['label']
        self.design_matrix, self.initial_ref_cols = utils.generate_design_matrix(self.X_train_cat)
        self.design_matrix['label'] = self.y_train
        self.design_matrix.to_csv(f'../output/{self.name}_data/design_matrix.csv', index=False)

    def start_flwr_client(self, variable_list, refit_model=False, change_ref=False):
        num_variables = len(variable_list.split(','))
        if refit_model:
            cmd = (f'python3 ../Flower/client.py ../output/{self.name}_data design_matrix.csv {variable_list}'
                   f' > ../output/{self.name}_data/flwr_output/{self.strategy}/output_flwr_refit.txt 2>&1 &')
        elif change_ref:
            cmd = (f'python3 ../Flower/client.py ../output/{self.name}_data design_matrix.csv {variable_list}'
                   f' > ../output/{self.name}_data/flwr_output/{self.strategy}/output_flwr_final.txt 2>&1 &')
        else:
            cmd = (f'python3 ../Flower/client.py ../output/{self.name}_data design_matrix.csv {variable_list}'
                   f' > ../output/{self.name}_data/flwr_output/{self.strategy}/output_flwr_{num_variables}.txt 2>&1 &')

        print(cmd)
        os.system(cmd)

    def federated_LR_client(self, n_min, n_max):
        time.sleep(5)
        self.strategy = self.sock.recv(self.BUFFER_SIZE).decode()
        print(f'FL Strategy: {self.strategy}')
        if not os.path.exists(f'../output/{self.name}_data/flwr_output'):
            os.mkdir(f'../output/{self.name}_data/flwr_output')
        if not os.path.exists(f'../output/{self.name}_data/flwr_output/{self.strategy}'):
            os.mkdir(f'../output/{self.name}_data/flwr_output/{self.strategy}')
        rank = pd.read_csv(os.path.join(f'../output/{self.name}_data', 'global_importance.csv'))
        for n in range(n_min, n_max + 1):
            variable_list = list(rank['var'][:n])
            variables = ','.join(variable_list)
            self.start_flwr_client(variables)
            time.sleep(15)
        os.system(f'python3 extract_flwr.py ../output/{self.name}_data/flwr_output/{self.strategy}')

    def compute_local_auc_val(self, validation_set, variable_list):
        k = len(variable_list)
        selected_df = validation_set[variable_list + ['label']]
        score_table = pd.read_csv(f'../output/{self.name}_data/flwr_output/{self.strategy}/coef_flwr_{k}.csv', index_col=0)
        score_table = score_table.squeeze()
        selected_df_score = utils.assign_score(selected_df, score_table)
        X_validation, y_validation = selected_df_score.drop('label', axis=1), selected_df_score['label']
        total_scores = np.sum(X_validation, axis=1)
        auc = roc_auc_score(y_validation, total_scores)
        print(f'Select {k} variables, {self.name} AUC: {auc}')
        return self.name, k, auc

    def compute_all_auc_val(self, validation_set, n_min, n_max, save_filename):
        local_AUC_df = pd.DataFrame({'client': [], 'num_var': [], 'AUC': []})
        rank = pd.read_csv(f'../output/{self.name}_data/global_importance.csv')
        for k in range(n_min, n_max + 1):
            variables = list(rank['var'][:k])
            client_name, num_var, auc = self.compute_local_auc_val(validation_set, variables)
            local_AUC_df = pd.concat(
                [local_AUC_df, pd.DataFrame([{'client': client_name, 'num_var': num_var, 'AUC': auc}])])
        print(local_AUC_df.to_string(index=False))
        save_path = f'../output/{self.name}_data/{self.name}_parsimony_local_AUC.csv'
        local_AUC_df.to_csv(save_path, index=False)
        self.send_to_server(save_path)
        self.receive_from_server(f"../output/{self.name}_data")
        mean_AUC_df = pd.read_csv(f"../output/{self.name}_data/parsimony_mean_AUC.csv", index_col=0)
        print(mean_AUC_df.to_string(index=False))
        utils.plot_AUC(mean_AUC_df['AUC'], variables=mean_AUC_df['var'], auc_lim_min=0.5, auc_lim_max='adaptive',
                       ylab='Mean AUC across all clients', title="Parsimony plot on validation set",
                       save_filename=save_filename)

    def fit_final_model_client(self, variable_list, max_score):
        variables = ','.join(variable_list)
        self.sock.send(variables.encode())
        time.sleep(5)
        self.start_flwr_client(variables, refit_model=True)
        time.sleep(25)
        os.system(f'python3 extract_flwr.py ../output/{self.name}_data/flwr_output/{self.strategy}')

        refit_coef = pd.read_csv(f'../output/{self.name}_data/flwr_output/{self.strategy}/coef_flwr_refit.csv', index_col=0)
        refit_coef = refit_coef.squeeze()

        design_matrix, initial_ref_cols, X_train = self.design_matrix.copy(), self.initial_ref_cols.copy(), self.X_train_cat.copy()
        design_matrix = design_matrix[
            [x for x in design_matrix.columns if '_'.join(x.split('_')[:-1]) in variable_list]]
        initial_ref_cols = initial_ref_cols[
            [x for x in initial_ref_cols.columns if '_'.join(x.split('_')[:-1]) in variable_list]]
        design_matrix_final, unchanged_vars = utils.change_reference(design_matrix, refit_coef)
        for v in initial_ref_cols.columns:
            for x in unchanged_vars:
                if x == '_'.join(v.split('_')[:-1]):
                    initial_ref_cols = initial_ref_cols.drop(v, axis=1)
        design_matrix_final = pd.concat([design_matrix_final, initial_ref_cols], axis=1)
        design_matrix_final['label'] = self.design_matrix['label']
        design_matrix_final.to_csv(f'../output/{self.name}_data/design_matrix.csv', index=False)

        self.start_flwr_client(variables, change_ref=True)
        time.sleep(15)
        os.system(f'python3 extract_flwr.py ../output/{self.name}_data/flwr_output/{self.strategy}')

        final_coef = pd.read_csv(f'../output/{self.name}_data/flwr_output/{self.strategy}/coef_flwr_final.csv', index_col=0)
        final_coef = final_coef.squeeze()

        if min(final_coef) < 0:
            coef_index = final_coef.index
            scaler = MinMaxScaler(feature_range=(0.001, 1))
            final_coef = scaler.fit_transform(final_coef.values.reshape(-1, 1))
            final_coef = pd.Series(final_coef.flatten(), index=coef_index)

        X_train = X_train[variable_list]
        score_table = pd.Series(data=[x / min(final_coef) for x in final_coef], index=final_coef.index)
        score_table = utils.add_baseline(X_train, score_table)

        total_max, total = max_score, 0
        for var in variable_list:
            indicator_vars = [x for x in score_table.index if var == '_'.join(x.split('_')[:-1])]
            total += max([score_table[x] for x in indicator_vars])
        score_table = score_table / (total / total_max)
        self.score_table = score_table.round(0).astype(int)
        utils.print_scoring_table(self.score_table, variable_list)
        time.sleep(10)

    def test_final_model(self, test_set, variable_list, cut_vec, score_table, threshold="best"):
        test_set_1 = test_set[variable_list]
        cut_vec = {k: v for (k, v) in cut_vec.items() if k in variable_list}
        test_set_1['label'] = test_set['label']
        test_set_2 = utils.transform_df_fixed(test_set_1, cut_vec)
        test_set_3 = utils.assign_score(test_set_2, score_table).fillna(0)
        X_test, y_test = test_set_3.drop('label', axis=1), test_set_3['label']
        total_score = np.sum(X_test, axis=1)
        print("***Performance using AutoScore:")
        AutoScore_binary.print_roc_performance(total_score, y_test, threshold)
        pred_score = pd.DataFrame(columns=['pred_score', 'label'])
        pred_score['pred_score'], pred_score['label'] = total_score, y_test
        local_AUC = roc_auc_score(y_test, total_score)
        local_AUPRC = average_precision_score(y_test, total_score)
        msg = f"{self.name} AUC: {local_AUC}\n{self.name} AUPRC: {local_AUPRC}"
        print(msg)
        final_AUC_save_path = f'../output/{self.name}_data/{self.name}_final_model_AUC.txt'
        with open(final_AUC_save_path, 'w') as f:
            f.write(msg)
        self.send_to_server(final_AUC_save_path)
        time.sleep(10)
        msg = self.sock.recv(self.BUFFER_SIZE).decode()
        print(msg)
        time.sleep(5)
        msg = self.sock.recv(self.BUFFER_SIZE).decode()
        print(msg)

    def cleanup(self):
        all_temp_files = [file for file in os.listdir(f"../output/{self.name}_data") if not 'sample' in file]
        for file in all_temp_files:
            full_path = os.path.join(f"../output/{self.name}_data", file)
            if os.path.isfile(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
