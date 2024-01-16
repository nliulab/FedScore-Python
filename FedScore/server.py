import json
import math
import os
import shutil
import socket
import threading
import time

import numpy as np
import pandas as pd


class Server:
    def __init__(self, min_clients, host, port, client_weight=None, strategy='fedavg'):
        self.BUFFER_SIZE = 4096
        self.clients = []
        self.min_clients = min_clients
        self.weight = client_weight
        self.strategy = strategy
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen()
        print(f"Server is waiting for {self.min_clients} client connections...")

    def federated_ranking(self, conn, addr):
        self.receive_file_from_clients(conn, addr)
        time.sleep(10)
        local_rankings = [file for file in os.listdir("../output/server_data")
                          if os.path.isfile(os.path.join("../output/server_data", file)) and 'rank.csv' in file]
        local_rankings = sorted(local_rankings)
        all_ranks = []
        for file in local_rankings:
            col_name = file.split('_')[0] + '_rank'
            rank = pd.read_csv(os.path.join('../output/server_data', file), names=['var', col_name], header=None, skiprows=1)
            rank.set_index('var', inplace=True)
            all_ranks.append(rank)
        combined_ranks = pd.concat(all_ranks, axis=1)
        print(combined_ranks.to_string(index=False))

        self.global_ranking = pd.DataFrame()
        if not self.weight:
            self.global_ranking['var'] = combined_ranks.index
            self.global_ranking['rank'] = combined_ranks.reset_index().iloc[:, 1:].mean(axis=1)
        else:
            if np.sum(self.weight) != 1:
                raise ValueError("Weight should be None (uniform by default) OR adds up to 1.")
            elif len(self.weight) != combined_ranks.shape[1]:
                raise ValueError("Length of weight should be the same as the number of clients.")
            self.global_ranking = combined_ranks.mul(self.weight).sum(axis=1).to_frame(name='rank')
            self.global_ranking.reset_index(inplace=True)

        self.global_ranking['rank'] = self.global_ranking['rank'].apply(round, args=(1, ))
        self.global_ranking.sort_values('rank', inplace=True)
        print('Global variable importance ranking:\n', self.global_ranking.to_string(index=False))
        self.global_ranking.to_csv('../output/server_data/global_importance.csv', index=False)
        self.send_to_client(conn, addr, load_path='../output/server_data/global_importance.csv')

    def get_unified_cut_vec(self, conn, addr):
        self.receive_file_from_clients(conn, addr)
        time.sleep(15)
        local_cut_vecs_files = [file for file in os.listdir("../output/server_data")
                          if os.path.isfile(os.path.join("../output/server_data", file)) and 'cut_vec.json' in file]
        local_cut_vecs_files = sorted(local_cut_vecs_files)

        check_cut_vec = {}
        for cut_vec_files in local_cut_vecs_files:
            with open(os.path.join("../output/server_data", cut_vec_files)) as f:
                client_name = cut_vec_files.split('.')[0].split('_')[0]
                print(client_name)
                cut_vec = json.loads(f.read())
                check_cut_vec[client_name] = cut_vec
        check_cut_vec = dict(sorted(check_cut_vec.items()))
        for var in cut_vec['numerical']:
            cut_vec_len = [len(check_cut_vec[c]['numerical'][var]) for c in check_cut_vec]
            if len(set(cut_vec_len)) > 1:
                for c in check_cut_vec:
                    check_cut_vec[c]['numerical'][var] = check_cut_vec[c]['numerical'][var][:1] + check_cut_vec[c]['numerical'][var][-1:]
        for i, c in enumerate(check_cut_vec.keys()):
            with open(os.path.join("../output/server_data", local_cut_vecs_files[i]), "w") as json_file:
                json.dump(check_cut_vec[c], json_file, indent=4)

        all_cut_vecs = {}
        categorical_cut_vecs = {}
        for cut_vec_files in local_cut_vecs_files:
            with open(os.path.join('../output/server_data', cut_vec_files)) as f:
                client_name = cut_vec_files.split('.')[0].split('_')[0]
                print(client_name)
                cut_vec = json.loads(f.read())
                for var in cut_vec['numerical']:
                    if var in all_cut_vecs:
                        all_cut_vecs[var][client_name] = cut_vec['numerical'][var]
                    else:
                        all_cut_vecs[var] = {}
                        all_cut_vecs[var][client_name] = cut_vec['numerical'][var]
                for var in cut_vec['categorical']:
                    if var in categorical_cut_vecs:
                        categorical_cut_vecs[var][client_name] = cut_vec['categorical'][var]
                    else:
                        categorical_cut_vecs[var] = {}
                        categorical_cut_vecs[var][client_name] = cut_vec['categorical'][var]
        print('Numerical cut vectors:\n', all_cut_vecs)
        print('Combined levels of categorical variables:\n', categorical_cut_vecs)
        self.unified_cut_vec = {}
        if not self.weight:
            for var in all_cut_vecs:
                self.unified_cut_vec[var] = []
                for i in range(len(list(all_cut_vecs[var].values())[0])):
                    self.unified_cut_vec[var].append(np.mean([x[i] for x in all_cut_vecs[var].values()]))
                self.unified_cut_vec[var] = sorted(list(set(self.unified_cut_vec[var])))
        else:
            if sum(self.weight) != 1:
                raise ValueError("Weight should be None (uniform by default) OR adds up to 1.")
            for var in all_cut_vecs:
                self.unified_cut_vec[var] = []
                for i in range(len(list(all_cut_vecs[var].values())[0])):
                    self.unified_cut_vec[var].append(round(np.dot(self.weight, [x[i] for x in all_cut_vecs[var].values()]), 1))
                self.unified_cut_vec[var] = sorted(list(set(self.unified_cut_vec[var])))
        self.categorical_vars = {}
        for var in categorical_cut_vecs:
            self.categorical_vars[var] = list(set(sum(list(categorical_cut_vecs[var].values()), [])))
        print('Global cut vector to convert numerical variables to categorical variables:\n', self.unified_cut_vec)
        with open('../output/server_data/global_cut_vector.json', 'w') as f:
            json.dump(self.unified_cut_vec, f, indent=4)
        with open('../output/server_data/combined_categorical_vars.json', 'w') as f:
            json.dump(self.categorical_vars, f, indent=4)
        self.send_to_client(conn, addr, load_path='../output/server_data/global_cut_vector.json')

    def federated_LR(self, conn, addr, n_min, n_max):
        time.sleep(5)
        conn.send(self.strategy.encode())
        rank = pd.read_csv('../output/server_data/global_importance.csv')
        for n in range(n_min, n_max + 1):
            variable_list = list(rank['var'][:n])
            self.start_flwr_server(variable_list)
            time.sleep(15)

    def federated_parsimony(self, conn, addr, n_min, n_max):
        self.receive_file_from_clients(conn, addr)
        time.sleep(10)
        local_AUC_files = [file for file in os.listdir("../output/server_data")
                          if os.path.isfile(os.path.join("../output/server_data", file)) and 'local_AUC.csv' in file]
        local_AUC_files = sorted(local_AUC_files)
        all_AUC = pd.DataFrame()
        for file in local_AUC_files:
            local_AUC = pd.read_csv(os.path.join("../output/server_data", file), header=0)
            all_AUC = pd.concat([all_AUC, local_AUC])
        if not self.weight:
            mean_AUC = all_AUC.groupby('num_var').mean(numeric_only=True).values.flatten()
        else:
            mean_AUC = all_AUC.groupby('num_var').apply(lambda x: np.average(x['AUC'], weights=self.weight)).values.flatten()
        variables = self.global_ranking['var'].iloc[(n_min-1):n_max]
        result_df = pd.DataFrame({'var': variables, 'AUC': mean_AUC})
        result_df.to_csv('../output/server_data/parsimony_mean_AUC.csv')
        self.send_to_client(conn, addr, load_path='../output/server_data/parsimony_mean_AUC.csv')

    def fit_final_model(self, conn):
        variable_list = conn.recv(self.BUFFER_SIZE).decode().split(',')
        self.start_flwr_server(variable_list)
        time.sleep(25)
        self.start_flwr_server(variable_list)

    def evaluate_final_model(self, conn, addr):
        local_AUC_list, local_AUPRC_list = [], []
        self.receive_file_from_clients(conn, addr)
        time.sleep(10)
        local_AUC_files = [file for file in os.listdir("../output/server_data")
                          if os.path.isfile(os.path.join("../output/server_data", file)) and 'final_model_AUC.txt' in file]
        local_AUC_files = sorted(local_AUC_files)
        for file in local_AUC_files:
            with open(os.path.join('../output/server_data', file)) as f:
                file_content = f.readlines()
                local_AUC_list.append(float(file_content[0].split(':')[-1]))
                local_AUPRC_list.append(float(file_content[1].split(':')[-1]))

        if len(local_AUC_list) != len(self.weight) or len(local_AUPRC_list) != len(self.weight):
            raise ValueError('Length of weight should be the same as the number of clients.')
        federated_AUC = np.dot(local_AUC_list, self.weight)
        federated_AUPRC = np.dot(local_AUPRC_list, self.weight)
        msg = (f'Weighted average of AUC across all sites: {federated_AUC}\n'
               f'Weighted average of AUPRC across all sites: {federated_AUPRC}')
        print(msg)
        conn.send(msg.encode())
        time.sleep(5)
        AUC_squared_diff = [(x - federated_AUC) ** 2 for x in local_AUC_list]
        AUPRC_squared_diff = [(x - federated_AUPRC) ** 2 for x in local_AUPRC_list]
        AUC_variation = np.sqrt(np.dot(AUC_squared_diff, self.weight))
        AUPRC_variation = np.sqrt(np.dot(AUPRC_squared_diff, self.weight))
        msg = (f'Weighted variation of AUC across all sites: {AUC_variation}\n'
               f'Weighted variation of AUPRC across all sites: {AUPRC_variation}')
        print(msg)
        conn.send(msg.encode())

    def start_flwr_server(self, variable_list):
        os.system('pkill -f Flower')
        variables = ','.join(variable_list)
        cmd = 'python3 ../Flower/server_{}.py {} {} &'.format(self.strategy, self.min_clients, variables)
        print(cmd)
        os.system(cmd)

    def receive_file_from_clients(self, conn, addr, save_path="../output/server_data"):
        msg = conn.recv(self.BUFFER_SIZE).decode()
        filename, file_size = msg.split("@")
        print(filename, file_size)
        n_recv = math.ceil(int(file_size) / self.BUFFER_SIZE)
        with open(os.path.join(save_path, filename), "w") as f:
            for _ in range(n_recv):
                data = conn.recv(self.BUFFER_SIZE).decode()
                if data:
                    f.write(data)
                else:
                    break

    def send_to_client(self, conn, addr, load_path):
        filename, filesize = os.path.basename(load_path), os.path.getsize(load_path)
        msg = f"{filename}@{filesize}"
        conn.send(msg.encode())
        print(conn.recv(self.BUFFER_SIZE).decode())
        with open(load_path, "r") as f:
            data = f.read(self.BUFFER_SIZE)
            while data:
                conn.send(data.encode())
                data = f.read(self.BUFFER_SIZE)

    def cleanup(self):
        all_temp_files = [file for file in os.listdir(f"../output/server_data")]
        for file in all_temp_files:
            full_path = os.path.join(f"../output/server_data", file)
            if os.path.isfile(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)


def main(server, conn, addr):
    server.federated_ranking(conn, addr)
    server.get_unified_cut_vec(conn, addr)
    server.federated_LR(conn, addr, n_min=1, n_max=8)
    server.federated_parsimony(conn, addr, n_min=1, n_max=8)
    server.fit_final_model(conn)
    server.evaluate_final_model(conn, addr)


if __name__ == "__main__":
    server = Server(min_clients=2, host="127.0.0.1", port=12345, client_weight=(0.5, 0.5), strategy='fedavg')
    while len(server.clients) < server.min_clients:
        conn, addr = server.sock.accept()
        server.clients.append((conn, addr))

        if len(server.clients) >= server.min_clients:
            print(f'{server.min_clients} clients have connected. Starting FedScore...')
            for conn, addr in server.clients:
                conn.send('READY'.encode())
            break

    try:
        for conn, addr in server.clients:
            thread = threading.Thread(
                target=main, args=(server, conn, addr, )
            )
            thread.start()
    except Exception as e:
        print('ERROR:', e)
