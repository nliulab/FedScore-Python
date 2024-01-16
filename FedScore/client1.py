import json
import os
import numpy as np
import pandas as pd
from client import Client
from context import utils
np.random.seed(42)

if __name__ == '__main__':
    client1 = Client(name="client01", host="127.0.0.1", port=12345)
    msg = client1.sock.recv(client1.BUFFER_SIZE).decode()
    print(msg)
    if msg == 'READY':
        data1 = pd.read_csv('../output/client01_data/sample1.csv')
        data1 = data1.rename({'Mortality_inpatient': 'label'}, axis=1)
        data1 = data1.iloc[:, 1:]
        data1 = utils.convert_categorical_vars(data1)
        train_set1, validation_set1, test_set1 = utils.split_data(data1, (0.7, 0.1, 0.2),
                                                                  cross_validation=False, strat_by_label=False)

        client1.get_local_ranking(train_set1, validation_set1, method="rf", ntree=100)
        client1.get_local_cut_vec(train_set1, quantiles=(0, 0.2, 0.4, 0.6, 0.8, 1))
        client1.generate_design_matrix(train_set1, validation_set1)
        client1.federated_LR_client(n_min=1, n_max=8)
        client1.compute_all_auc_val(validation_set1, n_min=1, n_max=8,
                                    save_filename=f'../output/fig/FL_{client1.strategy}_parsimony.png')

        rank = pd.read_csv(os.path.join(f'../output/{client1.name}_data', 'global_importance.csv'))
        variable_list = list(rank['var'][:3])
        client1.fit_final_model_client(variable_list, max_score=100)

        with open(f'../output/{client1.name}_data/global_cut_vector.json') as f:
            cut_vec = json.loads(f.read())
        client1.test_final_model(test_set1, variable_list, cut_vec, client1.score_table, threshold='best')
