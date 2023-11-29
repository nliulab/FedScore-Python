import json
import os
import numpy as np
import pandas as pd
from client import Client
from context import utils
np.random.seed(42)

if __name__ == '__main__':
    client2 = Client(name="client02", host="127.0.0.1", port=12345)
    msg = client2.sock.recv(client2.BUFFER_SIZE).decode()
    print(msg)
    if msg == 'READY':
        # try:
        #     data2 = pd.read_csv('../output/client02_data/sample2.csv')
        #     data2 = data2.rename({'Mortality_inpatient': 'label'}, axis=1)
            # data2 = data2.iloc[:, 1:]
            data2 = pd.read_csv("../../../Documents/nBox/FedScore-python/SGH_all.csv")
            # data2 = pd.read_csv("D:/nbox/W_ED_FL/FedScore-python/SGH_all.csv")
            
            # data2['GENDER'].replace({'FEMALE': 'F', 'MALE': 'M'}, inplace=True)
            data2 = utils.convert_categorical_vars(data2)
            train_set2, validation_set2, test_set2 = utils.split_data(data2, (0.7, 0.1, 0.2),
                                                                      cross_validation=False, strat_by_label=False)

            client2.get_local_ranking(train_set2, validation_set2, method="rf", ntree=100)
            client2.get_local_cut_vec(train_set2, quantiles=(0, 0.2, 0.4, 0.6, 0.8, 1))
            client2.generate_design_matrix(train_set2, validation_set2)
            client2.federated_LR_client(n_min=1, n_max=8)
            client2.compute_all_auc_val(validation_set2, n_min=1, n_max=8,
                                        save_filename=f'../output/fig/FL_{client2.strategy}_parsimony.png')

            rank = pd.read_csv(os.path.join(f'../output/{client2.name}_data', 'global_importance.csv'))
            variable_list = list(rank['var'][:3])
            client2.fit_final_model_client(variable_list, max_score=100)

            with open(f'../output/{client2.name}_data/global_cut_vector.json') as f:
                cut_vec = json.loads(f.read())
            client2.test_final_model(test_set2, variable_list, cut_vec, client2.score_table, threshold='best')
        # except Exception as e:
        #     print('ERROR:', e)
        # finally:
        #     client2.cleanup()
