import pickle
from datetime import datetime


def print_process(x):
    print(x, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '', sep='\n')


def load_file_names(path):
    train_file_names = [path + f'collected_jsonl/train_features_{i}.jsonl' for i in range(6)]
    test_file_names = [path + 'collected_jsonl/test_features.jsonl']
    return train_file_names, test_file_names


def save(save_path, save_id, *data):
    if save_id == 1:
        data[0].to_csv(save_path + 'data/one_train_df.csv', index=False)
        data[1].to_csv(save_path + 'data/one_test_df.csv', index=False)
        with open(save_path + 'properties/one_props.pickle', 'wb') as file:
            pickle.dump(data[2], file)

    elif save_id == 2:
        with open(save_path + 'data/two_train_df.pickle', 'wb') as file:
            pickle.dump(data[0], file)
        with open(save_path + 'data/two_test_df.pickle', 'wb') as file:
            pickle.dump(data[1], file)
        with open(save_path + 'properties/two_props.pickle', 'wb') as file:
            pickle.dump(data[2], file)

    elif save_id == 3:
        with open(save_path + 'model/one_rf_model.pickle', 'wb') as file:
            pickle.dump(data[0], file)
