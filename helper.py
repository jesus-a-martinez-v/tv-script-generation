import os
import pickle


def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data


def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables, save_file_path='preprocessed.p'):
    """
    Preprocess Text Data
    """
    text = load_data(dataset_path)
    
    # Ignore notice, since we don't use it for analysing the data
    text = text[81:]

    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open(save_file_path, 'wb'))


def load_preprocessed(file_path='preprocessed.p'):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open(file_path, mode='rb'))


def save_parameters(params, file_name='params.p'):
    """
    Save parameters to file
    """
    pickle.dump(params, open(file_name, 'wb'))


def load_parameters(file_name='params.p'):
    """
    Load parameters from file
    """
    return pickle.load(open(file_name, mode='rb'))
