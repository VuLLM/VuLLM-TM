from datasets import Dataset
import pandas as pd
import pickle
import re
import hashlib
from transformers import AutoTokenizer
import pickle


def pre_processing(df, with_spaces=True):
    """
    Pre-processes the given DataFrame by selecting specific columns and performing text cleaning operations.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        with_spaces (bool, optional): Flag indicating whether to remove extra spaces in the text. Defaults to True.

    Returns:
        pandas.DataFrame: The pre-processed DataFrame.
    """
    df = df[['nonvul', 'lines_after_fix']]
    if not with_spaces:
        df['lines_after_fix'] = df['lines_after_fix'].apply(lambda x: re.sub(r'\n\s*', '\n', x.strip()))
    df.loc[:, 'nonvul'] = df['nonvul'].str.lstrip()
    return df



def drop_duplicates(df):
    """
    Remove duplicate rows from a DataFrame based on a specific column.

    Args:
        df (pandas.DataFrame): The DataFrame to remove duplicates from.

    Returns:
        pandas.DataFrame: The DataFrame with duplicate rows removed.
    """
    function_groups = {}
    for i in range(len(df)):
        nonvul = df['nonvul'].iloc[i]
        nonvul = nonvul.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
        lines_after_fix = df['lines_after_fix'].iloc[i]
        lines_after_fix = lines_after_fix.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
        row = nonvul + lines_after_fix
        function_hash = hashlib.sha256(row.encode()).hexdigest()
        if function_hash not in function_groups:
            function_groups[function_hash] = []
        function_groups[function_hash].append(i)

    indexes_to_drop = [index for index_list in function_groups.values() if len(index_list) > 1 for index in index_list[1:]]
    df = df.drop(indexes_to_drop)
    df = df.reset_index(drop=True)
    return df




def change_pads_token_expect_the_first_one(labels, tokenizer_pad_token_id):
    """
    Modifies the labels tensor by replacing all occurrences of the pad token
    except for the first one with -100.

    Args:
        labels (torch.Tensor): The labels tensor.
        tokenizer_pad_token_id (int): The ID of the pad token in the tokenizer.

    Returns:
        torch.Tensor: The modified labels tensor.
    """
    for idx, inner_tensor in enumerate(labels):
        if inner_tensor.numel() > 0:
            pad_indices = (inner_tensor == tokenizer_pad_token_id).nonzero(as_tuple=False)
            if pad_indices.numel() > 0:
                pad_index = pad_indices.min()
                inner_tensor[pad_index.item() + 1:] = -100
    return labels



def tokenize(data, tokenizer):
    """
    Tokenizes the input data using the given tokenizer.

    Args:
        data (DataFrame): The input data containing 'nonvul' and 'lines_after_fix' columns.
        tokenizer (Tokenizer): The tokenizer to be used for tokenization.

    Returns:
        dict: A dictionary containing the tokenized model inputs and labels.
    """
    model_inputs = tokenizer(data['nonvul'].tolist(), return_tensors='pt', padding='max_length', max_length=2048, truncation=True)
    model_outputs = tokenizer(data['lines_after_fix'].tolist(), return_tensors='pt', padding='longest', truncation=True)
    labels = model_outputs['input_ids']
    labels = change_pads_token_expect_the_first_one(labels, tokenizer.pad_token_id)
    model_inputs['labels'] = labels
    return model_inputs



def create_datasets(tokenizer, with_spaces=True, path_trainset=None, path_testset=None, is_vulgen=False):
    """
    Create train and test datasets for fine-tuning a model.

    Args:
        tokenizer (Tokenizer): The tokenizer object to tokenize the data.
        with_spaces (bool, optional): Whether to include spaces in the tokenization process. Defaults to True.

    Returns:
        tokenized_train (Dataset): Tokenized train dataset.
        tokenized_test (Dataset): Tokenized test dataset.
    """
    if path_trainset is None or path_testset is None:
        raise ValueError("Path to trainset and testset must be provided.")
    train = pd.read_csv(path_trainset)
    test = pd.read_csv(path_testset)
    train = train.fillna("")
    test = test.fillna("")
    train = pre_processing(train, with_spaces=with_spaces)
    test = pre_processing(test, with_spaces=with_spaces)
    #---delete
    #-train
    if is_vulgen:
        file_path = 'pickle_files/location_with_spaces_to_delete_train.pkl'
        with open(file_path, 'rb') as file:
            indexes_to_delete = pickle.load(file)
        train = train.drop(indexes_to_delete)
        train = train.reset_index(drop=True)
        # #-test
        file_path = 'pickle_files/location_with_spaces_to_delete_test.pkl'
        with open(file_path, 'rb') as file:
            indexes_to_delete = pickle.load(file)
        test = test.drop(indexes_to_delete)
        test = test.reset_index(drop=True)
    train = drop_duplicates(train)
    test = drop_duplicates(test)
    # max_length_vul = train['vul'].str.len().max()
    print("num of samples TrainSet: ", len(train))
    print("num of samples TestSet: ", len(test))
    tokenized_train = tokenize(train, tokenizer)
    tokenized_test = tokenize(test, tokenizer)
    tokenized_train = Dataset.from_dict(tokenized_train)
    tokenized_test = Dataset.from_dict(tokenized_test)
    return tokenized_train, tokenized_test


def create_testset(tokenizer, with_spaces=True, path_testset=None, is_vulgen=False):
    """
    Create train and test datasets for fine-tuning a model.

    Args:
        tokenizer (Tokenizer): The tokenizer object to tokenize the data.
        with_spaces (bool, optional): Whether to include spaces in the tokenization process. Defaults to True.

    Returns:
        tokenized_train (Dataset): Tokenized train dataset.
        tokenized_test (Dataset): Tokenized test dataset.
    """
    if path_testset is None:
        raise ValueError("Path to testset must be provided.")
    test = pd.read_csv(path_testset)
    test = test.fillna("")
    test = pre_processing(test, with_spaces=with_spaces)
    #---delete
    #-train
    if is_vulgen:
        # #-test
        file_path = 'pickle_files/location_with_spaces_to_delete_test.pkl'
        with open(file_path, 'rb') as file:
            indexes_to_delete = pickle.load(file)
        test = test.drop(indexes_to_delete)
        test = test.reset_index(drop=True)
        test = drop_duplicates(test)
    # max_length_vul = train['vul'].str.len().max()
    print("num of samples TestSet: ", len(test))
    tokenized_test = tokenize(test, tokenizer)
    tokenized_test = Dataset.from_dict(tokenized_test)
    return tokenized_test


def create_testset_for_only_decoder(tokenizer, with_spaces=True, path_testset=None, is_vulgen=False):
    """
    Create train and test datasets for fine-tuning a model.

    Args:
        tokenizer (Tokenizer): The tokenizer object to tokenize the data.
        with_spaces (bool, optional): Whether to include spaces in the tokenization process. Defaults to True.

    Returns:
        tokenized_train (Dataset): Tokenized train dataset.
        tokenized_test (Dataset): Tokenized test dataset.
    """
    if path_testset is None:
        raise ValueError("Path to testset must be provided.")
    test = pd.read_csv(path_testset)
    test = test.fillna("")
    test = pre_processing(test, with_spaces=with_spaces)
    #---delete
    #-train
    if is_vulgen:
        # #-test
        file_path = 'pickle_files/location_with_spaces_to_delete_test.pkl'
        with open(file_path, 'rb') as file:
            indexes_to_delete = pickle.load(file)
        test = test.drop(indexes_to_delete)
        test = test.reset_index(drop=True)
        test = drop_duplicates(test)
    # max_length_vul = train['vul'].str.len().max()
    print("num of samples TestSet: ", len(test))
    return test


def create_datasets_for_only_decoder(tokenizer, with_spaces=True, path_trainset=None, path_testset=None, is_vulgen=False):
    """
    Create train and test datasets for fine-tuning a model.

    Args:
        tokenizer (Tokenizer): The tokenizer object to tokenize the data.
        with_spaces (bool, optional): Whether to include spaces in the tokenization process. Defaults to True.

    Returns:
        tokenized_train (Dataset): Tokenized train dataset.
        tokenized_test (Dataset): Tokenized test dataset.
    """
    if path_trainset is None or path_testset is None:
        raise ValueError("Path to trainset and testset must be provided.")
    train = pd.read_csv(path_trainset)
    test = pd.read_csv(path_testset)
    train = train.fillna("")
    test = test.fillna("")
    train = pre_processing(train, with_spaces=with_spaces)
    test = pre_processing(test, with_spaces=with_spaces)
    #---delete
    #-train
    if is_vulgen:
        file_path = 'pickle_files/location_with_spaces_to_delete_train.pkl'
        with open(file_path, 'rb') as file:
            indexes_to_delete = pickle.load(file)
        train = train.drop(indexes_to_delete)
        train = train.reset_index(drop=True)
        # #-test
        file_path = 'pickle_files/location_with_spaces_to_delete_test.pkl'
        with open(file_path, 'rb') as file:
            indexes_to_delete = pickle.load(file)
        test = test.drop(indexes_to_delete)
        test = test.reset_index(drop=True)
    train = drop_duplicates(train)
    # test = drop_duplicates(test)
    # max_length_vul = train['vul'].str.len().max()
    print("num of samples TrainSet: ", len(train))
    print("num of samples TestSet: ", len(test))
    return train, test


def too_large_input(tokenized_train, tokenized_test):
    """
    Identifies the indices of input tensors that are too large in the tokenized_train and tokenized_test datasets.

    Args:
        tokenized_train (dict): Tokenized training dataset.
        tokenized_test (dict): Tokenized test dataset.

    Returns:
        tuple: A tuple containing two lists. The first list contains the indices of input tensors that are too large in the tokenized_train dataset. The second list contains the indices of input tensors that are too large in the tokenized_test dataset.
    """
    too_long_to_delete_train = []
    too_long_to_delete_test = []
    for idx, tensor in enumerate(tokenized_train['input_ids']):
        if tensor[2047] != 50256:
            too_long_to_delete_train.append(idx)
    if len(tokenized_train['labels'][0]) > 500:
        for idx, tensor in enumerate(tokenized_train['labels']):
            if tensor[500] != -100:
                too_long_to_delete_train.append(idx)
    for idx, tensor in enumerate(tokenized_test['input_ids']):
        if tensor[2047] != 50256:
            too_long_to_delete_test.append(idx)
    if len(tokenized_test['labels'][0]) > 500:
        for idx, tensor in enumerate(tokenized_test['labels']):
            if tensor[500] != -100:
                too_long_to_delete_test.append(idx)
    return too_long_to_delete_train, too_long_to_delete_test


# tokenized_train, tokenized_test = create_datasets(AutoTokenizer.from_pretrained('Salesforce/codet5p-6b'), with_spaces=True, path_trainset="Dataset_VulGen/vulgen_train_drop_dup.csv", path_testset="Dataset_VulGen/vulgen_test_drop_dup.csv", is_vulgen=False)

# too_long_to_delete_train, too_long_to_delete_test = too_large_input(tokenized_train, tokenized_test)

# # Save tokenized_train and tokenized_test to pickle files
# with open('pickle_files/location_with_spaces_to_delete_train.pkl', 'wb') as file:
#     pickle.dump(too_long_to_delete_train, file)

# with open('pickle_files/location_with_spaces_to_delete_test.pkl', 'wb') as file:
#     pickle.dump(too_long_to_delete_test, file)