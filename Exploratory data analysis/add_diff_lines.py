import difflib
import pandas as pd
import os
from collections import defaultdict
import re


def last_diff(diff, changes):
    """
    Returns the  difference of the last line of the code, in the given diff list and appends it to the changes list.

    Parameters:
    diff (list): The list of differences.
    changes (list): The list to append the last difference to.

    Returns:
    list: The updated changes list.
    """
    i = len(diff) - 1
    if diff[i].startswith('- '):
        changes.append("R " + diff[i][2:])
    elif diff[i].startswith('+ '):
        # line present in f2 but not in f1
        changes.append(f"A:{diff[max(0,i - 1)][2:]}<endRow>{diff[i][2:]}")
    return changes



def get_index_to_add(diff, i):
    """
    Get the index to add based on the given diff and current index.

    Args:
        diff (list): The diff list.
        i (int): The current index.

    Returns:
        int: The index to add.

    """
    index_to_add = i
    while index_to_add >= 0 and diff[index_to_add][0] != ' ':
        index_to_add -= 1
    return index_to_add



def compare_functions(f1, f2):
    """
    Compare two functions and identify the changes made between them.

    Args:
        f1 (str): The first function as a string.
        f2 (str): The second function as a string.

    Returns:
        list: A list of changes made between the two functions.
    """
    changes = []
    f1_lines = f1.splitlines()
    f2_lines = f2.splitlines()
    d = difflib.Differ()
    diff = list(d.compare(f1_lines, f2_lines))
    amount_of_added_lines = 0
    i = 0
    while i < len(diff):
        if diff[i].startswith("+") or diff[i].startswith("?"):
            amount_of_added_lines += 1
        if i + 1 == len(diff):
            changes = last_diff(diff, changes)
            return changes
        if diff[i].startswith('  '):
            # unchanged line
            i += 1
            continue
        elif diff[i].startswith('- '):
            # line present in f1 but not in f2
            changes.append("R " + diff[i][2:])
            isAdded = False
            i += 1
            while not (diff[i].startswith('  ') or  diff[i].startswith('- ')):
                if diff[i].startswith("+") or diff[i].startswith("?"):
                    amount_of_added_lines += 1
                if diff[i].startswith('+ '):                 
                    changes.append("A " + diff[i][2:])
                    isAdded = True
                i += 1
                if i == len(diff):
                    return changes
            if not isAdded:
                changes.append("A " + "EmptyLine")
            i -= 1
        elif diff[i].startswith('+ '):
            # line present in f2 but not in f1
            if i == 0:
                changes.append(f"A:{diff[1][2:]}<endRow>{diff[i][2:]}")
            else:
                if diff[i-1].startswith('+ '):
                    index_to_add = get_index_to_add(diff, i)
                    changes.append(f"A:{diff[index_to_add][2:]}<endRow>{diff[i][2:]}")
                else:
                    changes.append(f"A:{diff[max(0,i - 1)][2:]}<endRow>{diff[i][2:]}")
        i += 1
    return changes




def one_edit(s, index):
    """
    Extracts edits from a list of strings starting from a given index.

    Args:
        s (list): List of strings.
        index (int): Starting index.

    Returns:
        tuple: A tuple containing the updated index and the extracted edits as a string.
    """
    edits = ""
    count = 0
    while s[index+1][0] == "A" and s[index+1][1] != ":":
        edits += (s[index+1][1:]) +"\n"
        count += 1
        index += 1
        if index+1 == len(s):
            break
    return index, edits



def remove_number_suffix(s):
    return re.sub(r"_\d{1,3}$", "", s)



def clear_edits(edits):
    """
    Removes redundant edits from the given list of edits.

    Args:
        edits (list): List of edits.

    Returns:
        list: List of edits with redundant edits removed.
    """
    indexes_to_remove = []
    i = 0
    while i < len(edits):
        if i+1 == len(edits):
            break
        first_line = edits[i].rstrip()  # Remove trailing whitespaces
        first_kind = first_line[0]
        first_line = first_line[2:]
        sec_line = edits[i + 1].rstrip()  # Remove trailing whitespaces
        sec_kind = sec_line[0]
        sec_line = sec_line[2:]
        if first_kind == "R" and sec_kind == "A" and first_line.lstrip() == sec_line.lstrip():
            if i + 2 < len(edits) and (edits[i + 2][:2] == "A:" or edits[i + 2][0] == "R"):
                indexes_to_remove.extend([i, i+1])
                i += 2
            elif i + 2 == len(edits):
                indexes_to_remove.extend([i, i+1])
                i += 2
            else:
                i += 1
        else:
            i += 1
    result_list = [edits[i] for i in range(len(edits)) if i not in set(indexes_to_remove)]
    return result_list



def get_edits(df):
    """
    Extracts edits from a DataFrame containing nonvul and vul columns.

    Args:
        df (pandas.DataFrame): DataFrame containing nonvul and vul columns.

    Returns:
        list: List of dictionaries, where each dictionary represents the edits for a sample.
    """
    all_edits = []
    for sample in range(len(df['nonvul'])):
        s = compare_functions(df['nonvul'].iloc[sample], df['vul'].iloc[sample])
        s = clear_edits(s)
        edits = {}
        i = 0
        postfix = 'a'
        while i < len(s):
            if s[i][0] == "R":
                if i+1 == len(s) or (i+1 < len(s) and s[i+1] == "A EmptyLine"):
                    edits[s[i][1:]] = "EmptyLine"
                    i += 1
                elif i+1 < len(s) and s[i][1] != ":":
                    index, edits[s[i][1:]] = one_edit(s,i)
                    i = index
            else:
                key, value = s[i].split("<endRow>")
                if key not in edits.keys():
                    edits[key] = value
                else:
                    edits[key + postfix] = value
                    postfix = chr(ord(postfix) + 1)
            i += 1
        all_edits.append(edits)
    return all_edits






# def compare_functions(lines_func1, lines_func2):
#     """
#     Compare two lists of lines and return the modified lines.

#     Args:
#         lines_func1 (list): The first list of lines.
#         lines_func2 (list): The second list of lines.

#     Returns:
#         tuple: A tuple containing two lists - modified_lines_func1 and modified_lines_func2.
#                modified_lines_func1 contains the lines that are modified in lines_func1.
#                modified_lines_func2 contains the lines that are modified in lines_func2.
#     """
#     # Perform the comparison
#     d = difflib.Differ()
#     diff = list(d.compare(lines_func1, lines_func2))
#     # Extract modified lines
#     modified_lines_func1 = [line[2:] for line in diff if line.startswith('- ')]
#     modified_lines_func2 = [line[2:] for line in diff if line.startswith('+ ')]
#     return modified_lines_func1, modified_lines_func2
#     return modified_lines_func1, modified_lines_func2




def process_c_functions(df, column_name):
    """
    Process C functions in a DataFrame column by splitting them into lines and
    handling special cases where lines contain only "{" or "}" characters.

    Args:
        df (pandas.DataFrame): The DataFrame containing the C functions.
        column_name (str): The name of the column containing the C functions.

    Returns:
        pandas.DataFrame: The DataFrame with the processed C functions.
    """
    for index, row in df.iterrows():
        lines = row[column_name].split('\n')  # Split the function into lines
        processed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line in ["{", "}"] and i > 0:  # Check if line contains only "{" or "}"
                # Append the character to the end of the previous line
                processed_lines[-1] = processed_lines[-1] + " " + line
            else:
                processed_lines.append(lines[i])  # Add the line as it is
            i += 1
        # Update the DataFrame with the processed function
        df.at[index, column_name] = '\n'.join(processed_lines)
    return df



def append_spaces_suffix_to_duplicates(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Appends spaces as suffix to duplicate lines in the specified column of the given DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        column (str): The column containing the function code.

    Returns:
        pd.DataFrame: The modified DataFrame with spaces appended to duplicate lines.
    """
    modified_data = data.copy()
    for i, row in modified_data.iterrows():
        function_code = row[column]
        lines = function_code.split('\n')
        modified_lines = []
        line_counts = defaultdict(int)  # Start all values equal to 1
        for line in lines:
            if line_counts[line.strip()] > 0:
                spaces = " " * line_counts[line.strip()]
                modified_lines.append(f"{line}{spaces}")
            else:
                modified_lines.append(line)
            line_counts[line.strip()] += 1
        modified_data.at[i, column] = '\n'.join(modified_lines)
    return modified_data


def create_diff_lines_cols(df, new_csv_path):
    """
    Creates new columns in the DataFrame to store the lines before and after the fix.

    Args:
        df (pandas.DataFrame): The input DataFrame containing 'vul' and 'nonvul' columns.

    Returns:
        None
    """
    lines_before = []
    lines_after = []
    nun_of_large = 0
    i = 0
    # Iterate through each row in the DataFrame
    all_edits = get_edits(df)
    for dic in all_edits:
        str_before = ""
        str_after = ""
        for key, value in dic.items():
            if key[0] == "A":
                str_before += value + "\n"
            else:
                if value != "EmptyLine":
                    str_before += value
                else:
                    str_after += ""
                str_after += key + "\n"
        if str_before.count("\n") < 15 and str_after.count("\n") < 15:
            lines_before.append(str_before[:-1])
            lines_after.append(str_after[:-1])
        else:
            nun_of_large += 1
            lines_before.append(None)
            lines_after.append(None)
        i += 1

    # Add the new columns to the DataFrame
    print("Num of nan: " + str(nun_of_large))
    df['lines_with_vul'] = lines_before
    df['lines_after_fix'] = lines_after
    print(df.shape)
    df = df.dropna(subset=['lines_with_vul', 'lines_after_fix'])
    print(df.shape)
    # Save the DataFrame with the new columns to a new CSV file
    df.to_csv(new_csv_path, index=False)



def main(csv_path, new_csv_path):
    """
    Process the CSV file located at `csv_path` and create a new CSV file with diff lines at `new_csv_path`.
    Adding spaces to the end of duplicate lines, to make each line unique.

    Parameters:
    csv_path (str): The path to the input CSV file.
    new_csv_path (str): The path to the output CSV file.

    Returns:
    None
    """
    df = pd.read_csv(csv_path)
    df = df[['vul', 'nonvul', 'name']]
    df = df.dropna(subset=['vul', 'nonvul'])
    df = process_c_functions(df, 'nonvul')
    df = process_c_functions(df, 'vul')
    df = append_spaces_suffix_to_duplicates(df, 'nonvul')
    create_diff_lines_cols(df, new_csv_path)


if __name__ == "__main__":
    main(csv_path='Dataset_VulGen/vulgen_test_with_diff_lines.csv', new_csv_path='vulgen_test_with_diff_lines_spaces.csv')
