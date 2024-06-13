import subprocess
import sys
import gdown
import os


def download_file(url, out_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdown.download(url, out_path, quiet=False)


def max_list_sliding_window(num_list, sli_win):
    result = []
    for i in range((len(num_list) - sli_win)+1):
        result.append(max(num_list[i:sli_win+i]))
    return result


def count_letter(string):
    letter_count = {}
    for letter in string:
        if letter.isalpha():
            letter_count[letter] = string.count(letter)
    return letter_count


def count_word_in_file(file_path):
    word_counts = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        content = content.lower()
    file.close()
    words = content.split()
    for word in sorted(words, reverse=False):
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    return word_counts


def levenshtein_distance(s1, s2):
    '''
    This is the function to calculate the Levenshtein. 
    Input: 2 strings A and B 
    It returns the min of step to convert A to B thought out 3 methods (Delete/Add/Remove)
    '''
    m = len(s1)
    n = len(s2)
    # Step 1
    result = [[0] * (n + 1) for _ in range(m + 1)]
    # Step 2
    for i in range(m + 1):
        result[i][0] = i
    for j in range(n + 1):
        result[0][j] = j
    # Step 3
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            result[i][j] = min(result[i - 1][j] + 1, result[i]
                               [j - 1] + 1, result[i - 1][j - 1] + cost)
    return result[m][n]


if __name__ == "__main__":
    print("Exercise 2 - Module 1 - 240609")
    num_list = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1]
    file_url = "https://drive.google.com/uc?id=1IBScGdW2xlNsc9v5zSAya548kNgiOrko"
    output_path = os.path.normpath("./context/file.txt")
    download_file(file_url, output_path)
    string_A = 'kitten'
    string_B = 'sitting'
    print(f"Question 1: {max_list_sliding_window(num_list,3)}")
    print(f"Question 2: {count_letter('smiles')}")
    print(f"Question 3: {count_word_in_file(output_path)['man']}")
    print(
        f"Question 4: The levenshtein distance between {string_A} and {string_B}  {levenshtein_distance(string_A,string_B)}")
