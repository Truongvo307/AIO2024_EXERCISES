from sliding_window import max_list_sliding_window
from levenshtein_distance import levenshtein_distance
from word_counting import count_letter, count_word_in_file, download_file
import os


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
