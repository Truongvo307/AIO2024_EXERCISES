
import gdown
import os


def download_file(url, out_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdown.download(url, out_path, quiet=False)


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
