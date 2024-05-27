import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from nltk import ngrams
from collections import Counter


def features_list(string, n=1):
    gram_str = list(ngrams(string.split(), n))
    return [ " ".join(gram).lower() for gram in gram_str ]


def clean_and_merge_array(src):
    # Loại bỏ các phần tử trùng lặp và giữ nguyên thứ tự của chúng
    clean_array = list(dict.fromkeys(src))
    
    # Ghép các phần tử còn lại thành một chuỗi duy nhất
    merge = ', '.join(clean_array)
    
    return merge

def top_frequent_values(df, column_name, n=3):
    # Đếm tần suất xuất hiện của mỗi giá trị trong cột 'column_name'
    counter = Counter(df[column_name])
    
    # Lấy 'n' giá trị có tần suất cao nhất
    most_common = counter.most_common(n)
    
    # In ra kết quả
    result = [f"{item}: {count}" for item, count in most_common]
    return result