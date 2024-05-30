import requests
import pandas as pd
from urllib.parse import urlparse
import os
import re
from tqdm import tqdm
from underthesea import sent_tokenize, word_tokenize, sentiment
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyvi import ViTokenizer
import gensim



def get_product_id(sendo_url):
    # Lấy ra product_id từ URL sản phẩm bằng urlparse
    parsed_url = urlparse(sendo_url)
    # Lấy phần "path" của URL, tách bằng dấu "-"
    path_parts = parsed_url.path.split('-')
    # Lấy phần cuối cùng và bỏ đi đuôi .html
    product_id = path_parts[-1].replace('.html', '')
    return product_id

#Hàm tạo ra cấu trúc để lưu dữ liệu sau khi crawl data
def comment_parser(json):
    d = dict()
    d['id'] = json.get('rating_id')
    d['title'] = json.get('comment_title')
    d['comment'] = json.get('comment')
    d['default_sentiment'] = json.get('status')
    d['like_count'] = json.get('like_count')
    d['customer_id'] = json.get('customer_id')
    d['rating_star'] = json.get('star')
    d['customer_name'] = json.get('user_name')
    return d

#Hàm lấy comment 
def get_comments(product_id):
    #Khai báo Header tương ứng như API cung cấp
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'vi,vi-VN;q=0.9,fr-FR;q=0.8,fr;q=0.7,en-US;q=0.6,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    #Khai báo Params tương ứng như API cung cấp
    params = {
        'page': 1,
        'limit': 10,
        'sort': 'review_score',
        'v': '2',
        'star': 'all'
    }

    result = []
    while True:
        response = requests.get(f'https://ratingapi.sendo.vn/product/{product_id}/rating',headers=headers,params=params)
        if response.status_code == 200:
            data = response.json().get('data')
            if not data:  # Check if there are no more comments
                break
            for comment in data:
                parsed_comment = comment_parser(comment)
                # Check if comment ID already exists in results before adding
                if parsed_comment['id'] not in [c['id'] for c in result]:
                    result.append(parsed_comment)
            params['page'] += 1
        else:
            print(f"Error getting comments for page {params['page']}. Status code: {response.status_code}")
            break

    df_comment = pd.DataFrame(result)
    return df_comment

#Hàm standardize_comment để chuẩn hóa comment trước khi Sentiment Analysis:
def standardize_comment(comment):
    comment = comment.replace('\n', ' ')\
                    .replace('\r', ' ')\
                    .replace('"', ' ').replace("”", " ")\
                    .replace(":", " ")\
                    .replace("!", " ")\
                    .replace("?", " ") \
                    .replace("-", " ")\
                    .replace(". .", ' ')\
                    .lower()
    return comment

#Hàm xóa dấu cách bị thừa trong comment
def remove_extra_space(comment):
    comment = ' '.join(comment.split())  # Loại bỏ các dấu cách thừa
    return comment

#Hàm xóa Emoji ra khỏi comment
def demoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

# Hàm để tokenize từng comment
def tokenize_comment(comment):
    # Tách câu
    sentences = sent_tokenize(comment)
    # Tách từ trong mỗi câu và lưu kết quả
    tokenized_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)    
        #Xóa dấu câu khỏi mỗi từ
        for i, word in enumerate(words):
            words[i] = re.sub(r'[^\w\s]', '', word)
            
        tokenized_sentences.append(words)
    return tokenized_sentences



# Khởi tạo mô hình và tokenizer 
checkpoint = "mr4/phobert-base-vi-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
#Hàm sentiment bằng PhoBert cho từng dòng trong cột comment:
def get_sentiment_scores_by_phobert(text, batch_size=32):
    neg_to_pos=['sản phẩm dịch vụ giống mô tả và tốt hơn mong đợi', 'sản phẩm giống mô tả','sản phẩm dịch vụ giống mô tả','dịch vụ giống mô tả']

    num_batches = len(text) // batch_size + (1 if len(text) % batch_size != 0 else 0)
    
    for i in tqdm(range(num_batches), desc="Đang tạo Sentiments..."):
        batch_texts = text[i*batch_size:(i+1)*batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        
        # Dự báo và lấy ra nhãn có trị số cao nhất
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        for prediction in predictions:
            scores = prediction.tolist()
            highest_score_index = scores.index(max(scores))
            label_mapping = model.config.id2label
            dominant_sentiment = label_mapping[highest_score_index]
    if text in neg_to_pos:
        dominant_sentiment = 'Tích cực'
    return dominant_sentiment

#Hàm dùng ViTokenzier để xử lý phục vụ cho việc feature extraction
def process_comments(text):
    # Preprocess với gensim
    lines = gensim.utils.simple_preprocess(str(text))
    # Ghép lại thành chuỗi
    lines = ' '.join(lines)
    # Tokenize với ViTokenizer
    tokenized_text = ViTokenizer.tokenize(lines)
    return tokenized_text



#Khai báo thư viện stopword và stopword_dash
stopword = open('./vietnamese-stopwords.txt',encoding='utf-8').read()
stopword_dash = open('./vietnamese-stopwords-dash.txt',encoding='utf-8').read()
# Tạo hàm loại bỏ stopwords
def remove_stopwords(line):
    words = []
    for word in line.strip().split():
        if word not in stopword:
            words.append(word)
        elif word not in stopword_dash:
            words.append(word)
    return ' '.join(words)


