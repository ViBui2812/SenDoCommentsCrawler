from crawlCommentSendo_def import *
from tqdm import tqdm
from spam_model_selection.vispamdetection_def import predict_spam_binary
tqdm.pandas()


#Lấy commment từ 1 URL
link = input('Nhập vào URL Sendo: ')
product_id = get_product_id(link)
#Lấy comment từ 1 product_id
df = get_comments(product_id)

#Xử lý xóa duplicate và emojis
df = df.drop_duplicates(subset='id', keep='first') # Xóa duplicates dựa trên cột 'id'
df['comment'] = df['comment'].apply(demoji) #Xóa emoji
df['comment'] = df['comment'].apply(standardize_comment)

#Đánh nhãn SPAM hay No-SPAM
df['SPAM_title'] = df['comment'].progress_apply(predict_spam_binary)
print('--> Đánh nhãn SPAM thành công')
# Lọc ra các dòng mà giá trị trong cột 'SPAM_title' không phải là 'SPAM'
df = df[df['SPAM_title'] != 'Spam']


#Tokennize cột 'comment' để tạo ra cột mới 'split_tokenized_comment' -> output là các mảng
df['split_tokenized_comment'] = df['comment'].progress_apply(tokenize_comment)
print('--> Tokenize comment thành công')

#Tách các mảng sau khi được tokenize thành từng dòng riêng biệt
df = df.explode('split_tokenized_comment')
print('--> Tách các mảng thành từng dòng riêng biệt thành công')

# Xóa các dòng rỗng trong cột 'split_tokenized_comment'
df.dropna(subset=['split_tokenized_comment'], inplace=True)


# Ghép các mảng trong cột 'split_tokenized_comment' thành text theo từng dòng
df['split_tokenized_comment'] = df['split_tokenized_comment'].progress_apply(lambda x: ' '.join(x))
print('--> Ghép các mảng thành text thành công')


#Xóa các dấu cách bị thừa
df['split_tokenized_comment'] = df['split_tokenized_comment'].apply(remove_extra_space)

#Sentiment Analysis với thư viện phoBert, kết quả lưu vào cột mới 'phobert_sentiment_score'
df['phobert_sentiment']= df['split_tokenized_comment'].progress_apply(get_sentiment_scores_by_phobert)
print('--> Phân tích cảm xúc thành công')


df.to_csv('Comment-Crawled.csv', encoding='utf-8-sig')
