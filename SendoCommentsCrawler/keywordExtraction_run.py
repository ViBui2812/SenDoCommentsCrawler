import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from keywordExtraction_def import *
from crawlCommentSendo_def import tokenize_comment

if torch.cuda.is_available():       
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

model = T5ForConditionalGeneration.from_pretrained("NlpHUST/t5-small-vi-summarization")
tokenizer = T5Tokenizer.from_pretrained("NlpHUST/t5-small-vi-summarization")

model.to(device)


df_comment =pd.read_csv('Comment-Crawled.csv', encoding='utf-8-sig')


df_comment_pos = df_comment.loc[df_comment['phobert_sentiment'] == 'Tích cực']
df_comment_neg = df_comment.loc[df_comment['phobert_sentiment'] == 'Tiêu cực']


src = df_comment_pos.split_tokenized_comment.to_list() #Thay đổi df_comment_... tùy theo ý muốn muốn tóm tắt tích cực hay tiêu cực
src = list(dict.fromkeys(src))
src = ', '.join(src)
tokenized_text = tokenizer.encode(src, return_tensors="pt").to(device)
model.eval()

summary_ids = model.generate(
                    tokenized_text,
                    max_length=256, 
                    num_beams=5,
                    repetition_penalty=2.5, 
                    length_penalty=1.0, 
                    early_stopping=True
                )
output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(src)