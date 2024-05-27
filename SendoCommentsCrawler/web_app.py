from keywordExtraction_def import *
from crawlCommentSendo_def import *
from tqdm import tqdm
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
from spam_model_selection.vispamdetection_def import *

tqdm.pandas()
st.set_page_config(page_title='Nhom2-BigData' ,layout="wide")


def main():
     # Thanh sidebar với option menu
    with st.sidebar:
        selected = option_menu(
            menu_title="Menu",  # Tiêu đề menu
            options=["Menu", "Comments Summarization", "Visualization"],  # Các tùy chọn
            icons=["house", "archive", "bar-chart-fill"],  # Các icon tương ứng, bạn có thể thay đổi phù hợp
            menu_icon="cast",  # Icon cho menu
            default_index=0,  # Index của mục được chọn mặc định
        )

    if selected  == "Menu":
        st.title("Giới thiệu")
        st.write("Chào mừng bạn đến với ứng dụng phân tích sản phẩm!")
        

    elif selected  == "Comments Summarization":
        st.title("Trích xuất ý chính của comment")
        url = st.text_input("Nhập vào URL sản phẩm:")
        
        if url:
            if 'url' not in st.session_state or st.session_state.url != url:
                st.session_state.url = url
                st.session_state.df = None
                st.session_state.df_comment = None
                st.session_state.df_spam = None
                st.session_state.output_pos = None
                st.session_state.output_neg = None
                # Crawl comments from the provided URL
                try:
                    with st.spinner('Đang lấy comment...'):
                        product_id = get_product_id(url)
                        df = get_comments(product_id)
                        df = df.drop_duplicates(subset='id', keep='first')
                        df['comment'] = df['comment'].apply(demoji)
                        df['comment'] = df['comment'].apply(standardize_comment)

                        


                        df.to_csv('web_app_files/Comment_crawled.csv', encoding='utf-8-sig')
                        

                        st.session_state.df = df
                        st.write('**Bảng Comment thu thập được:**')
                        st.write(df)
                except:
                    st.write("Lấy comment thất bại vui lòng xem lại đường dẫn sản phẩm")
                
                
                with st.spinner('Đang phân tích SPAM...'):
                    df_spam = df.copy()
                    #Đánh nhãn SPAM hay No-SPAM
                    df_spam['SPAM_title'] = df_spam['comment'].progress_apply(predict_spam_binary)
                    
                    st.session_state.df_spam = df_spam
                    st.write('**Bảng Comment sau khi đánh dấu SPAM thu thập được:**')
                    st.write(df_spam)
                    st.text('**-- Sau quá trình nhận diện Spam, sẽ xóa đi các comment được đánh dấu "SPAM" --**')
                    # Lọc ra các dòng mà giá trị trong cột 'SPAM_title' không phải là 'SPAM'
                    df_spam = df_spam[df_spam['SPAM_title'] != 'Spam']
                    
                

                with st.spinner('Đang phân tích... Hãy nhớ: Comment càng nhiều, thời gian càng dài!!!'):
                    
                    df_comment = df_spam.copy()
                    df_comment['comment'] = df_comment['comment'].apply(str)

                    df_comment['split_tokenized_comment'] = df_comment['comment'].apply(tokenize_comment)
                    df_comment = df_comment.explode('split_tokenized_comment')
                    df_comment.dropna(subset=['split_tokenized_comment'], inplace=True)
                    df_comment['split_tokenized_comment'] = df_comment['split_tokenized_comment'].apply(lambda x: ' '.join(x))
                    df_comment['split_tokenized_comment'] = df_comment['split_tokenized_comment'].apply(remove_extra_space)
                    df_comment['phobert_sentiment'] = df_comment['split_tokenized_comment'].apply(get_sentiment_scores_by_phobert)
                   

                    df_comment.to_csv('web_app_files/Comment_crawled_with_tokenized.csv', encoding='utf-8-sig')


                    df_comment_pos = df_comment.loc[df_comment['phobert_sentiment'] == 'Tích cực']
                    df_comment_neg = df_comment.loc[df_comment['phobert_sentiment'] == 'Tiêu cực']
                    
                    st.write('**Bảng Comment sau khi được xử lý Sentiment và Tokenize-tách câu:**')
                    st.session_state.df_comment = df_comment
                    st.write(df_comment)


                with st.spinner('Đang tóm tắt...'):
                    model_summarize = T5ForConditionalGeneration.from_pretrained("NlpHUST/t5-small-vi-summarization")
                    tokenizer_summarize = T5Tokenizer.from_pretrained("NlpHUST/t5-small-vi-summarization")
                    model_summarize.eval()

                    
                    src_pos = clean_and_merge_array(df_comment_pos.split_tokenized_comment.to_list())
                    tokenized_text_pos = tokenizer_summarize.encode(src_pos, return_tensors="pt")
                    
                    summary_ids_pos = model_summarize.generate(
                                        tokenized_text_pos,
                                        max_length=256, 
                                        num_beams=5,
                                        repetition_penalty=2.5, 
                                        length_penalty=1.0, 
                                        early_stopping=True
                                    )
                    output_pos = tokenizer_summarize.decode(summary_ids_pos[0], skip_special_tokens=True)
                    st.session_state.output_pos = output_pos

                    src_neg = clean_and_merge_array(df_comment_neg.split_tokenized_comment.to_list())
                    tokenized_text_neg = tokenizer_summarize.encode(src_neg, return_tensors="pt")
                    summary_ids_neg = model_summarize.generate(
                                        tokenized_text_neg,
                                        max_length=256, 
                                        num_beams=5,
                                        repetition_penalty=2.5, 
                                        length_penalty=1.0, 
                                        early_stopping=True
                                    )
                    output_neg = tokenizer_summarize.decode(summary_ids_neg[0], skip_special_tokens=True)
                    st.session_state.output_neg = output_neg
                    
                    st.write('**Các comment tích cực nhiều nhất:**')
                    st.write(top_frequent_values(df_comment_pos, 'split_tokenized_comment'))
                    st.write('**Các comment tiêu cực nhiều nhất:**')
                    st.write(top_frequent_values(df_comment_neg, 'split_tokenized_comment'))
                    st.write('Bảng Comment sau khi được xử lý Sentiment và Tokenize-tách câu:')
                    st.write('**Tóm tắt bình luận tích cực:**')
                    st.write(output_pos)
                    st.write('**Tóm tắt bình luận tiêu cực:**')
                    st.write(output_neg)


            else:
                st.write('**Bảng Comment thu thập được:**')
                st.write(st.session_state.df)
                st.write('**Bảng Comment sau khi đánh dấu SPAM thu thập được:**')
                st.write(st.session_state.df_spam)
                st.write('**Bảng Comment sau khi được xử lý Sentiment và Tokenize-tách câu:**')
                st.write(st.session_state.df_comment)
                st.write('**Tóm tắt bình luận tích cực:**')
                st.write(st.session_state.output_pos)
                st.write('**Tóm tắt bình luận tiêu cực:**')
                st.write(st.session_state.output_neg)


    elif selected == 'Visualization':
        st.title("Visualization")
        if 'df_comment' in st.session_state:
            # Tạo bar_chart với 3 màu cho từng bar
            removedup = pd.read_csv('web_app_files/Comment_crawled_with_tokenized.csv', encoding = 'utf-8-sig')
            removedup = removedup.drop_duplicates(subset='id', keep='first') #xóa duplicate để tính số lượng comment chính xác


            sentiment_counts = removedup['phobert_sentiment'].value_counts()
        
            fig, ax = plt.subplots(figsize=(8, 5))
            sentiment_counts.plot(kind='bar', color=['blue', 'red', 'grey'], ax=ax)
            
            # Gắn nhãn trục x, trục y, tên biểu đồ
            ax.set_xlabel("Loại cảm xúc")
            ax.set_ylabel("Số lượng")
            ax.set_title("Biểu đồ biểu diễn số lượng của từng loại cảm xúc của dữ liệu", color='r')

            # Gắn data label cho từng bar
            for i, value in enumerate(sentiment_counts):
                ax.text(i, value + 0.1, str(value), ha='center', va='bottom', fontsize=12)


            # Thêm lưới cho biểu đồ
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Điều chỉnh layout
            plt.tight_layout()
            
            # Hiển thị biểu đồ trong Streamlit
            st.pyplot(fig)
        else:
            st.write("Vui lòng trích xuất comment trước.") 


if __name__ == "__main__":
    main()