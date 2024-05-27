from spam_model_selection.vispamdetection_def import *


if __name__ == '__main__':
    if not os.path.isdir("spam_model_selection/model"):
        os.mkdir("spam_model_selection/model")

    # De dataset ViSpam trong thu muc dataset/
    train_raw, dev_raw, test_raw = read_dataset()
    # LUU Y: doc ky tung dong comment. chay task nao thi uncomment ra nhe

    # ================ BINARY CLASSIFICATION (SPAM VA NO SPAM) ================================================
    # model, tokenizer = train_model_binary(train_raw, dev_raw)
    # evaluate_model_binary(test_raw, model, tokenizer)

    # ---- TEST ------
    # result = predict_text_binary("comment để nhận xu nhe mn ")
    # print(result)




    # ================ MULTICLASS CLASSIFICATION (NHAN DIEN NO SPAM, SPAM-1, SPAM 2, SPAM-3) ========================
    
    # model, tokenizer = train_model_multiclass(train_raw, dev_raw)
    # evaluate_model_multiclass(test_raw, model, tokenizer)

    # ---- CODE (uncoment de chay) ------
    # result = predict_text_multiclass("comment để nhận xu nhe mn ")
    # print(result)

    pass