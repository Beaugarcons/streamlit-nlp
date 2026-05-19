import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from gensim.models import Word2Vec, FastText

import nltk
import re

from adjustText import adjust_text


def run():

    # ================= 页面样式 =================
    st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
    }

    h1, h2, h3 {
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    # ================= 下载 NLTK 数据 =================
    @st.cache_resource
    def download_nltk_data():
        nltk.download('punkt')

    download_nltk_data()

    st.title("自然语言处理：从统计到分布式的文本表示")

    # ================= 默认语料 =================
    default_corpus = """
    The quick brown fox jumps over the lazy dog.
    A king is a man who rules a country.
    A queen is a woman who rules a country.
    Paris is the capital of France.
    Beijing is the capital of China.
    Machine learning is a subset of artificial intelligence.
    The computer is a powerful tool for computation.
    A fast computer makes computation easier.
    """

    # ================= 创建标签页 =================
    tab1, tab2, tab3 = st.tabs([
        "1. TF-IDF & LSA",
        "2. Word2Vec",
        "3. FastText & Sent2Vec"
    ])

    # =========================================================
    # Tab 1
    # =========================================================
    with tab1:

        st.header("传统统计文本表示")

        corpus_input = st.text_area(
            "请输入英文语料：",
            value=default_corpus,
            height=180
        )

        if corpus_input:

            # 分句
            sentences = nltk.sent_tokenize(corpus_input)

            st.write(f"总句数：{len(sentences)}")

            # ================= TF-IDF =================
            st.subheader("TF-IDF 矩阵")

            tfidf_vec = TfidfVectorizer(stop_words='english')

            tfidf_matrix = tfidf_vec.fit_transform(sentences)

            df_tfidf = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=tfidf_vec.get_feature_names_out()
            )

            st.dataframe(df_tfidf)

            # ================= 关键词 =================
            st.subheader("关键词提取")

            scores = np.asarray(
                tfidf_matrix.sum(axis=0)
            ).ravel()

            words = tfidf_vec.get_feature_names_out()

            top_indices = scores.argsort()[::-1][:5]

            for idx in top_indices:
                st.write(
                    f"{words[idx]} : {scores[idx]:.4f}"
                )

            # ================= LSA =================
            st.subheader("LSA 二维可视化")

            cv = CountVectorizer(stop_words='english')

            X_cv = cv.fit_transform(sentences)

            vocab = np.array(cv.get_feature_names_out())

            # 选取重要词
            top_n = st.slider(
                "选择展示词数量",
                10,
                30,
                20
            )

            tfidf_scores = np.asarray(
                tfidf_matrix.sum(axis=0)
            ).ravel()

            top_indices = tfidf_scores.argsort()[::-1][:top_n]

            important_words = words[top_indices]

            mask = np.isin(vocab, important_words)

            X_cv_T = X_cv.T[mask]

            # 降维
            svd = TruncatedSVD(
                n_components=2,
                random_state=42
            )

            coords_2d = svd.fit_transform(X_cv_T)

            x = coords_2d[:, 0]
            y = coords_2d[:, 1]

            # 标准化
            x = (x - x.mean()) / x.std()
            y = (y - y.mean()) / y.std()

            x = x * 3
            y = y * 3

            fig, ax = plt.subplots(figsize=(10, 7))

            ax.scatter(x, y)

            texts = []

            for i, word in enumerate(vocab[mask]):

                texts.append(
                    ax.text(
                        x[i],
                        y[i],
                        word,
                        fontsize=10
                    )
                )

            adjust_text(texts)

            ax.set_title("LSA Word Visualization")

            st.pyplot(fig)

    # =========================================================
    # 分词
    # =========================================================
    sentences_tokenized = [
        nltk.word_tokenize(
            re.sub(r'[^\w\s]', '', s.lower())
        )
        for s in nltk.sent_tokenize(corpus_input)
    ]

    sentences_tokenized = [
        s for s in sentences_tokenized if s
    ]

    # =========================================================
    # Tab 2
    # =========================================================
    with tab2:

        st.header("Word2Vec")

        col1, col2 = st.columns(2)

        with col1:

            arch_choice = st.radio(
                "选择训练架构：",
                [
                    "CBOW (sg=0)",
                    "Skip-Gram (sg=1)"
                ]
            )

            sg_val = 1 if "Skip-Gram" in arch_choice else 0

        with col2:

            window_val = st.slider(
                "window",
                2,
                10,
                5
            )

        if len(sentences_tokenized) > 0:

            w2v_model = Word2Vec(
                sentences=sentences_tokenized,
                vector_size=50,
                window=window_val,
                sg=sg_val,
                min_count=1,
                workers=4
            )

            st.success(
                f"模型训练完成，词表大小：{len(w2v_model.wv.index_to_key)}"
            )

            target_word = st.text_input(
                "输入单词：",
                value=w2v_model.wv.index_to_key[0]
            )

            if target_word:

                target_word = target_word.lower()

                if target_word in w2v_model.wv:

                    similar_words = w2v_model.wv.most_similar(
                        target_word,
                        topn=5
                    )

                    st.write("最相似的5个词：")

                    for w, sim in similar_words:

                        st.write(f"{w} : {sim:.4f}")

                else:

                    st.warning("词不在词表中")

    # =========================================================
    # Tab 3
    # =========================================================
    with tab3:

        st.header("FastText 与 Sent2Vec")

        if len(sentences_tokenized) > 0:

            ft_model = FastText(
                sentences=sentences_tokenized,
                vector_size=50,
                window=5,
                min_count=1,
                workers=4
            )

            st.success("FastText 模型训练完成")

            # ================= OOV =================
            st.subheader("OOV 测试")

            oov_word = st.text_input(
                "输入词汇：",
                "computeer"
            ).lower()

            if st.button("测试 OOV"):

                col_w2v, col_ft = st.columns(2)

                # Word2Vec
                with col_w2v:

                    st.write("Word2Vec")

                    try:

                        w2v_sim = w2v_model.wv.most_similar(
                            oov_word,
                            topn=3
                        )

                        st.write(w2v_sim)

                    except:

                        st.error("Word2Vec 无法处理 OOV")

                # FastText
                with col_ft:

                    st.write("FastText")

                    try:

                        ft_sim = ft_model.wv.most_similar(
                            oov_word,
                            topn=3
                        )

                        for w, sim in ft_sim:

                            st.write(f"{w} : {sim:.4f}")

                    except Exception as e:

                        st.error(e)

            # ================= Sent2Vec =================
            st.subheader("句子相似度")

            sent1 = st.text_input(
                "句子1",
                "The computer is very fast."
            )

            sent2 = st.text_input(
                "句子2",
                "A machine that computes quickly."
            )

            if st.button("计算句子相似度"):

                def get_sentence_vector(sentence, model):

                    words = nltk.word_tokenize(
                        re.sub(
                            r'[^\w\s]',
                            '',
                            sentence.lower()
                        )
                    )

                    vectors = [
                        model.wv[w]
                        for w in words
                    ]

                    if vectors:

                        return np.mean(vectors, axis=0)

                    else:

                        return np.zeros(model.vector_size)

                vec1 = get_sentence_vector(
                    sent1,
                    ft_model
                )

                vec2 = get_sentence_vector(
                    sent2,
                    ft_model
                )

                sim = cosine_similarity(
                    [vec1],
                    [vec2]
                )[0][0]

                st.info(
                    f"句子余弦相似度：{sim:.4f}"
                )