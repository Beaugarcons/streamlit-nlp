import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim.models import Word2Vec, FastText
import gensim.downloader as api
import nltk
import re
from adjustText import adjust_text  
        from adjustText import adjust_text

def run():





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

    # 下载 NLTK 数据包
    @st.cache_resource
    def download_nltk_data():
        nltk.download('punkt')
        nltk.download('punkt_tab')

    download_nltk_data()
    st.title("自然语言处理：从统计到分布式的文本表示")

    # ================= 预加载耗时模型 =================
    @st.cache_resource
    def load_glove_model():
        # 使用较小的模型以保证下载和加载速度
        return api.load('glove-twitter-25')

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

    # 创建四个标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. 传统统计 (TF-IDF & LSA)", 
        "2. Word2Vec (CBOW vs Skip-Gram)", 
        "3. GloVe预训练 & 词类比", 
        "4. FastText & Sent2Vec"
    ])

    # ================= Tab 1: 传统统计 (TF-IDF & LSA) =================
    with tab1:
        st.header("基于传统统计的文本表示")
        corpus_input = st.text_area("请输入英文语料：", value=default_corpus, height=150)

        if corpus_input:
            # 切分句子
            sentences = nltk.sent_tokenize(corpus_input)
            st.write(f"**总句数：** {len(sentences)}")

            # 1. TF-IDF
            st.subheader("TF-IDF 矩阵与关键词提取")
            tfidf_vec = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf_vec.fit_transform(sentences)

            # 展示前5个句子和部分特征的TF-IDF矩阵
            df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vec.get_feature_names_out())
            st.dataframe(df_tfidf.head())

            # 提取权重最高的前5个关键词 (基于全局总和)
            sum_tfidf = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
            words = tfidf_vec.get_feature_names_out()
            top_indices = sum_tfidf.argsort()[::-1][:5]
            top_keywords = [(words[i], sum_tfidf[i]) for i in top_indices]
            st.write("**全局权重最高的 5 个关键词：**")
            for word, score in top_keywords:
                st.write(f"- {word}: {score:.4f}")

            # 2. LSA 降维与可视化 (对词汇降维)
            st.subheader("LSA (TruncatedSVD) 词汇 2D 可视化")
            cv = CountVectorizer(stop_words='english')
            X_cv = cv.fit_transform(sentences)
            vocab = np.array(cv.get_feature_names_out())

            # 👉 用 TF-IDF 选“重要词”
            tfidf_vec = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf_vec.fit_transform(sentences)
            tfidf_scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
            tfidf_vocab = tfidf_vec.get_feature_names_out()

            # 取 Top N 重要词
            top_n = st.slider("选择展示词数量", 10, 50, 25)

            top_indices = tfidf_scores.argsort()[::-1][:top_n]
            important_words = tfidf_vocab[top_indices]

            # 找到这些词在 CountVectorizer 中的位置
            mask = np.isin(vocab, important_words)

            X_cv_T = X_cv.T[mask]

            svd = TruncatedSVD(n_components=2, random_state=42)
            coords_2d = svd.fit_transform(X_cv_T)

            # ===== 画图 =====
            fig, ax = plt.subplots(figsize=(10, 6))

            x = coords_2d[:, 0]
            y = coords_2d[:, 1]

            ax.scatter(x, y, alpha=0.7)

            # 👉 解决重叠：偏移 + 小字体

            # ===== 拉开点（关键：增加分散度）=====
            x = coords_2d[:, 0]
            y = coords_2d[:, 1]

            # 标准化 + 放大（让点更分散）
            x = (x - x.mean()) / x.std()
            y = (y - y.mean()) / y.std()

            scale = 3   # 👉 可以调大（3~6）
            x = x * scale
            y = y * scale
            x += np.random.normal(0, 0.1, size=len(x))
            y += np.random.normal(0, 0.1, size=len(y))
            # ===== 画图 =====
            fig, ax = plt.subplots(figsize=(12, 8))

            ax.scatter(x, y, alpha=0.6)

            # ===== 标签 =====
            texts = []
            for i, word in enumerate(vocab[mask]):
                texts.append(
                    ax.text(
                        x[i],
                        y[i],
                        word,
                        fontsize=10,   # 👉 字体变大
                        weight='bold'
                    )
                )

            # ===== 自动避让（核心）=====
            adjust_text(
                texts,
                x=x,
                y=y,
                expand_points=(2.0, 2.0),   # 👉 点的排斥力（调大）
                expand_text=(2.0, 2.0),     # 👉 文字之间排斥（调大）
                force_points=0.5,           # 👉 点影响力
                force_text=1.0,             # 👉 文字之间推开力度（关键）
                arrowprops=dict(
                    arrowstyle='-',
                    color='gray',
                    lw=0.5
                )
            )

            ax.set_title("LSA Word Embedding (Readable Layout)", fontsize=14)
            ax.grid(True)

            st.pyplot(fig)



    # ================= 全局分词准备 (给 Tab 2 和 Tab 4 使用) =================
    sentences_tokenized = [nltk.word_tokenize(re.sub(r'[^\w\s]', '', s.lower())) for s in nltk.sent_tokenize(corpus_input)]
    sentences_tokenized = [s for s in sentences_tokenized if s] # 过滤空句

    # ================= Tab 2: Word2Vec (CBOW vs Skip-Gram) =================
    with tab2:
        st.header("Word2Vec 实时训练与测试")

        col1, col2 = st.columns(2)
        with col1:
            arch_choice = st.radio("选择训练架构：", ["CBOW (sg=0)", "Skip-Gram (sg=1)"])
            sg_val = 1 if "Skip-Gram" in arch_choice else 0
        with col2:
            window_val = st.slider("上下文窗口大小 (window)：", min_value=2, max_value=10, value=5)

        # 训练模型
        if len(sentences_tokenized) > 0:
            w2v_model = Word2Vec(sentences=sentences_tokenized, vector_size=50, window=window_val, sg=sg_val, min_count=1, workers=4)
            st.success(f"Word2Vec 模型训练完成！词表大小：{len(w2v_model.wv.index_to_key)}")

            target_word = st.text_input("输入一个单词以查找最相似的5个词：", value=w2v_model.wv.index_to_key[0] if len(w2v_model.wv.index_to_key)>0 else "")
            if target_word:
                target_word = target_word.lower()
                if target_word in w2v_model.wv:
                    similar_words = w2v_model.wv.most_similar(target_word, topn=5)
                    st.write(f"**与 '{target_word}' 余弦相似度最高的 5 个词：**")
                    for w, sim in similar_words:
                        st.write(f"- {w}: {sim:.4f}")
                else:
                    st.warning("该词不在词表中 (OOV)！")
        else:
            st.warning("语料为空，无法训练模型。")


    # ================= Tab 3: GloVe 预训练 & 词类比 =================
    with tab3:
        st.header("预训练 GloVe 模型与词类比任务")

        with st.spinner("正在加载预训练模型 glove-twitter-25 (初次加载可能需要稍等)..."):
            glove_model = load_glove_model()
        st.success("GloVe 模型加载成功！")

        st.subheader("词类比 (Word Analogy) 计算器")
        st.latex(r"Result \approx Vector(A) - Vector(B) + Vector(C)")
        st.markdown("说明：如果 A与B的关系 类似于 目标词与C的关系。例如：**king(A) - man(B) + woman(C) = queen**")

        c1, c2, c3 = st.columns(3)
        with c1: word_A = st.text_input("单词 A (如: king, paris)", "king").lower()
        with c2: word_B = st.text_input("单词 B (如: man, france)", "man").lower()
        with c3: word_C = st.text_input("单词 C (如: woman, china)", "woman").lower()

        if st.button("计算词类比"):
            try:
                # most_similar 的 positive=[A, C], negative=[B] 相当于 A + C - B
                result = glove_model.most_similar(positive=[word_A, word_C], negative=[word_B], topn=3)
                st.write("**最接近的结果 (Top 3)：**")
                for w, sim in result:
                    st.write(f"- {w}: {sim:.4f}")
            except KeyError as e:
                st.error(f"发生错误：有单词不在预训练词表中 {e}")

        st.markdown("---")
        st.subheader("基础功能：词义相似度")
        c4, c5 = st.columns(2)
        with c4: w1 = st.text_input("单词 1", "computer").lower()
        with c5: w2 = st.text_input("单词 2", "machine").lower()

        if st.button("计算相似度"):
            try:
                sim_score = glove_model.similarity(w1, w2)
                st.info(f"'{w1}' 与 '{w2}' 的余弦相似度为：**{sim_score:.4f}**")
            except KeyError as e:
                st.error(f"词汇不在字典中: {e}")


    # ================= Tab 4: FastText & Sent2Vec =================
    with tab4:
        st.header("FastText 与句子级表示 (Sent2Vec)")

        if len(sentences_tokenized) > 0:
            # 训练 FastText
            ft_model = FastText(sentences=sentences_tokenized, vector_size=50, window=5, min_count=1, workers=4)
            st.success("实时 FastText 模型训练完成！")

            st.subheader("1. OOV (未登录词) 测试")
            st.write("尝试输入一个带拼写错误的词，对比 Word2Vec 和 FastText 的处理能力。")
            oov_word = st.text_input("输入词汇 (例如：computeer)", "computeer").lower()

            if st.button("测试 OOV"):
                col_w2v, col_ft = st.columns(2)

                with col_w2v:
                    st.write("**:red[Word2Vec 结果]**")
                    try:
                        # 尝试从 Tab 2 的 w2v 中获取
                        w2v_sim = w2v_model.wv.most_similar(oov_word, topn=3)
                        st.write(w2v_sim)
                    except KeyError:
                        st.error(f"未登录词异常 (KeyError): '{oov_word}' not in vocabulary.")

                with col_ft:
                    st.write("**:green[FastText 结果]**")
                    try:
                        ft_sim = ft_model.wv.most_similar(oov_word, topn=3)
                        st.write(f"计算成功！与 '{oov_word}' 最相似的词：")
                        for w, sim in ft_sim:
                            st.write(f"- {w}: {sim:.4f}")
                    except Exception as e:
                        st.error(f"错误: {e}")

            st.markdown("---")
            st.subheader("2. 简单的 Sent2Vec (Average Pooling)")
            st.write("提取两个句子的单词向量，求均值后计算整体相似度。")
            sent1 = st.text_input("句子 1：", "The computer is very fast.")
            sent2 = st.text_input("句子 2：", "A machine that computes quickly.")

            if st.button("计算句子相似度"):
                def get_sentence_vector(sentence, model):
                    words = nltk.word_tokenize(re.sub(r'[^\w\s]', '', sentence.lower()))
                    # 获取向量，FastText 能处理 OOV
                    vectors = [model.wv[w] for w in words if w in model.wv or model.__class__ == FastText]
                    if vectors:
                        return np.mean(vectors, axis=0)
                    else:
                        return np.zeros(model.vector_size)

                vec1 = get_sentence_vector(sent1, ft_model)
                vec2 = get_sentence_vector(sent2, ft_model)

                if not np.all(vec1 == 0) and not np.all(vec2 == 0):
                    # cosine_similarity 需要 2D array
                    sim = cosine_similarity([vec1], [vec2])[0][0]
                    st.info(f"两句子的 Sent2Vec 余弦相似度：**{sim:.4f}**")
                else:
                    st.warning("句子中没有有效词汇提取向量。")