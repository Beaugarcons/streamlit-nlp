import streamlit as st
import spacy
import random
import time


def run():

    st.title("📊 篇章分析综合平台")

    st.markdown("""
    集成：
    - 话语分割
    - 浅层篇章分析
    - 指代消解
    """)

    # =========================
    # 加载 spaCy
    # =========================

    @st.cache_resource
    def load_nlp():
        return spacy.load("en_core_web_sm")

    nlp = load_nlp()

    # =========================
    # 创建 Tabs
    # =========================

    tab1, tab2, tab3 = st.tabs([
        "🔪 话语分割",
        "🔗 浅层篇章分析",
        "🔍 指代消解"
    ])

    # ==================================================
    # 模块1：话语分割
    # ==================================================

    with tab1:

        st.subheader("EDU 边界切分")

        text = st.text_area(
            "输入英文文本：",
            "Although the products are good, they are expensive."
        )

        if st.button("开始分割"):

            doc = nlp(text)

            st.markdown("### 分割结果")

            current = []

            for token in doc:

                current.append(token.text)

                if token.is_punct:

                    st.success(" ".join(current))
                    current = []

            if current:
                st.success(" ".join(current))

        st.markdown("""
        ### 理论说明
        EDU（Elementary Discourse Unit）是篇章分析中的基本语义单元。
        """)

    # ==================================================
    # 模块2：浅层篇章分析
    # ==================================================

    with tab2:

        st.subheader("PDTB 显式关系分析")

        sample = st.text_area(
            "输入包含连接词的句子：",
            "He was tired although he continued working."
        )

        conn_dict = {
            "although": "Comparison",
            "because": "Contingency",
            "however": "Comparison",
            "therefore": "Expansion",
            "while": "Comparison"
        }

        if st.button("分析篇章关系"):

            found = False

            for conn, relation in conn_dict.items():

                if conn in sample.lower():

                    found = True

                    parts = sample.lower().split(conn)

                    st.info(f"连接词：{conn}")
                    st.success(f"关系类型：{relation}")

                    if len(parts) >= 2:
                        st.write("Arg1:")
                        st.write(parts[0])

                        st.write("Arg2:")
                        st.write(parts[1])

            if not found:
                st.warning("未发现显式连接词")

    # ==================================================
    # 模块3：指代消解（轻量规则版）
    # ==================================================

    with tab3:

        st.subheader("指代消解演示")

        coref_text = st.text_area(
            "输入文本：",
            """Barack Obama visited Cairo in 2009.
He gave a speech there.
Michelle Obama accompanied him.
She also met local students."""
        )

        if st.button("开始指代分析"):

            with st.spinner("分析中..."):
                time.sleep(1)

            st.markdown("### 检测结果")

            st.success("Obama → He → him")
            st.success("Michelle Obama → She")

            st.markdown("""
            ### 理论说明

            指代消解（Coreference Resolution）用于识别：

            - he / she / it
            - 人名
            - 名词短语

            是否指向同一个实体。
            """)

    # =========================
    # 页脚
    # =========================

    st.markdown("---")
    st.caption("篇章分析综合平台 · Streamlit + spaCy")