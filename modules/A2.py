import streamlit as st
import spacy
from spacy import displacy
from nltk import CFG
from nltk.parse import ChartParser


def run():

    st.title("🧠 NLP句法分析可视化 Demo")

    # =========================
    # 加载 spaCy 模型
    # =========================
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        st.error("未检测到 en_core_web_sm 模型")
        st.info(
            "请在 requirements.txt 中添加：\n"
            "https://github.com/explosion/spacy-models/releases/download/"
            "en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz"
        )
        return

    # =========================
    # 示例句
    # =========================
    example_sentences = [
        "The boy saw the man with the telescope.",
        "I saw her duck.",
        "Visiting relatives can be annoying."
    ]

    selected_example = st.selectbox(
        "选择示例句：",
        example_sentences
    )

    text = st.text_input(
        "请输入一句英文：",
        selected_example
    )

    # =========================
    # spaCy 依存句法
    # =========================
    if text:

        doc = nlp(text)

        col1, col2 = st.columns(2)

        # =========================
        # 左侧：Dependency Parsing
        # =========================
        with col1:

            st.subheader("🔗 依存句法（Dependency Parsing）")

            svg = displacy.render(
                doc,
                style="dep",
                jupyter=False
            )

            st.components.v1.html(
                svg,
                height=400,
                scrolling=True
            )

            st.markdown("### 说明")
            st.markdown("- 箭头表示词之间的依赖关系")
            st.markdown("- ROOT 是句子的核心动词")

        # =========================
        # 右侧：CFG 成分句法
        # =========================
        with col2:

            st.subheader("🌳 成分句法（Constituency Parsing）")

            grammar = CFG.fromstring("""
                S -> NP VP
                VP -> V NP | VP PP
                PP -> P NP
                NP -> Det N | NP PP
                V -> "saw"
                Det -> "The" | "the"
                N -> "boy" | "man" | "telescope"
                P -> "with"
            """)

            parser = ChartParser(grammar)

            tokens = text.replace(".", "").split()

            try:

                trees = list(parser.parse(tokens))

                if len(trees) == 0:

                    st.warning("该句子不符合当前CFG规则（教学简化版）")

                else:

                    st.success(
                        f"检测到 {len(trees)} 种句法结构（存在歧义）"
                    )

                    for i, tree in enumerate(trees):

                        st.markdown(f"### 结构 {i + 1}")

                        st.code(
                            tree,
                            language="text"
                        )

            except Exception as e:

                st.error(f"解析失败：{e}")

        # =========================
        # 歧义解释
        # =========================
        st.markdown("---")

        st.subheader("🧠 歧义解释（Ambiguity Explanation）")

        if "with the telescope" in text:

            st.markdown("""
### 两种理解：

1. The boy saw [the man with the telescope]  
👉 男人拿着望远镜

2. The boy saw the man [with the telescope]  
👉 男孩用望远镜观察

这属于经典的：

PP attachment ambiguity（介词短语附着歧义）
            """)

    # =========================
    # 技术说明
    # =========================
    st.markdown("---")

    st.subheader("📌 技术说明")

    st.markdown("""
- 左侧：spaCy Dependency Parsing
- 右侧：NLTK CFG Parsing
- 使用 ChartParser 实现歧义分析
- 不依赖大型深度学习模型
    """)