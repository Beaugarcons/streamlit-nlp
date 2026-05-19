import streamlit as st

st.set_page_config(
    page_title="NLP实验平台",
    layout="wide"
)

st.sidebar.title("NLP实验平台")

page = st.sidebar.selectbox(
    "请选择模块",
    [
        "词汇分析",
        "句法分析",
        "语义分析",
        "语义分析2",
        "篇章分析",
        "语言模型",
        "信息抽取",
        "机器翻译",
        "情感分析"
    ]
)

if page == "词汇分析":
    from modules import A1
    A1.run()

elif page == "句法分析":
    from modules import A2
    A2.run()

elif page == "语义分析":
    from modules import A3
    A3.run()

elif page == "语义分析2":
    from modules import A4
    A4.run()

elif page == "篇章分析":
    from modules import A5
    A5.run()

elif page == "语言模型":
    from modules import A6
    A6.run()

elif page == "信息抽取":
    from modules import A7
    A7.run()

elif page == "机器翻译":
    from modules import A8
    A8.run()

elif page == "情感分析":
    from modules import A9
    A9.run()