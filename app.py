import streamlit as st

# ==========================================
# CRITICAL: set_page_config 必须是整个脚本的第一条 Streamlit 命令
# ==========================================
st.set_page_config(
    page_title="NLP 课程作业综合平台",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 动态/延迟导入模块以稍微缓解启动时的内存压力（Streamlit 推荐做法）
import A1
import A2
import A3
import A4
import A5
import A6
import A7
import A8
import A9

def main():
    st.sidebar.title("🧠 NLP 作业导航")
    st.sidebar.markdown("---")
    
    # 建立页面名称到对应运行函数的映射字典
    pages = {
        "🏠 首页 (Home)": None,
        "作业 1：词汇分析 (A1)": A1.run,
        "作业 2：句法分析 (A2)": A2.run,
        "作业 3：语义分析 (A3)": A3.run,
        "作业 4：语义分析2 (A4)": A4.run,
        "作业 5：篇章分析 (A5)": A5.run,
        "作业 6：语言模型 (A6)": A6.run,
        "作业 7：信息抽取 (A7)": A7.run,
        "作业 8：机器翻译 (A8)": A8.run,
        "作业 9：情感分析 (A9)": A9.run
    }
    
    # 侧边栏单选框导航
    selected_page = st.sidebar.radio(
        "请选择要查看的模块：",
        list(pages.keys())
    )

    st.sidebar.markdown("---")
    st.sidebar.info("📌 部署于 Streamlit Cloud")

    # 根据用户的选择渲染对应的页面
    if selected_page == "🏠 首页 (Home)":
        st.title("🚀 欢迎来到 NLP 课程作业综合平台")
        st.markdown("""
        本平台集成了自然语言处理（NLP）课程的全部 9 次作业演示。你可以在左侧边栏切换不同的模块，实时输入文本并查看算法或模型的输出结果。
        
        ### 📂 系统模块总览
        """)
        
        # 用两列并排展示，让首页显得更美观专业
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            * **A1：词汇分析** *🔎 中文分词、词性标注、规范化、繁简转换、拼音转换*
            * **A2：句法分析** *🌲 依存句法与成分句法分析可视化、歧义句解析*
            * **A3：语义分析** *📊 统计与分布式文本表示对比 (TF-IDF, Word2Vec, FastText)*
            * **A4：语义分析2** *🔍 词义消歧与语义角色标注 (Lesk 算法 vs BERT)*
            * **A5：篇章分析** *🔗 话语分割 (EDU)、浅层篇章分析与指代消解演示*
            """)
            
        with col2:
            st.markdown("""
            * **A6：语言模型** *🤖 N-gram 频率统计、掩码填空 (BERT)、文本生成 (GPT2) 与通顺度 (PPL) 评估*
            * **A7：信息抽取** *📍 命名实体识别 (NER)、底层 BIO 标注序列与实体关系抽取图谱*
            * **A8：机器翻译** *🌐 基于规则的逐词翻译对比统计机器翻译/深度学习翻译、自动化评测 (BLEU)*
            * **A9：情感分析** *📈 细粒度/批量文本情感分类、显式与隐式情感特征提取、舆情大屏可视化*
            """)
            
        st.markdown("---")
        st.caption("💡 提示：部分基于深度学习模型（如 A4, A6, A8, A9）的模块在第一次点击切换时，云端需要花费一点时间加载模型，请耐心等待。")
        
    else:
        # 动态调用对应作业文件的 run() 函数
        try:
            pages[selected_page]()
        except Exception as e:
            st.error(f"渲染页面时发生错误: {e}")
            st.info("这通常是由于云端内存不足(OOM)或依赖包未完全就绪导致的。如果反复出现，请刷新网页或尝试重新部署应用。")

if __name__ == "__main__":
    main()