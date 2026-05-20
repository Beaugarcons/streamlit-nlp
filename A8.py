import os
import streamlit as st
from transformers import pipeline
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from deep_translator import GoogleTranslator
import nltk

def run():

    # 1. 设置 Hugging Face 国内镜像源
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # 2. 安全检查并加载 NLTK 分词组件，避免重复下载引发平台卡顿
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass

    # -----------------------------
    # 缓存模型（避免重复加载）
    # -----------------------------
    @st.cache_resource
    def load_nmt():
        try:
            # 注意：该模型大小约 300MB+，在云端加载时必须确保安装了 torch 和 sentencepiece 库
            translator_pipeline = pipeline(
                task="translation",
                model="Helsinki-NLP/opus-mt-en-zh"
            )
            return translator_pipeline
        except Exception as e:
            # 捕获异常，返回 None 触发下方的优雅降级保护机制
            return None

    translator = load_nmt()

    # -----------------------------
    # 核心修复：统一的 NMT 翻译预测函数（实现优雅降级防护）
    # -----------------------------
    def predict_nmt(text):
        if not text.strip():
            return ""
        
        # 方案 A：如果本地 NMT 模型成功加载，优先使用本地 Hugging Face 模型
        if translator is not None:
            try:
                result = translator(text)
                return result[0]['translation_text']
            except Exception as e:
                st.warning(f"⚠️ 本地 NMT 模型推理失败 ({e})，正在为您自动切换至备用云端翻译引擎...")
        
        # 方案 B（备用防崩）：当云端环境因内存不足(OOM)导致模型无法加载时，使用在线全句翻译平替
        try:
            return GoogleTranslator(source='en', target='zh-CN').translate(text)
        except Exception as e:
            return f"[翻译引擎异常: {e}]"

    # -----------------------------
    # 简单规则翻译（逐词直译）
    # -----------------------------
    def rule_based_translate(text):
        words = text.split()
        translated = []

        for w in words:
            try:
                # 去除单词前后的常用标点符号干扰，使单词直译更准确
                clean_w = w.strip(".,!?;:\"'")
                if clean_w:
                    zh = GoogleTranslator(source='en', target='zh-CN').translate(clean_w)
                else:
                    zh = w
            except:
                zh = w
            translated.append(zh)

        return " ".join(translated)

    # -----------------------------
    # 页面结构
    # -----------------------------
    st.title("机器翻译机制对比与评测平台")

    # 如果本地模型由于 OOM 或依赖环境不就绪未加载成功，在页面顶部给用户一个友好的状态提示
    if translator is None:
        st.info("💡 **平台运行状态提示**：检测到当前云端服务器内存紧张 (OOM) 或未配置 `sentencepiece` 依赖，本地大模型未能加载成功。为了保障您的完整实验体验，系统已自动启用**在线高级翻译引擎**来模拟神经机器翻译（NMT）效果，您可以继续正常进行所有对比与评测。")

    tab1, tab2, tab3 = st.tabs([
        "NMT翻译引擎",
        "规则翻译 vs NMT",
        "BLEU评测"
    ])

    # =============================
    # 模块1：NMT翻译
    # =============================
    with tab1:
        st.subheader("神经机器翻译（NMT）")

        # 示例按钮
        if st.button("使用示例句", key="btn_ex1"):
            st.session_state["text1"] = "It rains cats and dogs."

        text = st.text_area("请输入英文句子", key="text1")

        if st.button("开始翻译", key="nmt"):
            if text.strip() != "":
                with st.spinner("模型翻译中..."):
                    translation = predict_nmt(text)

                st.success("翻译结果：")
                st.write(translation)

    # =============================
    # 模块2：规则 vs NMT
    # =============================
    with tab2:
        st.subheader("规则翻译 vs 神经翻译")

        # 示例按钮
        if st.button("使用示例句（对比）", key="btn_ex2"):
            st.session_state["text2"] = "It rains cats and dogs."

        text2 = st.text_area("请输入英文句子", key="text2")

        if st.button("对比翻译", key="btn_compare"):
            if text2.strip() != "":
                with st.spinner("生成中..."):
                    nmt_result = predict_nmt(text2)
                    rule_result = rule_based_translate(text2)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**规则直译（逐词）**")
                    st.write(rule_result)

                with col2:
                    st.markdown("**神经翻译（NMT/全句）**")
                    st.write(nmt_result)

    # =============================
    # 模块3：BLEU评测
    # =============================
    with tab3:
        st.subheader("BLEU自动评测")

        # -----------------------------
        # 示例按钮
        # -----------------------------
        if st.button("使用示例句（It rains cats and dogs.）", key="btn_ex3"):
            st.session_state["src"] = "It rains cats and dogs."
            st.session_state["ref1"] = "倾盆大雨"
            st.session_state["ref2"] = "雨 大 倾盆"
            st.session_state["ref3"] = "下着大雨"

        # -----------------------------
        # 输入区域
        # -----------------------------
        source = st.text_area("英文原文", key="src")
        reference1 = st.text_area("参考译文1（标准）", key="ref1")
        reference2 = st.text_area("参考译文2（语序错误）", key="ref2")
        reference3 = st.text_area("参考译文3（同义表达）", key="ref3")

        # Candidate 译文生成
        if st.button("生成机器译文", key="btn_gen_cand"):
            if source.strip():
                with st.spinner("生成中..."):
                    st.session_state["cand"] = predict_nmt(source)

        candidate = st.text_area("机器译文（Candidate）", key="cand")

        # -----------------------------
        # BLEU计算
        # -----------------------------
        def compute_bleu(ref, cand):
            if not ref or not cand:
                return 0.0

            ref_tokens = list(ref)
            cand_tokens = list(cand)

            smoothie = SmoothingFunction().method1

            score = sentence_bleu(
                [ref_tokens],
                cand_tokens,
                smoothing_function=smoothie
            )
            return round(score, 4)

        if st.button("计算BLEU", key="btn_calc_bleu"):
            score1 = compute_bleu(reference1, candidate)
            score2 = compute_bleu(reference2, candidate)
            score3 = compute_bleu(reference3, candidate)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**标准参考**")
                st.write(reference1)
                st.write(f"BLEU: `{score1}`")

            with col2:
                st.markdown("**语序错误**")
                st.write(reference2)
                st.write(f"BLEU: `{score2}`")

            with col3:
                st.markdown("**同义表达**")
                st.write(reference3)
                st.write(f"BLEU: `{score3}`")

            # -----------------------------
            # 自动解释（动态）
            # -----------------------------
            st.markdown("### 结果分析")

            st.markdown(f"""
            * **标准参考 BLEU** = `{score1}`  
            * **语序错误 BLEU** = `{score2}`  
            * **同义表达 BLEU** = `{score3}`  

            **通过数据可以观察到：**

            1. **如果语序错误但词汇相同**，BLEU 值仍可能较高（因为字面上的 n-gram 片段依然能够部分匹配成功）。  
            2. **如果语义完全正确但表达方式不同（如同义句）**，BLEU 值反而可能较低。  

            **这揭示了 BLEU 指标的经典优缺点：**

            * **优点**：能够完全自动化进行、计算速度极快、具备极高的客观可量化性。  
            * **缺点**：不具备真正的语义理解能力，完全依赖字面层面的严格匹配。  
            """)