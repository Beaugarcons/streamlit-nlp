import streamlit as st
import jieba
import jieba.posseg as pseg
from collections import Counter
import pandas as pd
import re
import unicodedata
from opencc import OpenCC
import numpy as np
from typing import List, Tuple, Dict, Any
import pypinyin
from pypinyin import Style, lazy_pinyin, pinyin, load_phrases_dict
import warnings
import plotly.express as px  # 引入 Plotly 替代 matplotlib

def run():

    warnings.filterwarnings('ignore')

    # 加载繁简体转换器
    t2s = OpenCC('t2s') # 繁转简
    s2t = OpenCC('s2t') # 简转繁

    # --- 增强函数定义 ---

    def enhanced_text_normalization(text):
        """增强的文本规范化（基于正则表达式和有限状态自动机思想）"""

        # 1. 构建规范化规则（类似有限状态转换机的规则集）
        normalization_rules = [
            # 规则1：处理重复标点（状态转移：多个相同标点 -> 单个标点）
            (r'[，。！？；：]{2,}', lambda m: m.group()[0]),

            # 规则2：处理数字连续（状态转移：中文数字标准化）
            (r'[一二三四五六七八九十]+', lambda m: m.group()),

            # 规则3：处理括号不匹配问题
            (r'（[^）]*$|\[[^\]]*$', lambda m: m.group().replace('（', '').replace('[', '')),

            # 规则4：去除特殊Unicode字符但保留中文标点
            (r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；："()《》【】、]', ''),

            # 规则5：标准化空格（多个空格变一个）
            (r'\s+', ' '),
        ]

        # 2. 应用规则序列（模拟有限状态转换机）
        result = text

        # 全角转半角（NFKC规范化）
        result = unicodedata.normalize('NFKC', result)

        # 繁转简
        result = t2s.convert(result)

        # 应用所有规则
        for pattern, replacement in normalization_rules:
            if callable(replacement):
                result = re.sub(pattern, replacement, result)
            else:
                result = re.sub(pattern, replacement, result)

        return result.strip()

    def segment_with_multiple_methods(text):
        """多种分词方法对比（展示不同分词原理）"""

        methods_results = {}

        # 方法1：Jieba默认分词（基于前缀词典和动态规划）
        seg_list = list(jieba.cut(text, cut_all=False))
        methods_results['Jieba精确模式'] = seg_list

        # 方法2：Jieba全模式（类似正向最大匹配的扩展）
        seg_list_full = list(jieba.cut(text, cut_all=True))
        methods_results['Jieba全模式'] = seg_list_full

        # 加载Jieba词典用于最大匹配算法
        jieba_dict = set()
        try:
            jieba_dict = set(jieba.dt.FREQ.keys())
        except:
            # 如果无法获取词典，使用一个简单的词典
            jieba_dict = {
                '一行', '行行', '不行', '要是', '都行', '干一行', '行一行',
                '一行行', '行行行', '干哪行', '哪行', '自然', '语言', '处理',
                '非常', '有趣', '领域'
            }

        # 方法3：模拟正向最大匹配（Forward Maximum Matching）
        def forward_max_match(sentence, max_len=4):
            """正向最大匹配算法"""
            result = []
            idx = 0

            while idx < len(sentence):
                matched = False
                # 从最大长度开始尝试匹配
                for length in range(min(max_len, len(sentence) - idx), 0, -1):
                    word = sentence[idx:idx + length]

                    # 检查是否在词典中或是单字
                    if length == 1 or word in jieba_dict:
                        result.append(word)
                        idx += length
                        matched = True
                        break

                # 如果没有匹配到，则取单个字符
                if not matched:
                    result.append(sentence[idx])
                    idx += 1

            return result

        # 方法4：模拟逆向最大匹配（Backward Maximum Matching）
        def backward_max_match(sentence, max_len=4):
            """逆向最大匹配算法"""
            result = []
            idx = len(sentence)

            while idx > 0:
                matched = False
                # 从最大长度开始尝试匹配
                for length in range(min(max_len, idx), 0, -1):
                    start = idx - length
                    word = sentence[start:idx]

                    # 检查是否在词典中或是单字
                    if length == 1 or word in jieba_dict:
                        result.insert(0, word)
                        idx = start
                        matched = True
                        break

                # 如果没有匹配到，则取单个字符
                if not matched:
                    result.insert(0, sentence[idx-1])
                    idx -= 1

            return result

        # 方法5：双向最大匹配（Bidirectional Maximum Matching）
        def bidirectional_max_match(sentence):
            """双向最大匹配算法"""
            forward = forward_max_match(sentence)
            backward = backward_max_match(sentence)

            # 选择分词数量少的（颗粒度更大的）
            if len(forward) < len(backward):
                return forward, '正向（词数少）'
            elif len(forward) > len(backward):
                return backward, '逆向（词数少）'
            else:
                # 分词数相同，选择单字少的
                forward_single = sum(1 for w in forward if len(w) == 1)
                backward_single = sum(1 for w in backward if len(w) == 1)
                if forward_single < backward_single:
                    return forward, '正向（单字少）'
                else:
                    return backward, '逆向（单字少）'

        # 执行各种方法
        methods_results['正向最大匹配'] = forward_max_match(text)
        methods_results['逆向最大匹配'] = backward_max_match(text)
        bidir_result, bidir_reason = bidirectional_max_match(text)
        methods_results[f'双向最大匹配（{bidir_reason}）'] = bidir_result

        # 方法6：基于Jieba的搜索引擎模式
        seg_list_search = list(jieba.cut_for_search(text))
        methods_results['Jieba搜索引擎模式'] = seg_list_search

        return methods_results

    def get_pos_tags_for_segmentation(segmented_text: List[str]) -> List[Tuple[str, str]]:
        """
        为分词结果进行词性标注
        使用Jieba的pseg.cut，但基于已分词的结果
        """
        # 将分词结果重新组合成文本
        combined_text = ''.join(segmented_text)

        # 使用Jieba进行词性标注
        pos_tags = []
        for word, pos in pseg.cut(combined_text):
            pos_tags.append((word, pos))

        return pos_tags

    def enhanced_pos_tagging_with_segmentation(text: str, segmentation: List[str]) -> Tuple[str, List[Dict], List[Tuple[str, str]]]:
        """
        增强的词性标注（基于指定的分词结果）
        """
        # 获取词性标注
        pos_tags = get_pos_tags_for_segmentation(segmentation)

        # 构建扩展的词性映射表（更详细的类别）
        pos_color_map = {
            # 主要词性类别
            'n': ('名词', '#FF6B6B', 'red'),        # 名词
            'v': ('动词', '#4ECDC4', 'blue'),       # 动词
            'a': ('形容词', '#45B7D1', 'green'),    # 形容词
            'd': ('副词', '#96CEB4', 'lightgreen'), # 副词
            'p': ('介词', '#FFEAA7', 'yellow'),     # 介词
            'c': ('连词', '#DDA0DD', 'purple'),     # 连词
            'u': ('助词', '#98D8C8', 'teal'),       # 助词
            'r': ('代词', '#F7DC6F', 'gold'),       # 代词
            'm': ('数词', '#BB8FCE', 'violet'),     # 数词
            'q': ('量词', '#85C1E9', 'lightblue'),  # 量词
        }

        # 生成带颜色标记的文本
        def get_pos_color(flag):
            """获取词性对应的颜色"""
            # 取第一个字母作为大类
            category = flag[0] if flag else 'x'
            if category in pos_color_map:
                return pos_color_map[category][2], pos_color_map[category][0]
            else:
                return 'gray', '其他'

        # 生成标注结果
        colored_parts = []
        detailed_info = []

        for word, flag in pos_tags:
            color, desc = get_pos_color(flag)
            colored_parts.append(f":{color}[{word} ({flag})]")
            detailed_info.append({
                '词语': word,
                '词性标签': flag,
                '词性说明': desc,
                '长度': len(word)
            })

        colored_text = " ".join(colored_parts)

        return colored_text, detailed_info, pos_tags

    def analyze_special_patterns(text):
        """分析文本中的特殊模式（如绕口令、成语等）"""
        patterns = []

        # 检测重复字符模式（类似词干提取的思想）
        # 查找连续重复的字符或词语
        repeat_patterns = re.findall(r'(.+?)\1+', text)
        if repeat_patterns:
            patterns.append({
                '类型': '重复模式',
                '模式': '字符/词语重复',
                '示例': list(set(repeat_patterns))[:3]
            })

        # 检测对称结构
        # 简单的回文检测
        clean_text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        if len(clean_text) >= 3 and clean_text == clean_text[::-1]:
            patterns.append({
                '类型': '对称结构',
                '模式': '回文',
                '示例': clean_text
            })

        # 检测成语模式（四字词语）
        idiom_pattern = re.findall(r'[\u4e00-\u9fa5]{4}', text)
        if idiom_pattern:
            # 去重并统计
            idiom_counter = Counter(idiom_pattern)
            patterns.append({
                '类型': '成语模式',
                '模式': '四字词语',
                '统计': dict(idiom_counter.most_common(5))
            })

        return patterns

    def compare_pos_distributions(pos_distributions: Dict[str, Dict[str, int]]) -> pd.DataFrame:
        """
        比较不同分词方法的词性分布
        """
        all_tags = set()
        for dist in pos_distributions.values():
            all_tags.update(dist.keys())

        # 创建比较表格
        comparison_data = []
        for tag in sorted(all_tags):
            tag_name = {
                'n': '名词', 'v': '动词', 'a': '形容词',
                'd': '副词', 'p': '介词', 'c': '连词',
                'u': '助词', 'r': '代词', 'm': '数词',
                'q': '量词', 'x': '其他'
            }.get(tag, tag)

            row = {'词性': tag_name}
            for method, dist in pos_distributions.items():
                row[method] = dist.get(tag, 0)
            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    # --- 主页面 ---

    st.title("🚀 NLP 中文文本分析助手")
    st.caption("基于多种NLP原理的增强文本分析工具 - 分词与词性标注联动分析")

    # 输入区域
    col_input, col_info = st.columns([3, 1])
    with col_input:
        input_text = st.text_area(
            "请输入要分析的文本（支持复杂文本如绕口令、诗词等）：", 
            height=150,
            value="干一行行一行，一行行行行行，行行行干哪行都行。要是不行，干一行不行一行，一行不行行行不行。"
        )

    with col_info:
        st.info("""
        **支持的原理：**
        - 有限状态自动机/转换机
        - 正则表达式模式匹配
        - 多种分词算法对比
        - 词性标注与可视化
        """)

    if input_text:
        # 处理进度指示器
        with st.spinner("正在分析文本..."):
            # --- 第1部分：文本规范化 ---
            st.header("📝 文本规范化分析")

            col_norm1, col_norm2 = st.columns(2)

            with col_norm1:
                st.subheader("原始文本")
                st.text(input_text)
                st.metric("字符数", len(input_text))

                # 字符类型统计
                char_types = {
                    "中文字符": len(re.findall(r'[\u4e00-\u9fa5]', input_text)),
                    "英文字母": len(re.findall(r'[a-zA-Z]', input_text)),
                    "数字": len(re.findall(r'\d', input_text)),
                    "标点符号": len(re.findall(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', input_text))
                }
                st.dataframe(pd.DataFrame(list(char_types.items()), columns=['类型', '数量']))

            with col_norm2:
                st.subheader("规范化后文本")
                normalized_text = enhanced_text_normalization(input_text)
                st.text(normalized_text)
                st.metric("规范化后字符数", len(normalized_text))

                # 显示规范化规则应用
                with st.expander("查看规范化规则应用详情"):
                    st.write("""
                    **应用的规则序列（有限状态转换机思想）：**
                    1. Unicode NFKC规范化（全角转半角）
                    2. 繁体转简体
                    3. 处理重复标点（状态转移）
                    4. 标准化空格
                    5. 清理特殊字符
                    """)

            # --- 第2部分：分词对比 ---
            st.header("🔪 分词方法对比")

            # 执行多种分词方法
            segmentation_results = segment_with_multiple_methods(normalized_text)

            # 存储所有方法的词性标注结果
            all_pos_results = {}
            all_pos_distributions = {}

            # 创建多个选项卡展示不同方法
            tabs = st.tabs(list(segmentation_results.keys()))

            for tab, (method_name, segmentation) in zip(tabs, segmentation_results.items()):
                with tab:
                    st.write(f"**{method_name}**")

                    # 显示分词结果
                    st.subheader("分词结果")
                    segmented_str = " | ".join(segmentation)
                    st.code(segmented_str)

                    # 统计信息
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("词数", len(segmentation))
                    with col_stat2:
                        avg_len = np.mean([len(w) for w in segmentation]) if segmentation else 0
                        st.metric("平均词长", f"{avg_len:.2f}")
                    with col_stat3:
                        single_chars = sum(1 for w in segmentation if len(w) == 1) if segmentation else 0
                        single_ratio = (single_chars/len(segmentation)*100) if segmentation else 0
                        st.metric("单字词比例", f"{single_ratio:.1f}%")

                    # 词性标注（基于当前分词结果）
                    st.subheader("词性标注结果")
                    colored_text, detailed_info, pos_result = enhanced_pos_tagging_with_segmentation(
                        normalized_text, segmentation
                    )

                    # 存储词性标注结果
                    all_pos_results[method_name] = {
                        'segmentation': segmentation,
                        'pos_result': pos_result,
                        'colored_text': colored_text,
                        'detailed_info': detailed_info
                    }

                    # 显示词性标注
                    st.markdown(colored_text)

                    # 词性分布统计
                    pos_distribution = {}
                    for _, flag in pos_result:
                        category = flag[0] if flag else 'x'
                        pos_distribution[category] = pos_distribution.get(category, 0) + 1

                    all_pos_distributions[method_name] = pos_distribution

                    # 显示词性分布
                    if pos_distribution:
                        st.subheader("词性分布")
                        pos_df = pd.DataFrame([
                            {'词性类别': k, '数量': v} 
                            for k, v in pos_distribution.items()
                        ])
                        st.dataframe(pos_df, use_container_width=True)

            # 分词方法原理说明
            with st.expander("📚 各分词方法原理说明"):
                st.markdown("""
                ### 分词算法原理对比

                1. **Jieba精确模式**
                   - **原理**: 基于前缀词典和动态规划（Viterbi算法）
                   - **特点**: 平衡准确率和召回率，最常用

                2. **Jieba全模式**
                   - **原理**: 类似正向最大匹配的扩展
                   - **特点**: 扫描所有可能的词语组合

                3. **正向最大匹配 (FMM)**
                   - **原理**: 从左到右，每次取最大可能长度的词
                   - **状态机**: 每次选择都基于当前状态和词典

                4. **逆向最大匹配 (BMM)**
                   - **原理**: 从右到左，每次取最大可能长度的词
                   - **特点**: 对中文后缀结构更敏感

                5. **双向最大匹配 (BI-MM)**
                   - **原理**: 综合FMM和BMM结果
                   - **决策**: 优先选词数少的结果

                6. **Jieba搜索引擎模式**
                   - **原理**: 在精确模式基础上对长词再切分
                   - **应用**: 适合搜索引擎索引
                """)

            # --- 第3部分：词性标注对比 ---
            st.header("🏷️ 词性标注对比分析")

            if all_pos_results:
                # 选择要对比的方法
                selected_methods = st.multiselect(
                    "选择要对比的分词方法：",
                    list(all_pos_results.keys()),
                    default=list(all_pos_results.keys())[:2]  # 默认选择前两种
                )

                if selected_methods:
                    # 创建多列对比
                    num_cols = min(len(selected_methods), 3)
                    cols = st.columns(num_cols)

                    for idx, method in enumerate(selected_methods):
                        if idx < num_cols:
                            with cols[idx % num_cols]:
                                st.subheader(f"{method}")

                                # 显示分词结果
                                st.write("分词：", " | ".join(all_pos_results[method]['segmentation'][:10]), 
                                        "..." if len(all_pos_results[method]['segmentation']) > 10 else "")

                                # 显示词性标注
                                st.markdown(all_pos_results[method]['colored_text'])

                                # 显示详细的词性统计
                                pos_df = pd.DataFrame(all_pos_results[method]['detailed_info'])
                                st.dataframe(pos_df, use_container_width=True, height=300)

                    # 词性分布对比图表
                    st.subheader("词性分布对比")

                    if len(selected_methods) >= 2:
                        # 创建对比数据
                        comparison_df = compare_pos_distributions(
                            {k: v for k, v in all_pos_distributions.items() if k in selected_methods}
                        )

                        # 显示对比表格
                        st.dataframe(comparison_df.set_index('词性'), use_container_width=True)

                        # 🛠️ 替换原本的 plt 堆叠柱状图，改用 Plotly
                        # 1. 将宽表格转换为 Plotly 识别的长表格 (Melt)
                        melted_df = comparison_df.melt(id_vars=['词性'], var_name='分词方法', value_name='数量')
                        
                        # 2. 颜色映射表
                        pos_color_map = {
                            '名词': '#FF6B6B', '动词': '#4ECDC4', '形容词': '#45B7D1',
                            '副词': '#96CEB4', '介词': '#FFEAA7', '连词': '#DDA0DD',
                            '助词': '#98D8C8', '代词': '#F7DC6F', '数词': '#BB8FCE',
                            '量词': '#85C1E9', '其他': '#CCCCCC'
                        }
                        
                        # 3. 绘制堆叠图
                        fig_stacked = px.bar(
                            melted_df, 
                            x='分词方法', 
                            y='数量', 
                            color='词性', 
                            title='不同分词方法的词性分布对比',
                            color_discrete_map=pos_color_map
                        )
                        fig_stacked.update_layout(xaxis_tickangle=-45, height=500)
                        st.plotly_chart(fig_stacked, use_container_width=True)
                    else:
                        st.info("请选择至少两种分词方法进行对比")

            # --- 第4部分：特殊模式分析 ---
            st.header("🔍 特殊模式检测")

            patterns = analyze_special_patterns(normalized_text)

            if patterns:
                for pattern in patterns:
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.success(pattern['类型'])
                        with col2:
                            st.write(f"**模式**: {pattern['模式']}")
                            if '示例' in pattern:
                                if isinstance(pattern['示例'], list):
                                    st.write(f"示例: {', '.join(pattern['示例'])}")
                                else:
                                    st.write(f"示例: {pattern['示例']}")
                            if '统计' in pattern:
                                for k, v in pattern['统计'].items():
                                    st.write(f"{k}: {v}次")
            else:
                st.info("未检测到明显的特殊语言模式")

            # --- 第5部分：综合统计 ---
            st.header("📊 综合统计信息")

            # 使用Jieba精确模式作为基准
            base_seg = segmentation_results.get('Jieba精确模式', [])

            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

            with col_stat1:
                st.metric("总字符数", len(input_text))
            with col_stat2:
                st.metric("分词数", len(base_seg))
            with col_stat3:
                unique_words = len(set(base_seg)) if base_seg else 0
                st.metric("唯一词数", unique_words)
            with col_stat4:
                ttr = unique_words / len(base_seg) if base_seg and len(base_seg) > 0 else 0
                st.metric("型例比(TTR)", f"{ttr:.2%}")

            # 词频统计
            if base_seg:
                word_freq = Counter(base_seg)
                top_words = word_freq.most_common(10)

                if top_words:
                    st.subheader("Top 10 高频词")
                    freq_df = pd.DataFrame(top_words, columns=['词语', '频次'])

                    col_chart, col_table = st.columns([2, 1])

                    with col_chart:
                        # 🛠️ 替换原本的 plt 条形图，改用 Plotly
                        fig_bar = px.bar(
                            freq_df, 
                            x='频次', 
                            y='词语', 
                            orientation='h', 
                            title='Top 10 高频词',
                            color_discrete_sequence=['#85C1E9']
                        )
                        # 让频次最高的词在顶部显示
                        fig_bar.update_layout(
                            yaxis={'categoryorder': 'total ascending'},
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=350
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                    with col_table:
                        st.dataframe(freq_df, use_container_width=True)

            # --- 第6部分：原理展示 ---
            with st.expander("🎓 NLP 原理详解"):
                st.markdown("""
                ## 分词与词性标注的联动关系

                ### 分词对词性标注的影响
                1. **粒度影响**：
                   - 不同分词粒度会导致不同的词语边界
                   - 例如："自然语言处理" 可以切分为：
                     - 大粒度：自然语言处理 (一个专有名词)
                     - 小粒度：自然 | 语言 | 处理 (三个普通名词/动词)

                2. **词性标注依赖分词**：
                   - 词性标注器基于分词结果进行分析
                   - 分词错误会导致词性标注错误
                   - 例如："一行行" 的标注：
                     - 切分为"一行|行"：数量词 + 动词
                     - 切分为"一行行"：可能被标注为副词

                3. **上下文影响**：
                   - 词性标注考虑词语上下文
                   - 相同词语在不同上下文中可能有不同词性
                   - 例如："行" 在不同位置可能是动词或量词

                ### 不同分词方法的词性标注差异
                | 分词方法 | 特点 | 对词性标注的影响 |
                |---------|------|----------------|
                | 精确模式 | 平衡粒度 | 词性标注最稳定 |
                | 全模式 | 过切分 | 可能产生更多单字词 |
                | 最大匹配 | 词典驱动 | 依赖词典覆盖度 |
                | 搜索引擎模式 | 细粒度 | 适合未登录词识别 |

                ### 词性标注技术对比
                1. **基于规则的方法**
                   - 优点：可解释性强
                   - 缺点：覆盖有限，难以处理歧义

                2. **统计方法 (HMM/CRF)**
                   - 优点：可处理未登录词
                   - 缺点：需要大量标注数据

                3. **深度学习方法**
                   - 优点：准确率高
                   - 缺点：计算资源需求大

                ### 评估指标
                - **分词准确率**：Precision, Recall, F1
                - **词性标注准确率**：POS Accuracy
                - **联合评估**：分词和词性标注的整体性能
                """)

    else:
        st.info("👆 请在上方输入框中输入文本以开始分析")

    # --- 侧边栏：配置选项 ---
    with st.sidebar:
        st.header("⚙️ 分析配置")

        st.subheader("分词设置")
        use_hmm = st.checkbox("使用HMM识别新词", value=True)

        st.subheader("词性标注设置")
        show_pos_details = st.checkbox("显示详细词性信息", value=True)

        st.subheader("显示选项")
        show_comparison = st.checkbox("显示对比分析", value=True)

        st.subheader("示例文本")
        example_selected = st.selectbox(
            "快速加载示例",
            ["绕口令", "新闻片段", "古诗", "技术文档", "自定义"]
        )

        if st.button("加载示例"):
            examples = {
                "绕口令": "干一行行一行，一行行行行行，行行行干哪行都行。要是不行，干一行不行一行，一行不行行行不行。",
                "新闻片段": "人工智能技术正在快速发展，改变着我们的生活方式和工作模式。",
                "古诗": "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
                "技术文档": "Python是一种解释型、面向对象、动态数据类型的高级程序设计语言。"
            }
            if example_selected in examples:
                st.rerun()

    # 添加自定义CSS样式
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 5px 5px 0px 0px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4CAF50;
            color: white;
        }
        .highlight-box {
            background-color: #f8f9fa;
            border-left: 4px solid #4CAF50;
            padding: 10px;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)