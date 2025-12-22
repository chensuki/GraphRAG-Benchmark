"""
下载并转换 HotpotQA 数据集
控制规模比 Medical 数据集小
"""
import json
import os
from pathlib import Path
from datasets import load_dataset
from typing import List, Dict, Any
import pandas as pd

# 目标规模（比 Medical 小）
TARGET_CORPUS_SIZE = 50  # 语料库文档数（Medical 约 100+）
TARGET_QUESTIONS_SIZE = 200  # 问题数（Medical 约 500+）

def check_medical_size():
    """检查 Medical 数据集规模"""
    try:
        base_dir = Path(__file__).parent.parent
        corpus_path = base_dir / 'Datasets/Corpus/medical.parquet'
        questions_path = base_dir / 'Datasets/Questions/medical_questions.parquet'
        
        if not corpus_path.exists() or not questions_path.exists():
            print(f"⚠️  Medical 数据集文件不存在，使用默认目标规模")
            return None, None
            
        corpus = pd.read_parquet(corpus_path)
        questions = pd.read_parquet(questions_path)
        print(f"📊 Medical 数据集规模:")
        print(f"  语料库文档数: {len(corpus)}")
        print(f"  问题数: {len(questions)}")
        print(f"  平均文档长度: {corpus['context'].str.len().mean():.0f} 字符")
        return len(corpus), len(questions)
    except Exception as e:
        print(f"⚠️  无法读取 Medical 数据集: {e}")
        return None, None

def download_hotpotqa():
    """下载 HotpotQA 数据集"""
    print("\n📥 开始下载 HotpotQA 数据集...")
    
    try:
        # 下载 dev 集（比 train 集小，适合测试）
        dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
        print(f"✅ 下载完成: {len(dataset)} 条数据")
        return dataset
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n💡 提示: 如果下载失败，可以手动下载:")
        print("   pip install datasets")
        print("   或访问: https://huggingface.co/datasets/hotpot_qa")
        return None

def convert_hotpotqa_to_framework_format(
    dataset,
    max_corpus: int = TARGET_CORPUS_SIZE,
    max_questions: int = TARGET_QUESTIONS_SIZE
):
    """
    将 HotpotQA 转换为评估框架格式
    
    Args:
        dataset: HotpotQA 数据集
        max_corpus: 最大语料库文档数
        max_questions: 最大问题数
    """
    print(f"\n🔄 开始转换 HotpotQA 数据集...")
    print(f"   目标规模: {max_corpus} 个文档, {max_questions} 个问题")
    
    # 创建输出目录（相对于项目根目录）
    base_dir = Path(__file__).parent.parent
    corpus_dir = base_dir / "Datasets/Corpus"
    questions_dir = base_dir / "Datasets/Questions"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    questions_dir.mkdir(parents=True, exist_ok=True)
    
    # 限制问题数量
    sample_size = min(len(dataset), max_questions)
    print(f"   采样 {sample_size} 个问题...")
    
    # 使用 select 方法安全地获取子集
    try:
        if hasattr(dataset, 'select'):
            dataset_subset = dataset.select(range(sample_size))
        else:
            dataset_subset = dataset[:sample_size]
    except Exception as e:
        print(f"   ⚠️  使用 select 失败，尝试直接迭代: {e}")
        dataset_subset = dataset
    
    # 第一步：先收集所有文档（不限制数量）
    print(f"   第一步：收集所有文档...")
    all_documents = {}  # {title: {'text': str, 'sentences': list}}
    debug_info = {'context_types': [], 'title_count': 0, 'empty_context': 0}
    
    for i in range(sample_size):
        try:
            item = dataset_subset[i]
        except (IndexError, KeyError) as e:
            print(f"   ⚠️  无法获取索引 {i} 的数据: {e}")
            break
        
        if not isinstance(item, dict):
            continue
        
        # 收集上下文文档
        context = item.get('context', [])
        
        # 调试：记录前几个 context 的格式
        if i < 3:
            print(f"   [调试] 问题 {i+1} 的 context 类型: {type(context)}")
            if isinstance(context, dict):
                print(f"   [调试] Context 是字典，键数量: {len(context)}")
                print(f"   [调试] 所有键: {list(context.keys())}")
                # 检查是否是特殊的 {'title': [...], 'sentences': [...]} 格式
                if 'title' in context and 'sentences' in context:
                    print(f"   [调试] 检测到特殊格式: {{'title': [...], 'sentences': [...]}}")
                    print(f"   [调试] title 类型: {type(context['title'])}, 长度: {len(context['title']) if hasattr(context['title'], '__len__') else 'N/A'}")
                    print(f"   [调试] sentences 类型: {type(context['sentences'])}, 长度: {len(context['sentences']) if hasattr(context['sentences'], '__len__') else 'N/A'}")
                    if isinstance(context['title'], list) and len(context['title']) > 0:
                        print(f"   [调试] 第一个标题: {context['title'][0]}")
                    if isinstance(context['sentences'], list) and len(context['sentences']) > 0:
                        print(f"   [调试] 第一个句子列表类型: {type(context['sentences'][0])}, 长度: {len(context['sentences'][0]) if hasattr(context['sentences'][0], '__len__') else 'N/A'}")
            elif isinstance(context, list):
                print(f"   [调试] Context 是列表，长度: {len(context)}")
                if len(context) > 0:
                    print(f"   [调试] 第一个元素类型: {type(context[0])}")
                    if isinstance(context[0], (list, tuple)) and len(context[0]) >= 2:
                        print(f"   [调试] 第一个元素: 标题='{context[0][0]}', 句子数={len(context[0][1]) if isinstance(context[0][1], list) else 'N/A'}")
        
        if not context:
            debug_info['empty_context'] += 1
            continue
        
        # HotpotQA 的 context 可能是列表或字典格式
        context_list = []
        if isinstance(context, list):
            # 列表格式: [[title, [sentences]], ...]
            context_list = context
        elif isinstance(context, dict):
            # 检查是否是特殊的 {'title': [...], 'sentences': [...]} 格式
            if 'title' in context and 'sentences' in context:
                # 特殊格式: {'title': [title1, title2, ...], 'sentences': [[sent1, sent2, ...], [sent3, sent4, ...], ...]}
                titles = context['title'] if isinstance(context['title'], list) else []
                sentences_list = context['sentences'] if isinstance(context['sentences'], list) else []
                
                # 配对标题和句子列表
                if len(titles) == len(sentences_list):
                    for title, sentences in zip(titles, sentences_list):
                        if isinstance(sentences, list):
                            context_list.append([title, sentences])
                        elif isinstance(sentences, str):
                            context_list.append([title, [sentences]])
                        else:
                            try:
                                sentences_list_item = list(sentences) if hasattr(sentences, '__iter__') else [str(sentences)]
                                context_list.append([title, sentences_list_item])
                            except:
                                continue
                else:
                    print(f"   [警告] 问题 {i+1}: title 和 sentences 长度不匹配 ({len(titles)} vs {len(sentences_list)})")
            else:
                # 标准字典格式: {title: [sentences], ...}
                # 转换为列表格式
                for title, sentences in context.items():
                    if isinstance(sentences, list):
                        context_list.append([title, sentences])
                    elif isinstance(sentences, str):
                        context_list.append([title, [sentences]])
                    else:
                        # 尝试转换其他格式
                        try:
                            sentences_list = list(sentences) if hasattr(sentences, '__iter__') else [str(sentences)]
                            context_list.append([title, sentences_list])
                        except:
                            continue
        else:
            continue
            
        # 收集文档到 all_documents
        for ctx_item in context_list:
            if not isinstance(ctx_item, (list, tuple)) or len(ctx_item) < 2:
                continue
                
            title = ctx_item[0]  # 文档标题
            sentences = ctx_item[1]  # 句子列表
            
            # 确保 title 是字符串
            if not isinstance(title, str):
                title = str(title)
            
            # 处理 sentences，确保是列表
            if not isinstance(sentences, list):
                if isinstance(sentences, str):
                    sentences = [sentences]
                elif hasattr(sentences, '__iter__'):
                    sentences = list(sentences)
                else:
                    sentences = [str(sentences)]
            
            # 过滤空句子
            sentences = [str(s).strip() for s in sentences if s and str(s).strip()]
            
            if not sentences:
                continue
            
            # 构建完整文档文本
            doc_text = " ".join(sentences)
            
            if not doc_text or not doc_text.strip():
                continue
            
            # 保存文档（去重，但保留所有文档）
            if title and title not in all_documents:
                all_documents[title] = {
                    'text': doc_text,
                    'sentences': sentences  # 保存句子列表用于证据提取
                }
                debug_info['title_count'] += 1
                if len(all_documents) <= 5:  # 打印前5个文档的标题
                    print(f"   [调试] 收集到文档 {len(all_documents)}: '{title}' ({len(sentences)} 个句子)")
    
    print(f"   ✅ 收集到 {len(all_documents)} 个唯一文档")
    print(f"   [调试] 总文档标题数: {debug_info['title_count']}, 空 context 数: {debug_info['empty_context']}")
    if len(all_documents) <= 10:
        print(f"   [调试] 所有文档标题: {list(all_documents.keys())}")
    
    # 第二步：处理问题并提取证据
    print(f"   第二步：处理问题并提取证据...")
    questions_list = []
    
    for i in range(sample_size):
        try:
            item = dataset_subset[i]
        except (IndexError, KeyError) as e:
            break
        
        if i % 50 == 0:
            print(f"   处理进度: {i}/{sample_size}")
        
        # 检查 item 类型
        if not isinstance(item, dict):
            continue
        
        # 提取问题信息
        question_id = f"HotpotQA-{item.get('id', str(i))}"
        question_text = item.get('question', '')
        answer = item.get('answer', '')
        
        if not question_text or not answer:
            continue
        
        # 提取支持事实作为证据
        supporting_facts = item.get('supporting_facts', [])
        evidence_sentences = []
        
        # 调试：打印前几个问题的 supporting_facts 信息
        if i < 3:
            print(f"   [调试] 问题 {i+1} 的 supporting_facts 类型: {type(supporting_facts)}")
            if isinstance(supporting_facts, dict):
                print(f"   [调试] 问题 {i+1} 的 supporting_facts 是字典，键: {list(supporting_facts.keys())}")
                print(f"   [调试] 问题 {i+1} 的 supporting_facts 完整内容: {supporting_facts}")
                for key, value in list(supporting_facts.items())[:5]:
                    print(f"   [调试]   键 '{key}': 类型={type(value)}")
                    if isinstance(value, list):
                        print(f"   [调试]     值长度: {len(value)}, 前3个: {value[:3]}")
                    else:
                        print(f"   [调试]     值: {str(value)[:200]}")
            elif isinstance(supporting_facts, list):
                print(f"   [调试] 问题 {i+1} 的 supporting_facts 是列表，长度: {len(supporting_facts)}")
                if len(supporting_facts) > 0:
                    print(f"   [调试]   前3个: {supporting_facts[:3]}")
                    print(f"   [调试]   第一个元素类型: {type(supporting_facts[0])}")
            else:
                print(f"   [调试] 问题 {i+1} 的 supporting_facts 是其他类型: {supporting_facts}")
        
        # 从 all_documents 中提取证据
        # 处理列表格式: [[title, idx], ...]
        if isinstance(supporting_facts, list) and len(supporting_facts) > 0:
            for fact in supporting_facts:
                if isinstance(fact, (list, tuple)) and len(fact) >= 2:
                    fact_title = str(fact[0]).strip()  # 文档标题
                    sentence_idx = fact[1]  # 句子索引
                    
                    # 调试：打印匹配过程
                    if i < 3:
                        print(f"   [调试] 查找证据: 标题='{fact_title}', 索引={sentence_idx}")
                    
                    # 在 all_documents 中查找匹配的文档
                    matched_doc = None
                    matched_title = None
                    for doc_title, doc_data in all_documents.items():
                        doc_title_clean = doc_title.strip()
                        if doc_title_clean == fact_title or doc_title == fact_title:
                            matched_doc = doc_data
                            matched_title = doc_title
                            break
                    
                    if i < 3:
                        if matched_doc:
                            print(f"   [调试] ✅ 找到匹配文档: '{matched_title}'")
                        else:
                            print(f"   [调试] ❌ 未找到匹配文档，可用文档: {list(all_documents.keys())[:5]}")
                    
                    if matched_doc and isinstance(sentence_idx, int):
                        sentences = matched_doc['sentences']
                        if 0 <= sentence_idx < len(sentences):
                            evidence_sentences.append(sentences[sentence_idx])
                            if i < 3:
                                print(f"   [调试] ✅ 提取证据: '{sentences[sentence_idx][:80]}...'")
                        elif i < 3:
                            print(f"   [调试] ❌ 索引超出范围: {sentence_idx} >= {len(sentences)}")
        
        # 处理字典格式: {title: [idx1, idx2, ...]} 或 {'title': [...], 'sent_id': [...]} 或 {'title': [...], 'sent_idx': [...]}
        elif isinstance(supporting_facts, dict):
            # 检查是否是特殊格式 {'title': [...], 'sent_id': [...]} 或 {'title': [...], 'sent_idx': [...]}
            # 注意：HotpotQA 使用的是 'sent_id' 而不是 'sent_idx'
            if 'title' in supporting_facts and ('sent_id' in supporting_facts or 'sent_idx' in supporting_facts):
                titles = supporting_facts['title'] if isinstance(supporting_facts['title'], list) else []
                # 优先使用 'sent_id'，如果没有则使用 'sent_idx'
                sent_indices = supporting_facts.get('sent_id') or supporting_facts.get('sent_idx')
                if not isinstance(sent_indices, list):
                    sent_indices = []
                
                if i < 3:
                    key_name = 'sent_id' if 'sent_id' in supporting_facts else 'sent_idx'
                    print(f"   [调试] 检测到特殊格式: {{'title': [...], '{key_name}': [...]}}")
                    print(f"   [调试]   title 长度: {len(titles)}, {key_name} 长度: {len(sent_indices)}")
                
                # 配对标题和句子索引
                if len(titles) == len(sent_indices):
                    for fact_title, sentence_idx in zip(titles, sent_indices):
                        fact_title = str(fact_title).strip()
                        
                        if i < 3:
                            print(f"   [调试] 查找证据: 标题='{fact_title}', 索引={sentence_idx}")
                        
                        # 在 all_documents 中查找匹配的文档
                        matched_doc = None
                        matched_title = None
                        for doc_title, doc_data in all_documents.items():
                            doc_title_clean = doc_title.strip()
                            if doc_title_clean == fact_title or doc_title == fact_title:
                                matched_doc = doc_data
                                matched_title = doc_title
                                break
                        
                        if i < 3:
                            if matched_doc:
                                print(f"   [调试] ✅ 找到匹配文档: '{matched_title}'")
                            else:
                                print(f"   [调试] ❌ 未找到匹配文档")
                        
                        if matched_doc and isinstance(sentence_idx, int):
                            sentences = matched_doc['sentences']
                            if 0 <= sentence_idx < len(sentences):
                                evidence_sentences.append(sentences[sentence_idx])
                                if i < 3:
                                    print(f"   [调试] ✅ 提取证据: '{sentences[sentence_idx][:80]}...'")
            else:
                # 标准字典格式: {title: [idx1, idx2, ...]}
                for fact_title, indices in supporting_facts.items():
                    fact_title = str(fact_title).strip()
                    
                    # indices 可能是单个索引或索引列表
                    if not isinstance(indices, list):
                        indices = [indices] if indices is not None else []
                    
                    if i < 3:
                        print(f"   [调试] 查找证据: 标题='{fact_title}', 索引列表={indices}")
                    
                    # 在 all_documents 中查找匹配的文档
                    matched_doc = None
                    matched_title = None
                    for doc_title, doc_data in all_documents.items():
                        doc_title_clean = doc_title.strip()
                        if doc_title_clean == fact_title or doc_title == fact_title:
                            matched_doc = doc_data
                            matched_title = doc_title
                            break
                    
                    if matched_doc:
                        sentences = matched_doc['sentences']
                        for sentence_idx in indices:
                            if isinstance(sentence_idx, int) and 0 <= sentence_idx < len(sentences):
                                evidence_sentences.append(sentences[sentence_idx])
                                if i < 3:
                                    print(f"   [调试] ✅ 提取证据: '{sentences[sentence_idx][:80]}...'")
        
        # 合并证据
        evidence = " ".join(evidence_sentences) if evidence_sentences else ""
        
        # 判断问题类型（HotpotQA 主要是复杂推理）
        question_type = "Complex Reasoning"  # HotpotQA 需要多跳推理
        
        # 构建问题项
        question_item = {
            "id": question_id,
            "source": "HotpotQA",  # 所有问题共享同一个语料库
            "question": question_text,
            "answer": answer,
            "question_type": question_type,
            "evidence": evidence
        }
        questions_list.append(question_item)
    
    # 第三步：从所有文档中选择目标数量的文档
    print(f"   第三步：选择 {max_corpus} 个文档...")
    
    # 选择文档：优先选择在 supporting_facts 中出现的文档
    selected_titles = set()
    for i in range(sample_size):
        try:
            item = dataset_subset[i]
            if not isinstance(item, dict):
                continue
            supporting_facts = item.get('supporting_facts', [])
            if isinstance(supporting_facts, list):
                for fact in supporting_facts:
                    if isinstance(fact, (list, tuple)) and len(fact) >= 2:
                        fact_title = str(fact[0]).strip()
                        # 在 all_documents 中查找匹配的标题
                        for doc_title in all_documents.keys():
                            if doc_title.strip() == fact_title or doc_title == fact_title:
                                selected_titles.add(doc_title)
                                if len(selected_titles) >= max_corpus:
                                    break
                        if len(selected_titles) >= max_corpus:
                            break
            if len(selected_titles) >= max_corpus:
                break
        except:
            continue
    
    # 如果还不够，随机补充
    if len(selected_titles) < max_corpus:
        import random
        remaining_titles = [t for t in all_documents.keys() if t not in selected_titles]
        needed = max_corpus - len(selected_titles)
        if len(remaining_titles) > 0:
            selected_titles.update(random.sample(remaining_titles, min(needed, len(remaining_titles))))
    
    # 构建最终语料库
    corpus_dict = {title: all_documents[title]['text'] for title in selected_titles}
    print(f"   ✅ 最终选择了 {len(corpus_dict)} 个文档")
    
    # 如果文档数仍然超过目标，进行最终筛选
    if len(corpus_dict) > max_corpus:
        import random
        selected_titles_list = random.sample(list(selected_titles), max_corpus)
        corpus_dict = {title: all_documents[title]['text'] for title in selected_titles_list}
        print(f"   ⚠️  最终筛选到 {len(corpus_dict)} 个文档")
    
    # 构建语料库数据
    corpus_data = []
    for title, context in corpus_dict.items():
        corpus_data.append({
            "corpus_name": "HotpotQA",
            "context": context
        })
    
    # 限制问题数量
    questions_list = questions_list[:max_questions]
    
    # 保存语料库
    corpus_file = corpus_dir / "hotpotqa.parquet"
    
    if len(corpus_data) == 0:
        print(f"\n❌ 错误: 没有提取到任何文档！")
        print(f"   请检查 HotpotQA 数据集的格式")
        return None, None
    
    corpus_df = pd.DataFrame(corpus_data)
    corpus_df.to_parquet(corpus_file, index=False)
    print(f"\n✅ 语料库已保存: {corpus_file}")
    print(f"   文档数: {len(corpus_data)}")
    if len(corpus_data) > 0:
        print(f"   平均长度: {corpus_df['context'].str.len().mean():.0f} 字符")
    
    # 保存问题
    questions_file = questions_dir / "hotpotqa_questions.json"
    with open(questions_file, 'w', encoding='utf-8') as f:
        json.dump(questions_list, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 问题集已保存: {questions_file}")
    print(f"   问题数: {len(questions_list)}")
    
    # 统计信息
    print(f"\n📊 转换完成统计:")
    print(f"   语料库文档数: {len(corpus_data)}")
    print(f"   问题数: {len(questions_list)}")
    print(f"   平均证据长度: {sum(len(q['evidence']) for q in questions_list) / len(questions_list):.0f} 字符")
    print(f"   问题类型分布:")
    from collections import Counter
    q_types = Counter(q['question_type'] for q in questions_list)
    for qtype, count in q_types.items():
        print(f"     {qtype}: {count} ({count/len(questions_list)*100:.1f}%)")
    
    return corpus_file, questions_file

def main():
    """主函数"""
    print("=" * 80)
    print("HotpotQA 数据集下载和转换工具")
    print("=" * 80)
    
    # 检查 Medical 数据集规模
    medical_corpus_size, medical_questions_size = check_medical_size()
    
    if medical_corpus_size:
        print(f"\n💡 目标: 创建比 Medical 更小的数据集")
        print(f"   Medical: {medical_corpus_size} 文档, {medical_questions_size} 问题")
        print(f"   目标: {TARGET_CORPUS_SIZE} 文档, {TARGET_QUESTIONS_SIZE} 问题")
    
    # 下载数据集
    dataset = download_hotpotqa()
    if dataset is None:
        return
    
    # 转换数据集
    corpus_file, questions_file = convert_hotpotqa_to_framework_format(
        dataset,
        max_corpus=TARGET_CORPUS_SIZE,
        max_questions=TARGET_QUESTIONS_SIZE
    )
    
    print("\n" + "=" * 80)
    print("✅ 完成！数据集已准备好使用")
    print("=" * 80)
    print(f"\n📁 输出文件:")
    print(f"   语料库: {corpus_file}")
    print(f"   问题集: {questions_file}")
    print(f"\n🚀 使用方法:")
    print(f"   python Examples/run_clearrag.py \\")
    print(f"     --subset hotpotqa \\")
    print(f"     --config_path ClearRAG/config/config.yaml")

if __name__ == "__main__":
    main()

