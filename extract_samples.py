"""
提取指定问题类型的样本数据
从 results/clearrag/predictions_Medical.json 中提取四种问题类型各10个完整检索结果
"""
import json
import os
import argparse
from typing import Dict, List


def extract_samples(
    input_file: str,
    output_file: str,
    question_types: List[str],
    samples_per_type: int = 10
):
    """
    从大JSON文件中提取指定问题类型的样本数据
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        question_types: 要提取的问题类型列表
        samples_per_type: 每种类型提取的样本数量
    """
    print(f"开始提取数据...")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"问题类型: {question_types}")
    print(f"每种类型提取数量: {samples_per_type}")
    
    # 检查文件大小
    file_size = os.path.getsize(input_file)
    file_size_mb = file_size / (1024 * 1024)
    print(f"\n文件大小: {file_size_mb:.2f} MB")
    
    # 用于存储提取的数据
    extracted_data = {q_type: [] for q_type in question_types}
    
    # 使用流式解析处理大文件
    if file_size > 100 * 1024 * 1024:  # 大于100MB使用流式解析
        print("\n检测到大文件，使用流式解析...")
        
        # 优先尝试使用 ijson（更高效）
        use_ijson = False
        try:
            import ijson
            use_ijson = True
        except ImportError:
            print("提示: 未安装 ijson，将使用备用解析方法（可能较慢）")
            print("建议安装 ijson 以提高性能: pip install ijson")
        
        item_count = 0
        
        if use_ijson:
            # 使用 ijson 流式解析
            with open(input_file, 'rb') as f:
                parser = ijson.items(f, 'item')
                
                for item in parser:
                    item_count += 1
                    q_type = item.get("question_type", "")
                    
                    if q_type in extracted_data and len(extracted_data[q_type]) < samples_per_type:
                        extracted_data[q_type].append(item)
                        print(f"✓ 提取 {q_type}: {len(extracted_data[q_type])}/{samples_per_type} - ID: {item.get('id', 'N/A')}")
                    
                    all_complete = all(
                        len(extracted_data[q_type]) >= samples_per_type 
                        for q_type in question_types
                    )
                    
                    if all_complete:
                        print(f"\n所有类型样本已收集完成，已处理 {item_count} 条记录")
                        break
                    
                    if item_count % 1000 == 0:
                        status = ", ".join([
                            f"{q_type}: {len(extracted_data[q_type])}/{samples_per_type}"
                            for q_type in question_types
                        ])
                        print(f"已处理 {item_count} 条记录 - [{status}]")
        else:
            # 备用方法：使用 JSONDecoder 增量解析
            print("使用备用解析方法（逐对象解析）...")
            decoder = json.JSONDecoder()
            buffer = ""
            bracket_depth = 0
            brace_depth = 0
            in_string = False
            escape_next = False
            current_obj_start = -1
            
            with open(input_file, 'r', encoding='utf-8', buffering=8192) as f:
                # 跳过开头的 '['
                first_char = f.read(1)
                if first_char != '[':
                    raise ValueError("文件格式错误: 期望JSON数组")
                
                chunk_size = 8192
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    buffer += chunk
                    
                    # 尝试从缓冲区中提取完整的JSON对象
                    i = 0
                    while i < len(buffer):
                        char = buffer[i]
                        
                        if escape_next:
                            escape_next = False
                            i += 1
                            continue
                        
                        if char == '\\':
                            escape_next = True
                            i += 1
                            continue
                        
                        if char == '"':
                            in_string = not in_string
                        elif not in_string:
                            if char == '[':
                                bracket_depth += 1
                            elif char == ']':
                                bracket_depth -= 1
                            elif char == '{':
                                if brace_depth == 0:
                                    current_obj_start = i
                                brace_depth += 1
                            elif char == '}':
                                brace_depth -= 1
                                if brace_depth == 0 and current_obj_start >= 0:
                                    # 找到一个完整的对象
                                    try:
                                        obj_str = buffer[current_obj_start:i+1]
                                        # 跳过对象前的逗号和空白
                                        obj_str = obj_str.lstrip(', \n\r\t')
                                        if obj_str.startswith('{'):
                                            item = json.loads(obj_str)
                                            item_count += 1
                                            
                                            q_type = item.get("question_type", "")
                                            
                                            if q_type in extracted_data and len(extracted_data[q_type]) < samples_per_type:
                                                extracted_data[q_type].append(item)
                                                print(f"✓ 提取 {q_type}: {len(extracted_data[q_type])}/{samples_per_type} - ID: {item.get('id', 'N/A')}")
                                            
                                            all_complete = all(
                                                len(extracted_data[q_type]) >= samples_per_type 
                                                for q_type in question_types
                                            )
                                            
                                            if all_complete:
                                                print(f"\n所有类型样本已收集完成，已处理 {item_count} 条记录")
                                                break
                                            
                                            if item_count % 1000 == 0:
                                                status = ", ".join([
                                                    f"{q_type}: {len(extracted_data[q_type])}/{samples_per_type}"
                                                    for q_type in question_types
                                                ])
                                                print(f"已处理 {item_count} 条记录 - [{status}]")
                                        
                                        # 移除已处理的部分
                                        buffer = buffer[i+1:].lstrip(', \n\r\t')
                                        i = 0
                                        current_obj_start = -1
                                        continue
                                    except json.JSONDecodeError:
                                        pass
                        
                        i += 1
                    
                    if all_complete:
                        break
                    
                    # 保留缓冲区末尾可能不完整的对象
                    if len(buffer) > 100000:  # 如果缓冲区太大，保留最后一部分
                        # 找到最后一个 '{' 的位置
                        last_brace = buffer.rfind('{')
                        if last_brace > 0:
                            buffer = buffer[last_brace:]
        
        print(f"\n总共处理了 {item_count} 条记录")
    else:
        # 小文件使用标准JSON加载
        print("\n使用标准JSON解析...")
        with open(input_file, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        
        print(f"总共 {len(file_data)} 条记录")
        
        for item in file_data:
            q_type = item.get("question_type", "")
            
            if q_type in extracted_data and len(extracted_data[q_type]) < samples_per_type:
                extracted_data[q_type].append(item)
                print(f"✓ 提取 {q_type}: {len(extracted_data[q_type])}/{samples_per_type} - ID: {item.get('id', 'N/A')}")
            
            # 检查是否所有类型都已收集足够样本
            all_complete = all(
                len(extracted_data[q_type]) >= samples_per_type 
                for q_type in question_types
            )
            
            if all_complete:
                break
    
    # 合并所有提取的数据
    final_data = []
    for q_type in question_types:
        count = len(extracted_data[q_type])
        if count > 0:
            final_data.extend(extracted_data[q_type])
            print(f"\n{q_type}: 提取了 {count} 个样本")
        else:
            print(f"\n警告: {q_type} 未找到任何数据")
    
    # 保存结果
    print(f"\n保存结果到 {output_file}...")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 成功保存 {len(final_data)} 条记录到 {output_file}")
    
    # 显示统计信息
    print("\n提取统计:")
    for q_type in question_types:
        count = len(extracted_data[q_type])
        print(f"  {q_type}: {count} 个样本")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="从大JSON文件中提取指定问题类型的样本数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python extract_samples.py --input results/clearrag/predictions_Medical.json --output results/samples.json
  python extract_samples.py --input results/clearrag/predictions_Medical.json --output results/samples.json --samples 20
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='results/clearrag/predictions_Medical.json',
        help='输入JSON文件路径 (默认: results/clearrag/predictions_Medical.json)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/extracted_samples.json',
        help='输出JSON文件路径 (默认: results/extracted_samples.json)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='每种类型提取的样本数量 (默认: 10)'
    )
    
    parser.add_argument(
        '--types',
        nargs='+',
        default=['Creative Generation', 'Contextual Summarize', 'Complex Reasoning', 'Fact Retrieval'],
        help='要提取的问题类型列表 (默认: Creative Generation Contextual Summarize Complex Reasoning Fact Retrieval)'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 执行提取
    extract_samples(
        input_file=args.input,
        output_file=args.output,
        question_types=args.types,
        samples_per_type=args.samples
    )


if __name__ == "__main__":
    main()

