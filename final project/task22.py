import pandas as pd
import random
import time
import requests
import json
import os
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_and_merge_similar_clusters():
    """
    读取聚类数据
    """
    print("开始读取聚类数据...")
    
    # 读取task21生成的聚类结果
    results_dir = Path(__file__).resolve().parent / "results"
    
    # 寻找task21生成的聚类数据文件
    clustering_files = list(results_dir.glob("clustered_data_task21.csv"))
    
    
    if not clustering_files:
        print("未找到task21的聚类结果文件")
        return None
    
    # 加载聚类数据
    clustering_file = clustering_files[0]
    df_clustered = pd.read_csv(clustering_file)
    
    if 'Cluster' not in df_clustered.columns:
        print("数据中未找到Cluster列")
        return df_clustered
    
    return df_clustered


def calculate_cluster_proportions():
    """
    计算并输出各聚类的占比
    """
    print("开始计算各聚类占比...")
    
    # 读取聚类数据
    results_dir = Path(__file__).resolve().parent / "results"
    
    # 寻找聚类数据文件
    clustering_files = list(results_dir.glob("*clustered_data_*.csv"))
    if not clustering_files:
        clustering_files = [results_dir / "imputed_data_task21.csv"]
    
    if not clustering_files:
        print("未找到聚类结果文件")
        return
    
    clustering_file = clustering_files[0]
    df_clustered = pd.read_csv(clustering_file)
    
    if 'Cluster' not in df_clustered.columns:
        print("数据中未找到Cluster列")
        return
    
    # 计算每个聚类的大小和占比
    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    total_count = len(df_clustered)
    cluster_proportions = (cluster_counts / total_count) * 100
    
    print("各聚类占比:")
    for cluster_id, count in cluster_counts.items():
        proportion = cluster_proportions[cluster_id]
        print(f"聚类 {cluster_id}: {count} 个样本 ({proportion:.2f}%)")
    
    # 保存聚类占比到CSV文件
    proportions_df = pd.DataFrame({
        'Cluster': cluster_counts.index,
        'Count': cluster_counts.values,
        'Proportion (%)': cluster_proportions.values
    })
    
    output_path = results_dir / "cluster_proportions.csv"
    proportions_df.to_csv(output_path, index=False)
    print(f"聚类占比已保存至: {output_path}")
    
    return proportions_df


def generate_user_portraits_with_ai():
    """
    使用AI API对聚类结果进行转写，生成人群画像文案
    """
    print("开始使用AI生成人群画像文案...")
    
    # 读取聚类数据
    df_clustered = find_and_merge_similar_clusters()
    
    if df_clustered is None:
        print("无法获取聚类数据，终止AI画像生成")
        return
    
    # 计算聚类占比
    proportions_df = calculate_cluster_proportions()
    
    # 提取聚类统计信息
    cluster_stats = df_clustered.groupby('Cluster').agg({
        'AI_Adoption': ['mean', 'median'],
        'AI_Attitude': ['mean', 'median'],
        'AI_Trust': ['mean', 'median'],
        'AI_UseForLearning': ['mean', 'median']
    }).round(2)
    
    # 展平列名
    cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns]
    cluster_stats = cluster_stats.reset_index()
    
    # 合并占比信息
    cluster_summary = cluster_stats.merge(
        proportions_df[['Cluster', 'Count', 'Proportion (%)']], 
        on='Cluster'
    )
    cluster_summary.rename(columns={'Count': 'n', 'Proportion (%)': 'pct'}, inplace=True)
    
    # 保存聚类摘要
    results_dir = Path(__file__).resolve().parent / "results"
    cluster_summary_path = results_dir / "cluster_summary_k7.csv"
    cluster_summary.to_csv(cluster_summary_path, index=False)
    print(f"聚类摘要已保存至: {cluster_summary_path}")
    
    # 读取背景变量数据
    original_data_files = []
    for year in ['2023', '2024', '2025']:
        original_data_path = results_dir / f"ai_indices_{year}_task2.csv"
        if original_data_path.exists():
            original_data_files.append(pd.read_csv(original_data_path))
    
    original_data = None
    if original_data_files:
        original_data = pd.concat(original_data_files, ignore_index=True)
    
    # 设置API参数
    API_KEY = "sk-cfe69ed331574c2dbd6cdad151ee08d2"
    API_URL = "https://api.deepseek.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 存储AI生成的人群画像
    portraits = []
    
    # 为每个聚类生成画像
    for _, cluster_row in cluster_summary.iterrows():
        cluster_id = cluster_row['Cluster']
        
        # 构建聚类信息
        cluster_info = {
            'cluster_id': int(cluster_row['Cluster']),
            'size': int(cluster_row['n']),
            'percentage': float(cluster_row['pct']),
        }
        
        # 添加聚类依据的核心变量信息（AI相关指标）
        core_ai_variables = [
            'AI_Adoption_mean', 'AI_Adoption_median',
            'AI_Attitude_mean', 'AI_Attitude_median',
            'AI_Trust_mean', 'AI_Trust_median',
            'AI_UseForLearning_mean', 'AI_UseForLearning_median'
        ]
        
        for var in core_ai_variables:
            if var in cluster_summary.columns and not pd.isna(cluster_row[var]):
                cluster_info[var] = float(cluster_row[var])
        
        # 添加其他可能的AI相关变量
        for col in df_clustered.columns:
            if col not in ['cluster_id', 'n', 'pct', 'RespondentID'] and col not in cluster_info:
                if col in cluster_row.index and not pd.isna(cluster_row[col]):
                    cluster_info[col] = float(cluster_row[col]) if isinstance(cluster_row[col], (int, float)) else str(cluster_row[col])

        # 如果有原始数据，添加背景变量信息
        if original_data is not None and 'Cluster' in original_data.columns:
            cluster_data = original_data[original_data['Cluster'] == cluster_id]
            if len(cluster_data) > 0:
                # 添加一些背景变量的统计信息
                background_vars = ['MainBranch', 'Employment', 'WorkExp', 'RemoteWork', 'Country', 'DevType']
                for var in background_vars:
                    if var in cluster_data.columns:
                        # 获取最常见的值
                        most_common = cluster_data[var].value_counts().head(1)
                        if len(most_common) > 0:
                            cluster_info[f'most_common_{var}'] = str(most_common.index[0])
        
        # 构建AI提示
        prompt = generate_prompt_for_cluster(cluster_info)
        
        # 构建API请求
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": generate_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        
        try:
            # 调用API
            print(f"正在处理聚类 {cluster_id}...")
            response = requests.post(API_URL, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                ai_portrait = result['choices'][0]['message']['content'].strip()
                
                # 保存结果
                portrait_result = {
                    'cluster_id': cluster_id,
                    'cluster_info': cluster_info,
                    'ai_portrait': ai_portrait
                }
                
                portraits.append(portrait_result)
                print(f"聚类 {cluster_id} 的画像生成完成")
            else:
                print(f"API错误: {response.status_code}")
                print(f"错误信息: {response.text}")
                
        except Exception as e:
            print(f"处理聚类 {cluster_id} 时发生错误: {e}")
        
        # 添加延迟以避免API限制
        time.sleep(1)
    
    # 保存AI生成的人群画像到文件
    output_path = results_dir / "ai_generated_portraits.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(portraits, f, ensure_ascii=False, indent=2)
    
    print(f"AI生成的人群画像已保存至: {output_path}")
    
    # 生成可读的文本文件
    txt_output_path = results_dir / "ai_generated_portraits.txt"
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        for portrait in portraits:
            f.write(f"聚类 {portrait['cluster_id']}:\n")
            f.write(f"AI画像描述:\n{portrait['ai_portrait']}\n")
            f.write("-" * 50 + "\n")
    
    print(f"AI生成的人群画像文本已保存至: {txt_output_path}")
    
    return portraits


def generate_system_prompt():
    """
    生成系统提示，约束AI的输出格式和内容
    """
    return """
你是一个专业的数据分析师，需要基于提供的聚类数据生成人群画像文案。
请按照以下要求进行描述：
1. 只基于提供的数据描述，不得杜撰任何信息
2. 每段必须引用 2–4 个关键数值，特别关注AI相关变量（如AI_Adoption、AI_Attitude、AI_Trust、AI_UseForLearning等）
3. 输出结构：`一句话标签` + `典型特征(3条，重点突出AI行为与态度)` + `与全体差异(2条，对比AI相关指标)` + `潜在需求/痛点(1条)`
4. 禁止项：不得新增未提供字段；不得解释因果（只能描述关联）；不得使用带偏见的推断
5. 请使用中文回答
"""


def generate_prompt_for_cluster(cluster_info):
    """
    为特定聚类生成提示
    """
    # 将聚类信息转换为文本格式
    info_text = "聚类信息:\n"
    for key, value in cluster_info.items():
        info_text += f"- {key}: {value}\n"
    
    return f"""
{info_text}

请根据以上聚类信息，生成该群体的人群画像文案，特别关注AI相关变量（如AI_Adoption、AI_Attitude、AI_Trust、AI_UseForLearning等）的分析，遵循以下结构：
1. 一句话标签（基于AI行为与态度特征）
2. 典型特征(3条，重点分析AI行为与态度指标)
3. 与全体差异(2条，对比AI相关指标)
4. 潜在需求/痛点(1条)

请确保只引用提供的数据，不得杜撰任何信息。
"""


if __name__ == "__main__":
    print("开始AI人群画像生成...")
    results = generate_user_portraits_with_ai()
    print("AI人群画像生成完成！")