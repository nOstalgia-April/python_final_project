import csv
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def load_survey_data(year):
    """
    加载指定年份的Stack Overflow开发者调查数据
    """
    base_dir = Path(__file__).resolve().parent.parent
    file_path = base_dir / f"stack-overflow-developer-survey-{year}" / "survey_results_public.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"找不到 {year} 年的调查数据文件: {file_path}")
        
    df = pd.read_csv(file_path, low_memory=False)
    return df

def calculate_ai_adoption_index(df, year):
    """
    计算AI采纳强度指数 (I1)
    根据不同年份使用不同的映射规则
    """
    if 'AISelect' not in df.columns:
        return None
    
    # 根据年份设置不同的映射规则
    if year in ['2023', '2024']:
        mapping = {
            'Yes': 100,
            'No, but I plan to soon': 30,
            'No, and I don\'t plan to': 0
        }
    else:  # 2025
        mapping = {
            'Yes, I use AI tools daily': 100,
            'Yes, I use AI tools weekly': 75,
            'Yes, I use AI tools monthly or infrequently': 40,
            'No, and I don\'t plan to': 0
        }
    
    # 应用映射并计算指数
    adoption_scores = df['AISelect'].map(mapping)
    return adoption_scores

def calculate_workflow_coverage_index(df, year):
    """
    计算AI工作流覆盖指数 (I2)
    """
    # 跨年可比的共同场景
    common_scenarios = [
        'Learning about a codebase',
        'Project planning', 
        'Writing code',
        'Documenting code',
        'Debugging and getting help',
        'Testing code',
        'Committing and reviewing code',
        'Deployment and monitoring'
    ]
    
    if year in ['2023', '2024']:
        if 'AIToolCurrently Using' not in df.columns:
            return None
            
        coverage_scores = []
        for _, row in df.iterrows():
            if pd.isna(row['AIToolCurrently Using']) or row['AIToolCurrently Using'] == 'NA':
                coverage_scores.append(0)
                continue
                
            scenarios = [s.strip() for s in str(row['AIToolCurrently Using']).split(';')]
            covered_count = len(set(scenarios) & set(common_scenarios))
            score = (covered_count / len(common_scenarios)) * 100
            coverage_scores.append(score)
            
    else:  # 2025
        if 'AIToolCurrently mostly AI' not in df.columns or 'AIToolCurrently partially AI' not in df.columns:
            return None
            
        coverage_scores = []
        for _, row in df.iterrows():
            mostly_scenarios = []
            partially_scenarios = []
            
            if not (pd.isna(row['AIToolCurrently mostly AI']) or row['AIToolCurrently mostly AI'] == 'NA'):
                mostly_scenarios = [s.strip() for s in str(row['AIToolCurrently mostly AI']).split(';')]
                
            if not (pd.isna(row['AIToolCurrently partially AI']) or row['AIToolCurrently partially AI'] == 'NA'):
                partially_scenarios = [s.strip() for s in str(row['AIToolCurrently partially AI']).split(';')]
            
            # 计算得分：mostly得1分，partially得0.5分
            score = 0
            for scenario in mostly_scenarios:
                if scenario in common_scenarios:
                    score += 1.0
                    
            for scenario in partially_scenarios:
                if scenario in common_scenarios:
                    score += 0.5
                    
            normalized_score = (score / len(common_scenarios)) * 100
            coverage_scores.append(normalized_score)
    
    return coverage_scores

def calculate_tool_breadth_index(df, year):
    """
    计算AI工具使用广度指数 (I3)
    返回两种计算方法的结果：绝对值和相对值
    """
    tools_used_columns = {
        '2023': ['AISearchHaveWorkedWith', 'AIDevHaveWorkedWith'],
        '2024': ['AISearchDevHaveWorkedWith'],
        '2025': ['DevEnvsHaveWorkedWith', 'AIModelsHaveWorkedWith']
    }
    
    if year not in tools_used_columns:
        return None, None
        
    columns = tools_used_columns[year]
    
    # 检查必要的列是否存在
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        return None, None
    
    tool_counts = []
    for _, row in df.iterrows():
        all_tools = set()
        for col in columns:
            if not (pd.isna(row[col]) or row[col] == 'NA'):
                tools = [t.strip() for t in str(row[col]).split(';') if t.strip()]
                all_tools.update(tools)
        tool_counts.append(len(all_tools))
    
    # 绝对值方法：直接使用工具数量
    absolute_values = tool_counts
    
    # 相对值方法：工具数量 / 该年工具选项总数 * 100
    # 根据问卷文本内容确定各年份的工具选项总数
    total_options = {
        '2023': 20,  # AISearch(12种) + AIDev(8种) = 20种
        '2024': 12,  # AISearchDev(12种)
        '2025': 45   # DevEnvs(25种) + AIModels(20种) = 45种
    }
    
    relative_values = [(count / total_options[year]) * 100 for count in tool_counts]
    
    return absolute_values, relative_values

def calculate_ai_attitude_index(df, year):
    """
    计算AI态度指数 (I4)
    """
    if 'AISent' not in df.columns:
        return None
        
    mapping = {
        'Very favorable': 100,
        'Favorable': 75,
        'Indifferent': 50,
        'Unsure': 40,
        'Unfavorable': 25,
        'Very unfavorable': 0
    }
    
    attitude_scores = df['AISent'].map(mapping)
    return attitude_scores

def calculate_ai_trust_index(df, year):
    """
    计算AI信任指数 (I5)
    """
    # 根据年份确定列名
    if year == '2023':
        column_name = 'AIBen'
    else:  # 2024, 2025
        column_name = 'AIAcc'
        
    if column_name not in df.columns:
        return None
        
    mapping = {
        'Highly trust': 100,
        'Somewhat trust': 75,
        'Neither trust nor distrust': 50,
        'Somewhat distrust': 25,
        'Highly distrust': 0
    }
    
    trust_scores = df[column_name].map(mapping)
    return trust_scores

def calculate_ai_threat_index(df, year):
    """
    计算AI威胁感知指数 (I6)
    仅适用于2024和2025年
    """
    if year not in ['2024', '2025']:
        return None
        
    if 'AIThreat' not in df.columns:
        return None
        
    mapping = {
        'Yes': 100,
        'I\'m not sure': 50,
        'No': 0
    }
    
    threat_scores = df['AIThreat'].map(mapping)
    return threat_scores

def calculate_expected_benefits_indices(df, year):
    """
    计算AI期望收益指数 (I7)
    仅适用于2023和2024年
    """
    if year not in ['2023', '2024']:
        return None, None, None, None
        
    # 根据年份确定列名
    if year == '2023':
        column_name = 'AIAcc'
    else:  # 2024
        column_name = 'AIBen'
        
    if column_name not in df.columns:
        return None, None, None, None
        
    # 跨年可比的选项
    efficiency_options = {'Increase productivity', 'Greater efficiency'}
    learning_option = 'Speed up learning'
    quality_option = 'Improve accuracy in coding'
    collaboration_option = 'Improve collaboration'
    
    efficiency_scores = []
    learning_scores = []
    quality_scores = []
    collaboration_scores = []
    
    for _, row in df.iterrows():
        if pd.isna(row[column_name]) or row[column_name] == 'NA':
            efficiency_scores.append(None)
            learning_scores.append(None)
            quality_scores.append(None)
            collaboration_scores.append(None)
            continue
            
        options = set([opt.strip() for opt in str(row[column_name]).split(';')])
        
        # 效率指数：选中 Increase productivity 和/或 Greater efficiency
        efficiency_count = len(options & efficiency_options)
        efficiency_score = (efficiency_count / len(efficiency_options)) * 100
        efficiency_scores.append(efficiency_score)
        
        # 学习指数：是否选中 Speed up learning
        learning_score = 100 if learning_option in options else 0
        learning_scores.append(learning_score)
        
        # 质量指数：是否选中 Improve accuracy in coding
        quality_score = 100 if quality_option in options else 0
        quality_scores.append(quality_score)
        
        # 协作指数：是否选中 Improve collaboration
        collaboration_score = 100 if collaboration_option in options else 0
        collaboration_scores.append(collaboration_score)
    
    return efficiency_scores, learning_scores, quality_scores, collaboration_scores

def calculate_complex_handling_index(df, year):
    """
    计算AI复杂任务胜任指数 (I8)
    仅适用于2024和2025年
    """
    if year not in ['2024', '2025']:
        return None
        
    if 'AIComplex' not in df.columns:
        return None
        
    mapping = {
        'Very well at handling complex tasks': 100,
        'Good, but not great at handling complex tasks': 75,
        'Neither good or bad at handling complex tasks': 50,
        'Bad at handling complex tasks': 25,
        'Very poor at handling complex tasks': 0
    }
    
    complex_scores = df['AIComplex'].map(mapping)
    return complex_scores

def calculate_agent_impact_indices(df, year):
    """
    计算AI Agent实际影响指数 (I9)
    仅适用于2025年
    """
    if year != '2025':
        return None, None, None, None, None, None
        
    # 需要的列名
    required_columns = [
        'AIAgentImpactStrongly agree',
        'AIAgentImpactSomewhat agree',
        'AIAgentImpactNeutral',
        'AIAgentImpactSomewhat disagree',
        'AIAgentImpactStrongly disagree'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return None, None, None, None, None, None
    
    # 主题定义
    productivity_statements = [
        'AI agents have increased my productivity.',
        'AI agents have reduced the time spent on specific development tasks.'
    ]
    
    automation_statements = [
        'AI agents have helped me automate repetitive tasks.'
    ]
    
    quality_statements = [
        'AI agents have improved the quality of my code.'
    ]
    
    learning_statements = [
        'AI agents have accelerated my learning about new technologies or codebases.'
    ]
    
    problem_solving_statements = [
        'AI agents have helped me solve complex problems more effectively.'
    ]
    
    collaboration_statements = [
        'AI agents have improved collaboration within my team.'
    ]
    
    def calculate_topic_score(statements, row):
        """计算特定主题的得分"""
        score = 0
        count = 0
        
        # 检查每个程度的列
        for level_col, weight in [
            ('AIAgentImpactStrongly agree', 1.0),
            ('AIAgentImpactSomewhat agree', 0.5),
            ('AIAgentImpactNeutral', 0.0),
            ('AIAgentImpactSomewhat disagree', -0.5),
            ('AIAgentImpactStrongly disagree', -1.0)
        ]:
            if not (pd.isna(row[level_col]) or row[level_col] == 'NA'):
                matched_statements = [s.strip() for s in str(row[level_col]).split(';')]
                matched_count = len(set(matched_statements) & set(statements))
                score += matched_count * weight
                count += matched_count
        
        # 如果没有任何匹配的语句，返回None
        if count == 0:
            return None
            
        # 归一化到0-100范围
        # 假设最大可能得分为1.0（所有语句都是"Strongly agree"）
        normalized_score = max(0, min(100, (score + len(statements)) / (2 * len(statements)) * 100))
        return normalized_score
    
    productivity_scores = []
    automation_scores = []
    quality_scores = []
    learning_scores = []
    problem_solving_scores = []
    collaboration_scores = []
    
    for _, row in df.iterrows():
        productivity_score = calculate_topic_score(productivity_statements, row)
        automation_score = calculate_topic_score(automation_statements, row)
        quality_score = calculate_topic_score(quality_statements, row)
        learning_score = calculate_topic_score(learning_statements, row)
        problem_solving_score = calculate_topic_score(problem_solving_statements, row)
        collaboration_score = calculate_topic_score(collaboration_statements, row)
        
        productivity_scores.append(productivity_score)
        automation_scores.append(automation_score)
        quality_scores.append(quality_score)
        learning_scores.append(learning_score)
        problem_solving_scores.append(problem_solving_score)
        collaboration_scores.append(collaboration_score)
    
    return (
        productivity_scores,
        automation_scores,
        quality_scores,
        learning_scores,
        problem_solving_scores,
        collaboration_scores
    )

def calculate_job_satisfaction_index(df, year):
    """
    计算工作满意度指数 (I10)
    仅适用于2024和2025年
    """
    if year not in ['2024', '2025']:
        return None
        
    if 'JobSat' not in df.columns:
        return None
        
    # JobSat是0-10的量表，转换为0-100
    job_satisfaction_scores = (df['JobSat'] / 10) * 100
    return job_satisfaction_scores

def calculate_ai_use_for_learning_index(df, year):
    """
    计算AI学习相关使用指数 (I11)
    """
    # 跨年可比口径
    learning_scenario = 'Learning about a codebase'
    
    if year in ['2023', '2024']:
        if 'AIToolCurrently Using' not in df.columns:
            return None
            
        learning_scores = []
        for _, row in df.iterrows():
            if pd.isna(row['AIToolCurrently Using']) or row['AIToolCurrently Using'] == 'NA':
                learning_scores.append(0)
                continue
                
            scenarios = [s.strip() for s in str(row['AIToolCurrently Using']).split(';')]
            score = 100 if learning_scenario in scenarios else 0
            learning_scores.append(score)
            
    else:  # 2025
        if 'AIToolCurrently mostly AI' not in df.columns or 'AIToolCurrently partially AI' not in df.columns:
            return None
            
        learning_scores = []
        for _, row in df.iterrows():
            mostly_scenarios = []
            partially_scenarios = []
            
            if not (pd.isna(row['AIToolCurrently mostly AI']) or row['AIToolCurrently mostly AI'] == 'NA'):
                mostly_scenarios = [s.strip() for s in str(row['AIToolCurrently mostly AI']).split(';')]
                
            if not (pd.isna(row['AIToolCurrently partially AI']) or row['AIToolCurrently partially AI'] == 'NA'):
                partially_scenarios = [s.strip() for s in str(row['AIToolCurrently partially AI']).split(';')]
            
            # 计算得分：mostly得100分，partially得50分
            score = 0
            if learning_scenario in mostly_scenarios:
                score = 100
            elif learning_scenario in partially_scenarios:
                score = 50
                
            learning_scores.append(score)
    
    return learning_scores

def calculate_ai_learn_engagement_index(df, year):
    """
    计算AI学习投入指数 (I12)
    仅适用于2025年
    """
    if year != '2025':
        return None
        
    if 'LearnCodeAI' not in df.columns or 'AILearnHow' not in df.columns:
        return None
        
    engagement_scores = []
    for _, row in df.iterrows():
        # LearnCodeAI映射：学过=100，没学过=0
        learn_code_score = 100 if row['LearnCodeAI'] == 'Yes' else 0
        
        # AILearnHow的渠道多样性
        if pd.isna(row['AILearnHow']) or row['AILearnHow'] == 'NA':
            diversity_score = 0
        else:
            channels = [c.strip() for c in str(row['AILearnHow']).split(';')]
            # 假设有大约15种学习渠道选项（根据问卷文本）
            diversity_score = min(100, (len(channels) / 15) * 100)
        
        # 组合得分
        engagement_score = (learn_code_score + diversity_score) / 2
        engagement_scores.append(engagement_score)
    
    return engagement_scores

def compute_indices_for_year(year):
    """
    为指定年份计算所有指数
    """
    print(f"正在处理 {year} 年的数据...")
    
    try:
        df = load_survey_data(year)
        print(f"成功加载 {len(df)} 条记录")
    except Exception as e:
        print(f"加载 {year} 年数据失败: {e}")
        return None
    
    # 计算AI采纳强度指数 (I1)
    adoption_index = calculate_ai_adoption_index(df, year)
    
    # 计算AI工作流覆盖指数 (I2)
    workflow_index = calculate_workflow_coverage_index(df, year)
    
    # 计算AI工具使用广度指数 (I3)
    tool_breadth_absolute, tool_breadth_relative = calculate_tool_breadth_index(df, year)
    
    # 计算AI态度指数 (I4)
    attitude_index = calculate_ai_attitude_index(df, year)
    
    # 计算AI信任指数 (I5)
    trust_index = calculate_ai_trust_index(df, year)
    
    # 计算AI威胁感知指数 (I6)
    threat_index = calculate_ai_threat_index(df, year)
    
    # 计算AI期望收益指数 (I7)
    efficiency_index, learning_index, quality_index, collaboration_index = calculate_expected_benefits_indices(df, year)
    
    # 计算AI复杂任务胜任指数 (I8)
    complex_index = calculate_complex_handling_index(df, year)
    
    # 计算AI Agent实际影响指数 (I9)
    (
        productivity_index,
        automation_index,
        quality_agent_index,
        learning_agent_index,
        problem_solving_index,
        collaboration_agent_index
    ) = calculate_agent_impact_indices(df, year)
    
    # 计算工作满意度指数 (I10)
    job_satisfaction_index = calculate_job_satisfaction_index(df, year)
    
    # 计算AI学习相关使用指数 (I11)
    ai_use_for_learning_index = calculate_ai_use_for_learning_index(df, year)
    
    # 计算AI学习投入指数 (I12)
    ai_learn_engagement_index = calculate_ai_learn_engagement_index(df, year)
    
    # 选择一些关键的背景变量用于用户画像
    # 定义关键背景变量
    background_vars = ['MainBranch', 'Employment', 'Country', 'WorkExp', 'RemoteWork', 'EdLevel', 'DevType']
    available_background_vars = [var for var in background_vars if var in df.columns]
    
    # 创建结果DataFrame
    result_data = {'RespondentID': range(len(df)), 'Year': year}
    
    if adoption_index is not None:
        result_data['AI_Adoption'] = adoption_index
        
    if workflow_index is not None:
        result_data['AI_WorkflowCoverage_Current'] = workflow_index
        
    if tool_breadth_absolute is not None and tool_breadth_relative is not None:
        result_data['AI_ToolBreadth_Absolute'] = tool_breadth_absolute
        result_data['AI_ToolBreadth_Relative'] = tool_breadth_relative
        
    if attitude_index is not None:
        result_data['AI_Attitude'] = attitude_index
        
    if trust_index is not None:
        result_data['AI_Trust'] = trust_index
        
    if threat_index is not None:
        result_data['AI_Threat'] = threat_index
        
    if efficiency_index is not None:
        result_data['AI_ExpectedBenefits_Efficiency'] = efficiency_index
        
    if learning_index is not None:
        result_data['AI_ExpectedBenefits_Learning'] = learning_index
        
    if quality_index is not None:
        result_data['AI_ExpectedBenefits_Quality'] = quality_index
        
    if collaboration_index is not None:
        result_data['AI_ExpectedBenefits_Collab'] = collaboration_index
        
    if complex_index is not None:
        result_data['AI_ComplexHandling'] = complex_index
        
    if productivity_index is not None:
        result_data['AI_AgentImpact_Productivity'] = productivity_index
        
    if automation_index is not None:
        result_data['AI_AgentImpact_Automation'] = automation_index
        
    if quality_agent_index is not None:
        result_data['AI_AgentImpact_Quality'] = quality_agent_index
        
    if learning_agent_index is not None:
        result_data['AI_AgentImpact_Learning'] = learning_agent_index
        
    if problem_solving_index is not None:
        result_data['AI_AgentImpact_ProblemSolving'] = problem_solving_index
        
    if collaboration_agent_index is not None:
        result_data['AI_AgentImpact_Collaboration'] = collaboration_agent_index
        
    if job_satisfaction_index is not None:
        result_data['JobSatisfaction'] = job_satisfaction_index
        
    if ai_use_for_learning_index is not None:
        result_data['AI_UseForLearning'] = ai_use_for_learning_index
        
    if ai_learn_engagement_index is not None:
        result_data['AI_LearnEngagement'] = ai_learn_engagement_index
    
    # 添加AIAgents和SOFriction字段（仅在2025年存在）
    if year == '2025':
        if 'AIAgents' in df.columns:
            result_data['AIAgents'] = df['AIAgents']
        if 'SOFriction' in df.columns:
            result_data['SOFriction'] = df['SOFriction']
    
    # 添加背景变量
    for var in available_background_vars:
        result_data[var] = df[var]
    
    result_df = pd.DataFrame(result_data)
    return result_df


def preprocess_data_for_clustering(df):
    """
    预处理数据用于聚类分析
    - 仅对数值型AI指数进行标准化和缺失值处理
    - 保留分类变量的原始形式用于后续分析
    - 添加缺失值统计功能
    """
    print(f"开始预处理 {len(df)} 条记录，{len(df.columns)} 个特征...")
    
    # 分离数值变量和分类变量
    numerical_columns = []
    categorical_columns = []
    
    for col in df.columns:
        if col in ['RespondentID', 'Year']:  # 特殊列处理
            continue
        elif df[col].dtype in ['int64', 'float64']:
            numerical_columns.append(col)
        else:
            categorical_columns.append(col)
    
    print(f"识别到 {len(numerical_columns)} 个数值列和 {len(categorical_columns)} 个分类列")
    
    # 统计缺失值
    print("\n缺失值统计:")
    missing_stats = df.isnull().sum()
    missing_stats = missing_stats[missing_stats > 0]  # 只显示有缺失值的列
    if len(missing_stats) > 0:
        for col, count in missing_stats.items():
            pct = (count / len(df)) * 100
            print(f"  {col}: {count} ({pct:.2f}%)")
    else:
        print("  没有发现缺失值")
    
    # 复制数据以避免修改原始数据
    df_processed = df.copy()
    
    
    
 
    
    # 仅对数值变量进行标准化
    print("开始数值变量标准化...")
    if numerical_columns:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_processed[numerical_columns] = scaler.fit_transform(df_processed[numerical_columns])
        print("标准化完成")
    else:
        print("没有数值列需要标准化")
    
    print("预处理完成")
    return df_processed


def main():
    """
    主函数：计算2023、2024、2025年的指数并保存结果
    """
    years = ['2023', '2024', '2025']
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # 存储所有年份的数据
    all_years_data = []
    
    for year in years:
        print(f"开始处理 {year} 年数据...")
        result_df = compute_indices_for_year(year)
        if result_df is not None:
            print(f"{year} 年数据计算成功，共 {len(result_df)} 条记录")
            all_years_data.append(result_df)
            output_file = output_dir / f"ai_indices_{year}.csv"
            # 保存原始结果
            original_cols = [col for col in result_df.columns if not col.startswith(('MainBranch_', 'Employment_', 'Country_', 'DevType_'))]
            result_df[original_cols].to_csv(output_file, index=False)
            print(f"{year} 年指数计算完成，结果已保存至: {output_file}")
        else:
            print(f"{year} 年指数计算失败")
    
    print(f"总共处理了 {len(all_years_data)} 个年份的数据")
    
    # 合并所有年份的数据
    if all_years_data:
        print("开始合并所有年份的数据...")
        combined_df = pd.concat(all_years_data, ignore_index=True)
        
        print("开始预处理数据用于聚类分析...")
        # 预处理数据用于聚类分析
        processed_df = preprocess_data_for_clustering(combined_df)
        
        # 保存预处理后的数据
        processed_output_file = output_dir / "processed_data_for_clustering.csv"
        processed_df.to_csv(processed_output_file, index=False)
        print(f"预处理后的聚类数据已保存至: {processed_output_file}")
        
        print(f"总共处理了 {len(combined_df)} 条记录用于聚类分析")
        print(f"特征维度从原始的 {len(combined_df.columns)} 扩展到 {len(processed_df.columns)} (经独热编码后)")
    else:
        print("没有成功处理任何年份的数据，跳过合并和预处理步骤")

if __name__ == "__main__":
    main()