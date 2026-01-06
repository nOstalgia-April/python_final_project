import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gower
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(2026)

def load_data():
    """
    从processed_data_for_clustering_task.csv加载数据
    """
    results_dir = Path(__file__).resolve().parent / "results"
    data_path = results_dir / "processed_data_for_clustering_task.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"数据加载成功，共有 {len(df)} 条记录，{len(df.columns)} 个特征")
    return df

def load_or_impute_data():
    """
    如果imputed_data_task21.csv存在则直接加载，否则执行插值处理
    """
    results_dir = Path(__file__).resolve().parent / "results"
    imputed_data_path = results_dir / "imputed_data_task21.csv"
    
    # 如果插值后的数据文件存在，直接加载
    if imputed_data_path.exists():
        print("检测到已存在的插值数据文件，直接加载...")
        df = pd.read_csv(imputed_data_path)
        print(f"已加载插值数据，共有 {len(df)} 条记录，{len(df.columns)} 个特征")
        return df
    else:
        print("未找到插值数据文件，开始执行插值处理...")
        # 加载原始数据
        df = load_data()
        # 执行插值处理
        df_imputed = impute_missing_values(df)
        # 处理异常值
        df_outliers_handled = handle_outliers(df_imputed)
        return df_outliers_handled

def impute_missing_values(df):
    """
    对缺失值进行填充
    AI_Adoption使用中位数填充
    AI_Attitude和AI_Trust使用KNN插值
    分类变量使用'Unknown'填充
    """
    print("开始处理缺失值...")
    
    df_imputed = df.copy()
    
    # 对AI_Adoption使用中位数填充
    if 'AI_Adoption' in df.columns:
        median_val = df['AI_Adoption'].median()
        df_imputed['AI_Adoption'] = df['AI_Adoption'].fillna(median_val)
        print(f"AI_Adoption 使用中位数 {median_val:.2f} 填充")
    
    # 对AI_Attitude和AI_Trust使用KNN插值
    for col_name in ['AI_Attitude', 'AI_Trust']:
        if col_name in df.columns:
            missing_before = df[col_name].isna().sum()
            if missing_before > 0:
                numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # 准备插值数据
                df_for_knn = df[numerical_cols].copy()
                
                # 使用KNN插值
                imputer = KNNImputer(n_neighbors=5)
                df_knn_imputed = pd.DataFrame(
                    imputer.fit_transform(df_for_knn),
                    columns=df_for_knn.columns,
                    index=df_for_knn.index
                )
                
                # 更新列
                df_imputed[col_name] = df_knn_imputed[col_name]
                missing_after = df_imputed[col_name].isna().sum()
                print(f"{col_name} KNN插值完成，缺失值从 {missing_before} 减少到 {missing_after}")
            else:
                print(f"{col_name} 没有缺失值，无需插值")
    
    # 对分类变量使用'Unknown'填充
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if col not in ['RespondentID', 'Year']:  # 排除不需要填充的列
            df_imputed[col] = df_imputed[col].fillna('Unknown')
    
    # 保存插值后的数据
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / "imputed_data_task21.csv"
    df_imputed.to_csv(output_file, index=False)
    print(f"插值后的数据已保存至: {output_file}")
    
    print("缺失值处理完成")
    return df_imputed

def handle_outliers(df):
    """
    使用IQR + MAD修正Z值（限定5%-95%）进行异常值截断
    """
    print("开始处理异常值...")
    
    df_outliers_handled = df.copy()
    
    # 选择数值列进行异常值处理
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numerical_cols:
        # 计算IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # 计算MAD (Median Absolute Deviation)
        median_val = df[col].median()
        mad = np.median(np.abs(df[col] - median_val))
        
        # 使用修正Z值方法（基于MAD）
        modified_z_scores = 0.6745 * (df[col] - median_val) / mad if mad != 0 else np.zeros(len(df[col]))
        
        # 限定在5%-95%范围内
        lower_bound = df[col].quantile(0.05)
        upper_bound = df[col].quantile(0.95)
        
        # 截断异常值
        df_outliers_handled[col] = df[col].clip(lower_bound, upper_bound)
    
    print("异常值处理完成")
    return df_outliers_handled

def sample_data_by_year(df, sample_size=6000):
    """
    按年份分别随机抽样
    """
    print(f"开始按年份抽样，每组抽样 {sample_size} 条...")
    
    sampled_data = {}
    for year in df['Year'].unique():
        year_data = df[df['Year'] == year]
        if len(year_data) > sample_size:
            sampled_year_data = year_data.sample(n=sample_size, random_state=2026)
        else:
            sampled_year_data = year_data  # 如果数据不足则使用全部数据
        
        sampled_data[year] = sampled_year_data
        print(f"{year} 年抽样完成，样本数: {len(sampled_year_data)}")
    
    print("按年份抽样完成")
    return sampled_data

def perform_clustering_per_year(sampled_data):
    """
    对每一年的数据分别进行聚类分析
    """
    print("开始对每年数据分别进行聚类分析...")
    
    # 选择聚类变量 - 添加AI_Trust
    clustering_vars = ['AI_Adoption', 'AI_Attitude', 'AI_UseForLearning', 'AI_Trust']
    
    results = {}
    
    # 创建一个列表来存储所有年份的数据
    all_years_clustered_data = []
    
    # 用于跟踪全局聚类ID
    global_cluster_id = 0
    
    for year, df_year in sampled_data.items():
        print(f"\n对 {year} 年数据进行聚类分析...")
        
        # 确保聚类变量存在
        available_vars = [var for var in clustering_vars if var in df_year.columns]
        clustering_data = df_year[available_vars].copy()
        
        # 计算Gower距离
        print(f"{year} 年: 计算Gower距离...")
        gower_dist = gower.gower_matrix(clustering_data)
        
        # 测试不同聚类数量的轮廓系数
        silhouette_scores = []
        cluster_range = range(2, 11)  # 测试2-10个聚类
        
        print(f"{year} 年: 计算不同聚类数量的轮廓系数...")
        for n_clusters in cluster_range:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='average',
                metric='precomputed'
            )
            cluster_labels = clustering.fit_predict(gower_dist)
            score = silhouette_score(gower_dist, cluster_labels, metric='precomputed')
            silhouette_scores.append(score)
            print(f"{year} 年聚类数 {n_clusters}: 轮廓系数 = {score:.3f}")
        
        # 选择轮廓系数最高的聚类数
        best_n_clusters = cluster_range[np.argmax(silhouette_scores)]
        print(f"{year} 年最佳聚类数: {best_n_clusters} (轮廓系数: {max(silhouette_scores):.3f})")
        
        # 使用最佳聚类数进行聚类
        best_clustering = AgglomerativeClustering(
            n_clusters=best_n_clusters,
            linkage='average',
            metric='precomputed'
        )
        cluster_labels = best_clustering.fit_predict(gower_dist)
        
        # 将聚类结果映射到全局聚类ID
        unique_labels = sorted(np.unique(cluster_labels))
        label_mapping = {}
        for unique_label in unique_labels:
            label_mapping[unique_label] = global_cluster_id
            global_cluster_id += 1
        
        # 应用映射
        mapped_cluster_labels = np.array([label_mapping[label] for label in cluster_labels])
        
        # 将聚类结果添加到数据中
        df_year_clustered = df_year.copy()
        df_year_clustered['Cluster'] = mapped_cluster_labels
        df_year_clustered['Year'] = year  # 添加年份标识
        
        print(f"{year} 年聚类完成，共生成 {len(np.unique(mapped_cluster_labels))} 个聚类")
        
        # 将当前年份的聚类数据添加到总列表中
        all_years_clustered_data.append(df_year_clustered)
        
        results[year] = {
            'data': df_year_clustered,
            'n_clusters': best_n_clusters,
            'silhouette_scores': silhouette_scores
        }
    
    # 将所有年份的聚类数据合并并保存到一个统一的文件中
    if all_years_clustered_data:
        combined_clustered_data = pd.concat(all_years_clustered_data, ignore_index=True)
        results_dir = Path(__file__).resolve().parent / "results"
        output_file = results_dir / "clustered_data_task21.csv"
        combined_clustered_data.to_csv(output_file, index=False)
        print(f"所有年份聚类结果已合并保存至: {output_file}")
    
    return results

def visualize_background_variables_per_year(clustering_results):
    """
    对每一年聚类结果的背景变量进行可视化
    """
    print("开始生成背景变量可视化...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 背景变量
    background_vars = ['MainBranch', 'Employment', 'WorkExp', 'RemoteWork', 'Country', 'DevType']
    
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    for year, result in clustering_results.items():
        df_clustered = result['data']
        
        # 筛选出存在的背景变量
        available_vars = [var for var in background_vars if var in df_clustered.columns]
        
        # 计算每个聚类的数量
        cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
        
        # 为每个背景变量生成可视化
        for var in available_vars:
            plt.figure(figsize=(12, 8))
            
            # 计算每个聚类中该变量的分布
            crosstab = pd.crosstab(df_clustered[var], df_clustered['Cluster'], normalize='columns') * 100
            
            # 绘制堆叠柱状图
            crosstab.plot(kind='bar', stacked=True, figsize=(14, 8))
            plt.title(f'{year}年: {var} 在各聚类中的分布')
            plt.xlabel(var)
            plt.ylabel('百分比 (%)')
            plt.legend(title='聚类', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # 保存图片
            plt.savefig(results_dir / f"{year}_{var}_cluster_distribution.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # 聚类大小分布
        plt.figure(figsize=(10, 6))
        cluster_counts.plot(kind='bar')
        plt.title(f'{year}年: 各聚类的样本数量')
        plt.xlabel('聚类')
        plt.ylabel('样本数量')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(results_dir / f"{year}_cluster_size_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()

def generate_background_summary_per_year(clustering_results):
    """
    生成每年背景变量分析，计算每个类别在聚类中的分布差异
    """
    print("生成每年背景变量分析...")
    
    background_vars = ['MainBranch', 'Employment', 'WorkExp', 'RemoteWork', 'Country', 'DevType']
    
    all_summaries = {}
    
    for year, result in clustering_results.items():
        df_clustered = result['data']
        available_vars = [var for var in background_vars if var in df_clustered.columns]
        
        summary = {}
        
        for var in available_vars:
            # 计算每个聚类中该变量的分布
            var_summary = {}
            
            for cluster in sorted(df_clustered['Cluster'].unique()):
                cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
                # 计算该变量前3个最常见的类别
                top_categories = cluster_data[var].value_counts().head(3)
                var_summary[cluster] = top_categories.to_dict()
            
            summary[var] = var_summary
        
        all_summaries[year] = summary
    
    return all_summaries

def generate_topn_features_per_year(clustering_results, n=5):
    """
    生成每年Top-N代表性特征，计算每类与总体差异最大的特征
    """
    print(f"生成每年Top-{n}代表性特征...")
    
    all_top_features = {}
    
    for year, result in clustering_results.items():
        df_clustered = result['data']
        
        # 选择数值变量进行分析
        numerical_vars = df_clustered.select_dtypes(include=[np.number]).columns.tolist()
        # 排除聚类变量和ID变量
        exclude_vars = ['Cluster', 'RespondentID']
        numerical_vars = [var for var in numerical_vars if var not in exclude_vars]
        
        # 计算总体均值
        overall_means = df_clustered[numerical_vars].mean()
        
        top_features = {}
        
        for cluster in sorted(df_clustered['Cluster'].unique()):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
            cluster_means = cluster_data[numerical_vars].mean()
            
            # 计算差异（绝对值）
            differences = np.abs(cluster_means - overall_means).sort_values(ascending=False)
            top_features[cluster] = differences.head(n).to_dict()
        
        all_top_features[year] = top_features
    
    return all_top_features

def main():
    """
    主函数
    """
    print("开始聚类分析...")
    
    # 1. 加载数据（如果存在插值数据则直接加载，否则进行插值处理）
    df = load_or_impute_data()
    
    # 2. 按年份抽样
    sampled_data = sample_data_by_year(df, sample_size=6000)
    
    # 3. 对每年数据分别聚类
    clustering_results = perform_clustering_per_year(sampled_data)
    
    # 4. 背景变量可视化
    visualize_background_variables_per_year(clustering_results)
    
    # 5. 生成背景变量分析
    background_summary = generate_background_summary_per_year(clustering_results)
    print("\n背景变量分析结果:")
    for year, summary in background_summary.items():
        print(f"\n{year}年:")
        for var, var_summary in summary.items():
            print(f"  {var}:")
            for cluster, top_cats in var_summary.items():
                print(f"    聚类 {cluster}: {top_cats}")
    
    # 6. 生成Top-N特征分析
    top_features = generate_topn_features_per_year(clustering_results, n=5)
    print("\nTop-N代表性特征分析结果:")
    for year, year_features in top_features.items():
        print(f"\n{year}年:")
        for cluster, features in year_features.items():
            print(f"  聚类 {cluster} 的Top特征:")
            for i, (feature, diff) in enumerate(list(features.items())[:3]):
                print(f"    {i+1}. {feature}: {diff:.3f}")
    
    print("\n聚类分析完成！")


if __name__ == "__main__":
    main()