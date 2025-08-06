import re
import csv
import os
from collections import defaultdict
from datetime import datetime
import argparse # 1. 导入 argparse 模块

def parse_training_log(log_file_path, max_iteration=None): # 2. 增加 max_iteration 参数
    """
    解析训练日志文件，提取每个学习迭代的指标数据
    只提取关键性能指标，忽略辅助信息
    """
    iteration_data = defaultdict(dict)
    current_iteration = None
    all_metrics = set()
    
    iteration_pattern = re.compile(r"Learning iteration (\d+)/\d+")
    metric_pattern = re.compile(r"│\s*([^:]+?)\s*:\s*([^\s│]+(?:\s[^\s│]+)*)\s*│")
    
    ignore_keywords = [
        "Computation", "Collection time", "Learning time", "Iteration time", 
        "Total time", "ETA", "Time Now", "Logging Directory", "Note"
    ]
    
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            iter_match = iteration_pattern.search(line)
            if iter_match:
                current_iteration = int(iter_match.group(1))
                
                # 3. 检查是否超过了最大迭代步数
                if max_iteration is not None and current_iteration > max_iteration:
                    print(f"\n已达到最大迭代步数 {max_iteration}，停止解析。")
                    break # 提前退出循环
                    
                iteration_data[current_iteration] = {}
                continue
            
            if current_iteration is not None:
                metric_match = metric_pattern.search(line)
                if metric_match:
                    key = metric_match.group(1).strip()
                    value = metric_match.group(2).strip()
                    
                    if any(ignore in key for ignore in ignore_keywords):
                        continue
                    
                    if key == "Total timesteps":
                        key = "total_timesteps"
                    
                    try:
                        if ' ' in value:
                            num_part = value.split()[0]
                            value = float(num_part) if '.' in num_part else int(num_part)
                        else:
                            value = float(value) if '.' in value else int(value)
                    except (ValueError, TypeError):
                        pass
                    
                    iteration_data[current_iteration][key] = value
                    all_metrics.add(key)
    
    all_metrics = sorted(all_metrics)
    for iteration, data in iteration_data.items():
        for metric in all_metrics:
            if metric not in data:
                iteration_data[iteration][metric] = None
    
    return dict(iteration_data), all_metrics

def save_metrics_to_csv(metrics_dict, all_metrics, output_path):
    """
    将提取的指标保存为CSV文件
    """
    headers = ["iteration"] + all_metrics
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        sorted_iterations = sorted(metrics_dict.keys())
        
        for iteration in sorted_iterations:
            data = metrics_dict[iteration]
            row = [iteration]
            
            for metric in all_metrics:
                row.append(data.get(metric, None))
            
            writer.writerow(row)

def main():
    # 4. 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="从训练日志中解析指标并保存为CSV。")
    
    # 添加一个可选参数来指定最大迭代步数
    # 使用 nargs='?' 和 const/default 来处理 `+iteration` 标志（如果存在但没有值）
    # 但更简单、更标准的做法是 `--iteration <值>` 或 `-i <值>`
    # 为了兼容您的 `+iteration=20000` 格式，我们需要手动解析
    parser.add_argument('params', nargs='*', help="Hydra-style arugments like '+iteration=20000'")

    args = parser.parse_args()
    
    max_iter = None
    for param in args.params:
        if param.startswith('+iteration='):
            try:
                max_iter = int(param.split('=')[1])
                print(f"设置最大迭代步数为: {max_iter}")
            except (IndexError, ValueError):
                print(f"警告: 无法解析参数 '{param}'，将读取所有迭代步数。")

    # 配置路径
    log_file = "nohup_bucket/nohup_FrontKick_opt_DR_grok"
    file_name = os.path.basename(log_file)
    output_dir = "extracted_csv_bucket"
    os.makedirs(output_dir, exist_ok=True)  # 创建目录（如果不存在）
    output_csv = os.path.join(output_dir, f"{file_name}_{max_iter}.csv")  # 拼接完整路径
    
    # 解析日志
    print(f"开始解析日志文件: {log_file}")
    start_time = datetime.now()
    # 5. 将解析到的最大迭代步数传递给函数
    metrics, all_metrics = parse_training_log(log_file, max_iteration=max_iter)
    elapsed = datetime.now() - start_time
    
    print(f"解析完成! 共找到 {len(metrics)} 个学习迭代")
    print(f"检测到 {len(all_metrics)} 个关键指标")
    print(f"耗时: {elapsed.total_seconds():.2f} 秒")
    
    # 保存结果为CSV
    save_metrics_to_csv(metrics, all_metrics, output_csv)
    print(f"结果已保存为CSV: {output_csv}")
    
    # 打印示例数据
    if metrics:
        sample_iter = min(metrics.keys())
        print("\n示例数据 (迭代 {}):".format(sample_iter))
        print("  iteration:", sample_iter)
        for i, metric in enumerate(all_metrics[:5]):
            print(f"  {metric}: {metrics[sample_iter].get(metric, 'N/A')}")

if __name__ == "__main__":
    main()