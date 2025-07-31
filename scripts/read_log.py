def analyze_times(file_path):
    model_time_total = 0.0  # 所有 model_time 的总时间（截断为 60）
    model_time_count = 0    # 满足条件的 model_time 数量（<60 且上一行匹配）

    total_time_total = 0.0  # 所有 total_time 的总时间（截断为 60）
    total_time_count = 0    # 满足条件的 total_time 数量（<60 且上一行匹配）

    previous_line = ""

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # 累加所有 model_time（无论上一行）
            if line.startswith("model_time:"):
                try:
                    time_value = float(line.split(":")[1].strip())
                    model_time_total += min(time_value, 60.0)

                    # 只在满足条件时 +1
                    if "Label is 1, stopping early and returning." in previous_line and time_value < 60:
                        model_time_count += 1
                except ValueError:
                    pass

            # 累加所有 total_time（无论上一行）
            elif line.startswith("total_time:"):
                try:
                    time_str = line.split(",")[0].split(":")[1].strip()
                    time_value = float(time_str)
                    total_time_total += min(time_value, 60.0)

                    # 只在满足条件时 +1
                    if "Bug found" in previous_line and time_value < 60:
                        total_time_count += 1
                except ValueError:
                    pass

            previous_line = line

    print("=== Summary ===")
    print(f"model_time total: {model_time_total:.2f}s, count(<60 & label): {model_time_count}")
    print(f"total_time total: {total_time_total:.2f}s, count(<60 & bug): {total_time_count}")

# 用法示例
file_path = 're.out'  # 替换为你的文件路径
analyze_times(file_path)
