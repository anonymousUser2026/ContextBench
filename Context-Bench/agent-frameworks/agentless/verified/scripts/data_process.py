import csv
import os

def process_data(input_file, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 动态存储不同 bench 类型的数据
    data = {}
    header = None

    # 读取原始 CSV 文件
    with open(input_file, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("错误：CSV 文件为空")
            return

        for row in reader:
            if not row or not any(row):
                continue
            bench = row[0]
            if bench not in data:
                data[bench] = []
            data[bench].append(row)

    # 将每一类写入对应的 CSV 文件
    for bt, rows in data.items():
        # 过滤掉非法字符或空名（如果有的话）
        filename = bt.replace('/', '_').replace('\\', '_') if bt else "unknown"
        output_file = os.path.join(output_dir, f"{filename}.csv")
        with open(output_file, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"成功创建 {output_file}，包含 {len(rows)} 条记录。")

if __name__ == "__main__":
    # 使用绝对路径或相对于项目根目录的路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(base_dir, "data", "selected_500_instances.csv")
    output_folder = os.path.join(base_dir, "data")
    
    process_data(input_csv, output_folder)

