from tqdm import tqdm
import argparse

def merge_jsonl_files(file1, file2, output_file):
    # 计算总行数以设置进度条总长度
    total_lines = sum(1 for _ in open(file1, 'r', encoding='utf-8')) + sum(1 for _ in open(file2, 'r', encoding='utf-8')) - 1  # 减去第二个文件的最后一行
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # 读取并写入第一个文件的内容
        with open(file1, 'r', encoding='utf-8') as f1:
            for line in tqdm(f1, total=total_lines, desc="Merging files"):
                out_f.write(line)
        
        # 读取并写入第二个文件的内容，跳过最后一行
        with open(file2, 'r', encoding='utf-8') as f2:
            lines = f2.readlines()
            for line in tqdm(lines[:-1], total=total_lines, desc="Merging files", leave=False):
                out_f.write(line)

# 文件路径
def main():
    parser = argparse.ArgumentParser(description='Merge two JSONL files')
    parser.add_argument('--file1', type=str, required=True, help='Path to first input file')
    parser.add_argument('--file2', type=str, required=True, help='Path to second input file')
    parser.add_argument('--output', type=str, required=True, help='Path to output file')

    args = parser.parse_args()

    # 合并文件
    merge_jsonl_files(args.file1, args.file2, args.output)

if __name__ == "__main__":
    main()
