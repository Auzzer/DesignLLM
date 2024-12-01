import base64

def print_invalid_base64(file_path):
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            # 提取第一列（按空格或制表符分隔）
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 1:
                continue  # 跳过空行或格式错误的行
            base64_str = parts[0]
            
            # 验证 Base64 格式
            try:
                base64.b64decode(base64_str)
            except Exception:
                print(f"Invalid Base64 on line {line_num}: {line.strip()}")

input_file = "/projectnb/ece601/24FallA2Group16/checkpoints/Llama3.1-8B-Instruct/tokenizer.model"

print_invalid_base64(input_file)
