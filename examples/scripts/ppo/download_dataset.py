from datasets import load_dataset
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:18669'  # 将端口号替换成你的实际端口
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:18669' # 将端口号替换成你的实际端口
ds = load_dataset("trl-internal-testing/descriptiveness-sentiment-trl-style",cache_dir="./.cache")

ds.save_to_disk("/Users/guoxing.lan/projects/dataset/for_rl/descriptiveness-sentiment-trl-style")
