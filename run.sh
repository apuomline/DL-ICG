# !/bin/bash

# 执行第一个脚本
./pipeline.sh --dataset-type 400 --config-dir configs --log-dir logs_400

# 检查第一个脚本是否成功执行
if [ $? -ne 0 ]; then
  echo "第一个脚本执行失败，退出程序。"
  exit 1
fi

# # 执行第二个脚本
./pipeline.sh --dataset-type 5000 --config-dir configs --log-dir logs_5000

# 检查第二个脚本是否成功执行
if [ $? -ne 0 ]; then
  echo "第二个脚本执行失败，退出程序。"
  exit 1
fi

echo "两个脚本均已成功执行。"