#!/bin/bash
# 在远程服务器上执行：更新代码并推送 checkpoints 中的 .pt 文件

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Step 1: 拉取最新代码"
echo "=========================================="
git pull origin main

echo ""
echo "=========================================="
echo "Step 2: 检查 checkpoints 目录中的 .pt 文件"
echo "=========================================="
if [ -d "checkpoints" ]; then
    echo "找到 checkpoints 目录，搜索 .pt 文件..."
    find checkpoints -name "*.pt" -type f | head -20
    PT_COUNT=$(find checkpoints -name "*.pt" -type f | wc -l)
    echo "共找到 $PT_COUNT 个 .pt 文件"
else
    echo "警告: checkpoints 目录不存在"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 3: 检查 git 状态"
echo "=========================================="
git status

echo ""
echo "=========================================="
echo "Step 4: 添加 checkpoints 目录中的 .pt 文件"
echo "=========================================="
# 强制添加 checkpoints 目录（即使之前被忽略）
git add -f checkpoints/**/*.pt 2>/dev/null || find checkpoints -name "*.pt" -type f -exec git add -f {} \;

echo ""
echo "=========================================="
echo "Step 5: 再次检查状态"
echo "=========================================="
git status

echo ""
echo "=========================================="
echo "Step 6: 提交更改"
echo "=========================================="
git commit -m "Add checkpoint .pt files from server" || echo "没有需要提交的文件"

echo ""
echo "=========================================="
echo "Step 7: 推送到 GitHub"
echo "=========================================="
git push origin main

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="

