#!/bin/bash
##############################################################################
# 中等规模测试脚本 - 测试 100 个实例
# 预计时间：10-20 小时
# 用途：中等规模评估
##############################################################################

set -e

echo "========================================="
echo "SWE-Bench Verified 中等规模测试"
echo "测试实例数：100"
echo "预计时间：10-20 小时"
echo "========================================="
echo ""

# 检查配置文件
if [ ! -f "config.toml" ]; then
    echo "❌ 错误：config.toml 文件不存在！"
    exit 1
fi

# 检查 API 密钥
if grep -q "sk-your-api-key-here" config.toml; then
    echo "❌ 错误：请先在 config.toml 中配置您的真实 API 密钥！"
    exit 1
fi

echo "✅ 配置已就绪"
echo ""

# 启用迭代评估模式
export ITERATIVE_EVAL_MODE=true
echo "✅ 已启用迭代评估模式（失败时自动重试）"
echo ""

# 确认运行
echo "⚠️  注意：此测试预计需要 10-20 小时完成"
echo "成本估算：约 $50-100 (GPT-4o)"
echo ""
read -p "确认要开始测试吗？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "测试已取消。"
    exit 0
fi

echo ""
echo "🚀 开始运行测试..."
echo ""

# 运行测试（4 并发）
./evaluation/benchmarks/swe_bench/scripts/run_infer.sh \
    llm.eval_gpt5 \
    HEAD \
    CodeActAgent \
    100 \
    100 \
    4 \
    princeton-nlp/SWE-bench_Verified \
    test

echo ""
echo "========================================="
echo "✅ 测试完成！"
echo "========================================="

