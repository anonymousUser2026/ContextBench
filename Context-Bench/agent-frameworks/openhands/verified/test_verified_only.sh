#!/bin/bash
##############################################################################
# 测试 Verified 实例脚本 - 只测试 CSV 中的 174 个 Verified 实例
# 预计时间：根据数量而定
##############################################################################

set -e

echo "========================================="
echo "SWE-Bench Verified 实例测试"
echo "实例来源：selected_500_instances.csv"
echo "Verified 实例数：174"
echo "========================================="
echo ""

# 检查配置文件
if [ ! -f "config.toml" ]; then
    echo "❌ 错误：config.toml 文件不存在！"
    exit 1
fi

# 检查 API 密钥
if grep -q 'api_key = ""' config.toml; then
    echo "❌ 错误：请先在 config.toml 中配置您的 API 密钥！"
    exit 1
fi

echo "✅ 配置已就绪"
echo ""

# 检查 verified_instances.txt 是否存在
if [ ! -f "verified_instances.txt" ]; then
    echo "生成 Verified 实例列表..."
    grep "^Verified" selected_500_instances.csv | cut -d',' -f2 > verified_instances.txt
    echo "✅ 已生成 verified_instances.txt（174 个实例）"
fi

# 显示实例数量
INSTANCE_COUNT=$(wc -l < verified_instances.txt)
echo "实例数量：$INSTANCE_COUNT"
echo ""

# 启用迭代评估模式
export ITERATIVE_EVAL_MODE=true
echo "✅ 已启用迭代评估模式"
echo ""

# 使用 Condenser
export EVAL_CONDENSER=summarizer_for_eval
echo "✅ 已启用 Condenser"
echo ""

# 询问测试数量
echo "您想测试多少个 Verified 实例？"
echo "  1) 测试 10 个（快速验证，约 1-2 小时）"
echo "  2) 测试 50 个（中等规模，约 5-10 小时）"
echo "  3) 测试全部 174 个（完整测试，约 20-35 小时）"
echo "  4) 自定义数量"
echo ""
read -p "请选择 (1-4): " choice

case $choice in
    1)
        EVAL_LIMIT=10
        ;;
    2)
        EVAL_LIMIT=50
        ;;
    3)
        EVAL_LIMIT=174
        ;;
    4)
        read -p "请输入要测试的实例数量: " EVAL_LIMIT
        ;;
    *)
        echo "无效选择，默认测试 10 个"
        EVAL_LIMIT=10
        ;;
esac

echo ""
echo "📊 测试配置："
echo "  - 实例数量：$EVAL_LIMIT"
echo "  - 数据集：princeton-nlp/SWE-bench_Verified"
echo "  - 模型：GPT-5"
echo "  - 最大迭代：100"
echo ""

# 确认运行
read -p "确认开始测试？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "测试已取消。"
    exit 0
fi

echo ""
echo "🚀 开始运行 Verified 测试..."
echo ""

# 运行测试（4 并发）
# 注意：这里使用 eval_limit 来限制测试数量
./evaluation/benchmarks/swe_bench/scripts/run_infer.sh \
    llm.eval_gpt5 \
    HEAD \
    CodeActAgent \
    $EVAL_LIMIT \
    100 \
    4 \
    princeton-nlp/SWE-bench_Verified \
    test

echo ""
echo "========================================="
echo "✅ Verified 测试完成！"
echo "========================================="
echo ""
echo "运行评估："
echo "./evaluate_results.sh"

