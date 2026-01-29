#!/bin/bash
##############################################################################
# 完整测试脚本 - 测试全部 500 个实例
# 预计时间：50-100 小时
# 用途：完整 SWE-Bench Verified 评估
##############################################################################

set -e

echo "========================================="
echo "SWE-Bench Verified 完整测试"
echo "测试实例数：500（全部）"
echo "预计时间：50-100 小时"
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

# 使用 Condenser
export EVAL_CONDENSER=summarizer_for_eval
echo "✅ 已启用 Condenser（管理上下文长度）"
echo ""

# 确认运行
echo "⚠️  警告：这是完整测试！"
echo "预计时间：50-100 小时"
echo "成本估算：约 $250-500 (GPT-4o)"
echo "建议：使用 screen 或 tmux 运行此脚本，避免会话断开"
echo ""
read -p "确认要开始完整测试吗？(yes/no) " REPLY
echo
if [[ ! $REPLY == "yes" ]]; then
    echo "测试已取消。"
    exit 0
fi

# 二次确认
echo ""
echo "最后确认："
read -p "真的要运行全部 500 个实例吗？(yes/no) " REPLY2
echo
if [[ ! $REPLY2 == "yes" ]]; then
    echo "测试已取消。"
    exit 0
fi

echo ""
echo "🚀 开始运行完整测试..."
echo "建议监控日志："
echo "  tail -f logs/llm/\$(date +%Y-%m-%d)/requests.log"
echo ""

# 运行测试（4 并发）
./evaluation/benchmarks/swe_bench/scripts/run_infer.sh \
    llm.eval_gpt5 \
    HEAD \
    CodeActAgent \
    500 \
    100 \
    4 \
    princeton-nlp/SWE-bench_Verified \
    test

echo ""
echo "========================================="
echo "🎉 完整测试完成！"
echo "========================================="
echo ""
echo "运行评估以查看结果..."

