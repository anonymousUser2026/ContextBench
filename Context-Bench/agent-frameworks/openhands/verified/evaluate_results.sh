#!/bin/bash
##############################################################################
# 评估结果脚本
# 用途：对测试输出进行评估并生成报告
##############################################################################

set -e

echo "========================================="
echo "SWE-Bench Verified 结果评估"
echo "========================================="
echo ""

# 查找最新的输出文件
OUTPUT_DIR="evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ 错误：未找到输出目录"
    echo "请先运行测试"
    exit 1
fi

echo "查找输出文件..."
echo ""

# 列出所有输出文件
FOUND_FILES=$(find "$OUTPUT_DIR" -name "output.jsonl" 2>/dev/null || true)

if [ -z "$FOUND_FILES" ]; then
    echo "❌ 错误：未找到 output.jsonl 文件"
    echo "请先运行测试并等待完成"
    exit 1
fi

echo "找到以下输出文件："
echo ""
select OUTPUT_FILE in $FOUND_FILES "退出"; do
    if [ "$OUTPUT_FILE" = "退出" ]; then
        echo "已取消评估"
        exit 0
    fi
    
    if [ -n "$OUTPUT_FILE" ]; then
        echo ""
        echo "选择的文件：$OUTPUT_FILE"
        echo ""
        
        # 检查文件大小和行数
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        LINE_COUNT=$(wc -l < "$OUTPUT_FILE")
        
        echo "文件信息："
        echo "  大小：$FILE_SIZE"
        echo "  实例数：$LINE_COUNT"
        echo ""
        
        # 运行评估
        echo "🚀 开始评估..."
        echo ""
        
        ./evaluation/benchmarks/swe_bench/scripts/eval_infer.sh "$OUTPUT_FILE"
        
        echo ""
        echo "========================================="
        echo "✅ 评估完成！"
        echo "========================================="
        echo ""
        
        # 查找并显示报告
        REPORT_DIR=$(dirname "$OUTPUT_FILE")
        
        if [ -f "$REPORT_DIR/report.json" ]; then
            echo "📊 评估报告："
            echo ""
            cat "$REPORT_DIR/report.json" | python3 -m json.tool
            echo ""
        fi
        
        if [ -f "$REPORT_DIR/README.md" ]; then
            echo "📄 详细报告位置："
            echo "  $REPORT_DIR/README.md"
            echo ""
            echo "查看详细报告："
            echo "  cat $REPORT_DIR/README.md"
            echo "  或"
            echo "  less $REPORT_DIR/README.md"
        fi
        
        break
    else
        echo "无效选择，请重试"
    fi
done

