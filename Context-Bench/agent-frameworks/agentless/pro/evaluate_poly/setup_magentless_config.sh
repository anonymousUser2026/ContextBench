#!/bin/bash
# 设置 MagentLess 的配置文件，指向 proxy

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MAGENTLESS_DIR="$PROJECT_ROOT/MagentLess"
API_KEY_FILE="$MAGENTLESS_DIR/script/api_key.sh"

# Proxy 配置（从环境变量或默认值获取）
PROXY_URL="${OPENAI_BASE_URL:-http://localhost:18888}"
PROXY_PORT="${PROXY_PORT:-18888}"
API_KEY="${OPENAI_API_KEY:-dummy-key}"
MODEL="${OPENAI_MODEL:-gpt-5}"
EMBED_URL="${OPENAI_EMBED_URL:-http://localhost:18888/v1}"

echo "配置 MagentLess API 设置..."
echo "Proxy URL: $PROXY_URL"
echo "Proxy Port: $PROXY_PORT"

# 创建 api_key.sh 文件
mkdir -p "$(dirname "$API_KEY_FILE")"
cat > "$API_KEY_FILE" << EOF
export OPENAI_API_KEY="$API_KEY"
export OPENAI_BASE_URL="$PROXY_URL"
export OPENAI_MODEL="$MODEL"
export OPENAI_EMBED_URL="$EMBED_URL"
EOF

echo "配置文件已创建: $API_KEY_FILE"
echo ""
echo "内容:"
cat "$API_KEY_FILE"

