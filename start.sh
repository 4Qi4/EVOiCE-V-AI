#!/bin/bash
# EVOiCE启动脚本

echo "========================================="
echo "      EVOiCE - 简易歌声编辑器"
echo "========================================="

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误：未找到Python3。请先安装Python 3.6或更高版本。"
    exit 1
fi

# 检查Python版本
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
REQUIRED_VERSION="3.6"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "警告：Python版本 $PYTHON_VERSION 可能不兼容，建议使用Python $REQUIRED_VERSION 或更高版本。"
fi

echo "检测到Python版本：$PYTHON_VERSION"

# 检查依赖包
echo "正在检查依赖包..."
MISSING_PACKAGES=()

for package in numpy scipy matplotlib pygame pyworld soundfile librosa; do
    if ! python3 -c "import $package" 2>/dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

# 安装缺失的依赖包
if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo "发现缺失的依赖包：${MISSING_PACKAGES[*]}"
    echo "正在安装依赖包..."
    
    if python3 -m pip install "${MISSING_PACKAGES[@]}"; then
        echo "依赖包安装成功！"
    else
        echo "错误：依赖包安装失败。请尝试运行 install.py 脚本。"
        exit 1
    fi
else
    echo "所有依赖包已安装。"
fi

# 检查音源文件夹
if [ ! -d "voicebank" ] || [ -z "$(ls -A voicebank 2>/dev/null)" ]; then
    echo "正在创建示例音源库..."
    if python3 create_voicebank.py; then
        echo "示例音源库创建成功！"
    else
        echo "警告：无法创建示例音源库。程序可能无法正常运行。"
    fi
fi

# 启动主程序
echo "正在启动EVOiCE编辑器..."
echo "提示：使用 Ctrl+C 可以退出程序"
echo "========================================="
echo ""

python3 evoice.py

# 程序退出后的清理工作
echo ""
echo "========================================="
echo "EVOiCE编辑器已退出。"