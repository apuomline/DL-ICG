#!/bin/bash

# 深度学习流程自动化脚本
# 支持不同数据集的统一处理流程

set -e  # 遇到错误立即退出

# 默认配置参数
DEFAULT_DATASET_TYPE="400"
DEFAULT_CONFIG_DIR="configs"
DEFAULT_SCRIPT_DIR="scripts"
DEFAULT_LOG_DIR="logs"

# 显示使用说明
show_usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --dataset-type TYPE        数据集类型 (400|5000, 默认: $DEFAULT_DATASET_TYPE)"
    echo "  --config-dir DIR           配置目录 (默认: $DEFAULT_CONFIG_DIR)"
    echo "  --script-dir DIR           脚本目录 (默认: $DEFAULT_SCRIPT_DIR)"
    echo "  --log-dir DIR              日志目录 (默认: $DEFAULT_LOG_DIR)"
    echo "  --skip-data-split          跳过数据划分步骤"
    echo "  --skip-training            跳过模型训练步骤"
    echo "  --skip-inference           跳过模型推理步骤"
    echo "  -h, --help                 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --dataset-type 400"
    echo "  $0 --dataset-type 5000 --log-dir logs_5000"
    echo "  $0 --dataset-type 400 --skip-data-split  # 仅运行训练和推理"
    echo "  $0 --help"
}

# 解析命令行参数
parse_arguments() {
    # 默认值
    DATASET_TYPE=$DEFAULT_DATASET_TYPE
    CONFIG_DIR=$DEFAULT_CONFIG_DIR
    SCRIPT_DIR=$DEFAULT_SCRIPT_DIR
    LOG_DIR=$DEFAULT_LOG_DIR
    SKIP_DATA_SPLIT=false
    SKIP_TRAINING=false
    SKIP_INFERENCE=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset-type)
                DATASET_TYPE="$2"
                shift 2
                ;;
            --config-dir)
                CONFIG_DIR="$2"
                shift 2
                ;;
            --script-dir)
                SCRIPT_DIR="$2"
                shift 2
                ;;
            --log-dir)
                LOG_DIR="$2"
                shift 2
                ;;
            --skip-data-split)
                SKIP_DATA_SPLIT=true
                shift
                ;;
            --skip-training)
                SKIP_TRAINING=true
                shift
                ;;
            --skip-inference)
                SKIP_INFERENCE=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                echo "错误: 未知参数 $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# 基于数据集类型生成配置文件路径
get_config_path() {
    local config_type=$1
    echo "${CONFIG_DIR}/${config_type}_${DATASET_TYPE}.yaml"
}

# 基于数据集类型生成日志文件名
get_log_suffix() {
    echo "${DATASET_TYPE}_$1"
}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 函数：检查文件是否存在
check_file_exists() {
    if [ ! -f "$1" ]; then
        echo "错误: 文件 $1 不存在"
        exit 1
    fi
}

# 函数：运行Python脚本并记录日志
run_python_script() {
    local script_name=$1
    local config_file=$2
    local step_name=$3
    local log_suffix=$4
    
    echo "------------------------------------------"
    echo "开始执行: $step_name"
    echo "脚本: $script_name"
    echo "配置文件: $config_file"
    echo "数据集类型: $DATASET_TYPE"
    echo "------------------------------------------"
    
    # 检查文件是否存在
    check_file_exists $script_name
    check_file_exists $config_file
    
    # 运行Python脚本
    python $script_name --config $config_file 2>&1 | tee $LOG_DIR/${TIMESTAMP}_${log_suffix}.log
    
    # 检查执行结果
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ $step_name 执行成功"
    else
        echo "❌ $step_name 执行失败"
        exit 1
    fi
}

# 主函数
main() {
    # 解析命令行参数
    parse_arguments "$@"
    
    # 验证数据集类型
    if [[ "$DATASET_TYPE" != "400" && "$DATASET_TYPE" != "5000" ]]; then
        echo "错误: 数据集类型必须为 400 或 5000"
        exit 1
    fi
    
    # 创建日志目录
    mkdir -p $LOG_DIR

    echo "=========================================="
    echo "开始执行深度学习流程 - $TIMESTAMP"
    echo "数据集类型: $DATASET_TYPE"
    echo "配置目录: $CONFIG_DIR"
    echo "脚本目录: $SCRIPT_DIR"
    echo "日志目录: $LOG_DIR"
    echo "跳过数据划分: $SKIP_DATA_SPLIT"
    echo "跳过训练: $SKIP_TRAINING"
    echo "跳过推理: $SKIP_INFERENCE"
    echo "=========================================="

    # 步骤1: 数据划分（可选）
    if [ "$SKIP_DATA_SPLIT" = false ]; then
        DATA_SPLIT_CONFIG=$(get_config_path "data_split")
        run_python_script \
            "$SCRIPT_DIR/data_split.py" \
            "$DATA_SPLIT_CONFIG" \
            "数据划分" \
            "$(get_log_suffix "data_split")"
    else
        echo "跳过数据划分步骤"
    fi

    # 步骤2: 模型训练（可选）
    if [ "$SKIP_TRAINING" = false ]; then
        TRAIN_CONFIG=$(get_config_path "train")
        run_python_script \
            "$SCRIPT_DIR/effnet_train4_done_yaml.py" \
            "$TRAIN_CONFIG" \
            "模型训练" \
            "$(get_log_suffix "training")"
    else
        echo "跳过模型训练步骤"
    fi

    # 步骤3: 模型推理（可选）
    if [ "$SKIP_INFERENCE" = false ]; then
        INFERENCE_CONFIG=$(get_config_path "inference")
        run_python_script \
            "$SCRIPT_DIR/inference4_arce.py" \
            "$INFERENCE_CONFIG" \
            "模型推理" \
            "$(get_log_suffix "inference")"
    else
        echo "跳过模型推理步骤"
    fi

    echo "=========================================="
    echo "所有步骤执行完成!"
    echo "数据集类型: $DATASET_TYPE"
    echo "日志文件保存在: $LOG_DIR/"
    echo "完成时间: $(date)"
    echo "=========================================="
}

# 执行主函数
main "$@"