# Windows启动脚本 - CNN AI服务

Write-Host "=====================================" -ForegroundColor Green
Write-Host "        CNN AI服务启动脚本          " -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host "本脚本将启动所有CNN AI相关服务：" -ForegroundColor Yellow
Write-Host "1. LabelStudio (端口: 16701)"
Write-Host "2. MLflow (端口: 16702)"
Write-Host "3. 训练服务 (包含LLaMA-Factory: 17860, NL2SQL: 17702)"
Write-Host "=====================================" -ForegroundColor Green

# 检查是否安装了Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "错误：未安装Docker！请先安装Docker Desktop。" -ForegroundColor Red
    Read-Host "按Enter键退出..."
    exit 1
}

# 检查Docker是否正在运行
if (-not (docker info | Select-String -Pattern "Server Version")) {
    Write-Host "错误：Docker未运行！请先启动Docker Desktop。" -ForegroundColor Red
    Read-Host "按Enter键退出..."
    exit 1
}

# 设置环境变量
$env:DATA_PATH = "D:\workspace_ai\CNNSession\data"
$env:MODEL_PATH = "D:\workspace_ai\CNNSession\model"
$env:MLFLOW_PATH = "D:\workspace_ai\CNNSession\mlflow"
$env:HUGGINGFACE_PATH = "D:\workspace_ai\CNNSession\huggingface"

# 显示环境变量
Write-Host "\n环境变量设置：" -ForegroundColor Cyan
Write-Host "DATA_PATH: $env:DATA_PATH"
Write-Host "MODEL_PATH: $env:MODEL_PATH"
Write-Host "MLFLOW_PATH: $env:MLFLOW_PATH"
Write-Host "HUGGINGFACE_PATH: $env:HUGGINGFACE_PATH"

# 创建必要的目录
Write-Host "\n创建必要的目录..." -ForegroundColor Cyan
foreach ($dir in $env:DATA_PATH, $env:MODEL_PATH, $env:MLFLOW_PATH, $env:HUGGINGFACE_PATH) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "   ✅ 创建目录: $dir"
    } else {
        Write-Host "   ✅ 目录已存在: $dir"
    }
}

# 询问用户操作
Write-Host "\n请选择操作：" -ForegroundColor Yellow
Write-Host "1. 启动所有服务"
Write-Host "2. 停止所有服务"
Write-Host "3. 重启所有服务"
Write-Host "4. 查看服务状态"
$choice = Read-Host "请输入选项 (1-4)"

switch ($choice) {
    "1" {
        Write-Host "\n启动所有服务..." -ForegroundColor Cyan
        docker-compose up -d
        
        # 等待服务启动
        Write-Host "\n等待服务启动..." -ForegroundColor Cyan
        Start-Sleep -Seconds 10
        
        # 显示服务状态
        docker-compose ps
        
        # 显示访问地址
        Write-Host "\n=====================================" -ForegroundColor Green
        Write-Host "          服务启动成功！           " -ForegroundColor Green
        Write-Host "=====================================" -ForegroundColor Green
        Write-Host "访问地址：" -ForegroundColor Yellow
        Write-Host "- LabelStudio: http://localhost:16701"
        Write-Host "- MLflow: http://localhost:16702"
        Write-Host "- LLaMA-Factory Web界面: http://localhost:17860"
        Write-Host "- NL2SQL服务: http://localhost:17702"
        Write-Host "=====================================" -ForegroundColor Green
    }
    
    "2" {
        Write-Host "\n停止所有服务..." -ForegroundColor Cyan
        docker-compose down
        Write-Host "\n服务已停止！" -ForegroundColor Green
    }
    
    "3" {
        Write-Host "\n重启所有服务..." -ForegroundColor Cyan
        docker-compose down
        docker-compose up -d
        
        # 等待服务启动
        Write-Host "\n等待服务启动..." -ForegroundColor Cyan
        Start-Sleep -Seconds 10
        
        # 显示服务状态
        docker-compose ps
        
        # 显示访问地址
        Write-Host "\n=====================================" -ForegroundColor Green
        Write-Host "          服务重启成功！           " -ForegroundColor Green
        Write-Host "=====================================" -ForegroundColor Green
        Write-Host "访问地址：" -ForegroundColor Yellow
        Write-Host "- LabelStudio: http://localhost:16701"
        Write-Host "- MLflow: http://localhost:16702"
        Write-Host "- LLaMA-Factory Web界面: http://localhost:17860"
        Write-Host "- NL2SQL服务: http://localhost:17702"
        Write-Host "=====================================" -ForegroundColor Green
    }
    
    "4" {
        Write-Host "\n服务状态：" -ForegroundColor Cyan
        docker-compose ps
    }
    
    default {
        Write-Host "\n错误：无效选项！" -ForegroundColor Red
    }
}

Write-Host "\n按Enter键退出..." -ForegroundColor Yellow
Read-Host