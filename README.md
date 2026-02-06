# Z-Image 图像生成系统

基于大模型的 AI 图像生成服务，支持本地模型和远程 API 两种调用方式。

## 🚀 快速开始

### 安装依赖



```bash
# 安装依赖
uv sync
```




### 启动服务

```bash
uv run uvicorn main:app --reload
```

服务启动后访问：
- 🌐 **前端界面**：http://localhost:8000
- 📚 **API 文档**：http://localhost:8000/docs

## 📁 项目结构

```
ZWeb/
├── main.py              # FastAPI 主入口
├── config.py            # 配置管理模块
├── config.yaml          # 模型和参数配置
├── image_service.py     # 图像生成服务层
├── static/              # 前端静态文件
│   ├── index.html       # 主页面
│   ├── style.css        # 样式
│   └── app.js           # 交互逻辑
└── pyproject.toml       # 项目依赖
```

## ⚙️ 配置说明

编辑 `config.yaml` 配置模型：


**注意事项：**
- 本地模型需要支持 Diffusers 格式
- 首次加载模型会比较慢，请耐心等待
- 如果没有 GPU，可以将 `device` 改为 `cpu`（速度会慢很多）
- Turbo 版本使用更少的推理步数，生成速度更快但质量可能略有下降

### 远程 API 模型

```yaml
models:
  my-remote-model:
    name: "远程模型"
    type: "remote"
    endpoint: "https://api.example.com/v1/generate"
    api_key: "your-api-key"
    default_params:
      width: 1024
      height: 1024
      steps: 30
      guidance_scale: 8.0
```

## 🔌 API 接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/generate` | POST | 生成图像 |
| `/api/models` | GET | 获取模型列表 |
| `/api/config/{model_id}` | GET | 获取模型配置 |
| `/api/config/reload` | POST | 重载配置 |

### 生成图像示例

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over mountains",
    "model_id": "z-image-local",
    "width": 512,
    "height": 512,
    "steps": 20
  }'
```

## 📝 许可证

MIT License
