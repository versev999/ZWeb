"""
Z-Image 图像生成系统 - FastAPI 后端
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from config import get_config, reload_config
from image_service import get_image_service


# 创建 FastAPI 应用
app = FastAPI(
    title="Z-Image 图像生成系统",
    description="基于 z-image 大模型的图像生成服务",
    version="1.0.0"
)


# ==================== 请求/响应模型 ====================

class GenerateRequest(BaseModel):
    """图像生成请求"""
    prompt: str
    model_id: Optional[str] = None
    negative_prompt: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    save_to_file: bool = False


class GenerateResponse(BaseModel):
    """图像生成响应"""
    success: bool
    image_base64: Optional[str] = None
    file_path: Optional[str] = None
    error: Optional[str] = None
    generation_time: float = 0.0


# ==================== API 端点 ====================

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest):
    """
    生成图像
    
    - **prompt**: 图像描述提示词
    - **model_id**: 模型ID（可选，默认使用配置中的默认模型）
    - **negative_prompt**: 负面提示词
    - **width**: 图像宽度
    - **height**: 图像高度
    - **steps**: 生成步数
    - **guidance_scale**: 引导系数
    - **seed**: 随机种子
    - **save_to_file**: 是否保存到文件
    """
    service = get_image_service()
    
    result = await service.generate_image(
        prompt=request.prompt,
        model_id=request.model_id,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        steps=request.steps,
        guidance_scale=request.guidance_scale,
        seed=request.seed,
        save_to_file=request.save_to_file
    )
    
    return GenerateResponse(**result.to_dict())


@app.get("/api/models")
async def get_models():
    """获取可用模型列表"""
    config = get_config()
    return {
        "models": config.get_model_list(),
        "default_model": config.default_model
    }


@app.get("/api/config/{model_id}")
async def get_model_config(model_id: str):
    """获取指定模型的配置"""
    config = get_config()
    model_config = config.get_model_config(model_id)
    
    if not model_config:
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")
    
    # 不返回敏感信息（如 api_key）
    safe_config = {
        "id": model_id,
        "name": model_config.get("name"),
        "type": model_config.get("type"),
        "default_params": model_config.get("default_params", {})
    }
    
    return safe_config


@app.post("/api/config/reload")
async def reload_configuration():
    """重新加载配置"""
    try:
        reload_config()
        return {"success": True, "message": "配置已重新加载"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 静态文件服务 ====================

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_index():
    """提供前端主页"""
    return FileResponse("static/index.html")


# ==================== 启动入口 ====================

def main():
    """启动服务"""
    config = get_config()
    server_config = config.server
    
    print(f"🚀 Z-Image 服务启动中...")
    print(f"📍 访问地址: http://{server_config.host}:{server_config.port}")
    print(f"📚 API文档: http://{server_config.host}:{server_config.port}/docs")
    
    uvicorn.run(
        "main:app",
        host=server_config.host,
        port=server_config.port,
        reload=True
    )


if __name__ == "__main__":
    main()
