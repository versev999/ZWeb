"""
图像生成服务层
支持本地模型和远程API两种调用方式
"""

import base64
import httpx
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime
from abc import ABC, abstractmethod

from config import get_config


class ImageGenerationResult:
    """图像生成结果"""
    
    def __init__(
        self,
        success: bool,
        image_data: Optional[bytes] = None,
        image_base64: Optional[str] = None,
        file_path: Optional[str] = None,
        error: Optional[str] = None,
        generation_time: float = 0.0
    ):
        self.success = success
        self.image_data = image_data
        self.image_base64 = image_base64
        self.file_path = file_path
        self.error = error
        self.generation_time = generation_time
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "image_base64": self.image_base64,
            "file_path": self.file_path,
            "error": self.error,
            "generation_time": self.generation_time
        }


class BaseImageGenerator(ABC):
    """图像生成器基类"""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> ImageGenerationResult:
        """生成图像"""
        pass


class LocalModelGenerator(BaseImageGenerator):
    """本地模型图像生成器"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._model = None
    
    def _load_model(self):
        """加载本地模型（需要根据实际模型实现）"""
        # TODO: 根据实际的 z-image 模型实现加载逻辑
        # 示例: self._model = ZImageModel.from_pretrained(self.model_path)
        pass
    
    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> ImageGenerationResult:
        """使用本地模型生成图像"""
        import time
        start_time = time.time()
        
        try:
            # TODO: 实现实际的本地模型调用
            # 目前返回模拟结果
            await asyncio.sleep(0.5)  # 模拟生成时间
            
            # 生成一个简单的占位图像（灰色图像）
            # 实际实现时替换为真实的模型调用
            placeholder_image = self._generate_placeholder(width, height)
            image_base64 = base64.b64encode(placeholder_image).decode("utf-8")
            
            generation_time = time.time() - start_time
            
            return ImageGenerationResult(
                success=True,
                image_data=placeholder_image,
                image_base64=image_base64,
                generation_time=generation_time
            )
        except Exception as e:
            return ImageGenerationResult(
                success=False,
                error=str(e),
                generation_time=time.time() - start_time
            )
    
    def _generate_placeholder(self, width: int, height: int) -> bytes:
        """生成占位图像（用于测试）"""
        # 生成一个简单的 PNG 图像
        # 这是一个最小的有效 PNG（1x1 灰色像素，然后缩放说明）
        import struct
        import zlib
        
        def create_png(w: int, h: int) -> bytes:
            """创建简单的灰色 PNG 图像"""
            # PNG 签名
            signature = b'\x89PNG\r\n\x1a\n'
            
            # IHDR chunk
            ihdr_data = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data)
            ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
            
            # IDAT chunk (简单的灰色图像)
            raw_data = b''
            for _ in range(h):
                raw_data += b'\x00'  # filter byte
                for _ in range(w):
                    raw_data += b'\x80\x80\x80'  # RGB 灰色
            
            compressed = zlib.compress(raw_data)
            idat_crc = zlib.crc32(b'IDAT' + compressed)
            idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)
            
            # IEND chunk
            iend_crc = zlib.crc32(b'IEND')
            iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
            
            return signature + ihdr + idat + iend
        
        return create_png(width, height)


class RemoteAPIGenerator(BaseImageGenerator):
    """远程API图像生成器"""
    
    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key
    
    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> ImageGenerationResult:
        """通过远程API生成图像"""
        import time
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "guidance_scale": guidance_scale,
                }
                
                if seed is not None:
                    payload["seed"] = seed
                
                response = await client.post(
                    self.endpoint,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # 假设API返回 base64 编码的图像
                    image_base64 = result.get("image", result.get("data", ""))
                    
                    return ImageGenerationResult(
                        success=True,
                        image_base64=image_base64,
                        generation_time=time.time() - start_time
                    )
                else:
                    return ImageGenerationResult(
                        success=False,
                        error=f"API错误: {response.status_code} - {response.text}",
                        generation_time=time.time() - start_time
                    )
                    
        except httpx.TimeoutException:
            return ImageGenerationResult(
                success=False,
                error="请求超时",
                generation_time=time.time() - start_time
            )
        except Exception as e:
            return ImageGenerationResult(
                success=False,
                error=str(e),
                generation_time=time.time() - start_time
            )


class ImageService:
    """图像生成服务"""
    
    def __init__(self):
        self.config = get_config()
        self._generators: dict[str, BaseImageGenerator] = {}
    
    def _get_generator(self, model_id: str) -> BaseImageGenerator:
        """获取或创建图像生成器"""
        if model_id not in self._generators:
            model_config = self.config.get_model_config(model_id)
            if not model_config:
                raise ValueError(f"未找到模型配置: {model_id}")
            
            model_type = model_config.get("type", "")
            
            if model_type == "local":
                self._generators[model_id] = LocalModelGenerator(
                    model_path=model_config.get("model_path", ""),
                    device=model_config.get("device", "cuda")
                )
            elif model_type == "remote":
                self._generators[model_id] = RemoteAPIGenerator(
                    endpoint=model_config.get("endpoint", ""),
                    api_key=model_config.get("api_key", "")
                )
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
        
        return self._generators[model_id]
    
    async def generate_image(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        save_to_file: bool = False
    ) -> ImageGenerationResult:
        """生成图像"""
        # 使用默认模型
        if not model_id:
            model_id = self.config.default_model
        
        # 获取模型配置
        model_config = self.config.get_model_config(model_id)
        if not model_config:
            return ImageGenerationResult(
                success=False,
                error=f"未找到模型: {model_id}"
            )
        
        # 使用模型默认参数填充缺失参数
        default_params = model_config.get("default_params", {})
        width = width or default_params.get("width", 512)
        height = height or default_params.get("height", 512)
        steps = steps or default_params.get("steps", 20)
        guidance_scale = guidance_scale or default_params.get("guidance_scale", 7.5)
        
        # 获取生成器并生成图像
        generator = self._get_generator(model_id)
        result = await generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        # 保存到文件
        if save_to_file and result.success and result.image_data:
            file_path = self._save_image(result.image_data)
            result.file_path = file_path
        
        return result
    
    def _save_image(self, image_data: bytes) -> str:
        """保存图像到文件"""
        output_config = self.config.output
        save_dir = Path(output_config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"generated_{timestamp}.{output_config.format}"
        file_path = save_dir / filename
        
        with open(file_path, "wb") as f:
            f.write(image_data)
        
        return str(file_path)


# 全局服务实例
_service: Optional[ImageService] = None


def get_image_service() -> ImageService:
    """获取图像服务实例"""
    global _service
    if _service is None:
        _service = ImageService()
    return _service
