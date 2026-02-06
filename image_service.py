"""
图像生成服务层
支持本地模型和远程API两种调用方式
"""

import base64
import httpx
import asyncio
import io
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
        self._pipe = None
    
    def _load_model(self):
        """加载本地模型"""
        if self._pipe is not None:
            return
        
        try:
            import torch
            from diffusers import DiffusionPipeline
            
            print(f"\n{'='*60}")
            print(f"正在加载模型: {self.model_path}")
            print(f"目标设备: {self.device}")
            
            # 检查模型路径是否存在
            model_path = Path(self.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
            
            # 检查设备可用性
            if self.device == "cuda" and not torch.cuda.is_available():
                print("⚠️  警告: CUDA不可用，切换到CPU模式")
                self.device = "cpu"
            elif self.device == "cuda":
                print(f"✅ 检测到 GPU: {torch.cuda.get_device_name(0)}")
                print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # 加载模型
            print("📦 正在加载模型文件...")
            self._pipe = DiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True,
            )
            
            # 移动到指定设备
            print(f"🚀 正在将模型移动到 {self.device}...")
            self._pipe = self._pipe.to(self.device)
            
            # 优化内存使用
            if self.device == "cuda":
                # 启用内存优化
                self._pipe.enable_attention_slicing()
                print("✅ 已启用 Attention Slicing 内存优化")
                
                # 如果支持，启用 xformers
                try:
                    self._pipe.enable_xformers_memory_efficient_attention()
                    print("✅ 已启用 xformers 内存优化")
                except Exception:
                    print("ℹ️  xformers 不可用，跳过（不影响正常使用）")
            
            print(f"✅ 模型加载成功!")
            print(f"{'='*60}\n")
            
        except ImportError as e:
            raise ImportError(
                f"❌ 缺少必要的依赖库。\n"
                f"请安装: uv pip install torch diffusers transformers accelerate\n"
                f"错误详情: {e}"
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"❌ {e}")
        except Exception as e:
            raise RuntimeError(f"❌ 模型加载失败: {e}")
    
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
        import io
        import torch
        
        start_time = time.time()
        
        try:
            # 确保模型已加载
            self._load_model()
            
            # 设置随机种子
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # 在线程池中运行模型推理（避免阻塞异步事件循环）
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(
                None,
                self._generate_sync,
                prompt,
                negative_prompt,
                width,
                height,
                steps,
                guidance_scale,
                generator
            )
            
            # 将PIL图像转换为字节
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            image_bytes = img_byte_arr.getvalue()
            
            # 转换为base64
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            generation_time = time.time() - start_time
            
            return ImageGenerationResult(
                success=True,
                image_data=image_bytes,
                image_base64=image_base64,
                generation_time=generation_time
            )
        except Exception as e:
            return ImageGenerationResult(
                success=False,
                error=f"图像生成失败: {str(e)}",
                generation_time=time.time() - start_time
            )
    
    def _generate_sync(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        generator
    ):
        """同步生成图像（在线程池中执行）"""
        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        # 返回第一张图像
        return result.images[0]


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
