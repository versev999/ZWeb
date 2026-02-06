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
    
    def __init__(self, model_path: str, device: str = "cuda", enable_optimizations: bool = True):
        self.model_path = model_path
        self.device = device
        self.enable_optimizations = enable_optimizations
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
            
            # 尝试加载VAE（如果存在单独的VAE）
            vae = None
            try:
                from diffusers import AutoencoderKL
                vae_path = model_path / "vae"
                if vae_path.exists():
                    print("📦 检测到独立VAE，正在加载...")
                    vae = AutoencoderKL.from_pretrained(
                        str(vae_path),
                        torch_dtype=torch.float32,  # VAE使用float32避免NaN
                    )
                    print("✅ VAE加载成功 (float32)")
            except Exception as e:
                print(f"ℹ️  未加载独立VAE: {e}")
            
            # 加载主模型
            # Z-Image模型需要使用float32避免NaN黑图问题
            load_kwargs = {
                "torch_dtype": torch.float32,
                "use_safetensors": True,
            }
            
            # 如果有独立VAE，使用它
            if vae is not None:
                load_kwargs["vae"] = vae
            
            self._pipe = DiffusionPipeline.from_pretrained(
                self.model_path,
                **load_kwargs
            )
            
            print("ℹ️  使用 float32 精度（Z-Image模型要求）")
            
            # 移动到指定设备
            print(f"🚀 正在将模型移动到 {self.device}...")
            self._pipe = self._pipe.to(self.device)
            
            # 检查模型组件
            print(f"📋 模型组件:")
            if hasattr(self._pipe, 'vae'):
                print(f"   ✅ VAE: {type(self._pipe.vae).__name__} ({self._pipe.vae.dtype})")
            if hasattr(self._pipe, 'transformer'):
                print(f"   ✅ Transformer: {type(self._pipe.transformer).__name__} ({self._pipe.transformer.dtype})")
            elif hasattr(self._pipe, 'unet'):
                print(f"   ✅ UNet: {type(self._pipe.unet).__name__} ({self._pipe.unet.dtype})")
            if hasattr(self._pipe, 'text_encoder'):
                print(f"   ✅ Text Encoder: {type(self._pipe.text_encoder).__name__}")
            
            # 应用优化
            if self.device == "cuda" and self.enable_optimizations:
                self._apply_optimizations()
            
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
    
    def _apply_optimizations(self):
        """应用各种推理优化"""
        import torch
        
        print(f"\n🚀 应用推理优化:")
        optimizations_applied = []
        
        # 1. Attention Slicing - 降低内存占用
        try:
            self._pipe.enable_attention_slicing(slice_size="auto")
            optimizations_applied.append("✅ Attention Slicing")
        except Exception as e:
            print(f"   ⚠️  Attention Slicing 失败: {e}")
        
        # 2. xformers - 内存高效的注意力机制
        try:
            self._pipe.enable_xformers_memory_efficient_attention()
            optimizations_applied.append("✅ xformers 内存优化")
        except Exception as e:
            optimizations_applied.append("ℹ️  xformers 不可用")
        
        # 3. VAE Slicing - 降低VAE解码的内存占用
        try:
            if hasattr(self._pipe, 'enable_vae_slicing'):
                self._pipe.enable_vae_slicing()
                optimizations_applied.append("✅ VAE Slicing")
        except Exception:
            pass
        
        # 4. VAE Tiling - 处理大图像时分块解码
        try:
            if hasattr(self._pipe, 'enable_vae_tiling'):
                self._pipe.enable_vae_tiling()
                optimizations_applied.append("✅ VAE Tiling")
        except Exception:
            pass
        
        # 5. Torch Compile (PyTorch 2.0+) - 最强加速
        # 注意: Z-Image模型的RoPE实现与torch.compile不兼容，暂时禁用
        # 如果是标准的Stable Diffusion模型，可以启用此优化
        if False and torch.__version__ >= "2.0.0":
            try:
                # 编译 UNet/Transformer 以获得最大加速
                if hasattr(self._pipe, 'transformer'):
                    self._pipe.transformer = torch.compile(
                        self._pipe.transformer,
                        mode="reduce-overhead",
                        fullgraph=True
                    )
                    optimizations_applied.append("✅ Torch Compile (Transformer)")
                elif hasattr(self._pipe, 'unet'):
                    self._pipe.unet = torch.compile(
                        self._pipe.unet,
                        mode="reduce-overhead",
                        fullgraph=True
                    )
                    optimizations_applied.append("✅ Torch Compile (UNet)")
            except Exception as e:
                optimizations_applied.append(f"ℹ️  Torch Compile 跳过: {str(e)[:50]}")
        
        # 6. CUDA优化
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            optimizations_applied.append("✅ CUDA TF32 + cuDNN Benchmark")
        except Exception:
            pass
        
        # 输出优化结果
        for opt in optimizations_applied:
            print(f"   {opt}")
        print()
    
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
        import numpy as np
        from PIL import Image
        
        # 生成图像（输出numpy数组）
        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="np",
        )
        
        # 获取numpy数组
        images_np = result.images
        
        # 检查异常值（调试用）
        if np.isnan(images_np).any() or np.isinf(images_np).any():
            print("⚠️  警告: 检测到NaN或Inf值，正在清理...")
            images_np = np.nan_to_num(images_np, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 确保值在 [0, 1] 范围内
        images_np = np.clip(images_np, 0, 1)
        
        # 转换为PIL图像
        image_np = images_np[0]  # 取第一张图像
        image_uint8 = (image_np * 255).round().astype(np.uint8)
        
        # 创建PIL图像
        if image_uint8.shape[-1] == 3:
            image = Image.fromarray(image_uint8, mode='RGB')
        else:
            image = Image.fromarray(image_uint8.squeeze(), mode='L')
        
        return image


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
                    device=model_config.get("device", "cuda"),
                    enable_optimizations=model_config.get("enable_optimizations", True)
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
