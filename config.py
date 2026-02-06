"""
配置管理模块
读取和管理模型配置，支持本地模型和远程API两种调用方式
"""

import yaml
from pathlib import Path
from typing import Optional
from pydantic import BaseModel


class ModelParams(BaseModel):
    """模型默认参数"""
    width: int = 512
    height: int = 512
    steps: int = 20
    guidance_scale: float = 7.5


class LocalModelConfig(BaseModel):
    """本地模型配置"""
    name: str
    type: str = "local"
    model_path: str
    device: str = "cuda"
    default_params: ModelParams = ModelParams()


class RemoteModelConfig(BaseModel):
    """远程API模型配置"""
    name: str
    type: str = "remote"
    endpoint: str
    api_key: str
    default_params: ModelParams = ModelParams()


class ServerConfig(BaseModel):
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8000


class OutputConfig(BaseModel):
    """输出配置"""
    save_dir: str = "./generated_images"
    format: str = "png"


class AppConfig:
    """应用配置管理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config: dict = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)
    
    def reload(self) -> None:
        """重新加载配置（热重载）"""
        self._load_config()
    
    @property
    def models(self) -> dict:
        """获取所有模型配置"""
        return self._config.get("models", {})
    
    @property
    def default_model(self) -> str:
        """获取默认模型ID"""
        return self._config.get("default_model", "")
    
    @property
    def server(self) -> ServerConfig:
        """获取服务器配置"""
        server_data = self._config.get("server", {})
        return ServerConfig(**server_data)
    
    @property
    def output(self) -> OutputConfig:
        """获取输出配置"""
        output_data = self._config.get("output", {})
        return OutputConfig(**output_data)
    
    def get_model_config(self, model_id: str) -> Optional[dict]:
        """获取指定模型的配置"""
        return self.models.get(model_id)
    
    def get_model_list(self) -> list[dict]:
        """获取模型列表（用于前端展示）"""
        result = []
        for model_id, model_config in self.models.items():
            result.append({
                "id": model_id,
                "name": model_config.get("name", model_id),
                "type": model_config.get("type", "unknown"),
                "default_params": model_config.get("default_params", {})
            })
        return result
    
    def is_local_model(self, model_id: str) -> bool:
        """判断是否为本地模型"""
        config = self.get_model_config(model_id)
        return config.get("type") == "local" if config else False
    
    def is_remote_model(self, model_id: str) -> bool:
        """判断是否为远程模型"""
        config = self.get_model_config(model_id)
        return config.get("type") == "remote" if config else False


# 全局配置实例
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """获取配置实例（单例模式）"""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reload_config() -> None:
    """重新加载配置"""
    global _config
    if _config is not None:
        _config.reload()
