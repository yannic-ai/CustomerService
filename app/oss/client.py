"""对象存储封装：阿里云 OSS，未配置时回退本地目录。"""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import DATA_DIR, get_settings

logger = logging.getLogger("cs.oss")


class OSSStorage:
    """阿里云 OSS；未配置密钥时回退到本地 data/oss。"""

    def __init__(self) -> None:
        """读取配置；凭证齐全则连接 OSS Bucket。"""
        self.settings = get_settings()
        self.local_root = DATA_DIR / "oss"
        self.local_root.mkdir(parents=True, exist_ok=True)
        self._bucket = None
        if self.settings.oss_enabled:
            import oss2

            auth = oss2.Auth(self.settings.oss_access_key_id, self.settings.oss_access_key_secret)
            self._bucket = oss2.Bucket(auth, self.settings.oss_endpoint, self.settings.oss_bucket)

    def put_text(
        self,
        key: str,
        content: str,
        content_type: str = "text/markdown",
        tenant_id: str = "demo",
    ) -> str:
        """上传文本对象，返回 OSS URL 或本地文件路径（按租户分前缀）。"""
        object_key = (
            f"{self.settings.oss_prefix.rstrip('/')}/{tenant_id}/{key.lstrip('/')}"
        )
        payload = content.encode("utf-8")
        if self._bucket is not None:
            self._bucket.put_object(object_key, payload, headers={"Content-Type": content_type})
            url = f"https://{self.settings.oss_bucket}.{self.settings.oss_endpoint}/{object_key}"
            logger.info("uploaded to oss: %s", url)
            return url

        path = self.local_root / object_key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)


def get_oss() -> OSSStorage:
    """创建 OSS 存储客户端。"""
    return OSSStorage()
