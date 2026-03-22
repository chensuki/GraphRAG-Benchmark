"""
OpenClaw 通知客户端

使用 WebSocket chat.send 发送消息到指定会话。

使用方式:
    from openclaw_notifier import OpenClawClient

    client = OpenClawClient(
        token="your-token",
        session_key="agent:main:feishu:group:oc_xxx",
    )
    client.notify_complete("Benchmark", {"accuracy": 0.95})
"""

import json
import uuid
import logging
import os
import asyncio
import concurrent.futures
import hashlib
import time
from typing import Optional, Dict, Any, Set
from dataclasses import dataclass, field
from datetime import datetime

try:
    import websockets
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ExperimentEvent:
    """实验事件"""
    name: str
    status: str  # started, progress, completed, error
    progress: float = 1.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_message(self) -> str:
        """转换为通知消息"""
        if self.status == "started":
            lines = [
                f"[{self.timestamp}] 实验开始",
                f"名称: {self.name}",
            ]
            if self.context:
                lines.append("配置:")
                for k, v in self.context.items():
                    lines.append(f"  - {k}: {v}")
            return "\n".join(lines)

        elif self.status == "progress":
            lines = [
                f"[{self.timestamp}] 实验进度",
                f"名称: {self.name}",
                f"进度: {self.progress:.1%}",
            ]
            if self.metrics:
                lines.append("当前指标:")
                for k, v in self.metrics.items():
                    lines.append(f"  - {k}: {v}")
            return "\n".join(lines)

        elif self.status == "completed":
            lines = [
                f"[{self.timestamp}] 实验完成",
                f"名称: {self.name}",
                "结果:",
            ]
            if self.metrics:
                for k, v in self.metrics.items():
                    lines.append(f"  - {k}: {v}")
            return "\n".join(lines)

        elif self.status == "error":
            lines = [
                f"[{self.timestamp}] 实验失败",
                f"名称: {self.name}",
                f"错误: {self.error}",
            ]
            if self.context:
                lines.append("上下文:")
                for k, v in self.context.items():
                    lines.append(f"  - {k}: {v}")
            return "\n".join(lines)

        return f"[{self.status}] {self.name}"


class OpenClawClient:
    """
    OpenClaw 通知客户端

    Args:
        gateway_url: Gateway 地址
        token: API token
        session_key: 默认会话 ID
        timeout: WebSocket 操作超时时间（秒）
        max_retries: 最大重试次数
        retry_base_delay: 重试基础延迟（秒）
        enable_dedup: 启用通知去重
        dedup_ttl: 去重缓存 TTL（秒）
    """

    def __init__(
        self,
        gateway_url: str = None,
        token: str = None,
        session_key: str = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        enable_dedup: bool = True,
        dedup_ttl: int = 60,
    ):
        if not WEBSOCKET_AVAILABLE:
            raise ImportError("websockets 库未安装，请运行: pip install websockets")

        self.gateway_url = (
            gateway_url or
            os.environ.get("OPENCLAW_GATEWAY_URL", "http://127.0.0.1:18789")
        ).rstrip("/")
        self.token = token or os.environ.get("OPENCLAW_TOKEN", "")
        self.default_session_key = (
            session_key or
            os.environ.get("OPENCLAW_SESSION_KEY", "")
        )
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.enable_dedup = enable_dedup
        self.dedup_ttl = dedup_ttl
        self._recent_hashes: Set[str] = set()
        self._dedup_timestamps: Dict[str, float] = {}

        if not self.token:
            raise ValueError("必须提供 token 参数或设置环境变量 OPENCLAW_TOKEN")

    def _get_ws_url(self) -> str:
        """将 HTTP URL 转换为 WebSocket URL"""
        return self.gateway_url.replace("http://", "ws://").replace("https://", "wss://")

    def _is_duplicate(self, message: str) -> bool:
        """检查是否为重复消息"""
        if not self.enable_dedup:
            return False

        msg_hash = hashlib.md5(message.encode('utf-8')).hexdigest()[:8]
        now = time.time()

        # 清理过期记录
        expired = [h for h, ts in self._dedup_timestamps.items() if now - ts > self.dedup_ttl]
        for h in expired:
            self._recent_hashes.discard(h)
            del self._dedup_timestamps[h]

        if msg_hash in self._recent_hashes:
            return True

        self._recent_hashes.add(msg_hash)
        self._dedup_timestamps[msg_hash] = now
        return False

    async def _send_connect_request(self, ws, request_id: str) -> None:
        """发送 connect 请求"""
        await ws.send(json.dumps({
            "type": "req",
            "id": f"connect-{request_id}",
            "method": "connect",
            "params": {
                "minProtocol": 3,
                "maxProtocol": 3,
                "role": "operator",
                "scopes": ["operator.read", "operator.write"],
                "caps": [],
                "commands": [],
                "permissions": {},
                "auth": {"token": self.token},
                "client": {
                    "id": "gateway-client",
                    "displayName": "Benchmark Runner",
                    "version": "1.0",
                    "platform": "python",
                    "mode": "cli",
                }
            }
        }))

    def send(
        self,
        message: str,
        session_key: str = None,
        deliver: bool = True,
        timeout_ms: int = 30000,
        skip_dedup: bool = False,
    ) -> Dict[str, Any]:
        """
        发送消息（带重试和去重）

        Args:
            message: 消息内容
            session_key: 会话 ID
            deliver: 是否发送到外部通道
            timeout_ms: 响应超时（毫秒）
            skip_dedup: 跳过去重检查

        Returns:
            API 响应
        """
        if not skip_dedup and self._is_duplicate(message):
            return {"status": "deduplicated", "ok": True}

        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self._send_once(message, session_key, deliver, timeout_ms)
            except (TimeoutError, ConnectionError, OSError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(f"🦞 Send failed, retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
                else:
                    raise
            except Exception:
                raise

        raise last_error or RuntimeError("Unexpected state")

    def _send_once(
        self,
        message: str,
        session_key: str = None,
        deliver: bool = True,
        timeout_ms: int = 30000,
    ) -> Dict[str, Any]:
        """单次发送"""
        effective_session_key = session_key or self.default_session_key
        ws_url = self._get_ws_url()

        async def _ws_send():
            request_id = str(uuid.uuid4())[:8]
            logger.info(f"🦞 WebSocket: {ws_url}")
            logger.info(f"🦞 sessionKey: {effective_session_key}")

            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
                connect_request_id = f"connect-{request_id}"
                send_request_id = f"send-{request_id}"
                connect_sent = False

                # 握手
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                    result = json.loads(response)

                    if result.get("type") == "event":
                        if result.get("event") == "connect.challenge":
                            nonce = result.get("payload", {}).get("nonce", "")
                            if not nonce:
                                raise ConnectionError("Missing nonce")
                            await self._send_connect_request(ws, request_id)
                            connect_sent = True
                        continue

                    if result.get("type") == "res" and result.get("id") == connect_request_id:
                        if not result.get("ok"):
                            raise ConnectionError(f"Connect failed: {result}")
                        break

                if not connect_sent:
                    raise ConnectionError("No connect.challenge received")

                # 发送消息
                await ws.send(json.dumps({
                    "type": "req",
                    "id": send_request_id,
                    "method": "chat.send",
                    "params": {
                        "sessionKey": effective_session_key,
                        "message": message,
                        "deliver": deliver,
                        "idempotencyKey": str(uuid.uuid4()),
                        "timeoutMs": timeout_ms,
                    }
                }))

                # 等待响应
                while True:
                    resp = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                    result = json.loads(resp)

                    if result.get("type") == "event":
                        continue

                    if result.get("type") == "res" and result.get("id") == send_request_id:
                        if result.get("ok"):
                            return result.get("payload", {"status": "ok"})
                        raise RuntimeError(f"chat.send failed: {result}")

            raise TimeoutError("WebSocket operation timeout")

        # 执行异步
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _ws_send())
                    return future.result(timeout=self.timeout + 5)
            return loop.run_until_complete(_ws_send())
        except RuntimeError:
            return asyncio.run(_ws_send())

    def notify(self, event: ExperimentEvent, session_key: str = None) -> Dict[str, Any]:
        """发送实验事件通知"""
        return self.send(message=event.to_message(), session_key=session_key)

    def notify_start(self, name: str, config: Dict[str, Any] = None, session_key: str = None) -> Dict[str, Any]:
        """通知实验开始"""
        return self.notify(ExperimentEvent(name=name, status="started", context=config or {}), session_key)

    def notify_progress(self, name: str, progress: float, metrics: Dict[str, Any] = None, session_key: str = None) -> Dict[str, Any]:
        """通知实验进度"""
        return self.notify(ExperimentEvent(name=name, status="progress", progress=progress, metrics=metrics or {}), session_key)

    def notify_complete(self, name: str, results: Dict[str, Any] = None, session_key: str = None) -> Dict[str, Any]:
        """通知实验完成"""
        return self.notify(ExperimentEvent(name=name, status="completed", metrics=results or {}), session_key)

    def notify_error(self, name: str, error: Exception, context: Dict[str, Any] = None, session_key: str = None) -> Dict[str, Any]:
        """通知实验错误"""
        return self.notify(
            ExperimentEvent(name=name, status="error", error=f"{type(error).__name__}: {error}", context=context or {}),
            session_key
        )

    def notify_evaluation(
        self,
        name: str,
        metrics: Dict[str, float],
        comparison: Optional[Dict[str, float]] = None,
        session_key: str = None,
    ) -> Dict[str, Any]:
        """通知评估结果"""
        lines = [
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 评估完成",
            f"实验: {name}",
            "指标:",
        ]

        for k, v in metrics.items():
            if comparison and k in comparison:
                delta = v - comparison[k]
                sign = "+" if delta >= 0 else ""
                lines.append(f"  - {k}: {v:.4f} ({sign}{delta:.4f})")
            else:
                lines.append(f"  - {k}: {v:.4f}")

        return self.send("\n".join(lines), session_key=session_key)