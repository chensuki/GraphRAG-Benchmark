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
from typing import Optional, Dict, Any, Callable
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

    使用 WebSocket chat.send 发送消息到指定会话。

    Args:
        gateway_url: Gateway 地址，默认 http://127.0.0.1:18789
        token: API token
        session_key: 默认会话 ID，格式 agent:{agentId}:{platform}:{type}:{id}
        timeout: WebSocket 操作超时时间（秒）
    """

    def __init__(
        self,
        gateway_url: str = None,
        token: str = None,
        session_key: str = None,
        timeout: int = 30,
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

        if not self.token:
            raise ValueError("必须提供 token 参数或设置环境变量 OPENCLAW_TOKEN")

    def _get_ws_url(self) -> str:
        """将 HTTP URL 转换为 WebSocket URL"""
        return self.gateway_url.replace("http://", "ws://").replace("https://", "wss://")

    async def _send_connect_request(self, ws, request_id: str, nonce: Optional[str] = None) -> None:
        """发送 OpenClaw Gateway `connect` 请求。

        注意：OpenClaw 的握手流程是先收到 `connect.challenge` 事件，
        然后再发送 `connect` 请求；并不是再发一个 `connect.challenge` 方法调用。
        """
        if nonce:
            logger.info("🦞 Received challenge nonce, sending connect request")

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
    ) -> Dict[str, Any]:
        """
        通过 WebSocket chat.send 发送消息

        Args:
            message: 消息内容
            session_key: 会话 ID（可选，使用默认值）
            deliver: 是否发送到外部通道（飞书群）
            timeout_ms: Agent 响应超时（毫秒）

        Returns:
            API 响应字典
        """
        effective_session_key = session_key or self.default_session_key
        ws_url = self._get_ws_url()

        async def _ws_send():
            request_id = str(uuid.uuid4())[:8]
            idempotency_key = str(uuid.uuid4())

            logger.info(f"🦞 WebSocket connect: {ws_url}")
            logger.info(f"🦞 chat.send sessionKey: {effective_session_key}")
            logger.info(f"🦞 message: {message[:100]}{'...' if len(message) > 100 else ''}")

            try:
                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    connect_request_id = f"connect-{request_id}"
                    send_request_id = f"send-{request_id}"
                    connect_sent = False

                    # Step 1: 等待 connect.challenge，然后发送 connect 请求
                    while True:
                        response = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                        result = json.loads(response)

                        if result.get("type") == "event":
                            event_name = result.get("event")
                            if event_name == "connect.challenge":
                                nonce = result.get("payload", {}).get("nonce", "")
                                if not nonce:
                                    raise ConnectionError("connect.challenge missing nonce")
                                await self._send_connect_request(ws, request_id, nonce)
                                connect_sent = True
                                continue
                            # 连接建立前可能会收到 tick 等事件，直接忽略
                            continue

                        if result.get("type") == "res" and result.get("id") == connect_request_id:
                            if not result.get("ok"):
                                raise ConnectionError(f"Connect failed: {result}")
                            break

                    if not connect_sent:
                        raise ConnectionError("Gateway did not send connect.challenge before connect response")

                    logger.info("🦞 WebSocket connected")

                    # Step 2: chat.send
                    await ws.send(json.dumps({
                        "type": "req",
                        "id": send_request_id,
                        "method": "chat.send",
                        "params": {
                            "sessionKey": effective_session_key,
                            "message": message,
                            "deliver": deliver,
                            "idempotencyKey": idempotency_key,
                            "timeoutMs": timeout_ms,
                        }
                    }))

                    # Step 3: 等待 chat.send 对应响应（忽略无关 event/tick）
                    while True:
                        send_response = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                        send_result = json.loads(send_response)

                        if send_result.get("type") == "event":
                            continue

                        if send_result.get("type") == "res" and send_result.get("id") == send_request_id:
                            logger.info(f"🦞 chat.send response: {send_result.get('ok', False)}")
                            if send_result.get("ok"):
                                return send_result.get("payload", {"status": "started"})
                            raise RuntimeError(f"chat.send failed: {send_result}")

            except asyncio.TimeoutError:
                raise TimeoutError("WebSocket operation timeout")

        # 在事件循环中运行
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _ws_send())
                    return future.result(timeout=self.timeout + 5)
            else:
                return loop.run_until_complete(_ws_send())
        except RuntimeError:
            return asyncio.run(_ws_send())

    def notify(self, event: ExperimentEvent, session_key: str = None) -> Dict[str, Any]:
        """发送实验事件通知"""
        return self.send(message=event.to_message(), session_key=session_key)

    def notify_start(
        self,
        name: str,
        config: Dict[str, Any] = None,
        session_key: str = None,
    ) -> Dict[str, Any]:
        """通知实验开始"""
        return self.notify(
            ExperimentEvent(name=name, status="started", context=config or {}),
            session_key=session_key,
        )

    def notify_progress(
        self,
        name: str,
        progress: float,
        metrics: Dict[str, Any] = None,
        session_key: str = None,
    ) -> Dict[str, Any]:
        """通知实验进度"""
        return self.notify(
            ExperimentEvent(
                name=name,
                status="progress",
                progress=progress,
                metrics=metrics or {},
            ),
            session_key=session_key,
        )

    def notify_complete(
        self,
        name: str,
        results: Dict[str, Any] = None,
        session_key: str = None,
    ) -> Dict[str, Any]:
        """通知实验完成"""
        return self.notify(
            ExperimentEvent(name=name, status="completed", metrics=results or {}),
            session_key=session_key,
        )

    def notify_error(
        self,
        name: str,
        error: Exception,
        context: Dict[str, Any] = None,
        session_key: str = None,
    ) -> Dict[str, Any]:
        """通知实验错误"""
        return self.notify(
            ExperimentEvent(
                name=name,
                status="error",
                error=f"{type(error).__name__}: {error}",
                context=context or {},
            ),
            session_key=session_key,
        )

    def send_custom(self, message: str, session_key: str = None) -> Dict[str, Any]:
        """发送自定义消息"""
        return self.send(message, session_key=session_key)


# 别名
OpenClawWebhookClient = OpenClawClient


class ExperimentTracker:
    """实验追踪器（上下文管理器）"""

    def __init__(
        self,
        name: str,
        client: OpenClawClient,
        session_key: str = None,
        notify_progress: bool = True,
        progress_interval: float = 0.1,
    ):
        self.name = name
        self.client = client
        self.session_key = session_key
        self.notify_progress_enabled = notify_progress
        self.progress_interval = progress_interval
        self.start_time: Optional[float] = None
        self.last_progress: float = 0.0
        self._context: Dict[str, Any] = {}

    def __enter__(self) -> "ExperimentTracker":
        self.start_time = __import__("time").time()
        self.client.notify_start(
            self.name,
            config=self._context,
            session_key=self.session_key,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        import time
        elapsed = time.time() - self.start_time

        if exc_type:
            self.client.notify_error(
                self.name,
                exc_val,
                context={"elapsed_seconds": f"{elapsed:.1f}s"},
                session_key=self.session_key,
            )
        else:
            self.client.notify_complete(
                self.name,
                results={"elapsed_seconds": f"{elapsed:.1f}s"},
                session_key=self.session_key,
            )

        return False

    def update_progress(
        self,
        progress: float,
        metrics: Dict[str, Any] = None,
        force: bool = False,
    ) -> None:
        if not self.notify_progress_enabled:
            return

        if not force and (progress - self.last_progress) < self.progress_interval:
            return

        self.last_progress = progress
        self.client.notify_progress(
            self.name,
            progress,
            metrics=metrics,
            session_key=self.session_key,
        )

    def set_context(self, key: str, value: Any) -> None:
        self._context[key] = value

    def send_message(self, message: str) -> Dict[str, Any]:
        return self.client.send_custom(message, session_key=self.session_key)


def create_notifier(
    token: str = None,
    session_key: str = None,
    gateway_url: str = "http://127.0.0.1:18789",
) -> OpenClawClient:
    """创建通知客户端"""
    return OpenClawClient(
        gateway_url=gateway_url,
        token=token,
        session_key=session_key,
    )


# ============ 命令行测试 ============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenClaw 通知客户端测试")
    parser.add_argument("--token", required=True, help="API token")
    parser.add_argument(
        "--session-key",
        default="agent:main:feishu:group:oc_05a3fdf2a569cc22f8f19a09653c49f5",
        help="会话 ID",
    )
    parser.add_argument(
        "--gateway-url",
        default="http://127.0.0.1:18789",
        help="Gateway URL",
    )
    parser.add_argument("--message", default="测试消息", help="要发送的消息")
    args = parser.parse_args()

    client = OpenClawClient(
        gateway_url=args.gateway_url,
        token=args.token,
        session_key=args.session_key,
    )

    print(f"发送消息到 {args.session_key}...")
    response = client.send_custom(args.message)
    print(f"响应: {response}")