import logging
import time
import threading
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

import base_policy as _base_policy
import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.
    / 通过 WebSocket 与服务器通信实现策略接口。

    See WebsocketPolicyServer for a corresponding server implementation.
    参见 WebsocketPolicyServer 以了解对应的服务器实现。
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        """Initialize WebSocket client policy.
        / 初始化 WebSocket 客户端策略。

        Args:
            host: Server host address. / 服务器主机地址。
            port: Server port number. / 服务器端口号。
            api_key: Optional API key for authentication. / 可选的 API 密钥用于身份验证。
        """
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws_lock = threading.Lock()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        """Get server metadata. / 获取服务器元数据。"""
        return self._server_metadata

    def _wait_for_server(self, max_retries: int = 3) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        """Wait for server connection and return connection with metadata.
        / 等待服务器连接并返回连接和元数据。
        """
        logging.info(f"Waiting for server at {self._uri}...")
        for attempt in range(max_retries):
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except Exception:
                logging.info(f"Still waiting for server... ({attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(5)
        raise ConnectionRefusedError(f"无法连接推理服务 {self._uri}，已重试 {max_retries} 次")

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        """Perform inference by sending observation to server and receiving actions.
        / 通过向服务器发送观察并接收动作来执行推理。
        """
        data = self._packer.pack(obs)
        with self._ws_lock:
            ws = self._ws
        if ws is None:
            raise RuntimeError("推理连接未建立")
        ws.send(data)
        response = ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            # 我们期望接收字节；如果服务器发送字符串，则说明出错了。
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        """Reset policy state. / 重置策略状态。"""
        logging.info("Resetting inference websocket connection...")
        with self._ws_lock:
            old_ws = self._ws
            self._ws = None

        if old_ws is not None:
            try:
                old_ws.close()
            except Exception:
                logging.exception("关闭旧的推理 websocket 连接失败")

        ws, metadata = self._wait_for_server()
        with self._ws_lock:
            self._ws = ws
            self._server_metadata = metadata
