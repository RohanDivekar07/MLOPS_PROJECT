"""
Custom Locust gRPC User Implementation.
Provides gRPC protocol instrumentation, metrics tracking, and event dispatching for Locust.
"""

import time
import grpc
from locust import User, events

class GrpcClient:
    """
    gRPC client wrapper that intercepts RPC calls and dispatches Locust events.
    """
    def __init__(self, host, port=None):
        target = f"{host}:{port}" if port else host
        self.channel = grpc.insecure_channel(
            target,
            options=[
                ('grpc.max_send_message_length', 100 * 1024 * 1024),
                ('grpc.max_receive_message_length', 100 * 1024 * 1024)
            ]
        )

    def invoke(self, name, rpc_func, request, timeout=10.0):
        start_time = time.perf_counter()
        exception = None
        response_length = 0
        try:
            response = rpc_func(request, timeout=timeout)
            response_length = len(response.SerializeToString()) if hasattr(response, "SerializeToString") else 0
            return response
        except grpc.RpcError as e:
            exception = e
            raise e
        finally:
            total_time = (time.perf_counter() - start_time) * 1000.0
            events.request.fire(
                request_type="gRPC",
                name=name,
                response_time=total_time,
                response_length=response_length,
                exception=exception,
                context={}
            )

class GrpcUser(User):
    """
    Locust User subclass for executing gRPC load tests.
    """
    abstract = True

    def __init__(self, environment):
        super().__init__(environment)
        host = self.host or "localhost"
        port = getattr(self, "port", None)
        self.client = GrpcClient(host, port)
