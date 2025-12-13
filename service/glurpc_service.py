import sys
import logging
import asyncio
import base64
import json
import statistics
from typing import Optional, List, Dict, Any
from concurrent import futures

import grpc

import service.common
from service.service_spec import glurpc_pb2, glurpc_pb2_grpc

# Import glurpc core functionality
from glurpc.core import (
    convert_to_unified_action,
    parse_and_schedule,
    generate_plot_from_handle,
    quick_plot_action,
    verify_api_key,
)
from glurpc.engine import ModelManager, BackgroundProcessor, check_queue_overload
from glurpc.state import InferenceCache, PlotCache, DisconnectTracker
from glurpc.data_classes import RequestTimeStats
from glurpc.schemas import RequestMetrics, FormattedWarnings
from glurpc.config import ENABLE_API_KEYS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)-18s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger("glurpc.grpc_service")


def format_warnings_to_proto(warnings: FormattedWarnings) -> glurpc_pb2.Warnings:
    """Convert FormattedWarnings to protobuf Warnings message."""
    return glurpc_pb2.Warnings(
        has_warnings=warnings.has_warnings,
        too_short=warnings.too_short,
        calibration=warnings.calibration,
        quality=warnings.quality,
        imputation=warnings.imputation,
        out_of_range=warnings.out_of_range,
        time_duplicates=warnings.time_duplicates,
        messages=warnings.messages
    )


class GlucosePredictionServicer(glurpc_pb2_grpc.GlucosePredictionServicer):
    """
    gRPC servicer for glucose prediction.
    Wraps existing REST endpoint logic with gRPC interface.
    Implements disconnect detection for graceful cancellation.
    """

    def __init__(self):
        self.request_metrics = RequestMetrics()
        self.model_manager = ModelManager()
        self.bg_processor = BackgroundProcessor()
        logger.info("GlucosePredictionServicer created")

    async def _initialize(self):
        """Initialize models and background processor."""
        if not self.model_manager.initialized:
            await self.model_manager.initialize()
        if not self.bg_processor._running:
            await self.bg_processor.start()
        logger.info("GlucosePredictionServicer initialized")

    def _check_api_key(self, context: grpc.ServicerContext, metadata_key: str = 'x-api-key') -> bool:
        """
        Check API key from gRPC metadata.
        Returns True if valid or if API keys disabled.
        Sets context error and returns False if invalid.
        """
        if not ENABLE_API_KEYS:
            return True

        metadata = dict(context.invocation_metadata())
        api_key = metadata.get(metadata_key, None)

        if not api_key:
            logger.warning("gRPC: API key missing in request")
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details("API key required in metadata (x-api-key)")
            return False

        if not verify_api_key(api_key):
            logger.warning(f"gRPC: Invalid API key provided: {api_key[:8]}...")
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details("Invalid API key")
            return False

        logger.info(f"gRPC: API key verified: {api_key[:8]}...")
        return True

    def _create_disconnect_future(self, context: grpc.ServicerContext) -> asyncio.Future:
        """
        Create a future that will be set when client disconnects.
        Monitors gRPC context cancellation to detect disconnects.
        """
        loop = asyncio.get_event_loop()
        disconnect_future = loop.create_future()

        async def watch_disconnect():
            """Monitor context for cancellation."""
            try:
                while context.is_active():
                    await asyncio.sleep(0.1)
                # Context is no longer active (client disconnected)
                if not disconnect_future.done():
                    disconnect_future.set_result(True)
                    logger.info("gRPC: Client disconnected")
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning(f"gRPC: Disconnect watch error: {e}")

        # Start watcher task
        asyncio.create_task(watch_disconnect())
        return disconnect_future

    def ConvertToUnified(
        self,
        request: glurpc_pb2.ConvertToUnifiedRequest,
        context: grpc.ServicerContext
    ) -> glurpc_pb2.ConvertToUnifiedResponse:
        """
        Convert CGM data to unified format (public endpoint, no auth).
        """
        logger.info("gRPC: ConvertToUnified called")
        self.request_metrics.total_http_requests += 1

        try:
            # Encode file content to base64
            content_base64 = base64.b64encode(request.file_content).decode()

            # Use asyncio to run the async action
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(convert_to_unified_action(content_base64))
            loop.close()

            if result.error:
                logger.info(f"gRPC: ConvertToUnified - error={result.error}")
                self.request_metrics.total_http_errors += 1
                return glurpc_pb2.ConvertToUnifiedResponse(error=result.error)

            logger.info(f"gRPC: ConvertToUnified - success, csv_length={len(result.csv_content) if result.csv_content else 0}")
            return glurpc_pb2.ConvertToUnifiedResponse(
                csv_content=result.csv_content or "",
                error=""
            )

        except Exception as e:
            logger.error(f"gRPC: Convert failed: {e}")
            self.request_metrics.total_http_errors += 1
            return glurpc_pb2.ConvertToUnifiedResponse(error=str(e))

    def ProcessUnified(
        self,
        request: glurpc_pb2.ProcessUnifiedRequest,
        context: grpc.ServicerContext
    ) -> glurpc_pb2.ProcessUnifiedResponse:
        """
        Process unified CSV and cache for plotting (requires auth).
        """
        logger.info(f"gRPC: ProcessUnified called - csv_base64_length={len(request.csv_base64)}, force={request.force_calculate}")
        self.request_metrics.total_http_requests += 1

        # Check API key
        if not self._check_api_key(context):
            self.request_metrics.total_http_errors += 1
            return glurpc_pb2.ProcessUnifiedResponse(error="Authentication failed")

        # Check for overload
        is_overloaded, load_status, _, _ = check_queue_overload()
        if is_overloaded:
            logger.warning(f"gRPC: Rejecting ProcessUnified due to overload (status={load_status})")
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details(f"Service overloaded. Queue is {load_status}. Please retry later.")
            self.request_metrics.total_http_errors += 1
            return glurpc_pb2.ProcessUnifiedResponse(error="Service overloaded")

        try:
            # Use asyncio to run the async action
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                parse_and_schedule(request.csv_base64, force_calculate=request.force_calculate)
            )
            loop.close()

            if result.error:
                logger.info(f"gRPC: ProcessUnified - error={result.error}")
                self.request_metrics.total_http_errors += 1
                return glurpc_pb2.ProcessUnifiedResponse(error=result.error)

            logger.info(f"gRPC: ProcessUnified - handle={result.handle}, has_warnings={result.warnings.has_warnings}")
            return glurpc_pb2.ProcessUnifiedResponse(
                handle=result.handle or "",
                total_samples=result.total_samples or 0,
                warnings=format_warnings_to_proto(result.warnings)
            )

        except Exception as e:
            logger.error(f"gRPC: Process failed: {e}")
            self.request_metrics.total_http_errors += 1
            return glurpc_pb2.ProcessUnifiedResponse(error=str(e))

    def DrawPlot(
        self,
        request: glurpc_pb2.PlotRequest,
        context: grpc.ServicerContext
    ) -> glurpc_pb2.PlotResponse:
        """
        Generate plot from cached dataset (requires auth).
        Implements disconnect detection and cancellation.
        """
        logger.info(f"gRPC: DrawPlot called - handle={request.handle}, index={request.index}, force={request.force_calculate}")
        self.request_metrics.total_http_requests += 1

        # Check API key
        if not self._check_api_key(context):
            self.request_metrics.total_http_errors += 1
            return glurpc_pb2.PlotResponse(error="Authentication failed")

        # Check for overload
        is_overloaded, load_status, _, _ = check_queue_overload()
        if is_overloaded:
            logger.warning(f"gRPC: Rejecting DrawPlot due to overload (status={load_status})")
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details(f"Service overloaded. Queue is {load_status}. Please retry later.")
            self.request_metrics.total_http_errors += 1
            return glurpc_pb2.PlotResponse(error="Service overloaded")

        try:
            # Register request and get request_id
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            disconnect_tracker = DisconnectTracker()
            request_id = loop.run_until_complete(
                disconnect_tracker.register_request(request.handle, request.index)
            )

            # Update latest request_id
            loop.run_until_complete(
                self.bg_processor.update_latest_request_id(request.handle, request.index, request_id)
            )

            # Create disconnect future with gRPC context monitoring
            disconnect_future = self._create_disconnect_future(context)

            try:
                # Generate plot
                plot_dict = loop.run_until_complete(
                    generate_plot_from_handle(
                        request.handle,
                        request.index,
                        force_calculate=request.force_calculate,
                        request_id=request_id,
                        disconnect_future=disconnect_future
                    )
                )

                logger.info(f"gRPC: DrawPlot - handle={request.handle}, index={request.index}, plot_keys={list(plot_dict.keys())}")
                
                # Convert plot dict to JSON string
                plot_json = json.dumps(plot_dict)
                return glurpc_pb2.PlotResponse(plot_json=plot_json)

            except asyncio.CancelledError:
                logger.info(f"gRPC: Request cancelled: {request.handle[:8]}:{request.index}:{request_id}")
                from glurpc.state import TaskRegistry
                loop.run_until_complete(
                    TaskRegistry().cancel_request(request.handle, request.index, request_id, "Client disconnected")
                )
                context.set_code(grpc.StatusCode.CANCELLED)
                context.set_details("Client closed request")
                self.request_metrics.total_http_errors += 1
                return glurpc_pb2.PlotResponse(error="Request cancelled")

            except asyncio.TimeoutError:
                logger.info(f"gRPC: Request timeout: {request.handle[:8]}:{request.index}:{request_id}")
                from glurpc.state import TaskRegistry
                loop.run_until_complete(
                    TaskRegistry().cancel_request(request.handle, request.index, request_id, "Request timeout")
                )
                context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
                context.set_details("Request timeout")
                self.request_metrics.total_http_errors += 1
                return glurpc_pb2.PlotResponse(error="Request timeout")

            except ValueError as e:
                logger.info(f"gRPC: DrawPlot - handle={request.handle}, index={request.index}, error={str(e)}")
                if "not found" in str(e).lower():
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                else:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                self.request_metrics.total_http_errors += 1
                return glurpc_pb2.PlotResponse(error=str(e))

            finally:
                # Unregister request
                loop.run_until_complete(
                    disconnect_tracker.unregister_request(request.handle, request.index, request_id)
                )
                loop.close()

        except Exception as e:
            logger.error(f"gRPC: Plot failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            self.request_metrics.total_http_errors += 1
            return glurpc_pb2.PlotResponse(error="Internal server error")

    def QuickPlot(
        self,
        request: glurpc_pb2.QuickPlotRequest,
        context: grpc.ServicerContext
    ) -> glurpc_pb2.QuickPlotResponse:
        """
        Quick plot: process + plot in one call (requires auth).
        Implements disconnect detection and cancellation.
        """
        logger.info(f"gRPC: QuickPlot called - csv_base64_length={len(request.csv_base64)}, force={request.force_calculate}")
        self.request_metrics.total_http_requests += 1

        # Check API key
        if not self._check_api_key(context):
            self.request_metrics.total_http_errors += 1
            return glurpc_pb2.QuickPlotResponse(error="Authentication failed")

        # Check for overload
        is_overloaded, load_status, _, _ = check_queue_overload()
        if is_overloaded:
            logger.warning(f"gRPC: Rejecting QuickPlot due to overload (status={load_status})")
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details(f"Service overloaded. Queue is {load_status}. Please retry later.")
            self.request_metrics.total_http_errors += 1
            return glurpc_pb2.QuickPlotResponse(error="Service overloaded")

        try:
            # Generate temporary request_id
            import uuid
            request_id = hash(str(uuid.uuid4())) & 0x7FFFFFFF

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Create disconnect future
            disconnect_future = self._create_disconnect_future(context)

            try:
                result = loop.run_until_complete(
                    quick_plot_action(
                        request.csv_base64,
                        force_calculate=request.force_calculate,
                        request_id=request_id,
                        disconnect_future=disconnect_future
                    )
                )

                if result.error:
                    logger.info(f"gRPC: QuickPlot - error={result.error}")
                    self.request_metrics.total_http_errors += 1
                    return glurpc_pb2.QuickPlotResponse(error=result.error)

                logger.info(f"gRPC: QuickPlot - success, plot_keys={list(result.plot_data.keys())}")
                plot_json = json.dumps(result.plot_data)
                return glurpc_pb2.QuickPlotResponse(
                    plot_json=plot_json,
                    warnings=format_warnings_to_proto(result.warnings)
                )

            except asyncio.CancelledError:
                logger.info(f"gRPC: Quick plot cancelled: req_id={request_id}")
                return glurpc_pb2.QuickPlotResponse(error="Request cancelled")

            finally:
                loop.close()

        except Exception as e:
            logger.error(f"gRPC: Quick plot failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            self.request_metrics.total_http_errors += 1
            return glurpc_pb2.QuickPlotResponse(error="Internal server error")

    def ManageCache(
        self,
        request: glurpc_pb2.CacheManagementRequest,
        context: grpc.ServicerContext
    ) -> glurpc_pb2.CacheManagementResponse:
        """
        Manage cache: flush, info, delete, save, load (requires auth).
        """
        logger.info(f"gRPC: ManageCache called - action={request.action}, handle={request.handle}")
        self.request_metrics.total_http_requests += 1

        # Check API key
        if not self._check_api_key(context):
            self.request_metrics.total_http_errors += 1
            return glurpc_pb2.CacheManagementResponse(
                success=False,
                message="Authentication failed"
            )

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            inf_cache = InferenceCache()
            plot_cache = PlotCache()

            if request.action == "flush":
                loop.run_until_complete(inf_cache.clear())
                loop.run_until_complete(plot_cache.clear())
                loop.close()
                return glurpc_pb2.CacheManagementResponse(
                    success=True,
                    message="Cache flushed successfully",
                    cache_size=0,
                    persisted_count=0,
                    items_affected=0
                )

            elif request.action == "info":
                size = loop.run_until_complete(inf_cache.get_size())
                plot_size = loop.run_until_complete(plot_cache.get_size())
                loop.close()
                return glurpc_pb2.CacheManagementResponse(
                    success=True,
                    message=f"Cache info retrieved (Inference: {size}, Plots: {plot_size})",
                    cache_size=size,
                    persisted_count=size + plot_size,
                    items_affected=0
                )

            elif request.action == "delete":
                if not request.handle:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    self.request_metrics.total_http_errors += 1
                    loop.close()
                    return glurpc_pb2.CacheManagementResponse(
                        success=False,
                        message="Handle parameter required for delete action"
                    )

                exists = loop.run_until_complete(inf_cache.contains(request.handle))
                loop.run_until_complete(inf_cache.delete(request.handle))
                size = loop.run_until_complete(inf_cache.get_size())
                loop.close()

                return glurpc_pb2.CacheManagementResponse(
                    success=exists,
                    message=f"Handle {request.handle} deleted successfully" if exists else f"Handle {request.handle} not found in cache",
                    cache_size=size,
                    persisted_count=size,
                    items_affected=1 if exists else 0
                )

            elif request.action == "save":
                size = loop.run_until_complete(inf_cache.get_size())
                loop.close()
                return glurpc_pb2.CacheManagementResponse(
                    success=True,
                    message="Cache is automatically persisted (No-op)",
                    cache_size=size,
                    persisted_count=size,
                    items_affected=0
                )

            elif request.action == "load":
                if not request.handle:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    self.request_metrics.total_http_errors += 1
                    loop.close()
                    return glurpc_pb2.CacheManagementResponse(
                        success=False,
                        message="Handle parameter required for load action"
                    )

                data = loop.run_until_complete(inf_cache.get(request.handle))
                size = loop.run_until_complete(inf_cache.get_size())
                loop.close()

                if data:
                    return glurpc_pb2.CacheManagementResponse(
                        success=True,
                        message=f"Handle {request.handle} loaded/verified in memory",
                        cache_size=size,
                        persisted_count=size,
                        items_affected=1
                    )
                else:
                    return glurpc_pb2.CacheManagementResponse(
                        success=False,
                        message=f"Handle {request.handle} not found on disk",
                        cache_size=size,
                        persisted_count=size,
                        items_affected=0
                    )

            else:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                self.request_metrics.total_http_errors += 1
                loop.close()
                return glurpc_pb2.CacheManagementResponse(
                    success=False,
                    message=f"Unknown action: {request.action}. Valid actions are: flush, info, delete, save, load"
                )

        except Exception as e:
            logger.error(f"gRPC: Cache management failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            self.request_metrics.total_http_errors += 1
            return glurpc_pb2.CacheManagementResponse(
                success=False,
                message=f"Internal error: {str(e)}"
            )

    def CheckHealth(
        self,
        request: glurpc_pb2.HealthRequest,
        context: grpc.ServicerContext
    ) -> glurpc_pb2.HealthResponse:
        """
        Health check endpoint (public, no auth).
        """
        logger.info("gRPC: CheckHealth called")
        self.request_metrics.total_http_requests += 1

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            inf_cache = InferenceCache()
            cache_size = loop.run_until_complete(inf_cache.get_size())

            stats = self.model_manager.get_stats()
            calc_stats = self.bg_processor.get_calc_stats()

            # Calculate request time stats
            request_times = self.request_metrics.request_times
            if request_times:
                avg_time = statistics.mean(request_times)
                median_time = statistics.median(request_times)
                min_time = min(request_times)
                max_time = max(request_times)
            else:
                avg_time = median_time = min_time = max_time = 0.0

            _, load_status, _, _ = check_queue_overload()

            loop.close()

            return glurpc_pb2.HealthResponse(
                status="ok" if self.model_manager.initialized else "degraded",
                load_status=load_status,
                cache_size=cache_size,
                models_initialized=self.model_manager.initialized,
                available_priority_models=stats.available_priority_models,
                available_general_models=stats.available_general_models,
                avg_fulfillment_time_ms=stats.avg_fulfillment_time_ms,
                vmem_usage_mb=stats.vmem_usage_mb,
                device=stats.device,
                total_http_requests=self.request_metrics.total_http_requests,
                total_http_errors=self.request_metrics.total_http_errors,
                avg_request_time_ms=avg_time,
                median_request_time_ms=median_time,
                min_request_time_ms=min_time,
                max_request_time_ms=max_time,
                inference_requests_by_priority=stats.inference_requests_by_priority,
                total_inference_errors=stats.total_inference_errors,
                total_calc_runs=calc_stats.total_calc_runs,
                total_calc_errors=calc_stats.total_calc_errors,
                inference_queue_size=calc_stats.inference_queue_size,
                inference_queue_capacity=calc_stats.inference_queue_capacity,
                calc_queue_size=calc_stats.calc_queue_size,
                calc_queue_capacity=calc_stats.calc_queue_capacity
            )

        except Exception as e:
            logger.error(f"gRPC: Health check failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            self.request_metrics.total_http_errors += 1
            return glurpc_pb2.HealthResponse(
                status="error",
                load_status="unknown",
                cache_size=0,
                models_initialized=False
            )


def serve(max_workers: int = 10, port: int = 7003) -> grpc.Server:
    """
    Create and configure gRPC server.
    
    Args:
        max_workers: Thread pool size for concurrent requests
        port: Port to bind gRPC service to
        
    Returns:
        Configured gRPC server (not started)
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = GlucosePredictionServicer()
    
    # Initialize servicer (models + background processor)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(servicer._initialize())
    loop.close()
    
    glurpc_pb2_grpc.add_GlucosePredictionServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]{port}")
    logger.info(f"gRPC server configured on port {port}")
    return server


if __name__ == "__main__":
    """
    Run the gRPC server standalone.
    """
    parser = service.common.common_parser(__file__)
    args = parser.parse_args(sys.argv[1:])
    service.common.main_loop(serve, args)
