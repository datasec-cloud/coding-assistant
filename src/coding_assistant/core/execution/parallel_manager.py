"""Enhanced parallel execution manager with improved task handling and monitoring"""
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic, Union
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from queue import PriorityQueue
import traceback
from contextlib import contextmanager
import signal
import time
import json

T = TypeVar('T')

class ExecutionPriority(Enum):
    """Task execution priorities"""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class ExecutionStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

@dataclass
class TaskMetrics:
    """Task execution metrics"""
    queue_time: float = 0.0
    execution_time: float = 0.0
    cpu_time: float = 0.0
    memory_usage: float = 0.0
    retries: int = 0
    error_count: int = 0

@dataclass
class TaskResult(Generic[T]):
    """Enhanced task execution result"""
    task_id: str
    status: ExecutionStatus
    result: Optional[T] = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": str(self.error) if self.error else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metrics": {
                "queue_time": self.metrics.queue_time,
                "execution_time": self.metrics.execution_time,
                "cpu_time": self.metrics.cpu_time,
                "memory_usage": self.metrics.memory_usage,
                "retries": self.metrics.retries,
                "error_count": self.metrics.error_count
            },
            "stack_trace": self.stack_trace
        }

@dataclass
class ParallelTask:
    """Enhanced parallel task definition"""
    task_id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: ExecutionPriority = ExecutionPriority.MEDIUM
    timeout: Optional[float] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    status: ExecutionStatus = ExecutionStatus.PENDING
    
    def __lt__(self, other):
        return (self.priority.value, self.created_at) < (other.priority.value, other.created_at)

class ExecutionError(Exception):
    """Custom execution error with enhanced details"""
    def __init__(self, message: str, task_id: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.task_id = task_id
        self.original_error = original_error
        self.timestamp = datetime.now()
        self.stack_trace = traceback.format_exc()

class ParallelExecutionManager:
    """Enhanced parallel execution manager with improved monitoring"""
    
    def __init__(self, max_workers: int = 4, base_dir: Optional[Path] = None):
        self.max_workers = max_workers
        self.base_dir = Path(base_dir) if base_dir else Path("data/execution")
        self.logger = logging.getLogger(__name__)
        
        # Enhanced execution components
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue: PriorityQueue[ParallelTask] = PriorityQueue()
        self.active_tasks: Dict[str, Future] = {}
        self.results: Dict[str, TaskResult] = {}
        
        # Enhanced synchronization
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Enhanced monitoring
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
            "active_workers": 0,
            "queue_size": 0,
            "error_rate": 0.0,
            "cpu_usage": 0.0,
            "memory_usage": 0.0
        }
        
        # Initialize storage
        self._initialize_storage()
        
        # Start monitoring threads
        self._start_monitoring()
        
        # Register signal handlers
        self._register_signals()

    def _initialize_storage(self):
        """Initialize execution storage"""
        try:
            for dir_name in ["logs", "results", "metrics"]:
                (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to initialize storage: {e}")
            raise

    def _start_monitoring(self):
        """Start monitoring threads"""
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._monitor_thread = threading.Thread(target=self._monitor_execution, daemon=True)
        self._worker_thread.start()
        self._monitor_thread.start()

    def _register_signals(self):
        """Register signal handlers for graceful shutdown"""
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    async def execute_async(self, tasks: List[ParallelTask]) -> Dict[str, TaskResult]:
        """Execute tasks asynchronously with enhanced monitoring"""
        try:
            if not tasks:
                return {}

            # Queue all tasks
            for task in tasks:
                self.task_queue.put(task)
                self.logger.debug(f"Queued task {task.task_id} with priority {task.priority}")
            
            # Wait for completion with timeout monitoring
            timeout = max(task.timeout for task in tasks if task.timeout is not None)
            start_time = time.time()
            
            while not all(task.task_id in self.results for task in tasks):
                if timeout and (time.time() - start_time) > timeout:
                    self._handle_timeout(tasks)
                    break
                await asyncio.sleep(0.1)
            
            # Return results
            return {
                task.task_id: self.results[task.task_id]
                for task in tasks
                if task.task_id in self.results
            }
            
        except Exception as e:
            self.logger.error(f"Error in async execution: {e}")
            self._handle_execution_error(e, tasks)
            raise

    def execute(self, tasks: List[ParallelTask], timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """Execute tasks synchronously with enhanced error handling"""
        try:
            if not tasks:
                return {}

            # Queue tasks with priorities
            for task in sorted(tasks):
                self.task_queue.put(task)
                task.status = ExecutionStatus.QUEUED
                self._update_metrics("queue_size", 1)
            
            # Wait for completion with monitoring
            start_time = time.time()
            completed_tasks = set()
            
            while len(completed_tasks) < len(tasks):
                if timeout and (time.time() - start_time) > timeout:
                    remaining_tasks = [t for t in tasks if t.task_id not in completed_tasks]
                    self._handle_timeout(remaining_tasks)
                    break
                
                # Check for completed tasks
                for task in tasks:
                    if task.task_id in self.results and task.task_id not in completed_tasks:
                        completed_tasks.add(task.task_id)
                        self._update_metrics("tasks_completed", 1)
                
                time.sleep(0.1)
            
            return {
                task.task_id: self.results[task.task_id]
                for task in tasks
                if task.task_id in self.results
            }
            
        except Exception as e:
            self.logger.error(f"Error in execution: {e}")
            self._handle_execution_error(e, tasks)
            raise

    def _process_queue(self):
        """Process tasks from queue with enhanced monitoring"""
        while not self._shutdown_event.is_set():
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None:
                    continue
                
                self._execute_task(task)
                self._update_metrics("queue_size", -1)
                
            except TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing task queue: {e}")
                self._handle_queue_error(e)

    def _execute_task(self, task: ParallelTask):
        """Execute single task with retry logic"""
        try:
            with self._lock:
                if task.task_id in self.active_tasks:
                    return
                
                task.status = ExecutionStatus.RUNNING
                task.metrics = TaskMetrics(queue_time=time.time() - task.created_at.timestamp())
                
                future = self.executor.submit(
                    self._run_task_with_monitoring,
                    task
                )
                
                self.active_tasks[task.task_id] = future
                future.add_done_callback(
                    lambda f: self._handle_completion(task.task_id, f)
                )
                
                self._update_metrics("active_workers", 1)
                
        except Exception as e:
            self.logger.error(f"Error executing task {task.task_id}: {e}")
            self._handle_task_error(task, e)

    def _run_task_with_monitoring(self, task: ParallelTask) -> Any:
        """Run task with comprehensive monitoring"""
        start_time = time.time()
        retry_count = 0
        
        while retry_count <= task.max_retries:
            try:
                if task.timeout:
                    # Run with timeout
                    future = self.executor.submit(task.func, *task.args, **task.kwargs)
                    result = future.result(timeout=task.timeout)
                else:
                    # Run normally
                    result = task.func(*task.args, **task.kwargs)
                
                execution_time = time.time() - start_time
                
                # Update metrics
                with self._lock:
                    self.metrics["total_execution_time"] += execution_time
                    self.metrics["average_execution_time"] = (
                        self.metrics["total_execution_time"] / 
                        (self.metrics["tasks_completed"] + 1)
                    )
                
                return result
                
            except Exception as e:
                retry_count += 1
                if retry_count <= task.max_retries:
                    self.logger.warning(
                        f"Retrying task {task.task_id} ({retry_count}/{task.max_retries})"
                    )
                    time.sleep(task.retry_delay)
                else:
                    raise ExecutionError(
                        f"Task failed after {task.max_retries} retries",
                        task.task_id,
                        e
                    )

    def _handle_completion(self, task_id: str, future: Future):
        """Handle task completion with enhanced error handling"""
        try:
            with self._lock:
                task = self.active_tasks.pop(task_id, None)
                if not task:
                    return
                
                end_time = datetime.now()
                
                if future.exception():
                    error = future.exception()
                    self.results[task_id] = TaskResult(
                        task_id=task_id,
                        status=ExecutionStatus.FAILED,
                        error=error,
                        start_time=datetime.fromtimestamp(time.time() - task.metrics.execution_time),
                        end_time=end_time,
                        metrics=task.metrics,
                        stack_trace=traceback.format_exception(
                            type(error), error, error.__traceback__
                        )
                    )
                    self._update_metrics("tasks_failed", 1)
                else:
                    self.results[task_id] = TaskResult(
                        task_id=task_id,
                        status=ExecutionStatus.COMPLETED,
                        result=future.result(),
                        start_time=datetime.fromtimestamp(time.time() - task.metrics.execution_time),
                        end_time=end_time,
                        metrics=task.metrics
                    )
                    
                self._update_metrics("active_workers", -1)
                self._save_result(task_id)
                
        except Exception as e:
            self.logger.error(f"Error handling task completion: {e}")
            self._handle_completion_error(task_id, e)

    def _handle_timeout(self, tasks: List[ParallelTask]):
        """Handle timeout for tasks"""
        with self._lock:
            for task in tasks:
                if task.task_id not in self.results:
                    self.results[task.task_id] = TaskResult(
                        task_id=task.task_id,
                        status=ExecutionStatus.TIMEOUT,
                        error=TimeoutError(f"Task {task.task_id} timed out"),
                        start_time=task.created_at,
                        end_time=datetime.now(),
                        metrics=TaskMetrics(
                            queue_time=time.time() - task.created_at.timestamp()
                        )
                    )
                    
                    future = self.active_tasks.get(task.task_id)
                    if future and not future.done():
                        future.cancel()
                        
                    self._update_metrics("tasks_failed", 1)

    def _save_result(self, task_id: str):
        """Save task result to disk"""
        try:
            result = self.results[task_id]
            result_path = self.base_dir / "results" / f"{task_id}.json"
            
            with result_path.open('w') as f:
                json.dump(result.to_dict(), f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving result for task {task_id}: {e}")

    def _update_metrics(self, metric: str, value: Union[int, float]):
        """Update execution metrics"""
        with self._lock:
            if metric in self.metrics:
                self.metrics[metric] += value
                
            # Calculate error rate
            total_tasks = self.metrics["tasks_completed"] + self.metrics["tasks_failed"]
            if total_tasks > 0:
                self.metrics["error_rate"] = (
                    self.metrics["tasks_failed"] / total_tasks
                ) * 100

    def _monitor_execution(self):
        """Monitor execution metrics"""
        while not self._shutdown_event.is_set():
            try:
                self._update_resource_usage()
                self._save_metrics()
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in execution monitoring: {e}")
                time.sleep(1)  # Retry after error

    def _update_resource_usage(self):
        """Update CPU and memory usage metrics"""
        try:
            import psutil
            process = psutil.Process()
            
            with self._lock:
                self.metrics["cpu_usage"] = process.cpu_percent()
                self.metrics["memory_usage"] = process.memory_info().rss / 1024 / 1024  # MB
                
        except ImportError:
            self.logger.warning("psutil not available for resource monitoring")
        except Exception as e:
            self.logger.error(f"Error updating resource usage: {e}")

    def _save_metrics(self):
        """Save execution metrics to disk"""
        try:
            metrics_path = self.base_dir / "metrics" / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with metrics_path.open('w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "metrics": self.metrics
                }, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")

    def _handle_execution_error(self, error: Exception, tasks: List[ParallelTask]):
        """Handle execution-wide errors"""
        try:
            error_id = f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            error_path = self.base_dir / "logs" / f"{error_id}.error"
            
            with error_path.open('w') as f:
                f.write(f"Error: {str(error)}\n")
                f.write(f"Stack Trace:\n{traceback.format_exc()}\n")
                f.write(f"Affected Tasks:\n")
                for task in tasks:
                    f.write(f"- {task.task_id} (Status: {task.status.value})\n")
                    
            self.logger.error(f"Execution error logged to {error_path}")
            
        except Exception as e:
            self.logger.error(f"Error handling execution error: {e}")

    def _handle_task_error(self, task: ParallelTask, error: Exception):
        """Handle individual task errors"""
        try:
            task.status = ExecutionStatus.FAILED
            task.metrics.error_count += 1
            
            self.results[task.task_id] = TaskResult(
                task_id=task.task_id,
                status=ExecutionStatus.FAILED,
                error=error,
                start_time=task.created_at,
                end_time=datetime.now(),
                metrics=task.metrics,
                stack_trace=traceback.format_exc()
            )
            
            self._update_metrics("tasks_failed", 1)
            self._save_result(task.task_id)
            
        except Exception as e:
            self.logger.error(f"Error handling task error: {e}")

    def _handle_queue_error(self, error: Exception):
        """Handle queue processing errors"""
        try:
            error_id = f"queue_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            error_path = self.base_dir / "logs" / f"{error_id}.error"
            
            with error_path.open('w') as f:
                f.write(f"Queue Error: {str(error)}\n")
                f.write(f"Stack Trace:\n{traceback.format_exc()}\n")
                f.write(f"Queue Size: {self.task_queue.qsize()}\n")
                f.write(f"Active Tasks: {len(self.active_tasks)}\n")
                
        except Exception as e:
            self.logger.error(f"Error handling queue error: {e}")

    def _handle_completion_error(self, task_id: str, error: Exception):
        """Handle task completion errors"""
        try:
            with self._lock:
                self.results[task_id] = TaskResult(
                    task_id=task_id,
                    status=ExecutionStatus.FAILED,
                    error=error,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    metrics=TaskMetrics(error_count=1),
                    stack_trace=traceback.format_exc()
                )
                
                self._update_metrics("tasks_failed", 1)
                self._save_result(task_id)
                
        except Exception as e:
            self.logger.error(f"Error handling completion error: {e}")

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received shutdown signal {signum}")
        self.shutdown()

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """Shutdown execution manager with enhanced cleanup"""
        try:
            self.logger.info("Initiating shutdown sequence")
            self._shutdown_event.set()
            
            if wait:
                # Cancel pending tasks
                with self._lock:
                    pending_tasks = []
                    while not self.task_queue.empty():
                        try:
                            task = self.task_queue.get_nowait()
                            task.status = ExecutionStatus.CANCELLED
                            pending_tasks.append(task)
                        except Exception:
                            break
                    
                    # Record cancelled tasks
                    for task in pending_tasks:
                        self.results[task.task_id] = TaskResult(
                            task_id=task.task_id,
                            status=ExecutionStatus.CANCELLED,
                            start_time=task.created_at,
                            end_time=datetime.now(),
                            metrics=TaskMetrics(
                                queue_time=time.time() - task.created_at.timestamp()
                            )
                        )
                        self._save_result(task.task_id)
                
                # Wait for active tasks
                if timeout:
                    self.executor.shutdown(wait=True, timeout=timeout)
                else:
                    self.executor.shutdown(wait=True)
                
                # Wait for monitoring threads
                self._worker_thread.join(timeout=timeout)
                self._monitor_thread.join(timeout=timeout)
                
                # Save final metrics
                self._save_metrics()
                
            else:
                self.executor.shutdown(wait=False)
                
            self.logger.info("Shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise

    def get_task_status(self, task_id: str) -> Optional[ExecutionStatus]:
        """Get current status of a task"""
        try:
            if task_id in self.results:
                return self.results[task_id].status
            
            with self._lock:
                if task_id in self.active_tasks:
                    return ExecutionStatus.RUNNING
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting task status: {e}")
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get current execution metrics"""
        with self._lock:
            return self.metrics.copy()

    def clear_results(self, older_than: Optional[timedelta] = None):
        """Clear stored results with age filter"""
        try:
            with self._lock:
                if older_than:
                    cutoff_time = datetime.now() - older_than
                    to_remove = [
                        task_id for task_id, result in self.results.items()
                        if result.end_time and result.end_time < cutoff_time
                    ]
                    for task_id in to_remove:
                        del self.results[task_id]
                else:
                    self.results.clear()
                    
        except Exception as e:
            self.logger.error(f"Error clearing results: {e}")

    def get_active_tasks(self) -> List[str]:
        """Get list of currently active task IDs"""
        with self._lock:
            return list(self.active_tasks.keys())

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task"""
        try:
            with self._lock:
                future = self.active_tasks.get(task_id)
                if future and not future.done():
                    future.cancel()
                    self.results[task_id] = TaskResult(
                        task_id=task_id,
                        status=ExecutionStatus.CANCELLED,
                        end_time=datetime.now()
                    )
                    self._save_result(task_id)
                    return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling task {task_id}: {e}")
            return False