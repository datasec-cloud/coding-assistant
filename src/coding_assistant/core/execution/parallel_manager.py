# src/coding_assistant/core/execution/parallel_manager.py

from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic, Union
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import threading
from queue import PriorityQueue

T = TypeVar('T')

class ExecutionPriority(Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class TaskResult(Generic[T]):
    """Represents the result of a task execution"""
    task_id: str
    status: ExecutionStatus
    result: Optional[T] = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: Dict[str, Any] = None

class ParallelTask:
    """Represents a task that can be executed in parallel"""
    
    def __init__(self, task_id: str, func: Callable, args: tuple = None, 
                 kwargs: dict = None, priority: ExecutionPriority = ExecutionPriority.MEDIUM,
                 timeout: Optional[float] = None):
        self.task_id = task_id
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.priority = priority
        self.timeout = timeout
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status = ExecutionStatus.PENDING
        
    def __lt__(self, other):
        return self.priority.value < other.priority.value

class ParallelExecutionManager:
    """Manages parallel execution of tasks with priority, monitoring, and error handling"""
    
    def __init__(self, max_workers: int = 4, base_dir: Optional[Path] = None):
        self.max_workers = max_workers
        self.base_dir = Path(base_dir) if base_dir else Path("data/execution")
        self.logger = logging.getLogger(__name__)
        
        # Core execution components
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue: PriorityQueue[ParallelTask] = PriorityQueue()
        self.active_tasks: Dict[str, Future] = {}
        self.results: Dict[str, TaskResult] = {}
        
        # Synchronization
        self._lock = threading.RLock()
        self._shutdown = threading.Event()
        
        # Monitoring
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0
        }
        
        # Start worker thread
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
        
    async def execute_async(self, tasks: List[ParallelTask]) -> Dict[str, TaskResult]:
        """Execute tasks asynchronously"""
        try:
            # Queue all tasks
            for task in tasks:
                self.task_queue.put(task)
            
            # Wait for completion
            while not all(task.task_id in self.results for task in tasks):
                await asyncio.sleep(0.1)
                
            # Return results
            return {
                task.task_id: self.results[task.task_id]
                for task in tasks
                if task.task_id in self.results
            }
            
        except Exception as e:
            self.logger.error(f"Error in async execution: {e}")
            raise
            
    def execute(self, tasks: List[ParallelTask], 
                timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """Execute tasks synchronously with optional timeout"""
        try:
            # Queue all tasks
            for task in tasks:
                self.task_queue.put(task)
            
            # Wait for completion
            start_time = datetime.now()
            while not all(task.task_id in self.results for task in tasks):
                if timeout:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed > timeout:
                        self._handle_timeout(tasks)
                        break
                threading.Event().wait(0.1)
            
            # Return results
            return {
                task.task_id: self.results[task.task_id]
                for task in tasks
                if task.task_id in self.results
            }
            
        except Exception as e:
            self.logger.error(f"Error in execution: {e}")
            raise
            
    def _process_queue(self):
        """Process tasks from the queue"""
        while not self._shutdown.is_set():
            try:
                # Get next task
                task = self.task_queue.get(timeout=1.0)
                if task is None:
                    continue
                    
                # Execute task
                self._execute_task(task)
                
            except TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing task queue: {e}")
                
    def _execute_task(self, task: ParallelTask):
        """Execute a single task"""
        try:
            with self._lock:
                if task.task_id in self.active_tasks:
                    return
                
                # Submit task
                task.start_time = datetime.now()
                task.status = ExecutionStatus.RUNNING
                
                future = self.executor.submit(
                    self._run_task_with_monitoring,
                    task
                )
                
                self.active_tasks[task.task_id] = future
                future.add_done_callback(
                    lambda f: self._handle_completion(task.task_id, f)
                )
                
        except Exception as e:
            self.logger.error(f"Error executing task {task.task_id}: {e}")
            self._record_failure(task, e)
            
    def _run_task_with_monitoring(self, task: ParallelTask) -> Any:
        """Run task with monitoring and timing"""
        try:
            start_time = datetime.now()
            
            if task.timeout:
                # Run with timeout
                future = self.executor.submit(task.func, *task.args, **task.kwargs)
                result = future.result(timeout=task.timeout)
            else:
                # Run normally
                result = task.func(*task.args, **task.kwargs)
                
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            with self._lock:
                self.metrics["tasks_completed"] += 1
                self.metrics["total_execution_time"] += execution_time
                self.metrics["average_execution_time"] = (
                    self.metrics["total_execution_time"] / 
                    self.metrics["tasks_completed"]
                )
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error in task execution: {e}")
            raise
            
    def _handle_completion(self, task_id: str, future: Future):
        """Handle task completion"""
        try:
            with self._lock:
                task = self.active_tasks.pop(task_id, None)
                if not task:
                    return
                    
                if future.exception():
                    self.results[task_id] = TaskResult(
                        task_id=task_id,
                        status=ExecutionStatus.FAILED,
                        error=future.exception()
                    )
                    self.metrics["tasks_failed"] += 1
                else:
                    self.results[task_id] = TaskResult(
                        task_id=task_id,
                        status=ExecutionStatus.COMPLETED,
                        result=future.result()
                    )
                    
        except Exception as e:
            self.logger.error(f"Error handling task completion: {e}")
            
    def _handle_timeout(self, tasks: List[ParallelTask]):
        """Handle timeout for tasks"""
        with self._lock:
            for task in tasks:
                if task.task_id not in self.results:
                    self.results[task_id] = TaskResult(
                        task_id=task.task_id,
                        status=ExecutionStatus.TIMEOUT
                    )
                    
                future = self.active_tasks.get(task.task_id)
                if future and not future.done():
                    future.cancel()
                    
    def _record_failure(self, task: ParallelTask, error: Exception):
        """Record task failure"""
        with self._lock:
            self.results[task.task_id] = TaskResult(
                task_id=task.task_id,
                status=ExecutionStatus.FAILED,
                error=error,
                start_time=task.start_time,
                end_time=datetime.now()
            )
            self.metrics["tasks_failed"] += 1
            
    def shutdown(self, wait: bool = True):
        """Shutdown the execution manager"""
        self._shutdown.set()
        self.executor.shutdown(wait=wait)
        if wait:
            self._worker_thread.join()
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics"""
        with self._lock:
            return self.metrics.copy()
            
    def clear_results(self):
        """Clear stored results"""
        with self._lock:
            self.results.clear()