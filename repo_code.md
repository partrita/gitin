# Repository Content


## setup.py
```
from setuptools import setup, find_packages
import os
from pathlib import Path
import shutil


# Create the .crawl4ai folder in the user's home directory if it doesn't exist
# If the folder already exists, remove the cache folder
base_dir = os.getenv("CRAWL4_AI_BASE_DIRECTORY")
crawl4ai_folder = Path(base_dir) if base_dir else Path.home()
crawl4ai_folder = crawl4ai_folder / ".crawl4ai"
cache_folder = crawl4ai_folder / "cache"
content_folders = [
    "html_content",
    "cleaned_html",
    "markdown_content",
    "extracted_content",
    "screenshots",
]

# Clean up old cache if exists
if cache_folder.exists():
    shutil.rmtree(cache_folder)

# Create new folder structure
crawl4ai_folder.mkdir(exist_ok=True)
cache_folder.mkdir(exist_ok=True)
for folder in content_folders:
    (crawl4ai_folder / folder).mkdir(exist_ok=True)

# Read requirements and version
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
with open(os.path.join(__location__, "requirements.txt")) as f:
    requirements = f.read().splitlines()

with open("crawl4ai/__version__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"')
            break

# Define requirements
default_requirements = requirements
torch_requirements = ["torch", "nltk", "scikit-learn"]
transformer_requirements = ["transformers", "tokenizers"]
cosine_similarity_requirements = ["torch", "transformers", "nltk"]
sync_requirements = ["selenium"]

setup(
    name="Crawl4AI",
    version=version,
    description="🔥🕷️ Crawl4AI: Open-source LLM Friendly Web Crawler & scraper",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/unclecode/crawl4ai",
    author="Unclecode",
    author_email="unclecode@kidocode.com",
    license="MIT",
    packages=find_packages(),
    package_data={
        'crawl4ai': ['js_snippet/*.js']  # This matches the exact path structure
    },
    install_requires=default_requirements
    + ["playwright", "aiofiles"],  # Added aiofiles
    extras_require={
        "torch": torch_requirements,
        "transformer": transformer_requirements,
        "cosine": cosine_similarity_requirements,
        "sync": sync_requirements,
        "all": default_requirements
        + torch_requirements
        + transformer_requirements
        + cosine_similarity_requirements
        + sync_requirements,
    },
    entry_points={
        "console_scripts": [
            "crawl4ai-download-models=crawl4ai.model_loader:main",
            "crawl4ai-migrate=crawl4ai.migrations:main",  
            'crawl4ai-setup=crawl4ai.install:post_install', 
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
)

```

## main.py
```
import asyncio, os
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware  
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import FileResponse
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, Security

from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Dict, Any, Union
import psutil
import time
import uuid
from collections import defaultdict
from urllib.parse import urlparse
import math
import logging
from enum import Enum
from dataclasses import dataclass
import json
from crawl4ai import AsyncWebCrawler, CrawlResult, CacheMode
from crawl4ai.config import MIN_WORD_THRESHOLD
from crawl4ai.extraction_strategy import (
    LLMExtractionStrategy,
    CosineStrategy,
    JsonCssExtractionStrategy,
)

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class CrawlerType(str, Enum):
    BASIC = "basic"
    LLM = "llm"
    COSINE = "cosine"
    JSON_CSS = "json_css"

class ExtractionConfig(BaseModel):
    type: CrawlerType
    params: Dict[str, Any] = {}

class ChunkingStrategy(BaseModel):
    type: str
    params: Dict[str, Any] = {}

class ContentFilter(BaseModel):
    type: str = "bm25"
    params: Dict[str, Any] = {}

class CrawlRequest(BaseModel):
    urls: Union[HttpUrl, List[HttpUrl]]
    word_count_threshold: int = MIN_WORD_THRESHOLD
    extraction_config: Optional[ExtractionConfig] = None
    chunking_strategy: Optional[ChunkingStrategy] = None
    content_filter: Optional[ContentFilter] = None
    js_code: Optional[List[str]] = None
    wait_for: Optional[str] = None
    css_selector: Optional[str] = None
    screenshot: bool = False
    magic: bool = False
    extra: Optional[Dict[str, Any]] = {}
    session_id: Optional[str] = None
    cache_mode: Optional[CacheMode] = CacheMode.ENABLED
    priority: int = Field(default=5, ge=1, le=10)
    ttl: Optional[int] = 3600    
    crawler_params: Dict[str, Any] = {}

@dataclass
class TaskInfo:
    id: str
    status: TaskStatus
    result: Optional[Union[CrawlResult, List[CrawlResult]]] = None
    error: Optional[str] = None
    created_at: float = time.time()
    ttl: int = 3600

class ResourceMonitor:
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.memory_threshold = 0.85
        self.cpu_threshold = 0.90
        self._last_check = 0
        self._check_interval = 1  # seconds
        self._last_available_slots = max_concurrent_tasks

    async def get_available_slots(self) -> int:
        current_time = time.time()
        if current_time - self._last_check < self._check_interval:
            return self._last_available_slots

        mem_usage = psutil.virtual_memory().percent / 100
        cpu_usage = psutil.cpu_percent() / 100

        memory_factor = max(0, (self.memory_threshold - mem_usage) / self.memory_threshold)
        cpu_factor = max(0, (self.cpu_threshold - cpu_usage) / self.cpu_threshold)

        self._last_available_slots = math.floor(
            self.max_concurrent_tasks * min(memory_factor, cpu_factor)
        )
        self._last_check = current_time

        return self._last_available_slots

class TaskManager:
    def __init__(self, cleanup_interval: int = 300):
        self.tasks: Dict[str, TaskInfo] = {}
        self.high_priority = asyncio.PriorityQueue()
        self.low_priority = asyncio.PriorityQueue()
        self.cleanup_interval = cleanup_interval
        self.cleanup_task = None

    async def start(self):
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

    async def add_task(self, task_id: str, priority: int, ttl: int) -> None:
        task_info = TaskInfo(id=task_id, status=TaskStatus.PENDING, ttl=ttl)
        self.tasks[task_id] = task_info
        queue = self.high_priority if priority > 5 else self.low_priority
        await queue.put((-priority, task_id))  # Negative for proper priority ordering

    async def get_next_task(self) -> Optional[str]:
        try:
            # Try high priority first
            _, task_id = await asyncio.wait_for(self.high_priority.get(), timeout=0.1)
            return task_id
        except asyncio.TimeoutError:
            try:
                # Then try low priority
                _, task_id = await asyncio.wait_for(self.low_priority.get(), timeout=0.1)
                return task_id
            except asyncio.TimeoutError:
                return None

    def update_task(self, task_id: str, status: TaskStatus, result: Any = None, error: str = None):
        if task_id in self.tasks:
            task_info = self.tasks[task_id]
            task_info.status = status
            task_info.result = result
            task_info.error = error

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        return self.tasks.get(task_id)

    async def _cleanup_loop(self):
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                current_time = time.time()
                expired_tasks = [
                    task_id
                    for task_id, task in self.tasks.items()
                    if current_time - task.created_at > task.ttl
                    and task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
                ]
                for task_id in expired_tasks:
                    del self.tasks[task_id]
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

class CrawlerPool:
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.active_crawlers: Dict[AsyncWebCrawler, float] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, **kwargs) -> AsyncWebCrawler:
        async with self._lock:
            # Clean up inactive crawlers
            current_time = time.time()
            inactive = [
                crawler
                for crawler, last_used in self.active_crawlers.items()
                if current_time - last_used > 600  # 10 minutes timeout
            ]
            for crawler in inactive:
                await crawler.__aexit__(None, None, None)
                del self.active_crawlers[crawler]

            # Create new crawler if needed
            if len(self.active_crawlers) < self.max_size:
                crawler = AsyncWebCrawler(**kwargs)
                await crawler.__aenter__()
                self.active_crawlers[crawler] = current_time
                return crawler

            # Reuse least recently used crawler
            crawler = min(self.active_crawlers.items(), key=lambda x: x[1])[0]
            self.active_crawlers[crawler] = current_time
            return crawler

    async def release(self, crawler: AsyncWebCrawler):
        async with self._lock:
            if crawler in self.active_crawlers:
                self.active_crawlers[crawler] = time.time()

    async def cleanup(self):
        async with self._lock:
            for crawler in list(self.active_crawlers.keys()):
                await crawler.__aexit__(None, None, None)
            self.active_crawlers.clear()

class CrawlerService:
    def __init__(self, max_concurrent_tasks: int = 10):
        self.resource_monitor = ResourceMonitor(max_concurrent_tasks)
        self.task_manager = TaskManager()
        self.crawler_pool = CrawlerPool(max_concurrent_tasks)
        self._processing_task = None

    async def start(self):
        await self.task_manager.start()
        self._processing_task = asyncio.create_task(self._process_queue())

    async def stop(self):
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        await self.task_manager.stop()
        await self.crawler_pool.cleanup()

    def _create_extraction_strategy(self, config: ExtractionConfig):
        if not config:
            return None

        if config.type == CrawlerType.LLM:
            return LLMExtractionStrategy(**config.params)
        elif config.type == CrawlerType.COSINE:
            return CosineStrategy(**config.params)
        elif config.type == CrawlerType.JSON_CSS:
            return JsonCssExtractionStrategy(**config.params)
        return None

    async def submit_task(self, request: CrawlRequest) -> str:
        task_id = str(uuid.uuid4())
        await self.task_manager.add_task(task_id, request.priority, request.ttl or 3600)
        
        # Store request data with task
        self.task_manager.tasks[task_id].request = request
        
        return task_id

    async def _process_queue(self):
        while True:
            try:
                available_slots = await self.resource_monitor.get_available_slots()
                if False and available_slots <= 0:
                    await asyncio.sleep(1)
                    continue

                task_id = await self.task_manager.get_next_task()
                if not task_id:
                    await asyncio.sleep(1)
                    continue

                task_info = self.task_manager.get_task(task_id)
                if not task_info:
                    continue

                request = task_info.request
                self.task_manager.update_task(task_id, TaskStatus.PROCESSING)

                try:
                    crawler = await self.crawler_pool.acquire(**request.crawler_params)
                    
                    extraction_strategy = self._create_extraction_strategy(request.extraction_config)
                    
                    if isinstance(request.urls, list):
                        results = await crawler.arun_many(
                            urls=[str(url) for url in request.urls],
                            word_count_threshold=MIN_WORD_THRESHOLD,
                            extraction_strategy=extraction_strategy,
                            js_code=request.js_code,
                            wait_for=request.wait_for,
                            css_selector=request.css_selector,
                            screenshot=request.screenshot,
                            magic=request.magic,
                            session_id=request.session_id,
                            cache_mode=request.cache_mode,
                            **request.extra,
                        )
                    else:
                        results = await crawler.arun(
                            url=str(request.urls),
                            extraction_strategy=extraction_strategy,
                            js_code=request.js_code,
                            wait_for=request.wait_for,
                            css_selector=request.css_selector,
                            screenshot=request.screenshot,
                            magic=request.magic,
                            session_id=request.session_id,
                            cache_mode=request.cache_mode,
                            **request.extra,
                        )

                    await self.crawler_pool.release(crawler)
                    self.task_manager.update_task(task_id, TaskStatus.COMPLETED, results)

                except Exception as e:
                    logger.error(f"Error processing task {task_id}: {str(e)}")
                    self.task_manager.update_task(task_id, TaskStatus.FAILED, error=str(e))

            except Exception as e:
                logger.error(f"Error in queue processing: {str(e)}")
                await asyncio.sleep(1)

app = FastAPI(title="Crawl4AI API")

# CORS configuration
origins = ["*"]  # Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of origins that are allowed to make requests
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# API token security
security = HTTPBearer()
CRAWL4AI_API_TOKEN = os.getenv("CRAWL4AI_API_TOKEN")

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not CRAWL4AI_API_TOKEN:
        return credentials  # No token verification if CRAWL4AI_API_TOKEN is not set
    if credentials.credentials != CRAWL4AI_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials

# Helper function to conditionally apply security
def secure_endpoint():
    return Depends(verify_token) if CRAWL4AI_API_TOKEN else None

# Check if site directory exists
if os.path.exists(__location__ + "/site"):
    # Mount the site directory as a static directory
    app.mount("/mkdocs", StaticFiles(directory="site", html=True), name="mkdocs")

site_templates = Jinja2Templates(directory=__location__ + "/site")

crawler_service = CrawlerService()

@app.on_event("startup")
async def startup_event():
    await crawler_service.start()

@app.on_event("shutdown")
async def shutdown_event():
    await crawler_service.stop()

@app.get("/")
def read_root():
    if os.path.exists(__location__ + "/site"):
        return RedirectResponse(url="/mkdocs")
    # Return a json response
    return {"message": "Crawl4AI API service is running"}


@app.post("/crawl", dependencies=[Depends(verify_token)])
async def crawl(request: CrawlRequest) -> Dict[str, str]:
    task_id = await crawler_service.submit_task(request)
    return {"task_id": task_id}

@app.get("/task/{task_id}", dependencies=[Depends(verify_token)])
async def get_task_status(task_id: str):
    task_info = crawler_service.task_manager.get_task(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")

    response = {
        "status": task_info.status,
        "created_at": task_info.created_at,
    }

    if task_info.status == TaskStatus.COMPLETED:
        # Convert CrawlResult to dict for JSON response
        if isinstance(task_info.result, list):
            response["results"] = [result.dict() for result in task_info.result]
        else:
            response["result"] = task_info.result.dict()
    elif task_info.status == TaskStatus.FAILED:
        response["error"] = task_info.error

    return response

@app.post("/crawl_sync", dependencies=[Depends(verify_token)])
async def crawl_sync(request: CrawlRequest) -> Dict[str, Any]:
    task_id = await crawler_service.submit_task(request)
    
    # Wait up to 60 seconds for task completion
    for _ in range(60):
        task_info = crawler_service.task_manager.get_task(task_id)
        if not task_info:
            raise HTTPException(status_code=404, detail="Task not found")
            
        if task_info.status == TaskStatus.COMPLETED:
            # Return same format as /task/{task_id} endpoint
            if isinstance(task_info.result, list):
                return {"status": task_info.status, "results": [result.dict() for result in task_info.result]}
            return {"status": task_info.status, "result": task_info.result.dict()}
            
        if task_info.status == TaskStatus.FAILED:
            raise HTTPException(status_code=500, detail=task_info.error)
            
        await asyncio.sleep(1)
    
    # If we get here, task didn't complete within timeout
    raise HTTPException(status_code=408, detail="Task timed out")

@app.post("/crawl_direct", dependencies=[Depends(verify_token)])
async def crawl_direct(request: CrawlRequest) -> Dict[str, Any]:
    try:
        crawler = await crawler_service.crawler_pool.acquire(**request.crawler_params)
        extraction_strategy = crawler_service._create_extraction_strategy(request.extraction_config)
        
        try:
            if isinstance(request.urls, list):
                results = await crawler.arun_many(
                    urls=[str(url) for url in request.urls],
                    extraction_strategy=extraction_strategy,
                    js_code=request.js_code,
                    wait_for=request.wait_for,
                    css_selector=request.css_selector,
                    screenshot=request.screenshot,
                    magic=request.magic,
                    cache_mode=request.cache_mode,
                    session_id=request.session_id,
                    **request.extra,
                )
                return {"results": [result.dict() for result in results]}
            else:
                result = await crawler.arun(
                    url=str(request.urls),
                    extraction_strategy=extraction_strategy,
                    js_code=request.js_code,
                    wait_for=request.wait_for,
                    css_selector=request.css_selector,
                    screenshot=request.screenshot,
                    magic=request.magic,
                    cache_mode=request.cache_mode,
                    session_id=request.session_id,
                    **request.extra,
                )
                return {"result": result.dict()}
        finally:
            await crawler_service.crawler_pool.release(crawler)
    except Exception as e:
        logger.error(f"Error in direct crawl: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
async def health_check():
    available_slots = await crawler_service.resource_monitor.get_available_slots()
    memory = psutil.virtual_memory()
    return {
        "status": "healthy",
        "available_slots": available_slots,
        "memory_usage": memory.percent,
        "cpu_usage": psutil.cpu_percent(),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11235)
```

## crawl4ai/async_database.py
```
import os, sys
from pathlib import Path
import aiosqlite
import asyncio
from typing import Optional, Tuple, Dict
from contextlib import asynccontextmanager
import logging
import json  # Added for serialization/deserialization
from .utils import ensure_content_dirs, generate_content_hash
from .models import CrawlResult
import xxhash
import aiofiles
from .config import NEED_MIGRATION
from .version_manager import VersionManager
from .async_logger import AsyncLogger
from .utils import get_error_context, create_box_message
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_directory = DB_PATH = os.path.join(os.getenv("CRAWL4_AI_BASE_DIRECTORY", Path.home()), ".crawl4ai")
os.makedirs(DB_PATH, exist_ok=True)
DB_PATH = os.path.join(base_directory, "crawl4ai.db")

class AsyncDatabaseManager:
    def __init__(self, pool_size: int = 10, max_retries: int = 3):
        self.db_path = DB_PATH
        self.content_paths = ensure_content_dirs(os.path.dirname(DB_PATH))
        self.pool_size = pool_size
        self.max_retries = max_retries
        self.connection_pool: Dict[int, aiosqlite.Connection] = {}
        self.pool_lock = asyncio.Lock()
        self.init_lock = asyncio.Lock()
        self.connection_semaphore = asyncio.Semaphore(pool_size)
        self._initialized = False  
        self.version_manager = VersionManager()
        self.logger = AsyncLogger(
            log_file=os.path.join(base_directory, ".crawl4ai", "crawler_db.log"),
            verbose=False,
            tag_width=10
        )
        
        
    async def initialize(self):
        """Initialize the database and connection pool"""
        try:
            self.logger.info("Initializing database", tag="INIT")
            # Ensure the database file exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Check if version update is needed
            needs_update = self.version_manager.needs_update()
            
            # Always ensure base table exists
            await self.ainit_db()
            
            # Verify the table exists
            async with aiosqlite.connect(self.db_path, timeout=30.0) as db:
                async with db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='crawled_data'"
                ) as cursor:
                    result = await cursor.fetchone()
                    if not result:
                        raise Exception("crawled_data table was not created")
            
            # If version changed or fresh install, run updates
            if needs_update:
                self.logger.info("New version detected, running updates", tag="INIT")
                await self.update_db_schema()
                from .migrations import run_migration  # Import here to avoid circular imports
                await run_migration()
                self.version_manager.update_version()  # Update stored version after successful migration
                self.logger.success("Version update completed successfully", tag="COMPLETE")
            else:
                self.logger.success("Database initialization completed successfully", tag="COMPLETE")

                
        except Exception as e:
            self.logger.error(
                message="Database initialization error: {error}",
                tag="ERROR",
                params={"error": str(e)}
            )
            self.logger.info(
                message="Database will be initialized on first use",
                tag="INIT"
            )
                        
            raise

            
    async def cleanup(self):
        """Cleanup connections when shutting down"""
        async with self.pool_lock:
            for conn in self.connection_pool.values():
                await conn.close()
            self.connection_pool.clear()

    @asynccontextmanager
    async def get_connection(self):
        """Connection pool manager with enhanced error handling"""
        if not self._initialized:
            async with self.init_lock:
                if not self._initialized:
                    try:
                        await self.initialize()
                        self._initialized = True
                    except Exception as e:
                        import sys
                        error_context = get_error_context(sys.exc_info())
                        self.logger.error(
                            message="Database initialization failed:\n{error}\n\nContext:\n{context}\n\nTraceback:\n{traceback}",
                            tag="ERROR",
                            force_verbose=True,
                            params={
                                "error": str(e),
                                "context": error_context["code_context"],
                                "traceback": error_context["full_traceback"]
                            }
                        )
                        raise

        await self.connection_semaphore.acquire()
        task_id = id(asyncio.current_task())
        
        try:
            async with self.pool_lock:
                if task_id not in self.connection_pool:
                    try:
                        conn = await aiosqlite.connect(
                            self.db_path,
                            timeout=30.0
                        )
                        await conn.execute('PRAGMA journal_mode = WAL')
                        await conn.execute('PRAGMA busy_timeout = 5000')
                        
                        # Verify database structure
                        async with conn.execute("PRAGMA table_info(crawled_data)") as cursor:
                            columns = await cursor.fetchall()
                            column_names = [col[1] for col in columns]
                            expected_columns = {
                                'url', 'html', 'cleaned_html', 'markdown', 'extracted_content',
                                'success', 'media', 'links', 'metadata', 'screenshot',
                                'response_headers', 'downloaded_files'
                            }
                            missing_columns = expected_columns - set(column_names)
                            if missing_columns:
                                raise ValueError(f"Database missing columns: {missing_columns}")
                        
                        self.connection_pool[task_id] = conn
                    except Exception as e:
                        import sys
                        error_context = get_error_context(sys.exc_info())
                        error_message = (
                            f"Unexpected error in db get_connection at line {error_context['line_no']} "
                            f"in {error_context['function']} ({error_context['filename']}):\n"
                            f"Error: {str(e)}\n\n"
                            f"Code context:\n{error_context['code_context']}"
                        )
                        self.logger.error(
                            message=create_box_message(error_message, type= "error"),
                        )

                        raise

            yield self.connection_pool[task_id]

        except Exception as e:
            import sys
            error_context = get_error_context(sys.exc_info())
            error_message = (
                f"Unexpected error in db get_connection at line {error_context['line_no']} "
                f"in {error_context['function']} ({error_context['filename']}):\n"
                f"Error: {str(e)}\n\n"
                f"Code context:\n{error_context['code_context']}"
            )
            self.logger.error(
                message=create_box_message(error_message, type= "error"),
            )
            raise
        finally:
            async with self.pool_lock:
                if task_id in self.connection_pool:
                    await self.connection_pool[task_id].close()
                    del self.connection_pool[task_id]
            self.connection_semaphore.release()


    async def execute_with_retry(self, operation, *args):
        """Execute database operations with retry logic"""
        for attempt in range(self.max_retries):
            try:
                async with self.get_connection() as db:
                    result = await operation(db, *args)
                    await db.commit()
                    return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(
                        message="Operation failed after {retries} attempts: {error}",
                        tag="ERROR",
                        force_verbose=True,
                        params={
                            "retries": self.max_retries,
                            "error": str(e)
                        }
                    )                    
                    raise
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

    async def ainit_db(self):
        """Initialize database schema"""
        async with aiosqlite.connect(self.db_path, timeout=30.0) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS crawled_data (
                    url TEXT PRIMARY KEY,
                    html TEXT,
                    cleaned_html TEXT,
                    markdown TEXT,
                    extracted_content TEXT,
                    success BOOLEAN,
                    media TEXT DEFAULT "{}",
                    links TEXT DEFAULT "{}",
                    metadata TEXT DEFAULT "{}",
                    screenshot TEXT DEFAULT "",
                    response_headers TEXT DEFAULT "{}",
                    downloaded_files TEXT DEFAULT "{}"  -- New column added
                )
            ''')
            await db.commit()

        

    async def update_db_schema(self):
        """Update database schema if needed"""
        async with aiosqlite.connect(self.db_path, timeout=30.0) as db:
            cursor = await db.execute("PRAGMA table_info(crawled_data)")
            columns = await cursor.fetchall()
            column_names = [column[1] for column in columns]
            
            # List of new columns to add
            new_columns = ['media', 'links', 'metadata', 'screenshot', 'response_headers', 'downloaded_files']
            
            for column in new_columns:
                if column not in column_names:
                    await self.aalter_db_add_column(column, db)
            await db.commit()

    async def aalter_db_add_column(self, new_column: str, db):
        """Add new column to the database"""
        if new_column == 'response_headers':
            await db.execute(f'ALTER TABLE crawled_data ADD COLUMN {new_column} TEXT DEFAULT "{{}}"')
        else:
            await db.execute(f'ALTER TABLE crawled_data ADD COLUMN {new_column} TEXT DEFAULT ""')
        self.logger.info(
            message="Added column '{column}' to the database",
            tag="INIT",
            params={"column": new_column}
        )        


    async def aget_cached_url(self, url: str) -> Optional[CrawlResult]:
        """Retrieve cached URL data as CrawlResult"""
        async def _get(db):
            async with db.execute(
                'SELECT * FROM crawled_data WHERE url = ?', (url,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                    
                # Get column names
                columns = [description[0] for description in cursor.description]
                # Create dict from row data
                row_dict = dict(zip(columns, row))
                
                # Load content from files using stored hashes
                content_fields = {
                    'html': row_dict['html'],
                    'cleaned_html': row_dict['cleaned_html'],
                    'markdown': row_dict['markdown'],
                    'extracted_content': row_dict['extracted_content'],
                    'screenshot': row_dict['screenshot'],
                    'screenshots': row_dict['screenshot'],
                }
                
                for field, hash_value in content_fields.items():
                    if hash_value:
                        content = await self._load_content(
                            hash_value, 
                            field.split('_')[0]  # Get content type from field name
                        )
                        row_dict[field] = content or ""
                    else:
                        row_dict[field] = ""

                # Parse JSON fields
                json_fields = ['media', 'links', 'metadata', 'response_headers']
                for field in json_fields:
                    try:
                        row_dict[field] = json.loads(row_dict[field]) if row_dict[field] else {}
                    except json.JSONDecodeError:
                        row_dict[field] = {}

                # Parse downloaded_files
                try:
                    row_dict['downloaded_files'] = json.loads(row_dict['downloaded_files']) if row_dict['downloaded_files'] else []
                except json.JSONDecodeError:
                    row_dict['downloaded_files'] = []

                # Remove any fields not in CrawlResult model
                valid_fields = CrawlResult.__annotations__.keys()
                filtered_dict = {k: v for k, v in row_dict.items() if k in valid_fields}
                
                return CrawlResult(**filtered_dict)

        try:
            return await self.execute_with_retry(_get)
        except Exception as e:
            self.logger.error(
                message="Error retrieving cached URL: {error}",
                tag="ERROR",
                force_verbose=True,
                params={"error": str(e)}
            )
            return None

    async def acache_url(self, result: CrawlResult):
        """Cache CrawlResult data"""
        # Store content files and get hashes
        content_map = {
            'html': (result.html, 'html'),
            'cleaned_html': (result.cleaned_html or "", 'cleaned'),
            'markdown': (result.markdown or "", 'markdown'),
            'extracted_content': (result.extracted_content or "", 'extracted'),
            'screenshot': (result.screenshot or "", 'screenshots')
        }
        
        content_hashes = {}
        for field, (content, content_type) in content_map.items():
            content_hashes[field] = await self._store_content(content, content_type)

        async def _cache(db):
            await db.execute('''
                INSERT INTO crawled_data (
                    url, html, cleaned_html, markdown,
                    extracted_content, success, media, links, metadata,
                    screenshot, response_headers, downloaded_files
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    html = excluded.html,
                    cleaned_html = excluded.cleaned_html,
                    markdown = excluded.markdown,
                    extracted_content = excluded.extracted_content,
                    success = excluded.success,
                    media = excluded.media,
                    links = excluded.links,
                    metadata = excluded.metadata,
                    screenshot = excluded.screenshot,
                    response_headers = excluded.response_headers,
                    downloaded_files = excluded.downloaded_files
            ''', (
                result.url,
                content_hashes['html'],
                content_hashes['cleaned_html'],
                content_hashes['markdown'],
                content_hashes['extracted_content'],
                result.success,
                json.dumps(result.media),
                json.dumps(result.links),
                json.dumps(result.metadata or {}),
                content_hashes['screenshot'],
                json.dumps(result.response_headers or {}),
                json.dumps(result.downloaded_files or [])
            ))

        try:
            await self.execute_with_retry(_cache)
        except Exception as e:
            self.logger.error(
                message="Error caching URL: {error}",
                tag="ERROR",
                force_verbose=True,
                params={"error": str(e)}
            )
            

    async def aget_total_count(self) -> int:
        """Get total number of cached URLs"""
        async def _count(db):
            async with db.execute('SELECT COUNT(*) FROM crawled_data') as cursor:
                result = await cursor.fetchone()
                return result[0] if result else 0

        try:
            return await self.execute_with_retry(_count)
        except Exception as e:
            self.logger.error(
                message="Error getting total count: {error}",
                tag="ERROR",
                force_verbose=True,
                params={"error": str(e)}
            )
            return 0

    async def aclear_db(self):
        """Clear all data from the database"""
        async def _clear(db):
            await db.execute('DELETE FROM crawled_data')

        try:
            await self.execute_with_retry(_clear)
        except Exception as e:
            self.logger.error(
                message="Error clearing database: {error}",
                tag="ERROR",
                force_verbose=True,
                params={"error": str(e)}
            )

    async def aflush_db(self):
        """Drop the entire table"""
        async def _flush(db):
            await db.execute('DROP TABLE IF EXISTS crawled_data')

        try:
            await self.execute_with_retry(_flush)
        except Exception as e:
            self.logger.error(
                message="Error flushing database: {error}",
                tag="ERROR",
                force_verbose=True,
                params={"error": str(e)}
            )
            
                
    async def _store_content(self, content: str, content_type: str) -> str:
        """Store content in filesystem and return hash"""
        if not content:
            return ""
            
        content_hash = generate_content_hash(content)
        file_path = os.path.join(self.content_paths[content_type], content_hash)
        
        # Only write if file doesn't exist
        if not os.path.exists(file_path):
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
                
        return content_hash

    async def _load_content(self, content_hash: str, content_type: str) -> Optional[str]:
        """Load content from filesystem by hash"""
        if not content_hash:
            return None
            
        file_path = os.path.join(self.content_paths[content_type], content_hash)
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except:
            self.logger.error(
                message="Failed to load content: {file_path}",
                tag="ERROR",
                force_verbose=True,
                params={"file_path": file_path}
            )
            return None

# Create a singleton instance
async_db_manager = AsyncDatabaseManager()

```

## crawl4ai/crawler_strategy.py
```
from abc import ABC, abstractmethod
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import InvalidArgumentException, WebDriverException
# from selenium.webdriver.chrome.service import Service as ChromeService
# from webdriver_manager.chrome import ChromeDriverManager
# from urllib3.exceptions import MaxRetryError

from .config import *
import logging, time
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from typing import List, Callable
import requests
import os
from pathlib import Path
from .utils import *

logger = logging.getLogger('selenium.webdriver.remote.remote_connection')
logger.setLevel(logging.WARNING)

logger_driver = logging.getLogger('selenium.webdriver.common.service')
logger_driver.setLevel(logging.WARNING)

urllib3_logger = logging.getLogger('urllib3.connectionpool')
urllib3_logger.setLevel(logging.WARNING)

# Disable http.client logging
http_client_logger = logging.getLogger('http.client')
http_client_logger.setLevel(logging.WARNING)

# Disable driver_finder and service logging
driver_finder_logger = logging.getLogger('selenium.webdriver.common.driver_finder')
driver_finder_logger.setLevel(logging.WARNING)




class CrawlerStrategy(ABC):
    @abstractmethod
    def crawl(self, url: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def take_screenshot(self, save_path: str):
        pass
    
    @abstractmethod
    def update_user_agent(self, user_agent: str):
        pass
    
    @abstractmethod
    def set_hook(self, hook_type: str, hook: Callable):
        pass

class CloudCrawlerStrategy(CrawlerStrategy):
    def __init__(self, use_cached_html = False):
        super().__init__()
        self.use_cached_html = use_cached_html
        
    def crawl(self, url: str) -> str:
        data = {
            "urls": [url],
            "include_raw_html": True,
            "forced": True,
            "extract_blocks": False,
        }

        response = requests.post("http://crawl4ai.uccode.io/crawl", json=data)
        response = response.json()
        html = response["results"][0]["html"]
        return sanitize_input_encode(html)

class LocalSeleniumCrawlerStrategy(CrawlerStrategy):
    def __init__(self, use_cached_html=False, js_code=None, **kwargs):
        super().__init__()
        print("[LOG] 🚀 Initializing LocalSeleniumCrawlerStrategy")
        self.options = Options()
        self.options.headless = True
        if kwargs.get("proxy"):
            self.options.add_argument("--proxy-server={}".format(kwargs.get("proxy")))
        if kwargs.get("user_agent"):
            self.options.add_argument("--user-agent=" + kwargs.get("user_agent"))
        else:
            user_agent = kwargs.get("user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            self.options.add_argument(f"--user-agent={user_agent}")
            self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
                  
        self.options.headless = kwargs.get("headless", True)
        if self.options.headless:
            self.options.add_argument("--headless")
        
        self.options.add_argument("--disable-gpu")  
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--disable-blink-features=AutomationControlled")     
        
        # self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--disable-gpu")
        # self.options.add_argument("--disable-extensions")
        # self.options.add_argument("--disable-infobars")
        # self.options.add_argument("--disable-logging")
        # self.options.add_argument("--disable-popup-blocking")
        # self.options.add_argument("--disable-translate")
        # self.options.add_argument("--disable-default-apps")
        # self.options.add_argument("--disable-background-networking")
        # self.options.add_argument("--disable-sync")
        # self.options.add_argument("--disable-features=NetworkService,NetworkServiceInProcess")
        # self.options.add_argument("--disable-browser-side-navigation")
        # self.options.add_argument("--dns-prefetch-disable")
        # self.options.add_argument("--disable-web-security")
        self.options.add_argument("--log-level=3")
        self.use_cached_html = use_cached_html
        self.use_cached_html = use_cached_html
        self.js_code = js_code
        self.verbose = kwargs.get("verbose", False)
        
        # Hooks
        self.hooks = {
            'on_driver_created': None,
            'on_user_agent_updated': None,
            'before_get_url': None,
            'after_get_url': None,
            'before_return_html': None
        }

        # chromedriver_autoinstaller.install()
        # import chromedriver_autoinstaller
        # crawl4ai_folder = os.path.join(os.getenv("CRAWL4_AI_BASE_DIRECTORY", Path.home()), ".crawl4ai")
        # driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=self.options)
        # chromedriver_path = chromedriver_autoinstaller.install()
        # chromedriver_path = chromedriver_autoinstaller.utils.download_chromedriver()
        # self.service = Service(chromedriver_autoinstaller.install())
        
        
        # chromedriver_path = ChromeDriverManager().install()
        # self.service = Service(chromedriver_path)
        # self.service.log_path = "NUL"
        # self.driver = webdriver.Chrome(service=self.service, options=self.options)
        
        # Use selenium-manager (built into Selenium 4.10.0+)
        self.service = Service()
        self.driver = webdriver.Chrome(options=self.options)
        
        self.driver = self.execute_hook('on_driver_created', self.driver)
        
        if kwargs.get("cookies"):
            for cookie in kwargs.get("cookies"):
                self.driver.add_cookie(cookie)
            
        

    def set_hook(self, hook_type: str, hook: Callable):
        if hook_type in self.hooks:
            self.hooks[hook_type] = hook
        else:
            raise ValueError(f"Invalid hook type: {hook_type}")
    
    def execute_hook(self, hook_type: str, *args):
        hook = self.hooks.get(hook_type)
        if hook:
            result = hook(*args)
            if result is not None:
                if isinstance(result, webdriver.Chrome):
                    return result
                else:
                    raise TypeError(f"Hook {hook_type} must return an instance of webdriver.Chrome or None.")
        # If the hook returns None or there is no hook, return self.driver
        return self.driver

    def update_user_agent(self, user_agent: str):
        self.options.add_argument(f"user-agent={user_agent}")
        self.driver.quit()
        self.driver = webdriver.Chrome(service=self.service, options=self.options)
        self.driver = self.execute_hook('on_user_agent_updated', self.driver)

    def set_custom_headers(self, headers: dict):
        # Enable Network domain for sending headers
        self.driver.execute_cdp_cmd('Network.enable', {})
        # Set extra HTTP headers
        self.driver.execute_cdp_cmd('Network.setExtraHTTPHeaders', {'headers': headers})

    def _ensure_page_load(self,  max_checks=6, check_interval=0.01):
        initial_length = len(self.driver.page_source)
        
        for ix in range(max_checks):
            # print(f"Checking page load: {ix}")
            time.sleep(check_interval)
            current_length = len(self.driver.page_source)
            
            if current_length != initial_length:
                break

        return self.driver.page_source
    
    def crawl(self, url: str, **kwargs) -> str:
        # Create md5 hash of the URL
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()
        
        if self.use_cached_html:
            cache_file_path = os.path.join(os.getenv("CRAWL4_AI_BASE_DIRECTORY", Path.home()), ".crawl4ai", "cache", url_hash)
            if os.path.exists(cache_file_path):
                with open(cache_file_path, "r") as f:
                    return sanitize_input_encode(f.read())

        try:
            self.driver = self.execute_hook('before_get_url', self.driver)
            if self.verbose:
                print(f"[LOG] 🕸️ Crawling {url} using LocalSeleniumCrawlerStrategy...")
            self.driver.get(url) #<html><head></head><body></body></html>
            
            WebDriverWait(self.driver, 20).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.TAG_NAME, "body"))
            )
            
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            self.driver = self.execute_hook('after_get_url', self.driver)
            html = sanitize_input_encode(self._ensure_page_load()) # self.driver.page_source                                        
            can_not_be_done_headless = False # Look at my creativity for naming variables
            
            # TODO: Very ugly approach, but promise to change it!
            if kwargs.get('bypass_headless', False) or html == "<html><head></head><body></body></html>":
                print("[LOG] 🙌 Page could not be loaded in headless mode. Trying non-headless mode...")
                can_not_be_done_headless = True
                options = Options()
                options.headless = False
                # set window size very small
                options.add_argument("--window-size=5,5")
                driver = webdriver.Chrome(service=self.service, options=options)
                driver.get(url)
                self.driver = self.execute_hook('after_get_url', driver)
                html = sanitize_input_encode(driver.page_source)
                driver.quit()
            
            # Execute JS code if provided
            self.js_code = kwargs.get("js_code", self.js_code)
            if self.js_code and type(self.js_code) == str:
                self.driver.execute_script(self.js_code)
                # Optionally, wait for some condition after executing the JS code
                WebDriverWait(self.driver, 10).until(
                    lambda driver: driver.execute_script("return document.readyState") == "complete"
                )
            elif self.js_code and type(self.js_code) == list:
                for js in self.js_code:
                    self.driver.execute_script(js)
                    WebDriverWait(self.driver, 10).until(
                        lambda driver: driver.execute_script("return document.readyState") == "complete"
                    )
            
            # Optionally, wait for some condition after executing the JS code : Contributed by (https://github.com/jonymusky)
            wait_for = kwargs.get('wait_for', False)
            if wait_for:
                if callable(wait_for):
                    print("[LOG] 🔄 Waiting for condition...")
                    WebDriverWait(self.driver, 20).until(wait_for)
                else:
                    print("[LOG] 🔄 Waiting for condition...")
                    WebDriverWait(self.driver, 20).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, wait_for))
                    ) 
            
            if not can_not_be_done_headless:
                html = sanitize_input_encode(self.driver.page_source)
            self.driver = self.execute_hook('before_return_html', self.driver, html)
            
            # Store in cache
            cache_file_path = os.path.join(os.getenv("CRAWL4_AI_BASE_DIRECTORY", Path.home()), ".crawl4ai", "cache", url_hash)
            with open(cache_file_path, "w", encoding="utf-8") as f:
                f.write(html)
                
            if self.verbose:
                print(f"[LOG] ✅ Crawled {url} successfully!")
            
            return html
        except InvalidArgumentException as e:
            if not hasattr(e, 'msg'):
                e.msg = sanitize_input_encode(str(e))
            raise InvalidArgumentException(f"Failed to crawl {url}: {e.msg}")
        except WebDriverException as e:
            # If e does nlt have msg attribute create it and set it to str(e)
            if not hasattr(e, 'msg'):
                e.msg = sanitize_input_encode(str(e))
            raise WebDriverException(f"Failed to crawl {url}: {e.msg}")  
        except Exception as e:
            if not hasattr(e, 'msg'):
                e.msg = sanitize_input_encode(str(e))
            raise Exception(f"Failed to crawl {url}: {e.msg}")

    def take_screenshot(self) -> str:
        try:
            # Get the dimensions of the page
            total_width = self.driver.execute_script("return document.body.scrollWidth")
            total_height = self.driver.execute_script("return document.body.scrollHeight")

            # Set the window size to the dimensions of the page
            self.driver.set_window_size(total_width, total_height)

            # Take screenshot
            screenshot = self.driver.get_screenshot_as_png()

            # Open the screenshot with PIL
            image = Image.open(BytesIO(screenshot))

            # Convert image to RGB mode (this will handle both RGB and RGBA images)
            rgb_image = image.convert('RGB')

            # Convert to JPEG and compress
            buffered = BytesIO()
            rgb_image.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            if self.verbose:
                print(f"[LOG] 📸 Screenshot taken and converted to base64")

            return img_base64
        except Exception as e:
            error_message = sanitize_input_encode(f"Failed to take screenshot: {str(e)}")
            print(error_message)

            # Generate an image with black background
            img = Image.new('RGB', (800, 600), color='black')
            draw = ImageDraw.Draw(img)
            
            # Load a font
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except IOError:
                font = ImageFont.load_default()

            # Define text color and wrap the text
            text_color = (255, 255, 255)
            max_width = 780
            wrapped_text = wrap_text(draw, error_message, font, max_width)

            # Calculate text position
            text_position = (10, 10)
            
            # Draw the text on the image
            draw.text(text_position, wrapped_text, fill=text_color, font=font)
            
            # Convert to base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return img_base64
        
    def quit(self):
        self.driver.quit()

```

## crawl4ai/__init__.py
```
# __init__.py

from .async_webcrawler import AsyncWebCrawler, CacheMode
from .async_configs import BrowserConfig, CrawlerRunConfig
from .extraction_strategy import ExtractionStrategy, LLMExtractionStrategy, CosineStrategy, JsonCssExtractionStrategy
from .chunking_strategy import ChunkingStrategy, RegexChunking
from .markdown_generation_strategy import DefaultMarkdownGenerator
from .content_filter_strategy import PruningContentFilter, BM25ContentFilter
from .models import CrawlResult
from .__version__ import __version__

__all__ = [
    "AsyncWebCrawler",
    "CrawlResult",
    "CacheMode",
    'BrowserConfig',
    'CrawlerRunConfig',
    'ExtractionStrategy',
    'LLMExtractionStrategy',
    'CosineStrategy',
    'JsonCssExtractionStrategy',
    'ChunkingStrategy',
    'RegexChunking',
    'DefaultMarkdownGenerator',
    'PruningContentFilter',
    'BM25ContentFilter',
]

def is_sync_version_installed():
    try:
        import selenium
        return True
    except ImportError:
        return False

if is_sync_version_installed():
    try:
        from .web_crawler import WebCrawler
        __all__.append("WebCrawler")
    except ImportError:
        import warnings
        print("Warning: Failed to import WebCrawler even though selenium is installed. This might be due to other missing dependencies.")
else:
    WebCrawler = None
    # import warnings
    # print("Warning: Synchronous WebCrawler is not available. Install crawl4ai[sync] for synchronous support. However, please note that the synchronous version will be deprecated soon.")
```

## crawl4ai/content_scraping_strategy.py
```
import re  # Point 1: Pre-Compile Regular Expressions
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import asyncio, requests, re, os
from .config import *
from bs4 import element, NavigableString, Comment
from bs4 import PageElement, Tag
from urllib.parse import urljoin
from requests.exceptions import InvalidSchema
# from .content_cleaning_strategy import ContentCleaningStrategy
from .content_filter_strategy import RelevantContentFilter, BM25ContentFilter#, HeuristicContentFilter
from .markdown_generation_strategy import MarkdownGenerationStrategy, DefaultMarkdownGenerator
from .models import MarkdownGenerationResult
from .utils import (
    extract_metadata,
    normalize_url,
    is_external_url    
)


# Pre-compile regular expressions for Open Graph and Twitter metadata
OG_REGEX = re.compile(r'^og:')
TWITTER_REGEX = re.compile(r'^twitter:')
DIMENSION_REGEX = re.compile(r"(\d+)(\D*)")

# Function to parse image height/width value and units
def parse_dimension(dimension):
    if dimension:
        # match = re.match(r"(\d+)(\D*)", dimension)
        match = DIMENSION_REGEX.match(dimension)
        if match:
            number = int(match.group(1))
            unit = match.group(2) or 'px'  # Default unit is 'px' if not specified
            return number, unit
    return None, None

# Fetch image file metadata to extract size and extension
def fetch_image_file_size(img, base_url):
    #If src is relative path construct full URL, if not it may be CDN URL
    img_url = urljoin(base_url,img.get('src'))
    try:
        response = requests.head(img_url)
        if response.status_code == 200:
            return response.headers.get('Content-Length',None)
        else:
            print(f"Failed to retrieve file size for {img_url}")
            return None
    except InvalidSchema as e:
        return None
    finally:
        return

class ContentScrapingStrategy(ABC):
    @abstractmethod
    def scrap(self, url: str, html: str, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def ascrap(self, url: str, html: str, **kwargs) -> Dict[str, Any]:
        pass

class WebScrapingStrategy(ContentScrapingStrategy):
    def __init__(self, logger=None):
        self.logger = logger

    def _log(self, level, message, tag="SCRAPE", **kwargs):
        """Helper method to safely use logger."""
        if self.logger:
            log_method = getattr(self.logger, level)
            log_method(message=message, tag=tag, **kwargs)
                
    def scrap(self, url: str, html: str, **kwargs) -> Dict[str, Any]:
        return self._scrap(url, html, is_async=False, **kwargs)

    async def ascrap(self, url: str, html: str, **kwargs) -> Dict[str, Any]:
        return await asyncio.to_thread(self._scrap, url, html, **kwargs)

    def _generate_markdown_content(self, 
                                 cleaned_html: str,
                                 html: str,
                                 url: str,
                                 success: bool,
                                 **kwargs) -> Dict[str, Any]:
        markdown_generator: Optional[MarkdownGenerationStrategy] = kwargs.get('markdown_generator', DefaultMarkdownGenerator())
        
        if markdown_generator:
            try:
                if kwargs.get('fit_markdown', False) and not markdown_generator.content_filter:
                        markdown_generator.content_filter = BM25ContentFilter(
                            user_query=kwargs.get('fit_markdown_user_query', None),
                            bm25_threshold=kwargs.get('fit_markdown_bm25_threshold', 1.0)
                        )
                
                markdown_result: MarkdownGenerationResult = markdown_generator.generate_markdown(
                    cleaned_html=cleaned_html,
                    base_url=url,
                    html2text_options=kwargs.get('html2text', {})
                )
                
                return {
                    'markdown': markdown_result.raw_markdown,  
                    'fit_markdown': markdown_result.fit_markdown,
                    'fit_html': markdown_result.fit_html, 
                    'markdown_v2': markdown_result
                }
            except Exception as e:
                self._log('error',
                    message="Error using new markdown generation strategy: {error}",
                    tag="SCRAPE",
                    params={"error": str(e)}
                )
                markdown_generator = None
                return {
                    'markdown': f"Error using new markdown generation strategy: {str(e)}",
                    'fit_markdown': "Set flag 'fit_markdown' to True to get cleaned HTML content.",
                    'fit_html': "Set flag 'fit_markdown' to True to get cleaned HTML content.",
                    'markdown_v2': None                    
                }

        # Legacy method
        """
        # h = CustomHTML2Text()
        # h.update_params(**kwargs.get('html2text', {}))            
        # markdown = h.handle(cleaned_html)
        # markdown = markdown.replace('    ```', '```')
        
        # fit_markdown = "Set flag 'fit_markdown' to True to get cleaned HTML content."
        # fit_html = "Set flag 'fit_markdown' to True to get cleaned HTML content."
        
        # if kwargs.get('content_filter', None) or kwargs.get('fit_markdown', False):
        #     content_filter = kwargs.get('content_filter', None)
        #     if not content_filter:
        #         content_filter = BM25ContentFilter(
        #             user_query=kwargs.get('fit_markdown_user_query', None),
        #             bm25_threshold=kwargs.get('fit_markdown_bm25_threshold', 1.0)
        #         )
        #     fit_html = content_filter.filter_content(html)
        #     fit_html = '\n'.join('<div>{}</div>'.format(s) for s in fit_html)
        #     fit_markdown = h.handle(fit_html)

        # markdown_v2 = MarkdownGenerationResult(
        #     raw_markdown=markdown,
        #     markdown_with_citations=markdown,
        #     references_markdown=markdown,
        #     fit_markdown=fit_markdown
        # )
        
        # return {
        #     'markdown': markdown,
        #     'fit_markdown': fit_markdown,
        #     'fit_html': fit_html,
        #     'markdown_v2' : markdown_v2
        # }
        """

    def flatten_nested_elements(self, node):
        if isinstance(node, NavigableString):
            return node
        if len(node.contents) == 1 and isinstance(node.contents[0], Tag) and node.contents[0].name == node.name:
            return self.flatten_nested_elements(node.contents[0])
        node.contents = [self.flatten_nested_elements(child) for child in node.contents]
        return node

    def find_closest_parent_with_useful_text(self, tag, **kwargs):
        image_description_min_word_threshold = kwargs.get('image_description_min_word_threshold', IMAGE_DESCRIPTION_MIN_WORD_THRESHOLD)
        current_tag = tag
        while current_tag:
            current_tag = current_tag.parent
            # Get the text content of the parent tag
            if current_tag:
                text_content = current_tag.get_text(separator=' ',strip=True)
                # Check if the text content has at least word_count_threshold
                if len(text_content.split()) >= image_description_min_word_threshold:
                    return text_content
        return None

    def remove_unwanted_attributes(self, element, important_attrs, keep_data_attributes=False):
        attrs_to_remove = []
        for attr in element.attrs:
            if attr not in important_attrs:
                if keep_data_attributes:
                    if not attr.startswith('data-'):
                        attrs_to_remove.append(attr)
                else:
                    attrs_to_remove.append(attr)
        
        for attr in attrs_to_remove:
            del element[attr]

    def process_image(self, img, url, index, total_images, **kwargs):
        parse_srcset = lambda s: [{'url': u.strip().split()[0], 'width': u.strip().split()[-1].rstrip('w') 
                        if ' ' in u else None} 
                        for u in [f"http{p}" for p in s.split("http") if p]]
        
        # Constants for checks
        classes_to_check = frozenset(['button', 'icon', 'logo'])
        tags_to_check = frozenset(['button', 'input'])
        
        # Pre-fetch commonly used attributes
        style = img.get('style', '')
        alt = img.get('alt', '')
        src = img.get('src', '')
        data_src = img.get('data-src', '')
        width = img.get('width')
        height = img.get('height')
        parent = img.parent
        parent_classes = parent.get('class', [])

        # Quick validation checks
        if ('display:none' in style or
            parent.name in tags_to_check or
            any(c in cls for c in parent_classes for cls in classes_to_check) or
            any(c in src for c in classes_to_check) or
            any(c in alt for c in classes_to_check)):
            return None

        # Quick score calculation
        score = 0
        if width and width.isdigit():
            width_val = int(width)
            score += 1 if width_val > 150 else 0
        if height and height.isdigit():
            height_val = int(height)
            score += 1 if height_val > 150 else 0
        if alt:
            score += 1
        score += index/total_images < 0.5
        
        image_format = ''
        if "data:image/" in src:
            image_format = src.split(',')[0].split(';')[0].split('/')[1].split(';')[0]
        else:
            image_format = os.path.splitext(src)[1].lower().strip('.').split('?')[0]
        
        if image_format in ('jpg', 'png', 'webp', 'avif'):
            score += 1

        if score <= kwargs.get('image_score_threshold', IMAGE_SCORE_THRESHOLD):
            return None

        # Use set for deduplication
        unique_urls = set()
        image_variants = []
        
        # Generate a unique group ID for this set of variants
        group_id = index 
        
        # Base image info template
        image_description_min_word_threshold = kwargs.get('image_description_min_word_threshold', IMAGE_DESCRIPTION_MIN_WORD_THRESHOLD)
        base_info = {
            'alt': alt,
            'desc': self.find_closest_parent_with_useful_text(img, **kwargs),
            'score': score,
            'type': 'image',
            'group_id': group_id # Group ID for this set of variants
        }

        # Inline function for adding variants
        def add_variant(src, width=None):
            if src and not src.startswith('data:') and src not in unique_urls:
                unique_urls.add(src)
                image_variants.append({**base_info, 'src': src, 'width': width})

        # Process all sources
        add_variant(src)
        add_variant(data_src)
        
        # Handle srcset and data-srcset in one pass
        for attr in ('srcset', 'data-srcset'):
            if value := img.get(attr):
                for source in parse_srcset(value):
                    add_variant(source['url'], source['width'])

        # Quick picture element check
        if picture := img.find_parent('picture'):
            for source in picture.find_all('source'):
                if srcset := source.get('srcset'):
                    for src in parse_srcset(srcset):
                        add_variant(src['url'], src['width'])

        # Framework-specific attributes in one pass
        for attr, value in img.attrs.items():
            if attr.startswith('data-') and ('src' in attr or 'srcset' in attr) and 'http' in value:
                add_variant(value)

        return image_variants if image_variants else None

    
    def process_element(self, url, element: PageElement, **kwargs) -> Dict[str, Any]:        
        media = {'images': [], 'videos': [], 'audios': []}
        internal_links_dict = {}
        external_links_dict = {}
        self._process_element(
            url,
            element,
            media,
            internal_links_dict,
            external_links_dict,
            **kwargs
        )
        return {
            'media': media,
            'internal_links_dict': internal_links_dict,
            'external_links_dict': external_links_dict
        }
        
    def _process_element(self, url, element: PageElement,  media: Dict[str, Any], internal_links_dict: Dict[str, Any], external_links_dict: Dict[str, Any], **kwargs) -> bool:
        try:
            if isinstance(element, NavigableString):
                if isinstance(element, Comment):
                    element.extract()
                return False
            
            # if element.name == 'img':
            #     process_image(element, url, 0, 1)
            #     return True

            if element.name in ['script', 'style', 'link', 'meta', 'noscript']:
                element.decompose()
                return False

            keep_element = False
            
            exclude_social_media_domains = SOCIAL_MEDIA_DOMAINS + kwargs.get('exclude_social_media_domains', [])
            exclude_social_media_domains = list(set(exclude_social_media_domains))
            
            try:
                if element.name == 'a' and element.get('href'):
                    href = element.get('href', '').strip()
                    if not href:  # Skip empty hrefs
                        return False
                        
                    url_base = url.split('/')[2]
                    
                    # Normalize the URL
                    try:
                        normalized_href = normalize_url(href, url)
                    except ValueError as e:
                        # logging.warning(f"Invalid URL format: {href}, Error: {str(e)}")
                        return False
                        
                    link_data = {
                        'href': normalized_href,
                        'text': element.get_text().strip(),
                        'title': element.get('title', '').strip()
                    }
                    
                    # Check for duplicates and add to appropriate dictionary
                    is_external = is_external_url(normalized_href, url_base)
                    if is_external:
                        if normalized_href not in external_links_dict:
                            external_links_dict[normalized_href] = link_data
                    else:
                        if normalized_href not in internal_links_dict:
                            internal_links_dict[normalized_href] = link_data
                            
                    keep_element = True
                    
                    # Handle external link exclusions
                    if is_external:
                        if kwargs.get('exclude_external_links', False):
                            element.decompose()
                            return False
                        elif kwargs.get('exclude_social_media_links', False):
                            if any(domain in normalized_href.lower() for domain in exclude_social_media_domains):
                                element.decompose()
                                return False
                        elif kwargs.get('exclude_domains', []):
                            if any(domain in normalized_href.lower() for domain in kwargs.get('exclude_domains', [])):
                                element.decompose()
                                return False
                                
            except Exception as e:
                raise Exception(f"Error processing links: {str(e)}")

            try:
                if element.name == 'img':
                    potential_sources = ['src', 'data-src', 'srcset' 'data-lazy-src', 'data-original']
                    src = element.get('src', '')
                    while not src and potential_sources:
                        src = element.get(potential_sources.pop(0), '')
                    if not src:
                        element.decompose()
                        return False
                    
                    # If it is srcset pick up the first image
                    if 'srcset' in element.attrs:
                        src = element.attrs['srcset'].split(',')[0].split(' ')[0]
                        
                    # Check flag if we should remove external images
                    if kwargs.get('exclude_external_images', False):
                        src_url_base = src.split('/')[2]
                        url_base = url.split('/')[2]
                        if url_base not in src_url_base:
                            element.decompose()
                            return False
                        
                    if not kwargs.get('exclude_external_images', False) and kwargs.get('exclude_social_media_links', False):
                        src_url_base = src.split('/')[2]
                        url_base = url.split('/')[2]
                        if any(domain in src for domain in exclude_social_media_domains):
                            element.decompose()
                            return False
                        
                    # Handle exclude domains
                    if kwargs.get('exclude_domains', []):
                        if any(domain in src for domain in kwargs.get('exclude_domains', [])):
                            element.decompose()
                            return False
                    
                    return True  # Always keep image elements
            except Exception as e:
                raise "Error processing images"
            
            
            # Check if flag to remove all forms is set
            if kwargs.get('remove_forms', False) and element.name == 'form':
                element.decompose()
                return False
            
            if element.name in ['video', 'audio']:
                media[f"{element.name}s"].append({
                    'src': element.get('src'),
                    'alt': element.get('alt'),
                    'type': element.name,
                    'description': self.find_closest_parent_with_useful_text(element, **kwargs)
                })
                source_tags = element.find_all('source')
                for source_tag in source_tags:
                    media[f"{element.name}s"].append({
                    'src': source_tag.get('src'),
                    'alt': element.get('alt'),
                    'type': element.name,
                    'description': self.find_closest_parent_with_useful_text(element, **kwargs)
                })
                return True  # Always keep video and audio elements

            if element.name in ONLY_TEXT_ELIGIBLE_TAGS:
                if kwargs.get('only_text', False):
                    element.replace_with(element.get_text())

            try:
                self.remove_unwanted_attributes(element, IMPORTANT_ATTRS, kwargs.get('keep_data_attributes', False))
            except Exception as e:
                # print('Error removing unwanted attributes:', str(e))
                self._log('error',
                    message="Error removing unwanted attributes: {error}",
                    tag="SCRAPE",
                    params={"error": str(e)}
                )
            # Process children
            for child in list(element.children):
                if isinstance(child, NavigableString) and not isinstance(child, Comment):
                    if len(child.strip()) > 0:
                        keep_element = True
                else:
                    if self._process_element(url, child, media, internal_links_dict, external_links_dict, **kwargs):
                        keep_element = True
                

            # Check word count
            word_count_threshold = kwargs.get('word_count_threshold', MIN_WORD_THRESHOLD)
            if not keep_element:
                word_count = len(element.get_text(strip=True).split())
                keep_element = word_count >= word_count_threshold

            if not keep_element:
                element.decompose()

            return keep_element
        except Exception as e:
            # print('Error processing element:', str(e))
            self._log('error',
                message="Error processing element: {error}",
                tag="SCRAPE",
                params={"error": str(e)}
            )                
            return False

    def _scrap(self, url: str, html: str, word_count_threshold: int = MIN_WORD_THRESHOLD, css_selector: str = None, **kwargs) -> Dict[str, Any]:
        success = True
        if not html:
            return None

        soup = BeautifulSoup(html, 'lxml')
        body = soup.body
        
        try:
            meta = extract_metadata("", soup)
        except Exception as e:
            self._log('error', 
                message="Error extracting metadata: {error}",
                tag="SCRAPE",
                params={"error": str(e)}
            )            
            meta = {}
        
        # Handle tag-based removal first - faster than CSS selection
        excluded_tags = set(kwargs.get('excluded_tags', []) or [])  
        if excluded_tags:
            for element in body.find_all(lambda tag: tag.name in excluded_tags):
                element.extract()
        
        # Handle CSS selector-based removal
        excluded_selector = kwargs.get('excluded_selector', '')
        if excluded_selector:
            is_single_selector = ',' not in excluded_selector and ' ' not in excluded_selector
            if is_single_selector:
                while element := body.select_one(excluded_selector):
                    element.extract()
            else:
                for element in body.select(excluded_selector):
                    element.extract()  
        
        if css_selector:
            selected_elements = body.select(css_selector)
            if not selected_elements:
                return {
                    'markdown': '',
                    'cleaned_html': '',
                    'success': True,
                    'media': {'images': [], 'videos': [], 'audios': []},
                    'links': {'internal': [], 'external': []},
                    'metadata': {},
                    'message': f"No elements found for CSS selector: {css_selector}"
                }
                # raise InvalidCSSSelectorError(f"Invalid CSS selector, No elements found for CSS selector: {css_selector}")
            body = soup.new_tag('div')
            for el in selected_elements:
                body.append(el)

        result_obj = self.process_element(
            url, 
            body, 
            word_count_threshold = word_count_threshold, 
            **kwargs
        )
        
        links = {'internal': [], 'external': []}
        media = result_obj['media']
        internal_links_dict = result_obj['internal_links_dict']
        external_links_dict = result_obj['external_links_dict']
        
        # Update the links dictionary with unique links
        links['internal'] = list(internal_links_dict.values())
        links['external'] = list(external_links_dict.values())

        # # Process images using ThreadPoolExecutor
        imgs = body.find_all('img')
        
        media['images'] = [
            img for result in (self.process_image(img, url, i, len(imgs)) 
                            for i, img in enumerate(imgs))
            if result is not None
            for img in result
        ]

        body = self.flatten_nested_elements(body)
        base64_pattern = re.compile(r'data:image/[^;]+;base64,([^"]+)')
        for img in imgs:
            src = img.get('src', '')
            if base64_pattern.match(src):
                # Replace base64 data with empty string
                img['src'] = base64_pattern.sub('', src)
                
        str_body = ""
        try:
            str_body = body.encode_contents().decode('utf-8')
        except Exception as e:
            # Reset body to the original HTML
            success = False
            body = BeautifulSoup(html, 'html.parser')
            
            # Create a new div with a special ID
            error_div = body.new_tag('div', id='crawl4ai_error_message')
            error_div.string = '''
            Crawl4AI Error: This page is not fully supported.
            
            Possible reasons:
            1. The page may have restrictions that prevent crawling.
            2. The page might not be fully loaded.
            
            Suggestions:
            - Try calling the crawl function with these parameters:
            magic=True,
            - Set headless=False to visualize what's happening on the page.
            
            If the issue persists, please check the page's structure and any potential anti-crawling measures.
            '''
            
            # Append the error div to the body
            body.body.append(error_div)
            str_body = body.encode_contents().decode('utf-8')
            
            print(f"[LOG] 😧 Error: After processing the crawled HTML and removing irrelevant tags, nothing was left in the page. Check the markdown for further details.")
            self._log('error',
                message="After processing the crawled HTML and removing irrelevant tags, nothing was left in the page. Check the markdown for further details.",
                tag="SCRAPE"
            )

        cleaned_html = str_body.replace('\n\n', '\n').replace('  ', ' ')

        # markdown_content = self._generate_markdown_content(
        #     cleaned_html=cleaned_html,
        #     html=html,
        #     url=url,
        #     success=success,
        #     **kwargs
        # )
        
        return {
            # **markdown_content,
            'cleaned_html': cleaned_html,
            'success': success,
            'media': media,
            'links': links,
            'metadata': meta
        }

```

## crawl4ai/async_crawler_strategy.py
```
import asyncio
import base64
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, List, Optional, Awaitable
import os, sys, shutil
import tempfile, subprocess
from playwright.async_api import async_playwright, Page, Browser, Error, BrowserContext
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from playwright.async_api import ProxySettings
from pydantic import BaseModel
import hashlib
import json
import uuid
from .js_snippet import load_js_script
from .models import AsyncCrawlResponse
from .utils import get_error_context
from .user_agent_generator import UserAgentGenerator
from .config import SCREENSHOT_HEIGHT_TRESHOLD, DOWNLOAD_PAGE_TIMEOUT
from .async_configs import BrowserConfig, CrawlerRunConfig
from playwright_stealth import StealthConfig, stealth_async


from io import BytesIO
import base64
from PIL import Image, ImageDraw, ImageFont

stealth_config = StealthConfig(
    webdriver=True,
    chrome_app=True,
    chrome_csi=True,
    chrome_load_times=True,
    chrome_runtime=True,
    navigator_languages=True,
    navigator_plugins=True,
    navigator_permissions=True,
    webgl_vendor=True,
    outerdimensions=True,
    navigator_hardware_concurrency=True,
    media_codecs=True,
)

BROWSER_DISABLE_OPTIONS = [
    "--disable-background-networking",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-breakpad",
    "--disable-client-side-phishing-detection",
    "--disable-component-extensions-with-background-pages",
    "--disable-default-apps",
    "--disable-extensions",
    "--disable-features=TranslateUI",
    "--disable-hang-monitor",
    "--disable-ipc-flooding-protection",
    "--disable-popup-blocking",
    "--disable-prompt-on-repost",
    "--disable-sync",
    "--force-color-profile=srgb",
    "--metrics-recording-only",
    "--no-first-run",
    "--password-store=basic",
    "--use-mock-keychain"
]

class ManagedBrowser:
    def __init__(self, browser_type: str = "chromium", user_data_dir: Optional[str] = None, headless: bool = False, logger = None, host: str = "localhost", debugging_port: int = 9222):
        self.browser_type = browser_type
        self.user_data_dir = user_data_dir
        self.headless = headless
        self.browser_process = None
        self.temp_dir = None
        self.debugging_port = debugging_port
        self.host = host
        self.logger = logger
        self.shutting_down = False

    async def start(self) -> str:
        """
        Starts the browser process and returns the CDP endpoint URL.
        If user_data_dir is not provided, creates a temporary directory.
        """
        
        # Create temp dir if needed
        if not self.user_data_dir:
            self.temp_dir = tempfile.mkdtemp(prefix="browser-profile-")
            self.user_data_dir = self.temp_dir

        # Get browser path and args based on OS and browser type
        browser_path = self._get_browser_path()
        args = self._get_browser_args()

        # Start browser process
        try:
            self.browser_process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            # Monitor browser process output for errors
            asyncio.create_task(self._monitor_browser_process())
            await asyncio.sleep(2)  # Give browser time to start
            return f"http://{self.host}:{self.debugging_port}"
        except Exception as e:
            await self.cleanup()
            raise Exception(f"Failed to start browser: {e}")

    async def _monitor_browser_process(self):
        """Monitor the browser process for unexpected termination."""
        if self.browser_process:
            try:
                stdout, stderr = await asyncio.gather(
                    asyncio.to_thread(self.browser_process.stdout.read),
                    asyncio.to_thread(self.browser_process.stderr.read)
                )
                
                # Check shutting_down flag BEFORE logging anything
                if self.browser_process.poll() is not None:
                    if not self.shutting_down:
                        self.logger.error(
                            message="Browser process terminated unexpectedly | Code: {code} | STDOUT: {stdout} | STDERR: {stderr}",
                            tag="ERROR",
                            params={
                                "code": self.browser_process.returncode,
                                "stdout": stdout.decode(),
                                "stderr": stderr.decode()
                            }
                        )                
                        await self.cleanup()
                    else:
                        self.logger.info(
                            message="Browser process terminated normally | Code: {code}",
                            tag="INFO",
                            params={"code": self.browser_process.returncode}
                        )
            except Exception as e:
                if not self.shutting_down:
                    self.logger.error(
                        message="Error monitoring browser process: {error}",
                        tag="ERROR",
                        params={"error": str(e)}
                    )

    def _get_browser_path(self) -> str:
        """Returns the browser executable path based on OS and browser type"""
        if sys.platform == "darwin":  # macOS
            paths = {
                "chromium": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "firefox": "/Applications/Firefox.app/Contents/MacOS/firefox",
                "webkit": "/Applications/Safari.app/Contents/MacOS/Safari"
            }
        elif sys.platform == "win32":  # Windows
            paths = {
                "chromium": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                "firefox": "C:\\Program Files\\Mozilla Firefox\\firefox.exe",
                "webkit": None  # WebKit not supported on Windows
            }
        else:  # Linux
            paths = {
                "chromium": "google-chrome",
                "firefox": "firefox",
                "webkit": None  # WebKit not supported on Linux
            }
        
        return paths.get(self.browser_type)

    def _get_browser_args(self) -> List[str]:
        """Returns browser-specific command line arguments"""
        base_args = [self._get_browser_path()]
        
        if self.browser_type == "chromium":
            args = [
                f"--remote-debugging-port={self.debugging_port}",
                f"--user-data-dir={self.user_data_dir}",
            ]
            if self.headless:
                args.append("--headless=new")
        elif self.browser_type == "firefox":
            args = [
                "--remote-debugging-port", str(self.debugging_port),
                "--profile", self.user_data_dir,
            ]
            if self.headless:
                args.append("--headless")
        else:
            raise NotImplementedError(f"Browser type {self.browser_type} not supported")
            
        return base_args + args

    async def cleanup(self):
        """Cleanup browser process and temporary directory"""
        # Set shutting_down flag BEFORE any termination actions
        self.shutting_down = True
        
        if self.browser_process:
            try:
                self.browser_process.terminate()
                # Wait for process to end gracefully
                for _ in range(10):  # 10 attempts, 100ms each
                    if self.browser_process.poll() is not None:
                        break
                    await asyncio.sleep(0.1)
                
                # Force kill if still running
                if self.browser_process.poll() is None:
                    self.browser_process.kill()
                    await asyncio.sleep(0.1)  # Brief wait for kill to take effect
                    
            except Exception as e:
                self.logger.error(
                    message="Error terminating browser: {error}",
                    tag="ERROR",
                    params={"error": str(e)}
                )

        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                self.logger.error(
                    message="Error removing temporary directory: {error}",
                    tag="ERROR",
                    params={"error": str(e)}
                )

class BrowserManager:
    def __init__(self, browser_config: BrowserConfig, logger=None):
        """
        Initialize the BrowserManager with a browser configuration.
        
        Args:
            browser_config (BrowserConfig): Configuration object containing all browser settings
            logger: Logger instance for recording events and errors
        """
        self.config = browser_config
        self.logger = logger
        
        # Browser state
        self.browser = None
        self.default_context = None
        self.managed_browser = None
        self.playwright = None
        
        # Session management
        self.sessions = {}
        self.session_ttl = 1800  # 30 minutes
        
        # Initialize ManagedBrowser if needed
        if self.config.use_managed_browser:
            self.managed_browser = ManagedBrowser(
                browser_type=self.config.browser_type,
                user_data_dir=self.config.user_data_dir,
                headless=self.config.headless,
                logger=self.logger
            )

    async def start(self):
        """Start the browser instance and set up the default context."""
        if self.playwright is None:
            from playwright.async_api import async_playwright
            self.playwright = await async_playwright().start()

        if self.config.use_managed_browser:
            cdp_url = await self.managed_browser.start()
            self.browser = await self.playwright.chromium.connect_over_cdp(cdp_url)
            contexts = self.browser.contexts
            if contexts:
                self.default_context = contexts[0]
            else:
                self.default_context = await self.browser.new_context(
                    viewport={"width": self.config.viewport_width, "height": self.config.viewport_height},
                    storage_state=self.config.storage_state,
                    user_agent=self.config.headers.get("User-Agent", self.config.user_agent),
                    accept_downloads=self.config.accept_downloads,
                    ignore_https_errors=self.config.ignore_https_errors,
                    java_script_enabled=self.config.java_script_enabled
                )
            await self.setup_context(self.default_context)
        else:
            browser_args = self._build_browser_args()
            
            # Launch appropriate browser type
            if self.config.browser_type == "firefox":
                self.browser = await self.playwright.firefox.launch(**browser_args)
            elif self.config.browser_type == "webkit":
                self.browser = await self.playwright.webkit.launch(**browser_args)
            else:
                self.browser = await self.playwright.chromium.launch(**browser_args)

            self.default_context = self.browser

    def _build_browser_args(self) -> dict:
        """Build browser launch arguments from config."""
        args = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-infobars",
            "--window-position=0,0",
            "--ignore-certificate-errors",
            "--ignore-certificate-errors-spki-list",
            "--disable-blink-features=AutomationControlled",
            "--window-position=400,0",
            f"--window-size={self.config.viewport_width},{self.config.viewport_height}",
        ]

        if self.config.light_mode:
            args.extend(BROWSER_DISABLE_OPTIONS)

        if self.config.text_only:
            args.extend(['--blink-settings=imagesEnabled=false', '--disable-remote-fonts'])

        if self.config.extra_args:
            args.extend(self.config.extra_args)

        browser_args = {
            "headless": self.config.headless,
            "args": args
        }

        if self.config.chrome_channel:
            browser_args["channel"] = self.config.chrome_channel

        if self.config.accept_downloads:
            browser_args["downloads_path"] = (self.config.downloads_path or 
                                           os.path.join(os.getcwd(), "downloads"))
            os.makedirs(browser_args["downloads_path"], exist_ok=True)

        if self.config.proxy or self.config.proxy_config:
            from playwright.async_api import ProxySettings
            proxy_settings = (
                ProxySettings(server=self.config.proxy) if self.config.proxy else
                ProxySettings(
                    server=self.config.proxy_config.get("server"),
                    username=self.config.proxy_config.get("username"),
                    password=self.config.proxy_config.get("password")
                )
            )
            browser_args["proxy"] = proxy_settings

        return browser_args

    async def setup_context(self, context: BrowserContext, is_default=False):
        """Set up a browser context with the configured options."""
        if self.config.headers:
            await context.set_extra_http_headers(self.config.headers)

        if self.config.cookies:
            await context.add_cookies(self.config.cookies)

        if self.config.storage_state:
            await context.storage_state(path=None)

        if self.config.accept_downloads:
            context.set_default_timeout(DOWNLOAD_PAGE_TIMEOUT)
            context.set_default_navigation_timeout(DOWNLOAD_PAGE_TIMEOUT)
            if self.config.downloads_path:
                context._impl_obj._options["accept_downloads"] = True
                context._impl_obj._options["downloads_path"] = self.config.downloads_path

        # Handle user agent and browser hints
        if self.config.user_agent:
            combined_headers = {
                "User-Agent": self.config.user_agent,
                "sec-ch-ua": self.config.browser_hint
            }
            combined_headers.update(self.config.headers)
            await context.set_extra_http_headers(combined_headers)

    async def get_page(self, session_id: Optional[str], user_agent: str):
        """Get a page for the given session ID, creating a new one if needed."""
        self._cleanup_expired_sessions()

        if session_id and session_id in self.sessions:
            context, page, _ = self.sessions[session_id]
            self.sessions[session_id] = (context, page, time.time())
            return page, context

        if self.config.use_managed_browser:
            context = self.default_context
            page = await context.new_page()
        else:
            context = await self.browser.new_context(
                user_agent=user_agent,
                viewport={"width": self.config.viewport_width, "height": self.config.viewport_height},
                proxy={"server": self.config.proxy} if self.config.proxy else None,
                accept_downloads=self.config.accept_downloads,
                storage_state=self.config.storage_state,
                ignore_https_errors=self.config.ignore_https_errors
            )
            await self.setup_context(context)
            page = await context.new_page()

        if session_id:
            self.sessions[session_id] = (context, page, time.time())

        return page, context

    async def kill_session(self, session_id: str):
        """Kill a browser session and clean up resources."""
        if session_id in self.sessions:
            context, page, _ = self.sessions[session_id]
            await page.close()
            if not self.config.use_managed_browser:
                await context.close()
            del self.sessions[session_id]

    def _cleanup_expired_sessions(self):
        """Clean up expired sessions based on TTL."""
        current_time = time.time()
        expired_sessions = [
            sid for sid, (_, _, last_used) in self.sessions.items()
            if current_time - last_used > self.session_ttl
        ]
        for sid in expired_sessions:
            asyncio.create_task(self.kill_session(sid))

    async def close(self):
        """Close all browser resources and clean up."""
        if self.config.sleep_on_close:
            await asyncio.sleep(0.5)
            
        session_ids = list(self.sessions.keys())
        for session_id in session_ids:
            await self.kill_session(session_id)

        if self.browser:
            await self.browser.close()
            self.browser = None

        if self.managed_browser:
            await asyncio.sleep(0.5)
            await self.managed_browser.cleanup()
            self.managed_browser = None

        if self.playwright:
            await self.playwright.stop()
            self.playwright = None

class AsyncCrawlerStrategy(ABC):
    @abstractmethod
    async def crawl(self, url: str, **kwargs) -> AsyncCrawlResponse:
        pass # 4 + 3
    
    @abstractmethod
    async def crawl_many(self, urls: List[str], **kwargs) -> List[AsyncCrawlResponse]:
        pass
    
    @abstractmethod
    async def take_screenshot(self, **kwargs) -> str:
        pass
    
    @abstractmethod
    def update_user_agent(self, user_agent: str):
        pass
    
    @abstractmethod
    def set_hook(self, hook_type: str, hook: Callable):
        pass

class AsyncPlaywrightCrawlerStrategy(AsyncCrawlerStrategy):
    def __init__(self, browser_config: BrowserConfig = None, logger = None, **kwargs):
        """
        Initialize the AsyncPlaywrightCrawlerStrategy with a browser configuration.
        
        Args:
            browser_config (BrowserConfig): Configuration object containing browser settings.
                                          If None, will be created from kwargs for backwards compatibility.
            logger: Logger instance for recording events and errors.
            **kwargs: Additional arguments for backwards compatibility and extending functionality.
        """
        # Initialize browser config, either from provided object or kwargs
        self.browser_config = browser_config or BrowserConfig.from_kwargs(kwargs)
        self.logger = logger
        
        # Initialize session management
        self._downloaded_files = []
        
        # Initialize hooks system
        self.hooks = {
            'on_browser_created': None,
            'on_user_agent_updated': None,
            'on_execution_started': None,
            'before_goto': None,
            'after_goto': None,
            'before_return_html': None,
            'before_retrieve_html': None
        }
        
        # Initialize browser manager with config
        self.browser_manager = BrowserManager(
            browser_config=self.browser_config,
            logger=self.logger
        )

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def start(self):
        await self.browser_manager.start()
        await self.execute_hook('on_browser_created', self.browser_manager.browser, context = self.browser_manager.default_context)
        
    async def close(self):
        await self.browser_manager.close()
        
    async def kill_session(self, session_id: str):
        # Log a warning message and no need kill session, in new version auto kill session
        self.logger.warning(
            message="Session auto-kill is enabled in the new version. No need to manually kill sessions.",
            tag="WARNING"
        )
        await self.browser_manager.kill_session(session_id)

    def set_hook(self, hook_type: str, hook: Callable):
        if hook_type in self.hooks:
            self.hooks[hook_type] = hook
        else:
            raise ValueError(f"Invalid hook type: {hook_type}")

    async def execute_hook(self, hook_type: str, *args, **kwargs):
        hook = self.hooks.get(hook_type)
        if hook:
            if asyncio.iscoroutinefunction(hook):
                return await hook(*args, **kwargs)
            else:
                return hook(*args, **kwargs)
        return args[0] if args else None

    def update_user_agent(self, user_agent: str):
        self.user_agent = user_agent

    def set_custom_headers(self, headers: Dict[str, str]):
        self.headers = headers
            
    async def smart_wait(self, page: Page, wait_for: str, timeout: float = 30000):
        wait_for = wait_for.strip()
        
        if wait_for.startswith('js:'):
            # Explicitly specified JavaScript
            js_code = wait_for[3:].strip()
            return await self.csp_compliant_wait(page, js_code, timeout)
        elif wait_for.startswith('css:'):
            # Explicitly specified CSS selector
            css_selector = wait_for[4:].strip()
            try:
                await page.wait_for_selector(css_selector, timeout=timeout)
            except Error as e:
                if 'Timeout' in str(e):
                    raise TimeoutError(f"Timeout after {timeout}ms waiting for selector '{css_selector}'")
                else:
                    raise ValueError(f"Invalid CSS selector: '{css_selector}'")
        else:
            # Auto-detect based on content
            if wait_for.startswith('()') or wait_for.startswith('function'):
                # It's likely a JavaScript function
                return await self.csp_compliant_wait(page, wait_for, timeout)
            else:
                # Assume it's a CSS selector first
                try:
                    await page.wait_for_selector(wait_for, timeout=timeout)
                except Error as e:
                    if 'Timeout' in str(e):
                        raise TimeoutError(f"Timeout after {timeout}ms waiting for selector '{wait_for}'")
                    else:
                        # If it's not a timeout error, it might be an invalid selector
                        # Let's try to evaluate it as a JavaScript function as a fallback
                        try:
                            return await self.csp_compliant_wait(page, f"() => {{{wait_for}}}", timeout)
                        except Error:
                            raise ValueError(f"Invalid wait_for parameter: '{wait_for}'. "
                                             "It should be either a valid CSS selector, a JavaScript function, "
                                             "or explicitly prefixed with 'js:' or 'css:'.")
    
    async def csp_compliant_wait(self, page: Page, user_wait_function: str, timeout: float = 30000):
        wrapper_js = f"""
        async () => {{
            const userFunction = {user_wait_function};
            const startTime = Date.now();
            while (true) {{
                if (await userFunction()) {{
                    return true;
                }}
                if (Date.now() - startTime > {timeout}) {{
                    throw new Error('Timeout waiting for condition');
                }}
                await new Promise(resolve => setTimeout(resolve, 100));
            }}
        }}
        """
        
        try:
            await page.evaluate(wrapper_js)
        except TimeoutError:
            raise TimeoutError(f"Timeout after {timeout}ms waiting for condition")
        except Exception as e:
            raise RuntimeError(f"Error in wait condition: {str(e)}")

    async def process_iframes(self, page):
        # Find all iframes
        iframes = await page.query_selector_all('iframe')
        
        for i, iframe in enumerate(iframes):
            try:
                # Add a unique identifier to the iframe
                await iframe.evaluate(f'(element) => element.id = "iframe-{i}"')
                
                # Get the frame associated with this iframe
                frame = await iframe.content_frame()
                
                if frame:
                    # Wait for the frame to load
                    await frame.wait_for_load_state('load', timeout=30000)  # 30 seconds timeout
                    
                    # Extract the content of the iframe's body
                    iframe_content = await frame.evaluate('() => document.body.innerHTML')
                    
                    # Generate a unique class name for this iframe
                    class_name = f'extracted-iframe-content-{i}'
                    
                    # Replace the iframe with a div containing the extracted content
                    _iframe = iframe_content.replace('`', '\\`')
                    await page.evaluate(f"""
                        () => {{
                            const iframe = document.getElementById('iframe-{i}');
                            const div = document.createElement('div');
                            div.innerHTML = `{_iframe}`;
                            div.className = '{class_name}';
                            iframe.replaceWith(div);
                        }}
                    """)
                else:
                    self.logger.warning(
                        message="Could not access content frame for iframe {index}",
                        tag="SCRAPE",
                        params={"index": i}
                    )                    
            except Exception as e:
                self.logger.error(
                    message="Error processing iframe {index}: {error}",
                    tag="ERROR",
                    params={"index": i, "error": str(e)}
                )                

        # Return the page object
        return page  
    
    async def create_session(self, **kwargs) -> str:
        """Creates a new browser session and returns its ID."""
        await self.start()
        
        session_id = kwargs.get('session_id') or str(uuid.uuid4())
        
        user_agent = kwargs.get("user_agent", self.user_agent)
        # Use browser_manager to get a fresh page & context assigned to this session_id
        page, context = await self.browser_manager.get_page(session_id, user_agent)
        return session_id
    
    async def crawl(self, url: str, config: CrawlerRunConfig,  **kwargs) -> AsyncCrawlResponse:
        """
        Crawls a given URL or processes raw HTML/local file content based on the URL prefix.

        Args:
            url (str): The URL to crawl. Supported prefixes:
                - 'http://' or 'https://': Web URL to crawl.
                - 'file://': Local file path to process.
                - 'raw:': Raw HTML content to process.
            **kwargs: Additional parameters:
                - 'screenshot' (bool): Whether to take a screenshot.
                - ... [other existing parameters]

        Returns:
            AsyncCrawlResponse: The response containing HTML, headers, status code, and optional screenshot.
        """
        config = config or CrawlerRunConfig.from_kwargs(kwargs)
        response_headers = {}
        status_code = 200  # Default for local/raw HTML
        screenshot_data = None

        if url.startswith(('http://', 'https://')):
            return await self._crawl_web(url, config)

        elif url.startswith('file://'):
            # Process local file
            local_file_path = url[7:]  # Remove 'file://' prefix
            if not os.path.exists(local_file_path):
                raise FileNotFoundError(f"Local file not found: {local_file_path}")
            with open(local_file_path, 'r', encoding='utf-8') as f:
                html = f.read()
            if config.screenshot:
                screenshot_data = await self._generate_screenshot_from_html(html)
            return AsyncCrawlResponse(
                html=html,
                response_headers=response_headers,
                status_code=status_code,
                screenshot=screenshot_data,
                get_delayed_content=None
            )

        elif url.startswith('raw:'):
            # Process raw HTML content
            raw_html = url[4:]  # Remove 'raw:' prefix
            html = raw_html
            if config.screenshot:
                screenshot_data = await self._generate_screenshot_from_html(html)
            return AsyncCrawlResponse(
                html=html,
                response_headers=response_headers,
                status_code=status_code,
                screenshot=screenshot_data,
                get_delayed_content=None
            )
        else:
            raise ValueError("URL must start with 'http://', 'https://', 'file://', or 'raw:'")

    async def _crawl_web(self, url: str, config: CrawlerRunConfig) -> AsyncCrawlResponse:
        """
        Internal method to crawl web URLs with the specified configuration.
        
        Args:
            url (str): The web URL to crawl
            config (CrawlerRunConfig): Configuration object controlling the crawl behavior
        
        Returns:
            AsyncCrawlResponse: The response containing HTML, headers, status code, and optional data
        """
        response_headers = {}
        status_code = None
        
        # Reset downloaded files list for new crawl
        self._downloaded_files = []
        
        # Handle user agent with magic mode
        user_agent = self.browser_config.user_agent
        if config.magic and self.browser_config.user_agent_mode != "random":
            user_agent = UserAgentGenerator().generate(
                **(self.browser_config.user_agent_generator_config or {})
            )
        
        # Get page for session
        page, context = await self.browser_manager.get_page(
            session_id=config.session_id,
            user_agent=user_agent
        )
        
        # Add default cookie
        await context.add_cookies([{"name": "cookiesEnabled", "value": "true", "url": url}])
        
        # Handle navigator overrides
        if config.override_navigator or config.simulate_user or config.magic:
            await context.add_init_script(load_js_script("navigator_overrider"))
        
        # Set up console logging if requested
        if config.log_console:
            page.on("console", lambda msg: self.logger.debug(
                message="Console: {msg}",
                tag="CONSOLE",
                params={"msg": msg.text}
            ))
            page.on("pageerror", lambda exc: self.logger.error(
                message="Page error: {exc}",
                tag="ERROR",
                params={"exc": exc}
            ))
        
        try:
            # Set up download handling
            if self.browser_config.accept_downloads:
                page.on("download", lambda download: asyncio.create_task(self._handle_download(download)))

            # Handle page navigation and content loading
            if not config.js_only:
                await self.execute_hook('before_goto', page, context=context)

                try:
                    response = await page.goto(
                        url,
                        wait_until=config.wait_until,
                        timeout=config.page_timeout
                    )
                except Error as e:
                    raise RuntimeError(f"Failed on navigating ACS-GOTO:\n{str(e)}")
                
                await self.execute_hook('after_goto', page, context=context)
                
                status_code = response.status
                response_headers = response.headers
            else:
                status_code = 200
                response_headers = {}

            # Wait for body element and visibility
            try:
                await page.wait_for_selector('body', state='attached', timeout=30000)
                await page.wait_for_function("""
                    () => {
                        const body = document.body;
                        const style = window.getComputedStyle(body);
                        return style.display !== 'none' && 
                            style.visibility !== 'hidden' && 
                            style.opacity !== '0';
                    }
                """, timeout=30000)
            except Error as e:
                visibility_info = await page.evaluate("""
                    () => {
                        const body = document.body;
                        const style = window.getComputedStyle(body);
                        return {
                            display: style.display,
                            visibility: style.visibility,
                            opacity: style.opacity,
                            hasContent: body.innerHTML.length,
                            classList: Array.from(body.classList)
                        }
                    }
                """)
                
                if self.config.verbose:
                    self.logger.debug(
                        message="Body visibility info: {info}",
                        tag="DEBUG",
                        params={"info": visibility_info}
                    )
                
                if not config.ignore_body_visibility:
                    raise Error(f"Body element is hidden: {visibility_info}")

            # Handle content loading and viewport adjustment
            if not self.browser_config.text_only and (config.wait_for_images or config.adjust_viewport_to_content):
                await page.wait_for_load_state("domcontentloaded")
                await asyncio.sleep(0.1)
                try:
                    await page.wait_for_function(
                        "Array.from(document.images).every(img => img.complete)",
                        timeout=1000
                    )
                except PlaywrightTimeoutError:
                    pass

            # Adjust viewport if needed
            if not self.browser_config.text_only and config.adjust_viewport_to_content:
                try:
                    page_width = await page.evaluate("document.documentElement.scrollWidth")
                    page_height = await page.evaluate("document.documentElement.scrollHeight")
                    
                    target_width = self.browser_config.viewport_width
                    target_height = int(target_width * page_width / page_height * 0.95)
                    await page.set_viewport_size({"width": target_width, "height": target_height})

                    scale = min(target_width / page_width, target_height / page_height)
                    cdp = await page.context.new_cdp_session(page)
                    await cdp.send('Emulation.setDeviceMetricsOverride', {
                        'width': page_width,
                        'height': page_height,
                        'deviceScaleFactor': 1,
                        'mobile': False,
                        'scale': scale
                    })
                except Exception as e:
                    self.logger.warning(
                        message="Failed to adjust viewport to content: {error}",
                        tag="VIEWPORT",
                        params={"error": str(e)}
                    )

            # Handle full page scanning
            if config.scan_full_page:
                await self._handle_full_page_scan(page, config.scroll_delay)

            # Execute JavaScript if provided
            if config.js_code:
                if isinstance(config.js_code, str):
                    await page.evaluate(config.js_code)
                elif isinstance(config.js_code, list):
                    for js in config.js_code:
                        await page.evaluate(js)
                
                await self.execute_hook('on_execution_started', page, context=context)

            # Handle user simulation
            if config.simulate_user or config.magic:
                await page.mouse.move(100, 100)
                await page.mouse.down()
                await page.mouse.up()
                await page.keyboard.press('ArrowDown')

            # Handle wait_for condition
            if config.wait_for:
                try:
                    await self.smart_wait(page, config.wait_for, timeout=config.page_timeout)
                except Exception as e:
                    raise RuntimeError(f"Wait condition failed: {str(e)}")

            # Update image dimensions if needed
            if not self.browser_config.text_only:
                update_image_dimensions_js = load_js_script("update_image_dimensions")
                try:
                    try:
                        await page.wait_for_load_state("domcontentloaded", timeout=5)
                    except PlaywrightTimeoutError:
                        pass
                    await page.evaluate(update_image_dimensions_js)
                except Exception as e:
                    self.logger.error(
                        message="Error updating image dimensions: {error}",
                        tag="ERROR",
                        params={"error": str(e)}
                    )

            # Process iframes if needed
            if config.process_iframes:
                page = await self.process_iframes(page)

            # Pre-content retrieval hooks and delay
            await self.execute_hook('before_retrieve_html', page, context=context)
            if config.delay_before_return_html:
                await asyncio.sleep(config.delay_before_return_html)

            # Handle overlay removal
            if config.remove_overlay_elements:
                await self.remove_overlay_elements(page)

            # Get final HTML content
            html = await page.content()
            await self.execute_hook('before_return_html', page, html, context=context)

            # Handle PDF and screenshot generation
            start_export_time = time.perf_counter()
            pdf_data = None
            screenshot_data = None

            if config.pdf:
                pdf_data = await self.export_pdf(page)

            if config.screenshot:
                if config.screenshot_wait_for:
                    await asyncio.sleep(config.screenshot_wait_for)
                screenshot_data = await self.take_screenshot(
                    page,
                    screenshot_height_threshold=config.screenshot_height_threshold
                )

            if screenshot_data or pdf_data:
                self.logger.info(
                    message="Exporting PDF and taking screenshot took {duration:.2f}s",
                    tag="EXPORT",
                    params={"duration": time.perf_counter() - start_export_time}
                )

            # Define delayed content getter
            async def get_delayed_content(delay: float = 5.0) -> str:
                if self.config.verbose:
                    self.logger.info(
                        message="Waiting for {delay} seconds before retrieving content for {url}",
                        tag="INFO",
                        params={"delay": delay, "url": url}
                    )                    
                await asyncio.sleep(delay)
                return await page.content()

            # Return complete response
            return AsyncCrawlResponse(
                html=html,
                response_headers=response_headers,
                status_code=status_code,
                screenshot=screenshot_data,
                pdf_data=pdf_data,
                get_delayed_content=get_delayed_content,
                downloaded_files=self._downloaded_files if self._downloaded_files else None
            )

        except Exception as e:
            raise e

    async def _handle_full_page_scan(self, page: Page, scroll_delay: float):
        """Helper method to handle full page scanning"""
        try:
            viewport_height = page.viewport_size.get("height", self.browser_config.viewport_height)
            current_position = viewport_height
            
            await page.evaluate(f"window.scrollTo(0, {current_position})")
            await asyncio.sleep(scroll_delay)
            
            total_height = await page.evaluate("document.documentElement.scrollHeight")
            
            while current_position < total_height:
                current_position = min(current_position + viewport_height, total_height)
                await page.evaluate(f"window.scrollTo(0, {current_position})")
                await asyncio.sleep(scroll_delay)
                
                new_height = await page.evaluate("document.documentElement.scrollHeight")
                if new_height > total_height:
                    total_height = new_height
            
            await page.evaluate("window.scrollTo(0, 0)")
            
        except Exception as e:
            self.logger.warning(
                message="Failed to perform full page scan: {error}",
                tag="PAGE_SCAN",
                params={"error": str(e)}
            )
        else:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    
    
    async def _handle_download(self, download):
        """Handle file downloads."""
        try:
            suggested_filename = download.suggested_filename
            download_path = os.path.join(self.downloads_path, suggested_filename)
            
            self.logger.info(
                message="Downloading {filename} to {path}",
                tag="FETCH",
                params={"filename": suggested_filename, "path": download_path}
            )
                
            start_time = time.perf_counter()
            await download.save_as(download_path)
            end_time = time.perf_counter()
            self._downloaded_files.append(download_path)

            self.logger.success(
                message="Downloaded {filename} successfully",
                tag="COMPLETE",
                params={"filename": suggested_filename, "path": download_path, "duration": f"{end_time - start_time:.2f}s"}
            )            
        except Exception as e:
            self.logger.error(
                message="Failed to handle download: {error}",
                tag="ERROR",
                params={"error": str(e)}
            )
            
    
    async def crawl_many(self, urls: List[str], **kwargs) -> List[AsyncCrawlResponse]:
        semaphore_count = kwargs.get('semaphore_count', 5)  # Adjust as needed
        semaphore = asyncio.Semaphore(semaphore_count)

        async def crawl_with_semaphore(url):
            async with semaphore:
                return await self.crawl(url, **kwargs)

        tasks = [crawl_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [result if not isinstance(result, Exception) else str(result) for result in results]

    async def remove_overlay_elements(self, page: Page) -> None:
        """
        Removes popup overlays, modals, cookie notices, and other intrusive elements from the page.
        
        Args:
            page (Page): The Playwright page instance
        """
        remove_overlays_js = load_js_script("remove_overlay_elements")
    
        try:
            await page.evaluate(remove_overlays_js)
            await page.wait_for_timeout(500)  # Wait for any animations to complete
        except Exception as e:
            self.logger.warning(
                message="Failed to remove overlay elements: {error}",
                tag="SCRAPE",
                params={"error": str(e)}
            )            

    async def export_pdf(self, page: Page) -> bytes:
        """
        Exports the current page as a PDF.
        """
        pdf_data = await page.pdf(print_background=True)
        return pdf_data

    async def take_screenshot(self, page, **kwargs) -> str:
        page_height = await page.evaluate("document.documentElement.scrollHeight")
        if page_height < kwargs.get("screenshot_height_threshold", SCREENSHOT_HEIGHT_TRESHOLD):
            # Page is short enough, just take a screenshot
            return await self.take_screenshot_naive(page)
        else:
            # Page is too long, try to take a full-page screenshot
            return await self.take_screenshot_scroller(page, **kwargs)
            # return await self.take_screenshot_from_pdf(await self.export_pdf(page))     

    async def take_screenshot_from_pdf(self, pdf_data: bytes) -> str:
        """
        Convert the first page of the PDF to a screenshot.
        Requires pdf2image and poppler.
        """
        try:
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(pdf_data)
            final_img = images[0].convert('RGB')
            buffered = BytesIO()
            final_img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            error_message = f"Failed to take PDF-based screenshot: {str(e)}"
            self.logger.error(
                message="PDF Screenshot failed: {error}",
                tag="ERROR",
                params={"error": error_message}
            )
            # Return error image as fallback
            img = Image.new('RGB', (800, 600), color='black')
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            draw.text((10, 10), error_message, fill=(255, 255, 255), font=font)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

    async def take_screenshot_scroller(self, page: Page, **kwargs) -> str:
        """
        Attempt to set a large viewport and take a full-page screenshot.
        If still too large, segment the page as before.
        """
        try:
            # Get page height
            page_height = await page.evaluate("document.documentElement.scrollHeight")
            page_width = await page.evaluate("document.documentElement.scrollWidth")

            # Set a large viewport
            large_viewport_height = min(page_height, kwargs.get("screenshot_height_threshold", SCREENSHOT_HEIGHT_TRESHOLD))
            await page.set_viewport_size({"width": page_width, "height": large_viewport_height})
            
            # Page still too long, segment approach
            segments = []
            viewport_size = page.viewport_size
            viewport_height = viewport_size["height"]

            num_segments = (page_height // viewport_height) + 1
            for i in range(num_segments):
                y_offset = i * viewport_height
                await page.evaluate(f"window.scrollTo(0, {y_offset})")
                await asyncio.sleep(0.01)  # wait for render
                seg_shot = await page.screenshot(full_page=False)
                img = Image.open(BytesIO(seg_shot)).convert('RGB')
                segments.append(img)

            total_height = sum(img.height for img in segments)
            stitched = Image.new('RGB', (segments[0].width, total_height))
            offset = 0
            for img in segments:
                # stitched.paste(img, (0, offset))
                stitched.paste(img.convert('RGB'), (0, offset))
                offset += img.height

            buffered = BytesIO()
            stitched = stitched.convert('RGB')
            stitched.save(buffered, format="BMP", quality=85)
            encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return encoded
        except Exception as e:
            error_message = f"Failed to take large viewport screenshot: {str(e)}"
            self.logger.error(
                message="Large viewport screenshot failed: {error}",
                tag="ERROR",
                params={"error": error_message}
            )
            # return error image
            img = Image.new('RGB', (800, 600), color='black')
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            draw.text((10, 10), error_message, fill=(255, 255, 255), font=font)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        finally:
            await page.close()
    
    async def take_screenshot_naive(self, page: Page) -> str:
        """
        Takes a screenshot of the current page.
        
        Args:
            page (Page): The Playwright page instance
            
        Returns:
            str: Base64-encoded screenshot image
        """
        try:
            # The page is already loaded, just take the screenshot
            screenshot = await page.screenshot(full_page=False)
            return base64.b64encode(screenshot).decode('utf-8')
        except Exception as e:
            error_message = f"Failed to take screenshot: {str(e)}"
            self.logger.error(
                message="Screenshot failed: {error}",
                tag="ERROR",
                params={"error": error_message}
            )
            

            # Generate an error image
            img = Image.new('RGB', (800, 600), color='black')
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            draw.text((10, 10), error_message, fill=(255, 255, 255), font=font)
            
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        finally:
            await page.close()
     
    async def export_storage_state(self, path: str = None) -> dict:
        """
        Exports the current storage state (cookies, localStorage, sessionStorage)
        to a JSON file at the specified path.
        """
        if self.default_context:
            state = await self.default_context.storage_state(path=path)
            self.logger.info(
                message="Exported storage state to {path}",
                tag="INFO",
                params={"path": path}
            )
            return state
        else:
            self.logger.warning(
                message="No default_context available to export storage state.",
                tag="WARNING"
            )
            
    async def _generate_screenshot_from_html(self, html: str) -> Optional[str]:
        """
        Generates a screenshot from raw HTML content.

        Args:
            html (str): The HTML content to render and capture.

        Returns:
            Optional[str]: Base64-encoded screenshot image or an error image if failed.
        """
        try:
            await self.start()
            # Create a temporary page without a session_id
            page, context = await self.browser_manager.get_page(None, self.user_agent)
            
            await page.set_content(html, wait_until='networkidle')
            screenshot = await page.screenshot(full_page=True)
            await page.close()
            return base64.b64encode(screenshot).decode('utf-8')
        except Exception as e:
            error_message = f"Failed to take screenshot: {str(e)}"
            self.logger.error(
                message="Screenshot failed: {error}",
                tag="ERROR",
                params={"error": error_message}
            )            

            # Generate an error image
            img = Image.new('RGB', (800, 600), color='black')
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            draw.text((10, 10), error_message, fill=(255, 255, 255), font=font)

            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')


```

## crawl4ai/migrations.py
```
import os
import asyncio
import logging
from pathlib import Path
import aiosqlite
from typing import Optional
import xxhash
import aiofiles
import shutil
import time
from datetime import datetime
from .async_logger import AsyncLogger, LogLevel

# Initialize logger
logger = AsyncLogger(log_level=LogLevel.DEBUG, verbose=True)

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

class DatabaseMigration:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.content_paths = self._ensure_content_dirs(os.path.dirname(db_path))
        
    def _ensure_content_dirs(self, base_path: str) -> dict:
        dirs = {
            'html': 'html_content',
            'cleaned': 'cleaned_html',
            'markdown': 'markdown_content', 
            'extracted': 'extracted_content',
            'screenshots': 'screenshots'
        }
        content_paths = {}
        for key, dirname in dirs.items():
            path = os.path.join(base_path, dirname)
            os.makedirs(path, exist_ok=True)
            content_paths[key] = path
        return content_paths

    def _generate_content_hash(self, content: str) -> str:
        x = xxhash.xxh64()
        x.update(content.encode())
        content_hash = x.hexdigest()
        return content_hash
        # return hashlib.sha256(content.encode()).hexdigest()

    async def _store_content(self, content: str, content_type: str) -> str:
        if not content:
            return ""
        
        content_hash = self._generate_content_hash(content)
        file_path = os.path.join(self.content_paths[content_type], content_hash)
        
        if not os.path.exists(file_path):
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
                
        return content_hash

    async def migrate_database(self):
        """Migrate existing database to file-based storage"""
        # logger.info("Starting database migration...")
        logger.info("Starting database migration...", tag="INIT")
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get all rows
                async with db.execute(
                    '''SELECT url, html, cleaned_html, markdown, 
                       extracted_content, screenshot FROM crawled_data'''
                ) as cursor:
                    rows = await cursor.fetchall()

                migrated_count = 0
                for row in rows:
                    url, html, cleaned_html, markdown, extracted_content, screenshot = row
                    
                    # Store content in files and get hashes
                    html_hash = await self._store_content(html, 'html')
                    cleaned_hash = await self._store_content(cleaned_html, 'cleaned')
                    markdown_hash = await self._store_content(markdown, 'markdown')
                    extracted_hash = await self._store_content(extracted_content, 'extracted')
                    screenshot_hash = await self._store_content(screenshot, 'screenshots')

                    # Update database with hashes
                    await db.execute('''
                        UPDATE crawled_data 
                        SET html = ?, 
                            cleaned_html = ?,
                            markdown = ?,
                            extracted_content = ?,
                            screenshot = ?
                        WHERE url = ?
                    ''', (html_hash, cleaned_hash, markdown_hash, 
                         extracted_hash, screenshot_hash, url))
                    
                    migrated_count += 1
                    if migrated_count % 100 == 0:
                        logger.info(f"Migrated {migrated_count} records...", tag="INIT")
                        

                await db.commit()
                logger.success(f"Migration completed. {migrated_count} records processed.", tag="COMPLETE")

        except Exception as e:
            # logger.error(f"Migration failed: {e}")
            logger.error(
                message="Migration failed: {error}",
                tag="ERROR",
                params={"error": str(e)}
            )
            raise e

async def backup_database(db_path: str) -> str:
    """Create backup of existing database"""
    if not os.path.exists(db_path):
        logger.info("No existing database found. Skipping backup.", tag="INIT")
        return None
        
    # Create backup with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{db_path}.backup_{timestamp}"
    
    try:
        # Wait for any potential write operations to finish
        await asyncio.sleep(1)
        
        # Create backup
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backup created at: {backup_path}", tag="COMPLETE")
        return backup_path
    except Exception as e:
        # logger.error(f"Backup failed: {e}")
        logger.error(
                message="Migration failed: {error}",
                tag="ERROR",
                params={"error": str(e)}
            )
        raise e
    
async def run_migration(db_path: Optional[str] = None):
    """Run database migration"""
    if db_path is None:
        db_path = os.path.join(Path.home(), ".crawl4ai", "crawl4ai.db")
    
    if not os.path.exists(db_path):
        logger.info("No existing database found. Skipping migration.", tag="INIT")
        return
        
    # Create backup first
    backup_path = await backup_database(db_path)
    if not backup_path:
        return
    
    migration = DatabaseMigration(db_path)
    await migration.migrate_database()
    
def main():
    """CLI entry point for migration"""
    import argparse
    parser = argparse.ArgumentParser(description='Migrate Crawl4AI database to file-based storage')
    parser.add_argument('--db-path', help='Custom database path')
    args = parser.parse_args()
    
    asyncio.run(run_migration(args.db_path))

if __name__ == "__main__":
    main()
```

## crawl4ai/async_configs.py
```
from .config import (
    MIN_WORD_THRESHOLD, 
    IMAGE_DESCRIPTION_MIN_WORD_THRESHOLD,
    SCREENSHOT_HEIGHT_TRESHOLD,
    PAGE_TIMEOUT
)
from .user_agent_generator import UserAgentGenerator
from .extraction_strategy import ExtractionStrategy
from .chunking_strategy import ChunkingStrategy
from .markdown_generation_strategy import MarkdownGenerationStrategy

class BrowserConfig:
    """
    Configuration class for setting up a browser instance and its context in AsyncPlaywrightCrawlerStrategy.

    This class centralizes all parameters that affect browser and context creation. Instead of passing
    scattered keyword arguments, users can instantiate and modify this configuration object. The crawler
    code will then reference these settings to initialize the browser in a consistent, documented manner.

    Attributes:
        browser_type (str): The type of browser to launch. Supported values: "chromium", "firefox", "webkit".
                            Default: "chromium".
        headless (bool): Whether to run the browser in headless mode (no visible GUI).
                         Default: True.
        use_managed_browser (bool): Launch the browser using a managed approach (e.g., via CDP), allowing
                                    advanced manipulation. Default: False.
        use_persistent_context (bool): Use a persistent browser context (like a persistent profile).
                                       Automatically sets use_managed_browser=True. Default: False.
        user_data_dir (str or None): Path to a user data directory for persistent sessions. If None, a
                                     temporary directory may be used. Default: None.
        chrome_channel (str): The Chrome channel to launch (e.g., "chrome", "msedge"). Only applies if browser_type
                              is "chromium". Default: "chrome".
        proxy (str or None): Proxy server URL (e.g., "http://username:password@proxy:port"). If None, no proxy is used.
                             Default: None.
        proxy_config (dict or None): Detailed proxy configuration, e.g. {"server": "...", "username": "..."}.
                                     If None, no additional proxy config. Default: None.
        viewport_width (int): Default viewport width for pages. Default: 1920.
        viewport_height (int): Default viewport height for pages. Default: 1080.
        verbose (bool): Enable verbose logging.
                        Default: True.
        accept_downloads (bool): Whether to allow file downloads. If True, requires a downloads_path.
                                 Default: False.
        downloads_path (str or None): Directory to store downloaded files. If None and accept_downloads is True,
                                      a default path will be created. Default: None.
        storage_state (str or dict or None): Path or object describing storage state (cookies, localStorage).
                                             Default: None.
        ignore_https_errors (bool): Ignore HTTPS certificate errors. Default: True.
        java_script_enabled (bool): Enable JavaScript execution in pages. Default: True.
        cookies (list): List of cookies to add to the browser context. Each cookie is a dict with fields like
                        {"name": "...", "value": "...", "url": "..."}.
                        Default: [].
        headers (dict): Extra HTTP headers to apply to all requests in this context.
                        Default: {}.
        user_agent (str): Custom User-Agent string to use. Default: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36".
        user_agent_mode (str or None): Mode for generating the user agent (e.g., "random"). If None, use the provided
                                       user_agent as-is. Default: None.
        user_agent_generator_config (dict or None): Configuration for user agent generation if user_agent_mode is set.
                                                    Default: None.
        text_only (bool): If True, disables images and other rich content for potentially faster load times.
                          Default: False.
        light_mode (bool): Disables certain background features for performance gains. Default: False.
        extra_args (list): Additional command-line arguments passed to the browser.
                           Default: [].
    """

    def __init__(
        self,
        browser_type: str = "chromium",
        headless: bool = True,
        use_managed_browser: bool = False,
        use_persistent_context: bool = False,
        user_data_dir: str = None,
        chrome_channel: str = "chrome",
        proxy: str = None,
        proxy_config: dict = None,
        viewport_width: int = 1920,
        viewport_height: int = 1080,
        accept_downloads: bool = False,
        downloads_path: str = None,
        storage_state=None,
        ignore_https_errors: bool = True,
        java_script_enabled: bool = True,
        sleep_on_close: bool = False,
        verbose: bool = True,
        cookies: list = None,
        headers: dict = None,
        user_agent: str = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/116.0.5845.187 Safari/604.1 Edg/117.0.2045.47"
        ),
        user_agent_mode: str = None,
        user_agent_generator_config: dict = None,
        text_only: bool = False,
        light_mode: bool = False,
        extra_args: list = None,
    ):
        self.browser_type = browser_type
        self.headless = headless
        self.use_managed_browser = use_managed_browser
        self.use_persistent_context = use_persistent_context
        self.user_data_dir = user_data_dir
        if self.browser_type == "chromium":
            self.chrome_channel = "chrome"
        elif self.browser_type == "firefox":
            self.chrome_channel = "firefox"
        elif self.browser_type == "webkit":
            self.chrome_channel = "webkit"
        else:
            self.chrome_channel = chrome_channel or "chrome"
        self.proxy = proxy
        self.proxy_config = proxy_config
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.accept_downloads = accept_downloads
        self.downloads_path = downloads_path
        self.storage_state = storage_state
        self.ignore_https_errors = ignore_https_errors
        self.java_script_enabled = java_script_enabled
        self.cookies = cookies if cookies is not None else []
        self.headers = headers if headers is not None else {}
        self.user_agent = user_agent
        self.user_agent_mode = user_agent_mode
        self.user_agent_generator_config = user_agent_generator_config
        self.text_only = text_only
        self.light_mode = light_mode
        self.extra_args = extra_args if extra_args is not None else []
        self.sleep_on_close = sleep_on_close
        self.verbose = verbose
        
        user_agenr_generator = UserAgentGenerator()
        if self.user_agent_mode != "random":
            self.user_agent = user_agenr_generator.generate(
                **(self.user_agent_generator_config or {})
            )
        self.browser_hint = user_agenr_generator.generate_client_hints(self.user_agent)
        self.headers.setdefault("sec-ch-ua", self.browser_hint)

        # If persistent context is requested, ensure managed browser is enabled
        if self.use_persistent_context:
            self.use_managed_browser = True

    @staticmethod
    def from_kwargs(kwargs: dict) -> "BrowserConfig":
        return BrowserConfig(
            browser_type=kwargs.get("browser_type", "chromium"),
            headless=kwargs.get("headless", True),
            use_managed_browser=kwargs.get("use_managed_browser", False),
            use_persistent_context=kwargs.get("use_persistent_context", False),
            user_data_dir=kwargs.get("user_data_dir"),
            chrome_channel=kwargs.get("chrome_channel", "chrome"),
            proxy=kwargs.get("proxy"),
            proxy_config=kwargs.get("proxy_config"),
            viewport_width=kwargs.get("viewport_width", 1920),
            viewport_height=kwargs.get("viewport_height", 1080),
            accept_downloads=kwargs.get("accept_downloads", False),
            downloads_path=kwargs.get("downloads_path"),
            storage_state=kwargs.get("storage_state"),
            ignore_https_errors=kwargs.get("ignore_https_errors", True),
            java_script_enabled=kwargs.get("java_script_enabled", True),
            cookies=kwargs.get("cookies", []),
            headers=kwargs.get("headers", {}),
            user_agent=kwargs.get("user_agent",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
            ),
            user_agent_mode=kwargs.get("user_agent_mode"),
            user_agent_generator_config=kwargs.get("user_agent_generator_config"),
            text_only=kwargs.get("text_only", False),
            light_mode=kwargs.get("light_mode", False),
            extra_args=kwargs.get("extra_args", [])
        )


class CrawlerRunConfig:
    """
    Configuration class for controlling how the crawler runs each crawl operation.
    This includes parameters for content extraction, page manipulation, waiting conditions,
    caching, and other runtime behaviors.

    This centralizes parameters that were previously scattered as kwargs to `arun()` and related methods.
    By using this class, you have a single place to understand and adjust the crawling options.

    Attributes:
        word_count_threshold (int): Minimum word count threshold before processing content.
                                    Default: MIN_WORD_THRESHOLD (typically 200).
        extraction_strategy (ExtractionStrategy or None): Strategy to extract structured data from crawled pages.
                                                          Default: None (NoExtractionStrategy is used if None).
        chunking_strategy (ChunkingStrategy): Strategy to chunk content before extraction.
                                              Default: RegexChunking().
        content_filter (RelevantContentFilter or None): Optional filter to prune irrelevant content.
                                                        Default: None.
        cache_mode (CacheMode or None): Defines how caching is handled.
                                        If None, defaults to CacheMode.ENABLED internally.
                                        Default: None.
        session_id (str or None):   Optional session ID to persist the browser context and the created 
                                    page instance. If the ID already exists, the crawler does not 
                                    create a new page and uses the current page to preserve the state;
                                    if not, it creates a new page and context then stores it in 
                                    memory with the given session ID.
        bypass_cache (bool): Legacy parameter, if True acts like CacheMode.BYPASS.
                             Default: False.
        disable_cache (bool): Legacy parameter, if True acts like CacheMode.DISABLED.
                              Default: False.
        no_cache_read (bool): Legacy parameter, if True acts like CacheMode.WRITE_ONLY.
                              Default: False.
        no_cache_write (bool): Legacy parameter, if True acts like CacheMode.READ_ONLY.
                               Default: False.
        css_selector (str or None): CSS selector to extract a specific portion of the page.
                                    Default: None.
        screenshot (bool): Whether to take a screenshot after crawling.
                           Default: False.
        pdf (bool): Whether to generate a PDF of the page.
                    Default: False.
        verbose (bool): Enable verbose logging.
                        Default: True.
        only_text (bool): If True, attempt to extract text-only content where applicable.
                          Default: False.
        image_description_min_word_threshold (int): Minimum words for image description extraction.
                                                    Default: IMAGE_DESCRIPTION_MIN_WORD_THRESHOLD (e.g., 50).
        prettiify (bool): If True, apply `fast_format_html` to produce prettified HTML output.
                          Default: False.
        js_code (str or list of str or None): JavaScript code/snippets to run on the page.
                                              Default: None.
        wait_for (str or None): A CSS selector or JS condition to wait for before extracting content.
                                Default: None.
        js_only (bool): If True, indicates subsequent calls are JS-driven updates, not full page loads.
                        Default: False.
        wait_until (str): The condition to wait for when navigating, e.g. "domcontentloaded".
                          Default: "domcontentloaded".
        page_timeout (int): Timeout in ms for page operations like navigation.
                            Default: 60000 (60 seconds).
        ignore_body_visibility (bool): If True, ignore whether the body is visible before proceeding.
                                       Default: True.
        wait_for_images (bool): If True, wait for images to load before extracting content. 
                                Default: True.
        adjust_viewport_to_content (bool): If True, adjust viewport according to the page content dimensions.
                                           Default: False.
        scan_full_page (bool): If True, scroll through the entire page to load all content.
                               Default: False.
        scroll_delay (float): Delay in seconds between scroll steps if scan_full_page is True.
                              Default: 0.2.
        process_iframes (bool): If True, attempts to process and inline iframe content.
                                Default: False.
        remove_overlay_elements (bool): If True, remove overlays/popups before extracting HTML.
                                        Default: False.
        delay_before_return_html (float): Delay in seconds before retrieving final HTML.
                                          Default: 0.1.
        log_console (bool): If True, log console messages from the page.
                            Default: False.
        simulate_user (bool): If True, simulate user interactions (mouse moves, clicks) for anti-bot measures.
                              Default: False.
        override_navigator (bool): If True, overrides navigator properties for more human-like behavior.
                                   Default: False.
        magic (bool): If True, attempts automatic handling of overlays/popups.
                      Default: False.
        screenshot_wait_for (float or None): Additional wait time before taking a screenshot.
                                             Default: None.
        screenshot_height_threshold (int): Threshold for page height to decide screenshot strategy.
                                           Default: SCREENSHOT_HEIGHT_TRESHOLD (from config, e.g. 20000).
        mean_delay (float): Mean base delay between requests when calling arun_many.
                            Default: 0.1.
        max_range (float): Max random additional delay range for requests in arun_many.
                           Default: 0.3.
        # session_id and semaphore_count might be set at runtime, not needed as defaults here.
    """

    def __init__(
        self,
        word_count_threshold: int =  MIN_WORD_THRESHOLD ,
        extraction_strategy : ExtractionStrategy=None,  # Will default to NoExtractionStrategy if None
        chunking_strategy : ChunkingStrategy= None,    # Will default to RegexChunking if None
        markdown_generator : MarkdownGenerationStrategy = None,
        content_filter=None,
        cache_mode=None,
        session_id: str = None,
        bypass_cache: bool = False,
        disable_cache: bool = False,
        no_cache_read: bool = False,
        no_cache_write: bool = False,
        css_selector: str = None,
        screenshot: bool = False,
        pdf: bool = False,
        verbose: bool = True,
        only_text: bool = False,
        image_description_min_word_threshold: int = IMAGE_DESCRIPTION_MIN_WORD_THRESHOLD,
        prettiify: bool = False,
        js_code=None,
        wait_for: str = None,
        js_only: bool = False,
        wait_until: str = "domcontentloaded",
        page_timeout: int = PAGE_TIMEOUT,
        ignore_body_visibility: bool = True,
        wait_for_images: bool = True,
        adjust_viewport_to_content: bool = False,
        scan_full_page: bool = False,
        scroll_delay: float = 0.2,
        process_iframes: bool = False,
        remove_overlay_elements: bool = False,
        delay_before_return_html: float = 0.1,
        log_console: bool = False,
        simulate_user: bool = False,
        override_navigator: bool = False,
        magic: bool = False,
        screenshot_wait_for: float = None,
        screenshot_height_threshold: int = SCREENSHOT_HEIGHT_TRESHOLD,
        mean_delay: float = 0.1,
        max_range: float = 0.3,
        semaphore_count: int = 5,
    ):
        self.word_count_threshold = word_count_threshold
        self.extraction_strategy = extraction_strategy
        self.chunking_strategy = chunking_strategy
        self.markdown_generator = markdown_generator
        self.content_filter = content_filter
        self.cache_mode = cache_mode
        self.session_id = session_id
        self.bypass_cache = bypass_cache
        self.disable_cache = disable_cache
        self.no_cache_read = no_cache_read
        self.no_cache_write = no_cache_write
        self.css_selector = css_selector
        self.screenshot = screenshot
        self.pdf = pdf
        self.verbose = verbose
        self.only_text = only_text
        self.image_description_min_word_threshold = image_description_min_word_threshold
        self.prettiify = prettiify
        self.js_code = js_code
        self.wait_for = wait_for
        self.js_only = js_only
        self.wait_until = wait_until
        self.page_timeout = page_timeout
        self.ignore_body_visibility = ignore_body_visibility
        self.wait_for_images = wait_for_images
        self.adjust_viewport_to_content = adjust_viewport_to_content
        self.scan_full_page = scan_full_page
        self.scroll_delay = scroll_delay
        self.process_iframes = process_iframes
        self.remove_overlay_elements = remove_overlay_elements
        self.delay_before_return_html = delay_before_return_html
        self.log_console = log_console
        self.simulate_user = simulate_user
        self.override_navigator = override_navigator
        self.magic = magic
        self.screenshot_wait_for = screenshot_wait_for
        self.screenshot_height_threshold = screenshot_height_threshold
        self.mean_delay = mean_delay
        self.max_range = max_range
        self.semaphore_count = semaphore_count

        # Validate type of extraction strategy and chunking strategy if they are provided
        if self.extraction_strategy is not None and not isinstance(self.extraction_strategy, ExtractionStrategy):
            raise ValueError("extraction_strategy must be an instance of ExtractionStrategy")
        if self.chunking_strategy is not None and not isinstance(self.chunking_strategy, ChunkingStrategy):
            raise ValueError("chunking_strategy must be an instance of ChunkingStrategy")

        # Set default chunking strategy if None
        if self.chunking_strategy is None:
            from .chunking_strategy import RegexChunking
            self.chunking_strategy = RegexChunking()
        

    @staticmethod
    def from_kwargs(kwargs: dict) -> "CrawlerRunConfig":
        return CrawlerRunConfig(
            word_count_threshold=kwargs.get("word_count_threshold", 200),
            extraction_strategy=kwargs.get("extraction_strategy"),
            chunking_strategy=kwargs.get("chunking_strategy"),
            markdown_generator=kwargs.get("markdown_generator"),
            content_filter=kwargs.get("content_filter"),
            cache_mode=kwargs.get("cache_mode"),
            session_id=kwargs.get("session_id"),
            bypass_cache=kwargs.get("bypass_cache", False),
            disable_cache=kwargs.get("disable_cache", False),
            no_cache_read=kwargs.get("no_cache_read", False),
            no_cache_write=kwargs.get("no_cache_write", False),
            css_selector=kwargs.get("css_selector"),
            screenshot=kwargs.get("screenshot", False),
            pdf=kwargs.get("pdf", False),
            verbose=kwargs.get("verbose", True),
            only_text=kwargs.get("only_text", False),
            image_description_min_word_threshold=kwargs.get("image_description_min_word_threshold",  IMAGE_DESCRIPTION_MIN_WORD_THRESHOLD),
            prettiify=kwargs.get("prettiify", False),
            js_code=kwargs.get("js_code"), # If not provided here, will default inside constructor
            wait_for=kwargs.get("wait_for"),
            js_only=kwargs.get("js_only", False),
            wait_until=kwargs.get("wait_until", "domcontentloaded"),
            page_timeout=kwargs.get("page_timeout", 60000),
            ignore_body_visibility=kwargs.get("ignore_body_visibility", True),
            adjust_viewport_to_content=kwargs.get("adjust_viewport_to_content", False),
            scan_full_page=kwargs.get("scan_full_page", False),
            scroll_delay=kwargs.get("scroll_delay", 0.2),
            process_iframes=kwargs.get("process_iframes", False),
            remove_overlay_elements=kwargs.get("remove_overlay_elements", False),
            delay_before_return_html=kwargs.get("delay_before_return_html", 0.1),
            log_console=kwargs.get("log_console", False),
            simulate_user=kwargs.get("simulate_user", False),
            override_navigator=kwargs.get("override_navigator", False),
            magic=kwargs.get("magic", False),
            screenshot_wait_for=kwargs.get("screenshot_wait_for"),
            screenshot_height_threshold=kwargs.get("screenshot_height_threshold", 20000),
            mean_delay=kwargs.get("mean_delay", 0.1),
            max_range=kwargs.get("max_range", 0.3),
            semaphore_count=kwargs.get("semaphore_count", 5)
        )

```

## crawl4ai/async_tools.py
```
import asyncio
import base64
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, List, Optional, Awaitable
import os, sys, shutil
import tempfile, subprocess
from playwright.async_api import async_playwright, Page, Browser, Error
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from playwright.async_api import ProxySettings
from pydantic import BaseModel
import hashlib
import json
import uuid
from .models import AsyncCrawlResponse
from .utils import create_box_message
from .user_agent_generator import UserAgentGenerator
from playwright_stealth import StealthConfig, stealth_async


class ManagedBrowser:
    def __init__(self, browser_type: str = "chromium", user_data_dir: Optional[str] = None, headless: bool = False, logger = None, host: str = "localhost", debugging_port: int = 9222):
        self.browser_type = browser_type
        self.user_data_dir = user_data_dir
        self.headless = headless
        self.browser_process = None
        self.temp_dir = None
        self.debugging_port = debugging_port
        self.host = host
        self.logger = logger
        self.shutting_down = False

    async def start(self) -> str:
        """
        Starts the browser process and returns the CDP endpoint URL.
        If user_data_dir is not provided, creates a temporary directory.
        """
        
        # Create temp dir if needed
        if not self.user_data_dir:
            self.temp_dir = tempfile.mkdtemp(prefix="browser-profile-")
            self.user_data_dir = self.temp_dir

        # Get browser path and args based on OS and browser type
        browser_path = self._get_browser_path()
        args = self._get_browser_args()

        # Start browser process
        try:
            self.browser_process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            # Monitor browser process output for errors
            asyncio.create_task(self._monitor_browser_process())
            await asyncio.sleep(2)  # Give browser time to start
            return f"http://{self.host}:{self.debugging_port}"
        except Exception as e:
            await self.cleanup()
            raise Exception(f"Failed to start browser: {e}")

    async def _monitor_browser_process(self):
        """Monitor the browser process for unexpected termination."""
        if self.browser_process:
            try:
                stdout, stderr = await asyncio.gather(
                    asyncio.to_thread(self.browser_process.stdout.read),
                    asyncio.to_thread(self.browser_process.stderr.read)
                )
                
                # Check shutting_down flag BEFORE logging anything
                if self.browser_process.poll() is not None:
                    if not self.shutting_down:
                        self.logger.error(
                            message="Browser process terminated unexpectedly | Code: {code} | STDOUT: {stdout} | STDERR: {stderr}",
                            tag="ERROR",
                            params={
                                "code": self.browser_process.returncode,
                                "stdout": stdout.decode(),
                                "stderr": stderr.decode()
                            }
                        )                
                        await self.cleanup()
                    else:
                        self.logger.info(
                            message="Browser process terminated normally | Code: {code}",
                            tag="INFO",
                            params={"code": self.browser_process.returncode}
                        )
            except Exception as e:
                if not self.shutting_down:
                    self.logger.error(
                        message="Error monitoring browser process: {error}",
                        tag="ERROR",
                        params={"error": str(e)}
                    )

    def _get_browser_path(self) -> str:
        """Returns the browser executable path based on OS and browser type"""
        if sys.platform == "darwin":  # macOS
            paths = {
                "chromium": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "firefox": "/Applications/Firefox.app/Contents/MacOS/firefox",
                "webkit": "/Applications/Safari.app/Contents/MacOS/Safari"
            }
        elif sys.platform == "win32":  # Windows
            paths = {
                "chromium": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                "firefox": "C:\\Program Files\\Mozilla Firefox\\firefox.exe",
                "webkit": None  # WebKit not supported on Windows
            }
        else:  # Linux
            paths = {
                "chromium": "google-chrome",
                "firefox": "firefox",
                "webkit": None  # WebKit not supported on Linux
            }
        
        return paths.get(self.browser_type)

    def _get_browser_args(self) -> List[str]:
        """Returns browser-specific command line arguments"""
        base_args = [self._get_browser_path()]
        
        if self.browser_type == "chromium":
            args = [
                f"--remote-debugging-port={self.debugging_port}",
                f"--user-data-dir={self.user_data_dir}",
            ]
            if self.headless:
                args.append("--headless=new")
        elif self.browser_type == "firefox":
            args = [
                "--remote-debugging-port", str(self.debugging_port),
                "--profile", self.user_data_dir,
            ]
            if self.headless:
                args.append("--headless")
        else:
            raise NotImplementedError(f"Browser type {self.browser_type} not supported")
            
        return base_args + args

    async def cleanup(self):
        """Cleanup browser process and temporary directory"""
        # Set shutting_down flag BEFORE any termination actions
        self.shutting_down = True
        
        if self.browser_process:
            try:
                self.browser_process.terminate()
                # Wait for process to end gracefully
                for _ in range(10):  # 10 attempts, 100ms each
                    if self.browser_process.poll() is not None:
                        break
                    await asyncio.sleep(0.1)
                
                # Force kill if still running
                if self.browser_process.poll() is None:
                    self.browser_process.kill()
                    await asyncio.sleep(0.1)  # Brief wait for kill to take effect
                    
            except Exception as e:
                self.logger.error(
                    message="Error terminating browser: {error}",
                    tag="ERROR",
                    params={"error": str(e)}
                )

        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                self.logger.error(
                    message="Error removing temporary directory: {error}",
                    tag="ERROR",
                    params={"error": str(e)}
                )


```

## crawl4ai/web_crawler.py
```
import os, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

from .models import UrlModel, CrawlResult
from .database import init_db, get_cached_url, cache_url, DB_PATH, flush_db
from .utils import *
from .chunking_strategy import *
from .extraction_strategy import *
from .crawler_strategy import *
from typing import List
from concurrent.futures import ThreadPoolExecutor
from .content_scraping_strategy import WebScrapingStrategy
from .config import *
import warnings
import json
warnings.filterwarnings("ignore", message='Field "model_name" has conflict with protected namespace "model_".')


class WebCrawler:
    def __init__(self, crawler_strategy: CrawlerStrategy = None, always_by_pass_cache: bool = False, verbose: bool = False):
        self.crawler_strategy = crawler_strategy or LocalSeleniumCrawlerStrategy(verbose=verbose)
        self.always_by_pass_cache = always_by_pass_cache
        self.crawl4ai_folder = os.path.join(os.getenv("CRAWL4_AI_BASE_DIRECTORY", Path.home()), ".crawl4ai")
        os.makedirs(self.crawl4ai_folder, exist_ok=True)
        os.makedirs(f"{self.crawl4ai_folder}/cache", exist_ok=True)
        init_db()
        self.ready = False
        
    def warmup(self):
        print("[LOG] 🌤️  Warming up the WebCrawler")
        self.run(
            url='https://google.com/',
            word_count_threshold=5,
            extraction_strategy=NoExtractionStrategy(),
            bypass_cache=False,
            verbose=False
        )
        self.ready = True
        print("[LOG] 🌞 WebCrawler is ready to crawl")
        
    def fetch_page(
        self,
        url_model: UrlModel,
        provider: str = DEFAULT_PROVIDER,
        api_token: str = None,
        extract_blocks_flag: bool = True,
        word_count_threshold=MIN_WORD_THRESHOLD,
        css_selector: str = None,
        screenshot: bool = False,
        use_cached_html: bool = False,
        extraction_strategy: ExtractionStrategy = None,
        chunking_strategy: ChunkingStrategy = RegexChunking(),
        **kwargs,
    ) -> CrawlResult:
        return self.run(
            url_model.url,
            word_count_threshold,
            extraction_strategy or NoExtractionStrategy(),
            chunking_strategy,
            bypass_cache=url_model.forced,
            css_selector=css_selector,
            screenshot=screenshot,
            **kwargs,
        )
        pass

    def fetch_pages(
        self,
        url_models: List[UrlModel],
        provider: str = DEFAULT_PROVIDER,
        api_token: str = None,
        extract_blocks_flag: bool = True,
        word_count_threshold=MIN_WORD_THRESHOLD,
        use_cached_html: bool = False,
        css_selector: str = None,
        screenshot: bool = False,
        extraction_strategy: ExtractionStrategy = None,
        chunking_strategy: ChunkingStrategy = RegexChunking(),
        **kwargs,
    ) -> List[CrawlResult]:
        extraction_strategy = extraction_strategy or NoExtractionStrategy()
        def fetch_page_wrapper(url_model, *args, **kwargs):
            return self.fetch_page(url_model, *args, **kwargs)

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(
                    fetch_page_wrapper,
                    url_models,
                    [provider] * len(url_models),
                    [api_token] * len(url_models),
                    [extract_blocks_flag] * len(url_models),
                    [word_count_threshold] * len(url_models),
                    [css_selector] * len(url_models),
                    [screenshot] * len(url_models),
                    [use_cached_html] * len(url_models),
                    [extraction_strategy] * len(url_models),
                    [chunking_strategy] * len(url_models),
                    *[kwargs] * len(url_models),
                )
            )

        return results

    def run(
            self,
            url: str,
            word_count_threshold=MIN_WORD_THRESHOLD,
            extraction_strategy: ExtractionStrategy = None,
            chunking_strategy: ChunkingStrategy = RegexChunking(),
            bypass_cache: bool = False,
            css_selector: str = None,
            screenshot: bool = False,
            user_agent: str = None,
            verbose=True,
            **kwargs,
        ) -> CrawlResult:
            try:
                extraction_strategy = extraction_strategy or NoExtractionStrategy()
                extraction_strategy.verbose = verbose
                if not isinstance(extraction_strategy, ExtractionStrategy):
                    raise ValueError("Unsupported extraction strategy")
                if not isinstance(chunking_strategy, ChunkingStrategy):
                    raise ValueError("Unsupported chunking strategy")
                
                word_count_threshold = max(word_count_threshold, MIN_WORD_THRESHOLD)

                cached = None
                screenshot_data = None
                extracted_content = None
                if not bypass_cache and not self.always_by_pass_cache:
                    cached = get_cached_url(url)
                
                if kwargs.get("warmup", True) and not self.ready:
                    return None
                
                if cached:
                    html = sanitize_input_encode(cached[1])
                    extracted_content = sanitize_input_encode(cached[4])
                    if screenshot:
                        screenshot_data = cached[9]
                        if not screenshot_data:
                            cached = None
                
                if not cached or not html:
                    if user_agent:
                        self.crawler_strategy.update_user_agent(user_agent)
                    t1 = time.time()
                    html = sanitize_input_encode(self.crawler_strategy.crawl(url, **kwargs))
                    t2 = time.time()
                    if verbose:
                        print(f"[LOG] 🚀 Crawling done for {url}, success: {bool(html)}, time taken: {t2 - t1:.2f} seconds")
                    if screenshot:
                        screenshot_data = self.crawler_strategy.take_screenshot()

                
                crawl_result = self.process_html(url, html, extracted_content, word_count_threshold, extraction_strategy, chunking_strategy, css_selector, screenshot_data, verbose, bool(cached), **kwargs)
                crawl_result.success = bool(html)
                return crawl_result
            except Exception as e:
                if not hasattr(e, "msg"):
                    e.msg = str(e)
                print(f"[ERROR] 🚫 Failed to crawl {url}, error: {e.msg}")    
                return CrawlResult(url=url, html="", success=False, error_message=e.msg)

    def process_html(
            self,
            url: str,
            html: str,
            extracted_content: str,
            word_count_threshold: int,
            extraction_strategy: ExtractionStrategy,
            chunking_strategy: ChunkingStrategy,
            css_selector: str,
            screenshot: bool,
            verbose: bool,
            is_cached: bool,
            **kwargs,
        ) -> CrawlResult:
            t = time.time()
            # Extract content from HTML
            try:
                t1 = time.time()
                scrapping_strategy = WebScrapingStrategy()
                extra_params = {k: v for k, v in kwargs.items() if k not in ["only_text", "image_description_min_word_threshold"]}
                result = scrapping_strategy.scrap(
                    url,
                    html,
                    word_count_threshold=word_count_threshold,
                    css_selector=css_selector,
                    only_text=kwargs.get("only_text", False),
                    image_description_min_word_threshold=kwargs.get(
                        "image_description_min_word_threshold", IMAGE_DESCRIPTION_MIN_WORD_THRESHOLD
                    ),
                    **extra_params,
                )
                
                # result = get_content_of_website_optimized(url, html, word_count_threshold, css_selector=css_selector, only_text=kwargs.get("only_text", False))
                if verbose:
                    print(f"[LOG] 🚀 Content extracted for {url}, success: True, time taken: {time.time() - t1:.2f} seconds")
                
                if result is None:
                    raise ValueError(f"Failed to extract content from the website: {url}")
            except InvalidCSSSelectorError as e:
                raise ValueError(str(e))
            
            cleaned_html = sanitize_input_encode(result.get("cleaned_html", ""))
            markdown = sanitize_input_encode(result.get("markdown", ""))
            media = result.get("media", [])
            links = result.get("links", [])
            metadata = result.get("metadata", {})
                        
            if extracted_content is None:
                if verbose:
                    print(f"[LOG] 🔥 Extracting semantic blocks for {url}, Strategy: {extraction_strategy.name}")

                sections = chunking_strategy.chunk(markdown)
                extracted_content = extraction_strategy.run(url, sections)
                extracted_content = json.dumps(extracted_content, indent=4, default=str, ensure_ascii=False)

                if verbose:
                    print(f"[LOG] 🚀 Extraction done for {url}, time taken: {time.time() - t:.2f} seconds.")
                
            screenshot = None if not screenshot else screenshot
            
            if not is_cached:
                cache_url(
                    url,
                    html,
                    cleaned_html,
                    markdown,
                    extracted_content,
                    True,
                    json.dumps(media),
                    json.dumps(links),
                    json.dumps(metadata),
                    screenshot=screenshot,
                )                
            
            return CrawlResult(
                url=url,
                html=html,
                cleaned_html=format_html(cleaned_html),
                markdown=markdown,
                media=media,
                links=links,
                metadata=metadata,
                screenshot=screenshot,
                extracted_content=extracted_content,
                success=True,
                error_message="",
            )
```

## crawl4ai/async_webcrawler.py
```
import os, sys
import time
import warnings
from enum import Enum
from colorama import init, Fore, Back, Style
from pathlib import Path
from typing import Optional, List, Union
import json
import asyncio
# from contextlib import nullcontext, asynccontextmanager
from contextlib import asynccontextmanager
from .models import CrawlResult, MarkdownGenerationResult
from .async_database import async_db_manager
from .chunking_strategy import *
from .content_filter_strategy import *
from .extraction_strategy import *
from .async_crawler_strategy import AsyncCrawlerStrategy, AsyncPlaywrightCrawlerStrategy, AsyncCrawlResponse
from .cache_context import CacheMode, CacheContext, _legacy_to_cache_mode
from .markdown_generation_strategy import DefaultMarkdownGenerator, MarkdownGenerationStrategy
from .content_scraping_strategy import WebScrapingStrategy
from .async_logger import AsyncLogger
from .async_configs import BrowserConfig, CrawlerRunConfig
from .config import (
    MIN_WORD_THRESHOLD, 
    IMAGE_DESCRIPTION_MIN_WORD_THRESHOLD,
    URL_LOG_SHORTEN_LENGTH
)
from .utils import (
    sanitize_input_encode,
    InvalidCSSSelectorError,
    format_html,
    fast_format_html,
    create_box_message
)

from urllib.parse import urlparse
import random
from .__version__ import __version__ as crawl4ai_version


class AsyncWebCrawler:
    """
    Asynchronous web crawler with flexible caching capabilities.
    
    Migration Guide:
    Old way (deprecated):
        crawler = AsyncWebCrawler(always_by_pass_cache=True, browser_type="chromium", headless=True)
    
    New way (recommended):
        browser_config = BrowserConfig(browser_type="chromium", headless=True)
        crawler = AsyncWebCrawler(browser_config=browser_config)
    """
    _domain_last_hit = {}

    def __init__(
        self,
        crawler_strategy: Optional[AsyncCrawlerStrategy] = None,
        config: Optional[BrowserConfig] = None,
        always_bypass_cache: bool = False,
        always_by_pass_cache: Optional[bool] = None,  # Deprecated parameter
        base_directory: str = str(os.getenv("CRAWL4_AI_BASE_DIRECTORY", Path.home())),
        thread_safe: bool = False,
        **kwargs,
    ):
        """
        Initialize the AsyncWebCrawler.

        Args:
            crawler_strategy: Strategy for crawling web pages. If None, will create AsyncPlaywrightCrawlerStrategy
            config: Configuration object for browser settings. If None, will be created from kwargs
            always_bypass_cache: Whether to always bypass cache (new parameter)
            always_by_pass_cache: Deprecated, use always_bypass_cache instead
            base_directory: Base directory for storing cache
            thread_safe: Whether to use thread-safe operations
            **kwargs: Additional arguments for backwards compatibility
        """  
        # Handle browser configuration
        browser_config = config
        if browser_config is not None:
            if any(k in kwargs for k in ["browser_type", "headless", "viewport_width", "viewport_height"]):
                self.logger.warning(
                    message="Both browser_config and legacy browser parameters provided. browser_config will take precedence.",
                    tag="WARNING"
                )
        else:
            # Create browser config from kwargs for backwards compatibility
            browser_config = BrowserConfig.from_kwargs(kwargs)

        self.browser_config = browser_config
        
        # Initialize logger first since other components may need it
        self.logger = AsyncLogger(
            log_file=os.path.join(base_directory, ".crawl4ai", "crawler.log"),
            verbose=self.browser_config.verbose,    
            tag_width=10
        )

        
        # Initialize crawler strategy
        self.crawler_strategy = crawler_strategy or AsyncPlaywrightCrawlerStrategy(
            browser_config=browser_config,
            logger=self.logger,
            **kwargs  # Pass remaining kwargs for backwards compatibility
        )
        
        # Handle deprecated cache parameter
        if always_by_pass_cache is not None:
            if kwargs.get("warning", True):
                warnings.warn(
                    "'always_by_pass_cache' is deprecated and will be removed in version 0.5.0. "
                    "Use 'always_bypass_cache' instead. "
                    "Pass warning=False to suppress this warning.",
                    DeprecationWarning,
                    stacklevel=2
                )
            self.always_bypass_cache = always_by_pass_cache
        else:
            self.always_bypass_cache = always_bypass_cache

        # Thread safety setup
        self._lock = asyncio.Lock() if thread_safe else None
        
        # Initialize directories
        self.crawl4ai_folder = os.path.join(base_directory, ".crawl4ai")
        os.makedirs(self.crawl4ai_folder, exist_ok=True)
        os.makedirs(f"{self.crawl4ai_folder}/cache", exist_ok=True)
        
        self.ready = False

    async def __aenter__(self):
        await self.crawler_strategy.__aenter__()
        await self.awarmup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.crawler_strategy.__aexit__(exc_type, exc_val, exc_tb)
    
    async def awarmup(self):
        """Initialize the crawler with warm-up sequence."""
        self.logger.info(f"Crawl4AI {crawl4ai_version}", tag="INIT")
        self.ready = True

    @asynccontextmanager
    async def nullcontext(self):
        """异步空上下文管理器"""
        yield
    
    async def arun(
            self,
            url: str,
            config: Optional[CrawlerRunConfig] = None,
            # Legacy parameters maintained for backwards compatibility
            word_count_threshold=MIN_WORD_THRESHOLD,
            extraction_strategy: ExtractionStrategy = None,
            chunking_strategy: ChunkingStrategy = RegexChunking(),
            content_filter: RelevantContentFilter = None,
            cache_mode: Optional[CacheMode] = None,
            # Deprecated cache parameters
            bypass_cache: bool = False,
            disable_cache: bool = False,
            no_cache_read: bool = False,
            no_cache_write: bool = False,
            # Other legacy parameters
            css_selector: str = None,
            screenshot: bool = False,
            pdf: bool = False,
            user_agent: str = None,
            verbose=True,
            **kwargs,
        ) -> CrawlResult:
            """
            Runs the crawler for a single source: URL (web, local file, or raw HTML).

            Migration Guide:
            Old way (deprecated):
                result = await crawler.arun(
                    url="https://example.com",
                    word_count_threshold=200,
                    screenshot=True,
                    ...
                )
            
            New way (recommended):
                config = CrawlerRunConfig(
                    word_count_threshold=200,
                    screenshot=True,
                    ...
                )
                result = await crawler.arun(url="https://example.com", crawler_config=config)

            Args:
                url: The URL to crawl (http://, https://, file://, or raw:)
                crawler_config: Configuration object controlling crawl behavior
                [other parameters maintained for backwards compatibility]
            
            Returns:
                CrawlResult: The result of crawling and processing
            """
            crawler_config = config
            if not isinstance(url, str) or not url:
                raise ValueError("Invalid URL, make sure the URL is a non-empty string")
            
            async with self._lock or self.nullcontext():
                try:
                    # Handle configuration
                    if crawler_config is not None:
                        if any(param is not None for param in [
                            word_count_threshold, extraction_strategy, chunking_strategy,
                            content_filter, cache_mode, css_selector, screenshot, pdf
                        ]):
                            self.logger.warning(
                                message="Both crawler_config and legacy parameters provided. crawler_config will take precedence.",
                                tag="WARNING"
                            )
                        config = crawler_config
                    else:
                        # Merge all parameters into a single kwargs dict for config creation
                        config_kwargs = {
                            "word_count_threshold": word_count_threshold,
                            "extraction_strategy": extraction_strategy,
                            "chunking_strategy": chunking_strategy,
                            "content_filter": content_filter,
                            "cache_mode": cache_mode,
                            "bypass_cache": bypass_cache,
                            "disable_cache": disable_cache,
                            "no_cache_read": no_cache_read,
                            "no_cache_write": no_cache_write,
                            "css_selector": css_selector,
                            "screenshot": screenshot,
                            "pdf": pdf,
                            "verbose": verbose,
                            **kwargs
                        }
                        config = CrawlerRunConfig.from_kwargs(config_kwargs)

                    # Handle deprecated cache parameters
                    if any([bypass_cache, disable_cache, no_cache_read, no_cache_write]):
                        if kwargs.get("warning", True):
                            warnings.warn(
                                "Cache control boolean flags are deprecated and will be removed in version 0.5.0. "
                                "Use 'cache_mode' parameter instead.",
                                DeprecationWarning,
                                stacklevel=2
                            )
                        
                        # Convert legacy parameters if cache_mode not provided
                        if config.cache_mode is None:
                            config.cache_mode = _legacy_to_cache_mode(
                                disable_cache=disable_cache,
                                bypass_cache=bypass_cache,
                                no_cache_read=no_cache_read,
                                no_cache_write=no_cache_write
                            )
                    
                    # Default to ENABLED if no cache mode specified
                    if config.cache_mode is None:
                        config.cache_mode = CacheMode.ENABLED

                    # Create cache context
                    cache_context = CacheContext(url, config.cache_mode, self.always_bypass_cache)

                    # Initialize processing variables
                    async_response: AsyncCrawlResponse = None
                    cached_result = None
                    screenshot_data = None
                    pdf_data = None
                    extracted_content = None
                    start_time = time.perf_counter()

                    # Try to get cached result if appropriate
                    if cache_context.should_read():
                        cached_result = await async_db_manager.aget_cached_url(url)

                    if cached_result:
                        html = sanitize_input_encode(cached_result.html)
                        extracted_content = sanitize_input_encode(cached_result.extracted_content or "")
                        # If screenshot is requested but its not in cache, then set cache_result to None
                        screenshot_data = cached_result.screenshot
                        pdf_data = cached_result.pdf
                        if config.screenshot and not screenshot or config.pdf and not pdf:
                            cached_result = None

                        self.logger.url_status(
                            url=cache_context.display_url,
                            success=bool(html),
                            timing=time.perf_counter() - start_time,
                            tag="FETCH"
                        )

                    # Fetch fresh content if needed
                    if not cached_result or not html:
                        t1 = time.perf_counter()
                        
                        if user_agent:
                            self.crawler_strategy.update_user_agent(user_agent)
                        
                        # Pass config to crawl method
                        async_response = await self.crawler_strategy.crawl(
                            url,
                            config=config  # Pass the entire config object
                        )
                        
                        html = sanitize_input_encode(async_response.html)
                        screenshot_data = async_response.screenshot
                        pdf_data = async_response.pdf_data
                        
                        t2 = time.perf_counter()
                        self.logger.url_status(
                            url=cache_context.display_url,
                            success=bool(html),
                            timing=t2 - t1,
                            tag="FETCH"
                        )

                    # Process the HTML content
                    crawl_result = await self.aprocess_html(
                        url=url,
                        html=html,
                        extracted_content=extracted_content,
                        config=config,  # Pass the config object instead of individual parameters
                        screenshot=screenshot_data,
                        pdf_data=pdf_data,
                        verbose=config.verbose,
                        **kwargs
                    )

                    # Set response data
                    if async_response:
                        crawl_result.status_code = async_response.status_code
                        crawl_result.response_headers = async_response.response_headers
                        crawl_result.downloaded_files = async_response.downloaded_files
                    else:
                        crawl_result.status_code = 200
                        crawl_result.response_headers = cached_result.response_headers if cached_result else {}

                    crawl_result.success = bool(html)
                    crawl_result.session_id = getattr(config, 'session_id', None)

                    self.logger.success(
                        message="{url:.50}... | Status: {status} | Total: {timing}",
                        tag="COMPLETE",
                        params={
                            "url": cache_context.display_url,
                            "status": crawl_result.success,
                            "timing": f"{time.perf_counter() - start_time:.2f}s"
                        },
                        colors={
                            "status": Fore.GREEN if crawl_result.success else Fore.RED,
                            "timing": Fore.YELLOW
                        }
                    )

                    # Update cache if appropriate
                    if cache_context.should_write() and not bool(cached_result):
                        await async_db_manager.acache_url(crawl_result)

                    return crawl_result

                except Exception as e:
                    error_context = get_error_context(sys.exc_info())
                
                    error_message = (
                        f"Unexpected error in _crawl_web at line {error_context['line_no']} "
                        f"in {error_context['function']} ({error_context['filename']}):\n"
                        f"Error: {str(e)}\n\n"
                        f"Code context:\n{error_context['code_context']}"
                    )
                    # if not hasattr(e, "msg"):
                    #     e.msg = str(e)
                    
                    self.logger.error_status(
                        url=url,
                        error=create_box_message(error_message, type="error"),
                        tag="ERROR"
                    )
                    
                    return CrawlResult(
                        url=url,
                        html="",
                        success=False,
                        error_message=error_message
                    )

    async def aprocess_html(
            self,
            url: str,
            html: str,
            extracted_content: str,
            config: CrawlerRunConfig,
            screenshot: str,
            pdf_data: str,
            verbose: bool,
            **kwargs,
        ) -> CrawlResult:
            """
            Process HTML content using the provided configuration.
            
            Args:
                url: The URL being processed
                html: Raw HTML content
                extracted_content: Previously extracted content (if any)
                config: Configuration object controlling processing behavior
                screenshot: Screenshot data (if any)
                verbose: Whether to enable verbose logging
                **kwargs: Additional parameters for backwards compatibility
            
            Returns:
                CrawlResult: Processed result containing extracted and formatted content
            """
            try:
                _url = url if not kwargs.get("is_raw_html", False) else "Raw HTML"
                t1 = time.perf_counter()

                # Initialize scraping strategy
                scrapping_strategy = WebScrapingStrategy(logger=self.logger)

                # Process HTML content
                result = scrapping_strategy.scrap(
                    url,
                    html,
                    word_count_threshold=config.word_count_threshold,
                    css_selector=config.css_selector,
                    only_text=config.only_text,
                    image_description_min_word_threshold=config.image_description_min_word_threshold,
                    content_filter=config.content_filter,
                    **kwargs
                )

                if result is None:
                    raise ValueError(f"Process HTML, Failed to extract content from the website: {url}")

            except InvalidCSSSelectorError as e:
                raise ValueError(str(e))
            except Exception as e:
                raise ValueError(f"Process HTML, Failed to extract content from the website: {url}, error: {str(e)}")

       

            # Extract results
            cleaned_html = sanitize_input_encode(result.get("cleaned_html", ""))
            fit_markdown = sanitize_input_encode(result.get("fit_markdown", ""))
            fit_html = sanitize_input_encode(result.get("fit_html", ""))
            media = result.get("media", [])
            links = result.get("links", [])
            metadata = result.get("metadata", {})

            # Markdown Generation
            markdown_generator: Optional[MarkdownGenerationStrategy] = config.markdown_generator or DefaultMarkdownGenerator()
            if not config.content_filter and not markdown_generator.content_filter:
                markdown_generator.content_filter = PruningContentFilter()
            
            markdown_result: MarkdownGenerationResult = markdown_generator.generate_markdown(
                cleaned_html=cleaned_html,
                base_url=url,
                # html2text_options=kwargs.get('html2text', {})
            )
            markdown_v2 = markdown_result
            markdown = sanitize_input_encode(markdown_result.raw_markdown)

            # Log processing completion
            self.logger.info(
                message="Processed {url:.50}... | Time: {timing}ms",
                tag="SCRAPE",
                params={
                    "url": _url,
                    "timing": int((time.perf_counter() - t1) * 1000)
                }
            )

            # Handle content extraction if needed
            if (extracted_content is None and 
                config.extraction_strategy and 
                config.chunking_strategy and 
                not isinstance(config.extraction_strategy, NoExtractionStrategy)):
                
                t1 = time.perf_counter()
                
                # Handle different extraction strategy types
                if isinstance(config.extraction_strategy, (JsonCssExtractionStrategy, JsonCssExtractionStrategy)):
                    config.extraction_strategy.verbose = verbose
                    extracted_content = config.extraction_strategy.run(url, [html])
                    extracted_content = json.dumps(extracted_content, indent=4, default=str, ensure_ascii=False)
                else:
                    sections = config.chunking_strategy.chunk(markdown)
                    extracted_content = config.extraction_strategy.run(url, sections)
                    extracted_content = json.dumps(extracted_content, indent=4, default=str, ensure_ascii=False)

                # Log extraction completion
                self.logger.info(
                    message="Completed for {url:.50}... | Time: {timing}s",
                    tag="EXTRACT",
                    params={
                        "url": _url,
                        "timing": time.perf_counter() - t1
                    }
                )

            # Handle screenshot and PDF data
            screenshot_data = None if not screenshot else screenshot
            pdf_data = None if not pdf_data else pdf_data

            # Apply HTML formatting if requested
            if config.prettiify:
                cleaned_html = fast_format_html(cleaned_html)

            # Return complete crawl result
            return CrawlResult(
                url=url,
                html=html,
                cleaned_html=cleaned_html,
                markdown_v2=markdown_v2,
                markdown=markdown,
                fit_markdown=fit_markdown,
                fit_html=fit_html,
                media=media,
                links=links,
                metadata=metadata,
                screenshot=screenshot_data,
                pdf=pdf_data,
                extracted_content=extracted_content,
                success=True,
                error_message="",
            )    

    async def arun_many(
            self,
            urls: List[str],
            config: Optional[CrawlerRunConfig] = None,
            # Legacy parameters maintained for backwards compatibility
            word_count_threshold=MIN_WORD_THRESHOLD,
            extraction_strategy: ExtractionStrategy = None,
            chunking_strategy: ChunkingStrategy = RegexChunking(),
            content_filter: RelevantContentFilter = None,
            cache_mode: Optional[CacheMode] = None,
            bypass_cache: bool = False,
            css_selector: str = None,
            screenshot: bool = False,
            pdf: bool = False,
            user_agent: str = None,
            verbose=True,
            **kwargs,
        ) -> List[CrawlResult]:
            """
            Runs the crawler for multiple URLs concurrently.

            Migration Guide:
            Old way (deprecated):
                results = await crawler.arun_many(
                    urls,
                    word_count_threshold=200,
                    screenshot=True,
                    ...
                )
            
            New way (recommended):
                config = CrawlerRunConfig(
                    word_count_threshold=200,
                    screenshot=True,
                    ...
                )
                results = await crawler.arun_many(urls, crawler_config=config)

            Args:
                urls: List of URLs to crawl
                crawler_config: Configuration object controlling crawl behavior for all URLs
                [other parameters maintained for backwards compatibility]
            
            Returns:
                List[CrawlResult]: Results for each URL
            """
            crawler_config = config
            # Handle configuration
            if crawler_config is not None:
                if any(param is not None for param in [
                    word_count_threshold, extraction_strategy, chunking_strategy,
                    content_filter, cache_mode, css_selector, screenshot, pdf
                ]):
                    self.logger.warning(
                        message="Both crawler_config and legacy parameters provided. crawler_config will take precedence.",
                        tag="WARNING"
                    )
                config = crawler_config
            else:
                # Merge all parameters into a single kwargs dict for config creation
                config_kwargs = {
                    "word_count_threshold": word_count_threshold,
                    "extraction_strategy": extraction_strategy,
                    "chunking_strategy": chunking_strategy,
                    "content_filter": content_filter,
                    "cache_mode": cache_mode,
                    "bypass_cache": bypass_cache,
                    "css_selector": css_selector,
                    "screenshot": screenshot,
                    "pdf": pdf,
                    "verbose": verbose,
                    **kwargs
                }
                config = CrawlerRunConfig.from_kwargs(config_kwargs)

            if bypass_cache:
                if kwargs.get("warning", True):
                    warnings.warn(
                        "'bypass_cache' is deprecated and will be removed in version 0.5.0. "
                        "Use 'cache_mode=CacheMode.BYPASS' instead. "
                        "Pass warning=False to suppress this warning.",
                        DeprecationWarning,
                        stacklevel=2
                    )
                if config.cache_mode is None:
                    config.cache_mode = CacheMode.BYPASS

            semaphore_count = config.semaphore_count or 5
            semaphore = asyncio.Semaphore(semaphore_count)

            async def crawl_with_semaphore(url):
                # Handle rate limiting per domain
                domain = urlparse(url).netloc
                current_time = time.time()
                
                self.logger.debug(
                    message="Started task for {url:.50}...",
                    tag="PARALLEL",
                    params={"url": url}
                )

                # Get delay settings from config
                mean_delay = config.mean_delay
                max_range = config.max_range
                
                # Apply rate limiting
                if domain in self._domain_last_hit:
                    time_since_last = current_time - self._domain_last_hit[domain]
                    if time_since_last < mean_delay:
                        delay = mean_delay + random.uniform(0, max_range)
                        await asyncio.sleep(delay)
                
                self._domain_last_hit[domain] = current_time

                async with semaphore:
                    return await self.arun(
                        url,
                        crawler_config=config,  # Pass the entire config object
                        user_agent=user_agent  # Maintain user_agent override capability
                    )

            # Log start of concurrent crawling
            self.logger.info(
                message="Starting concurrent crawling for {count} URLs...",
                tag="INIT",
                params={"count": len(urls)}
            )

            # Execute concurrent crawls
            start_time = time.perf_counter()
            tasks = [crawl_with_semaphore(url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.perf_counter()

            # Log completion
            self.logger.success(
                message="Concurrent crawling completed for {count} URLs | Total time: {timing}",
                tag="COMPLETE",
                params={
                    "count": len(urls),
                    "timing": f"{end_time - start_time:.2f}s"
                },
                colors={
                    "timing": Fore.YELLOW
                }
            )

            return [result if not isinstance(result, Exception) else str(result) for result in results]

    async def aclear_cache(self):
        """Clear the cache database."""
        await async_db_manager.cleanup()

    async def aflush_cache(self):
        """Flush the cache database."""
        await async_db_manager.aflush_db()

    async def aget_cache_size(self):
        """Get the total number of cached items."""
        return await async_db_manager.aget_total_count()



```

## tests/test_web_crawler.py
```
import unittest, os
from crawl4ai.web_crawler import WebCrawler
from crawl4ai.chunking_strategy import RegexChunking, FixedLengthWordChunking, SlidingWindowChunking
from crawl4ai.extraction_strategy import CosineStrategy, LLMExtractionStrategy, TopicExtractionStrategy, NoExtractionStrategy

class TestWebCrawler(unittest.TestCase):
    
    def setUp(self):
        self.crawler = WebCrawler()
    
    def test_warmup(self):
        self.crawler.warmup()
        self.assertTrue(self.crawler.ready, "WebCrawler failed to warm up")
    
    def test_run_default_strategies(self):
        result = self.crawler.run(
            url='https://www.nbcnews.com/business',
            word_count_threshold=5,
            chunking_strategy=RegexChunking(),
            extraction_strategy=CosineStrategy(), bypass_cache=True
        )
        self.assertTrue(result.success, "Failed to crawl and extract using default strategies")
    
    def test_run_different_strategies(self):
        url = 'https://www.nbcnews.com/business'
        
        # Test with FixedLengthWordChunking and LLMExtractionStrategy
        result = self.crawler.run(
            url=url,
            word_count_threshold=5,
            chunking_strategy=FixedLengthWordChunking(chunk_size=100),
            extraction_strategy=LLMExtractionStrategy(provider="openai/gpt-3.5-turbo", api_token=os.getenv('OPENAI_API_KEY')), bypass_cache=True
        )
        self.assertTrue(result.success, "Failed to crawl and extract with FixedLengthWordChunking and LLMExtractionStrategy")
        
        # Test with SlidingWindowChunking and TopicExtractionStrategy
        result = self.crawler.run(
            url=url,
            word_count_threshold=5,
            chunking_strategy=SlidingWindowChunking(window_size=100, step=50),
            extraction_strategy=TopicExtractionStrategy(num_keywords=5), bypass_cache=True
        )
        self.assertTrue(result.success, "Failed to crawl and extract with SlidingWindowChunking and TopicExtractionStrategy")
    
    def test_invalid_url(self):
        with self.assertRaises(Exception) as context:
            self.crawler.run(url='invalid_url', bypass_cache=True)
        self.assertIn("Invalid URL", str(context.exception))
    
    def test_unsupported_extraction_strategy(self):
        with self.assertRaises(Exception) as context:
            self.crawler.run(url='https://www.nbcnews.com/business', extraction_strategy="UnsupportedStrategy", bypass_cache=True)
        self.assertIn("Unsupported extraction strategy", str(context.exception))
    
    def test_invalid_css_selector(self):
        with self.assertRaises(ValueError) as context:
            self.crawler.run(url='https://www.nbcnews.com/business', css_selector="invalid_selector", bypass_cache=True)
        self.assertIn("Invalid CSS selector", str(context.exception))

    
    def test_crawl_with_cache_and_bypass_cache(self):
        url = 'https://www.nbcnews.com/business'
        
        # First crawl with cache enabled
        result = self.crawler.run(url=url, bypass_cache=False)
        self.assertTrue(result.success, "Failed to crawl and cache the result")
        
        # Second crawl with bypass_cache=True
        result = self.crawler.run(url=url, bypass_cache=True)
        self.assertTrue(result.success, "Failed to bypass cache and fetch fresh data")
    
    def test_fetch_multiple_pages(self):
        urls = [
            'https://www.nbcnews.com/business',
            'https://www.bbc.com/news'
        ]
        results = []
        for url in urls:
            result = self.crawler.run(
                url=url,
                word_count_threshold=5,
                chunking_strategy=RegexChunking(),
                extraction_strategy=CosineStrategy(),
                bypass_cache=True
            )
            results.append(result)
        
        self.assertEqual(len(results), 2, "Failed to crawl and extract multiple pages")
        for result in results:
            self.assertTrue(result.success, "Failed to crawl and extract a page in the list")
    
    def test_run_fixed_length_word_chunking_and_no_extraction(self):
        result = self.crawler.run(
            url='https://www.nbcnews.com/business',
            word_count_threshold=5,
            chunking_strategy=FixedLengthWordChunking(chunk_size=100),
            extraction_strategy=NoExtractionStrategy(), bypass_cache=True
        )
        self.assertTrue(result.success, "Failed to crawl and extract with FixedLengthWordChunking and NoExtractionStrategy")

    def test_run_sliding_window_and_no_extraction(self):
        result = self.crawler.run(
            url='https://www.nbcnews.com/business',
            word_count_threshold=5,
            chunking_strategy=SlidingWindowChunking(window_size=100, step=50),
            extraction_strategy=NoExtractionStrategy(), bypass_cache=True
        )
        self.assertTrue(result.success, "Failed to crawl and extract with SlidingWindowChunking and NoExtractionStrategy")

if __name__ == '__main__':
    unittest.main()

```

## tests/docker_example.py
```
import requests
import json
import time
import sys
import base64
import os
from typing import Dict, Any

class Crawl4AiTester:
    def __init__(self, base_url: str = "http://localhost:11235", api_token: str = None):
        self.base_url = base_url
        self.api_token = api_token or os.getenv('CRAWL4AI_API_TOKEN')  # Check environment variable as fallback
        self.headers = {'Authorization': f'Bearer {self.api_token}'} if self.api_token else {}
        
    def submit_and_wait(self, request_data: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
        # Submit crawl job
        response = requests.post(f"{self.base_url}/crawl", json=request_data, headers=self.headers)
        if response.status_code == 403:
            raise Exception("API token is invalid or missing")
        task_id = response.json()["task_id"]
        print(f"Task ID: {task_id}")
        
        # Poll for result
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
                
            result = requests.get(f"{self.base_url}/task/{task_id}", headers=self.headers)
            status = result.json()
            
            if status["status"] == "failed":
                print("Task failed:", status.get("error"))
                raise Exception(f"Task failed: {status.get('error')}")
                
            if status["status"] == "completed":
                return status
                
            time.sleep(2)
            
    def submit_sync(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/crawl_sync", json=request_data, headers=self.headers, timeout=60)
        if response.status_code == 408:
            raise TimeoutError("Task did not complete within server timeout")
        response.raise_for_status()
        return response.json()

def test_docker_deployment(version="basic"):
    tester = Crawl4AiTester(
        # base_url="http://localhost:11235" ,
        base_url="https://crawl4ai-sby74.ondigitalocean.app",
        api_token="test"
    )
    print(f"Testing Crawl4AI Docker {version} version")
    
    # Health check with timeout and retry
    max_retries = 5
    for i in range(max_retries):
        try:
            health = requests.get(f"{tester.base_url}/health", timeout=10)
            print("Health check:", health.json())
            break
        except requests.exceptions.RequestException as e:
            if i == max_retries - 1:
                print(f"Failed to connect after {max_retries} attempts")
                sys.exit(1)
            print(f"Waiting for service to start (attempt {i+1}/{max_retries})...")
            time.sleep(5)
    
    # Test cases based on version
    test_basic_crawl(tester)
    test_basic_crawl(tester)
    test_basic_crawl_sync(tester)
    
    # if version in ["full", "transformer"]:
    #     test_cosine_extraction(tester)

    # test_js_execution(tester)
    # test_css_selector(tester)
    # test_structured_extraction(tester)
    # test_llm_extraction(tester)
    # test_llm_with_ollama(tester)
    # test_screenshot(tester)
    

def test_basic_crawl(tester: Crawl4AiTester):
    print("\n=== Testing Basic Crawl ===")
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 10, 
        "session_id": "test"
    }
    
    result = tester.submit_and_wait(request)
    print(f"Basic crawl result length: {len(result['result']['markdown'])}")
    assert result["result"]["success"]
    assert len(result["result"]["markdown"]) > 0

def test_basic_crawl_sync(tester: Crawl4AiTester):
    print("\n=== Testing Basic Crawl (Sync) ===")
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 10,
        "session_id": "test"
    }
    
    result = tester.submit_sync(request)
    print(f"Basic crawl result length: {len(result['result']['markdown'])}")
    assert result['status'] == 'completed'
    assert result['result']['success']
    assert len(result['result']['markdown']) > 0
    
def test_js_execution(tester: Crawl4AiTester):
    print("\n=== Testing JS Execution ===")
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 8,
        "js_code": [
            "const loadMoreButton = Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('Load More')); loadMoreButton && loadMoreButton.click();"
        ],
        "wait_for": "article.tease-card:nth-child(10)",
        "crawler_params": {
            "headless": True
        }
    }
    
    result = tester.submit_and_wait(request)
    print(f"JS execution result length: {len(result['result']['markdown'])}")
    assert result["result"]["success"]

def test_css_selector(tester: Crawl4AiTester):
    print("\n=== Testing CSS Selector ===")
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 7,
        "css_selector": ".wide-tease-item__description",
        "crawler_params": {
            "headless": True
        },
        "extra": {"word_count_threshold": 10}
        
    }
    
    result = tester.submit_and_wait(request)
    print(f"CSS selector result length: {len(result['result']['markdown'])}")
    assert result["result"]["success"]

def test_structured_extraction(tester: Crawl4AiTester):
    print("\n=== Testing Structured Extraction ===")
    schema = {
        "name": "Coinbase Crypto Prices",
        "baseSelector": ".cds-tableRow-t45thuk",
        "fields": [
            {
                "name": "crypto",
                "selector": "td:nth-child(1) h2",
                "type": "text",
            },
            {
                "name": "symbol",
                "selector": "td:nth-child(1) p",
                "type": "text",
            },
            {
                "name": "price",
                "selector": "td:nth-child(2)",
                "type": "text",
            }
        ],
    }
    
    request = {
        "urls": "https://www.coinbase.com/explore",
        "priority": 9,
        "extraction_config": {
            "type": "json_css",
            "params": {
                "schema": schema
            }
        }
    }
    
    result = tester.submit_and_wait(request)
    extracted = json.loads(result["result"]["extracted_content"])
    print(f"Extracted {len(extracted)} items")
    print("Sample item:", json.dumps(extracted[0], indent=2))
    assert result["result"]["success"]
    assert len(extracted) > 0

def test_llm_extraction(tester: Crawl4AiTester):
    print("\n=== Testing LLM Extraction ===")
    schema = {
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the OpenAI model."
            },
            "input_fee": {
                "type": "string",
                "description": "Fee for input token for the OpenAI model."
            },
            "output_fee": {
                "type": "string",
                "description": "Fee for output token for the OpenAI model."
            }
        },
        "required": ["model_name", "input_fee", "output_fee"]
    }
    
    request = {
        "urls": "https://openai.com/api/pricing",
        "priority": 8,
        "extraction_config": {
            "type": "llm",
            "params": {
                "provider": "openai/gpt-4o-mini",
                "api_token": os.getenv("OPENAI_API_KEY"),
                "schema": schema,
                "extraction_type": "schema",
                "instruction": """From the crawled content, extract all mentioned model names along with their fees for input and output tokens."""
            }
        },
        "crawler_params": {"word_count_threshold": 1}
    }
    
    try:
        result = tester.submit_and_wait(request)
        extracted = json.loads(result["result"]["extracted_content"])
        print(f"Extracted {len(extracted)} model pricing entries")
        print("Sample entry:", json.dumps(extracted[0], indent=2))
        assert result["result"]["success"]
    except Exception as e:
        print(f"LLM extraction test failed (might be due to missing API key): {str(e)}")

def test_llm_with_ollama(tester: Crawl4AiTester):
    print("\n=== Testing LLM with Ollama ===")
    schema = {
        "type": "object",
        "properties": {
            "article_title": {
                "type": "string",
                "description": "The main title of the news article"
            },
            "summary": {
                "type": "string",
                "description": "A brief summary of the article content"
            },
            "main_topics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Main topics or themes discussed in the article"
            }
        }
    }
    
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 8,
        "extraction_config": {
            "type": "llm",
            "params": {
                "provider": "ollama/llama2",
                "schema": schema,
                "extraction_type": "schema",
                "instruction": "Extract the main article information including title, summary, and main topics."
            }
        },
        "extra": {"word_count_threshold": 1},
        "crawler_params": {"verbose": True}
    }
    
    try:
        result = tester.submit_and_wait(request)
        extracted = json.loads(result["result"]["extracted_content"])
        print("Extracted content:", json.dumps(extracted, indent=2))
        assert result["result"]["success"]
    except Exception as e:
        print(f"Ollama extraction test failed: {str(e)}")

def test_cosine_extraction(tester: Crawl4AiTester):
    print("\n=== Testing Cosine Extraction ===")
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 8,
        "extraction_config": {
            "type": "cosine",
            "params": {
                "semantic_filter": "business finance economy",
                "word_count_threshold": 10,
                "max_dist": 0.2,
                "top_k": 3
            }
        }
    }
    
    try:
        result = tester.submit_and_wait(request)
        extracted = json.loads(result["result"]["extracted_content"])
        print(f"Extracted {len(extracted)} text clusters")
        print("First cluster tags:", extracted[0]["tags"])
        assert result["result"]["success"]
    except Exception as e:
        print(f"Cosine extraction test failed: {str(e)}")

def test_screenshot(tester: Crawl4AiTester):
    print("\n=== Testing Screenshot ===")
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 5,
        "screenshot": True,
        "crawler_params": {
            "headless": True
        }
    }
    
    result = tester.submit_and_wait(request)
    print("Screenshot captured:", bool(result["result"]["screenshot"]))
    
    if result["result"]["screenshot"]:
        # Save screenshot
        screenshot_data = base64.b64decode(result["result"]["screenshot"])
        with open("test_screenshot.jpg", "wb") as f:
            f.write(screenshot_data)
        print("Screenshot saved as test_screenshot.jpg")
    
    assert result["result"]["success"]

if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "basic"
    # version = "full"
    test_docker_deployment(version)
```

## tests/test_main.py
```
import asyncio
import aiohttp
import json
import time
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, HttpUrl

class NBCNewsAPITest:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def submit_crawl(self, request_data: Dict[str, Any]) -> str:
        async with self.session.post(f"{self.base_url}/crawl", json=request_data) as response:
            result = await response.json()
            return result["task_id"]

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        async with self.session.get(f"{self.base_url}/task/{task_id}") as response:
            return await response.json()

    async def wait_for_task(self, task_id: str, timeout: int = 300, poll_interval: int = 2) -> Dict[str, Any]:
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")

            status = await self.get_task_status(task_id)
            if status["status"] in ["completed", "failed"]:
                return status

            await asyncio.sleep(poll_interval)

    async def check_health(self) -> Dict[str, Any]:
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()

async def test_basic_crawl():
    print("\n=== Testing Basic Crawl ===")
    async with NBCNewsAPITest() as api:
        request = {
            "urls": "https://www.nbcnews.com/business",
            "priority": 10
        }
        task_id = await api.submit_crawl(request)
        result = await api.wait_for_task(task_id)
        print(f"Basic crawl result length: {len(result['result']['markdown'])}")
        assert result["status"] == "completed"
        assert "result" in result
        assert result["result"]["success"]

async def test_js_execution():
    print("\n=== Testing JS Execution ===")
    async with NBCNewsAPITest() as api:
        request = {
            "urls": "https://www.nbcnews.com/business",
            "priority": 8,
            "js_code": [
                "const loadMoreButton = Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('Load More')); loadMoreButton && loadMoreButton.click();"
            ],
            "wait_for": "article.tease-card:nth-child(10)",
            "crawler_params": {
                "headless": True
            }
        }
        task_id = await api.submit_crawl(request)
        result = await api.wait_for_task(task_id)
        print(f"JS execution result length: {len(result['result']['markdown'])}")
        assert result["status"] == "completed"
        assert result["result"]["success"]

async def test_css_selector():
    print("\n=== Testing CSS Selector ===")
    async with NBCNewsAPITest() as api:
        request = {
            "urls": "https://www.nbcnews.com/business",
            "priority": 7,
            "css_selector": ".wide-tease-item__description"
        }
        task_id = await api.submit_crawl(request)
        result = await api.wait_for_task(task_id)
        print(f"CSS selector result length: {len(result['result']['markdown'])}")
        assert result["status"] == "completed"
        assert result["result"]["success"]

async def test_structured_extraction():
    print("\n=== Testing Structured Extraction ===")
    async with NBCNewsAPITest() as api:
        schema = {
            "name": "NBC News Articles",
            "baseSelector": "article.tease-card",
            "fields": [
                {
                    "name": "title",
                    "selector": "h2",
                    "type": "text"
                },
                {
                    "name": "description",
                    "selector": ".tease-card__description",
                    "type": "text"
                },
                {
                    "name": "link",
                    "selector": "a",
                    "type": "attribute",
                    "attribute": "href"
                }
            ]
        }
        
        request = {
            "urls": "https://www.nbcnews.com/business",
            "priority": 9,
            "extraction_config": {
                "type": "json_css",
                "params": {
                    "schema": schema
                }
            }
        }
        task_id = await api.submit_crawl(request)
        result = await api.wait_for_task(task_id)
        extracted = json.loads(result["result"]["extracted_content"])
        print(f"Extracted {len(extracted)} articles")
        assert result["status"] == "completed"
        assert result["result"]["success"]
        assert len(extracted) > 0

async def test_batch_crawl():
    print("\n=== Testing Batch Crawl ===")
    async with NBCNewsAPITest() as api:
        request = {
            "urls": [
                "https://www.nbcnews.com/business",
                "https://www.nbcnews.com/business/consumer",
                "https://www.nbcnews.com/business/economy"
            ],
            "priority": 6,
            "crawler_params": {
                "headless": True
            }
        }
        task_id = await api.submit_crawl(request)
        result = await api.wait_for_task(task_id)
        print(f"Batch crawl completed, got {len(result['results'])} results")
        assert result["status"] == "completed"
        assert "results" in result
        assert len(result["results"]) == 3

async def test_llm_extraction():
    print("\n=== Testing LLM Extraction with Ollama ===")
    async with NBCNewsAPITest() as api:
        schema = {
            "type": "object",
            "properties": {
                "article_title": {
                    "type": "string",
                    "description": "The main title of the news article"
                },
                "summary": {
                    "type": "string",
                    "description": "A brief summary of the article content"
                },
                "main_topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Main topics or themes discussed in the article"
                }
            },
            "required": ["article_title", "summary", "main_topics"]
        }

        request = {
            "urls": "https://www.nbcnews.com/business",
            "priority": 8,
            "extraction_config": {
                "type": "llm",
                "params": {
                    "provider": "openai/gpt-4o-mini",
                    "api_key": os.getenv("OLLAMA_API_KEY"),
                    "schema": schema,
                    "extraction_type": "schema",
                    "instruction": """Extract the main article information including title, a brief summary, and main topics discussed. 
                    Focus on the primary business news article on the page."""
                }
            },
            "crawler_params": {
                "headless": True,
                "word_count_threshold": 1
            }
        }
        
        task_id = await api.submit_crawl(request)
        result = await api.wait_for_task(task_id)
        
        if result["status"] == "completed":
            extracted = json.loads(result["result"]["extracted_content"])
            print(f"Extracted article analysis:")
            print(json.dumps(extracted, indent=2))
        
        assert result["status"] == "completed"
        assert result["result"]["success"]

async def test_screenshot():
    print("\n=== Testing Screenshot ===")
    async with NBCNewsAPITest() as api:
        request = {
            "urls": "https://www.nbcnews.com/business",
            "priority": 5,
            "screenshot": True,
            "crawler_params": {
                "headless": True
            }
        }
        task_id = await api.submit_crawl(request)
        result = await api.wait_for_task(task_id)
        print("Screenshot captured:", bool(result["result"]["screenshot"]))
        assert result["status"] == "completed"
        assert result["result"]["success"]
        assert result["result"]["screenshot"] is not None

async def test_priority_handling():
    print("\n=== Testing Priority Handling ===")
    async with NBCNewsAPITest() as api:
        # Submit low priority task first
        low_priority = {
            "urls": "https://www.nbcnews.com/business",
            "priority": 1,
            "crawler_params": {"headless": True}
        }
        low_task_id = await api.submit_crawl(low_priority)

        # Submit high priority task
        high_priority = {
            "urls": "https://www.nbcnews.com/business/consumer",
            "priority": 10,
            "crawler_params": {"headless": True}
        }
        high_task_id = await api.submit_crawl(high_priority)

        # Get both results
        high_result = await api.wait_for_task(high_task_id)
        low_result = await api.wait_for_task(low_task_id)

        print("Both tasks completed")
        assert high_result["status"] == "completed"
        assert low_result["status"] == "completed"

async def main():
    try:
        # Start with health check
        async with NBCNewsAPITest() as api:
            health = await api.check_health()
            print("Server health:", health)

        # Run all tests
        # await test_basic_crawl()
        # await test_js_execution()
        # await test_css_selector()
        # await test_structured_extraction()
        await test_llm_extraction()
        # await test_batch_crawl()
        # await test_screenshot()
        # await test_priority_handling()

    except Exception as e:
        print(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
```

## tests/test_docker.py
```
import requests
import json
import time
import sys
import base64
import os
from typing import Dict, Any

class Crawl4AiTester:
    def __init__(self, base_url: str = "http://localhost:11235"):
        self.base_url = base_url
        
    def submit_and_wait(self, request_data: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
        # Submit crawl job
        response = requests.post(f"{self.base_url}/crawl", json=request_data)
        task_id = response.json()["task_id"]
        print(f"Task ID: {task_id}")
        
        # Poll for result
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
                
            result = requests.get(f"{self.base_url}/task/{task_id}")
            status = result.json()
            
            if status["status"] == "failed":
                print("Task failed:", status.get("error"))
                raise Exception(f"Task failed: {status.get('error')}")
                
            if status["status"] == "completed":
                return status
                
            time.sleep(2)

def test_docker_deployment(version="basic"):
    tester = Crawl4AiTester()
    print(f"Testing Crawl4AI Docker {version} version")
    
    # Health check with timeout and retry
    max_retries = 5
    for i in range(max_retries):
        try:
            health = requests.get(f"{tester.base_url}/health", timeout=10)
            print("Health check:", health.json())
            break
        except requests.exceptions.RequestException as e:
            if i == max_retries - 1:
                print(f"Failed to connect after {max_retries} attempts")
                sys.exit(1)
            print(f"Waiting for service to start (attempt {i+1}/{max_retries})...")
            time.sleep(5)
    
    # Test cases based on version
    test_basic_crawl(tester)
    
    # if version in ["full", "transformer"]:
    #     test_cosine_extraction(tester)

    # test_js_execution(tester)
    # test_css_selector(tester)
    # test_structured_extraction(tester)
    # test_llm_extraction(tester)
    # test_llm_with_ollama(tester)
    # test_screenshot(tester)
    

def test_basic_crawl(tester: Crawl4AiTester):
    print("\n=== Testing Basic Crawl ===")
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 10
    }
    
    result = tester.submit_and_wait(request)
    print(f"Basic crawl result length: {len(result['result']['markdown'])}")
    assert result["result"]["success"]
    assert len(result["result"]["markdown"]) > 0

def test_js_execution(tester: Crawl4AiTester):
    print("\n=== Testing JS Execution ===")
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 8,
        "js_code": [
            "const loadMoreButton = Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('Load More')); loadMoreButton && loadMoreButton.click();"
        ],
        "wait_for": "article.tease-card:nth-child(10)",
        "crawler_params": {
            "headless": True
        }
    }
    
    result = tester.submit_and_wait(request)
    print(f"JS execution result length: {len(result['result']['markdown'])}")
    assert result["result"]["success"]

def test_css_selector(tester: Crawl4AiTester):
    print("\n=== Testing CSS Selector ===")
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 7,
        "css_selector": ".wide-tease-item__description",
        "crawler_params": {
            "headless": True
        },
        "extra": {"word_count_threshold": 10}
        
    }
    
    result = tester.submit_and_wait(request)
    print(f"CSS selector result length: {len(result['result']['markdown'])}")
    assert result["result"]["success"]

def test_structured_extraction(tester: Crawl4AiTester):
    print("\n=== Testing Structured Extraction ===")
    schema = {
        "name": "Coinbase Crypto Prices",
        "baseSelector": ".cds-tableRow-t45thuk",
        "fields": [
            {
                "name": "crypto",
                "selector": "td:nth-child(1) h2",
                "type": "text",
            },
            {
                "name": "symbol",
                "selector": "td:nth-child(1) p",
                "type": "text",
            },
            {
                "name": "price",
                "selector": "td:nth-child(2)",
                "type": "text",
            }
        ],
    }
    
    request = {
        "urls": "https://www.coinbase.com/explore",
        "priority": 9,
        "extraction_config": {
            "type": "json_css",
            "params": {
                "schema": schema
            }
        }
    }
    
    result = tester.submit_and_wait(request)
    extracted = json.loads(result["result"]["extracted_content"])
    print(f"Extracted {len(extracted)} items")
    print("Sample item:", json.dumps(extracted[0], indent=2))
    assert result["result"]["success"]
    assert len(extracted) > 0

def test_llm_extraction(tester: Crawl4AiTester):
    print("\n=== Testing LLM Extraction ===")
    schema = {
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the OpenAI model."
            },
            "input_fee": {
                "type": "string",
                "description": "Fee for input token for the OpenAI model."
            },
            "output_fee": {
                "type": "string",
                "description": "Fee for output token for the OpenAI model."
            }
        },
        "required": ["model_name", "input_fee", "output_fee"]
    }
    
    request = {
        "urls": "https://openai.com/api/pricing",
        "priority": 8,
        "extraction_config": {
            "type": "llm",
            "params": {
                "provider": "openai/gpt-4o-mini",
                "api_token": os.getenv("OPENAI_API_KEY"),
                "schema": schema,
                "extraction_type": "schema",
                "instruction": """From the crawled content, extract all mentioned model names along with their fees for input and output tokens."""
            }
        },
        "crawler_params": {"word_count_threshold": 1}
    }
    
    try:
        result = tester.submit_and_wait(request)
        extracted = json.loads(result["result"]["extracted_content"])
        print(f"Extracted {len(extracted)} model pricing entries")
        print("Sample entry:", json.dumps(extracted[0], indent=2))
        assert result["result"]["success"]
    except Exception as e:
        print(f"LLM extraction test failed (might be due to missing API key): {str(e)}")

def test_llm_with_ollama(tester: Crawl4AiTester):
    print("\n=== Testing LLM with Ollama ===")
    schema = {
        "type": "object",
        "properties": {
            "article_title": {
                "type": "string",
                "description": "The main title of the news article"
            },
            "summary": {
                "type": "string",
                "description": "A brief summary of the article content"
            },
            "main_topics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Main topics or themes discussed in the article"
            }
        }
    }
    
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 8,
        "extraction_config": {
            "type": "llm",
            "params": {
                "provider": "ollama/llama2",
                "schema": schema,
                "extraction_type": "schema",
                "instruction": "Extract the main article information including title, summary, and main topics."
            }
        },
        "extra": {"word_count_threshold": 1},
        "crawler_params": {"verbose": True}
    }
    
    try:
        result = tester.submit_and_wait(request)
        extracted = json.loads(result["result"]["extracted_content"])
        print("Extracted content:", json.dumps(extracted, indent=2))
        assert result["result"]["success"]
    except Exception as e:
        print(f"Ollama extraction test failed: {str(e)}")

def test_cosine_extraction(tester: Crawl4AiTester):
    print("\n=== Testing Cosine Extraction ===")
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 8,
        "extraction_config": {
            "type": "cosine",
            "params": {
                "semantic_filter": "business finance economy",
                "word_count_threshold": 10,
                "max_dist": 0.2,
                "top_k": 3
            }
        }
    }
    
    try:
        result = tester.submit_and_wait(request)
        extracted = json.loads(result["result"]["extracted_content"])
        print(f"Extracted {len(extracted)} text clusters")
        print("First cluster tags:", extracted[0]["tags"])
        assert result["result"]["success"]
    except Exception as e:
        print(f"Cosine extraction test failed: {str(e)}")

def test_screenshot(tester: Crawl4AiTester):
    print("\n=== Testing Screenshot ===")
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 5,
        "screenshot": True,
        "crawler_params": {
            "headless": True
        }
    }
    
    result = tester.submit_and_wait(request)
    print("Screenshot captured:", bool(result["result"]["screenshot"]))
    
    if result["result"]["screenshot"]:
        # Save screenshot
        screenshot_data = base64.b64decode(result["result"]["screenshot"])
        with open("test_screenshot.jpg", "wb") as f:
            f.write(screenshot_data)
        print("Screenshot saved as test_screenshot.jpg")
    
    assert result["result"]["success"]

if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "basic"
    # version = "full"
    test_docker_deployment(version)
```

## tests/async/test_0.4.2_browser_manager.py
```
import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
__location__ = os.path.realpath(    os.path.join(os.getcwd(), os.path.dirname(__file__)))

import os, sys
import asyncio
from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# Assuming that the changes made allow different configurations 
# for managed browser, persistent context, and so forth.

async def test_default_headless():
    async with AsyncWebCrawler(
        headless=True,
        verbose=True,
        user_agent_mode="random",
        user_agent_generator_config={"device_type": "mobile", "os_type": "android"},
        use_managed_browser=False,
        use_persistent_context=False,
        ignore_https_errors=True,
        # Testing normal ephemeral context
    ) as crawler:
        result = await crawler.arun(
            url='https://www.kidocode.com/degrees/technology',
            cache_mode=CacheMode.BYPASS,
            markdown_generator=DefaultMarkdownGenerator(options={"ignore_links": True}),
        )
        print("[test_default_headless] success:", result.success)
        print("HTML length:", len(result.html if result.html else ""))
        
async def test_managed_browser_persistent():
    # Treating use_persistent_context=True as managed_browser scenario.
    async with AsyncWebCrawler(
        headless=False,
        verbose=True,
        user_agent_mode="random",
        user_agent_generator_config={"device_type": "desktop", "os_type": "mac"},
        use_managed_browser=True,
        use_persistent_context=True,  # now should behave same as managed browser
        user_data_dir="./outpu/test_profile",
        # This should store and reuse profile data across runs
    ) as crawler:
        result = await crawler.arun(
            url='https://www.google.com',
            cache_mode=CacheMode.BYPASS,
            markdown_generator=DefaultMarkdownGenerator(options={"ignore_links": True})
        )
        print("[test_managed_browser_persistent] success:", result.success)
        print("HTML length:", len(result.html if result.html else ""))

async def test_session_reuse():
    # Test creating a session, using it for multiple calls
    session_id = "my_session"
    async with AsyncWebCrawler(
        headless=False,
        verbose=True,
        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        # Fixed user-agent for consistency
        use_managed_browser=False,
        use_persistent_context=False,
    ) as crawler:
        
        # First call: create session
        result1 = await crawler.arun(
            url='https://www.example.com',
            cache_mode=CacheMode.BYPASS,
            session_id=session_id,
            markdown_generator=DefaultMarkdownGenerator(options={"ignore_links": True})
        )
        print("[test_session_reuse first call] success:", result1.success)
        
        # Second call: same session, possibly cookie retained
        result2 = await crawler.arun(
            url='https://www.example.com/about',
            cache_mode=CacheMode.BYPASS,
            session_id=session_id,
            markdown_generator=DefaultMarkdownGenerator(options={"ignore_links": True})
        )
        print("[test_session_reuse second call] success:", result2.success)

async def test_magic_mode():
    # Test magic mode with override_navigator and simulate_user
    async with AsyncWebCrawler(
        headless=False,
        verbose=True,
        user_agent_mode="random",
        user_agent_generator_config={"device_type": "desktop", "os_type": "windows"},
        use_managed_browser=False,
        use_persistent_context=False,
        magic=True,
        override_navigator=True,
        simulate_user=True,
    ) as crawler:
        result = await crawler.arun(
            url='https://www.kidocode.com/degrees/business',
            cache_mode=CacheMode.BYPASS,
            markdown_generator=DefaultMarkdownGenerator(options={"ignore_links": True})
        )
        print("[test_magic_mode] success:", result.success)
        print("HTML length:", len(result.html if result.html else ""))

async def test_proxy_settings():
    # Test with a proxy (if available) to ensure code runs with proxy
    async with AsyncWebCrawler(
        headless=True,
        verbose=False,
        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        proxy="http://127.0.0.1:8080",  # Assuming local proxy server for test
        use_managed_browser=False,
        use_persistent_context=False,
    ) as crawler:
        result = await crawler.arun(
            url='https://httpbin.org/ip',
            cache_mode=CacheMode.BYPASS,
            markdown_generator=DefaultMarkdownGenerator(options={"ignore_links": True})
        )
        print("[test_proxy_settings] success:", result.success)
        if result.success:
            print("HTML preview:", result.html[:200] if result.html else "")

async def test_ignore_https_errors():
    # Test ignore HTTPS errors with a self-signed or invalid cert domain
    # This is just conceptual, the domain should be one that triggers SSL error.
    # Using a hypothetical URL that fails SSL:
    async with AsyncWebCrawler(
        headless=True,
        verbose=True,
        user_agent="Mozilla/5.0",
        ignore_https_errors=True,
        use_managed_browser=False,
        use_persistent_context=False,
    ) as crawler:
        result = await crawler.arun(
            url='https://self-signed.badssl.com/',
            cache_mode=CacheMode.BYPASS,
            markdown_generator=DefaultMarkdownGenerator(options={"ignore_links": True})
        )
        print("[test_ignore_https_errors] success:", result.success)

async def main():
    print("Running tests...")
    # await test_default_headless()
    # await test_managed_browser_persistent()
    # await test_session_reuse()
    # await test_magic_mode()
    # await test_proxy_settings()
    await test_ignore_https_errors()

if __name__ == "__main__":
    asyncio.run(main())

```

## tests/async/test_screenshot.py
```
import os
import sys
import pytest
import asyncio
import base64
from PIL import Image
import io

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from crawl4ai.async_webcrawler import AsyncWebCrawler

@pytest.mark.asyncio
async def test_basic_screenshot():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://example.com"  # A static website
        result = await crawler.arun(url=url, bypass_cache=True, screenshot=True)
        
        assert result.success
        assert result.screenshot is not None
        
        # Verify the screenshot is a valid image
        image_data = base64.b64decode(result.screenshot)
        image = Image.open(io.BytesIO(image_data))
        assert image.format == "PNG"

@pytest.mark.asyncio
async def test_screenshot_with_wait_for():
    async with AsyncWebCrawler(verbose=True) as crawler:
        # Using a website with dynamic content
        url = "https://www.youtube.com"
        wait_for = "css:#content"  # Wait for the main content to load
        
        result = await crawler.arun(
            url=url, 
            bypass_cache=True, 
            screenshot=True, 
            wait_for=wait_for
        )
        
        assert result.success
        assert result.screenshot is not None
        
        # Verify the screenshot is a valid image
        image_data = base64.b64decode(result.screenshot)
        image = Image.open(io.BytesIO(image_data))
        assert image.format == "PNG"
        
        # You might want to add more specific checks here, like image dimensions
        # or even use image recognition to verify certain elements are present

@pytest.mark.asyncio
async def test_screenshot_with_js_wait_for():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.amazon.com"
        wait_for = "js:() => document.querySelector('#nav-logo-sprites') !== null"
        
        result = await crawler.arun(
            url=url, 
            bypass_cache=True, 
            screenshot=True, 
            wait_for=wait_for
        )
        
        assert result.success
        assert result.screenshot is not None
        
        image_data = base64.b64decode(result.screenshot)
        image = Image.open(io.BytesIO(image_data))
        assert image.format == "PNG"

@pytest.mark.asyncio
async def test_screenshot_without_wait_for():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nytimes.com"  # A website with lots of dynamic content
        
        result = await crawler.arun(url=url, bypass_cache=True, screenshot=True)
        
        assert result.success
        assert result.screenshot is not None
        
        image_data = base64.b64decode(result.screenshot)
        image = Image.open(io.BytesIO(image_data))
        assert image.format == "PNG"

@pytest.mark.asyncio
async def test_screenshot_comparison():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.reddit.com"
        wait_for = "css:#SHORTCUT_FOCUSABLE_DIV"
        
        # Take screenshot without wait_for
        result_without_wait = await crawler.arun(
            url=url, 
            bypass_cache=True, 
            screenshot=True
        )
        
        # Take screenshot with wait_for
        result_with_wait = await crawler.arun(
            url=url, 
            bypass_cache=True, 
            screenshot=True, 
            wait_for=wait_for
        )
        
        assert result_without_wait.success and result_with_wait.success
        assert result_without_wait.screenshot is not None
        assert result_with_wait.screenshot is not None
        
        # Compare the two screenshots
        image_without_wait = Image.open(io.BytesIO(base64.b64decode(result_without_wait.screenshot)))
        image_with_wait = Image.open(io.BytesIO(base64.b64decode(result_with_wait.screenshot)))
        
        # This is a simple size comparison. In a real-world scenario, you might want to use
        # more sophisticated image comparison techniques.
        assert image_with_wait.size[0] >= image_without_wait.size[0]
        assert image_with_wait.size[1] >= image_without_wait.size[1]

# Entry point for debugging
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## tests/async/test_parameters_and_options.py
```
import os
import sys
import pytest
import asyncio
import json

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from crawl4ai.async_webcrawler import AsyncWebCrawler

@pytest.mark.asyncio
async def test_word_count_threshold():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        result_no_threshold = await crawler.arun(url=url, word_count_threshold=0, bypass_cache=True)
        result_with_threshold = await crawler.arun(url=url, word_count_threshold=50, bypass_cache=True)
        
        assert len(result_no_threshold.markdown) > len(result_with_threshold.markdown)

@pytest.mark.asyncio
async def test_css_selector():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        css_selector = "h1, h2, h3"
        result = await crawler.arun(url=url, css_selector=css_selector, bypass_cache=True)
        
        assert result.success
        assert "<h1" in result.cleaned_html or "<h2" in result.cleaned_html or "<h3" in result.cleaned_html

@pytest.mark.asyncio
async def test_javascript_execution():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"

        # Crawl without JS
        result_without_more = await crawler.arun(url=url, bypass_cache=True)
        
        js_code = ["const loadMoreButton = Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('Load More')); loadMoreButton && loadMoreButton.click();"]
        result_with_more = await crawler.arun(url=url, js=js_code, bypass_cache=True)
        
        assert result_with_more.success
        assert len(result_with_more.markdown) > len(result_without_more.markdown)

@pytest.mark.asyncio
async def test_screenshot():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        result = await crawler.arun(url=url, screenshot=True, bypass_cache=True)
        
        assert result.success
        assert result.screenshot
        assert isinstance(result.screenshot, str)  # Should be a base64 encoded string

@pytest.mark.asyncio
async def test_custom_user_agent():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        custom_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Crawl4AI/1.0"
        result = await crawler.arun(url=url, user_agent=custom_user_agent, bypass_cache=True)
        
        assert result.success
        # Note: We can't directly verify the user agent in the result, but we can check if the crawl was successful

@pytest.mark.asyncio
async def test_extract_media_and_links():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        result = await crawler.arun(url=url, bypass_cache=True)
        
        assert result.success
        assert result.media
        assert isinstance(result.media, dict)
        assert 'images' in result.media
        assert result.links
        assert isinstance(result.links, dict)
        assert 'internal' in result.links and 'external' in result.links

@pytest.mark.asyncio
async def test_metadata_extraction():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        result = await crawler.arun(url=url, bypass_cache=True)
        
        assert result.success
        assert result.metadata
        assert isinstance(result.metadata, dict)
        # Check for common metadata fields
        assert any(key in result.metadata for key in ['title', 'description', 'keywords'])

# Entry point for debugging
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## tests/async/test_error_handling.py
```
# import os
# import sys
# import pytest
# import asyncio

# # Add the parent directory to the Python path
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(parent_dir)

# from crawl4ai.async_webcrawler import AsyncWebCrawler
# from crawl4ai.utils import InvalidCSSSelectorError

# class AsyncCrawlerWrapper:
#     def __init__(self):
#         self.crawler = None

#     async def setup(self):
#         self.crawler = AsyncWebCrawler(verbose=True)
#         await self.crawler.awarmup()

#     async def cleanup(self):
#         if self.crawler:
#             await self.crawler.aclear_cache()

# @pytest.fixture(scope="module")
# def crawler_wrapper():
#     wrapper = AsyncCrawlerWrapper()
#     asyncio.get_event_loop().run_until_complete(wrapper.setup())
#     yield wrapper
#     asyncio.get_event_loop().run_until_complete(wrapper.cleanup())

# @pytest.mark.asyncio
# async def test_network_error(crawler_wrapper):
#     url = "https://www.nonexistentwebsite123456789.com"
#     result = await crawler_wrapper.crawler.arun(url=url, bypass_cache=True)
#     assert not result.success
#     assert "Failed to crawl" in result.error_message

# # @pytest.mark.asyncio
# # async def test_timeout_error(crawler_wrapper):
# #     # Simulating a timeout by using a very short timeout value
# #     url = "https://www.nbcnews.com/business"
# #     result = await crawler_wrapper.crawler.arun(url=url, bypass_cache=True, timeout=0.001)
# #     assert not result.success
# #     assert "timeout" in result.error_message.lower()

# # @pytest.mark.asyncio
# # async def test_invalid_css_selector(crawler_wrapper):
# #     url = "https://www.nbcnews.com/business"
# #     with pytest.raises(InvalidCSSSelectorError):
# #         await crawler_wrapper.crawler.arun(url=url, bypass_cache=True, css_selector="invalid>>selector")

# # @pytest.mark.asyncio
# # async def test_js_execution_error(crawler_wrapper):
# #     url = "https://www.nbcnews.com/business"
# #     invalid_js = "This is not valid JavaScript code;"
# #     result = await crawler_wrapper.crawler.arun(url=url, bypass_cache=True, js=invalid_js)
# #     assert not result.success
# #     assert "JavaScript" in result.error_message

# # @pytest.mark.asyncio
# # async def test_empty_page(crawler_wrapper):
# #     # Use a URL that typically returns an empty page
# #     url = "http://example.com/empty"
# #     result = await crawler_wrapper.crawler.arun(url=url, bypass_cache=True)
# #     assert result.success  # The crawl itself should succeed
# #     assert not result.markdown.strip()  # The markdown content should be empty or just whitespace

# # @pytest.mark.asyncio
# # async def test_rate_limiting(crawler_wrapper):
# #     # Simulate rate limiting by making multiple rapid requests
# #     url = "https://www.nbcnews.com/business"
# #     results = await asyncio.gather(*[crawler_wrapper.crawler.arun(url=url, bypass_cache=True) for _ in range(10)])
# #     assert any(not result.success and "rate limit" in result.error_message.lower() for result in results)

# # Entry point for debugging
# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])
```

## tests/async/test_0.4.2_config_params.py
```
import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

import asyncio
from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig      
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from crawl4ai.chunking_strategy import RegexChunking
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# Category 1: Browser Configuration Tests
async def test_browser_config_object():
    """Test the new BrowserConfig object with various browser settings"""
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=False,
        viewport_width=1920,
        viewport_height=1080,
        use_managed_browser=True,
        user_agent_mode="random",
        user_agent_generator_config={"device_type": "desktop", "os_type": "windows"}
    )
    
    async with AsyncWebCrawler(config=browser_config, verbose=True) as crawler:
        result = await crawler.arun('https://example.com', cache_mode=CacheMode.BYPASS)
        assert result.success, "Browser config crawl failed"
        assert len(result.html) > 0, "No HTML content retrieved"

async def test_browser_performance_config():
    """Test browser configurations focused on performance"""
    browser_config = BrowserConfig(
        text_only=True,
        light_mode=True,
        extra_args=['--disable-gpu', '--disable-software-rasterizer'],
        ignore_https_errors=True,
        java_script_enabled=False
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun('https://example.com')
        assert result.success, "Performance optimized crawl failed"
        assert result.status_code == 200, "Unexpected status code"

# Category 2: Content Processing Tests
async def test_content_extraction_config():
    """Test content extraction with various strategies"""
    crawler_config = CrawlerRunConfig(
        word_count_threshold=300,
        extraction_strategy=JsonCssExtractionStrategy(
            schema={
                "name": "article",
                "baseSelector": "div",
                "fields": [{
                    "name": "title",
                    "selector": "h1",
                    "type": "text"
                }]
            }
        ),
        chunking_strategy=RegexChunking(),
        content_filter=PruningContentFilter()
    )
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            'https://example.com/article',
            config=crawler_config
        )
        assert result.extracted_content is not None, "Content extraction failed"
        assert 'title' in result.extracted_content, "Missing expected content field"

# Category 3: Cache and Session Management Tests
async def test_cache_and_session_management():
    """Test different cache modes and session handling"""
    browser_config = BrowserConfig(use_persistent_context=True)
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.WRITE_ONLY,
        process_iframes=True,
        remove_overlay_elements=True
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # First request - should write to cache
        result1 = await crawler.arun(
            'https://example.com',
            config=crawler_config
        )
        
        # Second request - should use fresh fetch due to WRITE_ONLY mode
        result2 = await crawler.arun(
            'https://example.com',
            config=crawler_config
        )
        
        assert result1.success and result2.success, "Cache mode crawl failed"
        assert result1.html == result2.html, "Inconsistent results between requests"

# Category 4: Media Handling Tests
async def test_media_handling_config():
    """Test configurations related to media handling"""
    # Get the base path for home directroy ~/.crawl4ai/downloads, make sure it exists
    os.makedirs(os.path.expanduser("~/.crawl4ai/downloads"), exist_ok=True)
    browser_config = BrowserConfig(
        viewport_width=1920,
        viewport_height=1080,
        accept_downloads=True,
        downloads_path= os.path.expanduser("~/.crawl4ai/downloads")
    )
    crawler_config = CrawlerRunConfig(
        screenshot=True,
        pdf=True,
        adjust_viewport_to_content=True,
        wait_for_images=True,
        screenshot_height_threshold=20000
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            'https://example.com',
            config=crawler_config
        )
        assert result.screenshot is not None, "Screenshot capture failed"
        assert result.pdf is not None, "PDF generation failed"

# Category 5: Anti-Bot and Site Interaction Tests
async def test_antibot_config():
    """Test configurations for handling anti-bot measures"""
    crawler_config = CrawlerRunConfig(
        simulate_user=True,
        override_navigator=True,
        magic=True,
        wait_for="js:()=>document.querySelector('body')",
        delay_before_return_html=1.0,
        log_console=True,
        cache_mode=CacheMode.BYPASS
    )
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            'https://example.com',
            config=crawler_config
        )
        assert result.success, "Anti-bot measure handling failed"

# Category 6: Parallel Processing Tests
async def test_parallel_processing():
    """Test parallel processing capabilities"""
    crawler_config = CrawlerRunConfig(
        mean_delay=0.5,
        max_range=1.0,
        semaphore_count=5
    )
    
    urls = [
        'https://example.com/1',
        'https://example.com/2',
        'https://example.com/3'
    ]
    
    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun_many(
            urls,
            config=crawler_config
        )
        assert len(results) == len(urls), "Not all URLs were processed"
        assert all(r.success for r in results), "Some parallel requests failed"

# Category 7: Backwards Compatibility Tests
async def test_legacy_parameter_support():
    """Test that legacy parameters still work"""
    async with AsyncWebCrawler(
        headless=True,
        browser_type="chromium",
        viewport_width=1024,
        viewport_height=768
    ) as crawler:
        result = await crawler.arun(
            'https://example.com',
            screenshot=True,
            word_count_threshold=200,
            bypass_cache=True,
            css_selector=".main-content"
        )
        assert result.success, "Legacy parameter support failed"

# Category 8: Mixed Configuration Tests
async def test_mixed_config_usage():
    """Test mixing new config objects with legacy parameters"""
    browser_config = BrowserConfig(headless=True)
    crawler_config = CrawlerRunConfig(screenshot=True)
    
    async with AsyncWebCrawler(
        config=browser_config,
        verbose=True  # legacy parameter
    ) as crawler:
        result = await crawler.arun(
            'https://example.com',
            config=crawler_config,
            cache_mode=CacheMode.BYPASS,  # legacy parameter
            css_selector="body"  # legacy parameter
        )
        assert result.success, "Mixed configuration usage failed"

if __name__ == "__main__":
    async def run_tests():
        test_functions = [
            test_browser_config_object,
            # test_browser_performance_config,
            # test_content_extraction_config,
            # test_cache_and_session_management,
            # test_media_handling_config,
            # test_antibot_config,
            # test_parallel_processing,
            # test_legacy_parameter_support,
            # test_mixed_config_usage
        ]
        
        for test in test_functions:
            print(f"\nRunning {test.__name__}...")
            try:
                await test()
                print(f"✓ {test.__name__} passed")
            except AssertionError as e:
                print(f"✗ {test.__name__} failed: {str(e)}")
            except Exception as e:
                print(f"✗ {test.__name__} error: {str(e)}")
    
    asyncio.run(run_tests())
```

## tests/async/test_content_extraction.py
```
import os
import sys
import pytest
import asyncio
import json

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from crawl4ai.async_webcrawler import AsyncWebCrawler

@pytest.mark.asyncio
async def test_extract_markdown():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        result = await crawler.arun(url=url, bypass_cache=True)
        assert result.success
        assert result.markdown
        assert isinstance(result.markdown, str)
        assert len(result.markdown) > 0

@pytest.mark.asyncio
async def test_extract_cleaned_html():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        result = await crawler.arun(url=url, bypass_cache=True)
        assert result.success
        assert result.cleaned_html
        assert isinstance(result.cleaned_html, str)
        assert len(result.cleaned_html) > 0

@pytest.mark.asyncio
async def test_extract_media():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        result = await crawler.arun(url=url, bypass_cache=True)
        assert result.success
        assert result.media
        media = result.media
        assert isinstance(media, dict)
        assert "images" in media
        assert isinstance(media["images"], list)
        for image in media["images"]:
            assert "src" in image
            assert "alt" in image
            assert "type" in image

@pytest.mark.asyncio
async def test_extract_links():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        result = await crawler.arun(url=url, bypass_cache=True)
        assert result.success
        assert result.links
        links = result.links
        assert isinstance(links, dict)
        assert "internal" in links
        assert "external" in links
        assert isinstance(links["internal"], list)
        assert isinstance(links["external"], list)
        for link in links["internal"] + links["external"]:
            assert "href" in link
            assert "text" in link

@pytest.mark.asyncio
async def test_extract_metadata():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        result = await crawler.arun(url=url, bypass_cache=True)
        assert result.success
        assert result.metadata
        metadata = result.metadata
        assert isinstance(metadata, dict)
        assert "title" in metadata
        assert isinstance(metadata["title"], str)

@pytest.mark.asyncio
async def test_css_selector_extraction():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        css_selector = "h1, h2, h3"
        result = await crawler.arun(url=url, bypass_cache=True, css_selector=css_selector)
        assert result.success
        assert result.markdown
        assert all(heading in result.markdown for heading in ["#", "##", "###"])

# Entry point for debugging
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## tests/async/test_chunking_and_extraction_strategies.py
```
import os
import sys
import pytest
import asyncio
import json

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from crawl4ai.async_webcrawler import AsyncWebCrawler
from crawl4ai.chunking_strategy import RegexChunking, NlpSentenceChunking
from crawl4ai.extraction_strategy import CosineStrategy, LLMExtractionStrategy

@pytest.mark.asyncio
async def test_regex_chunking():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        chunking_strategy = RegexChunking(patterns=["\n\n"])
        result = await crawler.arun(
            url=url,
            chunking_strategy=chunking_strategy,
            bypass_cache=True
        )
        assert result.success
        assert result.extracted_content
        chunks = json.loads(result.extracted_content)
        assert len(chunks) > 1  # Ensure multiple chunks were created

# @pytest.mark.asyncio
# async def test_cosine_strategy():
#     async with AsyncWebCrawler(verbose=True) as crawler:
#         url = "https://www.nbcnews.com/business"
#         extraction_strategy = CosineStrategy(word_count_threshold=10, max_dist=0.2, linkage_method="ward", top_k=3, sim_threshold=0.3)
#         result = await crawler.arun(
#             url=url,
#             extraction_strategy=extraction_strategy,
#             bypass_cache=True
#         )
#         assert result.success
#         assert result.extracted_content
#         extracted_data = json.loads(result.extracted_content)
#         assert len(extracted_data) > 0
#         assert all('tags' in item for item in extracted_data)

@pytest.mark.asyncio
async def test_llm_extraction_strategy():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        extraction_strategy = LLMExtractionStrategy(
            provider="openai/gpt-4o-mini",
            api_token=os.getenv('OPENAI_API_KEY'),
            instruction="Extract only content related to technology"
        )
        result = await crawler.arun(
            url=url,
            extraction_strategy=extraction_strategy,
            bypass_cache=True
        )
        assert result.success
        assert result.extracted_content
        extracted_data = json.loads(result.extracted_content)
        assert len(extracted_data) > 0
        assert all('content' in item for item in extracted_data)

# @pytest.mark.asyncio
# async def test_combined_chunking_and_extraction():
#     async with AsyncWebCrawler(verbose=True) as crawler:
#         url = "https://www.nbcnews.com/business"
#         chunking_strategy = RegexChunking(patterns=["\n\n"])
#         extraction_strategy = CosineStrategy(word_count_threshold=10, max_dist=0.2, linkage_method="ward", top_k=3, sim_threshold=0.3)
#         result = await crawler.arun(
#             url=url,
#             chunking_strategy=chunking_strategy,
#             extraction_strategy=extraction_strategy,
#             bypass_cache=True
#         )
#         assert result.success
#         assert result.extracted_content
#         extracted_data = json.loads(result.extracted_content)
#         assert len(extracted_data) > 0
#         assert all('tags' in item for item in extracted_data)
#         assert all('content' in item for item in extracted_data)

# Entry point for debugging
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## tests/async/test_database_operations.py
```
import os
import sys
import pytest
import asyncio
import json

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from crawl4ai.async_webcrawler import AsyncWebCrawler

@pytest.mark.asyncio
async def test_cache_url():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.example.com"
        # First run to cache the URL
        result1 = await crawler.arun(url=url, bypass_cache=True)
        assert result1.success

        # Second run to retrieve from cache
        result2 = await crawler.arun(url=url, bypass_cache=False)
        assert result2.success
        assert result2.html == result1.html

@pytest.mark.asyncio
async def test_bypass_cache():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.python.org"
        # First run to cache the URL
        result1 = await crawler.arun(url=url, bypass_cache=True)
        assert result1.success

        # Second run bypassing cache
        result2 = await crawler.arun(url=url, bypass_cache=True)
        assert result2.success
        assert result2.html != result1.html  # Content might be different due to dynamic nature of websites

@pytest.mark.asyncio
async def test_cache_size():
    async with AsyncWebCrawler(verbose=True) as crawler:
        initial_size = await crawler.aget_cache_size()
        
        url = "https://www.nbcnews.com/business"
        await crawler.arun(url=url, bypass_cache=True)
        
        new_size = await crawler.aget_cache_size()
        assert new_size == initial_size + 1

@pytest.mark.asyncio
async def test_clear_cache():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.example.org"
        await crawler.arun(url=url, bypass_cache=True)
        
        initial_size = await crawler.aget_cache_size()
        assert initial_size > 0

        await crawler.aclear_cache()
        new_size = await crawler.aget_cache_size()
        assert new_size == 0

@pytest.mark.asyncio
async def test_flush_cache():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.example.net"
        await crawler.arun(url=url, bypass_cache=True)
        
        initial_size = await crawler.aget_cache_size()
        assert initial_size > 0

        await crawler.aflush_cache()
        new_size = await crawler.aget_cache_size()
        assert new_size == 0

        # Try to retrieve the previously cached URL
        result = await crawler.arun(url=url, bypass_cache=False)
        assert result.success  # The crawler should still succeed, but it will fetch the content anew

# Entry point for debugging
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## tests/async/test_performance.py
```
import os
import sys
import pytest
import asyncio
import time

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from crawl4ai.async_webcrawler import AsyncWebCrawler

@pytest.mark.asyncio
async def test_crawl_speed():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        start_time = time.time()
        result = await crawler.arun(url=url, bypass_cache=True)
        end_time = time.time()
        
        assert result.success
        crawl_time = end_time - start_time
        print(f"Crawl time: {crawl_time:.2f} seconds")
        
        assert crawl_time < 10, f"Crawl took too long: {crawl_time:.2f} seconds"

@pytest.mark.asyncio
async def test_concurrent_crawling_performance():
    async with AsyncWebCrawler(verbose=True) as crawler:
        urls = [
            "https://www.nbcnews.com/business",
            "https://www.example.com",
            "https://www.python.org",
            "https://www.github.com",
            "https://www.stackoverflow.com"
        ]
        
        start_time = time.time()
        results = await crawler.arun_many(urls=urls, bypass_cache=True)
        end_time = time.time()
        
        total_time = end_time - start_time
        print(f"Total time for concurrent crawling: {total_time:.2f} seconds")
        
        assert all(result.success for result in results)
        assert len(results) == len(urls)
        
        assert total_time < len(urls) * 5, f"Concurrent crawling not significantly faster: {total_time:.2f} seconds"

@pytest.mark.asyncio
async def test_crawl_speed_with_caching():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        
        start_time = time.time()
        result1 = await crawler.arun(url=url, bypass_cache=True)
        end_time = time.time()
        first_crawl_time = end_time - start_time
        
        start_time = time.time()
        result2 = await crawler.arun(url=url, bypass_cache=False)
        end_time = time.time()
        second_crawl_time = end_time - start_time
        
        assert result1.success and result2.success
        print(f"First crawl time: {first_crawl_time:.2f} seconds")
        print(f"Second crawl time (cached): {second_crawl_time:.2f} seconds")
        
        assert second_crawl_time < first_crawl_time / 2, "Cached crawl not significantly faster"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## tests/async/test_basic_crawling.py
```
import os
import sys
import pytest
import asyncio
import time

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from crawl4ai.async_webcrawler import AsyncWebCrawler

@pytest.mark.asyncio
async def test_successful_crawl():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        result = await crawler.arun(url=url, bypass_cache=True)
        assert result.success
        assert result.url == url
        assert result.html
        assert result.markdown
        assert result.cleaned_html

@pytest.mark.asyncio
async def test_invalid_url():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.invalidurl12345.com"
        result = await crawler.arun(url=url, bypass_cache=True)
        assert not result.success
        assert result.error_message

@pytest.mark.asyncio
async def test_multiple_urls():
    async with AsyncWebCrawler(verbose=True) as crawler:
        urls = [
            "https://www.nbcnews.com/business",
            "https://www.example.com",
            "https://www.python.org"
        ]
        results = await crawler.arun_many(urls=urls, bypass_cache=True)
        assert len(results) == len(urls)
        assert all(result.success for result in results)
        assert all(result.html for result in results)

@pytest.mark.asyncio
async def test_javascript_execution():
    async with AsyncWebCrawler(verbose=True) as crawler:
        js_code = "document.body.innerHTML = '<h1>Modified by JS</h1>';"
        url = "https://www.example.com"
        result = await crawler.arun(url=url, bypass_cache=True, js_code=js_code)
        assert result.success
        assert "<h1>Modified by JS</h1>" in result.html

@pytest.mark.asyncio
async def test_concurrent_crawling_performance():
    async with AsyncWebCrawler(verbose=True) as crawler:
        urls = [
            "https://www.nbcnews.com/business",
            "https://www.example.com",
            "https://www.python.org",
            "https://www.github.com",
            "https://www.stackoverflow.com"
        ]
        
        start_time = time.time()
        results = await crawler.arun_many(urls=urls, bypass_cache=True)
        end_time = time.time()
        
        total_time = end_time - start_time
        print(f"Total time for concurrent crawling: {total_time:.2f} seconds")
        
        assert all(result.success for result in results)
        assert len(results) == len(urls)
        
        # Assert that concurrent crawling is faster than sequential
        # This multiplier may need adjustment based on the number of URLs and their complexity
        assert total_time < len(urls) * 5, f"Concurrent crawling not significantly faster: {total_time:.2f} seconds"

# Entry point for debugging
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## tests/async/test_async_doanloader.py
```
import os
import sys
import asyncio
import shutil
from typing import List
import tempfile
import time

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from crawl4ai.async_webcrawler import AsyncWebCrawler

class TestDownloads:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="crawl4ai_test_")
        self.download_dir = os.path.join(self.temp_dir, "downloads")
        os.makedirs(self.download_dir, exist_ok=True)
        self.results: List[str] = []
        
    def cleanup(self):
        shutil.rmtree(self.temp_dir)
        
    def log_result(self, test_name: str, success: bool, message: str = ""):
        result = f"{'✅' if success else '❌'} {test_name}: {message}"
        self.results.append(result)
        print(result)
        
    async def test_basic_download(self):
        """Test basic file download functionality"""
        try:
            async with AsyncWebCrawler(
                accept_downloads=True,
                downloads_path=self.download_dir,
                verbose=True
            ) as crawler:
                # Python.org downloads page typically has stable download links
                result = await crawler.arun(
                    url="https://www.python.org/downloads/",
                    js_code="""
                    // Click first download link
                    const downloadLink = document.querySelector('a[href$=".exe"]');
                    if (downloadLink) downloadLink.click();
                    """
                )
                
                success = result.downloaded_files is not None and len(result.downloaded_files) > 0
                self.log_result(
                    "Basic Download",
                    success,
                    f"Downloaded {len(result.downloaded_files or [])} files" if success else "No files downloaded"
                )
        except Exception as e:
            self.log_result("Basic Download", False, str(e))

    async def test_persistent_context_download(self):
        """Test downloads with persistent context"""
        try:
            user_data_dir = os.path.join(self.temp_dir, "user_data")
            os.makedirs(user_data_dir, exist_ok=True)
            
            async with AsyncWebCrawler(
                accept_downloads=True,
                downloads_path=self.download_dir,
                use_persistent_context=True,
                user_data_dir=user_data_dir,
                verbose=True
            ) as crawler:
                result = await crawler.arun(
                    url="https://www.python.org/downloads/",
                    js_code="""
                    const downloadLink = document.querySelector('a[href$=".exe"]');
                    if (downloadLink) downloadLink.click();
                    """
                )
                
                success = result.downloaded_files is not None and len(result.downloaded_files) > 0
                self.log_result(
                    "Persistent Context Download",
                    success,
                    f"Downloaded {len(result.downloaded_files or [])} files" if success else "No files downloaded"
                )
        except Exception as e:
            self.log_result("Persistent Context Download", False, str(e))

    async def test_multiple_downloads(self):
        """Test multiple simultaneous downloads"""
        try:
            async with AsyncWebCrawler(
                accept_downloads=True,
                downloads_path=self.download_dir,
                verbose=True
            ) as crawler:
                result = await crawler.arun(
                    url="https://www.python.org/downloads/",
                    js_code="""
                    // Click multiple download links
                    const downloadLinks = document.querySelectorAll('a[href$=".exe"]');
                    downloadLinks.forEach(link => link.click());
                    """
                )
                
                success = result.downloaded_files is not None and len(result.downloaded_files) > 1
                self.log_result(
                    "Multiple Downloads",
                    success,
                    f"Downloaded {len(result.downloaded_files or [])} files" if success else "Not enough files downloaded"
                )
        except Exception as e:
            self.log_result("Multiple Downloads", False, str(e))

    async def test_different_browsers(self):
        """Test downloads across different browser types"""
        browsers = ["chromium", "firefox", "webkit"]
        
        for browser_type in browsers:
            try:
                async with AsyncWebCrawler(
                    accept_downloads=True,
                    downloads_path=self.download_dir,
                    browser_type=browser_type,
                    verbose=True
                ) as crawler:
                    result = await crawler.arun(
                        url="https://www.python.org/downloads/",
                        js_code="""
                        const downloadLink = document.querySelector('a[href$=".exe"]');
                        if (downloadLink) downloadLink.click();
                        """
                    )
                    
                    success = result.downloaded_files is not None and len(result.downloaded_files) > 0
                    self.log_result(
                        f"{browser_type.title()} Download",
                        success,
                        f"Downloaded {len(result.downloaded_files or [])} files" if success else "No files downloaded"
                    )
            except Exception as e:
                self.log_result(f"{browser_type.title()} Download", False, str(e))

    async def test_edge_cases(self):
        """Test various edge cases"""
        
        # Test 1: Downloads without specifying download path
        try:
            async with AsyncWebCrawler(
                accept_downloads=True,
                verbose=True
            ) as crawler:
                result = await crawler.arun(
                    url="https://www.python.org/downloads/",
                    js_code="document.querySelector('a[href$=\".exe\"]').click()"
                )
                self.log_result(
                    "Default Download Path",
                    True,
                    f"Downloaded to default path: {result.downloaded_files[0] if result.downloaded_files else 'None'}"
                )
        except Exception as e:
            self.log_result("Default Download Path", False, str(e))

        # Test 2: Downloads with invalid path
        try:
            async with AsyncWebCrawler(
                accept_downloads=True,
                downloads_path="/invalid/path/that/doesnt/exist",
                verbose=True
            ) as crawler:
                result = await crawler.arun(
                    url="https://www.python.org/downloads/",
                    js_code="document.querySelector('a[href$=\".exe\"]').click()"
                )
                self.log_result("Invalid Download Path", False, "Should have raised an error")
        except Exception as e:
            self.log_result("Invalid Download Path", True, "Correctly handled invalid path")

        # Test 3: Download with accept_downloads=False
        try:
            async with AsyncWebCrawler(
                accept_downloads=False,
                verbose=True
            ) as crawler:
                result = await crawler.arun(
                    url="https://www.python.org/downloads/",
                    js_code="document.querySelector('a[href$=\".exe\"]').click()"
                )
                success = result.downloaded_files is None
                self.log_result(
                    "Disabled Downloads",
                    success,
                    "Correctly ignored downloads" if success else "Unexpectedly downloaded files"
                )
        except Exception as e:
            self.log_result("Disabled Downloads", False, str(e))

    async def run_all_tests(self):
        """Run all test cases"""
        print("\n🧪 Running Download Tests...\n")
        
        test_methods = [
            self.test_basic_download,
            self.test_persistent_context_download,
            self.test_multiple_downloads,
            self.test_different_browsers,
            self.test_edge_cases
        ]
        
        for test in test_methods:
            print(f"\n📝 Running {test.__doc__}...")
            await test()
            await asyncio.sleep(2)  # Brief pause between tests
            
        print("\n📊 Test Results Summary:")
        for result in self.results:
            print(result)
            
        successes = len([r for r in self.results if '✅' in r])
        total = len(self.results)
        print(f"\nTotal: {successes}/{total} tests passed")
        
        self.cleanup()

async def main():
    tester = TestDownloads()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
```

## tests/async/test_caching.py
```
import os
import sys
import pytest
import asyncio

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from crawl4ai.async_webcrawler import AsyncWebCrawler

@pytest.mark.asyncio
async def test_caching():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        
        # First crawl (should not use cache)
        start_time = asyncio.get_event_loop().time()
        result1 = await crawler.arun(url=url, bypass_cache=True)
        end_time = asyncio.get_event_loop().time()
        time_taken1 = end_time - start_time
        
        assert result1.success
        
        # Second crawl (should use cache)
        start_time = asyncio.get_event_loop().time()
        result2 = await crawler.arun(url=url, bypass_cache=False)
        end_time = asyncio.get_event_loop().time()
        time_taken2 = end_time - start_time
        
        assert result2.success
        assert time_taken2 < time_taken1  # Cached result should be faster

@pytest.mark.asyncio
async def test_bypass_cache():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        
        # First crawl
        result1 = await crawler.arun(url=url, bypass_cache=False)
        assert result1.success
        
        # Second crawl with bypass_cache=True
        result2 = await crawler.arun(url=url, bypass_cache=True)
        assert result2.success
        
        # Content should be different (or at least, not guaranteed to be the same)
        assert result1.html != result2.html or result1.markdown != result2.markdown

@pytest.mark.asyncio
async def test_clear_cache():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        
        # Crawl and cache
        await crawler.arun(url=url, bypass_cache=False)
        
        # Clear cache
        await crawler.aclear_cache()
        
        # Check cache size
        cache_size = await crawler.aget_cache_size()
        assert cache_size == 0

@pytest.mark.asyncio
async def test_flush_cache():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.nbcnews.com/business"
        
        # Crawl and cache
        await crawler.arun(url=url, bypass_cache=False)
        
        # Flush cache
        await crawler.aflush_cache()
        
        # Check cache size
        cache_size = await crawler.aget_cache_size()
        assert cache_size == 0

# Entry point for debugging
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## tests/async/test_edge_cases.py
```
import os
import re
import sys
import pytest
import json
from bs4 import BeautifulSoup
import asyncio
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from crawl4ai.async_webcrawler import AsyncWebCrawler

# @pytest.mark.asyncio
# async def test_large_content_page():
#     async with AsyncWebCrawler(verbose=True) as crawler:
#         url = "https://en.wikipedia.org/wiki/List_of_largest_known_stars"  # A page with a large table
#         result = await crawler.arun(url=url, bypass_cache=True)
#         assert result.success
#         assert len(result.html) > 1000000  # Expecting more than 1MB of content

# @pytest.mark.asyncio
# async def test_minimal_content_page():
#     async with AsyncWebCrawler(verbose=True) as crawler:
#         url = "https://example.com"  # A very simple page
#         result = await crawler.arun(url=url, bypass_cache=True)
#         assert result.success
#         assert len(result.html) < 10000  # Expecting less than 10KB of content

# @pytest.mark.asyncio
# async def test_single_page_application():
#     async with AsyncWebCrawler(verbose=True) as crawler:
#         url = "https://reactjs.org/"  # React's website is a SPA
#         result = await crawler.arun(url=url, bypass_cache=True)
#         assert result.success
#         assert "react" in result.html.lower()

# @pytest.mark.asyncio
# async def test_page_with_infinite_scroll():
#     async with AsyncWebCrawler(verbose=True) as crawler:
#         url = "https://news.ycombinator.com/"  # Hacker News has infinite scroll
#         result = await crawler.arun(url=url, bypass_cache=True)
#         assert result.success
#         assert "hacker news" in result.html.lower()

# @pytest.mark.asyncio
# async def test_page_with_heavy_javascript():
#     async with AsyncWebCrawler(verbose=True) as crawler:
#         url = "https://www.airbnb.com/"  # Airbnb uses a lot of JavaScript
#         result = await crawler.arun(url=url, bypass_cache=True)
#         assert result.success
#         assert "airbnb" in result.html.lower()

# @pytest.mark.asyncio
# async def test_page_with_mixed_content():
#     async with AsyncWebCrawler(verbose=True) as crawler:
#         url = "https://github.com/"  # GitHub has a mix of static and dynamic content
#         result = await crawler.arun(url=url, bypass_cache=True)
#         assert result.success
#         assert "github" in result.html.lower()

# Add this test to your existing test file
@pytest.mark.asyncio
async def test_typescript_commits_multi_page():
    first_commit = ""
    async def on_execution_started(page):
        nonlocal first_commit 
        try:
            # Check if the page firct commit h4 text is different from the first commit (use document.querySelector('li.Box-sc-g0xbh4-0 h4'))
            while True:
                await page.wait_for_selector('li.Box-sc-g0xbh4-0 h4')
                commit = await page.query_selector('li.Box-sc-g0xbh4-0 h4')
                commit = await commit.evaluate('(element) => element.textContent')
                commit = re.sub(r'\s+', '', commit)
                if commit and commit != first_commit:
                    first_commit = commit
                    break
                await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Warning: New content didn't appear after JavaScript execution: {e}")


    async with AsyncWebCrawler(verbose=True) as crawler:
        crawler.crawler_strategy.set_hook('on_execution_started', on_execution_started)

        url = "https://github.com/microsoft/TypeScript/commits/main"
        session_id = "typescript_commits_session"
        all_commits = []

        js_next_page = """
        const button = document.querySelector('a[data-testid="pagination-next-button"]');
        if (button) button.click();
        """

        for page in range(3):  # Crawl 3 pages
            result = await crawler.arun(
                url=url,  # Only use URL for the first page
                session_id=session_id,
                css_selector="li.Box-sc-g0xbh4-0",
                js=js_next_page if page > 0 else None,  # Don't click 'next' on the first page
                bypass_cache=True,
                js_only=page > 0  # Use js_only for subsequent pages
            )

            assert result.success, f"Failed to crawl page {page + 1}"

            # Parse the HTML and extract commits
            soup = BeautifulSoup(result.cleaned_html, 'html.parser')
            commits = soup.select("li")
            # Take first commit find h4 extract text
            first_commit = commits[0].find("h4").text
            first_commit = re.sub(r'\s+', '', first_commit)
            all_commits.extend(commits)

            print(f"Page {page + 1}: Found {len(commits)} commits")

        # Clean up the session
        await crawler.crawler_strategy.kill_session(session_id)

        # Assertions
        assert len(all_commits) >= 90, f"Expected at least 90 commits, but got {len(all_commits)}"
        
        print(f"Successfully crawled {len(all_commits)} commits across 3 pages")                      

# Entry point for debugging
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## tests/async/test_crawler_strategy.py
```
import os
import sys
import pytest
import asyncio

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from crawl4ai.async_webcrawler import AsyncWebCrawler
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy

@pytest.mark.asyncio
async def test_custom_user_agent():
    async with AsyncWebCrawler(verbose=True) as crawler:
        custom_user_agent = "MyCustomUserAgent/1.0"
        crawler.crawler_strategy.update_user_agent(custom_user_agent)
        url = "https://httpbin.org/user-agent"
        result = await crawler.arun(url=url, bypass_cache=True)
        assert result.success
        assert custom_user_agent in result.html

@pytest.mark.asyncio
async def test_custom_headers():
    async with AsyncWebCrawler(verbose=True) as crawler:
        custom_headers = {"X-Test-Header": "TestValue"}
        crawler.crawler_strategy.set_custom_headers(custom_headers)
        url = "https://httpbin.org/headers"
        result = await crawler.arun(url=url, bypass_cache=True)
        assert result.success
        assert "X-Test-Header" in result.html
        assert "TestValue" in result.html

@pytest.mark.asyncio
async def test_javascript_execution():
    async with AsyncWebCrawler(verbose=True) as crawler:
        js_code = "document.body.innerHTML = '<h1>Modified by JS</h1>';"
        url = "https://www.example.com"
        result = await crawler.arun(url=url, bypass_cache=True, js_code=js_code)
        assert result.success
        assert "<h1>Modified by JS</h1>" in result.html

@pytest.mark.asyncio
async def test_hook_execution():
    async with AsyncWebCrawler(verbose=True) as crawler:
        async def test_hook(page):
            await page.evaluate("document.body.style.backgroundColor = 'red';")
            return page

        crawler.crawler_strategy.set_hook('after_goto', test_hook)
        url = "https://www.example.com"
        result = await crawler.arun(url=url, bypass_cache=True)
        assert result.success
        assert "background-color: red" in result.html

@pytest.mark.asyncio
async def test_screenshot():
    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://www.example.com"
        result = await crawler.arun(url=url, bypass_cache=True, screenshot=True)
        assert result.success
        assert result.screenshot
        assert isinstance(result.screenshot, str)
        assert len(result.screenshot) > 0

# Entry point for debugging
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## docs/examples/async_webcrawler_multiple_urls_example.py
```
# File: async_webcrawler_multiple_urls_example.py
import os, sys
# append 2 parent directories to sys.path to import crawl4ai
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

import asyncio
from crawl4ai import AsyncWebCrawler

async def main():
    # Initialize the AsyncWebCrawler
    async with AsyncWebCrawler(verbose=True) as crawler:
        # List of URLs to crawl
        urls = [
            "https://example.com",
            "https://python.org",
            "https://github.com",
            "https://stackoverflow.com",
            "https://news.ycombinator.com"
        ]

        # Set up crawling parameters
        word_count_threshold = 100

        # Run the crawling process for multiple URLs
        results = await crawler.arun_many(
            urls=urls,
            word_count_threshold=word_count_threshold,
            bypass_cache=True,
            verbose=True
        )

        # Process the results
        for result in results:
            if result.success:
                print(f"Successfully crawled: {result.url}")
                print(f"Title: {result.metadata.get('title', 'N/A')}")
                print(f"Word count: {len(result.markdown.split())}")
                print(f"Number of links: {len(result.links.get('internal', [])) + len(result.links.get('external', []))}")
                print(f"Number of images: {len(result.media.get('images', []))}")
                print("---")
            else:
                print(f"Failed to crawl: {result.url}")
                print(f"Error: {result.error_message}")
                print("---")

if __name__ == "__main__":
    asyncio.run(main())
```

## docs/examples/quickstart_async.config.py
```
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ['FIRECRAWL_API_KEY'] = "fc-84b370ccfad44beabc686b38f1769692"

import asyncio
import time
import json
import re
from typing import Dict, List
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, CacheMode, BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, LLMExtractionStrategy

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

print("Crawl4AI: Advanced Web Crawling and Data Extraction")
print("GitHub Repository: https://github.com/unclecode/crawl4ai")
print("Twitter: @unclecode")
print("Website: https://crawl4ai.com")

# Basic Example - Simple Crawl
async def simple_crawl():
    print("\n--- Basic Usage ---")
    browser_config = BrowserConfig(headless=True)
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://www.nbcnews.com/business",
            config=crawler_config
        )
        print(result.markdown[:500])

# JavaScript Execution Example
async def simple_example_with_running_js_code():
    print("\n--- Executing JavaScript and Using CSS Selectors ---")
    
    browser_config = BrowserConfig(
        headless=True,
        java_script_enabled=True
    )
    
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        js_code=["const loadMoreButton = Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('Load More')); loadMoreButton && loadMoreButton.click();"],
        # wait_for="() => { return Array.from(document.querySelectorAll('article.tease-card')).length > 10; }"
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://www.nbcnews.com/business",
            config=crawler_config
        )
        print(result.markdown[:500])

# CSS Selector Example
async def simple_example_with_css_selector():
    print("\n--- Using CSS Selectors ---")
    browser_config = BrowserConfig(headless=True)
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        css_selector=".wide-tease-item__description"
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://www.nbcnews.com/business",
            config=crawler_config
        )
        print(result.markdown[:500])

# Proxy Example
async def use_proxy():
    print("\n--- Using a Proxy ---")
    browser_config = BrowserConfig(
        headless=True,
        proxy="http://your-proxy-url:port"
    )
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://www.nbcnews.com/business",
            config=crawler_config
        )
        if result.success:
            print(result.markdown[:500])

# Screenshot Example
async def capture_and_save_screenshot(url: str, output_path: str):
    browser_config = BrowserConfig(headless=True)
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        screenshot=True
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url=url,
            config=crawler_config
        )
        
        if result.success and result.screenshot:
            import base64
            screenshot_data = base64.b64decode(result.screenshot)
            with open(output_path, 'wb') as f:
                f.write(screenshot_data)
            print(f"Screenshot saved successfully to {output_path}")
        else:
            print("Failed to capture screenshot")

# LLM Extraction Example
class OpenAIModelFee(BaseModel):
    model_name: str = Field(..., description="Name of the OpenAI model.")
    input_fee: str = Field(..., description="Fee for input token for the OpenAI model.")
    output_fee: str = Field(..., description="Fee for output token for the OpenAI model.")

async def extract_structured_data_using_llm(provider: str, api_token: str = None, extra_headers: Dict[str, str] = None):
    print(f"\n--- Extracting Structured Data with {provider} ---")
    
    if api_token is None and provider != "ollama":
        print(f"API token is required for {provider}. Skipping this example.")
        return

    browser_config = BrowserConfig(headless=True)
    
    extra_args = {
        "temperature": 0,
        "top_p": 0.9,
        "max_tokens": 2000
    }
    if extra_headers:
        extra_args["extra_headers"] = extra_headers

    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=1,
        page_timeout = 80000,
        extraction_strategy=LLMExtractionStrategy(
            provider=provider,
            api_token=api_token,
            schema=OpenAIModelFee.model_json_schema(),
            extraction_type="schema",
            instruction="""From the crawled content, extract all mentioned model names along with their fees for input and output tokens. 
            Do not miss any models in the entire content.""",
            extra_args=extra_args
        )
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://openai.com/api/pricing/",
            config=crawler_config
        )
        print(result.extracted_content)

# CSS Extraction Example
async def extract_structured_data_using_css_extractor():
    print("\n--- Using JsonCssExtractionStrategy for Fast Structured Output ---")
    schema = {
        "name": "KidoCode Courses",
        "baseSelector": "section.charge-methodology .w-tab-content > div",
        "fields": [
            {
                "name": "section_title",
                "selector": "h3.heading-50",
                "type": "text",
            },
            {
                "name": "section_description",
                "selector": ".charge-content",
                "type": "text",
            },
            {
                "name": "course_name",
                "selector": ".text-block-93",
                "type": "text",
            },
            {
                "name": "course_description",
                "selector": ".course-content-text",
                "type": "text",
            },
            {
                "name": "course_icon",
                "selector": ".image-92",
                "type": "attribute",
                "attribute": "src"
            }
        ]
    }

    browser_config = BrowserConfig(
        headless=True,
        java_script_enabled=True
    )
    
    js_click_tabs = """
    (async () => {
        const tabs = document.querySelectorAll("section.charge-methodology .tabs-menu-3 > div");
        for(let tab of tabs) {
            tab.scrollIntoView();
            tab.click();
            await new Promise(r => setTimeout(r, 500));
        }
    })();
    """
    
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=JsonCssExtractionStrategy(schema),
        js_code=[js_click_tabs]
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://www.kidocode.com/degrees/technology",
            config=crawler_config
        )

        companies = json.loads(result.extracted_content)
        print(f"Successfully extracted {len(companies)} companies")
        print(json.dumps(companies[0], indent=2))

# Dynamic Content Examples - Method 1
async def crawl_dynamic_content_pages_method_1():
    print("\n--- Advanced Multi-Page Crawling with JavaScript Execution ---")
    first_commit = ""

    async def on_execution_started(page, **kwargs):
        nonlocal first_commit
        try:
            while True:
                await page.wait_for_selector("li.Box-sc-g0xbh4-0 h4")
                commit = await page.query_selector("li.Box-sc-g0xbh4-0 h4")
                commit = await commit.evaluate("(element) => element.textContent")
                commit = re.sub(r"\s+", "", commit)
                if commit and commit != first_commit:
                    first_commit = commit
                    break
                await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Warning: New content didn't appear after JavaScript execution: {e}")

    browser_config = BrowserConfig(
        headless=False,
        java_script_enabled=True
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        crawler.crawler_strategy.set_hook("on_execution_started", on_execution_started)

        url = "https://github.com/microsoft/TypeScript/commits/main"
        session_id = "typescript_commits_session"
        all_commits = []

        js_next_page = """
        const button = document.querySelector('a[data-testid="pagination-next-button"]');
        if (button) button.click();
        """

        for page in range(3):
            crawler_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                css_selector="li.Box-sc-g0xbh4-0",
                js_code=js_next_page if page > 0 else None,
                js_only=page > 0,
                session_id=session_id
            )

            result = await crawler.arun(url=url, config=crawler_config)
            assert result.success, f"Failed to crawl page {page + 1}"

            soup = BeautifulSoup(result.cleaned_html, "html.parser")
            commits = soup.select("li")
            all_commits.extend(commits)

            print(f"Page {page + 1}: Found {len(commits)} commits")

        print(f"Successfully crawled {len(all_commits)} commits across 3 pages")

# Dynamic Content Examples - Method 2
async def crawl_dynamic_content_pages_method_2():
    print("\n--- Advanced Multi-Page Crawling with JavaScript Execution ---")

    browser_config = BrowserConfig(
        headless=False,
        java_script_enabled=True
    )

    js_next_page_and_wait = """
    (async () => {
        const getCurrentCommit = () => {
            const commits = document.querySelectorAll('li.Box-sc-g0xbh4-0 h4');
            return commits.length > 0 ? commits[0].textContent.trim() : null;
        };

        const initialCommit = getCurrentCommit();
        const button = document.querySelector('a[data-testid="pagination-next-button"]');
        if (button) button.click();

        while (true) {
            await new Promise(resolve => setTimeout(resolve, 100));
            const newCommit = getCurrentCommit();
            if (newCommit && newCommit !== initialCommit) {
                break;
            }
        }
    })();
    """

    schema = {
        "name": "Commit Extractor",
        "baseSelector": "li.Box-sc-g0xbh4-0",
        "fields": [
            {
                "name": "title",
                "selector": "h4.markdown-title",
                "type": "text",
                "transform": "strip",
            },
        ],
    }

    async with AsyncWebCrawler(config=browser_config) as crawler:
        url = "https://github.com/microsoft/TypeScript/commits/main"
        session_id = "typescript_commits_session"
        all_commits = []

        extraction_strategy = JsonCssExtractionStrategy(schema)

        for page in range(3):
            crawler_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                css_selector="li.Box-sc-g0xbh4-0",
                extraction_strategy=extraction_strategy,
                js_code=js_next_page_and_wait if page > 0 else None,
                js_only=page > 0,
                session_id=session_id
            )

            result = await crawler.arun(url=url, config=crawler_config)
            assert result.success, f"Failed to crawl page {page + 1}"

            commits = json.loads(result.extracted_content)
            all_commits.extend(commits)
            print(f"Page {page + 1}: Found {len(commits)} commits")

        print(f"Successfully crawled {len(all_commits)} commits across 3 pages")

# Browser Comparison
async def crawl_custom_browser_type():
    print("\n--- Browser Comparison ---")
    
    # Firefox
    browser_config_firefox = BrowserConfig(
        browser_type="firefox",
        headless=True
    )
    start = time.time()
    async with AsyncWebCrawler(config=browser_config_firefox) as crawler:
        result = await crawler.arun(
            url="https://www.example.com",
            config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        )
        print("Firefox:", time.time() - start)
        print(result.markdown[:500])

    # WebKit
    browser_config_webkit = BrowserConfig(
        browser_type="webkit",
        headless=True
    )
    start = time.time()
    async with AsyncWebCrawler(config=browser_config_webkit) as crawler:
        result = await crawler.arun(
            url="https://www.example.com",
            config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        )
        print("WebKit:", time.time() - start)
        print(result.markdown[:500])

    # Chromium (default)
    browser_config_chromium = BrowserConfig(
        browser_type="chromium",
        headless=True
    )
    start = time.time()
    async with AsyncWebCrawler(config=browser_config_chromium) as crawler:
        result = await crawler.arun(
            url="https://www.example.com",
            config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        )
        print("Chromium:", time.time() - start)
        print(result.markdown[:500])

# Anti-Bot and User Simulation
async def crawl_with_user_simulation():
    browser_config = BrowserConfig(
        headless=True,
        user_agent_mode="random",
        user_agent_generator_config={
            "device_type": "mobile",
            "os_type": "android"
        }
    )

    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        magic=True,
        simulate_user=True,
        override_navigator=True
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="YOUR-URL-HERE",
            config=crawler_config
        )
        print(result.markdown)

# Speed Comparison
async def speed_comparison():
    print("\n--- Speed Comparison ---")
    
    # Firecrawl comparison
    from firecrawl import FirecrawlApp
    app = FirecrawlApp(api_key=os.environ['FIRECRAWL_API_KEY'])
    start = time.time()
    scrape_status = app.scrape_url(
        'https://www.nbcnews.com/business',
        params={'formats': ['markdown', 'html']}
    )
    end = time.time()
    print("Firecrawl:")
    print(f"Time taken: {end - start:.2f} seconds")
    print(f"Content length: {len(scrape_status['markdown'])} characters")
    print(f"Images found: {scrape_status['markdown'].count('cldnry.s-nbcnews.com')}")
    print()

    # Crawl4AI comparisons
    browser_config = BrowserConfig(headless=True)
    
    # Simple crawl
    async with AsyncWebCrawler(config=browser_config) as crawler:
        start = time.time()
        result = await crawler.arun(
            url="https://www.nbcnews.com/business",
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                word_count_threshold=0
            )
        )
        end = time.time()
        print("Crawl4AI (simple crawl):")
        print(f"Time taken: {end - start:.2f} seconds")
        print(f"Content length: {len(result.markdown)} characters")
        print(f"Images found: {result.markdown.count('cldnry.s-nbcnews.com')}")
        print()

        # Advanced filtering
        start = time.time()
        result = await crawler.arun(
            url="https://www.nbcnews.com/business",
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                word_count_threshold=0,
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(
                        threshold=0.48,
                        threshold_type="fixed",
                        min_word_threshold=0
                    )
                )
            )
        )
        end = time.time()
        print("Crawl4AI (Markdown Plus):")
        print(f"Time taken: {end - start:.2f} seconds")
        print(f"Content length: {len(result.markdown_v2.raw_markdown)} characters")
        print(f"Fit Markdown: {len(result.markdown_v2.fit_markdown)} characters")
        print(f"Images found: {result.markdown.count('cldnry.s-nbcnews.com')}")
        print()

# Main execution
async def main():
    # Basic examples
    # await simple_crawl()
    # await simple_example_with_running_js_code()
    # await simple_example_with_css_selector()
    
    # Advanced examples
    # await extract_structured_data_using_css_extractor()
    await extract_structured_data_using_llm("openai/gpt-4o", os.getenv("OPENAI_API_KEY"))
    # await crawl_dynamic_content_pages_method_1()
    # await crawl_dynamic_content_pages_method_2()
    
    # Browser comparisons
    # await crawl_custom_browser_type()
    
    # Performance testing
    # await speed_comparison()

    # Screenshot example
    # await capture_and_save_screenshot(
    #     "https://www.example.com",
    #     os.path.join(__location__, "tmp/example_screenshot.jpg")
    # )

if __name__ == "__main__":
    asyncio.run(main())
```

## docs/examples/summarize_page.py
```
import os
import time
import json
from crawl4ai.web_crawler import WebCrawler
from crawl4ai.chunking_strategy import *
from crawl4ai.extraction_strategy import *
from crawl4ai.crawler_strategy import *

url = r'https://marketplace.visualstudio.com/items?itemName=Unclecode.groqopilot'

crawler = WebCrawler()
crawler.warmup()

from pydantic import BaseModel, Field

class PageSummary(BaseModel):
    title: str = Field(..., description="Title of the page.")
    summary: str = Field(..., description="Summary of the page.")
    brief_summary: str = Field(..., description="Brief summary of the page.")
    keywords: list = Field(..., description="Keywords assigned to the page.")

result = crawler.run(
    url=url,
    word_count_threshold=1,
    extraction_strategy= LLMExtractionStrategy(
        provider= "openai/gpt-4o", api_token = os.getenv('OPENAI_API_KEY'), 
        schema=PageSummary.model_json_schema(),
        extraction_type="schema",
        apply_chunking =False,
        instruction="From the crawled content, extract the following details: "\
            "1. Title of the page "\
            "2. Summary of the page, which is a detailed summary "\
            "3. Brief summary of the page, which is a paragraph text "\
            "4. Keywords assigned to the page, which is a list of keywords. "\
            'The extracted JSON format should look like this: '\
            '{ "title": "Page Title", "summary": "Detailed summary of the page.", "brief_summary": "Brief summary in a paragraph.", "keywords": ["keyword1", "keyword2", "keyword3"] }'
    ),
    bypass_cache=True,
)

page_summary = json.loads(result.extracted_content)

print(page_summary)

with open(".data/page_summary.json", "w", encoding="utf-8") as f:
    f.write(result.extracted_content)

```

## docs/examples/crawlai_vs_firecrawl.py
```
import os, time
# append the path to the root of the project
import sys
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from firecrawl import FirecrawlApp
from crawl4ai import AsyncWebCrawler
__data__ = os.path.join(os.path.dirname(__file__), '..', '..') + '/.data'

async def compare():
    app = FirecrawlApp(api_key=os.environ['FIRECRAWL_API_KEY'])

    # Tet Firecrawl with a simple crawl
    start = time.time()
    scrape_status = app.scrape_url(
    'https://www.nbcnews.com/business',
    params={'formats': ['markdown', 'html']}
    )
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    print(len(scrape_status['markdown']))
    # save the markdown content with provider name
    with open(f"{__data__}/firecrawl_simple.md", "w") as f:
        f.write(scrape_status['markdown'])
    # Count how many "cldnry.s-nbcnews.com" are in the markdown
    print(scrape_status['markdown'].count("cldnry.s-nbcnews.com"))
    


    async with AsyncWebCrawler() as crawler:
        start = time.time()
        result = await crawler.arun(
            url="https://www.nbcnews.com/business",
            # js_code=["const loadMoreButton = Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('Load More')); loadMoreButton && loadMoreButton.click();"],
            word_count_threshold=0,
            bypass_cache=True, 
            verbose=False
        )
        end = time.time()
        print(f"Time taken: {end - start} seconds")
        print(len(result.markdown))
        # save the markdown content with provider name  
        with open(f"{__data__}/crawl4ai_simple.md", "w") as f:
            f.write(result.markdown)
        # count how many "cldnry.s-nbcnews.com" are in the markdown
        print(result.markdown.count("cldnry.s-nbcnews.com"))

        start = time.time()
        result = await crawler.arun(
            url="https://www.nbcnews.com/business",
            js_code=["const loadMoreButton = Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('Load More')); loadMoreButton && loadMoreButton.click();"],
            word_count_threshold=0,
            bypass_cache=True, 
            verbose=False
        )
        end = time.time()
        print(f"Time taken: {end - start} seconds")
        print(len(result.markdown))
        # save the markdown content with provider name
        with open(f"{__data__}/crawl4ai_js.md", "w") as f:
            f.write(result.markdown)
        # count how many "cldnry.s-nbcnews.com" are in the markdown
        print(result.markdown.count("cldnry.s-nbcnews.com"))
        
if __name__ == "__main__":
    asyncio.run(compare())
    
```

## docs/examples/llm_extraction_openai_pricing.py
```
from crawl4ai.extraction_strategy import *
from crawl4ai.crawler_strategy import *
import asyncio
from pydantic import BaseModel, Field

url = r'https://openai.com/api/pricing/'

class OpenAIModelFee(BaseModel):
    model_name: str = Field(..., description="Name of the OpenAI model.")
    input_fee: str = Field(..., description="Fee for input token for the OpenAI model.")
    output_fee: str = Field(..., description="Fee for output token for the OpenAI model.")

from crawl4ai import AsyncWebCrawler

async def main():
    # Use AsyncWebCrawler
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
            word_count_threshold=1,
            extraction_strategy= LLMExtractionStrategy(
                # provider= "openai/gpt-4o", api_token = os.getenv('OPENAI_API_KEY'),
                provider= "groq/llama-3.1-70b-versatile", api_token = os.getenv('GROQ_API_KEY'),
                schema=OpenAIModelFee.model_json_schema(),
                extraction_type="schema",
                instruction="From the crawled content, extract all mentioned model names along with their " \
                            "fees for input and output tokens. Make sure not to miss anything in the entire content. " \
                            'One extracted model JSON format should look like this: ' \
                            '{ "model_name": "GPT-4", "input_fee": "US$10.00 / 1M tokens", "output_fee": "US$30.00 / 1M tokens" }'
            ),

        )
        print("Success:", result.success)
        model_fees = json.loads(result.extracted_content)
        print(len(model_fees))

        with open(".data/data.json", "w", encoding="utf-8") as f:
            f.write(result.extracted_content)

asyncio.run(main())

```

## docs/examples/v0.3.74.overview.py
```
import os, sys
# append the parent directory to the sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_parent_dir)
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__data__ = os.path.join(__location__, "__data")
import asyncio
from pathlib import Path
import aiohttp
import json
from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.content_filter_strategy import BM25ContentFilter

# 1. File Download Processing Example
async def download_example():
    """Example of downloading files from Python.org"""
    # downloads_path = os.path.join(os.getcwd(), "downloads")
    downloads_path = os.path.join(Path.home(), ".crawl4ai", "downloads")
    os.makedirs(downloads_path, exist_ok=True)
    
    print(f"Downloads will be saved to: {downloads_path}")
    
    async with AsyncWebCrawler(
        accept_downloads=True,
        downloads_path=downloads_path,
        verbose=True
    ) as crawler:
        result = await crawler.arun(
            url="https://www.python.org/downloads/",
            js_code="""
            // Find and click the first Windows installer link
            const downloadLink = document.querySelector('a[href$=".exe"]');
            if (downloadLink) {
                console.log('Found download link:', downloadLink.href);
                downloadLink.click();
            } else {
                console.log('No .exe download link found');
            }
            """,
            delay_before_return_html=1,  # Wait 5 seconds to ensure download starts
            cache_mode=CacheMode.BYPASS
        )
        
        if result.downloaded_files:
            print("\nDownload successful!")
            print("Downloaded files:")
            for file_path in result.downloaded_files:
                print(f"- {file_path}")
                print(f"  File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        else:
            print("\nNo files were downloaded")

# 2. Local File and Raw HTML Processing Example
async def local_and_raw_html_example():
    """Example of processing local files and raw HTML"""
    # Create a sample HTML file
    sample_file = os.path.join(__data__, "sample.html")
    with open(sample_file, "w") as f:
        f.write("""
        <html><body>
            <h1>Test Content</h1>
            <p>This is a test paragraph.</p>
        </body></html>
        """)
    
    async with AsyncWebCrawler(verbose=True) as crawler:
        # Process local file
        local_result = await crawler.arun(
            url=f"file://{os.path.abspath(sample_file)}"
        )
        
        # Process raw HTML
        raw_html = """
        <html><body>
            <h1>Raw HTML Test</h1>
            <p>This is a test of raw HTML processing.</p>
        </body></html>
        """
        raw_result = await crawler.arun(
            url=f"raw:{raw_html}"
        )
        
        # Clean up
        os.remove(sample_file)
        
        print("Local file content:", local_result.markdown)
        print("\nRaw HTML content:", raw_result.markdown)

# 3. Enhanced Markdown Generation Example
async def markdown_generation_example():
    """Example of enhanced markdown generation with citations and LLM-friendly features"""
    async with AsyncWebCrawler(verbose=True) as crawler:
        # Create a content filter (optional)
        content_filter = BM25ContentFilter(
            # user_query="History and cultivation",
            bm25_threshold=1.0
        )
        
        result = await crawler.arun(
            url="https://en.wikipedia.org/wiki/Apple",
            css_selector="main div#bodyContent",
            content_filter=content_filter,
            cache_mode=CacheMode.BYPASS
        )
        
        from crawl4ai import AsyncWebCrawler
        from crawl4ai.content_filter_strategy import BM25ContentFilter
        
        result = await crawler.arun(
            url="https://en.wikipedia.org/wiki/Apple",
            css_selector="main div#bodyContent",
            content_filter=BM25ContentFilter()
        )
        print(result.markdown_v2.fit_markdown)
        
        print("\nMarkdown Generation Results:")
        print(f"1. Original markdown length: {len(result.markdown)}")
        print(f"2. New markdown versions (markdown_v2):")
        print(f"   - Raw markdown length: {len(result.markdown_v2.raw_markdown)}")
        print(f"   - Citations markdown length: {len(result.markdown_v2.markdown_with_citations)}")
        print(f"   - References section length: {len(result.markdown_v2.references_markdown)}")
        if result.markdown_v2.fit_markdown:
            print(f"   - Filtered markdown length: {len(result.markdown_v2.fit_markdown)}")
        
        # Save examples to files
        output_dir = os.path.join(__data__, "markdown_examples")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save different versions
        with open(os.path.join(output_dir, "1_raw_markdown.md"), "w") as f:
            f.write(result.markdown_v2.raw_markdown)
            
        with open(os.path.join(output_dir, "2_citations_markdown.md"), "w") as f:
            f.write(result.markdown_v2.markdown_with_citations)
            
        with open(os.path.join(output_dir, "3_references.md"), "w") as f:
            f.write(result.markdown_v2.references_markdown)
            
        if result.markdown_v2.fit_markdown:
            with open(os.path.join(output_dir, "4_filtered_markdown.md"), "w") as f:
                f.write(result.markdown_v2.fit_markdown)
                
        print(f"\nMarkdown examples saved to: {output_dir}")
        
        # Show a sample of citations and references
        print("\nSample of markdown with citations:")
        print(result.markdown_v2.markdown_with_citations[:500] + "...\n")
        print("Sample of references:")
        print('\n'.join(result.markdown_v2.references_markdown.split('\n')[:10]) + "...")

# 4. Browser Management Example
async def browser_management_example():
    """Example of using enhanced browser management features"""
    # Use the specified user directory path
    user_data_dir = os.path.join(Path.home(), ".crawl4ai", "browser_profile")
    os.makedirs(user_data_dir, exist_ok=True)
    
    print(f"Browser profile will be saved to: {user_data_dir}")
    
    async with AsyncWebCrawler(
        use_managed_browser=True,
        user_data_dir=user_data_dir,
        headless=False,
        verbose=True
    ) as crawler:

        result = await crawler.arun(
            url="https://crawl4ai.com",
            # session_id="persistent_session_1",
            cache_mode=CacheMode.BYPASS
        )        
        # Use GitHub as an example - it's a good test for browser management
        # because it requires proper browser handling
        result = await crawler.arun(
            url="https://github.com/trending",
            # session_id="persistent_session_1",
            cache_mode=CacheMode.BYPASS
        )
        
        print("\nBrowser session result:", result.success)
        if result.success:
            print("Page title:", result.metadata.get('title', 'No title found'))

# 5. API Usage Example
async def api_example():
    """Example of using the new API endpoints"""
    api_token = os.getenv('CRAWL4AI_API_TOKEN') or "test_api_code"
    headers = {'Authorization': f'Bearer {api_token}'}    
    async with aiohttp.ClientSession() as session:
        # Submit crawl job
        crawl_request = {
            "urls": ["https://news.ycombinator.com"],  # Hacker News as an example
            "extraction_config": {
                "type": "json_css",
                "params": {
                    "schema": {
                        "name": "Hacker News Articles",
                        "baseSelector": ".athing",
                        "fields": [
                            {
                                "name": "title",
                                "selector": ".title a",
                                "type": "text"
                            },
                            {
                                "name": "score",
                                "selector": ".score",
                                "type": "text"
                            },
                            {
                                "name": "url",
                                "selector": ".title a",
                                "type": "attribute",
                                "attribute": "href"
                            }
                        ]
                    }
                }
            },
            "crawler_params": {
                "headless": True,
                # "use_managed_browser": True
            },
            "cache_mode": "bypass",
            # "screenshot": True,
            # "magic": True
        }
        
        async with session.post(
            "http://localhost:11235/crawl",
            json=crawl_request,
            headers=headers
        ) as response:
            task_data = await response.json()
            task_id = task_data["task_id"]
            
            # Check task status
            while True:
                async with session.get(
                    f"http://localhost:11235/task/{task_id}",
                    headers=headers
                ) as status_response:
                    result = await status_response.json()
                    print(f"Task status: {result['status']}")
                    
                    if result["status"] == "completed":
                        print("Task completed!")
                        print("Results:")
                        news = json.loads(result["results"][0]['extracted_content'])
                        print(json.dumps(news[:4], indent=2))
                        break
                    else:
                        await asyncio.sleep(1)

# Main execution
async def main():
    # print("Running Crawl4AI feature examples...")
    
    # print("\n1. Running Download Example:")
    # await download_example()
    
    # print("\n2. Running Markdown Generation Example:")
    # await markdown_generation_example()
    
    # # print("\n3. Running Local and Raw HTML Example:")
    # await local_and_raw_html_example()
    
    # # print("\n4. Running Browser Management Example:")
    await browser_management_example()
    
    # print("\n5. Running API Example:")
    await api_example()

if __name__ == "__main__":
    asyncio.run(main())
```

## docs/examples/docker_example.py
```
import requests
import json
import time
import sys
import base64
import os
from typing import Dict, Any

class Crawl4AiTester:
    def __init__(self, base_url: str = "http://localhost:11235", api_token: str = None):
        self.base_url = base_url
        self.api_token = api_token or os.getenv('CRAWL4AI_API_TOKEN') or "test_api_code"  # Check environment variable as fallback
        self.headers = {'Authorization': f'Bearer {self.api_token}'} if self.api_token else {}
        
    def submit_and_wait(self, request_data: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
        # Submit crawl job
        response = requests.post(f"{self.base_url}/crawl", json=request_data, headers=self.headers)
        if response.status_code == 403:
            raise Exception("API token is invalid or missing")
        task_id = response.json()["task_id"]
        print(f"Task ID: {task_id}")
        
        # Poll for result
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
                
            result = requests.get(f"{self.base_url}/task/{task_id}", headers=self.headers)
            status = result.json()
            
            if status["status"] == "failed":
                print("Task failed:", status.get("error"))
                raise Exception(f"Task failed: {status.get('error')}")
                
            if status["status"] == "completed":
                return status
                
            time.sleep(2)
            
    def submit_sync(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/crawl_sync", json=request_data, headers=self.headers, timeout=60)
        if response.status_code == 408:
            raise TimeoutError("Task did not complete within server timeout")
        response.raise_for_status()
        return response.json()
    
    def crawl_direct(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Directly crawl without using task queue"""
        response = requests.post(
            f"{self.base_url}/crawl_direct", 
            json=request_data, 
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

def test_docker_deployment(version="basic"):
    tester = Crawl4AiTester(
        base_url="http://localhost:11235" ,
        # base_url="https://api.crawl4ai.com" # just for example
        # api_token="test" # just for example
    )
    print(f"Testing Crawl4AI Docker {version} version")
    
    # Health check with timeout and retry
    max_retries = 5
    for i in range(max_retries):
        try:
            health = requests.get(f"{tester.base_url}/health", timeout=10)
            print("Health check:", health.json())
            break
        except requests.exceptions.RequestException as e:
            if i == max_retries - 1:
                print(f"Failed to connect after {max_retries} attempts")
                sys.exit(1)
            print(f"Waiting for service to start (attempt {i+1}/{max_retries})...")
            time.sleep(5)
    
    # Test cases based on version
    test_basic_crawl_direct(tester)
    test_basic_crawl(tester)
    test_basic_crawl(tester)
    test_basic_crawl_sync(tester)
    
    if version in ["full", "transformer"]:
        test_cosine_extraction(tester)

    test_js_execution(tester)
    test_css_selector(tester)
    test_structured_extraction(tester)
    test_llm_extraction(tester)
    test_llm_with_ollama(tester)
    test_screenshot(tester)
    

def test_basic_crawl(tester: Crawl4AiTester):
    print("\n=== Testing Basic Crawl ===")
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 10, 
        "session_id": "test"
    }
    
    result = tester.submit_and_wait(request)
    print(f"Basic crawl result length: {len(result['result']['markdown'])}")
    assert result["result"]["success"]
    assert len(result["result"]["markdown"]) > 0

def test_basic_crawl_sync(tester: Crawl4AiTester):
    print("\n=== Testing Basic Crawl (Sync) ===")
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 10,
        "session_id": "test"
    }
    
    result = tester.submit_sync(request)
    print(f"Basic crawl result length: {len(result['result']['markdown'])}")
    assert result['status'] == 'completed'
    assert result['result']['success']
    assert len(result['result']['markdown']) > 0
    
def test_basic_crawl_direct(tester: Crawl4AiTester):
    print("\n=== Testing Basic Crawl (Direct) ===")
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 10,
        # "session_id": "test"
        "cache_mode": "bypass"  # or "enabled", "disabled", "read_only", "write_only"
    }
    
    result = tester.crawl_direct(request)
    print(f"Basic crawl result length: {len(result['result']['markdown'])}")
    assert result['result']['success']
    assert len(result['result']['markdown']) > 0
    
def test_js_execution(tester: Crawl4AiTester):
    print("\n=== Testing JS Execution ===")
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 8,
        "js_code": [
            "const loadMoreButton = Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('Load More')); loadMoreButton && loadMoreButton.click();"
        ],
        "wait_for": "article.tease-card:nth-child(10)",
        "crawler_params": {
            "headless": True
        }
    }
    
    result = tester.submit_and_wait(request)
    print(f"JS execution result length: {len(result['result']['markdown'])}")
    assert result["result"]["success"]

def test_css_selector(tester: Crawl4AiTester):
    print("\n=== Testing CSS Selector ===")
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 7,
        "css_selector": ".wide-tease-item__description",
        "crawler_params": {
            "headless": True
        },
        "extra": {"word_count_threshold": 10}
        
    }
    
    result = tester.submit_and_wait(request)
    print(f"CSS selector result length: {len(result['result']['markdown'])}")
    assert result["result"]["success"]

def test_structured_extraction(tester: Crawl4AiTester):
    print("\n=== Testing Structured Extraction ===")
    schema = {
        "name": "Coinbase Crypto Prices",
        "baseSelector": ".cds-tableRow-t45thuk",
        "fields": [
            {
                "name": "crypto",
                "selector": "td:nth-child(1) h2",
                "type": "text",
            },
            {
                "name": "symbol",
                "selector": "td:nth-child(1) p",
                "type": "text",
            },
            {
                "name": "price",
                "selector": "td:nth-child(2)",
                "type": "text",
            }
        ],
    }
    
    request = {
        "urls": "https://www.coinbase.com/explore",
        "priority": 9,
        "extraction_config": {
            "type": "json_css",
            "params": {
                "schema": schema
            }
        }
    }
    
    result = tester.submit_and_wait(request)
    extracted = json.loads(result["result"]["extracted_content"])
    print(f"Extracted {len(extracted)} items")
    print("Sample item:", json.dumps(extracted[0], indent=2))
    assert result["result"]["success"]
    assert len(extracted) > 0

def test_llm_extraction(tester: Crawl4AiTester):
    print("\n=== Testing LLM Extraction ===")
    schema = {
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the OpenAI model."
            },
            "input_fee": {
                "type": "string",
                "description": "Fee for input token for the OpenAI model."
            },
            "output_fee": {
                "type": "string",
                "description": "Fee for output token for the OpenAI model."
            }
        },
        "required": ["model_name", "input_fee", "output_fee"]
    }
    
    request = {
        "urls": "https://openai.com/api/pricing",
        "priority": 8,
        "extraction_config": {
            "type": "llm",
            "params": {
                "provider": "openai/gpt-4o-mini",
                "api_token": os.getenv("OPENAI_API_KEY"),
                "schema": schema,
                "extraction_type": "schema",
                "instruction": """From the crawled content, extract all mentioned model names along with their fees for input and output tokens."""
            }
        },
        "crawler_params": {"word_count_threshold": 1}
    }
    
    try:
        result = tester.submit_and_wait(request)
        extracted = json.loads(result["result"]["extracted_content"])
        print(f"Extracted {len(extracted)} model pricing entries")
        print("Sample entry:", json.dumps(extracted[0], indent=2))
        assert result["result"]["success"]
    except Exception as e:
        print(f"LLM extraction test failed (might be due to missing API key): {str(e)}")

def test_llm_with_ollama(tester: Crawl4AiTester):
    print("\n=== Testing LLM with Ollama ===")
    schema = {
        "type": "object",
        "properties": {
            "article_title": {
                "type": "string",
                "description": "The main title of the news article"
            },
            "summary": {
                "type": "string",
                "description": "A brief summary of the article content"
            },
            "main_topics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Main topics or themes discussed in the article"
            }
        }
    }
    
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 8,
        "extraction_config": {
            "type": "llm",
            "params": {
                "provider": "ollama/llama2",
                "schema": schema,
                "extraction_type": "schema",
                "instruction": "Extract the main article information including title, summary, and main topics."
            }
        },
        "extra": {"word_count_threshold": 1},
        "crawler_params": {"verbose": True}
    }
    
    try:
        result = tester.submit_and_wait(request)
        extracted = json.loads(result["result"]["extracted_content"])
        print("Extracted content:", json.dumps(extracted, indent=2))
        assert result["result"]["success"]
    except Exception as e:
        print(f"Ollama extraction test failed: {str(e)}")

def test_cosine_extraction(tester: Crawl4AiTester):
    print("\n=== Testing Cosine Extraction ===")
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 8,
        "extraction_config": {
            "type": "cosine",
            "params": {
                "semantic_filter": "business finance economy",
                "word_count_threshold": 10,
                "max_dist": 0.2,
                "top_k": 3
            }
        }
    }
    
    try:
        result = tester.submit_and_wait(request)
        extracted = json.loads(result["result"]["extracted_content"])
        print(f"Extracted {len(extracted)} text clusters")
        print("First cluster tags:", extracted[0]["tags"])
        assert result["result"]["success"]
    except Exception as e:
        print(f"Cosine extraction test failed: {str(e)}")

def test_screenshot(tester: Crawl4AiTester):
    print("\n=== Testing Screenshot ===")
    request = {
        "urls": "https://www.nbcnews.com/business",
        "priority": 5,
        "screenshot": True,
        "crawler_params": {
            "headless": True
        }
    }
    
    result = tester.submit_and_wait(request)
    print("Screenshot captured:", bool(result["result"]["screenshot"]))
    
    if result["result"]["screenshot"]:
        # Save screenshot
        screenshot_data = base64.b64decode(result["result"]["screenshot"])
        with open("test_screenshot.jpg", "wb") as f:
            f.write(screenshot_data)
        print("Screenshot saved as test_screenshot.jpg")
    
    assert result["result"]["success"]

if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "basic"
    # version = "full"
    test_docker_deployment(version)
```

## docs/examples/research_assistant.py
```
# Make sure to install the required packageschainlit and groq
import os, time
from openai import AsyncOpenAI
import chainlit as cl
import re
import requests
from io import BytesIO
from chainlit.element import ElementBased
from groq import Groq

# Import threadpools to run the crawl_url function in a separate thread
from concurrent.futures import ThreadPoolExecutor

client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))

# Instrument the OpenAI client
cl.instrument_openai()

settings = {
    "model": "llama3-8b-8192",
    "temperature": 0.5,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

def extract_urls(text):
    url_pattern = re.compile(r'(https?://\S+)')
    return url_pattern.findall(text)

def crawl_url(url):
    data = {
        "urls": [url],
        "include_raw_html": True,
        "word_count_threshold": 10,
        "extraction_strategy": "NoExtractionStrategy",
        "chunking_strategy": "RegexChunking"
    }
    response = requests.post("https://crawl4ai.com/crawl", json=data)
    response_data = response.json()
    response_data = response_data['results'][0]
    return response_data['markdown']

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("session", {
        "history": [],
        "context": {}
    })  
    await cl.Message(
        content="Welcome to the chat! How can I assist you today?"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    user_session = cl.user_session.get("session")
    
    # Extract URLs from the user's message
    urls = extract_urls(message.content)
    
    
    futures = []
    with ThreadPoolExecutor() as executor:
        for url in urls:
            futures.append(executor.submit(crawl_url, url))

    results = [future.result() for future in futures]

    for url, result in zip(urls, results):
        ref_number = f"REF_{len(user_session['context']) + 1}"
        user_session["context"][ref_number] = {
            "url": url,
            "content": result
        }    


    user_session["history"].append({
        "role": "user",
        "content": message.content
    })

    # Create a system message that includes the context
    context_messages = [
        f'<appendix ref="{ref}">\n{data["content"]}\n</appendix>'
        for ref, data in user_session["context"].items()
    ]
    if context_messages:
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful bot. Use the following context for answering questions. "
                "Refer to the sources using the REF number in square brackets, e.g., [1], only if the source is given in the appendices below.\n\n"
                "If the question requires any information from the provided appendices or context, refer to the sources. "
                "If not, there is no need to add a references section. "
                "At the end of your response, provide a reference section listing the URLs and their REF numbers only if sources from the appendices were used.\n\n"
                "\n\n".join(context_messages)
            )
        }
    else:
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant."
        }


    msg = cl.Message(content="")
    await msg.send()

    # Get response from the LLM
    stream = await client.chat.completions.create(
        messages=[
            system_message,
            *user_session["history"]
        ],
        stream=True,
        **settings
    )

    assistant_response = ""
    async for part in stream:
        if token := part.choices[0].delta.content:
            assistant_response += token
            await msg.stream_token(token)

    # Add assistant message to the history
    user_session["history"].append({
        "role": "assistant",
        "content": assistant_response
    })
    await msg.update()

    # Append the reference section to the assistant's response
    reference_section = "\n\nReferences:\n"
    for ref, data in user_session["context"].items():
        reference_section += f"[{ref.split('_')[1]}]: {data['url']}\n"

    msg.content += reference_section
    await msg.update()


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # Write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)

    pass

@cl.step(type="tool")
async def speech_to_text(audio_file):
    cli = Groq()
       
    response = await client.audio.transcriptions.create(
        model="whisper-large-v3", file=audio_file
    )

    return response.text


@cl.on_audio_end
async def on_audio_end(elements: list[ElementBased]):
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")
    
    start_time = time.time()
    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)
    end_time = time.time()
    print(f"Transcription took {end_time - start_time} seconds")
    
    user_msg = cl.Message(
        author="You", 
        type="user_message",
        content=transcription
    )
    await user_msg.send()
    await on_message(user_msg)


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)



```

## docs/examples/quickstart_async.py
```
import os, sys
# append parent directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); os.environ['FIRECRAWL_API_KEY'] = "fc-84b370ccfad44beabc686b38f1769692";

import asyncio
# import nest_asyncio
# nest_asyncio.apply()

import time
import json
import os
import re
from typing import Dict, List
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
from crawl4ai.extraction_strategy import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
)

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

print("Crawl4AI: Advanced Web Crawling and Data Extraction")
print("GitHub Repository: https://github.com/unclecode/crawl4ai")
print("Twitter: @unclecode")
print("Website: https://crawl4ai.com")


async def simple_crawl():
    print("\n--- Basic Usage ---")
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url="https://www.nbcnews.com/business", cache_mode= CacheMode.BYPASS)
        print(result.markdown[:500])  # Print first 500 characters

async def simple_example_with_running_js_code():
    print("\n--- Executing JavaScript and Using CSS Selectors ---")
    # New code to handle the wait_for parameter
    wait_for = """() => {
        return Array.from(document.querySelectorAll('article.tease-card')).length > 10;
    }"""

    # wait_for can be also just a css selector
    # wait_for = "article.tease-card:nth-child(10)"

    async with AsyncWebCrawler(verbose=True) as crawler:
        js_code = [
            "const loadMoreButton = Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('Load More')); loadMoreButton && loadMoreButton.click();"
        ]
        result = await crawler.arun(
            url="https://www.nbcnews.com/business",
            js_code=js_code,
            # wait_for=wait_for,
            cache_mode=CacheMode.BYPASS,
        )
        print(result.markdown[:500])  # Print first 500 characters

async def simple_example_with_css_selector():
    print("\n--- Using CSS Selectors ---")
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(
            url="https://www.nbcnews.com/business",
            css_selector=".wide-tease-item__description",
            cache_mode=CacheMode.BYPASS,
        )
        print(result.markdown[:500])  # Print first 500 characters

async def use_proxy():
    print("\n--- Using a Proxy ---")
    print(
        "Note: Replace 'http://your-proxy-url:port' with a working proxy to run this example."
    )
    # Uncomment and modify the following lines to use a proxy
    async with AsyncWebCrawler(verbose=True, proxy="http://your-proxy-url:port") as crawler:
        result = await crawler.arun(
            url="https://www.nbcnews.com/business",
            cache_mode= CacheMode.BYPASS
        )
        if result.success:
            print(result.markdown[:500])  # Print first 500 characters

async def capture_and_save_screenshot(url: str, output_path: str):
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(
            url=url,
            screenshot=True,
            cache_mode= CacheMode.BYPASS
        )
        
        if result.success and result.screenshot:
            import base64
            
            # Decode the base64 screenshot data
            screenshot_data = base64.b64decode(result.screenshot)
            
            # Save the screenshot as a JPEG file
            with open(output_path, 'wb') as f:
                f.write(screenshot_data)
            
            print(f"Screenshot saved successfully to {output_path}")
        else:
            print("Failed to capture screenshot")

class OpenAIModelFee(BaseModel):
    model_name: str = Field(..., description="Name of the OpenAI model.")
    input_fee: str = Field(..., description="Fee for input token for the OpenAI model.")
    output_fee: str = Field(
        ..., description="Fee for output token for the OpenAI model."
    )

async def extract_structured_data_using_llm(provider: str, api_token: str = None, extra_headers: Dict[str, str] = None):
    print(f"\n--- Extracting Structured Data with {provider} ---")
    
    if api_token is None and provider != "ollama":
        print(f"API token is required for {provider}. Skipping this example.")
        return

    # extra_args = {}
    extra_args={
        "temperature": 0, 
        "top_p": 0.9,
        "max_tokens": 2000,
        # any other supported parameters for litellm
    }
    if extra_headers:
        extra_args["extra_headers"] = extra_headers

    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(
            url="https://openai.com/api/pricing/",
            word_count_threshold=1,
            extraction_strategy=LLMExtractionStrategy(
                provider=provider,
                api_token=api_token,
                schema=OpenAIModelFee.model_json_schema(),
                extraction_type="schema",
                instruction="""From the crawled content, extract all mentioned model names along with their fees for input and output tokens. 
                Do not miss any models in the entire content. One extracted model JSON format should look like this: 
                {"model_name": "GPT-4", "input_fee": "US$10.00 / 1M tokens", "output_fee": "US$30.00 / 1M tokens"}.""",
                extra_args=extra_args
            ),
            cache_mode=CacheMode.BYPASS,
        )
        print(result.extracted_content)

async def extract_structured_data_using_css_extractor():
    print("\n--- Using JsonCssExtractionStrategy for Fast Structured Output ---")
    schema = {
    "name": "KidoCode Courses",
    "baseSelector": "section.charge-methodology .w-tab-content > div",
    "fields": [
        {
            "name": "section_title",
            "selector": "h3.heading-50",
            "type": "text",
        },
        {
            "name": "section_description",
            "selector": ".charge-content",
            "type": "text",
        },
        {
            "name": "course_name",
            "selector": ".text-block-93",
            "type": "text",
        },
        {
            "name": "course_description",
            "selector": ".course-content-text",
            "type": "text",
        },
        {
            "name": "course_icon",
            "selector": ".image-92",
            "type": "attribute",
            "attribute": "src"
        }
    ]
}

    async with AsyncWebCrawler(
        headless=True,
        verbose=True
    ) as crawler:
        
        # Create the JavaScript that handles clicking multiple times
        js_click_tabs = """
        (async () => {
            const tabs = document.querySelectorAll("section.charge-methodology .tabs-menu-3 > div");
            
            for(let tab of tabs) {
                // scroll to the tab
                tab.scrollIntoView();
                tab.click();
                // Wait for content to load and animations to complete
                await new Promise(r => setTimeout(r, 500));
            }
        })();
        """     

        result = await crawler.arun(
            url="https://www.kidocode.com/degrees/technology",
            extraction_strategy=JsonCssExtractionStrategy(schema, verbose=True),
            js_code=[js_click_tabs],
            cache_mode=CacheMode.BYPASS
        )

        companies = json.loads(result.extracted_content)
        print(f"Successfully extracted {len(companies)} companies")
        print(json.dumps(companies[0], indent=2))

# Advanced Session-Based Crawling with Dynamic Content 🔄
async def crawl_dynamic_content_pages_method_1():
    print("\n--- Advanced Multi-Page Crawling with JavaScript Execution ---")
    first_commit = ""

    async def on_execution_started(page):
        nonlocal first_commit
        try:
            while True:
                await page.wait_for_selector("li.Box-sc-g0xbh4-0 h4")
                commit = await page.query_selector("li.Box-sc-g0xbh4-0 h4")
                commit = await commit.evaluate("(element) => element.textContent")
                commit = re.sub(r"\s+", "", commit)
                if commit and commit != first_commit:
                    first_commit = commit
                    break
                await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Warning: New content didn't appear after JavaScript execution: {e}")

    async with AsyncWebCrawler(verbose=True) as crawler:
        crawler.crawler_strategy.set_hook("on_execution_started", on_execution_started)

        url = "https://github.com/microsoft/TypeScript/commits/main"
        session_id = "typescript_commits_session"
        all_commits = []

        js_next_page = """
        (() => {
            const button = document.querySelector('a[data-testid="pagination-next-button"]');
            if (button) button.click();
        })();
        """

        for page in range(3):  # Crawl 3 pages
            result = await crawler.arun(
                url=url,
                session_id=session_id,
                css_selector="li.Box-sc-g0xbh4-0",
                js=js_next_page if page > 0 else None,
                cache_mode=CacheMode.BYPASS,
                js_only=page > 0,
                headless=False,
            )

            assert result.success, f"Failed to crawl page {page + 1}"

            soup = BeautifulSoup(result.cleaned_html, "html.parser")
            commits = soup.select("li")
            all_commits.extend(commits)

            print(f"Page {page + 1}: Found {len(commits)} commits")

        await crawler.crawler_strategy.kill_session(session_id)
        print(f"Successfully crawled {len(all_commits)} commits across 3 pages")

async def crawl_dynamic_content_pages_method_2():
    print("\n--- Advanced Multi-Page Crawling with JavaScript Execution ---")

    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://github.com/microsoft/TypeScript/commits/main"
        session_id = "typescript_commits_session"
        all_commits = []
        last_commit = ""

        js_next_page_and_wait = """
        (async () => {
            const getCurrentCommit = () => {
                const commits = document.querySelectorAll('li.Box-sc-g0xbh4-0 h4');
                return commits.length > 0 ? commits[0].textContent.trim() : null;
            };

            const initialCommit = getCurrentCommit();
            const button = document.querySelector('a[data-testid="pagination-next-button"]');
            if (button) button.click();

            // Poll for changes
            while (true) {
                await new Promise(resolve => setTimeout(resolve, 100)); // Wait 100ms
                const newCommit = getCurrentCommit();
                if (newCommit && newCommit !== initialCommit) {
                    break;
                }
            }
        })();
        """

        schema = {
            "name": "Commit Extractor",
            "baseSelector": "li.Box-sc-g0xbh4-0",
            "fields": [
                {
                    "name": "title",
                    "selector": "h4.markdown-title",
                    "type": "text",
                    "transform": "strip",
                },
            ],
        }
        extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)

        for page in range(3):  # Crawl 3 pages
            result = await crawler.arun(
                url=url,
                session_id=session_id,
                css_selector="li.Box-sc-g0xbh4-0",
                extraction_strategy=extraction_strategy,
                js_code=js_next_page_and_wait if page > 0 else None,
                js_only=page > 0,
                cache_mode=CacheMode.BYPASS,
                headless=False,
            )

            assert result.success, f"Failed to crawl page {page + 1}"

            commits = json.loads(result.extracted_content)
            all_commits.extend(commits)

            print(f"Page {page + 1}: Found {len(commits)} commits")

        await crawler.crawler_strategy.kill_session(session_id)
        print(f"Successfully crawled {len(all_commits)} commits across 3 pages")

async def crawl_dynamic_content_pages_method_3():
    print("\n--- Advanced Multi-Page Crawling with JavaScript Execution using `wait_for` ---")

    async with AsyncWebCrawler(verbose=True) as crawler:
        url = "https://github.com/microsoft/TypeScript/commits/main"
        session_id = "typescript_commits_session"
        all_commits = []

        js_next_page = """
        const commits = document.querySelectorAll('li.Box-sc-g0xbh4-0 h4');
        if (commits.length > 0) {
            window.firstCommit = commits[0].textContent.trim();
        }
        const button = document.querySelector('a[data-testid="pagination-next-button"]');
        if (button) button.click();
        """

        wait_for = """() => {
            const commits = document.querySelectorAll('li.Box-sc-g0xbh4-0 h4');
            if (commits.length === 0) return false;
            const firstCommit = commits[0].textContent.trim();
            return firstCommit !== window.firstCommit;
        }"""
        
        schema = {
            "name": "Commit Extractor",
            "baseSelector": "li.Box-sc-g0xbh4-0",
            "fields": [
                {
                    "name": "title",
                    "selector": "h4.markdown-title",
                    "type": "text",
                    "transform": "strip",
                },
            ],
        }
        extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)

        for page in range(3):  # Crawl 3 pages
            result = await crawler.arun(
                url=url,
                session_id=session_id,
                css_selector="li.Box-sc-g0xbh4-0",
                extraction_strategy=extraction_strategy,
                js_code=js_next_page if page > 0 else None,
                wait_for=wait_for if page > 0 else None,
                js_only=page > 0,
                cache_mode=CacheMode.BYPASS,
                headless=False,
            )

            assert result.success, f"Failed to crawl page {page + 1}"

            commits = json.loads(result.extracted_content)
            all_commits.extend(commits)

            print(f"Page {page + 1}: Found {len(commits)} commits")

        await crawler.crawler_strategy.kill_session(session_id)
        print(f"Successfully crawled {len(all_commits)} commits across 3 pages")

async def crawl_custom_browser_type():
    # Use Firefox
    start = time.time()
    async with AsyncWebCrawler(browser_type="firefox", verbose=True, headless = True) as crawler:
        result = await crawler.arun(url="https://www.example.com", cache_mode= CacheMode.BYPASS)
        print(result.markdown[:500])
        print("Time taken: ", time.time() - start)

    # Use WebKit
    start = time.time()
    async with AsyncWebCrawler(browser_type="webkit", verbose=True, headless = True) as crawler:
        result = await crawler.arun(url="https://www.example.com", cache_mode= CacheMode.BYPASS)
        print(result.markdown[:500])
        print("Time taken: ", time.time() - start)

    # Use Chromium (default)
    start = time.time()
    async with AsyncWebCrawler(verbose=True, headless = True) as crawler:
        result = await crawler.arun(url="https://www.example.com", cache_mode= CacheMode.BYPASS)
        print(result.markdown[:500])
        print("Time taken: ", time.time() - start)

async def crawl_with_user_simultion():
    async with AsyncWebCrawler(verbose=True, headless=True) as crawler:
        url = "YOUR-URL-HERE"
        result = await crawler.arun(
            url=url,            
            cache_mode=CacheMode.BYPASS,
            magic = True, # Automatically detects and removes overlays, popups, and other elements that block content
            # simulate_user = True,# Causes a series of random mouse movements and clicks to simulate user interaction
            # override_navigator = True # Overrides the navigator object to make it look like a real user
        )
        
        print(result.markdown)    

async def speed_comparison():
    # print("\n--- Speed Comparison ---")
    # print("Firecrawl (simulated):")
    # print("Time taken: 7.02 seconds")
    # print("Content length: 42074 characters")
    # print("Images found: 49")
    # print()
    # Simulated Firecrawl performance
    from firecrawl import FirecrawlApp
    app = FirecrawlApp(api_key=os.environ['FIRECRAWL_API_KEY'])
    start = time.time()
    scrape_status = app.scrape_url(
    'https://www.nbcnews.com/business',
    params={'formats': ['markdown', 'html']}
    )
    end = time.time()
    print("Firecrawl:")
    print(f"Time taken: {end - start:.2f} seconds")
    print(f"Content length: {len(scrape_status['markdown'])} characters")
    print(f"Images found: {scrape_status['markdown'].count('cldnry.s-nbcnews.com')}")
    print()    

    async with AsyncWebCrawler() as crawler:
        # Crawl4AI simple crawl
        start = time.time()
        result = await crawler.arun(
            url="https://www.nbcnews.com/business",
            word_count_threshold=0,
            cache_mode=CacheMode.BYPASS,
            verbose=False,
        )
        end = time.time()
        print("Crawl4AI (simple crawl):")
        print(f"Time taken: {end - start:.2f} seconds")
        print(f"Content length: {len(result.markdown)} characters")
        print(f"Images found: {result.markdown.count('cldnry.s-nbcnews.com')}")
        print()

        # Crawl4AI with advanced content filtering
        start = time.time()
        result = await crawler.arun(
            url="https://www.nbcnews.com/business",
            word_count_threshold=0,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter = PruningContentFilter(threshold=0.48, threshold_type="fixed", min_word_threshold=0)
                # content_filter=BM25ContentFilter(user_query=None, bm25_threshold=1.0)
            ),
            cache_mode=CacheMode.BYPASS,
            verbose=False,
        )
        end = time.time()
        print("Crawl4AI (Markdown Plus):")
        print(f"Time taken: {end - start:.2f} seconds")
        print(f"Content length: {len(result.markdown_v2.raw_markdown)} characters")
        print(f"Fit Markdown: {len(result.markdown_v2.fit_markdown)} characters")
        print(f"Images found: {result.markdown.count('cldnry.s-nbcnews.com')}")
        print()

        # Crawl4AI with JavaScript execution
        start = time.time()
        result = await crawler.arun(
            url="https://www.nbcnews.com/business",
            js_code=[
                "const loadMoreButton = Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('Load More')); loadMoreButton && loadMoreButton.click();"
            ],
            word_count_threshold=0,
            cache_mode=CacheMode.BYPASS,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter = PruningContentFilter(threshold=0.48, threshold_type="fixed", min_word_threshold=0)
                # content_filter=BM25ContentFilter(user_query=None, bm25_threshold=1.0)
            ),
            verbose=False,
        )
        end = time.time()
        print("Crawl4AI (with JavaScript execution):")
        print(f"Time taken: {end - start:.2f} seconds")
        print(f"Content length: {len(result.markdown)} characters")
        print(f"Fit Markdown: {len(result.markdown_v2.fit_markdown)} characters")
        print(f"Images found: {result.markdown.count('cldnry.s-nbcnews.com')}")

    print("\nNote on Speed Comparison:")
    print("The speed test conducted here may not reflect optimal conditions.")
    print("When we call Firecrawl's API, we're seeing its best performance,")
    print("while Crawl4AI's performance is limited by the local network speed.")
    print("For a more accurate comparison, it's recommended to run these tests")
    print("on servers with a stable and fast internet connection.")
    print("Despite these limitations, Crawl4AI still demonstrates faster performance.")
    print("If you run these tests in an environment with better network conditions,")
    print("you may observe an even more significant speed advantage for Crawl4AI.")

async def generate_knowledge_graph():
    class Entity(BaseModel):
        name: str
        description: str
        
    class Relationship(BaseModel):
        entity1: Entity
        entity2: Entity
        description: str
        relation_type: str

    class KnowledgeGraph(BaseModel):
        entities: List[Entity]
        relationships: List[Relationship]

    extraction_strategy = LLMExtractionStrategy(
            provider='openai/gpt-4o-mini', # Or any other provider, including Ollama and open source models
            api_token=os.getenv('OPENAI_API_KEY'), # In case of Ollama just pass "no-token"
            schema=KnowledgeGraph.model_json_schema(),
            extraction_type="schema",
            instruction="""Extract entities and relationships from the given text."""
    )
    async with AsyncWebCrawler() as crawler:
        url = "https://paulgraham.com/love.html"
        result = await crawler.arun(
            url=url,
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=extraction_strategy,
            # magic=True
        )
        # print(result.extracted_content)
        with open(os.path.join(__location__, "kb.json"), "w") as f:
            f.write(result.extracted_content)

async def fit_markdown_remove_overlay():
    
    async with AsyncWebCrawler(
            headless=True,  # Set to False to see what is happening
            verbose=True,
            user_agent_mode="random",
            user_agent_generator_config={
                "device_type": "mobile",
                "os_type": "android"
            },
    ) as crawler:
        result = await crawler.arun(
            url='https://www.kidocode.com/degrees/technology',
            cache_mode=CacheMode.BYPASS,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=0.48, threshold_type="fixed", min_word_threshold=0
                ),
                options={
                    "ignore_links": True
                }
            ),
            # markdown_generator=DefaultMarkdownGenerator(
            #     content_filter=BM25ContentFilter(user_query="", bm25_threshold=1.0),
            #     options={
            #         "ignore_links": True
            #     }
            # ),
        )
        
        if result.success:
            print(len(result.markdown_v2.raw_markdown))
            print(len(result.markdown_v2.markdown_with_citations))
            print(len(result.markdown_v2.fit_markdown))
            
            # Save clean html
            with open(os.path.join(__location__, "output/cleaned_html.html"), "w") as f:
                f.write(result.cleaned_html)
            
            with open(os.path.join(__location__, "output/output_raw_markdown.md"), "w") as f:
                f.write(result.markdown_v2.raw_markdown)
                
            with open(os.path.join(__location__, "output/output_markdown_with_citations.md"), "w") as f:
                f.write(result.markdown_v2.markdown_with_citations) 
                
            with open(os.path.join(__location__, "output/output_fit_markdown.md"), "w") as f:   
                f.write(result.markdown_v2.fit_markdown)
        
    print("Done")


async def main():
    # await extract_structured_data_using_llm("openai/gpt-4o", os.getenv("OPENAI_API_KEY"))
    
    # await simple_crawl()
    # await simple_example_with_running_js_code()
    # await simple_example_with_css_selector()
    # # await use_proxy()
    # await capture_and_save_screenshot("https://www.example.com", os.path.join(__location__, "tmp/example_screenshot.jpg"))
    # await extract_structured_data_using_css_extractor()

    # LLM extraction examples
    # await extract_structured_data_using_llm()
    # await extract_structured_data_using_llm("huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct", os.getenv("HUGGINGFACE_API_KEY"))
    # await extract_structured_data_using_llm("ollama/llama3.2")    

    # You always can pass custom headers to the extraction strategy
    # custom_headers = {
    #     "Authorization": "Bearer your-custom-token",
    #     "X-Custom-Header": "Some-Value"
    # }
    # await extract_structured_data_using_llm(extra_headers=custom_headers)
    
    await crawl_dynamic_content_pages_method_1()
    await crawl_dynamic_content_pages_method_2()
    await crawl_dynamic_content_pages_method_3()
    
    await crawl_custom_browser_type()
    
    await speed_comparison()


if __name__ == "__main__":
    asyncio.run(main())

```

## docs/examples/quickstart_sync.py
```
import os
import time
from crawl4ai.web_crawler import WebCrawler
from crawl4ai.chunking_strategy import *
from crawl4ai.extraction_strategy import *
from crawl4ai.crawler_strategy import *
from rich import print
from rich.console import Console
from functools import lru_cache

console = Console()

@lru_cache()
def create_crawler():
    crawler = WebCrawler(verbose=True)
    crawler.warmup()
    return crawler

def print_result(result):
    # Print each key in one line and just the first 10 characters of each one's value and three dots
    console.print(f"\t[bold]Result:[/bold]")
    for key, value in result.model_dump().items():
        if isinstance(value, str) and value:
            console.print(f"\t{key}: [green]{value[:20]}...[/green]")
    if result.extracted_content:
        items = json.loads(result.extracted_content)
        print(f"\t[bold]{len(items)} blocks is extracted![/bold]")


def cprint(message, press_any_key=False):
    console.print(message)
    if press_any_key:
        console.print("Press any key to continue...", style="")
        input()

def basic_usage(crawler):
    cprint("🛠️ [bold cyan]Basic Usage: Simply provide a URL and let Crawl4ai do the magic![/bold cyan]")
    result = crawler.run(url="https://www.nbcnews.com/business", only_text = True)
    cprint("[LOG] 📦 [bold yellow]Basic crawl result:[/bold yellow]")
    print_result(result)

def basic_usage_some_params(crawler):
    cprint("🛠️ [bold cyan]Basic Usage: Simply provide a URL and let Crawl4ai do the magic![/bold cyan]")
    result = crawler.run(url="https://www.nbcnews.com/business", word_count_threshold=1, only_text = True)
    cprint("[LOG] 📦 [bold yellow]Basic crawl result:[/bold yellow]")
    print_result(result)

def screenshot_usage(crawler):
    cprint("\n📸 [bold cyan]Let's take a screenshot of the page![/bold cyan]")
    result = crawler.run(url="https://www.nbcnews.com/business", screenshot=True)
    cprint("[LOG] 📦 [bold yellow]Screenshot result:[/bold yellow]")
    # Save the screenshot to a file
    with open("screenshot.png", "wb") as f:
        f.write(base64.b64decode(result.screenshot))
    cprint("Screenshot saved to 'screenshot.png'!")
    print_result(result)

def understanding_parameters(crawler):
    cprint("\n🧠 [bold cyan]Understanding 'bypass_cache' and 'include_raw_html' parameters:[/bold cyan]")
    cprint("By default, Crawl4ai caches the results of your crawls. This means that subsequent crawls of the same URL will be much faster! Let's see this in action.")
    
    # First crawl (reads from cache)
    cprint("1️⃣ First crawl (caches the result):", True)
    start_time = time.time()
    result = crawler.run(url="https://www.nbcnews.com/business")
    end_time = time.time()
    cprint(f"[LOG] 📦 [bold yellow]First crawl took {end_time - start_time} seconds and result (from cache):[/bold yellow]")
    print_result(result)

    # Force to crawl again
    cprint("2️⃣ Second crawl (Force to crawl again):", True)
    start_time = time.time()
    result = crawler.run(url="https://www.nbcnews.com/business", bypass_cache=True)
    end_time = time.time()
    cprint(f"[LOG] 📦 [bold yellow]Second crawl took {end_time - start_time} seconds and result (forced to crawl):[/bold yellow]")
    print_result(result)

def add_chunking_strategy(crawler):
    # Adding a chunking strategy: RegexChunking
    cprint("\n🧩 [bold cyan]Let's add a chunking strategy: RegexChunking![/bold cyan]", True)
    cprint("RegexChunking is a simple chunking strategy that splits the text based on a given regex pattern. Let's see it in action!")
    result = crawler.run(
        url="https://www.nbcnews.com/business",
        chunking_strategy=RegexChunking(patterns=["\n\n"])
    )
    cprint("[LOG] 📦 [bold yellow]RegexChunking result:[/bold yellow]")
    print_result(result)

    # Adding another chunking strategy: NlpSentenceChunking
    cprint("\n🔍 [bold cyan]Time to explore another chunking strategy: NlpSentenceChunking![/bold cyan]", True)
    cprint("NlpSentenceChunking uses NLP techniques to split the text into sentences. Let's see how it performs!")
    result = crawler.run(
        url="https://www.nbcnews.com/business",
        chunking_strategy=NlpSentenceChunking()
    )
    cprint("[LOG] 📦 [bold yellow]NlpSentenceChunking result:[/bold yellow]")
    print_result(result)

def add_extraction_strategy(crawler):
    # Adding an extraction strategy: CosineStrategy
    cprint("\n🧠 [bold cyan]Let's get smarter with an extraction strategy: CosineStrategy![/bold cyan]", True)
    cprint("CosineStrategy uses cosine similarity to extract semantically similar blocks of text. Let's see it in action!")
    result = crawler.run(
        url="https://www.nbcnews.com/business",
        extraction_strategy=CosineStrategy(word_count_threshold=10, max_dist=0.2, linkage_method="ward", top_k=3, sim_threshold = 0.3, verbose=True)
    )
    cprint("[LOG] 📦 [bold yellow]CosineStrategy result:[/bold yellow]")
    print_result(result)
    
    # Using semantic_filter with CosineStrategy
    cprint("You can pass other parameters like 'semantic_filter' to the CosineStrategy to extract semantically similar blocks of text. Let's see it in action!")
    result = crawler.run(
        url="https://www.nbcnews.com/business",
        extraction_strategy=CosineStrategy(
            semantic_filter="inflation rent prices",
        )
    )
    cprint("[LOG] 📦 [bold yellow]CosineStrategy result with semantic filter:[/bold yellow]")
    print_result(result)

def add_llm_extraction_strategy(crawler):
    # Adding an LLM extraction strategy without instructions
    cprint("\n🤖 [bold cyan]Time to bring in the big guns: LLMExtractionStrategy without instructions![/bold cyan]", True)
    cprint("LLMExtractionStrategy uses a large language model to extract relevant information from the web page. Let's see it in action!")
    result = crawler.run(
        url="https://www.nbcnews.com/business",
        extraction_strategy=LLMExtractionStrategy(provider="openai/gpt-4o", api_token=os.getenv('OPENAI_API_KEY'))
    )
    cprint("[LOG] 📦 [bold yellow]LLMExtractionStrategy (no instructions) result:[/bold yellow]")
    print_result(result)
    
    # Adding an LLM extraction strategy with instructions
    cprint("\n📜 [bold cyan]Let's make it even more interesting: LLMExtractionStrategy with instructions![/bold cyan]", True)
    cprint("Let's say we are only interested in financial news. Let's see how LLMExtractionStrategy performs with instructions!")
    result = crawler.run(
        url="https://www.nbcnews.com/business",
        extraction_strategy=LLMExtractionStrategy(
            provider="openai/gpt-4o",
            api_token=os.getenv('OPENAI_API_KEY'),
            instruction="I am interested in only financial news"
        )
    )
    cprint("[LOG] 📦 [bold yellow]LLMExtractionStrategy (with instructions) result:[/bold yellow]")
    print_result(result)
    
    result = crawler.run(
        url="https://www.nbcnews.com/business",
        extraction_strategy=LLMExtractionStrategy(
            provider="openai/gpt-4o",
            api_token=os.getenv('OPENAI_API_KEY'),
            instruction="Extract only content related to technology"
        )
    )
    cprint("[LOG] 📦 [bold yellow]LLMExtractionStrategy (with technology instruction) result:[/bold yellow]")
    print_result(result)

def targeted_extraction(crawler):
    # Using a CSS selector to extract only H2 tags
    cprint("\n🎯 [bold cyan]Targeted extraction: Let's use a CSS selector to extract only H2 tags![/bold cyan]", True)
    result = crawler.run(
        url="https://www.nbcnews.com/business",
        css_selector="h2"
    )
    cprint("[LOG] 📦 [bold yellow]CSS Selector (H2 tags) result:[/bold yellow]")
    print_result(result)

def interactive_extraction(crawler):
    # Passing JavaScript code to interact with the page
    cprint("\n🖱️ [bold cyan]Let's get interactive: Passing JavaScript code to click 'Load More' button![/bold cyan]", True)
    cprint("In this example we try to click the 'Load More' button on the page using JavaScript code.")
    js_code = """
    const loadMoreButton = Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('Load More'));
    loadMoreButton && loadMoreButton.click();
    """
    # crawler_strategy = LocalSeleniumCrawlerStrategy(js_code=js_code)
    # crawler = WebCrawler(crawler_strategy=crawler_strategy, always_by_pass_cache=True)
    result = crawler.run(
        url="https://www.nbcnews.com/business",
        js = js_code
    )
    cprint("[LOG] 📦 [bold yellow]JavaScript Code (Load More button) result:[/bold yellow]")
    print_result(result)

def multiple_scrip(crawler):
    # Passing JavaScript code to interact with the page
    cprint("\n🖱️ [bold cyan]Let's get interactive: Passing JavaScript code to click 'Load More' button![/bold cyan]", True)
    cprint("In this example we try to click the 'Load More' button on the page using JavaScript code.")
    js_code = ["""
    const loadMoreButton = Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('Load More'));
    loadMoreButton && loadMoreButton.click();
    """] * 2
    # crawler_strategy = LocalSeleniumCrawlerStrategy(js_code=js_code)
    # crawler = WebCrawler(crawler_strategy=crawler_strategy, always_by_pass_cache=True)
    result = crawler.run(
        url="https://www.nbcnews.com/business",
        js = js_code  
    )
    cprint("[LOG] 📦 [bold yellow]JavaScript Code (Load More button) result:[/bold yellow]")
    print_result(result)

def using_crawler_hooks(crawler):
    # Example usage of the hooks for authentication and setting a cookie
    def on_driver_created(driver):
        print("[HOOK] on_driver_created")
        # Example customization: maximize the window
        driver.maximize_window()
        
        # Example customization: logging in to a hypothetical website
        driver.get('https://example.com/login')
        
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, 'username'))
        )
        driver.find_element(By.NAME, 'username').send_keys('testuser')
        driver.find_element(By.NAME, 'password').send_keys('password123')
        driver.find_element(By.NAME, 'login').click()
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'welcome'))
        )
        # Add a custom cookie
        driver.add_cookie({'name': 'test_cookie', 'value': 'cookie_value'})
        return driver        
        

    def before_get_url(driver):
        print("[HOOK] before_get_url")
        # Example customization: add a custom header
        # Enable Network domain for sending headers
        driver.execute_cdp_cmd('Network.enable', {})
        # Add a custom header
        driver.execute_cdp_cmd('Network.setExtraHTTPHeaders', {'headers': {'X-Test-Header': 'test'}})
        return driver
    
    def after_get_url(driver):
        print("[HOOK] after_get_url")
        # Example customization: log the URL
        print(driver.current_url)
        return driver

    def before_return_html(driver, html):
        print("[HOOK] before_return_html")
        # Example customization: log the HTML
        print(len(html))
        return driver
    
    cprint("\n🔗 [bold cyan]Using Crawler Hooks: Let's see how we can customize the crawler using hooks![/bold cyan]", True)
    
    crawler_strategy = LocalSeleniumCrawlerStrategy(verbose=True)
    crawler_strategy.set_hook('on_driver_created', on_driver_created)
    crawler_strategy.set_hook('before_get_url', before_get_url)
    crawler_strategy.set_hook('after_get_url', after_get_url)
    crawler_strategy.set_hook('before_return_html', before_return_html)
    
    crawler = WebCrawler(verbose=True, crawler_strategy=crawler_strategy)
    crawler.warmup()    
    result = crawler.run(url="https://example.com")
    
    cprint("[LOG] 📦 [bold yellow]Crawler Hooks result:[/bold yellow]")
    print_result(result= result)
    
def using_crawler_hooks_dleay_example(crawler):
    def delay(driver):
        print("Delaying for 5 seconds...")
        time.sleep(5)
        print("Resuming...")
        
    def create_crawler():
        crawler_strategy = LocalSeleniumCrawlerStrategy(verbose=True)
        crawler_strategy.set_hook('after_get_url', delay)
        crawler = WebCrawler(verbose=True, crawler_strategy=crawler_strategy)
        crawler.warmup()
        return crawler

    cprint("\n🔗 [bold cyan]Using Crawler Hooks: Let's add a delay after fetching the url to make sure entire page is fetched.[/bold cyan]")
    crawler = create_crawler()
    result = crawler.run(url="https://google.com", bypass_cache=True)    
    
    cprint("[LOG] 📦 [bold yellow]Crawler Hooks result:[/bold yellow]")
    print_result(result)
    
    

def main():
    cprint("🌟 [bold green]Welcome to the Crawl4ai Quickstart Guide! Let's dive into some web crawling fun! 🌐[/bold green]")
    cprint("⛳️ [bold cyan]First Step: Create an instance of WebCrawler and call the `warmup()` function.[/bold cyan]")
    cprint("If this is the first time you're running Crawl4ai, this might take a few seconds to load required model files.")

    crawler = create_crawler()

    crawler.always_by_pass_cache = True
    basic_usage(crawler)
    # basic_usage_some_params(crawler)
    understanding_parameters(crawler)
    
    crawler.always_by_pass_cache = True
    screenshot_usage(crawler)
    add_chunking_strategy(crawler)
    add_extraction_strategy(crawler)
    add_llm_extraction_strategy(crawler)
    targeted_extraction(crawler)
    interactive_extraction(crawler)
    multiple_scrip(crawler)

    cprint("\n🎉 [bold green]Congratulations! You've made it through the Crawl4ai Quickstart Guide! Now go forth and crawl the web like a pro! 🕸️[/bold green]")

if __name__ == "__main__":
    main()


```

## docs/examples/language_support_example.py
```
import asyncio
from crawl4ai import AsyncWebCrawler, AsyncPlaywrightCrawlerStrategy

async def main():
    # Example 1: Setting language when creating the crawler
    crawler1 = AsyncWebCrawler(
        crawler_strategy=AsyncPlaywrightCrawlerStrategy(
            headers={"Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7"}
        )
    )
    result1 = await crawler1.arun("https://www.example.com")
    print("Example 1 result:", result1.extracted_content[:100])  # Print first 100 characters

    # Example 2: Setting language before crawling
    crawler2 = AsyncWebCrawler()
    crawler2.crawler_strategy.headers["Accept-Language"] = "es-ES,es;q=0.9,en-US;q=0.8,en;q=0.7"
    result2 = await crawler2.arun("https://www.example.com")
    print("Example 2 result:", result2.extracted_content[:100])

    # Example 3: Setting language when calling arun method
    crawler3 = AsyncWebCrawler()
    result3 = await crawler3.arun(
        "https://www.example.com",
        headers={"Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7"}
    )
    print("Example 3 result:", result3.extracted_content[:100])

    # Example 4: Crawling multiple pages with different languages
    urls = [
        ("https://www.example.com", "fr-FR,fr;q=0.9"),
        ("https://www.example.org", "es-ES,es;q=0.9"),
        ("https://www.example.net", "de-DE,de;q=0.9"),
    ]
    
    crawler4 = AsyncWebCrawler()
    results = await asyncio.gather(*[
        crawler4.arun(url, headers={"Accept-Language": lang})
        for url, lang in urls
    ])
    
    for url, result in zip([u for u, _ in urls], results):
        print(f"Result for {url}:", result.extracted_content[:100])

if __name__ == "__main__":
    asyncio.run(main())
```

## docs/examples/tmp/research_assistant_audio_not_completed.py
```
# Make sure to install the required packageschainlit and groq
import os, time
from openai import AsyncOpenAI
import chainlit as cl
import re
import requests
from io import BytesIO
from chainlit.element import ElementBased
from groq import Groq

# Import threadpools to run the crawl_url function in a separate thread
from concurrent.futures import ThreadPoolExecutor

client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))

# Instrument the OpenAI client
cl.instrument_openai()

settings = {
    "model": "llama3-8b-8192",
    "temperature": 0.5,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

def extract_urls(text):
    url_pattern = re.compile(r'(https?://\S+)')
    return url_pattern.findall(text)

def crawl_url(url):
    data = {
        "urls": [url],
        "include_raw_html": True,
        "word_count_threshold": 10,
        "extraction_strategy": "NoExtractionStrategy",
        "chunking_strategy": "RegexChunking"
    }
    response = requests.post("https://crawl4ai.com/crawl", json=data)
    response_data = response.json()
    response_data = response_data['results'][0]
    return response_data['markdown']

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("session", {
        "history": [],
        "context": {}
    })  
    await cl.Message(
        content="Welcome to the chat! How can I assist you today?"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    user_session = cl.user_session.get("session")
    
    # Extract URLs from the user's message
    urls = extract_urls(message.content)
    
    
    futures = []
    with ThreadPoolExecutor() as executor:
        for url in urls:
            futures.append(executor.submit(crawl_url, url))

    results = [future.result() for future in futures]

    for url, result in zip(urls, results):
        ref_number = f"REF_{len(user_session['context']) + 1}"
        user_session["context"][ref_number] = {
            "url": url,
            "content": result
        }    
    
    # for url in urls:
    #     # Crawl the content of each URL and add it to the session context with a reference number
    #     ref_number = f"REF_{len(user_session['context']) + 1}"
    #     crawled_content = crawl_url(url)
    #     user_session["context"][ref_number] = {
    #         "url": url,
    #         "content": crawled_content
    #     }

    user_session["history"].append({
        "role": "user",
        "content": message.content
    })

    # Create a system message that includes the context
    context_messages = [
        f'<appendix ref="{ref}">\n{data["content"]}\n</appendix>'
        for ref, data in user_session["context"].items()
    ]
    if context_messages:
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful bot. Use the following context for answering questions. "
                "Refer to the sources using the REF number in square brackets, e.g., [1], only if the source is given in the appendices below.\n\n"
                "If the question requires any information from the provided appendices or context, refer to the sources. "
                "If not, there is no need to add a references section. "
                "At the end of your response, provide a reference section listing the URLs and their REF numbers only if sources from the appendices were used.\n\n"
                "\n\n".join(context_messages)
            )
        }
    else:
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant."
        }


    msg = cl.Message(content="")
    await msg.send()

    # Get response from the LLM
    stream = await client.chat.completions.create(
        messages=[
            system_message,
            *user_session["history"]
        ],
        stream=True,
        **settings
    )

    assistant_response = ""
    async for part in stream:
        if token := part.choices[0].delta.content:
            assistant_response += token
            await msg.stream_token(token)

    # Add assistant message to the history
    user_session["history"].append({
        "role": "assistant",
        "content": assistant_response
    })
    await msg.update()

    # Append the reference section to the assistant's response
    reference_section = "\n\nReferences:\n"
    for ref, data in user_session["context"].items():
        reference_section += f"[{ref.split('_')[1]}]: {data['url']}\n"

    msg.content += reference_section
    await msg.update()


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # Write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)

    pass

@cl.step(type="tool")
async def speech_to_text(audio_file):
    cli = Groq()
    
    # response = cli.audio.transcriptions.create(
    #     file=audio_file, #(filename, file.read()),
    #     model="whisper-large-v3",
    # )
    
    response = await client.audio.transcriptions.create(
        model="whisper-large-v3", file=audio_file
    )

    return response.text


@cl.on_audio_end
async def on_audio_end(elements: list[ElementBased]):
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")

    # input_audio_el = cl.Audio(
    #     mime=audio_mime_type, content=audio_file, name=audio_buffer.name
    # )
    # await cl.Message(
    #     author="You", 
    #     type="user_message",
    #     content="",
    #     elements=[input_audio_el, *elements]
    # ).send()
    
    # answer_message = await cl.Message(content="").send()
    
    
    start_time = time.time()
    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)
    end_time = time.time()
    print(f"Transcription took {end_time - start_time} seconds")
    
    user_msg = cl.Message(
        author="You", 
        type="user_message",
        content=transcription
    )
    await user_msg.send()
    await on_message(user_msg)

    # images = [file for file in elements if "image" in file.mime]

    # text_answer = await generate_text_answer(transcription, images)
    
    # output_name, output_audio = await text_to_speech(text_answer, audio_mime_type)
    
    # output_audio_el = cl.Audio(
    #     name=output_name,
    #     auto_play=True,
    #     mime=audio_mime_type,
    #     content=output_audio,
    # )
    
    # answer_message.elements = [output_audio_el]
    
    # answer_message.content = transcription
    # await answer_message.update()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)



```

## docs/examples/tmp/chainlit_review.py
```
from openai import AsyncOpenAI
from chainlit.types import ThreadDict
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider
client = AsyncOpenAI()

# Instrument the OpenAI client
cl.instrument_openai()

settings = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.5,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

@cl.action_callback("action_button")
async def on_action(action: cl.Action):
    print("The user clicked on the action button!")

    return "Thank you for clicking on the action button!"

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="GPT-3.5",
            markdown_description="The underlying LLM model is **GPT-3.5**.",
            icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name="GPT-4",
            markdown_description="The underlying LLM model is **GPT-4**.",
            icon="https://picsum.photos/250",
        ),
    ]

@cl.on_chat_start
async def on_chat_start():
    
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
                initial_index=0,
            ),
            Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=1,
                min=0,
                max=2,
                step=0.1,
            ),
            Slider(
                id="SAI_Steps",
                label="Stability AI - Steps",
                initial=30,
                min=10,
                max=150,
                step=1,
                description="Amount of inference steps performed on image generation.",
            ),
            Slider(
                id="SAI_Cfg_Scale",
                label="Stability AI - Cfg_Scale",
                initial=7,
                min=1,
                max=35,
                step=0.1,
                description="Influences how strongly your generation is guided to match your prompt.",
            ),
            Slider(
                id="SAI_Width",
                label="Stability AI - Image Width",
                initial=512,
                min=256,
                max=2048,
                step=64,
                tooltip="Measured in pixels",
            ),
            Slider(
                id="SAI_Height",
                label="Stability AI - Image Height",
                initial=512,
                min=256,
                max=2048,
                step=64,
                tooltip="Measured in pixels",
            ),
        ]
    ).send()
    
    chat_profile = cl.user_session.get("chat_profile")
    await cl.Message(
        content=f"starting chat using the {chat_profile} chat profile"
    ).send()
    
    print("A new chat session has started!")
    cl.user_session.set("session", {
        "history": [],
        "context": []
    })  
    
    image = cl.Image(url="https://c.tenor.com/uzWDSSLMCmkAAAAd/tenor.gif", name="cat image", display="inline")

    # Attach the image to the message
    await cl.Message(
        content="You are such a good girl, aren't you?!",
        elements=[image],
    ).send()
    
    text_content = "Hello, this is a text element."
    elements = [
        cl.Text(name="simple_text", content=text_content, display="inline")
    ]

    await cl.Message(
        content="Check out this text element!",
        elements=elements,
    ).send()
    
    elements = [
        cl.Audio(path="./assets/audio.mp3", display="inline"),
    ]
    await cl.Message(
        content="Here is an audio file",
        elements=elements,
    ).send()
    
    await cl.Avatar(
        name="Tool 1",
        url="https://avatars.githubusercontent.com/u/128686189?s=400&u=a1d1553023f8ea0921fba0debbe92a8c5f840dd9&v=4",
    ).send()
    
    await cl.Message(
        content="This message should not have an avatar!", author="Tool 0"
    ).send()
    
    await cl.Message(
        content="This message should have an avatar!", author="Tool 1"
    ).send()
    
    elements = [
        cl.File(
            name="quickstart.py",
            path="./quickstart.py",
            display="inline",
        ),
    ]

    await cl.Message(
        content="This message has a file element", elements=elements
    ).send()
    
    # Sending an action button within a chatbot message
    actions = [
        cl.Action(name="action_button", value="example_value", description="Click me!")
    ]

    await cl.Message(content="Interact with this action button:", actions=actions).send()
    
    # res = await cl.AskActionMessage(
    #     content="Pick an action!",
    #     actions=[
    #         cl.Action(name="continue", value="continue", label="✅ Continue"),
    #         cl.Action(name="cancel", value="cancel", label="❌ Cancel"),
    #     ],
    # ).send()

    # if res and res.get("value") == "continue":
    #     await cl.Message(
    #         content="Continue!",
    #     ).send()
    
    # import plotly.graph_objects as go
    # fig = go.Figure(
    #     data=[go.Bar(y=[2, 1, 3])],
    #     layout_title_text="An example figure",
    # )
    # elements = [cl.Plotly(name="chart", figure=fig, display="inline")]

    # await cl.Message(content="This message has a chart", elements=elements).send()
    
    # Sending a pdf with the local file path
    # elements = [
    #   cl.Pdf(name="pdf1", display="inline", path="./pdf1.pdf")
    # ]

    # cl.Message(content="Look at this local pdf!", elements=elements).send()    

@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    
@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")

@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    print("The user resumed a previous chat session!")




# @cl.on_message
async def on_message(message: cl.Message):
    cl.user_session.get("session")["history"].append({
        "role": "user",
        "content": message.content
    })    
    response = await client.chat.completions.create(
        messages=[
            {
                "content": "You are a helpful bot",
                "role": "system"
            },
            *cl.user_session.get("session")["history"]
        ],
        **settings
    )
    

    # Add assitanr message to the history
    cl.user_session.get("session")["history"].append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    
    # msg.content = response.choices[0].message.content
    # await msg.update()
    
    # await cl.Message(content=response.choices[0].message.content).send()

@cl.on_message
async def on_message(message: cl.Message):
    cl.user_session.get("session")["history"].append({
        "role": "user",
        "content": message.content
    })    

    msg = cl.Message(content="")
    await msg.send()    
    
    stream = await client.chat.completions.create(
        messages=[
            {
                "content": "You are a helpful bot",
                "role": "system"
            },
            *cl.user_session.get("session")["history"]
        ],
        stream = True, 
        **settings
    )
    
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)
    
    # Add assitanr message to the history
    cl.user_session.get("session")["history"].append({
        "role": "assistant",
        "content": msg.content
    })    
    await msg.update()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
```
