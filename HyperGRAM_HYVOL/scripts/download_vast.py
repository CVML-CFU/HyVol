#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "rich",
#     "tyro",
#     "httpx",
# ]
# ///
"""
Downloader for the VAST-27M 150k subset (VAST150K).

Design notes:
- Uses as_completed() instead of wait() to avoid deadlocks.
- Drops the explicit Semaphore and relies on the ThreadPoolExecutor limit.
- Single background thread refreshes the proxy pool.
- Keeps the error classification and retry logic.
- Keeps the Rich progress display.
"""

import json
import os
import subprocess
import shlex
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import hashlib
from datetime import datetime
import threading
from queue import Queue, PriorityQueue, Empty
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set, Any
import random
import sys
import shutil  # For terminal size detection

# Rich imports for progress display
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    ProgressColumn,
    Text
)
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich import print as rprint

# tyro for CLI
import tyro

# Initialize console
console = Console()

# ============================================================================
# Error Categories and Classification
# ============================================================================

@dataclass
class FailureRecord:
    """Record of a failed download attempt"""
    video_id: str
    error_type: str
    error_message: str
    attempts: int = 1
    last_attempt: str = ""
    retry_eligible: bool = True
    cookies_used: bool = False
    fallback_tried: bool = False


class ErrorCategories:
    """Error category constants with emojis for display"""
    # Permanent errors (should not retry)
    PRIVATE = "private"
    DELETED = "deleted"
    NOT_FOUND = "not_found"
    AGE_RESTRICTED = "age_restricted"
    REGION_BLOCKED = "region_blocked"
    COPYRIGHT = "copyright"
    MEMBERS = "members"
    LIVESTREAM = "livestream"

    # Temporary errors (should retry)
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    SIGNIN = "signin"
    UNAVAILABLE = "unavailable"  # Generic unavailable, might be temporary

    # Processing errors
    FFMPEG = "ffmpeg"
    FILESYSTEM = "filesystem"

    # Other
    UNEXPECTED = "unexpected"
    UNKNOWN = "unknown"

    @classmethod
    def get_emoji(cls, category):
        """Get emoji for error category"""
        emoji_map = {
            cls.PRIVATE: "🔒",
            cls.DELETED: "🗑️",
            cls.NOT_FOUND: "❓",
            cls.AGE_RESTRICTED: "🔞",
            cls.REGION_BLOCKED: "🌍",
            cls.COPYRIGHT: "©️",
            cls.MEMBERS: "👥",
            cls.LIVESTREAM: "🔴",
            cls.NETWORK: "🌐",
            cls.TIMEOUT: "⏱️",
            cls.RATE_LIMITED: "🚦",
            cls.SIGNIN: "🔑",
            cls.UNAVAILABLE: "⚠️",
            cls.FFMPEG: "🎬",
            cls.FILESYSTEM: "💾",
            cls.UNEXPECTED: "💥",
            cls.UNKNOWN: "❌"
        }
        return emoji_map.get(category, "❌")

    @classmethod
    def is_permanent(cls, category):
        """Check if error is permanent (should not retry)"""
        return category in [
            cls.PRIVATE, cls.DELETED, cls.NOT_FOUND,
            cls.AGE_RESTRICTED, cls.REGION_BLOCKED,
            cls.COPYRIGHT, cls.MEMBERS, cls.LIVESTREAM
        ]

    @classmethod
    def is_temporary(cls, category):
        """Check if error is temporary (should retry)"""
        return category in [
            cls.NETWORK, cls.TIMEOUT, cls.RATE_LIMITED, cls.SIGNIN,
            cls.UNAVAILABLE
        ]


def classify_error(error_msg: str) -> str:
    """Classify error message into category"""
    if not error_msg:
        return ErrorCategories.UNKNOWN

    error_lower = error_msg.lower()

    # Check for specific error patterns (order matters - most specific first!)

    # Copyright/DMCA
    if any(term in error_lower for term in [
        "copyright", "dmca", "content from", "blocked it on copyright",
        "copyright claim", "copyright owner"
    ]):
        return ErrorCategories.COPYRIGHT

    # Members-only content
    elif any(term in error_lower for term in [
        "members only", "member-only", "channel membership",
        "join this channel", "join to get access"
    ]):
        return ErrorCategories.MEMBERS

    # Live stream specific
    elif any(term in error_lower for term in [
        "waiting for stream", "stream offline", "premiere will begin",
        "live event will begin", "scheduled for", "premieres in"
    ]):
        return ErrorCategories.LIVESTREAM

    # Private video
    elif "private" in error_lower:
        return ErrorCategories.PRIVATE

    # Deleted/Removed
    elif any(term in error_lower for term in [
        "removed", "deleted", "no longer available", "taken down"
    ]):
        return ErrorCategories.DELETED

    # Not found
    elif "not found" in error_lower or "404" in error_lower or "doesn't exist" in error_lower:
        return ErrorCategories.NOT_FOUND

    # Age restriction (more specific patterns)
    elif any(pattern in error_lower for pattern in [
        "age-restricted", "age restricted", "age_restricted",
        "confirm your age", "age verification",
        "inappropriate for some", "adult content",
        "mature content"
    ]):
        return ErrorCategories.AGE_RESTRICTED

    # Region blocking
    elif ("not available" in error_lower or "unavailable" in error_lower) and \
         any(term in error_lower for term in ["country", "region", "location", "territory"]):
        return ErrorCategories.REGION_BLOCKED

    # Sign-in required (but not age-related)
    elif any(term in error_lower for term in [
        "sign in", "sign-in", "signin", "log in", "login",
        "authenticate", "authentication required"
    ]):
        return ErrorCategories.SIGNIN

    # Network errors
    elif any(term in error_lower for term in [
        "network", "connection", "unable to connect", "connection reset",
        "connection refused", "ssl error", "certificate"
    ]):
        return ErrorCategories.NETWORK

    # Timeout
    elif "timeout" in error_lower or "timed out" in error_lower:
        return ErrorCategories.TIMEOUT

    # Rate limiting
    elif any(term in error_lower for term in [
        "429", "rate limit", "too many requests", "quota exceeded"
    ]):
        return ErrorCategories.RATE_LIMITED

    # FFmpeg errors
    elif "ffmpeg" in error_lower or "conversion" in error_lower:
        return ErrorCategories.FFMPEG

    # Filesystem errors
    elif any(term in error_lower for term in [
        "permission denied", "disk full", "no space", "file system",
        "cannot write", "access denied"
    ]):
        return ErrorCategories.FILESYSTEM

    # Generic unavailable (could be temporary)
    elif "unavailable" in error_lower or "not available" in error_lower:
        return ErrorCategories.UNAVAILABLE

    return ErrorCategories.UNKNOWN


# ============================================================================
# Retry Management
# ============================================================================

class RetryManager:
    """Manages retry logic with exponential backoff"""

    def __init__(self, max_retries=3, base_delay=5.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retry_counts = defaultdict(int)
        self.retry_queue = PriorityQueue()
        self.lock = threading.Lock()

    def should_retry(self, error_category: str, video_id: str, cookies_active: bool) -> bool:
        """Determine if a video should be retried"""
        # Don't retry permanent errors
        if ErrorCategories.is_permanent(error_category):
            return False

        # Check retry count
        key = f"{video_id}_{error_category}"
        with self.lock:
            count = self.retry_counts[key]
            if count >= self.max_retries:
                return False

        # Special case: sign-in errors without cookies
        if error_category == ErrorCategories.SIGNIN and not cookies_active:
            return False

        return True

    def add_to_retry(self, video_id: str, error_category: str, error_msg: str, cookies_active: bool):
        """Add video to retry queue with exponential backoff"""
        key = f"{video_id}_{error_category}"

        with self.lock:
            self.retry_counts[key] += 1
            count = self.retry_counts[key]

            # Calculate retry time with exponential backoff
            delay = self.base_delay * (2 ** (count - 1))
            retry_time = time.time() + delay

            # Add to priority queue (priority = retry_time)
            self.retry_queue.put((retry_time, video_id, {
                'error_category': error_category,
                'error_msg': error_msg,
                'attempt': count,
                'is_retry': True
            }))

    def get_ready_retries(self) -> List[Tuple[str, Dict]]:
        """Get videos ready for retry"""
        ready = []
        current_time = time.time()

        with self.lock:
            # Get all items ready for retry
            while not self.retry_queue.empty():
                retry_time, video_id, metadata = self.retry_queue.get()
                if retry_time <= current_time:
                    ready.append((video_id, metadata))
                else:
                    # Put it back if not ready
                    self.retry_queue.put((retry_time, video_id, metadata))
                    break

        return ready

    def get_stats(self) -> Dict[str, int]:
        """Get retry statistics"""
        with self.lock:
            stats = {}
            for key, count in self.retry_counts.items():
                _, category = key.rsplit('_', 1)
                stats[category] = stats.get(category, 0) + 1
            return stats


# ============================================================================
# Sign-in Error Monitoring
# ============================================================================

class SigninErrorMonitor:
    """Monitor consecutive sign-in errors and trigger pause when threshold reached"""

    def __init__(self, threshold: int = 20, pause_duration: int = 900, enabled: bool = True):
        self.threshold = threshold
        self.pause_duration = pause_duration
        self.enabled = enabled
        self.consecutive_errors = 0
        self.pause_end_time = None
        self.pause_count = 0
        self.total_signin_errors = 0
        self.lock = threading.Lock()

    def record_error(self, error_category: str) -> bool:
        """Record an error and return True if pause should be triggered"""
        if not self.enabled:
            return False

        with self.lock:
            # Don't count errors during pause period
            if self.pause_end_time and time.time() < self.pause_end_time:
                return False

            if error_category == ErrorCategories.SIGNIN:
                self.consecutive_errors += 1
                self.total_signin_errors += 1

                if self.consecutive_errors >= self.threshold:
                    self.trigger_pause()
                    return True
            # Don't reset on other errors - signin issue may persist
            return False

    def record_success(self):
        """Record a successful download, resetting the consecutive error count"""
        with self.lock:
            self.consecutive_errors = 0

    def trigger_pause(self):
        """Trigger a pause period"""
        self.pause_end_time = time.time() + self.pause_duration
        self.pause_count += 1
        self.consecutive_errors = 0  # Reset after triggering pause

    def is_paused(self) -> bool:
        """Check if currently in pause period"""
        if not self.enabled:
            return False

        with self.lock:
            if self.pause_end_time:
                if time.time() < self.pause_end_time:
                    return True
                else:
                    # Pause period ended - reset everything
                    self.pause_end_time = None
                    self.consecutive_errors = 0  # Clear counter for fresh start
            return False

    def get_remaining_pause_time(self) -> float:
        """Get remaining pause time in seconds"""
        with self.lock:
            if self.pause_end_time:
                remaining = self.pause_end_time - time.time()
                return max(0, remaining)
            return 0

    def get_resume_time(self) -> str:
        """Get formatted resume time"""
        with self.lock:
            if self.pause_end_time:
                resume_time = datetime.fromtimestamp(self.pause_end_time)
                return resume_time.strftime("%H:%M:%S")
            return "N/A"

    def get_status(self) -> Dict[str, Any]:
        """Get current monitor status"""
        with self.lock:
            return {
                'consecutive_errors': self.consecutive_errors,
                'total_signin_errors': self.total_signin_errors,
                'is_paused': self.is_paused(),
                'remaining_pause_time': self.get_remaining_pause_time(),
                'pause_count': self.pause_count
            }


# ============================================================================
# Proxy Management
# ============================================================================

class ProxyPool:
    """Simple proxy pool with background refresh"""

    def __init__(self, refresh_interval=3600, blacklist_ttl=1800):
        self.proxies = deque()
        self.blacklist = {}  # Changed to dict to store timestamp
        self.blacklist_ttl = blacklist_ttl  # 30 minutes TTL for blacklisted proxies
        self.lock = threading.Lock()
        self.refresh_interval = refresh_interval
        self.last_refresh = 0
        self.min_proxies = 10
        self.stop_event = threading.Event()

        # Start background refresh thread
        self.refresh_thread = threading.Thread(target=self._background_refresh, daemon=True)
        self.refresh_thread.start()

    def _background_refresh(self):
        """Background thread to refresh proxy pool"""
        while not self.stop_event.is_set():
            with self.lock:
                current_time = time.time()
                need_refresh = (
                    len(self.proxies) < self.min_proxies or
                    current_time - self.last_refresh > self.refresh_interval
                )

            if need_refresh:
                self._fetch_proxies()

            # Check every 30 seconds
            self.stop_event.wait(30)

    def _fetch_proxies(self):
        """Fetch new proxies from API or file"""
        # Clean expired blacklist entries first
        self._clean_blacklist()

        new_proxies = []

        # Try loading from local file first
        proxy_file = Path('proxylist.txt')
        if proxy_file.exists():
            try:
                with open(proxy_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and ':' in line:
                            if not line.startswith('http'):
                                line = f'http://{line}'
                            new_proxies.append(line)
            except Exception as e:
                print(f"Error loading proxy file: {e}")

        # Try API if no local proxies
        if not new_proxies:
            try:
                import httpx
                url = "https://www.proxy-list.download/api/v1/get?type=http"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }
                with httpx.Client(timeout=10.0, headers=headers) as client:
                    response = client.get(url)
                    if response.status_code == 200:
                        proxy_text = response.text.strip()
                        for line in proxy_text.split('\n'):
                            if line and ':' in line:
                                new_proxies.append(f"http://{line.strip()}")
                    else:
                        print(f"Failed to fetch proxies: HTTP {response.status_code}")
            except Exception as e:
                print(f"Failed to fetch proxies from API: {e}")

        # Update proxy pool
        if new_proxies:
            with self.lock:
                # Remove blacklisted proxies
                valid_proxies = [p for p in new_proxies if p not in self.blacklist]
                self.proxies = deque(valid_proxies)
                self.last_refresh = time.time()
                print(f"Proxy pool refreshed: {len(self.proxies)} available")

    def get_proxy(self) -> Optional[str]:
        """Get next available proxy"""
        with self.lock:
            if self.proxies:
                proxy = self.proxies.popleft()
                self.proxies.append(proxy)  # Rotate to back
                return proxy
            return None

    def blacklist_proxy(self, proxy: str):
        """Add proxy to blacklist with timestamp"""
        with self.lock:
            self.blacklist[proxy] = time.time()  # Store timestamp
            if proxy in self.proxies:
                self.proxies.remove(proxy)

    def _clean_blacklist(self):
        """Remove expired entries from blacklist"""
        current_time = time.time()
        expired = []
        for proxy, timestamp in list(self.blacklist.items()):
            if current_time - timestamp > self.blacklist_ttl:
                expired.append(proxy)
        for proxy in expired:
            del self.blacklist[proxy]
        if expired:
            print(f"[cyan]Removed {len(expired)} expired proxies from blacklist[/cyan]")

    def force_refresh(self):
        """Force immediate proxy pool refresh"""
        with self.lock:
            console.print("[cyan]Force refreshing proxy pool due to sign-in errors...[/cyan]")
            self.last_refresh = 0  # Reset to trigger immediate refresh
        self._fetch_proxies()

    def stop(self):
        """Stop background thread"""
        self.stop_event.set()
        self.refresh_thread.join(timeout=1)


class ProxyManager:
    """Manages proxy assignment to workers"""

    def __init__(self, num_workers: int, enable_proxy: bool = True):
        self.num_workers = num_workers
        self.enable_proxy = enable_proxy
        self.worker_proxies = {}
        self.proxy_pool = None

        if enable_proxy:
            self.proxy_pool = ProxyPool()
            # Assign proxies to workers (first 2 workers don't use proxy)
            for worker_id in range(num_workers):
                if worker_id < 2:
                    self.worker_proxies[worker_id] = None
                else:
                    self.worker_proxies[worker_id] = self.proxy_pool.get_proxy()

    def get_proxy_for_worker(self, worker_id: int) -> Optional[str]:
        """Get proxy for specific worker"""
        return self.worker_proxies.get(worker_id)

    def rotate_proxy_for_worker(self, worker_id: int):
        """Rotate proxy for worker after failure"""
        if worker_id < 2 or not self.proxy_pool:
            return None

        old_proxy = self.worker_proxies.get(worker_id)
        if old_proxy:
            self.proxy_pool.blacklist_proxy(old_proxy)

        new_proxy = self.proxy_pool.get_proxy()
        self.worker_proxies[worker_id] = new_proxy
        return new_proxy

    def refresh_proxy_pool(self):
        """Force refresh proxy pool"""
        if self.proxy_pool:
            self.proxy_pool.force_refresh()

    def stop(self):
        """Stop proxy pool"""
        if self.proxy_pool:
            self.proxy_pool.stop()


# ============================================================================
# Cookie Management
# ============================================================================

def verify_cookies(cookies_file: Optional[str], cookies_browser: Optional[str], use_uvx: bool = True) -> bool:
    """Verify if cookies are valid"""
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo" - first YouTube video

    if use_uvx:
        base_cmd = ["uvx", "yt-dlp"]
    else:
        base_cmd = ["yt-dlp"]

    cmd = base_cmd + ["--quiet", "--skip-download", "--no-warnings"]

    if cookies_file:
        cmd.extend(["--cookies", cookies_file])
    elif cookies_browser:
        cmd.extend(["--cookies-from-browser", cookies_browser])
    else:
        return False

    cmd.append(test_url)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except Exception:
        return False


# ============================================================================
# Download Functions
# ============================================================================

def download_video(video_id: str, output_dir: str, quality: str = '720p',
                  max_retries: int = 3, use_uvx: bool = True,
                  fallback_mode: bool = False, cookies_file: Optional[str] = None,
                  cookies_from_browser: Optional[str] = None, proxy: Optional[str] = None,
                  debug: bool = False) -> Tuple[bool, str, Optional[str]]:
    """Download a video from YouTube"""

    url = f"https://www.youtube.com/watch?v={video_id}"
    output_path = os.path.join(output_dir, f"{video_id}.mp4")

    # Check if already exists
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size > 1000:  # At least 1KB
            return True, "Already exists", None

    # Build yt-dlp command
    if use_uvx:
        cmd = ["uvx", "yt-dlp"]
    else:
        cmd = ["yt-dlp"]

    # Add quality selection
    if fallback_mode:
        cmd.extend([
            "-f", "best[ext=mp4]/best",
            "--no-check-certificate"
        ])
    else:
        height = quality.rstrip('p')
        cmd.extend([
            "-f", f"best[height<={height}][ext=mp4]/best[height<={height}]/best[ext=mp4]/best"
        ])

    # Add output options
    cmd.extend([
        "-o", output_path,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--no-progress"
    ])

    # Add cookies if provided
    if cookies_file:
        cmd.extend(["--cookies", cookies_file])
    elif cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])

    # Add proxy if provided
    if proxy:
        cmd.extend(["--proxy", proxy])

    # Add URL
    cmd.append(url)

    # Execute download
    try:
        if debug:
            print(f"[DEBUG] Downloading {video_id} with command: {' '.join(cmd[:3])}...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            # Verify file was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                return True, "Success", None
            else:
                return False, "File not created", ErrorCategories.FILESYSTEM
        else:
            # Parse error
            error_msg = result.stderr or result.stdout or "Unknown error"
            error_category = classify_error(error_msg)

            if debug:
                print(f"[DEBUG] Download failed for {video_id}: {error_msg[:100]}")

            return False, error_msg[:200], error_category

    except subprocess.TimeoutExpired:
        return False, "Download timeout", ErrorCategories.TIMEOUT
    except Exception as e:
        return False, str(e), ErrorCategories.UNEXPECTED


def convert_to_audio(video_path: str, audio_dir: str, sample_rate: int = 16000) -> Tuple[bool, Optional[str], Optional[str]]:
    """Convert video to audio using ffmpeg"""

    video_name = os.path.basename(video_path)
    video_id = os.path.splitext(video_name)[0]
    audio_path = os.path.join(audio_dir, f"{video_id}.wav")

    # Check if already exists
    if os.path.exists(audio_path):
        file_size = os.path.getsize(audio_path)
        if file_size > 1000:  # At least 1KB
            return True, audio_path, None

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-y",  # Overwrite
        audio_path
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )

        if result.returncode == 0:
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                return True, audio_path, None
            else:
                return False, None, "Audio file not created"
        else:
            error_msg = result.stderr or "FFmpeg error"
            return False, None, error_msg[:200]

    except subprocess.TimeoutExpired:
        return False, None, "FFmpeg timeout"
    except Exception as e:
        return False, None, str(e)


def process_video(video_id: str, video_dir: str, audio_dir: str,
                 quality: str = '720p', skip_existing: bool = True,
                 use_uvx: bool = True, cookies_file: Optional[str] = None,
                 cookies_from_browser: Optional[str] = None,
                 proxy: Optional[str] = None, worker_id: Optional[int] = None,
                 debug: bool = False, is_retry: bool = False) -> Tuple[str, str, str, Optional[str]]:
    """Process a single video: download and extract audio"""

    # Check if already processed
    audio_path = os.path.join(audio_dir, f"{video_id}.wav")
    if skip_existing and os.path.exists(audio_path):
        if os.path.getsize(audio_path) > 1000:
            return video_id, "Skipped", "Already processed", None

    # Download video
    success, message, error_category = download_video(
        video_id, video_dir, quality,
        use_uvx=use_uvx,
        cookies_file=cookies_file,
        cookies_from_browser=cookies_from_browser,
        proxy=proxy,
        debug=debug
    )

    if not success:
        # Try fallback mode if network error
        if error_category in [ErrorCategories.NETWORK, ErrorCategories.TIMEOUT] and not is_retry:
            success, message, error_category = download_video(
                video_id, video_dir, quality,
                use_uvx=use_uvx,
                fallback_mode=True,
                cookies_file=cookies_file,
                cookies_from_browser=cookies_from_browser,
                proxy=proxy,
                debug=debug
            )

    if not success:
        return video_id, "Failed", message, error_category

    # Convert to audio
    video_path = os.path.join(video_dir, f"{video_id}.mp4")
    success, audio_path, error_msg = convert_to_audio(video_path, audio_dir)

    if not success:
        return video_id, "Failed", f"FFmpeg: {error_msg}", ErrorCategories.FFMPEG

    return video_id, "Success", "Downloaded and converted", None


# ============================================================================
# Progress Display
# ============================================================================

class SpeedColumn(ProgressColumn):
    """Custom column for successful download speed"""

    def render(self, task) -> Text:
        speed = task.fields.get('speed', 0)
        if speed > 0:
            return Text(f"{speed:.1f}/min", style="cyan")
        return Text("--/min", style="dim")


class StatusColumn(ProgressColumn):
    """Custom column for status counts"""

    def render(self, task) -> Text:
        success = task.fields.get('success', 0)
        skipped = task.fields.get('skipped', 0)

        status_parts = [
            f"✅ {success:,}",
            f"⏭️{skipped:,}"
        ]

        return Text(" | ".join(status_parts))


class ErrorDetailsColumn(ProgressColumn):
    """Display error types with dynamic width limit"""

    def __init__(self, max_width: int = 40):
        self.max_width = max_width
        super().__init__()

    def render(self, task) -> Text:
        errors = task.fields.get('error_counts', {})
        if not errors or sum(errors.values()) == 0:
            return Text("")

        # Display errors, sorted by count (highest first)
        error_parts = []
        total_width = 0

        for error_type, count in sorted(errors.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                emoji = ErrorCategories.get_emoji(error_type)
                part = f"{emoji}{count}"
                part_width = len(part) + 1  # +1 for space separator

                # If adding this error would exceed width limit, add ellipsis
                if total_width > 0 and total_width + part_width > self.max_width:
                    error_parts.append("...")
                    break

                error_parts.append(part)
                total_width += part_width

        if error_parts:
            return Text(" ".join(error_parts), style="yellow")
        return Text("")


def create_progress_bar() -> Progress:
    """Create Rich progress bar with responsive layout based on terminal width"""
    # Get terminal width
    try:
        terminal_width = shutil.get_terminal_size().columns
    except:
        terminal_width = 80  # Default fallback

    # Build columns based on available width
    columns = [
        SpinnerColumn(),
    ]

    # Adjust title based on width
    if terminal_width < 100:
        columns.append(TextColumn("[bold blue]VAST-150k"))
    else:
        columns.append(TextColumn("[bold blue]Downloading VAST-150k"))

    # Progress bar - keep it short to save space
    columns.append(BarColumn(bar_width=20))  # Fixed short width

    # Always show progress count
    columns.append(MofNCompleteColumn())

    # Add time columns for wider terminals
    if terminal_width >= 80:
        columns.append(TimeElapsedColumn())
        if terminal_width >= 100:
            columns.append(TimeRemainingColumn())

    # Add status for medium+ terminals
    if terminal_width >= 80:
        columns.extend([
            TextColumn("•"),
            StatusColumn(),
        ])

    # Add error details for wide terminals
    if terminal_width >= 120:
        # Calculate available width for errors
        # Approximate: spinner(2) + title(20) + bar(30) + count(12) + times(20) + status(20) = ~104
        error_width = max(20, terminal_width - 110)
        columns.extend([
            TextColumn("•"),
            ErrorDetailsColumn(max_width=min(40, error_width)),
        ])

    # Add speed for medium+ terminals
    if terminal_width >= 90:
        columns.extend([
            TextColumn("•"),
            SpeedColumn(),
        ])

    return Progress(
        *columns,
        console=console,
        refresh_per_second=2
    )


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PathConfig:
    """Path configuration"""
    annotations: Path = Path("datasets/vast27m/annotations150k.json")
    base_dir: Path = Path(".")
    video_dir: Optional[Path] = None
    audio_dir: Optional[Path] = None

    def __post_init__(self):
        if self.video_dir is None:
            self.video_dir = self.base_dir / "videos"
        if self.audio_dir is None:
            self.audio_dir = self.base_dir / "audios"


@dataclass
class DownloadConfig:
    """Download configuration"""
    workers: int = 4
    quality: str = "360p"
    skip_existing: bool = True
    use_uvx: bool = True
    limit: Optional[int] = None
    offset: int = 0
    shuffle: bool = False


@dataclass
class ProxyConfig:
    """Proxy configuration"""
    enable_proxy: bool = True
    refresh_interval: int = 3600


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_retries: int = 3
    base_delay: float = 5.0
    retry_failed: bool = True
    # Sign-in error monitoring
    signin_error_threshold: int = 20  # Consecutive errors before pause
    signin_pause_duration: int = 900  # 15 minutes pause
    signin_pause_enabled: bool = True  # Enable/disable pause mechanism


@dataclass
class AuthConfig:
    """Authentication configuration"""
    cookies: Optional[Path] = None
    cookies_from_browser: Optional[str] = None


@dataclass
class DisplayConfig:
    """Display configuration"""
    debug: bool = False
    verbose: bool = False
    status_interval: float = 30.0
    save_progress: bool = True


@dataclass
class Config:
    """Main configuration"""
    paths: PathConfig = field(default_factory=PathConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)


# ============================================================================
# Helper Functions
# ============================================================================

def create_summary_table(success_count: int, skipped_count: int, error_counts: Dict[str, int],
                         total: int, title: str = "Download Summary") -> Table:
    """Create a summary table for display"""
    summary_table = Table(title=title, show_header=True, header_style="bold magenta")
    summary_table.add_column("Category", style="cyan")
    summary_table.add_column("Count", style="green")
    summary_table.add_column("Percentage", style="yellow")

    summary_table.add_row("✅ Success", f"{success_count:,}", f"{success_count/total*100:.1f}%")
    summary_table.add_row("⏭️Skipped", f"{skipped_count:,}", f"{skipped_count/total*100:.1f}%")

    for category, count in error_counts.items():
        if count > 0:
            emoji = ErrorCategories.get_emoji(category)
            summary_table.add_row(
                f"{emoji} {category.replace('_', ' ').title()}",
                f"{count:,}",
                f"{count/total*100:.1f}%"
            )

    return summary_table


# ============================================================================
# Main Function
# ============================================================================

def main(config: Config = Config()):
    """
    VAST-150k Downloader V3 - Simplified and robust version
    """

    # Print configuration
    console.print("\n[bold cyan]VAST-150k Dataset Downloader V3[/bold cyan]\n")

    config_table = Table(title="Configuration", show_header=True, header_style="bold magenta")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Annotations", str(config.paths.annotations))
    config_table.add_row("Output Directory", str(config.paths.base_dir))
    config_table.add_row("Workers", str(config.download.workers))
    config_table.add_row("Quality", config.download.quality)
    config_table.add_row("Proxy Enabled", str(config.proxy.enable_proxy))
    config_table.add_row("Max Retries", str(config.retry.max_retries))
    config_table.add_row("Signin Error Threshold", str(config.retry.signin_error_threshold))
    config_table.add_row("Signin Pause Duration", f"{config.retry.signin_pause_duration/60:.0f} min")
    config_table.add_row("Signin Pause Enabled", str(config.retry.signin_pause_enabled))

    console.print(config_table)
    console.print()

    # Create directories
    config.paths.video_dir.mkdir(parents=True, exist_ok=True)
    config.paths.audio_dir.mkdir(parents=True, exist_ok=True)

    # Load video IDs
    console.print("[yellow]Loading video IDs...[/yellow]")

    if not config.paths.annotations.exists():
        # Try in base directory
        alt_path = config.paths.base_dir / config.paths.annotations.name
        if alt_path.exists():
            config.paths.annotations = alt_path
        else:
            console.print(f"[red]Error: Annotations file not found: {config.paths.annotations}[/red]")
            return

    with open(config.paths.annotations, 'r') as f:
        data = json.load(f)

    # Extract video IDs
    video_ids = []
    for item in data:
        if 'url' in item:
            url = item['url']
            if 'youtube.com/watch?v=' in url:
                video_id = url.split('v=')[1].split('&')[0]
                video_ids.append(video_id)

    # Apply offset and limit
    if config.download.offset > 0:
        video_ids = video_ids[config.download.offset:]
    if config.download.limit:
        video_ids = video_ids[:config.download.limit]

    # Shuffle if requested
    if config.download.shuffle:
        random.shuffle(video_ids)

    console.print(f"[green]Found {len(video_ids)} videos to process[/green]\n")

    # Initialize components
    retry_manager = RetryManager(
        max_retries=config.retry.max_retries,
        base_delay=config.retry.base_delay
    )

    proxy_manager = ProxyManager(
        num_workers=config.download.workers,
        enable_proxy=config.proxy.enable_proxy
    )

    signin_monitor = SigninErrorMonitor(
        threshold=config.retry.signin_error_threshold,
        pause_duration=config.retry.signin_pause_duration,
        enabled=config.retry.signin_pause_enabled
    )

    # Debug output for signin monitor
    if config.display.debug or config.retry.signin_error_threshold != 20:
        console.print(f"[cyan]Sign-in monitor initialized: threshold={signin_monitor.threshold}, pause={signin_monitor.pause_duration/60:.0f}min, enabled={signin_monitor.enabled}[/cyan]")

    # Prepare task queue
    task_queue = PriorityQueue()
    for i, video_id in enumerate(video_ids):
        # Priority based on position (earlier = higher priority)
        task_queue.put((i, video_id, {'is_retry': False}))

    # Statistics
    completed_count = 0
    success_count = 0
    skipped_count = 0
    error_counts = defaultdict(int)
    failed_records = []
    start_time = time.time()
    last_status_time = time.time()

    # Worker tracking
    worker_counter = 0
    worker_map = {}

    # Create progress bar
    progress = create_progress_bar()

    with progress:
        main_task = progress.add_task(
            "Processing videos...",
            total=len(video_ids),
            success=0,
            skipped=0,
            error_counts={},
            speed=0.0
        )

        # Create thread pool
        with ThreadPoolExecutor(max_workers=config.download.workers) as executor:
            futures = {}

            # Main processing loop using as_completed()
            while completed_count < len(video_ids) or futures:

                # Check if we're in pause mode due to sign-in errors
                if signin_monitor.is_paused():
                    remaining = signin_monitor.get_remaining_pause_time()
                    if remaining > 0:
                        # Don't submit new tasks during pause, but continue processing existing ones
                        time.sleep(1)
                        # Skip to processing completed tasks
                    else:
                        # Pause ended, continue normally
                        console.print("[bold green]Resuming after sign-in error pause (counter reset to 0)[/bold green]")

                # Submit new tasks if we have room and not paused
                while len(futures) < config.download.workers and not task_queue.empty() and not signin_monitor.is_paused():
                    try:
                        priority, video_id, metadata = task_queue.get_nowait()

                        # Assign worker and proxy
                        worker_id = worker_counter % config.download.workers
                        worker_counter += 1

                        proxy = proxy_manager.get_proxy_for_worker(worker_id) if proxy_manager else None

                        # If worker 2+ needs proxy but none available, put task back and skip
                        if worker_id >= 2 and proxy_manager and proxy is None:
                            task_queue.put((priority, video_id, metadata))
                            if config.display.debug:
                                console.print(f"[yellow]Worker {worker_id} has no proxy available, postponing task {video_id}[/yellow]")
                            continue

                        # Submit task
                        future = executor.submit(
                            process_video,
                            video_id,
                            str(config.paths.video_dir),
                            str(config.paths.audio_dir),
                            config.download.quality,
                            config.download.skip_existing,
                            config.download.use_uvx,
                            str(config.auth.cookies) if config.auth.cookies else None,
                            config.auth.cookies_from_browser,
                            proxy,
                            worker_id,
                            config.display.debug,
                            metadata.get('is_retry', False)
                        )

                        futures[future] = (video_id, worker_id, proxy, metadata)
                        worker_map[future] = worker_id

                    except Empty:
                        break

                # Check for ready retries (but not during pause)
                if not signin_monitor.is_paused():
                    ready_retries = retry_manager.get_ready_retries()
                    for video_id, metadata in ready_retries:
                        if len(futures) < config.download.workers:
                            # Submit retry
                            worker_id = worker_counter % config.download.workers
                            worker_counter += 1
                            proxy = proxy_manager.get_proxy_for_worker(worker_id) if proxy_manager else None

                            # If worker 2+ needs proxy but none available, put back to retry queue
                            if worker_id >= 2 and proxy_manager and proxy is None:
                                retry_manager.add_to_retry(
                                    video_id,
                                    metadata.get('error_category', ErrorCategories.NETWORK),
                                    "No proxy available",
                                    bool(config.auth.cookies)
                                )
                                if config.display.debug:
                                    console.print(f"[yellow]Worker {worker_id} has no proxy for retry, delaying {video_id}[/yellow]")
                                continue

                            future = executor.submit(
                                process_video,
                                video_id,
                                str(config.paths.video_dir),
                                str(config.paths.audio_dir),
                                config.download.quality,
                                config.download.skip_existing,
                                config.download.use_uvx,
                                str(config.auth.cookies) if config.auth.cookies else None,
                                config.auth.cookies_from_browser,
                                proxy,
                                worker_id,
                                config.display.debug,
                                True  # is_retry
                            )

                            futures[future] = (video_id, worker_id, proxy, metadata)
                            worker_map[future] = worker_id

                # Process completed tasks
                if futures:
                    try:
                        # Wait for at least one task to complete
                        for future in as_completed(futures.keys(), timeout=1):
                            video_id, worker_id, proxy, metadata = futures[future]
                            del futures[future]

                            try:
                                video_id, status, message, error_category = future.result()
                            except Exception as e:
                                status = "Failed"
                                message = str(e)
                                error_category = ErrorCategories.UNEXPECTED

                            completed_count += 1

                            # Update statistics
                            if status == "Success":
                                success_count += 1
                                signin_monitor.record_success()  # Reset consecutive signin error count
                            elif status == "Skipped":
                                skipped_count += 1
                            else:
                                # Error occurred
                                if error_category:
                                    error_counts[error_category] += 1

                                    # Refresh proxy pool on SIGNIN errors
                                    if error_category == ErrorCategories.SIGNIN and proxy_manager:
                                        proxy_manager.refresh_proxy_pool()

                                    # Monitor sign-in errors
                                    if signin_monitor.record_error(error_category):
                                        # Pause triggered
                                        console.print(f"\n[bold red]🔴 Too many sign-in errors ({signin_monitor.threshold}), pausing for {signin_monitor.pause_duration/60:.0f} minutes...[/bold red]")
                                        console.print(f"[yellow]Resume time: {signin_monitor.get_resume_time()}[/yellow]\n")

                                    # Check if should retry
                                    if retry_manager.should_retry(error_category, video_id, bool(config.auth.cookies)):
                                        retry_manager.add_to_retry(
                                            video_id, error_category, message,
                                            bool(config.auth.cookies)
                                        )
                                    else:
                                        # Record failure
                                        failed_records.append(FailureRecord(
                                            video_id=video_id,
                                            error_type=error_category,
                                            error_message=message,
                                            attempts=metadata.get('attempt', 1),
                                            last_attempt=datetime.now().isoformat()
                                        ))

                                # Handle proxy failure
                                if proxy and error_category in [
                                    ErrorCategories.NETWORK,
                                    ErrorCategories.TIMEOUT,
                                    ErrorCategories.SIGNIN,
                                    ErrorCategories.AGE_RESTRICTED
                                ]:
                                    new_proxy = proxy_manager.rotate_proxy_for_worker(worker_id)
                                    if config.display.debug:
                                        if new_proxy:
                                            console.print(f"[yellow]Worker {worker_id} switched proxy due to {error_category}[/yellow]")
                                        elif worker_id >= 2:
                                            console.print(f"[yellow]Worker {worker_id} has no proxy available, will wait for refresh[/yellow]")

                            # Calculate speed (only successful downloads)
                            elapsed = time.time() - start_time
                            speed = (success_count / elapsed) * 60 if elapsed > 0 else 0

                            # Update progress
                            progress.update(
                                main_task,
                                advance=1,
                                success=success_count,
                                skipped=skipped_count,
                                error_counts=dict(error_counts),
                                speed=speed
                            )

                            # Print errors if verbose
                            if config.display.verbose and status == "Failed":
                                emoji = ErrorCategories.get_emoji(error_category)
                                console.print(f"[red]{emoji} {video_id}: {message[:100]}[/red]")

                            # Periodic status update
                            if time.time() - last_status_time > config.display.status_interval:
                                last_status_time = time.time()

                                # Calculate elapsed time and speed (only successful downloads)
                                elapsed = time.time() - start_time
                                current_speed = (success_count / elapsed) * 60 if elapsed > 0 else 0
                                current_time = datetime.now().strftime("%H:%M:%S")

                                # Calculate estimated remaining time
                                remaining_items = len(video_ids) - completed_count
                                if success_count > 0 and current_speed > 0:
                                    est_remaining_mins = remaining_items / current_speed
                                    remaining_str = f" | Est. remaining: {est_remaining_mins:.1f} min"
                                else:
                                    remaining_str = " | Est. remaining: N/A"

                                # Print timing info
                                console.print(f"\n[bold cyan][{current_time}] Elapsed: {elapsed/60:.1f} minutes | Speed: {current_speed:.1f} successful/min{remaining_str}[/bold cyan]\n")

                                # Show pause status if applicable
                                if signin_monitor.is_paused():
                                    pause_remaining = signin_monitor.get_remaining_pause_time()
                                    console.print(f"[bold yellow]⏸️PAUSED: {pause_remaining/60:.1f} min remaining (too many sign-in errors)[/bold yellow]")
                                    console.print(f"[yellow]Resume time: {signin_monitor.get_resume_time()}[/yellow]\n")
                                elif signin_monitor.consecutive_errors > signin_monitor.threshold * 0.5:
                                    # Show warning when approaching threshold
                                    console.print(f"[yellow]⚠️Sign-in errors: {signin_monitor.consecutive_errors}/{signin_monitor.threshold}[/yellow]\n")

                                # Print summary table
                                summary_table = create_summary_table(
                                    success_count, skipped_count, error_counts,
                                    len(video_ids), "Download Summary"
                                )
                                console.print(summary_table)
                                console.print()  # Add blank line after table

                            # Only process one future per iteration to allow new submissions
                            break

                    except TimeoutError:
                        # No tasks completed in timeout period, that's OK
                        pass

                # Break if no more work
                if not futures and task_queue.empty() and not retry_manager.retry_queue.qsize():
                    break

    # Clean up
    if proxy_manager:
        proxy_manager.stop()

    # Print final summary
    console.print("\n" + "="*60 + "\n")

    summary_table = create_summary_table(
        success_count, skipped_count, error_counts,
        len(video_ids), "Final Download Summary"
    )
    console.print(summary_table)

    # Save progress if requested
    if config.display.save_progress and failed_records:
        progress_file = config.paths.base_dir / 'download_progress.json'
        with open(progress_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'completed': completed_count,
                'success': success_count,
                'skipped': skipped_count,
                'error_counts': dict(error_counts),
                'failed_records': [
                    {
                        'video_id': r.video_id,
                        'error_type': r.error_type,
                        'error_message': r.error_message,
                        'attempts': r.attempts,
                        'last_attempt': r.last_attempt
                    } for r in failed_records
                ]
            }, f, indent=2)
        console.print(f"\n[green]Progress saved to {progress_file}[/green]")

    elapsed_time = time.time() - start_time
    console.print(f"\n[bold green]Download complete![/bold green]")
    console.print(f"Total time: {elapsed_time/60:.1f} minutes")
    console.print(f"Average speed: {success_count/elapsed_time*60:.1f} successful downloads/minute") if elapsed_time > 0 else console.print("Average speed: N/A")


def cli():
    """VAST-150k Dataset Downloader - Complete Rewrite."""
    config = tyro.cli(Config, description=__doc__)

    # Create directories using PathConfig defaults
    config.paths.video_dir.mkdir(parents=True, exist_ok=True)
    config.paths.audio_dir.mkdir(parents=True, exist_ok=True)

    # Run main program
    try:
        main(config)
    except KeyboardInterrupt:
        print("\n\nShutdown complete.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
