"""
Thapar Website & PDF Scraper for RAG Pipeline
===============================================
Crawls a list of Thapar-related websites (thapar.edu and subdomains, or any
sites you specify), downloads all PDF files it finds, and also extracts
clean text from HTML pages — both saved in a structure ready to feed into
a RAG ingestion/chunking pipeline.

Usage:
    python thapar_rag_scraper.py

Configure the SEED_URLS and ALLOWED_DOMAINS lists below before running.

Output structure:
    output/
        pdfs/              -> downloaded .pdf files
        html_text/         -> extracted plain text per page (.txt)
        manifest.csv        -> log of every URL visited, type, and status
"""

import os
import re
import csv
import time
import hashlib
import logging
from urllib.parse import urljoin, urlparse
from collections import deque

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# --------------------------------------------------------------------------
# CONFIGURATION — edit this section for your project
# --------------------------------------------------------------------------

SEED_URLS = [
    # Admissions — ALL PDFs / data wanted, strictly
    "https://www.thapar.edu/admissions",
    "https://admission.thapar.edu/",
    # Hostels
    "https://www.thapar.edu/students/pages/hostels",
    # Programmes / syllabus
    "https://www.thapar.edu/programmes",
    # Fee circular
    "https://www.thapar.edu/students/pages/fee-circular-and-fee-chart",
    # Faculties (general directory)
    "https://www.thapar.edu/faculties",
    # Department subdomains — faculty + syllabus PDFs
    "https://csed.thapar.edu/",
    "https://csed.thapar.edu/faculty",
    "https://csed.thapar.edu/programmes",
    "https://med.thapar.edu/",
    "https://med.thapar.edu/programmes",
    "https://eced.thapar.edu/",
    "https://eced.thapar.edu/faculty",
    "https://eced.thapar.edu/programmes",
    "https://eied.thapar.edu/",
    # Webkiosk — limited scope, just the entry page
    "https://webkiosk.thapar.edu/",
    # Misces — only 2025/2026 items are kept (filtered during link
    # discovery — see MISCES_YEAR_KEYWORDS below)
    "https://www.thapar.edu/misces",
]

ALLOWED_DOMAINS = [
    "thapar.edu",
]

# --------------------------------------------------------------------------
# SCOPE RESTRICTION — only these path prefixes (per host) are in-scope for
# HTML crawling. This keeps the crawler tightly focused on exactly the
# sections requested instead of the whole thapar.edu site. An empty list
# for a host means "the whole host is in scope" (used for small dedicated
# subdomains where everything on them is relevant).
# --------------------------------------------------------------------------
ALLOWED_PATH_PREFIXES = {
    "www.thapar.edu": [
        "/admissions",
        "/students/pages/hostels",
        "/programmes",
        "/students/pages/fee-circular-and-fee-chart",
        "/faculties",
        "/misces",
    ],
    "admission.thapar.edu": [],
    "csed.thapar.edu": [],
    "med.thapar.edu": [],
    "eced.thapar.edu": [],
    "eied.thapar.edu": [],
    "webkiosk.thapar.edu": [],
}

# Under /misces specifically, only keep pages/PDFs whose URL text mentions
# one of these years — everything else under /misces is skipped.
MISCES_YEAR_KEYWORDS = ["2025", "2026"]

OUTPUT_DIR = "output"
PDF_DIR = os.path.join(OUTPUT_DIR, "pdfs")
TEXT_DIR = os.path.join(OUTPUT_DIR, "html_text")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.csv")

MAX_PAGES = 1500
REQUEST_DELAY = 1.0
TIMEOUT = 10
FAILURE_THRESHOLD = 3
MAX_PDF_SIZE_MB = 5
USER_AGENT = (
    "Mozilla/5.0 (compatible; RAG-Research-Bot/1.0; "
    "+for-academic-project-use)"
)

# --------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("thapar_scraper")

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

_retry_strategy = requests.adapters.Retry(
    total=0,
    status_forcelist=[429, 500, 502, 503, 504],
)
_adapter = requests.adapters.HTTPAdapter(
    max_retries=_retry_strategy, pool_connections=20, pool_maxsize=20
)
session.mount("https://", _adapter)
session.mount("http://", _adapter)


def normalize_url(url: str) -> str:
    if url.startswith("http://") and "thapar.edu" in url:
        url = "https://" + url[len("http://"):]
    return url


HOST_FAILURES = {}
DEAD_HOSTS = set()

# Tracks MD5 hashes of PDF content already saved, so the same file reached
# via two different URLs (common with CDN redirects / mirrored links) is
# only kept once.
SEEN_PDF_HASHES = set()

# Same idea for extracted HTML text — avoids saving near-identical page
# text twice when reached via two different URL variants.
SEEN_TEXT_HASHES = set()


def record_failure(host: str):
    HOST_FAILURES[host] = HOST_FAILURES.get(host, 0) + 1
    if HOST_FAILURES[host] >= FAILURE_THRESHOLD:
        DEAD_HOSTS.add(host)
        log.warning(f"Host failed {FAILURE_THRESHOLD}x in a row, blacklisting: {host}")
    else:
        log.warning(
            f"Request failed for host {host} "
            f"({HOST_FAILURES[host]}/{FAILURE_THRESHOLD} before blacklist)"
        )


def record_success(host: str):
    if host in HOST_FAILURES:
        HOST_FAILURES[host] = 0


def is_allowed(url: str) -> bool:
    """
    A URL is allowed for HTML crawling only if:
      1. Its host is one of ALLOWED_PATH_PREFIXES' keys (or a subdomain of
         'thapar.edu' generally, as a fallback), AND
      2. If that host has a restricted path list, the URL's path starts
         with one of those prefixes, AND
      3. If the path is under /misces, the URL also mentions one of the
         MISCES_YEAR_KEYWORDS — otherwise it's out of scope.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()

    if not any(allowed in host for allowed in ALLOWED_DOMAINS):
        return False

    prefixes = ALLOWED_PATH_PREFIXES.get(host)
    if prefixes is None:
        # Host not explicitly listed (e.g. an unexpected www.thapar.edu
        # variant) — be conservative and disallow rather than crawl broadly.
        return False

    if prefixes:  # non-empty list means path-restricted host
        if not any(path.startswith(p) for p in prefixes):
            return False

    if "/misces" in path:
        is_misces_index = path.rstrip("/") == "/misces"
        if not is_misces_index and not any(year in url for year in MISCES_YEAR_KEYWORDS):
            return False

    return True


def is_pdf_allowed(url: str) -> bool:
    """
    PDFs are allowed regardless of host (they may live on external CDNs),
    but a PDF whose URL is under a /misces path must still mention 2025/2026.
    """
    path = urlparse(url).path.lower()
    if "/misces" in path:
        return any(year in url for year in MISCES_YEAR_KEYWORDS)
    return True


def safe_get(url: str, stream: bool = False):
    host = urlparse(url).netloc
    if host in DEAD_HOSTS:
        return None
    try:
        resp = session.get(url, timeout=TIMEOUT, stream=stream)
        record_success(host)
        return resp
    except (requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.SSLError):
        record_failure(host)
        return None
    except requests.exceptions.ReadTimeout:
        log.warning(f"Read timeout (page-specific, not counted toward blacklist): {url}")
        return None
    except requests.exceptions.TooManyRedirects:
        log.warning(f"Too many redirects, skipping: {url}")
        return None
    except requests.exceptions.InvalidURL:
        log.warning(f"Invalid URL, skipping: {url}")
        return None
    except requests.exceptions.RequestException as e:
        log.warning(f"Unexpected request error for {url}: {e}")
        return None
    except Exception as e:
        log.warning(f"Unhandled error fetching {url}: {e}")
        return None


def safe_filename(url: str, suffix: str = "") -> str:
    parsed = urlparse(url)
    base = os.path.basename(parsed.path) or "index"
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    if not base.lower().endswith(suffix.lower()) and suffix:
        base += suffix
    name, ext = os.path.splitext(base)
    return f"{name}_{url_hash}{ext}"


def ensure_dirs():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)


def init_manifest():
    if not os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["url", "type", "status", "saved_path"])


def log_manifest(url, type_, status, saved_path=""):
    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([url, type_, status, saved_path])


def download_pdf(url: str) -> bool:
    host = urlparse(url).netloc
    if host in DEAD_HOSTS:
        log_manifest(url, "pdf", "skipped_dead_host")
        return False

    try:
        filename = safe_filename(url, suffix=".pdf")
        filepath = os.path.join(PDF_DIR, filename)
    except Exception as e:
        log.warning(f"Could not build filename for {url}: {e}")
        log_manifest(url, "pdf", f"error:bad_filename:{e}")
        return False

    if os.path.exists(filepath):
        log_manifest(url, "pdf", "skipped_exists", filepath)
        return True

    resp = safe_get(url, stream=True)
    if resp is None:
        log_manifest(url, "pdf", "error:unreachable")
        return False

    try:
        if resp.status_code != 200:
            log_manifest(url, "pdf", f"skipped_status_{resp.status_code}")
            return False

        content_type = resp.headers.get("Content-Type", "")
        content_length = resp.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_PDF_SIZE_MB * 1024 * 1024:
            log_manifest(url, "pdf", "skipped_too_large")
            return False

        if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
            log_manifest(url, "pdf", "skipped_not_pdf")
            return False

        total_bytes = 0
        max_bytes = MAX_PDF_SIZE_MB * 1024 * 1024
        hasher = hashlib.md5()
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    log.warning(f"PDF exceeded size limit mid-download, aborting: {url}")
                    f.close()
                    os.remove(filepath)
                    log_manifest(url, "pdf", "skipped_too_large_mid_download")
                    return False
                hasher.update(chunk)
                f.write(chunk)

        content_hash = hasher.hexdigest()
        if content_hash in SEEN_PDF_HASHES:
            os.remove(filepath)
            log_manifest(url, "pdf", f"skipped_duplicate_content:{content_hash[:8]}")
            return False
        SEEN_PDF_HASHES.add(content_hash)

        log_manifest(url, "pdf", "downloaded", filepath)
        return True

    except (OSError, IOError) as e:
        log.warning(f"Disk/file error saving PDF {url}: {e}")
        log_manifest(url, "pdf", f"error:disk:{e}")
        return False
    except Exception as e:
        log.warning(f"Unexpected error downloading PDF {url}: {e}")
        log_manifest(url, "pdf", f"error:{e}")
        return False
    finally:
        resp.close()


def extract_and_save_text(url: str, html: str):
    try:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [ln.strip() for ln in text.splitlines()]
        clean_text = "\n".join(ln for ln in lines if ln)

        if len(clean_text) < 50:
            log_manifest(url, "html", "skipped_too_short")
            return

        text_hash = hashlib.md5(clean_text.encode("utf-8", errors="ignore")).hexdigest()
        if text_hash in SEEN_TEXT_HASHES:
            log_manifest(url, "html", f"skipped_duplicate_content:{text_hash[:8]}")
            return
        SEEN_TEXT_HASHES.add(text_hash)

        filename = safe_filename(url, suffix=".txt")
        filepath = os.path.join(TEXT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"SOURCE_URL: {url}\n\n{clean_text}")
        log_manifest(url, "html", "text_extracted", filepath)
    except Exception as e:
        log.warning(f"Failed to extract/save text for {url}: {e}")
        log_manifest(url, "html", f"error:extract_failed:{e}")


def find_links(base_url: str, html: str):
    links = set()
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        log.warning(f"Failed to parse HTML for links on {base_url}: {e}")
        return links

    for a in soup.find_all("a", href=True):
        try:
            href = a["href"].strip()
            if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            full_url = urljoin(base_url, href)
            full_url = full_url.split("#")[0]
            full_url = normalize_url(full_url)
            if full_url.startswith(("http://", "https://")):
                links.add(full_url)
        except Exception:
            continue
    return links


def crawl():
    ensure_dirs()
    init_manifest()

    visited = set()
    page_queue = deque(u for u in SEED_URLS if not u.lower().endswith(".pdf"))
    pdf_queue = deque(u for u in SEED_URLS if u.lower().endswith(".pdf"))
    pages_crawled = 0
    pdf_count = 0
    error_count = 0

    pbar = tqdm(desc="Fetching documents (pages + PDFs)")

    try:
        while page_queue or pdf_queue:

            while pdf_queue:
                url = pdf_queue.popleft()
                if url in visited:
                    continue
                visited.add(url)
                try:
                    if download_pdf(url):
                        pdf_count += 1
                        pbar.update(1)
                except Exception as e:
                    error_count += 1
                    log.warning(f"Unhandled exception downloading PDF {url}: {e}")
                    log_manifest(url, "pdf", f"error:unhandled:{e}")
                time.sleep(REQUEST_DELAY)

            if not page_queue or pages_crawled >= MAX_PAGES:
                break

            url = page_queue.popleft()

            try:
                if url in visited or not is_allowed(url):
                    continue
                visited.add(url)

                host = urlparse(url).netloc
                if host in DEAD_HOSTS:
                    log_manifest(url, "html", "skipped_dead_host")
                    continue

                resp = safe_get(url)
                if resp is None:
                    error_count += 1
                    log_manifest(url, "html", "error:unreachable")
                    continue

                if resp.status_code == 404:
                    log_manifest(url, "html", "404_not_found")
                    continue
                if resp.status_code == 403:
                    log_manifest(url, "html", "403_forbidden")
                    continue
                if resp.status_code >= 400:
                    log_manifest(url, "html", f"http_error_{resp.status_code}")
                    continue

                content_type = resp.headers.get("Content-Type", "")

                if "application/pdf" in content_type.lower():
                    if download_pdf(url):
                        pdf_count += 1
                        pbar.update(1)
                    continue

                if "text/html" not in content_type.lower():
                    log_manifest(url, "html", f"skipped_content_type:{content_type}")
                    continue

                try:
                    html = resp.text
                except Exception as e:
                    log.warning(f"Could not decode response body for {url}: {e}")
                    log_manifest(url, "html", f"error:decode:{e}")
                    continue

                extract_and_save_text(url, html)
                pages_crawled += 1
                pbar.update(1)

                for link in find_links(url, html):
                    if link in visited:
                        continue
                    if link.lower().endswith(".pdf"):
                        if is_pdf_allowed(link):
                            pdf_queue.append(link)
                        else:
                            log_manifest(link, "pdf", "skipped_misces_year_filter")
                    elif is_allowed(link):
                        page_queue.append(link)

                if pages_crawled % 25 == 0:
                    log.info(
                        f"Progress: {pages_crawled} pages, {pdf_count} PDFs, "
                        f"{error_count} errors, {len(page_queue)} pages queued, "
                        f"{len(pdf_queue)} PDFs queued, "
                        f"{len(DEAD_HOSTS)} dead hosts blacklisted"
                    )

                time.sleep(REQUEST_DELAY)

            except Exception as e:
                error_count += 1
                log.warning(f"Unhandled exception on {url}: {e}")
                log_manifest(url, "unknown", f"error:unhandled:{e}")
                continue

    except KeyboardInterrupt:
        log.info("Interrupted by user — saving progress and exiting cleanly.")

    finally:
        pbar.close()
        log.info(
            f"Done. Pages crawled: {pages_crawled}, PDFs downloaded: {pdf_count}, "
            f"errors: {error_count}, dead hosts: {sorted(DEAD_HOSTS)}"
        )
        log.info(f"Text files saved in: {TEXT_DIR}")
        log.info(f"PDFs saved in: {PDF_DIR}")
        log.info(f"Manifest log: {MANIFEST_PATH}")

if __name__ == "__main__":
    crawl()
