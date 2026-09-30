"""Bounded SEC transport only. Does not change filing selection or financial rules."""
from __future__ import annotations
import collections
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import http.client
import io
import json
import pathlib
import threading
import time
import urllib.error
import urllib.parse

ALLOWED_HOSTS = frozenset({'www.sec.gov', 'data.sec.gov'})

class AccessStopped(RuntimeError):
    pass

class RateGate:
    """One no-burst request-start limiter shared by all workers and both SEC hosts."""
    def __init__(self, rate=5.0, clock=time.monotonic, sleep=time.sleep):
        if not 0 < rate <= 5.0:
            raise ValueError('Rate must be positive and <=5 requests/second')
        self.interval, self.clock, self.sleep = 1.0 / rate, clock, sleep
        self.next_start, self.lock = 0.0, threading.Lock()

    def enter(self, halted):
        with self.lock:
            if halted():
                raise AccessStopped('SEC_ACCESS_STOP')
            self.sleep(max(0.0, self.next_start - self.clock()))
            if halted():
                raise AccessStopped('SEC_ACCESS_STOP')
            self.next_start = self.clock() + self.interval

class Transport:
    def __init__(self, user_agent, metadata, counts, halted, on_stop, rate=5.0):
        if not user_agent or '@' not in user_agent:
            raise ValueError('Existing declared contact User-Agent required')
        self.ua, self.meta, self.counts = user_agent, pathlib.Path(metadata), counts
        self.halted, self.on_stop, self.gate = halted, on_stop, RateGate(rate)
        self.local, self.write_lock = threading.local(), threading.Lock()
        self.stopped = threading.Event()
        self.meta.mkdir(parents=True, exist_ok=True)

    def is_stopped(self):
        return self.stopped.is_set() or self.halted()

    @staticmethod
    def split_url(url):
        p = urllib.parse.urlsplit(url)
        if p.scheme != 'https' or p.hostname not in ALLOWED_HOSTS or p.username or p.password or p.port not in (None, 443) or p.fragment:
            raise ValueError('Only canonical HTTPS SEC URLs are allowed')
        return p.hostname, urllib.parse.urlunsplit(('', '', p.path or '/', p.query, ''))

    def connection(self, host):
        if not hasattr(self.local, 'connections'):
            self.local.connections = {}
        if host not in self.local.connections:
            self.local.connections[host] = http.client.HTTPSConnection(host, timeout=45)
        return self.local.connections[host]

    def close_connection(self, host):
        connections = getattr(self.local, 'connections', {})
        old = connections.pop(host, None)
        if old is not None:
            old.close()

    def get(self, url, bound):
        if bound <= 0:
            raise ValueError('positive byte bound required')
        original_url = url
        for redirect in range(6):
            host, path = self.split_url(url)
            self.gate.enter(self.is_stopped)
            begun = time.monotonic()
            receipt = dict(url=url, requested_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            try:
                conn = self.connection(host)
                conn.request('GET', path, headers={'User-Agent': self.ua, 'Accept-Encoding': 'gzip', 'Connection': 'keep-alive'})
                response = conn.getresponse()
                status = response.status
                receipt.update(http_status=status, response_url=url, content_type=response.getheader('Content-Type'), content_encoding=response.getheader('Content-Encoding'))
                if status in (401, 403, 429):
                    self.stopped.set()
                    self.on_stop()
                    raise urllib.error.HTTPError(url, status, 'SEC access stop', response.headers, None)
                if status in (301, 302, 303, 307, 308):
                    next_url = urllib.parse.urljoin(url, response.getheader('Location', ''))
                    self.split_url(next_url)
                    self.close_connection(host)
                    if redirect == 5:
                        raise ValueError('Too many redirects')
                    receipt['status'] = 'REDIRECT'
                    url = next_url
                    continue
                if status != 200:
                    raise urllib.error.HTTPError(url, status, 'SEC HTTP error', response.headers, None)
                wire = response.read(bound + 1)
                if len(wire) > bound:
                    raise ValueError('WIRE_BYTE_BOUND_EXCEEDED')
                receipt['wire_bytes'] = len(wire)
                encoding = (response.getheader('Content-Encoding') or 'identity').lower().strip()
                if encoding == 'gzip':
                    with gzip.GzipFile(fileobj=io.BytesIO(wire)) as stream:
                        body = stream.read(bound + 1)
                elif encoding in ('', 'identity'):
                    body = wire
                else:
                    raise ValueError('Unexpected content encoding: ' + encoding)
                if len(body) > bound:
                    raise ValueError('DECODED_BYTE_BOUND_EXCEEDED')
                receipt.update(status='RETRIEVED', bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), original_url=original_url)
                return body, receipt
            except Exception as exc:
                self.close_connection(host)
                receipt.update(status='ERROR', error=repr(exc))
                raise
            finally:
                receipt['elapsed_seconds'] = time.monotonic() - begun
                with self.write_lock:
                    self.counts['http_requests'] += 1
                    self.counts['network_bytes'] += receipt.get('bytes', 0)
                    self.counts['wire_bytes'] += receipt.get('wire_bytes', 0)
                    with (self.meta / 'requests.jsonl').open('a', encoding='utf-8') as out:
                        out.write(json.dumps(receipt, separators=(',', ':')) + '\n')
        raise RuntimeError('redirect exhaustion')

def primary_url(task):
    return ('https://www.sec.gov/Archives/edgar/data/' + str(int(task['cik'])) + '/' + task['accession'].replace('-', '') + '/' + urllib.parse.quote(task['primary_document']))

def prefetch(tasks, seen, fetch, halted, workers=4):
    """At most four in-flight/buffered bodies; deterministic original task order."""
    if not 1 <= workers <= 4:
        raise ValueError('workers must be 1..4')
    remaining = iter(t for t in tasks if t['cik'] + '/' + t['accession'] not in seen)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix='sec-body')
    pending = collections.deque()
    def fill():
        while len(pending) < workers and not halted():
            task = next(remaining, None)
            if task is None:
                break
            pending.append((task, executor.submit(fetch, primary_url(task), 64 * 1024**2)))
    try:
        fill()
        while pending:
            task, future = pending.popleft()
            if halted():
                future.cancel()
                break
            yield task, future
            fill()
    finally:
        for _, future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
