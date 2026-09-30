"""Patch the already-bootstrapped research collector in its isolated runner only."""
from __future__ import annotations
import hashlib
import json
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent

def patch(source):
    if 'from fast_transport import Transport, prefetch' in source:
        return source
    begin = source.index('def get(url,bound):\n')
    end = source.index('def emit(final=False):\n', begin)
    replacement = '''from fast_transport import Transport, prefetch
# One shared limiter also covers issuer-index requests and redirects.
def _access_stop():
 global stop
 stop=True
_transport=Transport(UA,META,counts,lambda: stop or terminate,_access_stop,rate=5.0)
def get(url,bound):
 return _transport.get(url,bound)
'''
    source = source[:begin] + replacement + source[end:]
    old = '   for task in tasks:\n'
    if source.count(old) != 1:
        raise ValueError('Unexpected task loop; refusing an unsafe patch')
    source = source.replace(old, '   for task,fetched in prefetch(tasks,seen,get,lambda: stop or terminate,workers=4):\n')
    old = '     body,r=get(url,64*1024**2)\n'
    if source.count(old) != 1:
        raise ValueError('Unexpected body retrieval; refusing patch')
    source = source.replace(old, '     body,r=fetched.result()\n')
    old = " if sys.stdin.readline().strip()!='CONTINUE':raise RuntimeError('DURABLE_UPLOAD_NOT_CONFIRMED')"
    if source.count(old) != 1:
        raise ValueError('Unexpected transfer handshake; refusing patch')
    source = source.replace(old, " wait_started=time.monotonic()\n" + old + "\n counts['transfer_wait_seconds']+=time.monotonic()-wait_started")
    compile(source, 'direct_sec_collector.py', 'exec')
    return source

def main():
    path = BASE / 'direct_sec_collector.py'
    original = path.read_bytes()
    upgraded = patch(original.decode()).encode()
    path.write_bytes(upgraded)
    meta = pathlib.Path('retry_work/metadata')
    meta.mkdir(parents=True, exist_ok=True)
    proof = {'source_sha256_before': hashlib.sha256(original).hexdigest(),
             'source_sha256_after': hashlib.sha256(upgraded).hexdigest(),
             'max_workers': 4, 'global_sec_requests_per_second': 5,
             'gzip_transfer': True, 'keep_alive': True,
             'financial_contract_changed': False, 'artifact_ceiling_changed': False,
             'original_bytes_hashing_preserved': True}
    (meta / 'transport_config.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof))
if __name__ == '__main__':
    main()
