"""Reconcile completed-run artifacts after external durable-copy verification.
No financial inputs, SEC requests, or unacknowledged originals are changed.
"""
from __future__ import annotations
import base64, datetime, gzip, hashlib, io, json, os, pathlib, re, time
import urllib.request, urllib.error, zipfile
BRANCH='chatgpt-v73-raw-recovery-20260930'
STATE='research/v73_archive_scope/state/durable_official_keys.txt'
RAW=re.compile(r'v73-official-raw-(\d+)-(\d{4})$')
KEY=re.compile(r'\d{10}/\d{10}-\d{2}-\d{6}$')
SHA=re.compile(r'[0-9a-f]{64}$')
MAX_ZIP=64*1024**2

def sha(data): return hashlib.sha256(data).hexdigest()
def safe_zip(raw):
    z=zipfile.ZipFile(io.BytesIO(raw)); names=z.namelist()
    if len(names)!=len(set(names)): raise ValueError('duplicate ZIP member')
    for i in z.infolist():
        p=pathlib.PurePosixPath(i.filename)
        if p.is_absolute() or '..' in p.parts or '\\' in i.filename or ':' in i.filename or (i.external_attr>>16)&0o170000==0o120000:
            raise ValueError('unsafe ZIP member')
        if i.file_size>96*1024**2: raise ValueError('oversized member')
    if sum(i.file_size for i in z.infolist())>512*1024**2: raise ValueError('oversized ZIP content')
    if z.testzip() is not None: raise ValueError('CRC failed')
    return z

def verify_raw(raw, artifact, ack):
    rid=artifact['workflow_run']['id']; aid=artifact['id']
    match=RAW.fullmatch(artifact['name'])
    if not match or int(match[1])!=rid: raise ValueError('artifact identity mismatch')
    digest=artifact.get('digest','')
    if not digest.startswith('sha256:') or digest!='sha256:'+sha(raw): raise ValueError('ZIP digest mismatch')
    if ack.get('verified') is not True or ack.get('run_id')!=rid or ack.get('artifact_id')!=aid or ack.get('downloaded_zip_sha256')!=sha(raw):
        raise ValueError('external acknowledgment mismatch')
    with safe_zip(raw) as z:
        mbytes=z.read('SHA256SUMS.json')
        if sha(mbytes)!=ack.get('manifest_sha256'): raise ValueError('manifest hash mismatch')
        manifest=json.loads(mbytes)
        actual={i.filename for i in z.infolist() if not i.is_dir()}-{'SHA256SUMS.json'}
        if set(manifest)!=actual: raise ValueError('unmanifested content')
        for name, item in manifest.items():
            b=z.read(name)
            if len(b)!=item['bytes'] or sha(b)!=item['sha256']: raise ValueError('member hash mismatch')
        records=json.loads(z.read('records.json')); keys=set(); total=0
        if len(records)!=ack.get('records_verified'): raise ValueError('record count mismatch')
        for r in records:
            if r.get('source_level')!='OFFICIAL_SEC_HTTP200_ORIGINAL_PENDING_FIELD_VALIDATION': raise ValueError('not official source')
            with gzip.GzipFile(fileobj=io.BytesIO(z.read(r['file']))) as f: b=f.read(64*1024**2+1)
            if len(b)!=r['archive_content_bytes'] or sha(b)!=r['archive_content_sha256']: raise ValueError('original hash mismatch')
            total+=len(b)
            for cik in r['issuer_ciks']:
                key=cik+'/'+r['accession']
                if not KEY.fullmatch(key): raise ValueError('invalid issuer/accession')
                keys.add(key)
        if total!=ack.get('raw_bytes_verified'): raise ValueError('original byte total mismatch')
    return keys

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*a,**kw): return None

class Client:
    def __init__(self):
        self.repo=os.environ['GITHUB_REPOSITORY']; self.token=os.environ['GITHUB_TOKEN']
        if self.repo!='seungyun-eng/investment': raise ValueError('wrong repository')
        self.open=urllib.request.build_opener(NoRedirect)
    def api(self,route,method='GET',payload=None):
        req=urllib.request.Request('https://api.github.com/repos/'+self.repo+route,
            data=None if payload is None else json.dumps(payload).encode(),method=method,
            headers={'Authorization':'Bearer '+self.token,'Accept':'application/vnd.github+json','Content-Type':'application/json','User-Agent':'V73-verified-handoff'})
        for attempt in range(5):
            try:
                with self.open.open(req,timeout=30) as r: b=r.read(8*1024**2+1)
                if len(b)>8*1024**2: raise ValueError('API response too large')
                return json.loads(b) if b else {}
            except urllib.error.HTTPError as e:
                if e.code==404 and method=='GET': return None
                if e.code not in (500,502,503,504) or attempt==4: raise
            except (urllib.error.URLError,TimeoutError):
                if attempt==4: raise
            time.sleep(2**attempt)
    def download(self,aid):
        req=urllib.request.Request('https://api.github.com/repos/'+self.repo+'/actions/artifacts/'+str(aid)+'/zip',headers={'Authorization':'Bearer '+self.token,'User-Agent':'V73-verified-handoff'})
        try:
            r=self.open.open(req,timeout=30)
        except urllib.error.HTTPError as e:
            if e.code not in (301,302,303,307,308): raise
            url=e.headers.get('Location','')
            if not url.startswith('https://'): raise ValueError('unsafe artifact redirect')
            r=urllib.request.urlopen(url,timeout=120)
        with r: b=r.read(MAX_ZIP+1)
        if len(b)>MAX_ZIP: raise ValueError('artifact too large for bounded cleanup')
        return b
    def text(self,path):
        o=self.api('/contents/'+path+'?ref='+BRANCH)
        if o is None: return None
        if o.get('encoding')!='base64': raise ValueError('unreadable state file')
        return json.loads(base64.b64decode(o['content']))
    def inventory(self):
        out=[]; total=None
        for page in range(1,101):
            o=self.api('/actions/artifacts?per_page=100&page='+str(page))
            if total is None: total=o['total_count']
            out.extend(o['artifacts'])
            if len(o['artifacts'])<100:
                if len(out)<total: raise ValueError('incomplete artifact inventory')
                return out
        raise ValueError('inventory pagination limit')
    def persist(self,newkeys):
        if not newkeys: return
        for attempt in range(5):
            current=self.api('/contents/'+STATE+'?ref='+BRANCH)
            if current is None: raise ValueError('durable state missing; refusing destructive cleanup')
            encoded=current.get('content') or ''
            if current.get('encoding')=='base64' and encoded:
                raw=base64.b64decode(encoded)
            else:
                blobsha=current.get('sha')
                if not blobsha: raise ValueError('durable state missing blob sha')
                blob=self.api('/git/blobs/'+blobsha)
                if not blob or blob.get('sha')!=blobsha or blob.get('encoding')!='base64':
                    raise ValueError('durable state blob unavailable')
                raw=base64.b64decode(blob['content'])
                if int(blob.get('size',-1))!=len(raw): raise ValueError('durable state blob size mismatch')
            keys=set(raw.decode().splitlines())
            if any(not KEY.fullmatch(k) for k in keys): raise ValueError('invalid durable state')
            union=keys|newkeys
            if union==keys: return
            data=('\n'.join(sorted(union))+'\n').encode()
            try:
                self.api('/contents/'+STATE,'PUT',{'branch':BRANCH,'sha':current['sha'],'message':'research: preserve inherited verified SEC keys before temporary cleanup','content':base64.b64encode(data).decode()})
                return
            except urllib.error.HTTPError as e:
                if e.code not in (409,422) or attempt==4: raise
                time.sleep(1+attempt)
        raise RuntimeError('durable state update failed')

def reconcile(client,current_run=0,explicit=None):
    report={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'current_run':current_run,'deleted':[],'retained':[]}
    all_items=client.inventory(); report['before_bytes']=sum(a['size_in_bytes'] for a in all_items if not a['expired'])
    known_runs={}; explicit={int(x['artifact_id']):x for x in (explicit or [])}
    for a in all_items:
        if a['expired']: continue
        rid=a['workflow_run']['id']
        if rid==current_run: continue
        match=RAW.fullmatch(a['name']); proof=explicit.get(a['id'])
        if not match and proof is None: continue
        if rid not in known_runs: known_runs[rid]=client.api('/actions/runs/'+str(rid))
        run=known_runs[rid]
        if not run or run['status']!='completed' or run['head_branch']!=BRANCH:
            report['retained'].append({'id':a['id'],'reason':'not completed isolated research run'}); continue
        keys=set()
        if match:
            ack=client.text('research/v73_archive_scope/acks/'+str(rid)+'_'+str(a['id'])+'.json')
            if not ack:
                report['retained'].append({'id':a['id'],'reason':'external preservation acknowledgment missing'}); continue
            keys=verify_raw(client.download(a['id']),a,ack)
        else:
            if proof.get('durable_external_copy_verified') is not True or proof.get('run_id')!=rid or proof.get('name')!=a['name'] or a.get('digest')!='sha256:'+proof.get('zip_sha256',''):
                raise ValueError('explicit preservation mismatch')
            b=client.download(a['id'])
            if sha(b)!=proof['zip_sha256']: raise ValueError('explicit ZIP mismatch')
            with safe_zip(b): pass
        client.persist(keys)
        latest=client.api('/actions/artifacts/'+str(a['id']))
        if latest is None: continue
        if any(latest.get(k)!=a.get(k) for k in ('id','name','digest','workflow_run')): raise ValueError('artifact changed before deletion')
        client.api('/actions/artifacts/'+str(a['id']),'DELETE')
        report['deleted'].append({'id':a['id'],'name':a['name'],'bytes':a['size_in_bytes'],'sha256':a['digest'][7:],'preserved_keys':len(keys)})
    report['after_bytes']=sum(a['size_in_bytes'] for a in client.inventory() if not a['expired'])
    return report

def main():
    client=Client(); rid=int(os.environ.get('GITHUB_RUN_ID','0'))
    p=pathlib.Path(__file__).with_name('verified_temporary_copies.json')
    explicit=json.loads(p.read_text()) if p.exists() else []
    out=reconcile(client,rid,explicit)
    d=pathlib.Path('retry_work/metadata'); d.mkdir(parents=True,exist_ok=True)
    (d/'inherited_artifact_handoff.json').write_text(json.dumps(out,indent=2))
    print('V73_INHERITED_HANDOFF '+json.dumps(out),flush=True)
if __name__=='__main__': main()
