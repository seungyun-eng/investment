"""Resume existing SEC acquisition using only durably preserved document keys."""
from __future__ import annotations
import base64, gzip, hashlib, io, json, os, pathlib, re, time, urllib.request, urllib.error, zipfile
ROOT = pathlib.Path(__file__).resolve().parent
REPO = os.environ['GITHUB_REPOSITORY']
TOKEN = os.environ['GITHUB_TOKEN']
BRANCH = 'chatgpt-v73-raw-recovery-20260930'
STATE = 'research/v73_archive_scope/state/durable_official_keys.txt'
PROOF = json.loads((ROOT / 'preserved_official_36707672045.json').read_text())
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl): return None
opener = urllib.request.build_opener(NoRedirect)
def request(route, method='GET', payload=None, binary=False):
    url = 'https://api.github.com/repos/' + REPO + route
    data = None if payload is None else json.dumps(payload).encode()
    for attempt in range(6):
        req = urllib.request.Request(url, data=data, method=method, headers={'Authorization':'Bearer '+TOKEN,'Accept':'application/vnd.github+json','Content-Type':'application/json','User-Agent':'V73-verified-transfer'})
        try:
            with opener.open(req, timeout=45) as resp: raw = resp.read()
            return raw if binary else (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and method == 'GET': return None
            if exc.code in (301,302,303,307,308) and binary:
                location = exc.headers.get('Location','')
                if not location.startswith('https://'): raise ValueError('non-HTTPS artifact redirect')
                with urllib.request.urlopen(location, timeout=120) as resp: return resp.read(64*1024**2+1)
            if exc.code not in (429,500,502,503,504) or attempt == 5: raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 5: raise
        time.sleep(min(30,2**attempt))
    raise RuntimeError('unreachable request state')
def decode_state(obj):
    if obj is None: return set()
    keys = set(base64.b64decode(obj['content']).decode().splitlines())
    if any(not re.fullmatch(r'\d{10}/\d{10}-\d{2}-\d{6}',k) for k in keys): raise ValueError('invalid preserved key')
    return keys
state = request('/contents/' + STATE + '?ref=' + BRANCH)
keys = decode_state(state)
if not keys:
    recovered = set()
    for spec in PROOF['temporary_artifacts']:
        meta = request('/actions/artifacts/' + str(spec['id']))
        if not meta or meta['workflow_run']['id'] != PROOF['run_id'] or meta['name'] != spec['name']: raise ValueError('missing or wrong bootstrap artifact')
        raw = request('/actions/artifacts/' + str(spec['id']) + '/zip', binary=True)
        if raw is None or hashlib.sha256(raw).hexdigest() != spec['zip_sha256']: raise ValueError('bootstrap ZIP checksum mismatch')
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            manifest_raw = z.read('SHA256SUMS.json')
            if hashlib.sha256(manifest_raw).hexdigest() != spec['manifest_sha256']: raise ValueError('bootstrap manifest mismatch')
            for name,m in json.loads(manifest_raw).items():
                b = z.read(name)
                if len(b) != m['bytes'] or hashlib.sha256(b).hexdigest() != m['sha256']: raise ValueError('bootstrap member mismatch')
            records = json.loads(z.read('records.json'))
            if len(records) != spec['records_verified']: raise ValueError('bootstrap row count mismatch')
            for r in records:
                body = gzip.decompress(z.read(r['file']))
                if len(body) != r['archive_content_bytes'] or hashlib.sha256(body).hexdigest() != r['archive_content_sha256']: raise ValueError('original body mismatch')
                recovered.update(c+'/'+r['accession'] for c in r['issuer_ciks'])
    key_bytes = ('\n'.join(sorted(recovered))+'\n').encode()
    if len(recovered) != PROOF['key_count'] or hashlib.sha256(key_bytes).hexdigest() != PROOF['keys_sha256']: raise ValueError('preserved key set mismatch')
    payload = {'message':'research: persist verified official-original resume keys','branch':BRANCH,'content':base64.b64encode(key_bytes).decode()}
    if state: payload['sha'] = state['sha']
    request('/contents/' + STATE,'PUT',payload)
    keys = recovered
elif len(keys) < PROOF['key_count']:
    raise ValueError('durable resume index unexpectedly contracted')
for spec in PROOF['temporary_artifacts']:
    meta = request('/actions/artifacts/' + str(spec['id']))
    if meta is None: continue
    if spec['durable_external_copy_verified'] is not True or meta['workflow_run']['id'] != PROOF['run_id'] or meta['name'] != spec['name'] or meta.get('digest') != 'sha256:'+spec['zip_sha256']: raise ValueError('unsafe temporary cleanup')
    request('/actions/artifacts/'+str(spec['id']),'DELETE')
pathlib.Path('retry_work').mkdir(exist_ok=True)
pathlib.Path('retry_work/preserved_keys.txt').write_text('\n'.join(sorted(keys))+'\n')
p = ROOT/'direct_sec_collector.py'
s = p.read_text()
assert s.count('seen=set();') == 1
s = s.replace('seen=set();',"seen=set(pathlib.Path('retry_work/preserved_keys.txt').read_text().splitlines());")
s = s.replace('3.55*3600','5.2*3600')
compile(s,str(p),'exec');p.write_text(s)
p = pathlib.Path('.github/actions/v73-retry-driver/driver.js')
s = p.read_text();a = s.index('async function api(');b = s.index('\nfunction receipt()',a)
s = s[:a]+'''async function api(route,method='GET',body){
 let last;
 for(let attempt=0;attempt<8;attempt++){
  try{
   const r=await fetch('https://api.github.com/repos/'+repo+route,{method,signal:AbortSignal.timeout(60000),headers:{Authorization:'Bearer '+token,Accept:'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
   if(r.status===404&&method==='GET')return null;if(r.status===204)return {};
   if(!r.ok){const e=Error('GitHub '+r.status+' '+route);e.retry=[429,500,502,503,504].includes(r.status);throw e;}
   return await r.json();
  }catch(e){last=e;if(e.retry===false||attempt===7)throw e;await sleep(Math.min(30000,1000*2**attempt));}
 }
 throw last;
}
async function persistDurableKeys(spool){
 const endpoint='/contents/research/v73_archive_scope/state/durable_official_keys.txt';
 const current=await api(endpoint+'?ref='+encodeURIComponent(branch));
 const keys=new Set(current?Buffer.from(current.content,'base64').toString('utf8').trim().split('\\n').filter(Boolean):[]);
 const docs=JSON.parse(fs.readFileSync(path.join(spool,'records.json'),'utf8'));
 for(const doc of docs){if(doc.source_level!=='OFFICIAL_SEC_HTTP200_ORIGINAL_PENDING_FIELD_VALIDATION')throw Error('non-official record in official resume ledger');for(const cik of doc.issuer_ciks)keys.add(cik+'/'+doc.accession);}
 const text=[...keys].sort().join('\\n')+'\\n';
 await api(endpoint,'PUT',{message:'research: checkpoint durably preserved official SEC bodies',branch,content:Buffer.from(text).toString('base64'),...(current?{sha:current.sha}:{})});
}
'''+s[b:]
needle="  await api('/actions/artifacts/'+x.artifact_id,'DELETE');"
assert s.count(needle)==1
s=s.replace(needle,"  await persistDurableKeys(x.spool);\n"+needle)
s=s.replace('3.7*3600*1000','5.35*3600*1000')
p.write_text(s)
print(json.dumps({'resume_keys':len(keys),'source':'verified durable-copy index','unflushed_old_keys_skipped':False,'paid_storage_enabled':False}))
