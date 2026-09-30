"""Official SEC primary-body acquisition. No score/model changes or paid sources.
Receipts distinguish API index bytes, filing bodies, time availability and failures.
"""
from __future__ import annotations
import collections,datetime,gzip,hashlib,json,pathlib,re,signal,sys,time,urllib.request,urllib.error,urllib.parse
BASE=pathlib.Path(__file__).resolve().parent;OUT=pathlib.Path('retry_work');META=OUT/'metadata';CHUNK=OUT/'chunk';META.mkdir(parents=True,exist_ok=True);(CHUNK/'raw').mkdir(parents=True,exist_ok=True)
UA='Seungyun Lee V73 academic research seungyun@berkeley.edu';ALLOWED={'10-K','10-K/A','10-Q','10-Q/A','20-F','20-F/A'}
start=time.monotonic();started=datetime.datetime.now(datetime.timezone.utc).isoformat();last=0.;stop=False;terminate=False;rows=[];cb=0;cid=0;counts=collections.Counter();seen=set();issuer_done=[];all_tasks=[];errors=[]
scope_meta=json.loads((BASE/'scope_manifest.json').read_text());scope=set((BASE/'ciks.txt').read_text().split());scope.difference_update(scope_meta['transport_corrections']['remove']);scope.update(scope_meta['transport_corrections']['add']);canon=('\n'.join(sorted(scope))+'\n').encode();assert len(scope)==809 and hashlib.sha256(canon).hexdigest()==scope_meta['canonical_sha256'];(META/'scope_ciks.txt').write_bytes(canon)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,v):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,separators=(',',':')));q.replace(p)
def summary():return dict(started_at_utc=started,updated_at_utc=stamp(),elapsed_seconds=time.monotonic()-start,source='OFFICIAL_SEC_EDGAR_PRIMARY',scope_issuers=809,baseline_filing_rows=75253,counts=dict(counts),source_index_issuers_completed=len(issuer_done),discovered_tasks=len(all_tasks),access_stopped=stop,approved_inputs=0,onedrive_written=False,backtest=False)
def checkpoint():save(META/'summary.json',summary());save(META/'checkpoint.json',dict(issuer_done=issuer_done,seen=sorted(seen),tasks=all_tasks,errors=errors))
def ending(*args):
 global terminate
 terminate=True
signal.signal(signal.SIGTERM,ending);signal.signal(signal.SIGINT,ending)
def get(url,bound):
 global last,stop
 if stop:raise RuntimeError('SEC_ACCESS_STOP')
 time.sleep(max(0,.5-(time.monotonic()-last)));last=time.monotonic();row=dict(url=url,requested_at_utc=stamp())
 try:
  req=urllib.request.Request(url,headers={'User-Agent':UA,'Accept-Encoding':'identity'})
  with urllib.request.urlopen(req,timeout=45) as r:data=r.read(bound+1);row.update(http_status=r.status,response_url=r.geturl(),content_type=r.headers.get('Content-Type'))
  if len(data)>bound:raise ValueError('BYTE_BOUND_EXCEEDED')
  row.update(status='RETRIEVED',bytes=len(data),sha256=hashlib.sha256(data).hexdigest());counts['network_bytes']+=len(data)
  return data,row
 except urllib.error.HTTPError as e:
  row.update(status='HTTP_ERROR',http_status=e.code,error=str(e));stop=e.code in (401,403,429);raise
 except Exception as e:row.update(status='ERROR',error=repr(e));raise
 finally:
  counts['http_requests']+=1
  with (META/'requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
def emit(final=False):
 global rows,cb,cid
 checkpoint()
 if not rows:return
 save(CHUNK/'records.json',rows);save(CHUNK/'progress.json',summary());save(CHUNK/'checkpoint.json',dict(issuer_done=issuer_done,seen=sorted(seen),errors=errors,final=final))
 m={str(p.relative_to(CHUNK)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(CHUNK.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'};save(CHUNK/'SHA256SUMS.json',m)
 print('V73_CHUNK '+json.dumps(dict(chunk=cid,root=str(CHUNK.resolve()),manifest_sha256=hashlib.sha256((CHUNK/'SHA256SUMS.json').read_bytes()).hexdigest(),records=len(rows),final=final,progress=summary())),flush=True)
 if sys.stdin.readline().strip()!='CONTINUE':raise RuntimeError('DURABLE_UPLOAD_NOT_CONFIRMED')
 (CHUNK/'raw').mkdir(parents=True,exist_ok=True);rows=[];cb=0;cid+=1
def candidates(blob,cik):
 arr=blob.get('filings',{}).get('recent',blob);n=len(arr.get('accessionNumber',[]));out=[]
 for i in range(n):
  def value(k):
   a=arr.get(k,[]);return a[i] if len(a)>i else None
  form=value('form');acc=value('accessionNumber');date=value('filingDate');doc=value('primaryDocument')
  if form not in ALLOWED:continue
  if not isinstance(acc,str) or not re.fullmatch(r'\d{10}-\d{2}-\d{6}',acc):continue
  if not isinstance(date,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',date) or date>'2026-09-23':continue
  if not isinstance(doc,str) or not doc or '/' in doc or '\\' in doc or '..' in doc:errors.append(dict(cik=cik,accession=acc,status='PRIMARY_FILENAME_UNAVAILABLE'));continue
  out.append(dict(cik=cik,accession=acc,form=form,filing_date=date,period=value('reportDate'),accepted_at=value('acceptanceDateTime'),primary_document=doc))
 return out
try:
 print('V73_STARTED '+json.dumps(summary()),flush=True)
 for cik in sorted(scope):
  if stop or terminate or time.monotonic()-start>3.55*3600:break
  tasks=[];index_refs=[]
  try:
   data,rec=get('https://data.sec.gov/submissions/CIK'+cik+'.json',24*1024**2);main=json.loads(data);assert str(main.get('cik','')).zfill(10)==cik
   target=META/'api';target.mkdir(exist_ok=True);(target/(rec['sha256']+'.json.gz')).write_bytes(gzip.compress(data,mtime=0));index_refs.append(rec);tasks+=candidates(main,cik)
   for history in main.get('filings',{}).get('files',[]):
    if stop or terminate:break
    name=history.get('name','')
    if not re.fullmatch(r'CIK\d{10}-submissions-\d+\.json',name):errors.append(dict(cik=cik,status='HISTORY_NAME_UNEXPECTED',name=name));continue
    body,hr=get('https://data.sec.gov/submissions/'+name,48*1024**2);(target/(hr['sha256']+'.json.gz')).write_bytes(gzip.compress(body,mtime=0));index_refs.append(hr);tasks+=candidates(json.loads(body),cik)
   tasks=list({x['accession']:x for x in tasks}.values());tasks.sort(key=lambda x:(x['filing_date'],x['accession']));all_tasks.extend(tasks);issuer_done.append(cik);checkpoint()
   for task in tasks:
    if stop or terminate or time.monotonic()-start>3.55*3600:break
    key=cik+'/'+task['accession']
    if key in seen:continue
    url='https://www.sec.gov/Archives/edgar/data/'+str(int(cik))+'/'+task['accession'].replace('-','')+'/'+urllib.parse.quote(task['primary_document'])
    try:
     body,r=get(url,64*1024**2)
     if len(body)<500 or any(x in body[:20000].lower() for x in (b'request rate threshold exceeded',b'your request originates from an undeclared automated tool')):raise ValueError('NOT_A_FILING_BODY')
     if not any(x in body[:10000].lower() for x in (b'<html',b'<document',b'<!doctype',b'<?xml',b'securities and exchange',b'<page>')):raise ValueError('UNRECOGNIZED_BODY_FORMAT')
     h=r['sha256'];comp=gzip.compress(body,compresslevel=6,mtime=0);p=CHUNK/'raw'/(h+'.content.gz');p.write_bytes(comp)
     rows.append(dict(accession=task['accession'],issuer_ciks=[cik],filing_date=task['filing_date'],period=task['period'],accepted_timestamp_from_api=task['accepted_at'],primary_document=dict(type=task['form'],filename=task['primary_document']),source_url=url,source_level='OFFICIAL_SEC_HTTP200_ORIGINAL_PENDING_FIELD_VALIDATION',archive_content_sha256=h,archive_content_bytes=len(body),gzip_bytes=len(comp),file=str(p.relative_to(CHUNK)),retrieved_at_utc=r['requested_at_utc'],api_source_receipts=index_refs,financial_approved=False,clean_text_used=False))
     seen.add(key);cb+=len(comp);counts['retained_documents']+=1;counts['retained_content_bytes']+=len(body);counts['retained_gzip_bytes']+=len(comp)
     if cb>=24*1024**2:emit()
    except Exception as e:
     errors.append(dict(cik=cik,accession=task['accession'],url=url,status='BODY_REQUEST_OR_VALIDATION_FAILED',error=repr(e)));checkpoint()
     if stop:break
   print('V73_PROGRESS '+json.dumps(summary()),flush=True)
  except Exception as e:
   errors.append(dict(cik=cik,status='ISSUER_INDEX_FAILED',error=repr(e)));checkpoint();print('V73_INDEX_ERROR '+json.dumps(errors[-1]),flush=True)
   if stop:break
 emit(final=True)
finally:
 checkpoint();print('V73_FINISHED '+json.dumps(summary()),flush=True)
