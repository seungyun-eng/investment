"""Resume-safe free public-archive acquisition; never requests the stopped SEC host.
Retained `content` is a secondary archive copy, NOT approved SEC input.
"""
from __future__ import annotations
import collections, datetime, gzip, hashlib, json, pathlib, re, signal, sys, time
import urllib.request, urllib.error, urllib.parse, urllib.robotparser
import pyarrow.parquet as pq

BASE=pathlib.Path(__file__).resolve().parent
OUT=pathlib.Path('retry_work'); META=OUT/'metadata'; CHUNK=OUT/'chunk'
META.mkdir(parents=True,exist_ok=True); (CHUNK/'raw').mkdir(parents=True,exist_ok=True)
UA='V73Research/1.0 (research contact: seungyun@berkeley.edu)'
REPOS=[('10-K','TeraflopAI/10-K-text','bdcc7da848820c5f74891f0a72fa27229bcaf972'),('10-Q','TeraflopAI/10-Q-text','1db4291296a24f270b572c76ca46a71074371ba6')]
ALLOWED={'10-K','10-K/A','10-K405','10-K405/A','10-KT','10-KT/A','10-Q','10-Q/A','10-QT','10-QT/A'}
counts=collections.Counter(); years=collections.Counter(); forms=collections.Counter()
seen=set(); issuers=set(); completed=[]; errors=[]; rows=[]; chunk_bytes=0; chunk_id=0
started=time.monotonic(); started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
last_request=0.; access_stop=False; terminate=False; plan=[]
scope_meta=json.loads((BASE/'scope_manifest.json').read_text())
scope=set((BASE/'ciks.txt').read_text().split()); scope.difference_update(scope_meta['transport_corrections']['remove']); scope.update(scope_meta['transport_corrections']['add'])
b=('\n'.join(sorted(scope))+'\n').encode()
assert len(scope)==809 and hashlib.sha256(b).hexdigest()==scope_meta['canonical_sha256']
(META/'scope_ciks.txt').write_bytes(b)
(META/'scope_manifest.json').write_text(json.dumps(scope_meta,indent=2))

def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,separators=(',',':'))); tmp.replace(path)
def summary():
    return dict(started_at_utc=started_utc,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=round(time.monotonic()-started,2),scope_issuers=809,baseline_filing_rows=75253,counts=dict(counts),years=dict(sorted(years.items())),forms=dict(forms),acquired_issuers=len(issuers),completed_partitions=len(completed),partition_total=len(plan),archive_access_stop=access_stop,sec_access_stop_preserved=True,approved_inputs=0,onedrive_written=False,backtest=False)
def checkpoint():
    atomic(META/'summary.json',summary()); atomic(META/'checkpoint.json',dict(completed_partitions=completed,errors=errors,seen_source_keys=sorted(seen)))
def on_signal(*unused):
    global terminate
    terminate=True
signal.signal(signal.SIGTERM,on_signal); signal.signal(signal.SIGINT,on_signal)
def request(url,bound,destination=None):
    global last_request,access_stop
    if access_stop:raise RuntimeError('ARCHIVE_ACCESS_STOP_PRESERVED')
    time.sleep(max(0,1-(time.monotonic()-last_request))); last_request=time.monotonic()
    h=hashlib.sha256(); size=0; parts=[]; handle=None
    try:
        req=urllib.request.Request(url,headers={'User-Agent':UA,'Accept-Encoding':'identity'})
        with urllib.request.urlopen(req,timeout=120) as r:
            if destination:handle=destination.open('wb')
            while True:
                part=r.read(1024*1024)
                if not part:break
                size+=len(part)
                if size>bound:raise ValueError('source object exceeds byte bound')
                h.update(part)
                if handle:handle.write(part)
                else:parts.append(part)
        receipt=dict(url=url,bytes=size,sha256=h.hexdigest(),status='RETRIEVED',retrieved_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        with (META/'requests.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
        counts['network_bytes']+=size
        return receipt if destination else b''.join(parts)
    except urllib.error.HTTPError as e:
        if e.code in (401,403,429):access_stop=True
        raise
    finally:
        if handle:handle.close()
def emit(final=False):
    global rows,chunk_bytes,chunk_id
    checkpoint()
    if not rows:return
    atomic(CHUNK/'records.json',rows); atomic(CHUNK/'progress.json',summary())
    atomic(CHUNK/'checkpoint.json',dict(completed_partitions=completed,errors=errors,seen_source_keys=sorted(seen),final=final))
    hashes={str(p.relative_to(CHUNK)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(CHUNK.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
    atomic(CHUNK/'SHA256SUMS.json',hashes)
    mh=hashlib.sha256((CHUNK/'SHA256SUMS.json').read_bytes()).hexdigest()
    print('V73_CHUNK '+json.dumps(dict(chunk=chunk_id,root=str(CHUNK.resolve()),manifest_sha256=mh,records=len(rows),final=final,progress=summary())),flush=True)
    if sys.stdin.readline().strip()!='CONTINUE':raise RuntimeError('durable artifact upload acknowledgment missing')
    # Driver moves the chunk only after artifact upload; no original is deleted here.
    (CHUNK/'raw').mkdir(parents=True,exist_ok=True)
    rows=[];chunk_bytes=0;chunk_id+=1

def parse(v):
    if not isinstance(v,str):return v
    try:return json.loads(v)
    except (ValueError,TypeError):return None
def filers(v):
    if isinstance(v,list):
        for x in v:yield from filers(x)
    elif isinstance(v,dict):
        if isinstance(v.get('company-data'),dict) and v['company-data'].get('cik') is not None:yield v
        elif 'filer' in v:yield from filers(v['filer'])

try:
    robots=request('https://huggingface.co/robots.txt',2*1024**2);(META/'robots.txt').write_bytes(robots)
    rp=urllib.robotparser.RobotFileParser();rp.parse(robots.decode('utf-8','replace').splitlines())
    sources=[]
    for form,repo,rev in REPOS:
        data=request('https://huggingface.co/api/datasets/'+repo+'/revision/'+rev,20*1024**2);m=json.loads(data)
        assert not m.get('gated') and not m.get('private') and m.get('sha')==rev
        names=sorted(x['rfilename'] for x in m['siblings'] if x['rfilename'].endswith('.parquet'))
        (META/(form+'_source.json')).write_bytes(data);sources.append((form,repo,rev,names))
    for i in range(max(len(x[3]) for x in sources)):
        for form,repo,rev,names in sources:
            if i<len(names):plan.append((form,repo,rev,names[i]))
    atomic(META/'plan.json',plan)
    print('V73_STARTED '+json.dumps(summary()),flush=True)
    columns=['metadata_accession-number','metadata_filing-date','metadata_documents','metadata_period','metadata_filer']
    for pi,(family,repo,rev,name) in enumerate(plan):
        if terminate or time.monotonic()-started>3.6*3600:break
        key=repo+'@'+rev+'/'+name;url='https://huggingface.co/datasets/'+repo+'/resolve/'+rev+'/'+urllib.parse.quote(name,safe='/')
        if not rp.can_fetch(UA,url):errors.append(dict(partition=key,status='ROBOTS_DISALLOWED'));checkpoint();continue
        tmp=OUT/'partition.parquet';bad=False;chosen=0;base=0
        try:
            receipt=request(url,2*1024**3,tmp);pf=pq.ParquetFile(tmp)
            assert set(columns+['content']).issubset(pf.schema_arrow.names)
            for rg in range(pf.num_row_groups):
                n=pf.metadata.row_group(rg).num_rows
                if terminate:break
                try:
                    metadata=pf.read_row_group(rg,columns=columns,use_threads=False).to_pylist();counts['indexed_archive_rows']+=len(metadata)
                    wanted=[]
                    for ri,row in enumerate(metadata):
                        acc=row.get(columns[0]);date=str(row.get(columns[1]) or '')
                        if not isinstance(acc,str) or not re.fullmatch(r'\d{10}-\d{2}-\d{6}',acc):counts['bad_accession']+=1;continue
                        if not re.fullmatch(r'\d{8}',date) or date>'20260923':counts['outside_clock']+=1;continue
                        fs=list(filers(parse(row.get('metadata_filer'))));matched=[str(f['company-data']['cik']).zfill(10) for f in fs if str(f['company-data']['cik']).zfill(10) in scope]
                        if not matched:continue
                        docs=parse(row.get('metadata_documents'))
                        if not isinstance(docs,list) or len(docs)!=1 or not isinstance(docs[0],dict):counts['ambiguous_primary']+=1;continue
                        if str(docs[0].get('type','')).upper() not in ALLOWED:counts['other_form']+=1;continue
                        wanted.append((ri,row,matched,docs[0]))
                    if not wanted:base+=n;continue
                    content=pf.read_row_group(rg,columns=['content'],use_threads=False).column('content')
                    for ri,row,matched,doc in wanted:
                        value=content[ri].as_py()
                        if not isinstance(value,str) or len(value)<1000:counts['invalid_content']+=1;continue
                        data=value.encode('utf-8');digest=hashlib.sha256(data).hexdigest();acc=row[columns[0]];sid=acc+'|'+digest
                        if sid in seen:counts['duplicate_archive_record']+=1;continue
                        if len(data)>64*1024**2:errors.append(dict(accession=acc,status='OVERSIZE_NOT_APPROVED'));continue
                        compressed=gzip.compress(data,compresslevel=6,mtime=0);dest=CHUNK/'raw'/(digest+'.content.gz');dest.write_bytes(compressed)
                        rows.append(dict(accession=acc,issuer_ciks=matched,original_filer_metadata=row['metadata_filer'],filing_date=row[columns[1]],period=row['metadata_period'],primary_document=doc,source_repository=repo,source_revision=rev,source_partition=name,source_partition_sha256=receipt['sha256'],source_row=base+ri,archive_content_sha256=digest,archive_content_bytes=len(data),gzip_bytes=len(compressed),file=str(dest.relative_to(CHUNK)),declared_primary_bytes_equal=doc.get('secsgml_size_bytes')==len(data),source_level='PUBLIC_ARCHIVE_RETAINED_PRIMARY_CONTENT_PENDING_VALIDATION',clean_text_used=False,financial_approved=False))
                        seen.add(sid);chunk_bytes+=len(compressed);chosen+=1;counts['retained_documents']+=1;counts['retained_content_bytes']+=len(data);counts['retained_gzip_bytes']+=len(compressed);years[str(row[columns[1]])[:4]]+=1;forms[str(doc['type'])]+=1;issuers.update(matched)
                        if chunk_bytes>=96*1024**2:emit()
                    del content
                except Exception as e:
                    bad=True;errors.append(dict(partition=key,row_group=rg,status='ROW_GROUP_ERROR',error=type(e).__name__+': '+str(e)));print('V73_ROW_GROUP_ERROR '+json.dumps(errors[-1]),flush=True)
                    # A damaged group never invalidates intact groups and is never marked complete.
                base+=n
            if not bad and not terminate:completed.append(key)
            checkpoint();print('V73_PROGRESS '+json.dumps(dict(partition_index=pi,partition=name,selected=chosen,**summary())),flush=True)
        except Exception as e:
            errors.append(dict(partition=key,status='PARTITION_ERROR',error=type(e).__name__+': '+str(e),http_status=getattr(e,'code',None)));checkpoint();print('V73_ERROR '+json.dumps(errors[-1]),flush=True)
            if access_stop:break
        finally:tmp.unlink(missing_ok=True)
    emit(final=True)
finally:
    checkpoint();print('V73_FINISHED '+json.dumps(summary()),flush=True)
