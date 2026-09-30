"""Acquire archived primary-document content for every supplied PIT issuer.

This reads released public Parquet archives, never the stopped SEC host. The
archive's cleaned `text` column is not used. Archive content is not automatically
promoted to a verified SEC input. Acquisition and final input approval are separate.
"""
from __future__ import annotations
import collections, datetime, gzip, hashlib, json, os, pathlib, re, shutil, sys, time
import urllib.request, urllib.error, urllib.parse, urllib.robotparser
import pyarrow.parquet as pq

BASE=pathlib.Path(__file__).resolve().parent
OUT=pathlib.Path('archive_work'); OUT.mkdir(exist_ok=True)
META=OUT/'metadata'; META.mkdir(exist_ok=True)
CHUNK=OUT/'chunk'; CHUNK.mkdir(exist_ok=True)
(CHUNK/'raw').mkdir(exist_ok=True)
UA='V73Research/1.0 (research contact: seungyun@berkeley.edu)'
REPOS=[('10-K','TeraflopAI/10-K-text','bdcc7da848820c5f74891f0a72fa27229bcaf972'),('10-Q','TeraflopAI/10-Q-text','1db4291296a24f270b572c76ca46a71074371ba6')]
STOP_CODES={401,403,429}; DENIED=False; LAST_REQUEST=0.0
CHUNK_LIMIT=40*1024*1024
chunk_rows=[]; chunk_bytes=0; chunk_id=0; seen=set(); completed=[]; errors=[]
counts=collections.Counter(); years=collections.Counter(); forms=collections.Counter(); issuers=set()
START=time.monotonic(); START_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()
manifest=json.loads((BASE/'scope_manifest.json').read_text())
scope=set((BASE/'ciks.txt').read_text().split())
scope.difference_update(manifest['transport_corrections']['remove']);scope.update(manifest['transport_corrections']['add'])
canonical=('\n'.join(sorted(scope))+'\n').encode()
if len(scope)!=manifest['issuer_count'] or hashlib.sha256(canonical).hexdigest()!=manifest['canonical_sha256']:
    raise RuntimeError('PIT issuer scope checksum mismatch; no acquisition attempted')
(META/'scope_ciks.txt').write_bytes(canonical)
(META/'scope_manifest.json').write_text(json.dumps(manifest,indent=2))

def request(url,bound,destination=None):
    global DENIED,LAST_REQUEST
    if DENIED:raise RuntimeError('PUBLIC_ARCHIVE_ACCESS_STOP_PRESERVED')
    delay=1-(time.monotonic()-LAST_REQUEST)
    if delay>0:time.sleep(delay)
    LAST_REQUEST=time.monotonic()
    h=hashlib.sha256();size=0;parts=[]
    try:
        req=urllib.request.Request(url,headers={'User-Agent':UA,'Accept-Encoding':'identity'})
        with urllib.request.urlopen(req,timeout=120) as res:
            handle=destination.open('wb') if destination else None
            try:
                while True:
                    b=res.read(1024*1024)
                    if not b:break
                    size+=len(b)
                    if size>bound:raise ValueError('bounded source object limit exceeded')
                    h.update(b)
                    if handle:handle.write(b)
                    else:parts.append(b)
            finally:
                if handle:handle.close()
        receipt={'url':url,'bytes':size,'sha256':h.hexdigest(),'retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'RETRIEVED'}
        with (META/'requests.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
        counts['network_bytes']+=size
        return (b''.join(parts) if destination is None else receipt)
    except urllib.error.HTTPError as e:
        if e.code in STOP_CODES:DENIED=True
        raise

def summary():
    return dict(started_at_utc=START_UTC,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=round(time.monotonic()-START,3),scope_issuers=len(scope),baseline_filing_rows=75253,membership_observations=2225543,source_security_records=912,counts=dict(counts),years=dict(sorted(years.items())),forms=dict(forms),acquired_issuers=len(issuers),completed_partitions=len(completed),partition_total=sum(len(x[3]) for x in repositories) if 'repositories' in globals() else None,archive_access_stop=DENIED,sec_access_stop_preserved=True,verified_financial_inputs=0,full_pit_ready=False,onedrive_written=False,backtest=False)

def emit_chunk(final=False):
    global chunk_rows,chunk_bytes,chunk_id
    if not chunk_rows and not final:return
    (CHUNK/'records.json').write_text(json.dumps(chunk_rows,separators=(',',':')))
    (CHUNK/'progress.json').write_text(json.dumps(summary(),indent=2))
    (CHUNK/'checkpoint.json').write_text(json.dumps({'completed_partitions':completed,'errors':errors,'seen_source_keys':sorted(seen),'final':final},separators=(',',':')))
    (CHUNK/'source_scope.json').write_text(json.dumps(manifest,indent=2))
    receipts={str(p.relative_to(CHUNK)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(CHUNK.rglob('*')) if p.is_file()}
    (CHUNK/'SHA256SUMS.json').write_text(json.dumps(receipts,sort_keys=True,indent=2))
    mh=hashlib.sha256((CHUNK/'SHA256SUMS.json').read_bytes()).hexdigest()
    message={'chunk':chunk_id,'root':str(CHUNK.resolve()),'manifest_sha256':mh,'records':len(chunk_rows),'compressed_original_bytes':chunk_bytes,'final':final,'progress':summary()}
    print('V73_CHUNK '+json.dumps(message),flush=True)
    reply=sys.stdin.readline().strip()
    if reply!='CONTINUE':raise RuntimeError('verified transfer acknowledgment missing; original chunk retained')
    shutil.rmtree(CHUNK);(CHUNK/'raw').mkdir(parents=True)
    chunk_rows=[];chunk_bytes=0;chunk_id+=1

def loadjson(v):
    if not isinstance(v,str):return v
    try:return json.loads(v)
    except (ValueError,TypeError):return None

def filer_records(obj):
    if isinstance(obj,list):
        for x in obj:yield from filer_records(x)
    elif isinstance(obj,dict):
        c=obj.get('company-data')
        if isinstance(c,dict) and c.get('cik') is not None:yield obj
        elif 'filer' in obj:yield from filer_records(obj['filer'])

try:
    robots_data=request('https://huggingface.co/robots.txt',2*1024*1024)
    (META/'robots.txt').write_bytes(robots_data);robot=urllib.robotparser.RobotFileParser();robot.parse(robots_data.decode('utf-8','replace').splitlines())
    repositories=[]
    for form,repo,rev in REPOS:
        raw=request('https://huggingface.co/api/datasets/'+repo+'/revision/'+rev,20*1024*1024)
        m=json.loads(raw)
        if m.get('gated') or m.get('private') or m.get('sha')!=rev:raise RuntimeError('archive is not the pinned public revision')
        files=sorted(x['rfilename'] for x in m.get('siblings',[]) if x['rfilename'].endswith('.parquet'))
        (META/(form+'_repository.json')).write_bytes(raw);repositories.append((form,repo,rev,files))
    plan=[]
    # Interleave form families without selecting by investment outcomes.
    for i in range(max(len(r[3]) for r in repositories)):
        for form,repo,rev,files in repositories:
            if i<len(files):plan.append((form,repo,rev,files[i]))
    (META/'plan.json').write_text(json.dumps(plan))
    print('V73_STARTED '+json.dumps(summary()),flush=True)
    for part_no,(form,repo,rev,name) in enumerate(plan):
        key=repo+'@'+rev+'/'+name
        url='https://huggingface.co/datasets/'+repo+'/resolve/'+rev+'/'+urllib.parse.quote(name,safe='/')
        if not robot.can_fetch(UA,url):errors.append({'partition':key,'status':'ROBOTS_DISALLOWED'});continue
        tmp=OUT/'partition.parquet'
        try:
            receipt=request(url,2*1024*1024*1024,tmp)
            pf=pq.ParquetFile(tmp)
            required=['content','metadata_accession-number','metadata_filing-date','metadata_documents','metadata_period','metadata_filer']
            if not set(required).issubset(pf.schema_arrow.names):raise ValueError('archive schema is not retained primary HTML')
            meta=pf.read(columns=required[1:],use_threads=False).to_pylist()
            wanted={}
            for row_index,row in enumerate(meta):
                counts['indexed_archive_rows']+=1
                acc=row.get('metadata_accession-number');fd=str(row.get('metadata_filing-date') or '')
                if not isinstance(acc,str) or not re.fullmatch(r'\d{10}-\d{2}-\d{6}',acc):counts['ambiguous_accession_rows']+=1;continue
                if not re.fullmatch(r'\d{8}',fd) or fd>'20260923':counts['outside_clock_rows']+=1;continue
                original_filers=list(filer_records(loadjson(row.get('metadata_filer'))))
                matched=[]
                for filer in original_filers:
                    cik=str(filer.get('company-data',{}).get('cik','')).zfill(10)
                    if cik in scope:matched.append((cik,filer))
                if not matched:continue
                docs=loadjson(row.get('metadata_documents'))
                if not isinstance(docs,list) or len(docs)!=1:counts['ambiguous_document_rows']+=1;continue
                doc=docs[0]
                if not isinstance(doc,dict) or str(doc.get('type','')).upper() not in ('10-K','10-K/A','10-K405','10-K405/A','10-KT','10-KT/A','10-Q','10-Q/A','10-QT','10-QT/A'):
                    counts['other_form_rows']+=1;continue
                wanted[row_index]=(row,matched,doc)
            base=0;selected=0
            for rg in range(pf.num_row_groups):
                n=pf.metadata.row_group(rg).num_rows
                positions=[x for x in wanted if base<=x<base+n]
                if positions:
                    content=pf.read_row_group(rg,columns=['content'],use_threads=False).column('content')
                    for absolute in positions:
                        row,matched,doc=wanted[absolute];v=content[absolute-base].as_py()
                        if not isinstance(v,str) or len(v)<1000:counts['invalid_primary_content']+=1;continue
                        b=v.encode('utf-8');digest=hashlib.sha256(b).hexdigest();acc=row['metadata_accession-number'];src_key=acc+'|'+digest
                        if src_key in seen:counts['duplicate_archive_rows']+=1;continue
                        if len(b)>64*1024*1024:counts['oversize_primary_content']+=1;errors.append({'accession':acc,'status':'OVER_64_MIB_NOT_DROPPED_FROM_SOURCE_QUEUE'});continue
                        compressed=gzip.compress(b,compresslevel=6,mtime=0)
                        dest=CHUNK/'raw'/(digest+'.content.gz');dest.write_bytes(compressed)
                        declared=doc.get('secsgml_size_bytes')
                        record={'accession':acc,'issuer_ciks':[c for c,f in matched],'original_filer_metadata':row['metadata_filer'],'filing_date':row['metadata_filing-date'],'period':row['metadata_period'],'primary_document':doc,'source_repository':repo,'source_revision':rev,'source_partition':name,'source_partition_sha256':receipt['sha256'],'source_row':absolute,'archive_content_sha256':digest,'archive_content_bytes':len(b),'gzip_bytes':len(compressed),'file':str(dest.relative_to(CHUNK)),'declared_primary_bytes_equal':declared==len(b),'source_level':'PUBLIC_ARCHIVE_RETAINED_PRIMARY_CONTENT_PENDING_SOURCE_VALIDATION','clean_text_used':False,'accepted_timestamp_from_archive':None,'financial_approved':False}
                        chunk_rows.append(record);chunk_bytes+=len(compressed);seen.add(src_key);counts['retained_primary_documents']+=1;counts['retained_content_bytes']+=len(b);counts['retained_gzip_bytes']+=len(compressed);selected+=1
                        years[str(row['metadata_filing-date'])[:4]]+=1;forms[str(doc['type'])]+=1;issuers.update(c for c,f in matched)
                        if chunk_bytes>=CHUNK_LIMIT:emit_chunk()
                    del content
                base+=n
            completed.append(key);counts['completed_partitions']+=1
            print('V73_PROGRESS '+json.dumps({'partition_index':part_no,'partition':name,'partition_form':form,'selected':selected,**summary()}),flush=True)
        except Exception as e:
            errors.append({'partition':key,'status':'ERROR','error':type(e).__name__+': '+str(e),'http_status':getattr(e,'code',None)})
            print('V73_PARTITION_ERROR '+json.dumps(errors[-1]),flush=True)
            if DENIED:break
        finally:
            tmp.unlink(missing_ok=True)
        if time.monotonic()-START>3.6*3600:
            errors.append({'status':'CHECKPOINT_BEFORE_RUNNER_DEADLINE','remaining_partitions':len(plan)-part_no-1});break
    emit_chunk(final=True)
except Exception as e:
    errors.append({'status':'STOPPED','error':type(e).__name__+': '+str(e)})
    print('V73_FATAL '+json.dumps(errors[-1]),flush=True)
    if chunk_rows:emit_chunk(final=True)
finally:
    (META/'summary.json').write_text(json.dumps(summary(),indent=2))
    (META/'checkpoint.json').write_text(json.dumps({'completed_partitions':completed,'errors':errors,'seen_source_keys':sorted(seen)},separators=(',',':')))
    print('V73_FINISHED '+json.dumps(summary()),flush=True)
