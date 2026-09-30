"""Apply only a race-safe state write to the bootstrapped acquisition driver."""
from pathlib import Path
p=Path('.github/actions/v73-retry-driver/driver.js')
s=p.read_text()
a=s.index('async function persistDurableKeys(spool){')
b=s.index('\nfunction receipt()',a)
old=s[a:b]
if 'async function' in old[len('async function persistDurableKeys(spool){'):]:
    raise ValueError('Unexpected function layout; refusing patch')
s=s[:a]+'''async function persistDurableKeys(spool){
 const docs=JSON.parse(fs.readFileSync(path.join(spool,'records.json'),'utf8'));
 const keys=new Set();
 for(const doc of docs){
  if(doc.source_level!=='OFFICIAL_SEC_HTTP200_ORIGINAL_PENDING_FIELD_VALIDATION')throw Error('non-official source in resume ledger');
  for(const cik of doc.issuer_ciks)keys.add(cik+'/'+doc.accession);
 }
 await require(path.resolve('research/v73_archive_scope/durable_key_store.cjs')).persist({api,branch,newKeys:keys});
}
'''+s[b:]
p.write_text(s)
print('V73_DURABLE_CONFLICT_RETRY_INSTALLED: refetch, validate, union, retry 409/422; no SEC or financial changes')
