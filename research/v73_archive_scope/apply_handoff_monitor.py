"""Add inherited-ACK reconciliation and observable state to the existing driver."""
from pathlib import Path

INSERT = r'''
// V73_HANDOFF_MONITOR_V1: source progress is distinct from job liveness.
let health={run_id:run,phase:'starting',updated_at_utc:new Date().toISOString()},healthBusy=false,lastInherited=0,lastPublished=0;
function observeProgress(line){
 for(const tag of ['V73_STARTED ','V73_PROGRESS ','V73_CHUNK ','V73_FINISHED ']){
  if(!line.startsWith(tag))continue;
  try{const value=JSON.parse(line.slice(tag.length));const p=value.progress||value;
   health={...health,phase:tag==='V73_FINISHED '?'collector_finished':'collecting',source_updated_at_utc:p.updated_at_utc,counts:p.counts,source_index_issuers_completed:p.source_index_issuers_completed};
  }catch(e){console.error('V73_HEALTH_PARSE_FAILED '+e.message);}
 }
}
async function publishHealth(){
 if(healthBusy||Date.now()-lastPublished<60000)return;healthBusy=true;lastPublished=Date.now();
 try{
  const endpoint='/contents/research/v73_archive_scope/state/collector_health_'+run+'.json';
  const old=await api(endpoint+'?ref='+encodeURIComponent(branch));
  const h={...health,updated_at_utc:new Date().toISOString(),pending_artifact_ids:pending.filter(x=>!x.acknowledged).map(x=>x.artifact_id)};
  fs.mkdirSync('retry_work/metadata',{recursive:true});fs.writeFileSync('retry_work/metadata/collector_health.json',JSON.stringify(h));
  await api(endpoint,'PUT',{message:'research: publish actual collector phase and source checkpoint',branch,content:Buffer.from(JSON.stringify(h)+'\n').toString('base64'),...(old?{sha:old.sha}:{})});
 }catch(e){console.error('V73_HEALTH_PUBLISH_FAILED '+e.message);}finally{healthBusy=false;}
}
async function reconcileInherited(){
 if(Date.now()-lastInherited<60000)return;lastInherited=Date.now();
 await new Promise((resolve,reject)=>{
  const script="import sys;sys.path.insert(0,'research/v73_archive_scope');import artifact_handoff as h;h.MAX_ZIP=160*1024**2;h.main()";
  cp.execFile('python3',['-c',script],{timeout:180000,maxBuffer:2*1024*1024},(err,out,stderr)=>{
   if(out)console.log(out.trim());if(stderr)console.error(stderr.trim());if(err)reject(err);else resolve();
  });
 });
}
const healthTimer=setInterval(()=>{publishHealth();},60000);healthTimer.unref();
'''

def patch(source):
    if 'V73_HANDOFF_MONITOR_V1' in source:return source
    if source.count('function receipt(){')!=1:raise ValueError('unknown driver receipt layout')
    source=source.replace('function receipt(){',INSERT+'\nfunction receipt(){',1)
    before='async function space(required){while(true){await reconcile();'
    if source.count(before)!=1:raise ValueError('unknown storage wait layout')
    source=source.replace(before,'async function space(required){while(true){await reconcile();await reconcileInherited();',1)
    before="console.log('V73_STORAGE_BACKPRESSURE '+JSON.stringify("
    if source.count(before)!=1:raise ValueError('unknown pressure reporting layout')
    source=source.replace(before,"health={...health,phase:'waiting_for_storage',used_bytes:used,required_bytes:required,ceiling_bytes:ceiling};await publishHealth();"+before,1)
    before='console.log(line);if(!line.startsWith('
    if source.count(before)!=1:raise ValueError('unknown source progress layout')
    source=source.replace(before,'console.log(line);observeProgress(line);if(!line.startsWith(',1)
    before="console.log('V73_TRANSFER_READY '+JSON.stringify(row));p.stdin.write('CONTINUE\\n');"
    if source.count(before)!=1:raise ValueError('unknown upload handshake')
    source=source.replace(before,"health={...health,phase:'collecting',last_uploaded_artifact_id:uploaded.id};"+before,1)
    return source

def main():
    p=Path('.github/actions/v73-retry-driver/driver.js')
    p.write_text(patch(p.read_text()))
    print('V73_HANDOFF_MONITOR_INSTALLED: inherited completed-run ACKs plus live source/phase checkpoint')
if __name__=='__main__':main()
