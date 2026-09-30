'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto'),cp=require('child_process'),readline=require('readline');
const {DefaultArtifactClient}=require('@actions/artifact');
const client=new DefaultArtifactClient(),repo=process.env.GITHUB_REPOSITORY,run=Number(process.env.GITHUB_RUN_ID),token=process.env.GITHUB_TOKEN;
const branch='chatgpt-v73-raw-recovery-20260930',ceiling=400*1024*1024,start=Date.now();
const pending=[],sleep=ms=>new Promise(r=>setTimeout(r,ms));
function list(root){let a=[];for(const x of fs.readdirSync(root,{withFileTypes:true})){const p=path.join(root,x.name);if(x.isDirectory())a.push(...list(p));else if(x.isFile())a.push(p);}return a;}
async function api(route,method='GET'){const r=await fetch('https://api.github.com/repos/'+repo+route,{method,headers:{Authorization:'Bearer '+token,Accept:'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'}});if(r.status===404)return null;if(r.status===204)return {};if(!r.ok)throw Error('GitHub '+r.status+' '+route);return r.json();}
function receipt(){fs.mkdirSync('retry_work/metadata',{recursive:true});fs.writeFileSync('retry_work/metadata/transfers.json',JSON.stringify(pending,null,2));}
async function reconcile(){
 for(const x of pending){if(x.acknowledged)continue;
  const o=await api('/contents/research/v73_archive_scope/acks/'+run+'_'+x.artifact_id+'.json?ref='+encodeURIComponent(branch));if(!o||!o.content)continue;
  const a=JSON.parse(Buffer.from(o.content,'base64').toString('utf8'));
  if(a.verified!==true||a.run_id!==run||a.artifact_id!==x.artifact_id||a.manifest_sha256!==x.manifest_sha256||!/^[0-9a-f]{64}$/.test(a.downloaded_zip_sha256||''))throw Error('invalid external-copy acknowledgment');
  const m=await api('/actions/artifacts/'+x.artifact_id);
  if(!m||m.name!==x.name||m.workflow_run.id!==run)throw Error('artifact identity changed');
  if(m.digest&&m.digest!=='sha256:'+a.downloaded_zip_sha256)throw Error('ZIP digest differs');
  // Only this run's temporary artifact can be deleted, and only after a verified external copy.
  await api('/actions/artifacts/'+x.artifact_id,'DELETE');
  if(fs.existsSync(x.spool))fs.rmSync(x.spool,{recursive:true});
  x.acknowledged=true;x.external_zip_sha256=a.downloaded_zip_sha256;receipt();console.log('V73_TRANSFER_ACKNOWLEDGED '+JSON.stringify(x));
 }
}
async function space(required){while(true){await reconcile();const data=await api('/actions/artifacts?per_page=100');if(data.total_count>100)throw Error('artifact listing incomplete; no unmeasured storage expansion');const used=(data.artifacts||[]).filter(x=>!x.expired).reduce((s,x)=>s+x.size_in_bytes,0);if(used+required<ceiling)return;if(Date.now()-start>3.7*3600*1000)throw Error('runner checkpoint deadline, all unacknowledged originals retained');console.log('V73_STORAGE_BACKPRESSURE '+JSON.stringify({run_id:run,used_bytes:used,required_bytes:required,ceiling_bytes:ceiling,pending:pending.filter(x=>!x.acknowledged).map(x=>x.artifact_id)}));await sleep(15000);}}
async function main(){
 if(!token)throw Error('scoped Actions token missing');
 const p=cp.spawn('python3',['-u','research/v73_archive_scope/retry_collector.py'],{stdio:['pipe','pipe','inherit'],env:process.env});
 const closed=new Promise(resolve=>p.once('close',(code,signal)=>resolve({code,signal})));
 try{
  for await(const line of readline.createInterface({input:p.stdout})){console.log(line);if(!line.startsWith('V73_CHUNK '))continue;
   const c=JSON.parse(line.slice(10)),root=path.resolve(c.root);
   const manifest=fs.readFileSync(path.join(root,'SHA256SUMS.json'));if(crypto.createHash('sha256').update(manifest).digest('hex')!==c.manifest_sha256)throw Error('chunk manifest changed');
   const paths=list(root),size=paths.reduce((s,f)=>s+fs.statSync(f).size,0);await space(size+1024*1024);
   const name='v73-retry-raw-'+run+'-'+String(c.chunk).padStart(4,'0');
   const uploaded=await client.uploadArtifact(name,paths,root,{retentionDays:1,compressionLevel:0});if(!uploaded.id)throw Error('no artifact ID');
   const spool=path.resolve('retry_work/spool/'+String(c.chunk).padStart(4,'0'));fs.mkdirSync(path.dirname(spool),{recursive:true});fs.renameSync(root,spool);
   const row={run_id:run,artifact_id:uploaded.id,name,manifest_sha256:c.manifest_sha256,records:c.records,archive_size:uploaded.size,final:c.final,spool,acknowledged:false,progress:c.progress};pending.push(row);receipt();
   console.log('V73_TRANSFER_READY '+JSON.stringify(row));p.stdin.write('CONTINUE\n');
   await reconcile();
  }
  const result=await closed;if(result.code!==0)throw Error('collector stopped '+JSON.stringify(result));
  // Normal completion leaves any unacknowledged originals available as artifacts. No 90-minute idle wait.
  await reconcile();console.log('V73_RUN_FINISHED '+JSON.stringify({run_id:run,preserved_unacknowledged_artifacts:pending.filter(x=>!x.acknowledged).map(x=>x.artifact_id)}));
 }catch(e){receipt();p.stdin.end();p.kill('SIGTERM');console.error('V73_DRIVER_STOP '+e.stack);process.exitCode=1;}
}
main().catch(e=>{console.error(e.stack);process.exitCode=1;});
