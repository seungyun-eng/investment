'use strict';
const crypto = require('crypto');
const KEY = /^\d{10}\/\d{10}-\d{2}-\d{6}$/;
const STATE = 'research/v73_archive_scope/state/durable_official_keys.txt';
function decode(obj) {
  if (!obj || obj.encoding !== 'base64' || typeof obj.content !== 'string') throw Error('durable bytes unavailable');
  const clean = obj.content.replace(/\s/g,'');
  if (!/^[A-Za-z0-9+/]*={0,2}$/.test(clean) || clean.length % 4) throw Error('malformed state encoding');
  const data = Buffer.from(clean,'base64');
  if (data.toString('base64') !== clean) throw Error('noncanonical state encoding');
  if (typeof obj.size === 'number' && data.length !== obj.size) throw Error('state byte size mismatch');
  if (crypto.createHash('sha1').update(Buffer.from('blob '+data.length+'\0')).update(data).digest('hex') !== obj.sha) throw Error('state git blob mismatch');
  const text = new TextDecoder('utf-8',{fatal:true}).decode(data);
  const rows = text.trimEnd().split('\n');
  if (!rows.length || rows.some(k=>!KEY.test(k))) throw Error('invalid or empty durable keys');
  return new Set(rows);
}
async function read(api, branch) {
  const meta = await api('/contents/'+STATE+'?ref='+encodeURIComponent(branch));
  if (!meta || !/^[0-9a-f]{40}$/.test(meta.sha || '')) throw Error('durable state missing');
  let body = meta;
  if (meta.encoding !== 'base64' || !meta.content) {
    body = await api('/git/blobs/'+meta.sha);
    if (!body || body.sha !== meta.sha) throw Error('durable fallback identity mismatch');
  }
  return {sha:meta.sha, keys:decode(body)};
}
async function persist({api, branch, newKeys, sleep=ms=>new Promise(r=>setTimeout(r,ms)), attempts=8}) {
  const incoming=new Set(newKeys);
  if ([...incoming].some(k=>!KEY.test(k))) throw Error('invalid incoming durable key');
  if (!incoming.size) return;
  for(let attempt=0;attempt<attempts;attempt++) {
    const current=await read(api,branch);
    const combined=new Set([...current.keys,...incoming]);
    if(combined.size===current.keys.size) return;
    const bytes=Buffer.from([...combined].sort().join('\n')+'\n');
    try {
      await api('/contents/'+STATE,'PUT',{message:'research: merge verified durable SEC keys with conflict-safe retry',branch,sha:current.sha,content:bytes.toString('base64')});
      return;
    } catch(e) {
      if(!/GitHub (409|422)\b/.test(e.message || '') || attempt===attempts-1) throw e;
      // Re-read the current SHA and union on every retry; never reuse a stale write.
      await sleep(Math.min(12000,250*2**attempt));
    }
  }
  throw Error('durable update exhausted');
}
module.exports={decode,read,persist};
