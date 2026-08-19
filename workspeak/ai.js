async function callWorkSpeakAI(payload){
  const r=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  if(!r.ok){let msg='AI 분석 서버에 연결하지 못했습니다.';try{const e=await r.json();if(e.error)msg+=' '+e.error}catch{}throw new Error(msg)}
  return r.json();
}
function aiBusy(el,on,label='AI가 분석 중...'){
  if(!el)return;
  if(on){el.dataset.prev=el.textContent;el.textContent=label;el.disabled=true}else{el.textContent=el.dataset.prev||el.textContent;el.disabled=false}
}
function findButton(fnName){return [...document.querySelectorAll('button')].find(b=>(b.getAttribute('onclick')||'').includes(fnName))}
function listText(items,empty='없음'){return items&&items.length?items.map(x=>'• '+x).join('\n'):empty}

analyzeSpeech=async function(){
  const t=cleanText($('speech').textContent);
  if(!t)return alert('먼저 영어로 말하거나 직접 입력해보세요.');
  const btn=findButton('analyzeSpeech');aiBusy(btn,true);
  $('speakScore').textContent='…';
  try{
    const a=await callWorkSpeakAI({mode:'speaking',prompt:$('topic')?.value||'',answer:t});
    $('mine').textContent=t;
    $('natural').textContent=a.natural_version||'—';
    $('pro').textContent=a.executive_version||'—';
    $('speakScore').textContent=Math.round(a.score);
    $('why').textContent=(a.valid?'':`평가 불가: `)+(a.explanation_ko||a.verdict||'');
    latestUpgrade=a.executive_version||a.natural_version||'';
    if(a.valid){state.speaking.push({date:now(),score:a.score,text:t,upgrade:latestUpgrade,ai:true,english:a.english,structure:a.structure,judgment:a.judgment,exec:a.executive_presence});state.lastDate=now();save('workspeak_v2',state);renderHome()}
  }catch(e){$('speakScore').textContent='—';$('why').textContent=e.message;alert(e.message)}finally{aiBusy(btn,false)}
};

scoreCase=async function(){
  const t=cleanText($('mbaSpeech').textContent);
  if(!t)return alert('먼저 MBA 케이스에 영어로 말하거나 직접 입력해보세요.');
  const c=cases[caseIdx];const btn=findButton('scoreCase');aiBusy(btn,true);
  $('mbaFeedbackCard').style.display='block';$('mbaOverall').textContent='…';
  $('mbaStrong').textContent='답변의 의미와 케이스 사실을 비교하고 있습니다.';$('mbaImprove').textContent='';
  try{
    const a=await callWorkSpeakAI({mode:'mba',caseData:{category:c.cat,difficulty:c.diff,title:c.title,context:c.context,facts:c.facts,prompt:c.prompt,challenge:c.challenge},answer:t});
    $('mbaOverall').textContent=Math.round(a.score);
    $('rEnglish').textContent=Math.round(a.english);
    $('rStructure').textContent=Math.round(a.structure);
    $('rJudgment').textContent=Math.round(a.judgment);
    $('rExec').textContent=Math.round(a.executive_presence);
    $('mbaStrong').textContent=a.valid?listText(a.strengths,'특별히 잘한 점으로 평가할 부분이 없습니다.'):'평가 가능한 장점이 없습니다.';
    $('mbaImprove').textContent=listText(a.weaknesses,a.explanation_ko||a.verdict);
    if(a.explanation_ko)$('mbaImprove').textContent+=(a.weaknesses?.length?'\n\n':'')+a.explanation_ko;
    $('mbaChallenge').textContent=a.challenge_question||c.challenge;
    $('mbaModel').textContent=a.executive_version||c.model;
    $('modelWrap').style.display='none';
    if(a.valid){state.mba.push({date:now(),score:a.score,english:a.english,structure:a.structure,judgment:a.judgment,exec:a.executive_presence,text:t,case:c.title,ai:true,verdict:a.verdict});state.lastDate=now();save('workspeak_v2',state);renderHome()}
    setTimeout(()=>$('mbaFeedbackCard').scrollIntoView({behavior:'smooth',block:'start'}),100);
  }catch(e){$('mbaOverall').textContent='—';$('mbaStrong').textContent='AI 분석을 완료하지 못했습니다.';$('mbaImprove').textContent=e.message;alert(e.message)}finally{aiBusy(btn,false)}
};

checkDrill=async function(){
  const d=drills[drillIdx];const t=cleanText($('drillSpeech').textContent);
  if(!t)return alert('먼저 영어로 말하거나 직접 입력해보세요.');
  const btn=findButton('checkDrill');aiBusy(btn,true);
  $('drillMine').textContent=t;$('drillNatural').textContent='AI가 분석 중...';$('drillRef').textContent=d[2];$('drillPhrase').textContent=d[3];
  try{
    const a=await callWorkSpeakAI({mode:'drill',prompt:`Korean target meaning: ${d[1]}\nReference English: ${d[2]}`,answer:t});
    $('drillNatural').textContent=a.natural_version||a.executive_version||'—';
    $('drillWhy').textContent=`${a.valid?'':'의미가 충분히 맞지 않습니다. '}${a.explanation_ko||a.verdict}${a.weaknesses?.length?'\n'+listText(a.weaknesses):''}`;
    state.drills.unshift({date:now(),index:drillIdx,cat:d[0],prompt:d[1],answer:t,natural:a.natural_version||'',reference:d[2],phrase:d[3],score:a.score,valid:a.valid,ai:true,feedback:a.explanation_ko||a.verdict});
    state.lastDate=now();save('workspeak_v2',state);renderDrillHistory();renderHome();
  }catch(e){$('drillNatural').textContent='—';$('drillWhy').textContent=e.message;alert(e.message)}finally{aiBusy(btn,false)}
};
