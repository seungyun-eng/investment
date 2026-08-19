(function(){
  const configs={
    speech:'영어로 직접 입력하거나 마이크로 말해보세요.',
    mbaSpeech:'MBA 케이스 답변을 영어로 직접 입력하거나 말해보세요.',
    drillSpeech:'영어 답변을 직접 입력하거나 말해보세요.'
  };

  function makeEditable(id,placeholder){
    const el=document.getElementById(id);
    if(!el)return;
    if(el.tagName==='TEXTAREA'){
      el.placeholder=placeholder;
      el.rows=6;
      el.spellcheck=true;
      el.style.width='100%';
      return;
    }
    const ta=document.createElement('textarea');
    ta.id=id;
    ta.className=el.className;
    ta.rows=6;
    ta.spellcheck=true;
    ta.placeholder=placeholder;
    const current=(el.textContent||'').trim();
    if(current && !/표시됩니다|말해보세요/.test(current)) ta.value=current;
    el.replaceWith(ta);
  }

  function apply(){
    Object.entries(configs).forEach(([id,p])=>makeEditable(id,p));
    document.querySelectorAll('.micWrap').forEach(w=>{
      const title=w.querySelector('b');
      const tip=w.querySelector('.tip');
      if(title) title.textContent='말하거나 직접 입력하세요';
      if(tip) tip.textContent='키보드 입력과 마이크를 함께 사용할 수 있습니다.';
    });
  }

  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',apply);
  else apply();

  const obs=new MutationObserver(()=>apply());
  obs.observe(document.documentElement,{childList:true,subtree:true});
})();