(() => {
  const ENDPOINT =
    'https://script.google.com/macros/s/AKfycbyBFKjH7XmNCU8d6pU35v4Lw4xbBt2vPrbAWfHB3tmNlFydf1zzwvq6dZWWBTUzLqGzGw/exec';

  const variants = ['hip', 'shoulder', 'torso', 'procrustes'];
  const dataFiles = {
    hip:         'data/hip_norm_eval.json',
    shoulder:    'data/shoulder_norm_eval.json',
    torso:       'data/torso_norm_eval.json',
    procrustes:  'data/procrustes_norm_eval.json'
  };

  const RESP_KEY = 'similarity_responses';

  function toClipName(raw) {
    return raw
      .split('/').pop()
      .replace(/(\.csv|\.mp4)$/, '')
      .replace(/_timed$/, '') + '.mp4';
  }

  function sendResponse(record) {
    fetch(ENDPOINT, {
      method: 'POST',
      mode:   'no-cors',
      headers:{ 'Content-Type':'application/json' },
      body:    JSON.stringify(record)
    }).catch(e=>console.error('sendResponse failed', e));
  }

  const variantData = {};
  let queryClips = [];

  async function loadAllData() {
    const allQs = new Set();
    await Promise.all(variants.map(async v => {
      const resp = await fetch(dataFiles[v]);
      if (!resp.ok) throw new Error(`Failed to load ${v}: ${resp.status}`);
      const json = await resp.json();
      variantData[v] = json;
      Object.keys(json).forEach(k => allQs.add(k));
    }));
    queryClips = Array.from(allQs);
    console.log('Total distinct queries:', queryClips.length);
  }

  function saveLocally(rec) {
    const arr = JSON.parse(localStorage.getItem(RESP_KEY) || '[]');
    arr.push(rec);
    localStorage.setItem(RESP_KEY, JSON.stringify(arr));
  }

  window.addEventListener('load', async () => {
    const qVid    = document.getElementById('queryVideo');
    const nVid    = document.getElementById('neighbourVideo');
    const yesBtn   = document.getElementById('yesBtn');
    const maybeBtn = document.getElementById('maybeBtn');
    const noBtn    = document.getElementById('noBtn');

    function setButtons(enabled) {
      yesBtn.disabled = maybeBtn.disabled = noBtn.disabled = !enabled;
    }

    let curQ, curVariant, curN;

    function respond(ans) {
      const rec = {
        coach:     window.COACH_ID || 'UNKNOWN',  // ← pull fresh each time
        query:     toClipName(curQ),
        neighbour: toClipName(curN),
        variant:   curVariant,
        response:  ans,
        timestamp: new Date().toISOString()
      };
      sendResponse(rec);
      saveLocally(rec);
      console.log('Recorded & sent:', rec);
      nextPair();
    }

    function nextPair() {
      setButtons(false);

      curQ       = queryClips[Math.floor(Math.random() * queryClips.length)];
      curVariant = variants[Math.floor(Math.random() * variants.length)];
      const neigh = (variantData[curVariant][curQ] || [])[0];
      if (!neigh) return nextPair();
      curN = neigh;

      const qSrc = 'clips/' + toClipName(curQ);
      const nSrc = 'clips/' + toClipName(curN);

      let loadedCount = 0;
      function onLoad() {
        if (++loadedCount === 2) setButtons(true);
      }
      function onError(evt) {
        console.error('Video failed, skipping:', evt.target.src);
        qVid.onerror = nVid.onerror = null;
        nextPair();
      }

      qVid.onloadeddata = onLoad;
      nVid.onloadeddata = onLoad;
      qVid.onerror      = onError;
      nVid.onerror      = onError;

      qVid.src = qSrc; qVid.load();
      nVid.src = nSrc; nVid.load();
    }

    yesBtn.addEventListener('click',   () => respond('yes'));
    maybeBtn.addEventListener('click', () => respond('maybe'));
    noBtn.addEventListener('click',    () => respond('no'));

    setButtons(false);
    await loadAllData();
    nextPair();
  });
})();
