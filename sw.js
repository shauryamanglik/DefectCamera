const CACHE='defectcam-v1';
const CORE=['./','./index.html','./app.js','./manifest.json'];
self.addEventListener('install',e=>{e.waitUntil(caches.open(CACHE).then(c=>c.addAll(CORE)).then(()=>self.skipWaiting()));});
self.addEventListener('activate',e=>{e.waitUntil(self.clients.claim());});
self.addEventListener('fetch',e=>{
  const u=e.request.url;
  // cache-first for our files + CDN model libs (so it works offline after first run)
  if(CORE.some(f=>u.endsWith(f.replace('./','')))||u.includes('cdn.jsdelivr.net')||u.includes('tfhub')||u.includes('storage.googleapis.com')){
    e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request).then(resp=>{
      const clone=resp.clone(); caches.open(CACHE).then(c=>c.put(e.request,clone)); return resp;
    }).catch(()=>r)));
  }
});
