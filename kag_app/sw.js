const CACHE = "rag-kag-v1";
const SHELL = ["/", "/static/icon-512.jpg"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys =>
    Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
  ));
  self.clients.claim();
});

self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  // Ne pas mettre en cache les appels API
  if (["/upload","/rag","/kag","/compare","/smart","/stats","/health"].some(p => url.pathname.startsWith(p))) return;
  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request))
  );
});
