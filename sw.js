/* Jam phase 1: offline cache. After the first visit the page opens and works with no network at all. */
const VERSION = '20261109151919-a9e11ac5'; // stamped per publish by publish-jam.sh (a new version = a fresh cache on the next start)
const C = 'jam-p1-' + VERSION, FILES = ['./', 'index.html', 'style.css', 'app.js', 'analyzer.js', 'worker.js', 'capture-worklet.js', 'inst-model.js', 'icon.svg', 'manifest.webmanifest'];
self.addEventListener('install', (e) => { e.waitUntil(caches.open(C).then((c) => c.addAll(FILES)).then(() => self.skipWaiting())); });
self.addEventListener('activate', (e) => { e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== C).map((k) => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener('fetch', (e) => { if (e.request.method !== 'GET') return; e.respondWith(caches.match(e.request, { ignoreSearch: true }).then((r) => r || fetch(e.request))); });
