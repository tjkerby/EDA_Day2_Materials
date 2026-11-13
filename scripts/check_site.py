"""Check local page, download and asset links under the rendered docs folder."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit

ROOT=Path(__file__).resolve().parents[1]
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,t,a):
        a=dict(a)
        self.links.extend(a[k] for k in ['src','href'] if k in a)

missing=[]
for path in (ROOT/'docs').rglob('*.html'):
    p=Links();p.feed(path.read_text())
    for link in p.links:
        url=urlsplit(link)
        if url.scheme or url.netloc or not url.path:continue
        target=path.parent/unquote(url.path)
        if not target.exists():missing.append((str(path.relative_to(ROOT)),link))
if missing:raise SystemExit('Missing local links: '+repr(missing))
print('All local website page, download, and asset links resolve')
