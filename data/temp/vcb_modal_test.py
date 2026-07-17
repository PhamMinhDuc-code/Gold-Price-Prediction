import urllib.request, ssl, re, json

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# The VCB composite modal returned a JSON string inside double quotes - parse that
url = 'https://www.vietcombank.com.vn/api/composites/modal?v={E9998ED4-6853-4C3F-BE7E-1230060BA001}&itemid={AF386341-2740-4B05-B366-ABFE5F134C10}'
req = urllib.request.Request(url, headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'Accept': 'application/json,text/html,*/*;q=0.8',
    'Referer': 'https://www.vietcombank.com.vn/ca-nhan/tiet-kiem',
    'X-Requested-With': 'XMLHttpRequest',
})
resp = urllib.request.urlopen(req, context=ctx, timeout=12)
raw = resp.read().decode('utf-8', errors='replace')
print("Length:", len(raw))

# Look for double-encoded JSON (JSON inside JSON string)
m = re.search(r'"\\\\n', raw)
if m:
    print("Double-encoded JSON detected (escaped embedded JSON string)")
    try:
        outer = json.loads(raw)
        inner = json.loads(outer)
        print("Inner keys:", list(inner)[:5] if isinstance(inner, dict) else "list of len " + str(len(inner)))
        s = str(inner)
        print("Inner content (600 chars):", s[:600])
    except Exception as e2:
        print("Failed to decode inner:", e2)
        # Try just parsing outer
        try:
            o = json.loads(raw)
            print("Outer keys:", list(o)[:5])
            print("Outer raw[:500]:", repr(o)[:500])
        except Exception as e1:
            print("Failed to decode outer:", e1)
            print("Raw first 300:", raw[:300])
else:
    print("First 200 chars:", raw[:200])
    try:
        j = json.loads(raw)
        print("Keys:", list(j)[:5])
    except:
        pass
