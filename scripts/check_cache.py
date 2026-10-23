import json

m = json.load(
    open(
        r'data\tactical\h6\value_cache_v3\manifest.json',
        encoding='utf-8'
    )
)

print('labels_sha256 ', m['labels_sha256'])
print(
    'h6_report     ',
    m['h6_report_sha256'],
    'restricted',
    m['h6_report_restricted']
)

total = 0

for split, s in m['splits'].items():
    print(split, {k: v for k, v in s.items() if k != 'sha256'})
    total += s['states']

print('total states   ', total)