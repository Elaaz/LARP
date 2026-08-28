import re

paths = [r'f:/LARP/LARP/log_gcn.txt', r'f:/LARP/LARP/log_func.txt', r'f:/LARP/LARP/log_head.txt', r'f:/LARP/LARP/log_query.txt']
encs = ['utf-8','utf-16','utf-16le','latin1']
pat = re.compile(r"(test_seen|test_unseen_objects|test_unseen_configs|GAT\(obs\)|symbolic\(obs\)|symbolic\(oracle\)|realistic gap|val F1=|saved model)", re.I)

for p in paths:
    print('\n----', p, '----')
    try:
        b = open(p,'rb').read()
    except Exception as e:
        print('ERROR reading', p, e)
        continue
    s = None
    for e in encs:
        try:
            s = b.decode(e)
            break
        except Exception:
            s = None
    if s is None:
        print('failed to decode')
        continue
    for ln in s.splitlines():
        if pat.search(ln):
            print(ln)