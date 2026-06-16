import json

with open('/c/workbench/Kronos/outputs/models/ma60_predictor_lb60_pd5_samp50000/eval_results.jsonl') as f:
    results = [json.loads(l) for l in f if l.strip()]

features = ['open', 'high', 'low', 'close', 'vol', 'amt']
steps = [1, 2, 3, 4, 5]

print('=' * 100)
print('Multi-step (predict=5, 50k samples) -- Per-Feature Per-Step IC Across Epochs 1-3')
print('=' * 100)

for step in steps:
    sfx = '_step%d' % step
    print('\n--- Step +%d ---' % step)
    header = '%8s' % 'F'
    for e in range(1, 4):
        header += '  %7s_%d' % ('IC', e)
        header += ' %7s_%d' % ('RIC', e)
        header += ' %7s_%d' % ('DA', e)
    print(header)
    print('-' * 80)
    for f in features:
        line = '%8s' % f
        for r in results:
            ic_key = '%s_ic%s' % (f, sfx)
            ric_key = '%s_rank_ic%s' % (f, sfx)
            da_key = '%s_da%s' % (f, sfx)
            line += ' %8.4f' % r.get(ic_key, float('nan'))
            line += ' %8.4f' % r.get(ric_key, float('nan'))
            line += ' %8.4f' % r.get(da_key, float('nan'))
        print(line)

print()
print('=' * 100)
print('Best IC per feature across all steps/epochs:')
print('=' * 100)
for f in features:
    best_ic = -99
    best_step = None
    best_epoch = None
    for step in steps:
        sfx = '_step%d' % step
        ic_key = '%s_ic%s' % (f, sfx)
        for ei, r in enumerate(results):
            v = r.get(ic_key, float('nan'))
            if v == v and v > best_ic:
                best_ic = v
                best_step = step
                best_epoch = ei + 1
    print('  %s: best IC=%.4f at step +%d, epoch %d' % (f, best_ic, best_step, best_epoch))

print()
print('Eval loss trend: %.4f -> %.4f -> %.4f' % (
    results[0]['eval_loss'], results[1]['eval_loss'], results[2]['eval_loss']))
