import csv

rows = list(csv.DictReader(open('debug_per_frame_yolo/v_gQNyhv8y0QY_c013_stats.csv')))

print('\nFrames with >=4 confident kp but H failed sanity:')
print(f'{"frame":>6}  {"kp":>3}  {"ransac_in":>9}  {"reproj":>7}  fail_reason')
count = 0
for r in rows:
    if r['fail_reason'] == 'H failed sanity check':
        print(f"{r['frame_id']:>6}  {r['n_confident']:>3}  {r['n_ransac_inliers']:>9}  {r['mean_reproj_px']:>7}")
        count += 1
        if count >= 20:
            break

print(f'\n--- Distribution of kp counts in failures vs successes ---')
from collections import Counter
fail_kp   = Counter(int(r['n_confident']) for r in rows if r['fail_reason'] == 'H failed sanity check')
ok_kp     = Counter(int(r['n_confident']) for r in rows if r['sane'] == '1')
print('kp_count  failed  ok')
for k in sorted(set(fail_kp) | set(ok_kp)):
    print(f'  {k:>3}     {fail_kp.get(k, 0):>4}    {ok_kp.get(k, 0):>4}')
