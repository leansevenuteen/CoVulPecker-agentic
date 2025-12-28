import json
import os
from collections import Counter

# Try to find results file
results_file = None
for f in ['full_results.json', 'detection_results.json']:
    if os.path.exists(f):
        results_file = f
        break

if not results_file:
    print("No results file found")
    exit(1)

print(f"Analyzing {results_file}...")
with open(results_file, 'r') as f:
    data = json.load(f)

successful = [r for r in data if r.get('status') == 'success']
print(f'\nTotal successful: {len(successful)}')

pre_preds = [r.get('pre_patch_prediction', {}).get('predicted_label') for r in successful]
post_preds = [r.get('post_patch_prediction', {}).get('predicted_label') for r in successful]

print('\nPre-patch predictions:', Counter(pre_preds))
print('Post-patch predictions:', Counter(post_preds))

pv_samples = [r for r in successful if r.get('pairwise_category') == 'P-V']
print(f'\nP-V samples: {len(pv_samples)}/{len(successful)} ({len(pv_samples)/len(successful)*100:.1f}%)')

# Check true labels for P-V samples
pv_true_labels = [r.get('true_label') for r in pv_samples]
print(f'P-V true labels: {Counter(pv_true_labels)}')

# Sample P-V case
if pv_samples:
    r = pv_samples[0]
    print('\nSample P-V case:')
    print(f'  Pre: {r.get("pre_patch_prediction", {}).get("predicted_label")}')
    print(f'  Post: {r.get("post_patch_prediction", {}).get("predicted_label")}')
    print(f'  True label: {r.get("true_label")}')

