import json
import glob
import os

files = sorted(glob.glob('evaluation_reports/summary_*.json'))

columns = [
    'timestamp', 'query_count', 'top_k', 'avg_gt_cnt',
    'single_gt_ratio', 'p@1', 'h@3',
    'h@k', 'ndcg@k', 'faithfulness', 'ans_corr', 'ragas_cov'
]

def format_val(v):
    if v is None: return "N/A"
    if isinstance(v, float): return f"{v:.3f}"
    return str(v)

print(" | ".join(columns))
print("-" * 140)

best_ac = -1.0
best_ac_ts = "N/A"
best_ndcg = -1.0
best_ndcg_ts = "N/A"

rows = []
for f in files:
    try:
        with open(f, 'r') as jf:
            data = json.load(jf)
            run_config = data.get('run_config', {})
            summary = data.get('summary', {})
            metrics = summary.get('metrics_avg', {})
            q_diag = run_config.get('query_set_diagnostics', {})
            ragas_cov = summary.get('ragas_coverage', {})
            
            ts = run_config.get('timestamp', os.path.basename(f).replace('summary_', '').replace('.json', ''))
            # Simplify timestamp for display
            display_ts = ts.split('T')[0] + "_" + ts.split('T')[1][:8] if 'T' in ts else ts
            
            # Calculating ragas coverage as a string/float for display
            f_cov = ragas_cov.get('faithfulness_coverage', 1.0)
            a_cov = ragas_cov.get('answer_correctness_coverage', 1.0)
            avg_cov = (f_cov + a_cov) / 2.0
            
            row = [
                display_ts,
                summary.get('query_count'),
                summary.get('top_k'),
                q_diag.get('avg_ground_truth_count'),
                q_diag.get('single_ground_truth_ratio'),
                metrics.get('precision_at_1'),
                metrics.get('hit_at_3'),
                metrics.get('hit_at_k'),
                metrics.get('ndcg_at_k'),
                metrics.get('faithfulness'),
                metrics.get('answer_correctness'),
                avg_cov
            ]
            rows.append(row)
            
            ac = metrics.get('answer_correctness')
            if ac is not None and ac > best_ac:
                best_ac = ac
                best_ac_ts = display_ts
                
            ndcg = metrics.get('ndcg_at_k')
            if ndcg is not None and ndcg > best_ndcg:
                best_ndcg = ndcg
                best_ndcg_ts = display_ts
    except Exception as e:
        # print(f"Error processing {f}: {e}")
        pass

rows.sort(key=lambda x: x[0])

for r in rows:
    print(" | ".join(format_val(x) for x in r))

print("\nBest by answer_correctness:")
print(f"Timestamp: {best_ac_ts}, Score: {format_val(best_ac)}")

print("\nBest by ndcg_at_k:")
print(f"Timestamp: {best_ndcg_ts}, Score: {format_val(best_ndcg)}")
