# Verified Evidence Manifest

Every reported numerical value, re-derived from repository
artifacts on 2026-07-24. Base model everywhere: Qwen2.5-1.5B-Instruct (policy and
shared-pool verifiers); LoRA r=16, beta=0.1, lr 5e-6, 1 epoch; training seed 42
(single seed; see statistical note). Downstream eval: 500 held-out HH test pairs +
500 refusal prompts (250 unsafe / 250 benign), greedy decoding.

| Metric | Value | Source artifact | Script | Seed | Dataset | Split | Regime | 95% CI |
|---|---|---|---|---|---|---|---|---|
| pref_acc raw eta=0 | 0.548 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF | test 500 pairs | structured_unsafe |  |
| unsafe_refusal raw eta=0 | 0.428 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| retention raw eta=0 | 1.0 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| harmful_survival raw eta=0 | 0.0 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| pref_acc noise_aware eta=0 | 0.55 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF | test 500 pairs | structured_unsafe |  |
| unsafe_refusal noise_aware eta=0 | 0.412 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| pref_acc single eta=0 | 0.558 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF | test 500 pairs | structured_unsafe |  |
| unsafe_refusal single eta=0 | 0.416 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| retention single eta=0 | 0.599 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| harmful_survival single eta=0 | 0.0 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| pref_acc consensus_k3 eta=0 | 0.55 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF | test 500 pairs | structured_unsafe |  |
| unsafe_refusal consensus_k3 eta=0 | 0.408 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| retention consensus_k3 eta=0 | 0.4134 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| harmful_survival consensus_k3 eta=0 | 0.0 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| pref_acc oracle eta=0 | 0.546 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF | test 500 pairs | structured_unsafe |  |
| unsafe_refusal oracle eta=0 | 0.404 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| retention oracle eta=0 | 1.0 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| harmful_survival oracle eta=0 | 0.0 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| pref_acc raw eta=10 | 0.552 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF | test 500 pairs | structured_unsafe |  |
| unsafe_refusal raw eta=10 | 0.404 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| retention raw eta=10 | 1.0 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| harmful_survival raw eta=10 | 0.0986 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| pref_acc noise_aware eta=10 | 0.544 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF | test 500 pairs | structured_unsafe |  |
| unsafe_refusal noise_aware eta=10 | 0.416 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| pref_acc single eta=10 | 0.548 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF | test 500 pairs | structured_unsafe |  |
| unsafe_refusal single eta=10 | 0.424 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| retention single eta=10 | 0.5944 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| harmful_survival single eta=10 | 0.0908479138627187 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| pref_acc consensus_k3 eta=10 | 0.546 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF | test 500 pairs | structured_unsafe |  |
| unsafe_refusal consensus_k3 eta=10 | 0.432 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| retention consensus_k3 eta=10 | 0.408 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| harmful_survival consensus_k3 eta=10 | 0.08333333333333333 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| pref_acc oracle eta=10 | 0.548 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF | test 500 pairs | structured_unsafe |  |
| unsafe_refusal oracle eta=10 | 0.412 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| retention oracle eta=10 | 0.9014 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| harmful_survival oracle eta=10 | 0.0 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| pref_acc raw eta=20 | 0.548 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF | test 500 pairs | structured_unsafe |  |
| unsafe_refusal raw eta=20 | 0.404 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| retention raw eta=20 | 1.0 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| harmful_survival raw eta=20 | 0.2031 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| pref_acc noise_aware eta=20 | 0.554 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF | test 500 pairs | structured_unsafe |  |
| unsafe_refusal noise_aware eta=20 | 0.416 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| pref_acc single eta=20 | 0.544 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF | test 500 pairs | structured_unsafe |  |
| unsafe_refusal single eta=20 | 0.412 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| retention single eta=20 | 0.5905 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| harmful_survival single eta=20 | 0.1913632514817951 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| pref_acc consensus_k3 eta=20 | 0.552 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF | test 500 pairs | structured_unsafe |  |
| unsafe_refusal consensus_k3 eta=20 | 0.42 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| retention consensus_k3 eta=20 | 0.402 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| harmful_survival consensus_k3 eta=20 | 0.1808457711442786 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| pref_acc oracle eta=20 | 0.55 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF | test 500 pairs | structured_unsafe |  |
| unsafe_refusal oracle eta=20 | 0.424 | results/tables/main_results.csv | run_pilot_hh.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| retention oracle eta=20 | 0.7969 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| harmful_survival oracle eta=20 | 0.0 | results/tables/main_results.csv | consensus_filter_precomputed.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| pref_acc rdpo eta=0 | 0.542 | results/eval/structured_unsafe_eta0_rdpo.json | run_p3_robust_baselines.sh | 42 | HH-RLHF 10K train | test 500 pairs | structured_unsafe |  |
| unsafe_refusal rdpo eta=0 | 0.416 | results/eval/structured_unsafe_eta0_rdpo.json | run_p3_robust_baselines.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| pref_acc ipo eta=0 | 0.54 | results/eval/structured_unsafe_eta0_ipo.json | run_p3_robust_baselines.sh | 42 | HH-RLHF 10K train | test 500 pairs | structured_unsafe |  |
| unsafe_refusal ipo eta=0 | 0.416 | results/eval/structured_unsafe_eta0_ipo.json | run_p3_robust_baselines.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| pref_acc rdpo eta=10 | 0.548 | results/eval/structured_unsafe_eta10_rdpo.json | run_p3_robust_baselines.sh | 42 | HH-RLHF 10K train | test 500 pairs | structured_unsafe |  |
| unsafe_refusal rdpo eta=10 | 0.412 | results/eval/structured_unsafe_eta10_rdpo.json | run_p3_robust_baselines.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| pref_acc ipo eta=10 | 0.552 | results/eval/structured_unsafe_eta10_ipo.json | run_p3_robust_baselines.sh | 42 | HH-RLHF 10K train | test 500 pairs | structured_unsafe |  |
| unsafe_refusal ipo eta=10 | 0.408 | results/eval/structured_unsafe_eta10_ipo.json | run_p3_robust_baselines.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| pref_acc rdpo eta=20 | 0.55 | results/eval/structured_unsafe_eta20_rdpo.json | run_p3_robust_baselines.sh | 42 | HH-RLHF 10K train | test 500 pairs | structured_unsafe |  |
| unsafe_refusal rdpo eta=20 | 0.396 | results/eval/structured_unsafe_eta20_rdpo.json | run_p3_robust_baselines.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| pref_acc ipo eta=20 | 0.55 | results/eval/structured_unsafe_eta20_ipo.json | run_p3_robust_baselines.sh | 42 | HH-RLHF 10K train | test 500 pairs | structured_unsafe |  |
| unsafe_refusal ipo eta=20 | 0.424 | results/eval/structured_unsafe_eta20_ipo.json | run_p3_robust_baselines.sh | 42 | HH-RLHF 10K train | refusal eval 250+250 | structured_unsafe |  |
| paired-bootstrap raw-consensus harm gap eta=10 | 0.0152 | analysis/outputs/e1_paired_bootstrap.json | inline (e1) | 42 | HH-RLHF 10K train | train | structured_unsafe | [0.0084, 0.0222] |
| paired-bootstrap raw-consensus harm gap eta=20 | 0.0224 | analysis/outputs/e1_paired_bootstrap.json | inline (e1) | 42 | HH-RLHF 10K train | train | structured_unsafe | [0.0129, 0.0317] |
| rho_bar shared full-set eta=10 | 0.6822 | analysis/outputs/e2_correlation.json | e2_correlation.py | 42 | HH-RLHF 10K train | train | structured_unsafe | [0.6727, 0.6919] |
| n_eff shared full-set eta=10 | 1.313 | analysis/outputs/e2_correlation.json | e2_correlation.py | 42 | HH-RLHF 10K train | train | structured_unsafe | [1.3, 1.325] |
| rho_bar shared full-set eta=20 | 0.6683 | analysis/outputs/e2_correlation.json | e2_correlation.py | 42 | HH-RLHF 10K train | train | structured_unsafe | [0.658, 0.6786] |
| n_eff shared full-set eta=20 | 1.331 | analysis/outputs/e2_correlation.json | e2_correlation.py | 42 | HH-RLHF 10K train | train | structured_unsafe | [1.318, 1.345] |
| alpha_subset harm (retention-matched) eta=20 | 0.1809 | analysis/outputs/e2_correlation.json | e2_correlation.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| rho_bar shared subsample eta=10 | 0.6678 | analysis/outputs/p2_diverse_pool.json | p2_diverse_analysis.py | 42 | HH-RLHF 3,000-pair subsample (seed 42) | train | structured_unsafe | [0.6515, 0.6846] |
| n_eff shared subsample eta=10 | 1.332 | analysis/outputs/p2_diverse_pool.json | p2_diverse_analysis.py | 42 | HH-RLHF 3,000-pair subsample (seed 42) | train | structured_unsafe | [1.31, 1.354] |
| retention k3 shared eta=10 | 0.4087 | analysis/outputs/p2_diverse_pool.json | p2_diverse_analysis.py | 42 | 3,000-pair subsample | train | structured_unsafe |  |
| harmful_survival k3 shared eta=10 | 0.0775 | analysis/outputs/p2_diverse_pool.json | p2_diverse_analysis.py | 42 | 3,000-pair subsample | train | structured_unsafe |  |
| rho_bar diverse subsample eta=10 | 0.2665 | analysis/outputs/p2_diverse_pool.json | p2_diverse_analysis.py | 42 | HH-RLHF 3,000-pair subsample (seed 42) | train | structured_unsafe | [0.2464, 0.2862] |
| n_eff diverse subsample eta=10 | 2.223 | analysis/outputs/p2_diverse_pool.json | p2_diverse_analysis.py | 42 | HH-RLHF 3,000-pair subsample (seed 42) | train | structured_unsafe | [2.152, 2.3] |
| retention k3 diverse eta=10 | 0.5607 | analysis/outputs/p2_diverse_pool.json | p2_diverse_analysis.py | 42 | 3,000-pair subsample | train | structured_unsafe |  |
| harmful_survival k3 diverse eta=10 | 0.0826 | analysis/outputs/p2_diverse_pool.json | p2_diverse_analysis.py | 42 | 3,000-pair subsample | train | structured_unsafe |  |
| rho_bar shared subsample eta=20 | 0.6518 | analysis/outputs/p2_diverse_pool.json | p2_diverse_analysis.py | 42 | HH-RLHF 3,000-pair subsample (seed 42) | train | structured_unsafe | [0.6346, 0.6695] |
| n_eff shared subsample eta=20 | 1.353 | analysis/outputs/p2_diverse_pool.json | p2_diverse_analysis.py | 42 | HH-RLHF 3,000-pair subsample (seed 42) | train | structured_unsafe | [1.33, 1.377] |
| retention k3 shared eta=20 | 0.4017 | analysis/outputs/p2_diverse_pool.json | p2_diverse_analysis.py | 42 | 3,000-pair subsample | train | structured_unsafe |  |
| harmful_survival k3 shared eta=20 | 0.1809 | analysis/outputs/p2_diverse_pool.json | p2_diverse_analysis.py | 42 | 3,000-pair subsample | train | structured_unsafe |  |
| rho_bar diverse subsample eta=20 | 0.2602 | analysis/outputs/p2_diverse_pool.json | p2_diverse_analysis.py | 42 | HH-RLHF 3,000-pair subsample (seed 42) | train | structured_unsafe | [0.2405, 0.2813] |
| n_eff diverse subsample eta=20 | 2.246 | analysis/outputs/p2_diverse_pool.json | p2_diverse_analysis.py | 42 | HH-RLHF 3,000-pair subsample (seed 42) | train | structured_unsafe | [2.169, 2.324] |
| retention k3 diverse eta=20 | 0.553 | analysis/outputs/p2_diverse_pool.json | p2_diverse_analysis.py | 42 | 3,000-pair subsample | train | structured_unsafe |  |
| harmful_survival k3 diverse eta=20 | 0.1826 | analysis/outputs/p2_diverse_pool.json | p2_diverse_analysis.py | 42 | 3,000-pair subsample | train | structured_unsafe |  |
| margin-rule shared_eta10_k2 retention | 0.3745 | analysis/outputs/p2_margin_rule.json | inline (p2 margin) | 42 | full 10K (shared) / 3,000 subsample (diverse) | train | structured_unsafe |  |
| margin-rule shared_eta10_k2 harmful_survival | 0.0761 | analysis/outputs/p2_margin_rule.json | inline (p2 margin) | 42 | full 10K (shared) / 3,000 subsample (diverse) | train | structured_unsafe |  |
| margin-rule shared_eta10_k3 retention | 0.3203 | analysis/outputs/p2_margin_rule.json | inline (p2 margin) | 42 | full 10K (shared) / 3,000 subsample (diverse) | train | structured_unsafe |  |
| margin-rule shared_eta10_k3 harmful_survival | 0.074 | analysis/outputs/p2_margin_rule.json | inline (p2 margin) | 42 | full 10K (shared) / 3,000 subsample (diverse) | train | structured_unsafe |  |
| margin-rule shared_eta20_k2 retention | 0.3634 | analysis/outputs/p2_margin_rule.json | inline (p2 margin) | 42 | full 10K (shared) / 3,000 subsample (diverse) | train | structured_unsafe |  |
| margin-rule shared_eta20_k2 harmful_survival | 0.1607 | analysis/outputs/p2_margin_rule.json | inline (p2 margin) | 42 | full 10K (shared) / 3,000 subsample (diverse) | train | structured_unsafe |  |
| margin-rule shared_eta20_k3 retention | 0.3107 | analysis/outputs/p2_margin_rule.json | inline (p2 margin) | 42 | full 10K (shared) / 3,000 subsample (diverse) | train | structured_unsafe |  |
| margin-rule shared_eta20_k3 harmful_survival | 0.1587 | analysis/outputs/p2_margin_rule.json | inline (p2 margin) | 42 | full 10K (shared) / 3,000 subsample (diverse) | train | structured_unsafe |  |
| margin-rule diverse_eta10_k2 retention | 0.5827 | analysis/outputs/p2_margin_rule.json | inline (p2 margin) | 42 | full 10K (shared) / 3,000 subsample (diverse) | train | structured_unsafe |  |
| margin-rule diverse_eta10_k2 harmful_survival | 0.0721 | analysis/outputs/p2_margin_rule.json | inline (p2 margin) | 42 | full 10K (shared) / 3,000 subsample (diverse) | train | structured_unsafe |  |
| margin-rule diverse_eta10_k3 retention | 0.4523 | analysis/outputs/p2_margin_rule.json | inline (p2 margin) | 42 | full 10K (shared) / 3,000 subsample (diverse) | train | structured_unsafe |  |
| margin-rule diverse_eta10_k3 harmful_survival | 0.07 | analysis/outputs/p2_margin_rule.json | inline (p2 margin) | 42 | full 10K (shared) / 3,000 subsample (diverse) | train | structured_unsafe |  |
| margin-rule diverse_eta20_k2 retention | 0.5647 | analysis/outputs/p2_margin_rule.json | inline (p2 margin) | 42 | full 10K (shared) / 3,000 subsample (diverse) | train | structured_unsafe |  |
| margin-rule diverse_eta20_k2 harmful_survival | 0.1511 | analysis/outputs/p2_margin_rule.json | inline (p2 margin) | 42 | full 10K (shared) / 3,000 subsample (diverse) | train | structured_unsafe |  |
| margin-rule diverse_eta20_k3 retention | 0.4367 | analysis/outputs/p2_margin_rule.json | inline (p2 margin) | 42 | full 10K (shared) / 3,000 subsample (diverse) | train | structured_unsafe |  |
| margin-rule diverse_eta20_k3 harmful_survival | 0.1519 | analysis/outputs/p2_margin_rule.json | inline (p2 margin) | 42 | full 10K (shared) / 3,000 subsample (diverse) | train | structured_unsafe |  |
| retention-matched diverse harm eta=10 | 0.0776 | analysis/outputs/p2_retention_matched.json | inline (p2 matched) | 42 | 3,000-pair subsample | train | structured_unsafe |  |
| retention-matched diverse harm eta=20 | 0.1799 | analysis/outputs/p2_retention_matched.json | inline (p2 matched) | 42 | 3,000-pair subsample | train | structured_unsafe |  |
| rejection frac_truly_corrupted eta=10 | 0.1091 | analysis/outputs/e4_rejection.json | e4_rejection.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| rejection frac_clean_pair_unusable eta=10 | 0.7782 | analysis/outputs/e4_rejection.json | e4_rejection.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| rejection frac_clean_direction_flip eta=10 | 0.1127 | analysis/outputs/e4_rejection.json | e4_rejection.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| rejection mean_chosen_score_clean_kept eta=10 | 0.9108 | analysis/outputs/e4_rejection.json | e4_rejection.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| rejection mean_chosen_score_clean_rejected eta=10 | 0.2129 | analysis/outputs/e4_rejection.json | e4_rejection.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| rejection frac_truly_corrupted eta=20 | 0.2181 | analysis/outputs/e4_rejection.json | e4_rejection.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| rejection frac_clean_pair_unusable eta=20 | 0.6841 | analysis/outputs/e4_rejection.json | e4_rejection.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| rejection frac_clean_direction_flip eta=20 | 0.0978 | analysis/outputs/e4_rejection.json | e4_rejection.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| rejection mean_chosen_score_clean_kept eta=20 | 0.9104 | analysis/outputs/e4_rejection.json | e4_rejection.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| rejection mean_chosen_score_clean_rejected eta=20 | 0.212 | analysis/outputs/e4_rejection.json | e4_rejection.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| coordinated total eta=10 | 10000 | analysis/outputs/e5_coordinated_counts.json | inline (phase7) | 42 | HH-RLHF 10K train | train | coordinated_poisoning |  |
| coordinated trigger eta=10 | 443 | analysis/outputs/e5_coordinated_counts.json | inline (phase7) | 42 | HH-RLHF 10K train | train | coordinated_poisoning |  |
| coordinated poisoned eta=10 | 44 | analysis/outputs/e5_coordinated_counts.json | inline (phase7) | 42 | HH-RLHF 10K train | train | coordinated_poisoning |  |
| coordinated detected eta=10 | 37 | analysis/outputs/e5_coordinated_counts.json | inline (phase7) | 42 | HH-RLHF 10K train | train | coordinated_poisoning |  |
| coordinated surviving eta=10 | 7 | analysis/outputs/e5_coordinated_counts.json | inline (phase7) | 42 | HH-RLHF 10K train | train | coordinated_poisoning |  |
| coordinated kept eta=10 | 4132 | analysis/outputs/e5_coordinated_counts.json | inline (phase7) | 42 | HH-RLHF 10K train | train | coordinated_poisoning |  |
| coordinated recall eta=10 | 0.8409 | analysis/outputs/e5_coordinated_counts.json | inline (phase7) | 42 | HH-RLHF 10K train | train | coordinated_poisoning |  |
| coordinated rel_reduction eta=10 | 0.615 | analysis/outputs/e5_coordinated_counts.json | inline (phase7) | 42 | HH-RLHF 10K train | train | coordinated_poisoning |  |
| coordinated total eta=20 | 10000 | analysis/outputs/e5_coordinated_counts.json | inline (phase7) | 42 | HH-RLHF 10K train | train | coordinated_poisoning |  |
| coordinated trigger eta=20 | 443 | analysis/outputs/e5_coordinated_counts.json | inline (phase7) | 42 | HH-RLHF 10K train | train | coordinated_poisoning |  |
| coordinated poisoned eta=20 | 82 | analysis/outputs/e5_coordinated_counts.json | inline (phase7) | 42 | HH-RLHF 10K train | train | coordinated_poisoning |  |
| coordinated detected eta=20 | 70 | analysis/outputs/e5_coordinated_counts.json | inline (phase7) | 42 | HH-RLHF 10K train | train | coordinated_poisoning |  |
| coordinated surviving eta=20 | 12 | analysis/outputs/e5_coordinated_counts.json | inline (phase7) | 42 | HH-RLHF 10K train | train | coordinated_poisoning |  |
| coordinated kept eta=20 | 4129 | analysis/outputs/e5_coordinated_counts.json | inline (phase7) | 42 | HH-RLHF 10K train | train | coordinated_poisoning |  |
| coordinated recall eta=20 | 0.8537 | analysis/outputs/e5_coordinated_counts.json | inline (phase7) | 42 | HH-RLHF 10K train | train | coordinated_poisoning |  |
| coordinated rel_reduction eta=20 | 0.6456 | analysis/outputs/e5_coordinated_counts.json | inline (phase7) | 42 | HH-RLHF 10K train | train | coordinated_poisoning |  |
| broad consensus recall eta=10 | 0.6552 | analysis/outputs/e5_regimes.json | e5_regimes.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
| broad consensus recall eta=20 | 0.642 | analysis/outputs/e5_regimes.json | e5_regimes.py | 42 | HH-RLHF 10K train | train | structured_unsafe |  |
