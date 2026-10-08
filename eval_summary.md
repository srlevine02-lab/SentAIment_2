# AccessIQ review agents: evaluation results

Model: `claude-sonnet-5-5` | test cases: 40 | runs per case: 1 | ranking batches: 6

| Scope | Metric | What it checks | Average | n |
|---|---|---|---|---|
| End-to-end | E1_sentiment_accuracy | Final report sentiment matches label | 0.900 | 40 |
| End-to-end | E2_outcome_correct | Right departments + reply iff needed, report matches actions | 1.000 | 40 |
| End-to-end | E3_reply_quality | LLM-judge reply quality (1-5 scaled to 0-1) | 0.966 | 29 |
| Trajectory | T1_tool_order_correct | Valid tool order, no actions on 4-5 star reviews | 1.000 | 40 |
| Trajectory | T2_step_efficiency | Expected calls / actual calls | 0.977 | 40 |
| Trajectory | T3_task_completion | Correct executed state after human approval | 1.000 | 40 |
| Component | C1_routing_arg_f1 | Department arguments to route_to_department (F1) | 1.000 | 40 |
| Component | C2_policy_arg_correct | issue_category argument to check_return_policy | 1.000 | 29 |
| Component | C3_offer_arg_correct | offer argument to post_customer_reply | 1.000 | 29 |
| Component | C4_analyst_accuracy | Sentiment Analyst alone: sentiment accuracy | 0.925 | 40 |
| Component | C5_evidence_grounded | Analyst evidence quotes found verbatim in review | 1.000 | 34 |
| End-to-end | R1_top1_correct | Priority report: #1 issue matches the label-based #1 | 0.833 | 6 |
| End-to-end | R2_top2_overlap | Priority report: overlap with the label-based top 2 | 1.000 | 6 |
| Trajectory | R3_ranking_trajectory | One rank_issues call over all IDs, no routing/reply calls | 1.000 | 6 |
| Component | R4_ranker_top2_overlap | Ranking step alone: overlap with the label-based top 2 | 1.000 | 6 |

- Sentiment macro F1 (end-to-end, 4 classes): **0.780**
- Case pass rate (E2, T1 and T3 all pass, slide 37): **1.000**

## Interpretation (auto-generated starting point; edit before submitting)

- Strongest metric: E2_outcome_correct (1.00); weakest: E1_sentiment_accuracy (0.90).
- Sentiment in the final report is 0.03 below the analyst run alone, showing how much the manager changes or loses the analyst's label.
- Most common sentiment confusions (label -> predicted): positive->neutral x2, neutral->negative x2.
- Lowest case-pass rates by review type: mixed 1.00 (n=10), negative 1.00 (n=16).
- The policy guardrail blocked 0 proposed repl(ies); each block is an agent mistake caught before it reached a customer.
- Priority ranking (6 batches of 20 reviews): the report's #1 issue matched the label-based #1 in 5 of 6 batches.
- The ranking step alone scores 0.00 above the full agent run, which shows whether errors come from the analyst or from how the manager reports them.
