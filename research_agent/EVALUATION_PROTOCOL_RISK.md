# Final Evaluation Protocol Risk

## Confirmed local facts

- The current local experiment contract uses `long_view` as the positive label.
- The current local evaluator uses `GAUC` and `nDCG@5`.

## Unconfirmed risk

The final organizer evaluation protocol may differ from the current local protocol and may use:

- positive label: `is_click`;
- metrics: `NDCG@10` and `Recall@50`.

Until the organizer confirms the final protocol, this possibility must not be described as an
established official requirement.

## Potential impact

If final evaluation uses `is_click + NDCG@10/Recall@50`, scores, model comparisons, and
optimization directions derived from `long_view + GAUC/nDCG@5` will not directly represent
final evaluation performance. The team would need to reassess:

1. label construction and the training objective;
2. leakage-sensitive and forbidden fields;
3. train and validation splits;
4. ranking, user grouping, and metric computation;
5. baseline reproduction and comparability with later experiments.

## Handling policy

- Mark all current results as local `long_view` experiment results until the protocol is final.
- Do not claim that the current metrics are the final official metrics.
- Keep the label and evaluator replaceable instead of hard-coding them into future experiment
  implementations.
- After the organizer publishes the final protocol, decide whether to migrate to
  `is_click + NDCG@10/Recall@50` and reproduce the corresponding baseline.
