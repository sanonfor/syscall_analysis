# Sequence model interpretability

## Fournier models

First, compute system-call level anomaly scores by running `do_eval_steps.py`.

Then run `eval_steps_fournier.ipynb` to visualize the interpretability results.


## Kim models

Similarly to Fournier models, first compute system-call level anomaly scores by running `do_eval_steps.py`.

Then run `eval_steps_fournier.ipynb` to visualize the interpretability results.


## Catak models

### Deepaid

Evaluate with Deepaid by running `do_deepaid.py`.
By default, the results will be written to `catak/deepaid-results/<timestamp>/summary.json`.

### LIME

Evaluate with LIME by running `do_lime.py`.
By default, the results will be written to `catak/lime-results/<timestamp>/results_log.txt`.