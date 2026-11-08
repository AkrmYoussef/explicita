# Explicita

A small service that explains the decisions of a machine learning model.

The model decides if a bank should **approve** or **reject** a loan.
The service answers three questions for every decision:

1. **Why?** Which features pushed the decision up or down? (SHAP and LIME)
2. **What can I change?** What small change would flip the decision? (DiCE counterfactuals)
3. **Can I trust this?** Is the explanation faithful, stable, and do the methods agree?

The data is the public [German Credit dataset](https://www.openml.org/d/31) (1,000 loan applicants, 20 features).

## Quick start

You need Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
make install   # install the package with UI and test tools
make train     # download the data and train the model
make api       # start the API at http://localhost:8000
```

Open a second terminal for the UI:

```bash
source .venv/bin/activate
make ui        # open http://localhost:8501
```

Run the tests (they use synthetic data, so they need no internet):

```bash
make test
```

## API

Interactive docs: http://localhost:8000/docs

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Check that the service is running |
| GET | `/schema` | List every input feature, its type, and its allowed values |
| POST | `/predict` | Return the decision and the approval probability |
| POST | `/explain?method=shap` | Return feature contributions (`shap` or `lime`) |
| POST | `/counterfactuals?total=3` | Return small changes that flip the decision |
| POST | `/evaluate?k=3` | Run the three trust checks for this input |

Every POST takes the same body: `{"features": {...}}`, with one value per feature.
Get the feature names and default values from `GET /schema`.

## How the trust checks work

| Check | Question | How |
|---|---|---|
| Faithfulness | Does the explanation point to features the model really uses? | Replace the top-k features with "typical" values. The prediction should change more than when we replace k random features. |
| Stability | Does a tiny change in the input give a similar explanation? | Add small noise to the numbers 5 times. Measure the overlap of the top-k features (0 to 1). |
| Agreement | Do SHAP and LIME agree? | Overlap of their top-k features, plus rank correlation over all features. |

## Project structure

```
src/explicita/
    data.py            load the dataset, list immutable features
    model.py           the model pipeline and the ModelBundle class
    train.py           train and save the model
    explain.py         SHAP and LIME
    counterfactual.py  DiCE counterfactuals
    evaluate.py        faithfulness, stability, agreement
    api.py             FastAPI app
ui/app.py              Streamlit front end
tests/                 pytest tests on synthetic data
```

## Design notes

- **Ordinal encoding, not one-hot.** Each original feature stays one column, so every
  explanation maps to a feature a person understands.
- **Immutable features.** Counterfactuals never change `age`, `personal_status`,
  `foreign_worker`, or `num_dependents`. A useful suggestion must be something the
  person can actually change.
- **Repeatable results.** LIME and DiCE use fixed random seeds, so the same input
  gives the same explanation.
- **Fairness warning.** The dataset has sensitive features (age, sex inside
  `personal_status`, foreign worker). This project is for learning. Do not use the
  model for real decisions.

## Roadmap

- [x] Week 1: train the model
- [x] Week 2: API with SHAP and LIME
- [x] Week 3: counterfactuals with DiCE
- [x] Week 4: Streamlit UI
- [x] Week 5: trust checks and tests
- [x] Week 6: Docker
- [ ] Let users upload their own model and data
- [ ] Explain a text classifier with Captum (Integrated Gradients)
- [ ] Write a plain-language summary of each explanation with an LLM
- [ ] Cache explanations for repeated inputs
- [ ] Run the trust checks over the whole test set, not only one input

## Docker

```bash
make docker   # build and run the API on port 8000
```
