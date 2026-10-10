# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**: Louis de Perier
- **Student ID**: 20159
- **Email**: ldeperier.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

The Ames Housing dataset (De Cock, 2011), covering residential property sales in
Ames, Iowa between 2006 and 2010. One row is a single completed sale: the
assessor's record of the property joined to the transaction price and date.
[N] rows and [K] raw columns, committed to this repo as `data/ames.csv`
([size] MB), so `main.py train` retrains from scratch with no network access.

I picked it because the feature work is genuinely non-trivial — it mixes ordinal
quality ratings, high-cardinality neighborhood labels, and "missing means the
house doesn't have one" nulls — and because the sales span the 2008 housing
downturn, which forces a real decision about how to split rather than a reflexive
random shuffle.

## Business / real-life framing

*The hypothetical scenario this model serves, and what that scenario implies
for how you built the pipeline — target definition, whether a time-based
split was necessary and why (or why not), which metric should drive the
decision threshold and why.*

**The decision.** A small buy-renovate-resell investor operating in Ames makes
same-day cash offers on homes before any inspection, working only from public
assessor records. The model estimates the open-market sale price of a property;
the offer is that estimate minus a margin. The cost of error is asymmetric:
overpaying destroys margin on a deal already committed to, while underpaying only
loses a deal that someone else takes.

**What that implies for the pipeline:**

*Target definition needs care.* The target is `SalePrice`, but not every row is
the quantity the business cares about. Sales recorded as `Abnorml` (foreclosure,
short sale) or `Family` are not arm's-length market prices, and `Partial` sales
are incomplete new construction. I [kept / excluded] these because [...]. Leaving
them in would train the model to predict a price the investor will never actually
face when reselling.

I model `log(SalePrice)` rather than raw dollars, because pricing error is
naturally proportional — being $20k off on a $400k home is a very different
business outcome than on a $90k home — and because the raw target is strongly
right-skewed. All reported errors are converted back to dollars so they remain
readable as business quantities. [Note how you handled retransformation bias, or
state that you ignored it and why.]

*A random split would leak the future into training.* The data spans 2006–2010
and straddles the crash, and the model is deployed to price sales that have not
happened yet. A random shuffle lets 2010 comparables inform predictions about
2008, inflating test scores in a way that would not survive deployment. I split
on `YrSold`/`MoSold`: train on [2006–2008], validate on [2009], test on [2010].
[If you also ran a random split as a contrast, report both numbers here — the gap
is strong evidence the decision mattered.]

*Which metric sets the decision rule.* This is a regression problem, so there is
no classification threshold; the equivalent knob is the offer margin applied to
the prediction. I report **MAE in dollars** as the headline metric because it is
directly interpretable as average mispricing per home and is robust to the
handful of luxury outliers, alongside **RMSE** (which penalizes the large misses
that are most expensive) and the **share of homes priced within ±10%**, which is
what determines how wide a safety margin the investor must hold.


## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
