# Stage3 Condition Vector Validity Report

This report summarizes the current internal validity checks for the frozen Stage3 condition vector. It validates trajectory-informed operational stress proxies against realized Stage1 measurements; it does not claim AV crash risk or ADS failure probability.

## Core / extended overall split

The target definition is now split into:

- `core_overall_high_stress = LCS_tail OR PMIS_tail OR RTS_tail`
- `extended_overall_high_stress = core_overall_high_stress OR IIS_tail`

Three-fold mean results:

| model | label | slice | AUC | AP | Lift@Top10 |
|---|---|---|---:|---:|---:|
| Core | core_overall | all | 0.7872 | 0.5422 | 2.8039 |
| Core+IIS+dropout | core_overall | all | 0.7880 | 0.5430 | 2.7780 |
| Core | extended_overall | all | 0.7846 | 0.5823 | 2.6008 |
| Core+IIS+dropout | extended_overall | all | 0.7869 | 0.5894 | 2.6090 |
| Core | extended_overall | high IIS applicability | 0.7877 | 0.5712 | 2.6113 |
| Core+IIS+dropout | extended_overall | high IIS applicability | 0.7926 | 0.5788 | 2.6727 |

Interpretation: IIS provides its clearest incremental signal for extended-overall and high-IIS-applicability orders. Its gain for the core overall target is very small, so IIS remains an optional intersection-specific auxiliary branch rather than a mandatory modality for every Stage3 head.

## Dimension validity

Three-fold mean internal validity:

| dimension | Spearman(pred, realized) | Pearson(pred, realized) | Top10 lift |
|---|---:|---:|---:|
| LCS | 0.5888 | 0.5098 | 3.5372 |
| PMIS | 0.5761 | 0.5348 | 3.4889 |
| RTS | 0.5483 | 0.5062 | 3.4252 |
| IIS | 0.4544 | 0.4225 | 2.6160 |

The three Core dimensions show meaningful but not identical signals. IIS is weaker but still interpretable in intersection-relevant contexts.

## Case index

Typical high/low/high-confidence/low-confidence cases are exported to:

`stage3/output/condition_vector_cases/condition_vector_case_index.csv`

The cases include route length, link count, predicted stress, realized pressure, POI/IIS exposure and uncertainty.

## Output files

- `stage3/output/core_extended_ablation/core_extended_ablation_summary.csv`
- `stage3/output/condition_vector_validity/condition_vector_correlations.csv`
- `stage3/output/condition_vector_validity/condition_vector_decile_validity.csv`
- `stage3/output/condition_vector_cases/condition_vector_case_index.csv`
