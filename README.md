# Hotel Customer Segmentation — Clustering Project

Unsupervised segmentation of a European hotel's customer base (83,590 customers, 31 features). Three algorithm families covered in the ML course are compared under CRISP-DM: K-Means (partitioning), Agglomerative Hierarchical, and DBSCAN (density-based).

---

## Problem

Hotel H currently segments its customer base by sales origin — which channel booked the customer. That view is coarse: two customers arriving through the same channel can have completely different revenue, tenure and behaviour patterns. The new marketing manager wants a data-driven segmentation to define target groups, tailor pricing and promotions per group, and allocate marketing budget across channels based on the value each channel actually delivers.

This project delivers that segmentation using unsupervised clustering on three years of customer behaviour data.

## Dataset

- 83,590 customers
- 31 raw features across demographics, behaviour, revenue, and room preferences
- No target — this is unsupervised learning
- ~4.5% of the rows have a missing `Age`

## Result

The final segmentation uses **K-Means on PCA-reduced features** at K = 5. Each cluster is characterised by aggregate means of the interpretable features (age, revenue, cancellations, tenure) and named as a business persona for the marketing team.

Three algorithm families were compared on the same 5,000-customer sample:

| Method | Family | Clusters | Silhouette | Notes |
|---|---|---|---|---|
| **K-Means (K=5)** | Partitioning | 5 | **0.595** | Balanced sizes (31% / 27% / 14% / 14% / 14%), one persona per cluster |
| Agglomerative (Ward) | Hierarchical | 5 | 0.579 | Very close to K-Means — stability signal across method families |
| DBSCAN (eps=0.5, min_samples=20) | Density-based | 15 | 0.428 | 0.014% noise, one dominant pocket (~57% of customers) |

For a marketing team that needs the same customer's cluster refreshed every week, K-Means is the operational choice: it scales to the full 83k customers, produces balanced groups, and each centroid maps to a defensible persona.

### The five personas

- **Cluster 2 — Loyal high-value (27%).** Top on revenue, room-nights and check-ins; oldest account tenure. Their last stay was ~2 years ago — a lapsed premium segment worth retaining.
- **Cluster 4 — Reliable-but-cancellers (14%).** Oldest age (~51). The only cluster with `BookingsCanceled` > 0 — riskier for non-refundable offers.
- **Cluster 0 — New, low-engagement (31%).** Newest tenure, lowest check-in rate, lowest revenue. The onboarding target.
- **Cluster 1 — Long-lead planners (14%).** `AverageLeadTime` of 100 days — book far ahead. Perfect for early-bird promotions.
- **Cluster 3 — Middle-of-road active (14%).** Mid-tenure, short lead time, moderate metrics. The steady regulars.

## Why these three algorithms

The set matches the three main families of clustering methods taught in the course. Grid-based methods (STING, CLIQUE) were introduced in class but not implemented — this project follows that scope.

- Partitioning (K-Means) — hard clusters, spherical shape, distance to centroid.
- Hierarchical (Agglomerative with Ward linkage) — bottom-up merging, dendrogram as a visual audit of the group structure.
- Density-based (DBSCAN) — clusters as dense regions, arbitrary shape, built-in noise detection.

Comparing across families is how a portfolio project shows understanding of the whole clustering paradigm, not just one algorithm.

## What surprised me during the analysis

Multicollinearity in this dataset is very high. `DaysSinceLastStay` ↔ `DaysSinceFirstStay` correlate at **1.00** (for a single-stay customer they *are* the same value); `PersonsNights` ↔ `RoomNights` at **0.85**; and the tenure signals (`DaysSinceCreation` ↔ both stay-recency features) at **0.91**. PCA is the right tool: **5 components** cover 91% of the variance and turn the 23-dimensional space into a clean substrate for clustering.

Individual room-preference flags (`SR*` columns) are almost all zero — a large fraction of customers express no preference at all. They add dimensions without adding signal, so they were dropped. Aggregating them into a single "number of preferences expressed" flag would be a natural next iteration.

DBSCAN behaved better than expected here — 15 clusters found with only 12 noise points (0.014%). The reason is the one-hot encoded `MarketSegment` and `DistributionChannel` columns: each combination of segment × channel produces its own dense pocket in the PCA space, and DBSCAN follows that geometry. The largest DBSCAN cluster holds 57% of customers (the "Other" market segment). Fifteen sub-groups is too many for marketing to act on directly, but it exposes that the coarse K-Means groups can be split further if a specific campaign needs it.

Agglomerative clustering on the full 83k rows requires a full distance matrix (memory grows as N²) and is infeasible on a laptop. Running it on a stratified 5,000-customer sample and comparing the result to K-Means is standard practice and preserves the methodological comparison the class asks for.

## In production

If this segmentation were productised for the hotel:

1. Ingestion. Nightly refresh from the reservations warehouse, joined with the customer master.
2. Scoring. The persisted scaler → PCA → K-Means pipeline scores every active customer.
3. Action. Marketing receives a table of `customer_id → cluster`. Cluster-specific promotions, email templates and pricing rules are pre-approved per cluster.
4. Monitoring. Track (a) cluster sizes week over week (large drift = customer behaviour shifting), (b) revenue per cluster (which segment is worth investing more in), (c) fraction of new customers landing in each cluster (which segments are growing).
5. Retraining. Full retrain of the pipeline quarterly, or on a large drift alert. Adding a new marketing programme (e.g. a new loyalty tier) is a reason to retrain.

## Scope and limitations

One hotel. Generalisation to the group's other properties has not been validated.

`Nationality` was dropped because of high cardinality. A follow-up should reintroduce it as region groupings (Iberian, Central Europe, Rest of World, etc.) — nationality is likely a strong signal this pass leaves on the table.

Sparse room-preference flags (`SR*`) were dropped individually. Aggregating them into an "expressed preferences count" would recover some signal.

Agglomerative and DBSCAN ran on a stratified 5,000-customer sample due to memory constraints on the full 83k rows. Cluster stability across bootstrapped samples was not measured.

The persona names are illustrative. In a real deployment, the analyst provides the numeric characterisation and the marketing team names the personas in the language their organisation already uses.

## Methodology

The work follows CRISP-DM:

1. Business Understanding. Marketing use case and problem definition.
2. Data Understanding. Categorical exploration (MarketSegment × DistributionChannel), numeric distributions, correlation heatmap, missing value analysis.
3. Data Preparation. Drop identifiers and redundant/sparse features, impute `Age` with median, cap `Age` to [18, 90], one-hot encode categoricals, MinMax scale.
4. Dimensionality reduction. PCA with enough components to retain ~90% variance.
5. Modeling. Three families: K-Means (K chosen by elbow + silhouette), Agglomerative (dendrogram across four linkages, Ward chosen), DBSCAN (eps from k-distance plot).
6. Evaluation. Silhouette, cluster size balance, business interpretability. K-Means chosen for deployment.
7. Deployment. Segment labels exported. Scaler, PCA and K-Means pipeline persisted with `joblib`.

## Stack

Python 3.10+, pandas, numpy, scikit-learn (KMeans, AgglomerativeClustering, DBSCAN, PCA, MinMaxScaler, silhouette_score, NearestNeighbors), scipy (linkage, dendrogram, fcluster), matplotlib, seaborn, joblib.

---

Author: Camilla Alves.
