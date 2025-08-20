## Email

![alt text](image.png)

- Falar que me concentrei em 3

## Ideias to insert

- Macro topic: Nodes degree distribution across communities
  - All communities follow a power low degrees distribution, with specially similar magnitudes for cummunity 0 and 2, and commniuty 1 with lower magninitudes. It can be noticed in the figure where I plot the 3 nodes distribution, one plot for each of the 3 communities.
  - There is another figure where communities nodes degress are sobrepostos in a log scale to easly identify the differeces among them, it is posible to see commnity 2 is the one with more vomlume along all the nodes

  - Top 5 high degree nodes for each commnity
    Community 0 - Top High-Degree Nodes (95th percentile, threshold=71.0):

------------------------------------------------------------

   1. Techstars-seed                           (degree: 582, type: Early-stage)
   2. 500 Global-seed                          (degree: 564, type: Early-stage)
   3. Gaingels-series_b                        (degree: 509, type: Late-stage)
   4. Greycroft-series_a                       (degree: 483, type: Early-stage)
   5. Bossa Invest-series_b                    (degree: 450, type: Late-stage)

Community 1 - Top High-Degree Nodes (95th percentile, threshold=41.0)
------------------------------------------------------------

   1. Intel Capital-series_b                   (degree: 240, type: Late-stage)
   2. Norwest Venture Partners-series_a        (degree: 222, type: Early-stage)
   3. Canaan Partners-series_a                 (degree: 219, type: Early-stage)
   4. Intel Capital-series_c                   (degree: 175, type: Late-stage)
   5. SOSV-series_b                            (degree: 163, type: Late-stage)

Community 2 - Top High-Degree Nodes (95th percentile, threshold=135.0)
------------------------------------------------------------

   1. SV Angel-seed                            (degree: 974, type: Early-stage)
   2. SV Angel-series_a                        (degree: 754, type: Early-stage)
   3. Andreessen Horowitz-series_a             (degree: 664, type: Early-stage)
   4. Khosla Ventures-series_a                 (degree: 659, type: Early-stage)
   5. New Enterprise Associates-series_a       (degree: 619, type: Early-stage)

    - We compare in a graph degree vs investment activity where we see sort of positive relationship between such variables, with higher degree nodes showing more activity

- Macro topic: Nestedness evolution on community 2
  - We try to understand along the time how this community has increased its nestedness
  - The analysis is cumulative, so the first year contain only info about the first year, but the second contain the first and the second, and so on. The last year contain info about all year.
  - The results are below, I wil plot graphs based on them: ""Computing nestedness evolution for Community 2 (21-year cumulative windows)...
    ======================================================================
    Year 2004: Error - Input matrix rows with only zeros, abort.
    Year 2005: Error - Input matrix rows with only zeros, abort.
    Year 2006: Error - Input matrix rows with only zeros, abort.
    Year 2007: Generating 100 null models...

    Year 2007: 137 nodes (94 left, 43 right), 789 edges, Connectance: 0.1952, Nestedness: 0.3761, Z-score: -1.6195, P-value: 0.9400 (not_significant)
    Year 2008: Generating 100 null models...

    Year 2008: 177 nodes (125 left, 52 right), 1057 edges, Connectance: 0.1626, Nestedness: 0.3597, Z-score: -1.3174, P-value: 0.9000 (not_significant)
    Year 2009: Generating 100 null models...

    Year 2009: 206 nodes (153 left, 53 right), 1233 edges, Connectance: 0.1521, Nestedness: 0.3521, Z-score: -0.3254, P-value: 0.6500 (not_significant)
    Year 2010: Generating 100 null models...

    Year 2010: 258 nodes (189 left, 69 right), 1883 edges, Connectance: 0.1444, Nestedness: 0.3519, Z-score: -0.8475, P-value: 0.8200 (not_significant)
    Year 2011: Generating 100 null models...

    Year 2011: 345 nodes (240 left, 105 right), 3147 edges, Connectance: 0.1249, Nestedness: 0.3325, Z-score: -2.1528, P-value: 0.9800 (not_significant)
    Year 2012: Generating 100 null models...

    Year 2012: 397 nodes (280 left, 117 right), 3718 edges, Connectance: 0.1135, Nestedness: 0.3117, Z-score: -2.5701, P-value: 0.9900 (not_significant)
    Year 2013: Generating 100 null models...

    Year 2013: 497 nodes (340 left, 157 right), 5278 edges, Connectance: 0.0989, Nestedness: 0.2758, Z-score: -3.6828, P-value: 1.0000 (not_significant)
    Year 2014: Generating 100 null models...

    Year 2014: 695 nodes (460 left, 235 right), 8467 edges, Connectance: 0.0783, Nestedness: 0.2350, Z-score: -2.6249, P-value: 0.9800 (not_significant)
    Year 2015: Generating 100 null models...

    Year 2015: 981 nodes (632 left, 349 right), 13439 edges, Connectance: 0.0609, Nestedness: 0.1983, Z-score: -1.2423, P-value: 0.9000 (not_significant)
    Year 2016: Generating 100 null models...

    Year 2016: 1250 nodes (778 left, 472 right), 18215 edges, Connectance: 0.0496, Nestedness: 0.1728, Z-score: -1.2377, P-value: 0.8700 (not_significant)
    Year 2017: Generating 100 null models...

    Year 2017: 1530 nodes (952 left, 578 right), 22608 edges, Connectance: 0.0411, Nestedness: 0.1507, Z-score: -0.8990, P-value: 0.8400 (not_significant)
    Year 2018: Generating 100 null models...

    Year 2018: 1920 nodes (1217 left, 703 right), 27893 edges, Connectance: 0.0326, Nestedness: 0.1306, Z-score: 0.4901, P-value: 0.3000 (not_significant)
    Year 2019: Generating 100 null models...
    Year 2019: Generating 100 null models...

    Year 2019: 2364 nodes (1523 left, 841 right), 33805 edges, Connectance: 0.0264, Nestedness: 0.1177, Z-score: 2.9125, P-value: 0.0000 (significant)
    Year 2020: Generating 100 null models...
    Year 2020: Generating 100 null models...

    Year 2020: 2789 nodes (1839 left, 950 right), 38612 edges, Connectance: 0.0221, Nestedness: 0.1065, Z-score: 3.5526, P-value: 0.0000 (significant)
    Year 2021: Generating 100 null models...
    Year 2021: Generating 100 null models...

    Year 2021: 3514 nodes (2455 left, 1059 right), 44387 edges, Connectance: 0.0171, Nestedness: 0.0949, Z-score: 4.4923, P-value: 0.0000 (significant)
    Year 2022: Generating 100 null models...
    Year 2022: Generating 100 null models...

    Year 2022: 3784 nodes (2689 left, 1095 right), 46199 edges, Connectance: 0.0157, Nestedness: 0.0906, Z-score: 4.7584, P-value: 0.0000 (significant)
    Year 2023: Generating 100 null models...
    Year 2023: Generating 100 null models...

    Year 2023: 3890 nodes (2784 left, 1106 right), 46646 edges, Connectance: 0.0151, Nestedness: 0.0896, Z-score: 4.2853, P-value: 0.0000 (significant)
    Year 2024: Generating 100 null models...
    Year 2024: Generating 100 null models...

    Year 2024: 3959 nodes (2844 left, 1115 right), 47031 edges, Connectance: 0.0148, Nestedness: 0.0882, Z-score: 4.3246, P-value: 0.0000 (significant)

    Successfully analyzed 18 years for Community 2
    ""
  - Statistical analysis below: ""Community 2 Nestedness Evolution Summary:
    ============================================================
    Years analyzed: 18
    Period: 2007 to 2024
    Significant periods (p < 0.05): 6

    Nestedness Statistics Over Time:
    Mean observed nestedness: 0.2130 ± 0.1106
    Mean null nestedness: 0.2165 ± 0.1162
    Mean Z-score: 0.3498 ± 2.8676
    Mean connectance: 0.0758 ± 0.0596

    Detailed Year-by-Year Results:
    year  num_pairs  num_nodes  connectance  observed_nestedness  z_score  \
    0   2007        246        137       0.1952               0.3761  -1.6195
    1   2008        327        177       0.1626               0.3597  -1.3174
    2   2009        409        206       0.1521               0.3521  -0.3254
    3   2010        574        258       0.1444               0.3519  -0.8475
    4   2011        916        345       0.1249               0.3325  -2.1528
    5   2012       1169        397       0.1135               0.3117  -2.5701
    6   2013       1700        497       0.0989               0.2758  -3.6828
    7   2014       2731        695       0.0783               0.2350  -2.6249
    8   2015       4751        981       0.0609               0.1983  -1.2423
    9   2016       7784       1250       0.0496               0.1728  -1.2377
    10  2017      11088       1530       0.0411               0.1507  -0.8990
    11  2018      16522       1920       0.0326               0.1306   0.4901
    12  2019      24228       2364       0.0264               0.1177   2.9125
    13  2020      31727       2789       0.0221               0.1065   3.5526
    14  2021      46340       3514       0.0171               0.0949   4.4923
    15  2022      52374       3784       0.0157               0.0906   4.7584
    16  2023      53812       3890       0.0151               0.0896   4.2853
    17  2024      55675       3959       0.0148               0.0882   4.3246

        p_value     significance  
    0      0.94  not_significant  
    1      0.90  not_significant  
    2      0.65  not_significant  
    3      0.82  not_significant  
    4      0.98  not_significant  
    5      0.99  not_significant  
    6      1.00  not_significant  
    7      0.98  not_significant  
    8      0.90  not_significant  
    9      0.87  not_significant  
    10     0.84  not_significant  
    11     0.30  not_significant  
    12     0.00      significant  
    13     0.00      significant  
    14     0.00      significant  
    15     0.00      significant  
    16     0.00      significant  
    17     0.00      significant  ""

  - We have 2 figures, each one with 3 plots
    - Figure 1:
      - 1. Nestedness evolution: red line representing "observed nestedness" and blue the null models mechanisms
      - 2. Statistical Significance (Z-socres), yellow line on Z-socre below and above (2 and -2)
      - 3. Network Size Evolution gree line as total nodes, purple being total edges
    - Figure 2:
            1. Connectance evolution (decreasing)
            2. VC Types Over Time (late stage in blue early stagve in red)
            3. P-values over time (log scale)

  - The detailes analysis of significant periods

  Detailed Analysis of Significant Periods for Community 2
    ======================================================================

    Year 2019 (Window: 1998-2019):
    Investment pairs: 24228
    Network: 1523 late-stage VCs, 841 early-stage VCs
    Edges: 33805, Connectance: 0.0264
    Observed nestedness: 0.1177
    Null mean ± std: 0.1148 ± 0.0010
    Z-score: 2.9125, P-value: 0.000000

    Year 2020 (Window: 1999-2020):
    Investment pairs: 31727
    Network: 1839 late-stage VCs, 950 early-stage VCs
    Edges: 38612, Connectance: 0.0221
    Observed nestedness: 0.1065
    Null mean ± std: 0.1031 ± 0.0009
    Z-score: 3.5526, P-value: 0.000000

    Year 2021 (Window: 2000-2021):
    Investment pairs: 46340
    Network: 2455 late-stage VCs, 1059 early-stage VCs
    Edges: 44387, Connectance: 0.0171
    Observed nestedness: 0.0949
    Null mean ± std: 0.0912 ± 0.0008
    Z-score: 4.4923, P-value: 0.000000

    Year 2022 (Window: 2001-2022):
    Investment pairs: 52374
    Network: 2689 late-stage VCs, 1095 early-stage VCs
    Edges: 46199, Connectance: 0.0157
    Observed nestedness: 0.0906
    Null mean ± std: 0.0869 ± 0.0008
    Z-score: 4.7584, P-value: 0.000000

    Year 2023 (Window: 2002-2023):
    Investment pairs: 53812
    Network: 2784 late-stage VCs, 1106 early-stage VCs
    Edges: 46646, Connectance: 0.0151
    Observed nestedness: 0.0896
    Null mean ± std: 0.0859 ± 0.0008
    Z-score: 4.2853, P-value: 0.000000

  - Such date ir ploted with Null Model distribution and network structure side by side
  - For coincidence or not ([explain it better using references]), the late stage are increasing during time and right side decreasing in terms of number of VCs
