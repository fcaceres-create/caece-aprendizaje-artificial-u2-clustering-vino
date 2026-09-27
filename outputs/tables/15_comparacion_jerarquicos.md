| modelo                          | linkage   | métrica   |   silueta (euclídea) |   silueta (métrica propia) | tamaños      |   ARI vs. K-Means |
|:--------------------------------|:----------|:----------|---------------------:|---------------------------:|:-------------|------------------:|
| K-Means (k-means++)             | —         | euclidean |                0.285 |                      0.285 | [65, 62, 51] |             1     |
| Jerárquico ward (euclidean)     | ward      | euclidean |                0.277 |                      0.277 | [64, 58, 56] |             0.853 |
| Jerárquico complete (euclidean) | complete  | euclidean |                0.204 |                      0.204 | [69, 58, 51] |             0.596 |
| Jerárquico average (euclidean)  | average   | euclidean |                0.158 |                      0.158 | [174, 3, 1]  |            -0.002 |
| Jerárquico single (euclidean)   | single    | euclidean |                0.183 |                      0.183 | [174, 3, 1]  |            -0.003 |
| Jerárquico average (manhattan)  | average   | manhattan |                0.252 |                      0.277 | [126, 51, 1] |             0.487 |
| Jerárquico complete (manhattan) | complete  | manhattan |                0.19  |                      0.202 | [97, 52, 29] |             0.518 |
