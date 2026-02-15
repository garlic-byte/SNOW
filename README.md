## LOG
### 2026.01.30

#### dataset: libero_10, joint nums: 7, step: 30,000, loss: 0.04 

#### conclusion: action from dataset is noisy, loss is higher.

### 2026.01.31

#### dataset: accad, joint nums: 30, step: 30,000, loss: 0.02 (revised dataset, )

#### dataset: accad, joint nums: 7, step: 30,000, loss: 0.01

#### conclusion: less joints, less loss. (normal)

### 2026.02.09

#### dataset: libero, joint nums: 7, step: 30,000, loss: 0.05

#### conclusion: more datasets, more accurate in simulator


| TASK           | result            |
|----------------|-------------------|
| 10 (Long)      | 173/200 (86.50%)  |
| Goal           | 200/200 (100%)    |
| Object         | 200/200 (100%)    |
| Spatial        | 195/200 (97.65%)  |


### 2026.02.14

#### dataset: libero_10, joint nums: 7, step: 30,000, loss: 0.03

#### conclusion: pre-train + post-train, less loss.


| TASK           | result            |
|----------------|-------------------|
| 10 (Long)      | 177/200 (88.50%)  |

### 2026.02.15

#### use drift method 

#### dataset: libero, joint nums: 7, step: 30,000, loss: 0.03

#### simulator result


