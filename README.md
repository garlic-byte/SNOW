## Partial visualization results from the libero dataset
### task 1：KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_ac8
<video src="outputs/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_ac8.mp4" width="720" controls></video>

### task 2：KITCHEN_SCENE8_put_both_moka_pots_on_the_stove
<video src="outputs/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_ac8.mp4" width="720" controls></video>

### task 3：LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
<video src="outputs/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_ac8.mp4" width="720" controls></video>

### task LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate
<video src="outputs/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_ac8.mp4" width="720" controls></video>

Here's a sample video:

### 任务演示：将黄白色马克杯放入微波炉并关闭门
<video src="https://github.com/garlic-byte/SNOW/raw/main/outputs/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_ac8.mp4" width="720" controls="controls"></video>


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

#### use drift method dataset: libero, joint nums: 7, step: 30,000, loss: 0.03

#### simulator result: 0.0, code must be revise.

#### add action-horzion to 16, dataset: libero, joint nums: 7, step: 30,000, loss: 0.06

### 2026.02.17

#### use qwen25-vl-3b method dataset: libero, joint nums: 7, step: 30,000, loss: 0.09

### 2026.02.23

#### use qwen3-vl-2b method，remake flow_matching_head, dataset: libero, joint nums: 7, step: 30,000, loss: 0.07


### 2026.02.24

#### use qwen3-vl-2b method，remake flow_matching_head, input_embedding_dim: 2048, dataset: libero, step: 30,000, loss: 0.06

#### use qwen3-vl-2b method，remake flow_matching_head, hidden_size: 2048, dataset: libero, step: 30,000, loss: 0.06

### 2026.02.27

#### use qwen3-vl-2b method，remake qwen3, dataset: libero, step: 30,000, loss: 0.05

#### use qwen3-vl-2b method，remake qwen3, select_layer: 24, dataset: libero, step: 30,000, loss: 0.06

### 2026.03.02

#### use qwen3-vl-2b method，pre-train qwen3, constant lr:0.04, dataset: libero, step: 30,000, loss: 0.05

#### use qwen3-vl-2b method，post-train qwen3, dataset: libero, step: 30,000, loss: 0.03

| TASK           | result            |
|----------------|-------------------|
| 10 (Long)      | 182/200 (91%)  |