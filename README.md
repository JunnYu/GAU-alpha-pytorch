# GAU-alpha-pytorch
pytorch版本https://github.com/ZhuiyiTechnology/GAU-alpha

## News
- 2022/05/11 感谢苏神提醒，添加了一个注释，其中RoFormerV2<sup>*</sup>表示未经多任务学习的RoFormerV2模型。
- 2022/04/22 初始化仓库，添加初步的代码, 添加`paddle`版本`gau_alpha`代码。
- 2022/04/30 添加CLUE分类任务代码。

## Install
```bash
pip install git+https://github.com/JunnYu/GAU-alpha-pytorch.git
or
pip install gau_alpha
```

## 精度对齐
```bash
python compare.py
# bert4keras vs pytorch
# mean diff : tensor(6.9320e-07)
# max diff : tensor(3.9101e-05)
```

## torch版本使用
### 依赖：
- torch
- transformers

```python
import torch
from gau_alpha import GAUAlphaForMaskedLM, GAUAlphaTokenizer

text = "今天[MASK]很好，我[MASK]去公园玩。"
tokenizer = GAUAlphaTokenizer.from_pretrained(
    "junnyu/chinese_GAU-alpha-char_L-24_H-768"
)
pt_model = GAUAlphaForMaskedLM.from_pretrained(
    "junnyu/chinese_GAU-alpha-char_L-24_H-768"
)
pt_model.eval()
pt_inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    pt_outputs = pt_model(**pt_inputs).logits[0]
pt_outputs_sentence = "pytorch: "
for i, id in enumerate(tokenizer.encode(text)):
    if id == tokenizer.mask_token_id:
        val, idx = pt_outputs[i].softmax(-1).topk(k=5)
        tokens = tokenizer.convert_ids_to_tokens(idx)
        new_tokens = []
        for v, t in zip(val.cpu(), tokens):
            new_tokens.append(f"{t}+{round(v.item(),4)}")
        pt_outputs_sentence += "[" + "||".join(new_tokens) + "]"
    else:
        pt_outputs_sentence += "".join(
            tokenizer.convert_ids_to_tokens([id], skip_special_tokens=True)
        )
print(pt_outputs_sentence)
# pytorch: 今天[天+0.8657||气+0.0535||阳+0.0165||，+0.0126||晴+0.0111]很好，我[要+0.4619||想+0.4352||又+0.0252||就+0.0157||跑+0.0064]去公园玩。
```

## Paddle版本使用
### 依赖：
- paddlepaddle>=2.2.0
- paddlenlp

```python
import paddle
from transformers import BertTokenizer as GAUAlphaTokenizer
from gau_alpha_paddle import GAUAlphaForMaskedLM

text = "今天[MASK]很好，我[MASK]去公园玩。"
tokenizer = GAUAlphaTokenizer.from_pretrained(
    "junnyu/chinese_GAU-alpha-char_L-24_H-768"
)
pd_model = GAUAlphaForMaskedLM.from_pretrained("chinese_GAU-alpha-char_L-24_H-768")
pd_model.eval()
pd_inputs = tokenizer(text)
pd_inputs = {k: paddle.to_tensor([v]) for k, v in pd_inputs.items()}

with paddle.no_grad():
    pd_outputs = pd_model(**pd_inputs)[0][0]

pd_outputs_sentence = "paddle: "
for i, id in enumerate(tokenizer.encode(text)):
    if id == tokenizer.mask_token_id:
        val, idx = paddle.nn.functional.softmax(pd_outputs[i], -1).topk(k=5)
        tokens = tokenizer.convert_ids_to_tokens(idx)
        new_tokens = []
        for v, t in zip(val.cpu(), tokens):
            new_tokens.append(f"{t}+{round(v.item(),4)}")
        pd_outputs_sentence += "[" + "||".join(new_tokens) + "]"
    else:
        pd_outputs_sentence += "".join(
            tokenizer.convert_ids_to_tokens([id], skip_special_tokens=True)
        )
print(pd_outputs_sentence)
# paddle: 今天[天+0.8657||气+0.0535||阳+0.0165||，+0.0126||晴+0.0111]很好，我[要+0.4619||想+0.4352||又+0.0252||就+0.0157||跑+0.0064]去公园玩。
```

## 介绍

- GAU-α：https://kexue.fm/archives/9052
- GAU：https://kexue.fm/archives/8934
- 原始论文：https://arxiv.org/abs/2202.10447

## 评测对比
### CLUE-dev榜单分类任务结果，base版本。

|         | iflytek | tnews | afqmc | cmnli | ocnli | wsc | csl |
| :-----: | :-----: | :---: | :---: | :---: | :---: | :---: | :---: |
| BERT | 60.06 | 56.80 | 72.41 | 79.56 | 73.93 | 78.62 | 83.93 |
| RoBERTa | 60.64 | 58.06 | 74.05 | **81.24** | 76.00 | 87.50 | 84.50 |
| RoFormer | 60.91 | 57.54 | 73.52 | 80.92 | 76.07 | 86.84 | 84.63 |
| RoFormerV2<sup>*</sup> | 60.87 | 56.54 | 72.75 | 80.34 | 75.36 | 80.92 | 84.67 |
| GAU-α | 61.41 | 57.76 | 74.17 | 81.82 | 75.86 | 79.93 | 85.67 |
| RoFormerV2-pytorch| **62.87** | **59.03** | **76.20** | 80.85 | **79.73** |   **87.82**   | **91.87** |
| GAU-α-pytorch（Adafactor） | 61.18 | 57.52 | 73.42 | 80.91 | 75.69 | 80.59 | 85.5 |
| GAU-α-pytorch（AdamW wd0.01 warmup0.1） | 60.68 | 57.95 | 73.08 | 81.02 | 75.36 | 81.25 | 83.93 |

### CLUE-test榜单分类任务结果，base版本。

|         | iflytek | tnews | afqmc | cmnli | ocnli | wsc | csl |
| :-----: | :-----: | :---: | :---: | :---: | :---: | :---: | :---: |
| RoFormerV2-pytorch | **63.15** | **58.24** | **75.42** | **80.59** | **74.17** |   **83.79**   | 83.73 |
| GAU-α-pytorch（Adafactor） | 61.38 | 57.08 | 74.05 | 80.37 | 73.53 | 74.83 | **85.6** |
| GAU-α-pytorch（AdamW wd0.01 warmup0.1） | 60.54 | 57.67 | 72.44 | 80.32 | 72.97 | 76.55 | 84.13 |


### CLUE-dev集榜单阅读理解和NER结果

|         | cmrc2018 | c3 | chid | cluener |
| :-----: | :-----: | :---: | :---: | :---: |
| BERT | 56.17 | 60.54 | 85.69 | 79.45 |
| RoBERTa | 56.54 | 67.66 | 86.71 | 79.47 |
| RoFormer | 56.26 | 67.24 | 86.57 | 79.72 |
| RoFormerV2<sup>*</sup> | 57.91 | 64.62 | 85.09 | **81.08** |
| GAU-α | **58.09** | **68.24** | **87.91** | 80.01 |

### 注：
- 其中RoFormerV2<sup>*</sup>表示的是未进行多任务学习的RoFormerV2模型，该模型苏神并未开源，感谢苏神的提醒。
- 其中不带有pytorch后缀结果都是从[GAU-alpha](https://github.com/ZhuiyiTechnology/GAU-alpha)仓库复制过来的。
- 其中带有pytorch后缀的结果都是自己训练得出的。
## 引用
Bibtex：

```tex
@techreport{gau-alpha,
  title={GAU-α: GAU-based Transformers for NLP - ZhuiyiAI},
  author={Jianlin Su, Shengfeng Pan, Bo Wen, Yunfeng Liu},
  year={2022},
  url="https://github.com/ZhuiyiTechnology/GAU-alpha",
}
```

## Tips:
- 感谢苏神提供的模型和代码！