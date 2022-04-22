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
