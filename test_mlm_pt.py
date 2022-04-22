import torch

from gau_alpha import GAUAlphaForMaskedLM, GAUAlphaTokenizer

text = "今天[MASK]很好，我[MASK]去公园玩。"
tokenizer = GAUAlphaTokenizer.from_pretrained(
    "junnyu/chinese_GAU-alpha-char_L-24_H-768"
)
pt_model = GAUAlphaForMaskedLM.from_pretrained(
    "junnyu/chinese_GAU-alpha-char_L-24_H-768"
)

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
