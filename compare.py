import tensorflow as tf
import torch
from bert4keras.models import *

from gau_alpha import GAUAlphaModel, GAUAlphaTokenizer


# copied from https://github.com/ZhuiyiTechnology/GAU-alpha/blob/main/models.py
class GAU_alpha(RoFormerV2):
    """GAU-α
    改动：基本模块换成GAU
    链接：https://kexue.fm/archives/9052
    """

    def initializer(self, shape, dtype=None, order=3, gain=1.0):
        return super(GAU_alpha, self).initializer(shape, dtype, order, gain)

    def apply_main_layers(self, inputs, index):
        """GAU-α 的主体是基于Gated Attention Unit的模块
        顺序：GAU  --> Add --> LN
        """
        x = inputs

        attention_name = "Transformer-%d-GatedAttentionUnit" % index
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)

        # Self Attention
        xi = x
        x = [x, position_bias]
        arguments = {"a_bias": None, "p_bias": "rotary"}
        if attention_mask is not None:
            arguments["a_bias"] = True
            x.insert(1, attention_mask)
        x = self.apply(
            inputs=x,
            layer=GatedAttentionUnit,
            arguments=arguments,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=False,
            normalization="softmax_plus",
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=attention_name,
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name="%s-Dropout" % attention_name,
        )
        x = self.apply(inputs=[xi, x], layer=Add, name="%s-Add" % attention_name)
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name="%s-Norm" % attention_name,
        )

        return x

    def variable_mapping(self):
        """重新定义权重映射"""
        mapping = {
            "Embedding-Token": ["bert/embeddings/word_embeddings"],
            "Embedding-Segment": ["bert/embeddings/token_type_embeddings"],
        }

        for i in range(self.num_hidden_layers):
            prefix = "GAU_alpha/encoder/layer_%d/" % i
            mapping["Transformer-%d-GatedAttentionUnit" % i] = [
                prefix + "gau/i_dense/kernel",
                # prefix + 'gau/i_dense/bias',
                prefix + "gau/o_dense/kernel",
                # prefix + 'gau/o_dense/bias',
                # prefix + 'gau/q_scaleoffset/beta',
                prefix + "gau/q_scaleoffset/gamma",
                # prefix + 'gau/k_scaleoffset/beta',
                prefix + "gau/k_scaleoffset/gamma",
            ]
        return mapping

# huggingface.co上的
converted_ckpt_path = "junnyu/chinese_GAU-alpha-char_L-24_H-768"
# 从苏神提供的链接下载的
config_path = "chinese_GAU-alpha-char_L-24_H-768/bert_config.json"
checkpoint_path = "chinese_GAU-alpha-char_L-24_H-768/bert_model.ckpt"

text = [
    "人生的路丰富而漫长,就像在宇宙中穿梭。",
    "时而太空漫步,无拘无束;时而速度失控,遭遇窘境。这意味着，漫漫人生路，同时涌现着异彩纷呈和艰难险阻。",
    "不知不觉已然忙碌了半生，对人生这道深刻的命题，也有了层次不一的理解。",
    "十八岁：成长成人，前方有路",
]
tokenizer = GAUAlphaTokenizer.from_pretrained(converted_ckpt_path)
inputs = tokenizer(text, return_tensors="pt", padding=True, max_length=256)

tf_inputs = [
    tf.convert_to_tensor(inputs["input_ids"].numpy()),
    tf.convert_to_tensor(torch.zeros_like(inputs["input_ids"]).numpy()),
]
tf_model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model=GAU_alpha,
)
bert4keras_outputs = torch.tensor(tf_model(tf_inputs, training=False).numpy())

# pt
pt_model = GAUAlphaModel.from_pretrained(converted_ckpt_path)
with torch.no_grad():
    pt_outputs = pt_model(**inputs).last_hidden_state

print("bert4keras vs pytorch")
print("mean diff :", (bert4keras_outputs - pt_outputs).abs().mean())
print("max diff :", (bert4keras_outputs - pt_outputs).abs().max())
# bert4keras vs pytorch
# mean diff : tensor(6.9320e-07)
# max diff : tensor(3.9101e-05)
