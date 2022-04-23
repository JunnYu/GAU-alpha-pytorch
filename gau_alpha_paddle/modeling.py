import paddle
import paddle.nn as nn
from paddlenlp.transformers import PretrainedModel, register_base_model

from gau_alpha_paddle.layer import GAULayer, Norm

__all__ = [
    "GAUAlphaModel",
    "GAUAlphaPretrainedModel",
    "GAUAlphaForSequenceClassification",
    "GAUAlphaForTokenClassification",
    "GAUAlphaForQuestionAnswering",
    "GAUAlphaForMaskedLM",
]


class GAUAlphaPretrainedModel(PretrainedModel):
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "chinese_GAU-alpha-char_L-24_H-768": {
            "vocab_size": 12000,
            "hidden_size": 768,
            "intermediate_size": 1536,
            "num_hidden_layers": 24,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "attention_key_size": 128,
            "norm_eps": 1e-12,
            "pad_token_id": 0,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "swish",
            "use_bias": False,
            "normalization": "softmax_plus",
            "attention_scale": True,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "chinese_GAU-alpha-char_L-24_H-768": "https://huggingface.co/junnyu/chinese_GAU-alpha-char_L-24_H-768-paddle/resolve/main/model_state.pdparams",
        }
    }
    base_model_prefix = "gau_alpha"

    def init_weights(self, layer):
        """Initialization hook"""
        pass
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range")
                        else self.gau_alpha.config["initializer_range"],
                        shape=layer.weight.shape,
                    )
                )


@register_base_model
class GAUAlphaModel(GAUAlphaPretrainedModel):
    def __init__(
        self,
        vocab_size=12000,
        hidden_size=768,
        intermediate_size=1536,
        num_hidden_layers=24,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        attention_key_size=128,
        norm_eps=1e-12,
        pad_token_id=0,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        hidden_act="swish",
        use_bias=False,
        normalization="softmax_plus",
        attention_scale=True,
    ):
        super(GAUAlphaModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = GAUAlphaEmbeddings(
            vocab_size,
            hidden_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            norm_eps,
        )

        self.encoder = GAUAlphaEncoder(
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            attention_key_size,
            hidden_act,
            use_bias,
            normalization,
            attention_scale,
            attention_probs_dropout_prob,
            hidden_dropout_prob,
            norm_eps,
            max_position_embeddings,
        )

        self.apply(self.init_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=False,
        output_hidden_states=False,
    ):

        if attention_mask is None:
            attention_mask = input_ids != self.pad_token_id
        if attention_mask.ndim == 2:
            attention_mask = attention_mask.unsqueeze(1)  # bs, 1, seqlen
        attention_mask = attention_mask.astype(paddle.get_default_dtype())

        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return encoder_outputs


class GAUAlphaEmbeddings(nn.Layer):
    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        norm_eps=1e-12,
    ):
        super(GAUAlphaEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.norm = Norm(eps=norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embedings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + token_type_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GAUAlphaEncoder(nn.Layer):
    def __init__(
        self,
        num_hidden_layers,
        hidden_size,
        intermediate_size,
        attention_key_size,
        hidden_act,
        use_bias,
        normalization,
        attention_scale,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
        norm_eps,
        max_position_embeddings,
    ):
        super().__init__()
        self.layer = nn.LayerList(
            [
                GAULayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    attention_key_size=attention_key_size,
                    activation=hidden_act,
                    use_bias=use_bias,
                    normalization=normalization,
                    attention_scale=attention_scale,
                    attention_dropout=attention_probs_dropout_prob,
                    hidden_dropout=hidden_dropout_prob,
                    eps=norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        sinusoidal_id = self.get_sinusoidal_id(
            max_position_embeddings, attention_key_size
        )
        self.register_buffer("sin_pos", sinusoidal_id.sin(), persistable=False)
        self.register_buffer("cos_pos", sinusoidal_id.cos(), persistable=False)

    def get_sinusoidal_id(self, max_length, output_dim):
        position_ids = paddle.arange(0, max_length, dtype=paddle.get_default_dtype())
        indices = paddle.arange(0, output_dim // 2, dtype=paddle.get_default_dtype())
        indices = 10000.0 ** (-2 * indices / output_dim)
        sinusoidal_id = paddle.einsum("n,d->nd", position_ids, indices)
        return sinusoidal_id[None, :, :]

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        seqlen = hidden_states.shape[1]
        sinusoidal_pos = self.sin_pos[:, :seqlen, :], self.cos_pos[:, :seqlen, :]

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                sinusoidal_pos,
                output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [
                hidden_states,
                all_hidden_states,
                all_self_attentions,
            ]
            if v is not None
        )


class GAUAlphaForQuestionAnswering(GAUAlphaPretrainedModel):
    def __init__(self, gau_alpha, num_classes=2, dropout=None):
        super().__init__()
        self.gau_alpha = gau_alpha  # allow gau_alpha to be config
        self.dropout = nn.Dropout(
            dropout
            if dropout is not None
            else self.gau_alpha.config["hidden_dropout_prob"]
        )
        self.classifier = nn.Linear(self.gau_alpha.config["hidden_size"], num_classes)
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        start_positions=None,
        end_positions=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        sequence_output = self.gau_alpha(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )[0]

        outputs = self.gau_alpha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output)
        start_logits, end_logits = paddle.unbind(logits, axis=-1)

        total_loss = None

        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.shape[1]
            start_positions = paddle.clip(start_positions, a_min=0, a_max=ignored_index)
            end_positions = paddle.clip(end_positions, a_min=0, a_max=ignored_index)
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits) + outputs[1:]
        return ((total_loss,) + output) if total_loss is not None else output


class GAUAlphaForSequenceClassification(GAUAlphaPretrainedModel):
    def __init__(self, gau_alpha, num_classes=2, dropout=None):
        super().__init__()
        self.num_classes = num_classes
        self.gau_alpha = gau_alpha  # allow gau_alpha to be config
        self.dropout = nn.Dropout(
            dropout
            if dropout is not None
            else self.gau_alpha.config["hidden_dropout_prob"]
        )
        self.classifier = nn.Linear(self.gau_alpha.config["hidden_size"], num_classes)
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        outputs = self.gau_alpha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        pooled_output = self.dropout(sequence_output[:, 0])
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape([-1, self.num_classes]), labels.flatten())

        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output


class GAUAlphaForTokenClassification(GAUAlphaPretrainedModel):
    def __init__(self, gau_alpha, num_classes=2, dropout=None):
        super().__init__()
        self.num_classes = num_classes
        self.gau_alpha = gau_alpha  # allow gau_alpha to be config
        self.dropout = nn.Dropout(
            dropout
            if dropout is not None
            else self.gau_alpha.config["hidden_dropout_prob"]
        )
        self.classifier = nn.Linear(self.gau_alpha.config["hidden_size"], num_classes)
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        outputs = self.gau_alpha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape([-1, self.num_classes]), labels.flatten())

        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output


class GAUAlphaForMaskedLM(GAUAlphaPretrainedModel):
    def __init__(self, gau_alpha):
        super().__init__()
        self.gau_alpha = gau_alpha
        self.mlm_out = lambda x: paddle.matmul(
            x, self.gau_alpha.embeddings.word_embeddings.weight, transpose_y=True
        )

        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
    ):

        outputs = self.gau_alpha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        prediction_scores = self.mlm_out(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.reshape([-1, self.gau_alpha.config["vocab_size"]]),
                labels.flatten(),
            )
        output = (prediction_scores,) + outputs[1:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
