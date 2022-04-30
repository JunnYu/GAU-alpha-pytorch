from transformers.models.bert import BertTokenizer, BertTokenizerFast


class GAUAlphaTokenizerFast(BertTokenizerFast):
    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + token_ids_1 + sep) * [0]


class GAUAlphaTokenizer(BertTokenizer):
    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + token_ids_1 + sep) * [0]


from gau_alpha.layer import GatedAttentionUnit, GAULayer
from gau_alpha.modeling_gau_alpha import (
    GAUAlphaConfig,
    GAUAlphaForMaskedLM,
    GAUAlphaForMultipleChoice,
    GAUAlphaForQuestionAnswering,
    GAUAlphaForSequenceClassification,
    GAUAlphaForTokenClassification,
    GAUAlphaModel,
    GAUAlphaPreTrainedModel,
)
