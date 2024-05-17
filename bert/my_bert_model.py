from torchcrf import CRF
import torch
from torch import nn
from torchvision.models import resnet152
from transformers import BertModel, BertPreTrainedModel
from utils import ContextAwareGate, DynamicAttentionModule


class BertCrossAttention(nn.Module):
    """Implements cross-attention between two different modalities using a decoder layer."""

    def __init__(self, config, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
            for _ in range(num_layers)
        ])

    def forward(self, query, key, mask=None):
        output = query
        for layer in self.layers:
            output = layer(output, key, tgt_key_padding_mask=mask)
        return output


class MTCCMBertForMMTokenClassificationCRF(BertPreTrainedModel):
    def __init__(self, config, num_labels, add_context_aware_gate=False, use_dynamic_cross_modal_fusion=False):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.resnet = resnet152(pretrained=True)
        self.resnet.fc = nn.Identity()  # Adapt ResNet to remove the final fully connected layer

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(2048, config.hidden_size)

        self.txt2img_attention = BertCrossAttention(config, num_layers=1)
        self.img2txt_attention = BertCrossAttention(config, num_layers=1)

        self.add_context_aware_gate = add_context_aware_gate
        # Initialize the visual filter gate
        if add_context_aware_gate:
            self.visual_gate = ContextAwareGate(config.hidden_size, 768)

        if use_dynamic_cross_modal_fusion:
            self.dynamic_attention = DynamicAttentionModule(config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.init_weights()

        self.use_dynamic_cross_modal_fusion = use_dynamic_cross_modal_fusion

    def forward(self, input_ids, attention_mask, visual_embeds, labels=None):
        # Text feature extraction
        text_outputs = self.bert(input_ids, attention_mask=attention_mask)
        text_features = self.dropout(text_outputs.last_hidden_state)

        # Image feature extraction with ResNet-152
        visual_features = self.resnet(visual_embeds)  # Assuming visual_embeds is [batch_size, 3, 224, 224]
        visual_features = visual_features.view(visual_features.size(0), -1)  # Flatten the output of the ResNet
        visual_features = self.vismap2text(visual_features)  # Transform to match BERT hidden size
        visual_features = visual_features.unsqueeze(1).expand(-1, text_features.size(1),
                                                              -1)  # Expand to match text sequence length

        if self.add_context_aware_gate:
            visual_features = self.visual_gate(text_features, visual_features)

        # Cross-modal attention
        if self.use_dynamic_cross_modal_fusion:
            attended_text, attended_visuals = self.dynamic_attention(text_features, visual_features)
            combined_features = torch.cat([attended_text, attended_visuals], dim=-1)
        else:
            txt_attended_visuals = self.txt2img_attention(text_features, visual_features)
            img_attended_text = self.img2txt_attention(visual_features, text_features)
            combined_features = torch.cat([txt_attended_visuals, img_attended_text], dim=-1)

        logits = self.classifier(combined_features)

        # crf processing
        if labels is not None:
            # Ensure labels and logits have the same sequence length
            labels = torch.where(labels == -100, torch.zeros_like(labels), labels)

            seq_length = logits.size(1)
            if labels.size(1) < seq_length:
                padding_size = seq_length - labels.size(1)
                # Use a valid label index for padding, e.g., 0
                labels_padded = torch.full((labels.size(0), padding_size), fill_value=0, dtype=torch.long,
                                           device=labels.device)
                labels = torch.cat([labels, labels_padded], dim=1)

                # Adjust attention_mask to cover only the non-padded areas
                attention_mask_padded = torch.zeros((attention_mask.size(0), seq_length), dtype=torch.uint8,
                                                    device=attention_mask.device)
                attention_mask_padded[:, :attention_mask.size(1)] = attention_mask
                attention_mask = attention_mask_padded

            # CRF loss calculation
            loss = -self.crf(logits, labels, mask=attention_mask.byte(), reduction='mean')
            return loss
        else:
            return self.crf.decode(logits, mask=attention_mask.byte())
