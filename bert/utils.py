import torch
from torch import nn


class ContextAwareGate(nn.Module):
    def __init__(self, text_dim, visual_dim):
        super().__init__()
        # network to calculate threshold
        self.threshold_network = nn.Sequential(
            nn.Linear(text_dim, 1),  # Averaging text features to a single value
            nn.Sigmoid()  # Ensure the threshold is between 0 and 1
        )
        self.gate = nn.Sequential(
            nn.Linear(visual_dim * 2, visual_dim),  # Combine visual and transformed text features
            nn.Tanh(),
            nn.Linear(visual_dim, visual_dim),
            nn.Sigmoid()
        )

    def forward(self, text_features, visual_features):
        combined_features = torch.cat([text_features, visual_features], dim=-1)
        # Compute gating values
        gate_values = self.gate(combined_features)
        # Apply the gate to the visual features only
        text_mean = torch.mean(text_features, dim=1)
        update_threshold = self.threshold_network(text_mean).squeeze()  # Ensuring scalar output per batch item

        # if the threshold is larger than certain value apply combined features to the visual features
        update_threshold_expanded = update_threshold.unsqueeze(-1).unsqueeze(-1)
        update_mask = (gate_values > update_threshold_expanded).float()

        # Apply the gate to the visual features conditionally
        updated_visual_features = visual_features * (1 - update_mask) + (visual_features * gate_values) * update_mask

        return updated_visual_features


class DynamicAttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.text_weight_predictor = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        self.visual_weight_predictor = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, text_features, visual_features):
        text_weights = self.text_weight_predictor(text_features).expand_as(text_features)
        visual_weights = self.visual_weight_predictor(visual_features).expand_as(visual_features)
        attended_text = text_features * text_weights
        attended_visuals = visual_features * visual_weights
        return attended_text, attended_visuals
