from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from msrhgnn_model import (
    GENE_NETWORK_RELATION,
    LOCAL_DD_TRAIN_RELATION,
    LOCAL_DG_RELATION,
    LOCAL_DTI_RELATION,
    META_DISEASE_INTERACT_RELATION,
    META_DISEASE_SHARED_RELATION,
    META_DRUG_DISEASE_RELATION,
    SIM_DISEASE_RELATIONS,
    SIM_DRUG_RELATIONS,
    MultiViewMSRHGNN,
    merge_relation_graphs,
    reverse_edges,
    weighted_mean_aggregate,
)


@dataclass(frozen=True)
class AblationConfig:
    disable_similarity_branch: bool = False
    disable_low_order_relations: bool = False
    disable_high_order_relations: bool = False


class AblationMultiViewMSRHGNN(MultiViewMSRHGNN):
    def __init__(
        self,
        drug_dim: int,
        disease_dim: int,
        gene_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        ablation: Optional[AblationConfig] = None,
    ) -> None:
        super().__init__(
            drug_dim=drug_dim,
            disease_dim=disease_dim,
            gene_dim=gene_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.ablation = ablation or AblationConfig()

    @staticmethod
    def _zero_branch(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros_like(x), torch.zeros((1,), device=x.device, dtype=x.dtype)

    def encode(
        self,
        drug_x: torch.Tensor,
        disease_x: torch.Tensor,
        gene_x: torch.Tensor,
        relation_edges: Dict[str, torch.Tensor],
        relation_weights: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[float]]]:
        drug_base = self.dropout(F.relu(self.drug_proj(drug_x)))
        disease_base = self.dropout(F.relu(self.disease_proj(disease_x)))
        gene_base = self.dropout(F.relu(self.gene_proj(gene_x)))

        if self.ablation.disable_similarity_branch:
            drug_sim, drug_sim_weights = self._zero_branch(drug_base)
            disease_sim, disease_sim_weights = self._zero_branch(disease_base)
        else:
            drug_sim_views: List[torch.Tensor] = []
            for layer, relation in zip(self.drug_sim_layers, SIM_DRUG_RELATIONS):
                if relation in relation_edges:
                    view = self.dropout(F.relu(layer(drug_base, relation_edges[relation], relation_weights.get(relation))))
                    drug_sim_views.append(view)
            disease_sim_views: List[torch.Tensor] = []
            for layer, relation in zip(self.disease_sim_layers, SIM_DISEASE_RELATIONS):
                if relation in relation_edges:
                    view = self.dropout(F.relu(layer(disease_base, relation_edges[relation], relation_weights.get(relation))))
                    disease_sim_views.append(view)

            drug_sim_fused, drug_sim_weights = self.drug_sim_fusion(drug_sim_views if drug_sim_views else [drug_base])
            disease_sim_fused, disease_sim_weights = self.disease_sim_fusion(disease_sim_views if disease_sim_views else [disease_base])

            available_drug_relations = [r for r in SIM_DRUG_RELATIONS if r in relation_edges]
            available_disease_relations = [r for r in SIM_DISEASE_RELATIONS if r in relation_edges]
            drug_merge_weights = drug_sim_weights if available_drug_relations else torch.ones((1,), device=drug_base.device, dtype=drug_base.dtype)
            disease_merge_weights = disease_sim_weights if available_disease_relations else torch.ones((1,), device=disease_base.device, dtype=disease_base.dtype)
            merged_drug_edges, merged_drug_weights = merge_relation_graphs(
                available_drug_relations,
                relation_edges,
                relation_weights,
                drug_merge_weights,
                drug_base.size(0),
            )
            merged_disease_edges, merged_disease_weights = merge_relation_graphs(
                available_disease_relations,
                relation_edges,
                relation_weights,
                disease_merge_weights,
                disease_base.size(0),
            )
            drug_sim = self.drug_sim_transformer(drug_sim_fused, merged_drug_edges, merged_drug_weights)
            disease_sim = self.disease_sim_transformer(disease_sim_fused, merged_disease_edges, merged_disease_weights)

        gene_ctx = gene_base
        if GENE_NETWORK_RELATION in relation_edges:
            gene_ctx = self.gene_transformer(
                gene_base,
                relation_edges[GENE_NETWORK_RELATION],
                relation_weights.get(GENE_NETWORK_RELATION),
            )

        if self.ablation.disable_low_order_relations:
            drug_local, drug_local_weights = self._zero_branch(drug_base)
            disease_local, disease_local_weights = self._zero_branch(disease_base)
        else:
            drug_local_inputs: List[torch.Tensor] = []
            if LOCAL_DTI_RELATION in relation_edges:
                rev_edges, rev_weights = reverse_edges(relation_edges[LOCAL_DTI_RELATION], relation_weights.get(LOCAL_DTI_RELATION))
                agg = weighted_mean_aggregate(gene_ctx, rev_edges, drug_base.size(0), rev_weights)
                drug_local_inputs.append(self.dropout(F.relu(self.drug_local_gene(agg))))
            if LOCAL_DD_TRAIN_RELATION in relation_edges and relation_edges[LOCAL_DD_TRAIN_RELATION].numel() > 0:
                rev_edges, rev_weights = reverse_edges(relation_edges[LOCAL_DD_TRAIN_RELATION], relation_weights.get(LOCAL_DD_TRAIN_RELATION))
                agg = weighted_mean_aggregate(disease_base, rev_edges, drug_base.size(0), rev_weights)
                drug_local_inputs.append(self.dropout(F.relu(self.drug_local_disease(agg))))
            drug_local, drug_local_weights = self.local_drug_fusion(drug_local_inputs if drug_local_inputs else [drug_base])

            disease_local_inputs: List[torch.Tensor] = []
            if LOCAL_DG_RELATION in relation_edges:
                rev_edges, rev_weights = reverse_edges(relation_edges[LOCAL_DG_RELATION], relation_weights.get(LOCAL_DG_RELATION))
                agg = weighted_mean_aggregate(gene_ctx, rev_edges, disease_base.size(0), rev_weights)
                disease_local_inputs.append(self.dropout(F.relu(self.disease_local_gene(agg))))
            if LOCAL_DD_TRAIN_RELATION in relation_edges and relation_edges[LOCAL_DD_TRAIN_RELATION].numel() > 0:
                agg = weighted_mean_aggregate(drug_base, relation_edges[LOCAL_DD_TRAIN_RELATION], disease_base.size(0), relation_weights.get(LOCAL_DD_TRAIN_RELATION))
                disease_local_inputs.append(self.dropout(F.relu(self.disease_local_drug(agg))))
            disease_local, disease_local_weights = self.local_disease_fusion(disease_local_inputs if disease_local_inputs else [disease_base])

        if self.ablation.disable_high_order_relations:
            drug_global, drug_global_weights = self._zero_branch(drug_base)
            disease_global, disease_global_weights = self._zero_branch(disease_base)
        else:
            drug_global_inputs: List[torch.Tensor] = []
            if META_DRUG_DISEASE_RELATION in relation_edges:
                rev_edges, rev_weights = reverse_edges(relation_edges[META_DRUG_DISEASE_RELATION], relation_weights.get(META_DRUG_DISEASE_RELATION))
                agg = weighted_mean_aggregate(disease_base, rev_edges, drug_base.size(0), rev_weights)
                drug_global_inputs.append(self.dropout(F.relu(self.drug_global_relation(agg))))
            drug_global, drug_global_weights = self.global_drug_fusion(drug_base, drug_global_inputs if drug_global_inputs else [drug_local])

            disease_global_inputs: List[torch.Tensor] = []
            if META_DRUG_DISEASE_RELATION in relation_edges:
                agg = weighted_mean_aggregate(drug_base, relation_edges[META_DRUG_DISEASE_RELATION], disease_base.size(0), relation_weights.get(META_DRUG_DISEASE_RELATION))
                disease_global_inputs.append(self.dropout(F.relu(self.disease_global_drug(agg))))
            if META_DISEASE_SHARED_RELATION in relation_edges:
                agg = weighted_mean_aggregate(disease_base, relation_edges[META_DISEASE_SHARED_RELATION], disease_base.size(0), relation_weights.get(META_DISEASE_SHARED_RELATION))
                disease_global_inputs.append(self.dropout(F.relu(self.disease_global_shared(agg))))
            if META_DISEASE_INTERACT_RELATION in relation_edges:
                agg = weighted_mean_aggregate(disease_base, relation_edges[META_DISEASE_INTERACT_RELATION], disease_base.size(0), relation_weights.get(META_DISEASE_INTERACT_RELATION))
                disease_global_inputs.append(self.dropout(F.relu(self.disease_global_interact(agg))))
            disease_global, disease_global_weights = self.global_disease_fusion(disease_base, disease_global_inputs if disease_global_inputs else [disease_local])

        drug_gate = torch.sigmoid(self.drug_hetero_gate(torch.cat([drug_local, drug_global], dim=-1)))
        disease_gate = torch.sigmoid(self.disease_hetero_gate(torch.cat([disease_local, disease_global], dim=-1)))
        drug_hetero = drug_gate * drug_local + (1.0 - drug_gate) * drug_global
        disease_hetero = disease_gate * disease_local + (1.0 - disease_gate) * disease_global

        drug_repr = self.final_drug_norm(torch.cat([drug_sim, drug_hetero], dim=-1))
        disease_repr = self.final_disease_norm(torch.cat([disease_sim, disease_hetero], dim=-1))

        aux = {
            "drug_sim_weights": [float(x) for x in drug_sim_weights.detach().cpu().tolist()],
            "disease_sim_weights": [float(x) for x in disease_sim_weights.detach().cpu().tolist()],
            "drug_local_weights": [float(x) for x in drug_local_weights.detach().cpu().tolist()],
            "disease_local_weights": [float(x) for x in disease_local_weights.detach().cpu().tolist()],
            "drug_global_weights": [float(x) for x in drug_global_weights.detach().cpu().tolist()],
            "disease_global_weights": [float(x) for x in disease_global_weights.detach().cpu().tolist()],
        }
        return drug_repr, disease_repr, aux
