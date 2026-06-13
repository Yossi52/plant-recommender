"""
12_model.py — GCN / GAT 링크 예측 모델 정의
════════════════════════════════════════════════════════════════════

■ 전체 구조
─────────────────────────────────────────────────────────────────

  [입력]
  노드 ID (정수) → Embedding 테이블 → 노드 벡터

  [인코더]
  노드 벡터 + 그래프 구조 → GCN 또는 GAT → 업데이트된 노드 벡터 (z)

  [디코더]
  z[식물 노드] · z[조건 노드] → 연결 확률 (logit)

  흐름 요약:
  node_id → Embedding → (GCNConv or GATConv) × 2 → z → dot product → logit


■ Learnable Embedding 이란?
─────────────────────────────────────────────────────────────────
  nn.Embedding(n_total, hidden_dim) 은 (n_total × hidden_dim) 크기의
  가중치 행렬. 노드 ID 를 인덱스로 넘기면 해당 행(row)을 벡터로 반환.

  예: hidden_dim=64 일 때
    embedding(torch.tensor([0]))   → shape [1, 64]  (가울테리아 벡터)
    embedding(torch.tensor([216])) → shape [1, 64]  (광도_낮음 벡터)

  처음엔 랜덤 초기화, 역전파로 업데이트됨.
  GCN/GAT 는 이 벡터를 이웃 정보로 갱신하는 역할.


■ GCN vs GAT
─────────────────────────────────────────────────────────────────
  GCN (Graph Convolutional Network) — 베이스라인
    이웃 노드의 벡터를 동일한 가중치로 평균:
      z_i = σ( Σ_{j ∈ N(i)} (1/√d_i·d_j) · W · x_j )
    빠르고 단순하지만 "어떤 조건이 더 중요한지" 구별 못함.

  GAT (Graph Attention Network) — 메인 모델
    이웃마다 attention 가중치(α)를 계산해 가중 평균:
      z_i = σ( Σ_{j ∈ N(i)} α_ij · W · x_j )
      α_ij = softmax( LeakyReLU( a^T [Wh_i || Wh_j] ) )
    α 값이 "이 식물 추천에 어떤 조건이 얼마나 중요했는가"를 나타냄.
    → 15_analysis.ipynb 에서 attention 히트맵 시각화에 활용.


■ Decoder — Dot Product (기본값) + 대안 옵션
─────────────────────────────────────────────────────────────────
  기본: 두 노드 벡터의 내적(dot product)으로 연결 확률을 계산.
  값이 클수록 연결 가능성 높음.

  logit = z[plant] · z[cond]          (스칼라)
  prob  = sigmoid(logit)              (0~1, 예측 확률)

  ※ 모델은 sigmoid 전 logit 을 반환.
     Loss: BCEWithLogitsLoss (내부에서 sigmoid 처리 → 수치 안정성)
     추론: sigmoid(logit) 으로 확률 변환

  decoder= 인자로 다른 방식도 선택 가능 (기본값 "dot", 13_train_v2.py 에서 비교):
    "dot"      : 내적 (기존)
    "cosine"   : 정규화된 벡터의 코사인 유사도 × 학습 가능한 scale
                 → embedding norm 영향 제거, 방향만 비교
    "distmult" : (z_plant * r * z_cond).sum()  — r 은 차원별 학습 가중치
    "mlp"      : [z_plant; z_cond] → 작은 MLP → logit
    "l2"       : -||z_plant - z_cond||^2  (TransE 스타일 거리 기반)


■ 하이퍼파라미터 기본값
─────────────────────────────────────────────────────────────────
  HIDDEN_DIM = 64   : Embedding 및 중간 레이어 차원
  OUT_DIM    = 32   : 최종 노드 표현 차원 (디코더 입력)
  GAT_HEADS  = 4    : GAT multi-head attention 수
                      (각 head 가 독립적으로 attention 계산 후 concat)
  DROPOUT    = 0.3  : 과적합 방지용 드롭아웃 비율


■ 제공 클래스
─────────────────────────────────────────────────────────────────
  GCNLinkPred  : GCN 기반 링크 예측 모델 (베이스라인)
  GATLinkPred  : GAT 기반 링크 예측 모델 (메인)
  build_model  : 문자열로 모델 선택하는 헬퍼 함수

════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

# ── 기본 하이퍼파라미터 ────────────────────────────────────
HIDDEN_DIM = 64    # Embedding + GCN/GAT 중간 차원
OUT_DIM    = 32    # 최종 노드 표현 차원
GAT_HEADS  = 4     # GAT multi-head 수
DROPOUT    = 0.3   # 드롭아웃 비율


# ══════════════════════════════════════════════════════════
# Decoder 공통 로직 (dot / cosine / distmult / mlp / l2)
# ══════════════════════════════════════════════════════════
DECODER_TYPES = ["dot", "cosine", "distmult", "mlp", "l2"]


class _DecoderMixin:
    """
    GCNLinkPred / GATLinkPred 공용 decoder.

    self.decoder 값에 따라 두 노드 벡터로부터 logit 을 계산하는 방식이 달라짐.
    "dot" 이 기존 동작과 100% 동일 (기본값).
    """

    def _init_decoder(self, out_dim: int, decoder: str = "dot"):
        if decoder not in DECODER_TYPES:
            raise ValueError(f"decoder 는 {DECODER_TYPES} 중 하나. 입력값: {decoder!r}")
        self.decoder = decoder

        if decoder == "cosine":
            # 코사인 유사도(-1~1)는 sigmoid 통과 시 0.27~0.73 으로 압축됨
            # → 학습 가능한 scale 로 보정 (CLIP의 logit_scale과 동일한 아이디어)
            self.cos_scale = nn.Parameter(torch.tensor(5.0))
        elif decoder == "distmult":
            # 차원별 학습 가능한 relation 가중치
            self.rel = nn.Parameter(torch.ones(out_dim))
        elif decoder == "mlp":
            self.mlp_decoder = nn.Sequential(
                nn.Linear(out_dim * 2, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, 1),
            )

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """두 노드 벡터로부터 logit 계산 (decoder 종류에 따라 분기)"""
        src, dst = edge_index
        if self.decoder == "dot":
            return (z[src] * z[dst]).sum(dim=-1)
        elif self.decoder == "cosine":
            zn = F.normalize(z, dim=-1)
            return self.cos_scale * (zn[src] * zn[dst]).sum(dim=-1)
        elif self.decoder == "distmult":
            return (z[src] * self.rel * z[dst]).sum(dim=-1)
        elif self.decoder == "mlp":
            h = torch.cat([z[src], z[dst]], dim=-1)
            return self.mlp_decoder(h).squeeze(-1)
        elif self.decoder == "l2":
            return -((z[src] - z[dst]) ** 2).sum(dim=-1)

    def score(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        """
        임의의 두 벡터 집합 간 decoder 기반 pairwise score 행렬.

        Parameters
        ----------
        za : FloatTensor [N, out_dim]
        zb : FloatTensor [M, out_dim]

        Returns
        -------
        FloatTensor [N, M]  — za[i] 와 zb[j] 사이의 logit (decoder 종류에 따라 분기)

        ※ decode_all_plants() 와 14_recommend.py / 15_analysis.ipynb 의
          "조건 부분 입력" 점수 계산에서 공통으로 사용.
          raw dot product (za @ zb.t()) 를 직접 쓰면 cosine/distmult/mlp/l2
          decoder 로 학습한 모델에서는 학습 때와 다른 점수 체계가 되어버리므로,
          반드시 이 함수를 통해 decoder 와 일치하는 점수를 계산해야 함.
        """
        if self.decoder == "dot":
            return za @ zb.t()
        elif self.decoder == "cosine":
            zan = F.normalize(za, dim=-1)
            zbn = F.normalize(zb, dim=-1)
            return self.cos_scale * (zan @ zbn.t())
        elif self.decoder == "distmult":
            return (za * self.rel) @ zb.t()
        else:  # "mlp", "l2" — 모든 (a, b) 쌍을 직접 계산
            n_a, n_b = za.shape[0], zb.shape[0]
            za_exp = za.unsqueeze(1).expand(-1, n_b, -1).reshape(-1, za.shape[-1])
            zb_exp = zb.unsqueeze(0).expand(n_a, -1, -1).reshape(-1, zb.shape[-1])
            if self.decoder == "mlp":
                out = self.mlp_decoder(torch.cat([za_exp, zb_exp], dim=-1)).squeeze(-1)
            else:  # l2
                out = -((za_exp - zb_exp) ** 2).sum(dim=-1)
            return out.view(n_a, n_b)

    def decode_all_plants(
        self, z: torch.Tensor, n_plants: int, n_conditions: int
    ) -> torch.Tensor:
        """모든 식물 × 모든 조건 쌍의 logit 행렬 (추천용)"""
        return self.score(z[:n_plants], z[n_plants:])


# ══════════════════════════════════════════════════════════
# GCN 기반 링크 예측 모델 (베이스라인)
# ══════════════════════════════════════════════════════════
class GCNLinkPred(nn.Module, _DecoderMixin):
    """
    Embedding → GCNConv × 2 → decoder (기본: dot product)

    Parameters
    ----------
    num_nodes  : 전체 노드 수 (식물 + 조건)
    hidden_dim : Embedding 및 1번째 GCN 출력 차원
    out_dim    : 2번째 GCN 출력 차원 (최종 node representation)
    dropout    : 드롭아웃 비율
    decoder    : "dot"(기본) / "cosine" / "distmult" / "mlp" / "l2"
    """

    def __init__(
        self,
        num_nodes:  int,
        hidden_dim: int = HIDDEN_DIM,
        out_dim:    int = OUT_DIM,
        dropout:    float = DROPOUT,
        decoder:    str = "dot",
    ):
        super().__init__()

        # 각 노드의 학습 가능한 초기 벡터
        self.embedding = nn.Embedding(num_nodes, hidden_dim)

        # GCN 레이어 1: hidden_dim → hidden_dim
        self.conv1 = GCNConv(hidden_dim, hidden_dim)

        # GCN 레이어 2: hidden_dim → out_dim
        self.conv2 = GCNConv(hidden_dim, out_dim)

        self.dropout = nn.Dropout(dropout)
        self._init_decoder(out_dim, decoder)

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        그래프 구조를 이용해 노드 벡터를 갱신.

        Parameters
        ----------
        edge_index : LongTensor [2, E]  — 학습용 그래프 엣지 (train_edge_index)

        Returns
        -------
        z : FloatTensor [num_nodes, out_dim]  — 갱신된 노드 표현
        """
        # 전체 노드 ID 생성: [0, 1, 2, ..., num_nodes-1]
        node_ids = torch.arange(self.embedding.num_embeddings,
                                device=edge_index.device)

        x = self.embedding(node_ids)           # [N, hidden_dim]
        x = self.conv1(x, edge_index).relu()   # 이웃 정보 집계 후 활성화
        x = self.dropout(x)
        z = self.conv2(x, edge_index)          # 최종 표현, 활성화 없음
        return z                               # [N, out_dim]

    def forward(self, edge_index: torch.Tensor, target_edge: torch.Tensor) -> torch.Tensor:
        """학습용 forward: encode → decode → logit"""
        z = self.encode(edge_index)
        return self.decode(z, target_edge)


# ══════════════════════════════════════════════════════════
# GAT 기반 링크 예측 모델 (메인)
# ══════════════════════════════════════════════════════════
class GATLinkPred(nn.Module, _DecoderMixin):
    """
    Embedding → GATConv × 2 → decoder (기본: dot product)

    GCNLinkPred 와 구조는 동일하나 GATConv 를 사용.
    마지막 레이어에서 attention 가중치를 반환할 수 있어
    "어떤 조건이 추천에 더 중요했는가" 해석에 활용 가능.

    Parameters
    ----------
    num_nodes  : 전체 노드 수
    hidden_dim : Embedding 차원 (= heads × head_dim)
    out_dim    : 최종 표현 차원
    heads      : multi-head attention 수
                 (1번째 레이어에서 heads 개 attention 병렬 계산 후 concat)
    dropout    : 드롭아웃 비율
    decoder    : "dot"(기본) / "cosine" / "distmult" / "mlp" / "l2"
    """

    def __init__(
        self,
        num_nodes:  int,
        hidden_dim: int = HIDDEN_DIM,
        out_dim:    int = OUT_DIM,
        heads:      int = GAT_HEADS,
        dropout:    float = DROPOUT,
        decoder:    str = "dot",
    ):
        super().__init__()

        self.embedding = nn.Embedding(num_nodes, hidden_dim)

        # GAT 레이어 1: hidden_dim → hidden_dim (heads 개 head, 각 hidden_dim//heads 차원)
        # concat=True(기본) → 출력 차원 = (hidden_dim // heads) * heads = hidden_dim
        self.conv1 = GATConv(
            in_channels  = hidden_dim,
            out_channels = hidden_dim // heads,  # head 당 차원
            heads        = heads,
            dropout      = dropout,
            concat       = True,                 # 출력 = heads 개 벡터를 이어붙임
        )

        # GAT 레이어 2: hidden_dim → out_dim (head=1, attention 가중치 추출용)
        # concat=False → 출력 차원 = out_dim
        self.conv2 = GATConv(
            in_channels  = hidden_dim,
            out_channels = out_dim,
            heads        = 1,
            dropout      = dropout,
            concat       = False,                # 출력 = out_dim (단일 벡터)
        )

        self.dropout = nn.Dropout(dropout)
        self._init_decoder(out_dim, decoder)

    def encode(
        self,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Parameters
        ----------
        edge_index        : LongTensor [2, E]
        return_attention  : True 이면 마지막 레이어의 attention 가중치 반환

        Returns (return_attention=False)
        -------
        z : FloatTensor [N, out_dim]

        Returns (return_attention=True)
        -------
        z         : FloatTensor [N, out_dim]
        attn_edge : LongTensor [2, E]   — attention 이 계산된 엣지
        attn_w    : FloatTensor [E, 1]  — 각 엣지의 attention 가중치 α
        """
        node_ids = torch.arange(self.embedding.num_embeddings,
                                device=edge_index.device)

        x = self.embedding(node_ids)              # [N, hidden_dim]
        x = self.conv1(x, edge_index).relu()      # [N, hidden_dim]
        x = self.dropout(x)

        if return_attention:
            # return_attention_weights=True: (출력벡터, (엣지인덱스, attention가중치)) 반환
            z, (attn_edge, attn_w) = self.conv2(
                x, edge_index, return_attention_weights=True
            )
            return z, attn_edge, attn_w            # 시각화용
        else:
            z = self.conv2(x, edge_index)
            return z

    def forward(self, edge_index: torch.Tensor, target_edge: torch.Tensor) -> torch.Tensor:
        """학습용 forward: encode → decode → logit"""
        z = self.encode(edge_index)
        return self.decode(z, target_edge)


# ══════════════════════════════════════════════════════════
# 헬퍼: 문자열로 모델 선택
# ══════════════════════════════════════════════════════════
def build_model(
    model_type: str,
    num_nodes:  int,
    hidden_dim: int = HIDDEN_DIM,
    out_dim:    int = OUT_DIM,
    **kwargs,
) -> nn.Module:
    """
    Parameters
    ----------
    model_type : "gcn" 또는 "gat"
    num_nodes  : 전체 노드 수
    hidden_dim : Embedding 차원
    out_dim    : 최종 표현 차원
    **kwargs   : GAT 의 경우 heads=, dropout= 전달 가능
                 decoder= "dot"(기본)/"cosine"/"distmult"/"mlp"/"l2"

    Returns
    -------
    GCNLinkPred 또는 GATLinkPred 인스턴스

    Examples
    --------
    model = build_model("gat", num_nodes=261)
    model = build_model("gcn", num_nodes=261, hidden_dim=128, out_dim=64)
    model = build_model("gat", num_nodes=261, decoder="cosine")
    """
    model_type = model_type.lower()
    if model_type == "gcn":
        return GCNLinkPred(num_nodes, hidden_dim, out_dim,
                           dropout=kwargs.get("dropout", DROPOUT),
                           decoder=kwargs.get("decoder", "dot"))
    elif model_type == "gat":
        return GATLinkPred(num_nodes, hidden_dim, out_dim,
                           heads=kwargs.get("heads", GAT_HEADS),
                           dropout=kwargs.get("dropout", DROPOUT),
                           decoder=kwargs.get("decoder", "dot"))
    else:
        raise ValueError(f"model_type 은 'gcn' 또는 'gat' 만 지원. 입력값: {model_type!r}")


# ══════════════════════════════════════════════════════════
# 직접 실행 시 모델 구조 출력
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    NUM_NODES = 261   # 10_build_graph.py 결과 기준

    print("=" * 55)
    print("GCN 모델")
    print("=" * 55)
    gcn = build_model("gcn", num_nodes=NUM_NODES)
    print(gcn)
    n_params = sum(p.numel() for p in gcn.parameters() if p.requires_grad)
    print(f"학습 파라미터 수: {n_params:,}")

    print()
    print("=" * 55)
    print("GAT 모델")
    print("=" * 55)
    gat = build_model("gat", num_nodes=NUM_NODES)
    print(gat)
    n_params = sum(p.numel() for p in gat.parameters() if p.requires_grad)
    print(f"학습 파라미터 수: {n_params:,}")

    # 더미 forward 테스트
    print()
    print("── Forward 테스트 ──────────────────────────────────")
    dummy_edge_index  = torch.randint(0, NUM_NODES, (2, 100))
    dummy_target_edge = torch.randint(0, NUM_NODES, (2, 20))

    logit_gcn = gcn(dummy_edge_index, dummy_target_edge)
    logit_gat = gat(dummy_edge_index, dummy_target_edge)
    print(f"GCN logit shape : {logit_gcn.shape}  (기대: [20])")
    print(f"GAT logit shape : {logit_gat.shape}  (기대: [20])")
    print("────────────────────────────────────────────────────")
