from typing import Optional
from enum import IntEnum
import logging
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from lpl_planner.model.modules import LayerNorm, Mlp, MixerBlock
from lpl_planner.planning.scene.scene_feature.features import (SceneFeature, 
                                                                  RoadFeature, 
                                                                  AgentFeature, 
                                                                  StaticObstacleFeature, 
                                                                  EgoFeature,
                                                                  RouteFeature
                                                                  )
from lpl_planner.planning.planner.utils.int_enum import RoadType
LOGIT_CLAMP = 30.0
logger = logging.getLogger(__name__)

def _ensure_finite(x: torch.Tensor, name: str, clamp: Optional[float] = None) -> torch.Tensor:
    """
    Ensure a tensor has no NaN/Inf. If found, sanitize and optionally clamp.
    """
    if not torch.is_tensor(x):
        return x
    finite = torch.isfinite(x)
    if not finite.all():
        num_bad = (~finite).sum().item()
        logger.warning("Tensor %s has %s non-finite values. Sanitizing with nan_to_num.", name, num_bad)
        x = torch.nan_to_num(x, nan=0.0, posinf=LOGIT_CLAMP if clamp is None else clamp, neginf=-(LOGIT_CLAMP if clamp is None else clamp))
    if clamp is not None:
        x = torch.clamp(x, -clamp, clamp)
    return x

class SceneFeatureIDX(IntEnum):
    ROAD = 0
    ROUTE = 1
    STATIC = 2
    AGENT = 3
    EGO = 4

class SceneStateEncoder(nn.Module):
    """
    Unified encoder for trajectory planning state representation
    Integrates road context, dynamic agents, static obstacles, reference path and ego state
    """
    
    def __init__(self,
                 unified_dim=256,
                 num_heads=8,
                 encoder_depth=2,
                 debug: bool = False,
                 ):
        super().__init__()
        self.debug = bool(debug)
        
        # ----------------- Modular Encoders -----------------
        self.road_encoder = RoadEncoder(hidden_dim=unified_dim)
        self.agent_encoder = AgentEncoder(d_model=unified_dim)
        self.static_encoder = StaticObstacleEncoder(hidden_dim=unified_dim)
        self.route_encoder = RouteEncoder(hidden_dim=unified_dim)
        self.ego_encoder = EgoStateEncoder(hidden_dim=unified_dim)
        
        # ----------------- Feature Projection -----------------
        self.proj_road = nn.Linear(unified_dim, unified_dim)
        self.proj_agent = nn.Linear(unified_dim, unified_dim)
        self.proj_static = nn.Linear(unified_dim, unified_dim)
        self.proj_route = nn.Linear(unified_dim, unified_dim)
        self.proj_ego = nn.Linear(unified_dim, unified_dim)
        # self.type_embed = nn.Embedding(5, unified_dim)  # 5 types: road, dynamic, static, refpath, ego
        

        
        # ----------------- Context Aggregation -----------------
        self.type_embed = nn.Embedding(5, unified_dim)  # 0:road,1:agent,2:static,3:ref,4:ego
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=unified_dim, nhead=num_heads, batch_first=True
        )
        self.fusion = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_depth,
            enable_nested_tensor=False,
        )
        self.num_heads = num_heads

    def _maybe_debug_tensor(self, x: torch.Tensor, name: str, clamp: Optional[float] = None) -> torch.Tensor:
        if not self.debug:
            return x
        return _ensure_finite(x, name, clamp=clamp)

    def forward(self, batch_inputs: SceneFeature) -> Tensor:
        """Process batched scene inputs with padding
        Args:
            batch_inputs (SceneFeature): Batched scene features containing:
                - road_feature: RoadFeature object.
                - agent_feature: AgentFeature object.
                - static_obstacle_feature: StaticObstacleFeature object.
                - route_feature: RoadFeature object
                - ego_feature: List of variable-length Tensors [6]
        Returns:
            (Tensor, Tensor, Tensor): Padded features, mask and position IDs
        """
        # model_dtype = self.proj_road.weight.dtype  # Get model dtype from road encoder
        # batch_inputs = batch_inputs.to_feature_tensor(dtype=model_dtype)  # Convert to tensor format
        batch_size = batch_inputs.ego_feature.ego_current_state.shape[0]
        device = batch_inputs.ego_feature.ego_current_state.device

        # ----------------- Batch Preprocessing -----------------
        road_batch = batch_inputs.road_feature
        agent_batch = batch_inputs.agent_feature
        static_batch = batch_inputs.static_obstacle_feature
        route_batch = batch_inputs.route_feature
        ego_batch = batch_inputs.ego_feature

        # ----------------- Modular Encoding -----------------
        road_feat, road_mask = self.road_encoder(road_batch)  # [B, N, D]
        agent_feat, agent_mask = self.agent_encoder(agent_batch)  # [B, M, D]
        static_feat, static_mask = self.static_encoder(static_batch)  # [B, K, D]
        route_feat, route_mask = self.route_encoder(route_batch)  # [B, R, D]
        ego_feat = self.ego_encoder(ego_batch).unsqueeze(1)  # [B, 1, D]
        # road_feat = self._maybe_debug_tensor(road_feat, "scene_encoder.road_feat_preproj", clamp=1e6)
        # agent_feat = self._maybe_debug_tensor(agent_feat, "scene_encoder.agent_feat_preproj", clamp=1e6)
        # static_feat = self._maybe_debug_tensor(static_feat, "scene_encoder.static_feat_preproj", clamp=1e6)
        # route_feat = self._maybe_debug_tensor(route_feat, "scene_encoder.route_feat_preproj", clamp=1e6)
        # ego_feat = self._maybe_debug_tensor(ego_feat, "scene_encoder.ego_feat_preproj", clamp=1e6)

        # ----------------- Feature Projection -----------------
        road_feat = self.proj_road(road_feat)
        agent_feat = self.proj_agent(agent_feat)
        static_feat = self.proj_static(static_feat)
        route_feat = self.proj_route(route_feat)
        ego_feat = self.proj_ego(ego_feat)
        # road_feat = self._maybe_debug_tensor(road_feat, "scene_encoder.road_feat_postproj", clamp=1e6)
        # agent_feat = self._maybe_debug_tensor(agent_feat, "scene_encoder.agent_feat_postproj", clamp=1e6)
        # static_feat = self._maybe_debug_tensor(static_feat, "scene_encoder.static_feat_postproj", clamp=1e6)
        # route_feat = self._maybe_debug_tensor(route_feat, "scene_encoder.route_feat_postproj", clamp=1e6)
        # ego_feat = self._maybe_debug_tensor(ego_feat, "scene_encoder.ego_feat_postproj", clamp=1e6)

        # ----------------- Cross-modal Fusion -----------------
        ego_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)  # [B,1]
        concat_feat = torch.cat([road_feat, route_feat, static_feat, agent_feat, ego_feat], dim=1)
        concat_mask = torch.cat([road_mask, route_mask, static_mask, agent_mask, ego_mask], dim=1)

        # 生成带batch维度的position_ids
        position_ids = torch.cat([
            torch.full((road_feat.shape[1],), SceneFeatureIDX.ROAD,),
            torch.full((route_feat.shape[1],), SceneFeatureIDX.ROUTE,),
            torch.full((static_feat.shape[1],), SceneFeatureIDX.STATIC,),
            torch.full((agent_feat.shape[1],), SceneFeatureIDX.AGENT,),
            torch.tensor([SceneFeatureIDX.EGO])
        ]).to(device).unsqueeze(0).expand(batch_size, -1)  # [B, N]
        
        # Transformer fusion
        type_emb = self.type_embed(position_ids)              # [B, N, D]
        concat_feat = concat_feat + type_emb
        fused_feat = self.fusion(concat_feat, src_key_padding_mask=~concat_mask)

        return fused_feat, concat_mask, position_ids
    

    def _build_position_ids(self, road_max, dynamic_max, static_max, refpath_max, batch_size, device):
        template = torch.cat([
            torch.full((road_max,), SceneFeatureIDX.ROAD,),
            torch.full((dynamic_max,), SceneFeatureIDX.DYNAMIC,),
            torch.full((static_max,), SceneFeatureIDX.STATIC,),
            torch.full((refpath_max,), SceneFeatureIDX.REF_PATH,),
            torch.tensor([SceneFeatureIDX.EGO])
        ]).to(device)
        return template.unsqueeze(0).expand(batch_size, -1)
    
    def count_parameters(self):
        """统计模型参数数量
        Returns:
            tuple: (总参数数量, 活动参数数量)
        """
        total = sum(p.numel() for p in self.parameters())
        active = total  # 对于普通编码器，所有参数都是活动的
        return total, active

class EgoStateEncoder(nn.Module):
    """Encodes ego vehicle's dynamic states with temporal Mixer over history + current."""
    def __init__(self,
                 hidden_dim: int = 256,
                 history_len: int = 15,
                 num_layers: int = 2,
                 time_embed_dropout: float = 0.0,
                 drop_path_rate: float = 0.0):
        super().__init__()
        # Per-timestep feature: x, y, cos(yaw), sin(yaw), v, a, r, half_width, half_length, rear_to_center
        self.state_encoder = Mlp(10, hidden_dim, hidden_dim)
        self.temporal_blocks = nn.ModuleList([
            MixerBlock(dim=hidden_dim, seq_len=history_len + 1, drop_path=drop_path_rate)
            for _ in range(num_layers)
        ])
        self.time_embed = nn.Embedding(history_len + 1, hidden_dim)
        self.time_drop = nn.Dropout(time_embed_dropout)
        self.norm = LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.history_len = history_len

    def forward(self, ego_data: EgoFeature) -> Tensor:
        """
        Args:
            ego_data.ego_history_state: [B, T, 6] (x, y, yaw, v, a, r)
            ego_data.ego_current_state: [B, 6]
            ego_data.ego_geometry:      [B, 3] (half_width, half_length, rear_to_center)
        Returns:
            Tensor [B, hidden_dim]
        """
        hist = ego_data.ego_history_state
        curr = ego_data.ego_current_state.unsqueeze(1)          # [B, 1, 6]
        all_state = torch.cat([hist, curr], dim=1)              # [B, T+1, 6]
        B, T1, _ = all_state.shape

        # Build per-timestep features
        xy = all_state[..., :2]
        yaw = all_state[..., 2:3]
        v_a_r = all_state[..., 3:6]
        geom = ego_data.ego_geometry.unsqueeze(1).expand(-1, T1, -1)  # [B, T+1, 3]
        feats = torch.cat([xy, torch.cos(yaw), torch.sin(yaw), v_a_r, geom], dim=-1)  # [B, T+1, 10]

        x = self.state_encoder(feats)  # [B, T+1, H]

        # Time embedding
        t_idx = torch.arange(T1, device=x.device, dtype=torch.long)
        t_pe = self.time_embed(t_idx).to(dtype=x.dtype).view(1, T1, -1).expand(B, T1, -1)
        x = x + self.time_drop(t_pe)

        # Temporal mixing
        for blk in self.temporal_blocks:
            x = blk(x)

        # Mean pool over time
        x = x.mean(dim=1)  # [B, H]
        x = self.proj(self.norm(x))
        return x
    

class AgentEncoder(nn.Module):
    """Dynamic obstacle encoder with mixer-based temporal aggregation."""
    def __init__(self, 
                 feat_dim=9,
                 agent_type_num=4, # 3 types + 1 invalid
                 d_model=256,
                 num_layers=2,
                 history_len=15,
                 time_embed_dropout: float = 0.0):
        super().__init__()

        self.history_len = history_len
        self.state_encoder = Mlp(9, d_model, d_model)  # per-timestep feature -> d_model

        # Mixer blocks for temporal mixing (T+1 tokens, dim=d_model)
        self.temporal_blocks = nn.ModuleList([
            MixerBlock(dim=d_model, seq_len=history_len + 1)
            for _ in range(num_layers)
        ])

        # Time embedding
        self.time_embed = nn.Embedding(history_len + 1, d_model)
        self.time_drop = nn.Dropout(time_embed_dropout)

        # Agent type embedding and fusion
        self.type_embed = nn.Embedding(agent_type_num, d_model)
        self.norm = LayerNorm(d_model)
        self.fusion_proj = nn.Linear(d_model, d_model)

    def forward(self, batch_data: AgentFeature):
        B, max_agents, T, _ = batch_data.agent_history_state.shape
        assert T == self.history_len, f"AgentEncoder history_len mismatch: data T={T}, configured history_len={self.history_len}"
        device = batch_data.agent_history_state.device
        agent_mask = batch_data.agent_mask  # [B, N]
        flat_mask = agent_mask.view(-1)     # [B*N]
        valid_idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)
        num_valid = valid_idx.numel()

        if num_valid == 0:
            empty = torch.zeros(B, max_agents, self.fusion_proj.out_features, device=device,
                                dtype=batch_data.agent_history_state.dtype)
            return empty, agent_mask


        # Build per-timestep features (x, y, cos, sin, v, a, r, half_length, half_width)
        curr_state = batch_data.agent_current_state                              # [B, N, 6]
        history_state = batch_data.agent_history_state                           # [B, N, T, 6]
        geom_state = batch_data.agent_geometry                                   # [B, N, 2]
        all_state = torch.cat((history_state, curr_state.unsqueeze(2)), dim=2)   # [B, N, T+1, 6]
        all_state = torch.cat((
            all_state[..., :2],
            torch.cos(all_state[..., 2:3]),
            torch.sin(all_state[..., 2:3]),
            all_state[..., 3:6],
            geom_state.unsqueeze(2).expand(-1, -1, all_state.shape[2], -1)
        ), dim=-1)  # [B, N, T+1, 9]
        all_state = all_state.view(B * max_agents, T + 1, -1)  # [B*N, T+1, 9]
        all_state = all_state[valid_idx]  # [num_valid, T+1, 9]

        # Encode per timestep
        state_emb = self.state_encoder(all_state)  # [num_valid, T+1, D]

        # Time embedding (T+1)
        T1 = state_emb.size(1)
        t_idx = torch.arange(T1, device=device, dtype=torch.long)
        time_pe = self.time_embed(t_idx).to(dtype=state_emb.dtype)  # [T+1, D]
        time_pe = time_pe.unsqueeze(0).expand(num_valid, -1, -1)    # [V, T+1, D]
        state_emb = state_emb + self.time_drop(time_pe)

        # Temporal mask: history + current
        hist_mask = batch_data.agent_history_mask                                # [B, N, T]
        temporal_mask = torch.cat(
            (hist_mask, torch.ones((B, max_agents, 1), device=device, dtype=torch.bool)),
            dim=2
        )  # [B, N, T+1]
        temporal_mask = temporal_mask.view(B * max_agents, T1)[valid_idx]  # [num_valid, T+1]

        # Zero out invalid timesteps before Mixer
        state_emb = state_emb * temporal_mask.unsqueeze(-1)

        # Mixer over time
        x = state_emb
        tmask = temporal_mask
        for block in self.temporal_blocks:
            x = block(x)
            x = x * tmask.unsqueeze(-1)  # keep padded steps zeroed

        # Masked mean pooling over time
        tmask_f = tmask.unsqueeze(-1).float()
        numer = (x * tmask_f).sum(dim=1)
        denom = tmask_f.sum(dim=1).clamp(min=1e-6)
        pooled = numer / denom  # [num_valid, D]


        # Type embedding and fusion
        agent_type = batch_data.agent_type.view(B * max_agents)[valid_idx]  # [num_valid]
        type_emb = self.type_embed(torch.clamp(agent_type, max=self.type_embed.num_embeddings - 1))
        fused_valid = self.fusion_proj(self.norm(pooled + type_emb))

        # Scatter back to padded layout
        fused_full = torch.zeros(B * max_agents, fused_valid.shape[-1],
                                 device=device, dtype=fused_valid.dtype)
        fused_full[valid_idx] = fused_valid
        fused_full = fused_full.view(B, max_agents, -1)

        return fused_full, agent_mask
    
class RoadEncoder(nn.Module):
    """Batch-processable road encoder with fixed-length input support"""
    def __init__(self, 
                 hidden_dim=256, 
                 type_dim=64,
                 road_type_num=len(RoadType), # 8 types + 1 invalid,
                 traffic_light_num=5, # 4 types + 1 invalid
                 max_points=20, 
                 max_edges=20,
                 depth=2,
                 drop_path_rate=0.):
        super().__init__()
        # lane branch
        # self.lane_pre_token = Mlp(max_points, tokens_mlp_dim, tokens_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.lane_pre_channel = Mlp(4, hidden_dim, hidden_dim, act_layer=nn.GELU, drop=0.)
        self.lane_blocks = nn.ModuleList([
            MixerBlock(dim=hidden_dim, seq_len=max_points, drop_path=drop_path_rate) 
            for _ in range(depth)
            ])

        # region branch
        # self.region_pre_token = Mlp(max_edges, tokens_mlp_dim, tokens_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.region_pre_channel = Mlp(2, hidden_dim, hidden_dim, act_layer=nn.GELU, drop=0.)
        self.region_blocks = nn.ModuleList([
            MixerBlock(dim=hidden_dim, seq_len=max_edges, drop_path=drop_path_rate) 
            for _ in range(depth)
            ])

        # shared attrs + fusion
        self.type_embed = nn.Embedding(road_type_num, type_dim)
        self.tl_embed = nn.Embedding(traffic_light_num, type_dim)
        self.speed_embed = nn.Sequential(nn.Linear(1, type_dim), nn.ReLU())
        self.norm = LayerNorm(hidden_dim + type_dim*3)
        self.fusion = Mlp(hidden_dim + type_dim*3, 4*hidden_dim, hidden_dim)

    def _encode_lane(self, center_line, mask):
        """
        center_line: Tensor [B, N, max_points, 3] (x, y, heading)
        mask: Tensor [B, N] (validity mask for lanes)
        """
        B, N, max_points, _ = center_line.shape
        center_line_xy = center_line[..., :2]  # [B, N, max_points, 2]
        center_line_yaw = center_line[..., 2:3]  # [B, N, max_points, 1]
        center_line_feat = torch.cat(
            [center_line_xy, 
             torch.sin(center_line_yaw), 
             torch.cos(center_line_yaw)], 
             dim=-1
             )  # [B, N, max_points, 4]
        
        center_line_feat = center_line_feat.view(B*N, max_points, -1)
        # MLP-Mixer encoding
        center_line_feat = center_line_feat[mask.view(B*N)] # [num_valid, max_points, 4]
        x = self.lane_pre_channel(center_line_feat)  # [num_valid, max_points, hidden_dim]
        for block in self.lane_blocks:
            x = block(x)
        geom = torch.mean(x, dim=1)  # [num_valid, hidden_dim]
        geom = geom.to(center_line.dtype)
        geom_full = torch.zeros(B*N, geom.shape[-1], device=center_line.device, dtype=center_line.dtype)
        geom_full[mask.view(B*N)] = geom
        geom = geom_full.view(B, N, -1)
        
        return geom

    def _encode_roadblock(self, polygon, mask):
        B, N, max_edges, _ = polygon.shape
        polygon_xy = polygon[..., :2]  # [B, N, max_edges, 2]
        polygon_feat = polygon_xy.view(B*N, max_edges, -1)
        # MLP-Mixer encoding
        polygon_feat = polygon_feat[mask.view(B*N)]
        x = self.region_pre_channel(polygon_feat)  # [num_valid, max_edges, hidden_dim]
        for block in self.region_blocks:
            x = block(x)
        geom = torch.mean(x, dim=1)  # [num_valid, hidden_dim]
        geom = geom.to(polygon.dtype)
        geom_full = torch.zeros(B*N, geom.shape[-1], device=polygon.device, dtype=polygon.dtype)
        geom_full[mask.view(B*N)] = geom
        geom = geom_full.view(B, N, -1)

        return geom

    def forward(self, road_data: RoadFeature):
        lane_mask = (road_data.road_type == 1) | (road_data.road_type == 2)
        region_mask = road_data.road_mask & (~lane_mask)

        lane_feat = self._encode_lane(road_data.center_line, lane_mask)
        region_feat = self._encode_roadblock(road_data.road_geometry, region_mask)

        geom = lane_feat + region_feat  # [B, N, hidden_dim]
        type_emb = self.type_embed(road_data.road_type)
        tl_emb = self.tl_embed(road_data.road_traffic_light)
        speed_emb = self.speed_embed(road_data.road_speed_limit.unsqueeze(-1))

        fused = torch.cat([geom, type_emb, tl_emb, speed_emb], dim=-1)
        fused = self.fusion(self.norm(fused))
        fused = torch.where(road_data.road_mask.unsqueeze(-1), fused, torch.zeros_like(fused))
        return fused, road_data.road_mask
    
class StaticObstacleEncoder(nn.Module):
    """
    Batch-processable static obstacle encoder with tensor padding
    Processes batched input with variable number of obstacles per sample
    """
    
    def __init__(self,
                 hidden_dim=256,
                 num_type_obstacles=5,  # 4 types + 1 invalid
                 ):
        super().__init__()
        
        # Core feature encoders
        self.encoder = Mlp(6, 2*hidden_dim, hidden_dim)  # x, y, sin(heading), cos(heading), width, length
        self.hidden_dim = hidden_dim
        self.type_embed = nn.Embedding(num_type_obstacles, hidden_dim)  # different agent types
        

    def forward(self, batch_obstacles: StaticObstacleFeature) -> dict:
        """
        Args:
            batch_obstacles (StaticObstacleFeature): Batched static obstacle data containing:
                - static_obstacle_position: Tensor [B, N, 3] (x, y, heading)
                - static_object_dimension: Tensor [B, N, 2] (width, length)
                - static_obstacle_mask: Tensor [B, N] (validity mask for each obstacle)
        Returns:
            dict: Contains either element-wise features or global context
        """
        B, N = batch_obstacles.static_obstacle_position.shape[:2]
        if N == 0:
            empty = torch.zeros(B, 0, self.hidden_dim, 
                                device=batch_obstacles.static_obstacle_position.device, 
                                dtype=batch_obstacles.static_obstacle_position.dtype)
            return empty, batch_obstacles.static_obstacle_mask
        
        # Encode features
        pos_feat = batch_obstacles.static_obstacle_position  # [B, N, 3]
        pos_feat = torch.cat([
            pos_feat[..., :2],  # x, y
            torch.sin(pos_feat[..., 2:3]),  # sin(heading)
            torch.cos(pos_feat[..., 2:3])   # cos(heading)
        ], dim=-1) # [B, N, 4]
        size_feat = batch_obstacles.static_object_dimension    # [B, N, 2]
        all_feat = torch.cat([pos_feat, size_feat], dim=-1)  # [B, N, 6]
        fused_feature = self.encoder(all_feat)                # [B, N, hidden_dim]
        # Type embedding
        type_embed = self.type_embed(batch_obstacles.static_object_type)
        
        fused_feature = fused_feature + type_embed

        # mask out invalid obstacles
        fused_feature = torch.where(batch_obstacles.static_obstacle_mask.unsqueeze(-1), 
                                fused_feature, 
                                torch.zeros_like(fused_feature))

        # Element-wise features
        return  fused_feature, batch_obstacles.static_obstacle_mask

class RouteEncoder(nn.Module):
    """
    Encode an ordered list of route polygons.
    route_geometry: [B, R, max_edges, 2] (x, y) per edge (padded)
    route_mask:     [B, R] bool, valid polygons
    """
    def __init__(self,
                 hidden_dim: int = 256,
                 max_edges: int = 20,
                 max_routes: int = 10,
                 depth_geom: int = 2,
                 depth_route: int = 1,
                 drop_path_rate: float = 0.0):
        super().__init__()
        # Per-polygon edge encoding
        self.edge_proj = Mlp(2, hidden_dim, hidden_dim, act_layer=nn.GELU, drop=0.)
        self.geom_blocks = nn.ModuleList([
            MixerBlock(dim=hidden_dim, seq_len=max_edges, drop_path=drop_path_rate)
            for _ in range(depth_geom)
        ])
        # Route-level order-aware mixing over polygon tokens
        self.route_blocks = nn.ModuleList([
            MixerBlock(dim=hidden_dim, seq_len=max_routes, drop_path=drop_path_rate)
            for _ in range(depth_route)
        ])
        self.max_routes = max_routes
        self.max_edges = max_edges

    def forward(self, route_data: RouteFeature):
        route_geom = route_data.route_geometry          # [B, R, E, 2]
        route_mask = route_data.route_mask              # [B, R] bool
        B, R, E, _ = route_geom.shape
        device = route_geom.device

        # Guard shape
        assert E <= self.max_edges, f"route_edges {E} > max_edges {self.max_edges}, route_geom shape: {route_geom.shape}"
        # assert R <= self.max_routes, f"route_len {R} > max_routes {self.max_routes}, route_geom shape: {route_geom.shape}"

        # Flatten polygons, keep only valid for efficiency
        flat_mask = route_mask.view(B * R)              # [B*R]
        geom_flat = route_geom.view(B * R, E, -1)       # [B*R, E, 2]
        valid_geom = geom_flat[flat_mask]               # [V, E, 2]

        # Per-polygon encoding
        x = self.edge_proj(valid_geom)                  # [V, E, H]
        for blk in self.geom_blocks:
            x = blk(x)
        poly_feat = x.mean(dim=1)                       # [V, H]
        poly_feat = poly_feat.to(route_geom.dtype)

        # Scatter back to padded layout
        poly_full = torch.zeros(B * R, poly_feat.size(-1), device=device, dtype=poly_feat.dtype)
        poly_full[flat_mask] = poly_feat
        poly_full = poly_full.view(B, R, -1)            # [B, R, H]

        # Zero invalid tokens before route-level mixer
        poly_full = poly_full * route_mask.unsqueeze(-1)

        # Route-level order-aware mixing
        # Pad to max_routes if needed
        if R < self.max_routes:
            pad_len = self.max_routes - R
            poly_full = F.pad(poly_full, (0, 0, 0, pad_len))            # [B, max_routes, H]
            route_mask = F.pad(route_mask, (0, pad_len))
        elif R > self.max_routes:
            # Trim to max_routes
            poly_full = poly_full[:, :self.max_routes, :]
            route_mask = route_mask[:, :self.max_routes]

        for blk in self.route_blocks:
            poly_full = blk(poly_full)
            poly_full = poly_full * route_mask.unsqueeze(-1)

        # Trim back to original R length
        poly_full = poly_full[:, :R]
        route_mask = route_mask[:, :R]

        return poly_full, route_mask
