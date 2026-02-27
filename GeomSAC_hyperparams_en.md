## Geom-SAC Tunable Parameters (English)

Translation of the parameter summary table in section 10 of `GeomSAC_tong_quan_vi.md`.

| Parameter | Group | Location (file) | Default / Example Value | Short description |
|----------|-------|------------------|-------------------------|-------------------|
| **Input SMILES set** | Chemistry | `main.py` | List of 1 SMILES | Pool of starting molecules for each episode. |
| **`reference_mol`** | Chemistry | `main.py` | 1 Mol (from SMILES) | Reference molecule used to compute similarity. |
| **`target_sim`** | Chemistry | `main.py` → env | `1` | Target similarity level with the reference molecule. |
| **`allowed_atoms`** | Chemistry | `MolGraphEnv.py` | `["C","Cl","F","I","K","N","Na","O","S","Br"]` | Atom types allowed to be added to molecules. |
| **`max_atom`** | Chemistry | `MolGraphEnv.py` / `main.py` | `35` (env), `40` (main) | Maximum number of atoms allowed in a molecule. |
| **`max_action`** | Chemistry / Pipeline | `MolGraphEnv.py` | `130` | Maximum number of steps per episode; also threshold for early stop when `invalid_actions` is too large. |
| **`min_action`** | Chemistry | `MolGraphEnv.py` | `21` | Stored in env; can be used to enforce a minimum number of steps. |
| **`reward_type`** | Chemistry | `MolGraphEnv.py` | `"qed"` | Final reward type (currently QED only). |
| **Reward coefficients (step & final episode)** | Config | `MolGraphEnv.py` | Several constants (valency ±1/max_action, QED terms, reward_valid/geom/qed, similarity) | Balance between validity, QED and similarity in the reward function. |
| **`frame_work`** | Pipeline | `MolGraphEnv.py`, `main.py` | `'pyg'` | Graph backend: `'pyg'` (PyTorch Geometric) or `'dgl'`. |
| **`n_episodes`** | Pipeline | `main.py` | `5000` | Number of training episodes. |
| **QED threshold for `top`** | Pipeline | `main.py` | `0.79` | Only keeps molecules with QED above this threshold in `top`. |
| **`gamma`** | ML/RL – Agent | `agent.py` | `0.99` | Discount factor. |
| **`tau`** | ML/RL – Agent | `agent.py` | `0.005` | Soft-update factor for the target value network. |
| **`batch_size`** | ML/RL – Agent | `agent.py`, `buffer.py` | `32` | Number of samples per training update from the replay buffer. |
| **`reward_scale`** | ML/RL – Agent | `agent.py` | `10` | Multiplier applied to rewards when computing Q targets. |
| **`lr_actor`, `lr_v`, `lr_q1`, `lr_q2`** | ML/RL – Agent | `agent.py` | `0.003` | Learning rates for the actor and critic networks. |
| **`maxlen` (replay buffer)** | ML/RL – Agent | `buffer.py` | `500` | Maximum number of transitions stored in the replay buffer. |
| **`n_layers`** (GAT/GIN) | ML/RL – GAT/GIN | `neural_networks.py` (GraphEncoder) | `1` | Number of GAT and GIN layers. |
| **`dim_h`** | ML/RL – GAT/GIN | `neural_networks.py` | `128` | Hidden dimension for GAT, GIN and MLPs. |
| **`heads`** (GAT) | ML/RL – GAT | `neural_networks.py` (GraphEncoder) | `4` | Number of attention heads in GAT. |
| **`dim_hidden` (MLP V, Q)** | ML/RL – Agent (V/Q) | `neural_networks.py` (StateValueNetwork, ActionValueNetwork) | `64` | Hidden size of the second layer in V/Q MLPs (128 → 64 → 1). |
| **`epsilon`** (policy sample) | ML/RL – Agent (Policy) | `neural_networks.py` (PolicyNetwork.sample) | `1e-6` | Small constant for numerical stability when sampling actions (avoid log 0). |
| **`chkpt_dir`** | Config | `neural_networks.py` | `'tmp/GeomSac'` | Directory for saving actor / value / critic checkpoints. |
| **`device`** (replay buffer) | Config | `buffer.py` | `'cpu'` | Device used to store and stack tensors in `ReplayBuffer.sample()`. |
| **Number of molecules printed (summary)** | Config | `main.py` | `5` | Number of SMILES shown for `mols` and `top` at the end of a run (`mols[:5]`, `unique_top[:5]`). |

