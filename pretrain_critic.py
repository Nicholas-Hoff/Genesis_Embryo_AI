import duckdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from genesis_embryo_core import Critic
from logging_config import configure_logging

logger = logging.getLogger(__name__)

# ←— point at the merged DB you just built ——
#DB_PATH         = 'godseed_training.db'
DB_PATH         = 'godseed_aggressive_full.db'
PRETRAIN_EPOCHS = 10
BATCH_SIZE      = 128
LR              = 1e-3
GAMMA           = 0.9
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_transitions(db_path=DB_PATH):
    con = duckdb.connect(db_path)
    df = con.execute("""
      -- pull in every transition, sorted by timestamp
      SELECT
        hb,
        [survival_before, novelty_before, efficiency_before, mutation_error_before, cycle_before]   AS state,
        reward,
        [survival_after,  novelty_after,  efficiency_after,  mutation_error_after,  cycle_after]    AS next_state,
        ts
      FROM transitions
      WHERE next_state IS NOT NULL
      ORDER BY ts
    """).fetchdf()
    con.close()


    # ensure arrays are simple Python lists for compatibility
    df["state"] = df["state"].apply(list)
    df["next_state"] = df["next_state"].apply(list)

    # ensure arrays are simple Python floats for compatibility
    df["state"] = df["state"].apply(lambda arr: [float(str(x)) for x in arr])
    df["next_state"] = df["next_state"].apply(lambda arr: [float(str(x)) for x in arr])


    # sanity-check
    logger.info(f"Loaded {len(df)} transitions")
    logger.info(f"time span: {df.ts.min()} -> {df.ts.max()}")
    return df

def pretrain_critic(df, critic, optimizer,
                    epochs=PRETRAIN_EPOCHS,
                    batch_size=BATCH_SIZE,
                    gamma=GAMMA,
                    device=DEVICE):

    states_np = np.stack(df['state'].values).astype(np.float32)  # shape: (N, 5)

    states    = torch.tensor(states_np, dtype=torch.float32, device=device)
    rewards     = torch.tensor(df['reward'].values,        dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(df['next_state'].tolist(),  dtype=torch.float32, device=device)

    dataset = torch.utils.data.TensorDataset(states, rewards, next_states)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for s, r, ns in loader:
            optimizer.zero_grad()
            with torch.no_grad():
                target = r + gamma * critic(ns)
            pred = critic(s)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * s.size(0)
        avg_loss = total_loss / len(loader.dataset)
        logger.info(f'[Pretrain] Epoch {epoch}/{epochs}  Loss={avg_loss:.6f}')
    return critic

def main():
    configure_logging()
    df = load_transitions()
    critic    = Critic(input_dim=5, hidden=32)
    optimizer = optim.Adam(critic.parameters(), lr=LR)

    critic = pretrain_critic(df, critic, optimizer)
    torch.save(critic.state_dict(), 'critic_pretrained.pt')
    logger.info('Saved pretrained weights to critic_pretrained.pt')

if __name__ == '__main__':
    main()
