import logging

from MolGraphEnv import *
from agent import *
from utils import get_final_mols

if __name__ == "__main__":
    # Basic logging configuration; adjust level to DEBUG for more detail.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger = logging.getLogger("GeomSAC")

    scores = []
    mols = []
    top = []
    actor_loss = []
    n_episodes = 100
    ref_mol = Chem.MolFromSmiles('CC(=CCC1=CC2=C(C(=C1OC)O)C(=O)C3=C(O2)C=CC(=C3)O)C')

    logger.info("Starting training for %d episodes", n_episodes)

    for episode in range(n_episodes):
        rn = np.random.choice(["O=C1CSC(=Nc2cc(F)ccc2Cc2cncs2)N1", ])
        init_mol = Chem.MolFromSmiles(rn)
        logger.debug("Episode %d: initial SMILES %s", episode + 1, rn)

        env = MolecularGraphEnv(mol_g=init_mol, reference_mol=ref_mol, target_sim=1, max_atom=40)
        state = env.reset(frame_work='pyg')
        graph_encoder = GraphEncoder(state)
        state = graph_encoder(state)
        agent = SoftActorCriticAgent(env, state)
        rewards = 0
        done = False
        steps = 0

        logger.info("Episode %d: environment reset, starting interaction loop", episode + 1)

        while not done:
            steps += 1
            probabilities, actions, log_p = agent.select_actions(state)
            next_state, reward, done, info = env.step(actions[0].detach().cpu().numpy())
            graph_encoder_ = GraphEncoder(next_state)
            next_state = graph_encoder_(next_state)
            agent.replay_buffer.add(
                (state.detach().view(-1), probabilities.view(-1), reward, next_state.detach().view(-1), done,))
            agent.train()
            state = next_state
            rewards += reward

            logger.debug(
                "Episode %d, step %d: reward=%.4f, done=%s, cumulative_reward=%.4f",
                episode + 1,
                steps,
                float(reward),
                done,
                float(rewards),
            )

            if done:
                break

        scores.append(rewards)
        logger.info("Episode %d finished in %d steps with total reward %.4f", episode + 1, steps, float(rewards))
        actor_loss.append([x.item() for x in agent.ac_loss])

        try:
            final_smiles = Chem.MolToSmiles(env.get_final_mol())
            mols.append(final_smiles)
            logger.debug("Episode %d final molecule: %s", episode + 1, final_smiles)
        except Exception as e:
            logger.warning("Episode %d: failed to convert final molecule to SMILES (%s)", episode + 1, e)

        try:
            final_mol = env.get_final_mol()
            qed_final = Chem.QED.qed(final_mol)
            qed_largest_frag = Chem.QED.qed(get_final_mols(final_mol))
            if qed_final > 0.79:
                top.append(Chem.MolToSmiles(final_mol))
            if qed_largest_frag > 0.79:
                top.append(Chem.MolToSmiles(final_mol))
            logger.info(
                "Episode %d QED scores: final=%.4f, largest_fragment=%.4f",
                episode + 1,
                float(qed_final),
                float(qed_largest_frag),
            )
        except Exception as e:
            logger.warning("Episode %d: failed to compute QED/top molecules (%s)", episode + 1, e)
