from envs.calvano_pricing import CalvanoPricingEnv

env = CalvanoPricingEnv(n=2, m=15, mu=0.25, max_steps=5, seed=0)
obs, info = env.reset()
for t in range(5):
    acts = {ag: env.action_spaces[ag].sample() for ag in env.agents}
    obs, rew, term, trunc, inf = env.step(acts)
    print(f"t={t+1} rewards={rew} prices={inf['firm_0']['prices']}")
