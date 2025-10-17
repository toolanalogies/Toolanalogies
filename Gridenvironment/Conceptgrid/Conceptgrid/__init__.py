from gym.envs.registration import register

register(
    id='ConceptGrid-v0',
    entry_point='Conceptgrid.envs:GridEnv',
)
register(
    id='ConceptGrid-v1',
    entry_point='Conceptgrid.envs:GridEnv2',
)
register(
    id='PDDLWriter-v0',
    entry_point='Conceptgrid.envs:PDDLWriter',
)
