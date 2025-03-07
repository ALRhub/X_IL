python run.py  --config-name=libero_horeka_config \
            --multirun agents=bc_agent \
            agent_name=bc_mamba \
            group=bc_decoder_only \
            agents/model=bc/bc_dec_mamba \
            task_suite=libero_object,libero_goal,libero_10,libero_spatial \
            traj_per_task=10,50 \
            seed=0,1,2