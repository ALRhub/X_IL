python run_test.py  --config-name=robocasa_horeka_config \
            --multirun agents=fm_agent \
            agent_name=fm_mamba \
            group=fm_decoder_only_benchmark \
            agents/model=fm/fm_dec_mamba \
            mamba_n_layer_encoder=10 \
            seed=0,1,2