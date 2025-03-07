python run_test.py  --config-name=robocasa_horeka_config \
            --multirun agents=fm_agent \
            agent_name=fm_mamba \
            group=fm_encoder_decoder_benchmark \
            agents/model=fm/fm_encdec_mamba \
            mamba_n_layer_encoder=4 \
            mamba_n_layer_decoder=8 \
            seed=0,1,2