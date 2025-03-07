python run_test.py  --config-name=robocasa_horeka_config \
            --multirun agents=fm_agent \
            agent_name=fm_transformer \
            group=fm_decoder_only_benchmark \
            agents/model=fm/fm_dec_transformer \
            encoder_n_layer=6 \
            seed=0,1,2