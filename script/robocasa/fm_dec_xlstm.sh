python run_test.py  --config-name=robocasa_horeka_config \
            --multirun agents=fm_agent \
            agent_name=fm_xlstm \
            group=fm_decoder_only_benchmark \
            agents/model=fm/fm_dec_xlstm \
            seed=0,1,2