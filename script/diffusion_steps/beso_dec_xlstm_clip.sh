python run_test.py  --config-name=robocasa_horeka_config \
            --multirun agents=beso_agent \
            agent_name=beso_xlstm \
            group=beso_decoder_only_clip \
            agents/model=beso/beso_dec_xlstm_rgbcasa \
            agents/obs_encoders=clip_img_encoder \
            xlstm_encoder_blocks=8 \
            seed=0,1,2