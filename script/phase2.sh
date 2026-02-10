python src/run.py phase2 \
  --data_dir /data/GNN-RAG/datasets/webqsp \
  --split dev \
  --reader hf \
  --model_path /data/GNN-RAG/models/Llama-2-7b-chat-hf \
  --topn 50 \
  --save_evidence \
  --save_prompt \
  --progress