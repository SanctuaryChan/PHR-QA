python src/build_question_embeddings.py \
  --data_dir /data/GNN-RAG/datasets/webqsp \
  --split dev \
  --model /data/GNN-RAG/models/all-mpnet-base-v2 \
  --device cuda \
  --batch_size 256 \
  --offline
