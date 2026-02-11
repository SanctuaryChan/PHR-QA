python script/build_sbert_embeddings.py \
  --data_dir /data/GNN-RAG/datasets/webqsp \
  --model all-mpnet-base-v2 \
  --dim 100 \
  --batch_size 256 \
  --pca_batch_size 2048 \
  --device cuda \
  --entity_out entity_emb_sbert_100d.npy \
  --relation_out relation_emb_sbert_100d.npy