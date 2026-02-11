python src/build_sbert_embeddings.py \
  --data_dir /data/GNN-RAG/datasets/webqsp \
  --model /data/GNN-RAG/models/all-mpnet-base-v2 \
  --no_pca \
  --batch_size 256 \
  --device cuda \
  --entity_out entity_emb_sbert_768d.npy \
  --relation_out relation_emb_sbert_768d.npy \
  --offline
