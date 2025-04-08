from src.algo.gSASRec.config import gSASRecExperimentConfig

config = gSASRecExperimentConfig(
    name = "Electronics",
    sequence_len = 40,
    num_sequence = 10,
    least_interaction = 5,
    dataset="electronics",
    train_dir='model',
    dataset_dir='dataset',
    batch_size=128,
    lr=0.001,
    maxlen=200,
    hidden_units=50,
    num_blocks=1,
    num_epochs=100,
    num_heads=2,
    num_workers=1,
    num_negs=256,
    gbce_t = 0.75,
    dropout_rate=0.5,
    l2_emb=0.0,
    inference_only=False,
    state_dict_path=None,
    sampling_method='normal', ## "normal" or "graph" model or "hybrid" 
    pe_method = "normal" ## position encoding method, "normal" or "cape"
)