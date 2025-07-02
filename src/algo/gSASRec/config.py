import torch


class gSASRecExperimentConfig():
    """
    Configuration class for gSASRecExperiment.
    """
    def __init__(self,name = "Amazon",
                categories = ["Electronics"],
                sequence_len = 40,
                num_sequence = 10,
                output_graph_sequence = "dataset/sequences-graph.txt",
                output_normal_sequence = "dataset/sequences-normal.txt",
                least_interaction = 5,
                dataset="electronics",
                train_dir='model',
                dataset_dir='dataset',
                batch_size=512,
                lr=0.001,
                maxlen=200,
                hidden_units=50,
                num_blocks=1,
                num_epochs=100,
                num_heads=1,
                num_workers=0,
                num_negs=1,
                dropout_rate=0.2,
                l2_emb=0.0,
                gbce_t = 0.5,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                inference_only=False,
                state_dict_path=None,
                sampling_method='normal', ## "normal" or "graph" model or "hybrid" 
                pe_method = "normal" ## position encoding method, "normal" or "cape"
                ):
        
        self.name = name
        self.categories = categories
        self.sequence_len = sequence_len
        self.num_sequence = num_sequence
        self.output_graph_sequence = output_graph_sequence
        self.output_normal_sequence = output_normal_sequence
        self.least_interaction = least_interaction
        self.dataset = dataset
        self.train_dir = train_dir
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.lr = lr
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_epochs = num_epochs
        self.num_heads = num_heads
        self.num_workers = num_workers
        self.num_negs = num_negs
        self.dropout_rate = dropout_rate
        self.gbce_t = gbce_t
        self.l2_emb = l2_emb
        self.device = device
        self.inference_only = inference_only
        self.state_dict_path = state_dict_path if state_dict_path is not None else f"model/{self.name}.pt"
        self.sampling_method = sampling_method 
        self.pe_method = pe_method
        self.model = None
