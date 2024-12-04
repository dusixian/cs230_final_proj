# cs230_final_proj

## Generative Modeling of RNA Pairing Prediction with Biochemical Features

Accurate prediction and generation of RNA sequences are critical for understanding RNA editing mechanisms and designing functional sequences. In this study, we use a paired ADAR dataset to explore the potential of LSTM and Transformer models in predicting paired RNA sequence given one RNA sequence. We introduce a novel approach by embedding biochemical features into sequence representations to enhance model performance. Our findings suggest that LSTMs can serve as effective baselines for token-level prediction tasks, as it work well on short sequences, while Transformers require larger datasets or structural modifications to perform effectively. Furthermore, the integration of feature embeddings shows promise, offering potential for future work. 

## Files and Folders

- dataloader.py: Load the ADAR dataset.
- lstm.py: LSTM model, mentioned in the report. 
- transformer_greedy: Transformer model with greedy decoding, mentioned in the report. 
- transformer_hptuning: hyperparameter tuning for the transformer model. 
- transformer.py: Transformer model without greedy decoding, mentioned in the report. 

Folders: 

- data: ADAR dataset, in txt format.
- data_extraction: codes used to get the ADAR dataset
- data_preprocessing: divide and shuffle the dataset, tokenize the sequence and features
- model: saved model that are optimized
- other architectures: other model that are tested but not included in the report