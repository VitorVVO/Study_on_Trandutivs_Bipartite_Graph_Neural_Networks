# Study on Bipartite Graph Neural Networks with Keyphrase and Attention for Transductive Text Classification

  

This repository contains the codes used to carry out the final master's work of PPGI/UND: "Study on Bipartite Graph Neural Networks with Keyphrase and Attention for Transductive Text Classification", by student VItor Vasonccelos de Oliveira.

  

## Abstract:

In contemporary times, Natural Language Processing (NLP) has swiftly evolved in a wide range of tasks, especially thanks to Machine Learning (ML), and Deep Learning (DL) great advancements over the years. However, due to these technologies’ complexity and data prerequisites, current conventional NLP text classification methodologies often require large numbers of labeled documents and large computational power. This paper mainly investigates three techniques to address such challenges. Firstly and most significantly, the use of transductive graph-based approaches for the text classification task aims to reduce the amount of required labeled data. For this initial process, we employ the classic and well-established Graph Convolutional Networks (GCN) and the more contemporary Graph Attention Networks (GAT), on a novel document-concept bipartite graph framework that uses Keyphrase(concepts) for topic knowledge acquisition and model in formation enrichment. The second technique is applying coarsening for graph reduction, hence reducing computational costs. Lastly, we aim to employ Large Language Models (LLM) as low-cost labelers effectively removing or reducing the need for human labelers. Results show GAT as the best performing model for transductive text classification tasks using the document-concept bipartite graph approach, GAT showed that it can perform on equal levels to traditional inductive models despite using only 1 to 30 labeled documents per class. The coarsening application presented 40%-50% graph size reduction while maintaining 82% of the model performance at average, ranging from 68% to 95% on various datasets. LLMs were able to train several efficient models, but compared to models trained on human-labeled data revealed inferior results, demonstrating that transductive learning favors small amounts of highly accurate data rather than a large quantity of moderately accurate data.

  
  

# Index
* **Datset_Preparation**: contains code for dataset preprocessing, such as applying BERT, KeyBERT and, sentence tokenization for KNN, and fixing class name in the case of some datasets.
	* **Generating_processed_datasets**: Applies BERT, KeyBERT generating the processed datasets.
	* **Adjusting_Sentences_for_KNN**: Applies sentence tokenization and BERT for KNN application and in the future improving graph connectivity.
	* **Fixing_dataset_classes**: fixes class names for some datasets which were incorrect.
* **Traditional**: application and training of the traditional GAT and GCN models.
	* **Create_masks_and_train_Traditional**: Create the "masks" used on all experiments as the selection of training and testing data. After the creation of masks, also trains all traditional GAT and GCN models for all datasets, keyphrases, and number of human-labeled data combinations.
	* **80_60_40_20_Traditional**: Create masks and trains traditional GAT and GCN models with 20%, 40%, 60%, and 80% human-labeled data.
* **Coarsening**: Applies the CLPK coarsening algorithm and train models on coarsened graphs.
	* **Generating_graphs_to_CLPK**: Generates and saves graph structure, based on mask previously defined on **Create_masks_and_train_Traditional**, for future CLPK coarsening algorithm application.
	* **apply_clpk**: Applies CLPK coarsening algorithm, effectively reducing graph nodes and edges by 40% to 50%.
	* **Creating_Embeddings_and_Training_to_CLPK**: Applies BERT on the concatenated super nodes texts, and subsequently trains coarsened GAT and GCN models.
* **LLM**: Applies LLM based labeling  and train models on LLM-labeled data.
	* **Generating_graphs_to_LLM**: Generates and saves graph structure, based on mask previously defined on **Create_masks_and_train_Traditional**, for future LLM labeling application.
	* **LLM_labeling**: Applies Llama 3.1 8B to label 10, 50, and 100 documents for each dataset.
	* **training_llm**: Trains GAT and GCN models combined the LLM-labeled documents with the human-labeled documents.
	* **create_mask_LLM_only_models**: Creates mask for training and testing models using only LLM-labeled data
	* **training_llm_only_models**: Trains GAT and GCN models using only LLM-labeled data.
* **Results**: Code used to generate the results presented on text, chart, tables, and metrics.
	* **Average_and_Std**: Calculates the average and standard deviation of all metrics recorded for all trained models.
	* **Create_tables**: Creates F1-Score tables used in the study's text.
	* **Create_charts**:  Creates F1-Score charts used in the study's text.
	* **CD_graphs**: Creates Critical Difference figures used in the study's text.
	* **CONF_coarsenig**: Creates the coarsening reduction table used in the study's text.
