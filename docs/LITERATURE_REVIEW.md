# Integrated literature review

This study connects four architectural families under common text preprocessing. Prior disaster studies [1–5] establish the application context, while averaging,CNN,recurrent and attention foundations [6–12] explain the chosen representations. AdamW [13] motivates the optimization mechanism. Published scores use different labels,splits and pretraining and are not ranked against our measured results.

| Paper | Method | Data or task | Finding and relevance |
|---|---|---|---|
| [1] Balakrishnan et al. (2022) | Transformer representations with neural classifiers | Kaggle disaster tweets | Studies combinations of contextual representations and neural heads; motivates broader comparisons. Different split and pretrained representations prevent direct score comparisons. |
| [2] Dharrao et al. (2024) | BERT representations,CNN and optimizer comparison | Disaster tweet classification | Reports approximately0.83 accuracy and0.80 F1 for its proposed system. Its pretraining and metric protocol differ from our static-vector conditions. |
| [3] Nguyen et al. (2017) | CNN and cross-event training | Crisis-related social messages | Uses out-of-event labeled data to support a new crisis with scarce labels. Our random split is not an event-held-out replication. |
| [4] Kabir and Madria (2019) | Attention BiLSTM,CNN and auxiliary features | Harvey,Irma,CrisisNLP and CrisisLex | Combines text classification with rescue scheduling under resource constraints. Uses operational categories and auxiliary features beyond text-only binary classification. |
| [5] Alam et al. (2021) | Dataset and multiclass baselines | Approximately77K tweets from19 events | Provides a larger humanitarian benchmark and classic/fastText/Transformer baselines. Useful future evaluation; labels do not directly match the present binary task. |
| [6] Kim (2014) | Parallel convolution filters over word vectors | Sentence classification benchmarks | Shows that a shallow CNN over pretrained word vectors can be effective. Motivates phrase filters and max pooling; our data and tuning differ. |
| [7] Iyyer et al. (2015) | Deep averaging networks | Sentiment and question answering | Demonstrates a computationally simple unordered composition approach. Motivates pooled-embedding MLP; our mean+max classifier is a variant. |
| [8] Devlin et al. (2019) | Bidirectional Transformer pretraining | General NLP benchmarks | Shows the value of large-scale bidirectional language-model pretraining. Our encoder has no language-model pretraining; BERT is future transfer-learning work. |
| [9] Zhang and Wallace (2015,revised2016) | Controlled CNN hyperparameter analysis | Sentence classification benchmarks | Studies sensitivity to architecture and training choices. Supports validation-based search; does not prescribe optimal disaster-tweet settings. |
| [10] Hochreiter and Schmidhuber (1997) | Gated recurrent memory | Synthetic long-lag sequence tasks | Introduces memory cells and gates for long-term gradient flow. Our bidirectional LSTM uses a modern library and short tweet sequences. |
| [11] Vaswani et al. (2017) | Self-attention and positional encoding | Machine translation | Introduces attention-based encoder-decoder sequence modeling. We use only the encoder for classification; no translation decoder is present. |
| [12] Pennington,Socher and Manning (2014) | Global co-occurrence word representations | Large unlabeled text corpora | Provides pretrained static word vectors used as common inputs here. Static vectors cannot fully distinguish context-dependent word meanings. |
| [13] Loshchilov and Hutter (2019) | AdamW optimizer | Optimization/generalization benchmarks | Separates weight decay from adaptive gradient updates. Training mechanism reference,not a disaster-classification score baseline. |

## Sources

[1] Balakrishnan et al., A Comprehensive Analysis of Transformer-Deep Neural Network Models in Twitter Disaster Detection, Mathematics 10(24),4664, 2022. https://www.mdpi.com/2227-7390/10/24/4664
[2] Dharrao et al., An efficient method for disaster tweets classification using gradient-based optimized convolutional neural networks with BERT embeddings, MethodsX13,102843, 2024. https://doi.org/10.1016/j.mex.2024.102843
[3] Nguyen et al., Robust Classification of Crisis-Related Data on Social Networks Using Convolutional Neural Networks, ICWSM11(1),632–635, 2017. https://doi.org/10.1609/icwsm.v11i1.14950
[4] Kabir and Madria, A Deep Learning Approach for Tweet Classification and Rescue Scheduling for Effective Disaster Management, ACM SIGSPATIAL,269–278, 2019. https://arxiv.org/abs/1908.01456
[5] Alam et al., HumAID Human-Annotated Disaster Incidents Data from Twitter with Deep Learning Benchmarks, ICWSM15(1),933–942, 2021. https://doi.org/10.1609/icwsm.v15i1.18116
[6] Kim, Convolutional Neural Networks for Sentence Classification, EMNLP,1746–1751, 2014. https://aclanthology.org/D14-1181/
[7] Iyyer et al., Deep Unordered Composition Rivals Syntactic Methods for Text Classification, ACL-IJCNLP,1681–1691, 2015. https://aclanthology.org/P15-1162/
[8] Devlin et al., BERT Pre-training of Deep Bidirectional Transformers for Language Understanding, NAACL-HLT,4171–4186, 2019. https://aclanthology.org/N19-1423/
[9] Zhang and Wallace, A Sensitivity Analysis of and Practitioners Guide to Convolutional Neural Networks for Sentence Classification, arXiv1510.03820, 2015,revised2016. https://arxiv.org/abs/1510.03820
[10] Hochreiter and Schmidhuber, Long Short-Term Memory, Neural Computation9(8),1735–1780, 1997. https://www.bioinf.jku.at/publications/older/2604.pdf
[11] Vaswani et al., Attention Is All You Need, NeurIPS30, 2017. https://arxiv.org/abs/1706.03762
[12] Pennington,Socher and Manning, GloVe Global Vectors for Word Representation, EMNLP,1532–1543, 2014. https://aclanthology.org/D14-1162/
[13] Loshchilov and Hutter, Decoupled Weight Decay Regularization, ICLR2019, 2019. https://arxiv.org/abs/1711.05101