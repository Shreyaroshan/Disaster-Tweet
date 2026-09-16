# Literature review

These nine papers connect the simple baselines to crisis classification and contextual language models. Published numbers use different datasets, splits, preprocessing, and sometimes metric averaging. They are context, not directly comparable scores for this implementation. Qualitative findings are given where a single comparable numeric result is inappropriate.

| Paper | Method | Dataset or task | Key result | Relevance and limitation |
|---|---|---|---|---|
| [1] Balakrishnan et al. (2022), A Comprehensive Analysis of Transformer-Deep Neural Network Models in Twitter Disaster Detection | Transformer representations combined with neural classifiers | 7,613 Kaggle disaster tweets | Reports F-scores of 0.80–0.84 for simpler transformer variants, particularly with BiLSTM | Direct task match; motivates later contextual models. Pretraining and its 80/20 protocol differ from our from-scratch experiment. |
| [2] Dharrao et al. (2024), An efficient method for disaster tweets classification using gradient-based optimized convolutional neural networks with BERT embeddings | BERT embeddings, CNN, optimizer comparison | Disaster tweet classification dataset | BERT-CNN-RMSProp: reported macro F1 0.80 and validation accuracy 0.83 | Motivates CNN and optimization studies. Macro F1 is not our positive-class F1; embeddings and splits differ. |
| [3] Nguyen et al. (2017), Robust Classification of Crisis-Related Data on Social Networks Using Convolutional Neural Networks | CNN with transfer from other crisis events | Crisis-related social-network messages; event-based useful-tweet classification | Shows that out-of-event labeled data can support classification when a new event has no labels | Motivates local phrase features and future event-held-out tests. Useful-tweet labels differ from disaster/non-disaster labels. |
| [4] Kabir and Madria (2019), A Deep Learning Approach for Tweet Classification and Rescue Scheduling for Effective Disaster Management | Attention-based BiLSTM plus CNN, pretrained embeddings and auxiliary features | Hurricanes Harvey/Irma; four CrisisNLP and fifteen CrisisLex disaster datasets | Reports improvements over comparison classifiers; Nepal earthquake AUC 87.5 for CNN with attention and auxiliary features versus 84.8 for CNN comparison | Motivates recurrent/attention extensions. Uses metadata and rescue categories; it is not a Transformer-only study. |
| [5] Alam et al. (2021), HumAID Human-Annotated Disaster Incidents Data from Twitter with Deep Learning Benchmarks | Classical, fastText and Transformer multiclass benchmarks | Approximately 77,000 labeled tweets across 19 disaster events | Releases a larger human-annotated benchmark and multiclass baselines | Useful future external evaluation; humanitarian multiclass categories do not map directly to the present binary task. |
| [6] Kim (2014), Convolutional Neural Networks for Sentence Classification | One-layer CNN over pretrained word vectors, multiple filter widths, max pooling | Seven sentence-classification tasks including sentiment and questions | CNN variants improve the then state of the art on four of seven tasks | Architectural basis for our 3/4/5-token CNN; ours learns embeddings from scratch, so it is not a reproduction. |
| [7] Iyyer et al. (2015), Deep Unordered Composition Rivals Syntactic Methods for Text Classification | Deep averaging network over embeddings | Sentiment analysis and factoid question answering | An unordered neural model competes with syntactic models at lower training cost | Motivates masked mean embeddings plus MLP. Our one-hidden-layer baseline is a simplified variant without their word-dropout procedure. |
| [8] Devlin et al. (2019), BERT Pre-training of Deep Bidirectional Transformers for Language Understanding | Bidirectional Transformer pretraining and task fine-tuning | GLUE, MultiNLI, SQuAD and other NLP benchmarks | Reports GLUE 80.5 and MultiNLI accuracy 86.7% | Motivates contextual representation for the final phase; these are not disaster-tweet scores. Pretraining must be disclosed in any comparison. |
| [9] Zhang and Wallace (2015; revised 2016), A Sensitivity Analysis of and Practitioners' Guide to Convolutional Neural Networks for Sentence Classification | Controlled one-layer CNN hyperparameter experiments | Sentence classification benchmarks | Establishes practical guidance on architecture and hyperparameter sensitivity | Supports validation-only tuning of filters and dropout. Does not establish the best settings for this dataset. |

## Comparative synthesis

The averaging approach [7] offers a computationally simple baseline but loses word order. CNNs [3,6] capture short phrases, which may help distinguish disaster descriptions from isolated disaster-related words. Their benefits depend on configuration [9], training data, and word representations. Contextual models [1,2,8] address meanings that depend on surrounding words, but pretrained representations introduce an additional source of information beyond architecture alone. The hybrid rescue pipeline [4] and HumAID [5] broaden the task to operational categories; those results cannot be treated as binary disaster-detection scores.

This interim experiment therefore compares two from-scratch text-only models using the same vocabulary, embedding dimension, data partitions, threshold, training budget and metrics. Model capacity differs and is reported through parameter counts. Later LSTM and encoder-only Transformer experiments should reuse these saved partitions. A pretrained Transformer should be identified as an additional transfer-learning condition rather than attributed solely to architecture.

## Sources

1. Balakrishnan et al., Mathematics 10(24), 4664, 2022. https://www.mdpi.com/2227-7390/10/24/4664
2. Dharrao et al., MethodsX 13, 102843, 2024. https://doi.org/10.1016/j.mex.2024.102843 ; full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11296064/
3. Nguyen et al., ICWSM 11(1), 632–635, 2017. https://ojs.aaai.org/index.php/ICWSM/article/view/14950
4. Kabir and Madria, ACM SIGSPATIAL, 269–278, 2019. https://doi.org/10.1145/3347146.3359097 ; author preprint: https://arxiv.org/abs/1908.01456
5. Alam et al., ICWSM 15(1), 933–942, 2021. https://ojs.aaai.org/index.php/ICWSM/article/view/18116
6. Kim, EMNLP, 1746–1751, 2014. https://aclanthology.org/D14-1181/
7. Iyyer et al., ACL-IJCNLP, 1681–1691, 2015. https://aclanthology.org/P15-1162/
8. Devlin et al., NAACL-HLT, 4171–4186, 2019. https://aclanthology.org/N19-1423/
9. Zhang and Wallace, arXiv:1510.03820, 2015, revised 2016. https://arxiv.org/abs/1510.03820

Sources checked on 16 September 2026. The team should read the linked papers and verify the summaries before signing the report.
