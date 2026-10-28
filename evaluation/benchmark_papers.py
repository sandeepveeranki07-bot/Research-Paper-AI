"""Generates realistic multi-page scientific benchmark PDFs for repeatable evaluation when no local corpus is present.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List
import pymupdf as fitz

from config import get_settings

logger = logging.getLogger(__name__)

BENCHMARK_PAPERS = [
    {
        "filename": "attention_is_all_you_need.pdf",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
        "year": 2017,
        "pages": [
            # Page 1
            "Attention Is All You Need\n\nAshish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit\nGoogle Brain, Google Research\n\nAbstract\nThe dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.\n\n1. Introduction\nRecurrent neural networks, especially long short-term memory (LSTM) and gated recurrent (GRU) neural networks, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation. Inherently sequential computation precludes parallelization within training examples. In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.",

            # Page 2
            "2. Background\nThe goal of reducing sequential computation also forms the foundation of the Extended Neural GPU, ByteNet and ConvS2S, all of which use convolutional neural networks as basic building blocks. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions. In the Transformer this is reduced to a constant number of operations.\n\n3. Model Architecture\nMost competitive neural sequence transduction models have an encoder-decoder structure. Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.",

            # Page 3
            "3.1 Encoder and Decoder Stacks\nEncoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension d_model = 512.\nDecoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking ensures that predictions for position i depend only on known outputs.",

            # Page 4
            "3.2 Attention\nAn attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.\nScaled Dot-Product Attention: We call our particular attention Scaled Dot-Product Attention. The input consists of queries and keys of dimension d_k, and values of dimension d_v. We compute the dot products of the query with all keys, divide each by sqrt(d_k), and apply a softmax function to obtain the weights on the values. In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed into matrices K and V. We compute the matrix of outputs as Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V.",

            # Page 5
            "3.2.2 Multi-Head Attention\nInstead of performing a single attention function with d_model-dimensional queries, keys and values, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to d_k, d_k and d_v dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding d_v-dimensional output values. These are concatenated and once again projected, resulting in the final values.\nMulti-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.\n\n3.3 Position-wise Feed-Forward Networks\nIn addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between: FFN(x) = max(0, xW1 + b1)W2 + b2.",

            # Page 6
            "3.5 Positional Encoding\nSince our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add 'positional encodings' to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension d_model as the embeddings, so that the two can be summed. In this work, we use sine and cosine functions of different frequencies: PE(pos, 2i) = sin(pos/10000^(2i/d_model)) and PE(pos, 2i+1) = cos(pos/10000^(2i/d_model)).\n\n4. Why Self-Attention\nSelf-attention complexity is O(n^2 * d) operations per layer, connecting all positions with a constant number of sequentially executed operations, compared to recurrent layers which require O(n * d^2) sequential operations.",

            # Page 7
            "5. Training\nThis section describes the training regime for our models.\nTraining Data and Batching: We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens.\nHardware and Schedule: We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models, step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).\nOptimizer: We used the Adam optimizer with beta_1 = 0.9, beta_2 = 0.98 and eps = 10^-9. We varied the learning rate over the course of training according to the formula: lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5)), with warmup_steps = 4000.\nRegularization: We employ residual dropout of P_drop = 0.1 and label smoothing of epsilon_ls = 0.1 during training.",

            # Page 8
            "6. Results\nOn the WMT 2014 English-to-German translation task, the big transformer model outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The base model achieved 27.3 BLEU on English-to-German, surpassing all previously published models and ensembles, at a fraction of the training cost.\nOn the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all previously published single models at 1/4 the training cost of the previous state-of-the-art model.\n\n7. Conclusion\nIn this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-head self-attention. For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers."
        ]
    },
    {
        "filename": "vision_transformers_scale.pdf",
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "authors": ["Alexey Dosovitskiy", "Lucas Beyer", "Alexander Kolesnikov", "Dirk Weissenborn"],
        "year": 2020,
        "pages": [
            "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale\n\nAlexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn\nGoogle Research, Brain Team\n\nAbstract\nWhile the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on substantial amounts of data and transferred to multiple mid-sized or small image recognition benchmarks, Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.",
            "1. Introduction\nSelf-attention-based architectures, in particular Transformers, have become the model of choice in natural language processing (NLP). The dominant approach is to pre-train on a large text corpus and then fine-tune on a smaller task-specific dataset. Inspired by the Transformer scaling successes in NLP, we experiment with applying a standard Transformer directly to images, with the fewest possible modifications. To do so, we split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer. Image patches are treated the same way as tokens (words) in an NLP application.",
            "2. Method\nIn model design we follow the original Transformer as closely as possible. An overview of the model is depicted in Figure 1. To handle 2D images, we reshape the image x into a sequence of flattened 2D patches x_p of size P x P. The Transformer uses constant latent vector size D through all of its layers, so we flatten the patches and map to D dimensions with a trainable linear projection. We refer to the output of this projection as the patch embeddings. Similar to BERT's [class] token, we prepend a learnable embedding to the sequence of embedded patches, whose state at the output of the Transformer encoder serves as the image representation y. Position embeddings are added to the patch embeddings to retain positional information. We use standard learnable 1D position embeddings.",
            "3. Experiments\nWe evaluate the representation learning capabilities of ResNet, Vision Transformer (ViT), and hybrid models. To understand the data requirements of each model, we pre-train on datasets of varying size: ImageNet (1.3M images), ImageNet-21k (14M images), and JFT-300M (300M images). When trained on mid-sized datasets such as ImageNet without strong regularization, ViT models yield modest accuracies of a few percentage points below ResNets of comparable size. This outcome is expected: Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data.",
            "4. Scaling and Results\nHowever, the picture changes when the models are trained on larger datasets (14M-300M images). We find that large scale training trumps inductive bias. Our Vision Transformer (ViT-H/14) attains 88.55% top-1 accuracy on ImageNet when pre-trained on JFT-300M, outperforming competitive CNN models like BiT-L while taking substantially fewer TPU compute days to train."
        ]
    },
    {
        "filename": "deep_residual_learning.pdf",
        "title": "Deep Residual Learning for Image Recognition",
        "authors": ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"],
        "year": 2016,
        "pages": [
            "Deep Residual Learning for Image Recognition\n\nKaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun\nMicrosoft Research\n\nAbstract\nDeeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers—8x deeper than VGG nets but still having lower complexity.",
            "1. Introduction\nDeep convolutional neural networks have led to a series of breakthroughs for image classification. Driven by the significance of depth, a natural question arises: Is learning better networks as easy as stacking more layers? An obstacle to answering this question was the notorious problem of vanishing/exploding gradients, which hamper convergence from the beginning. When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing, accuracy gets saturated and then degrades rapidly. Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error.",
            "2. Deep Residual Learning\nInstead of hoping each few stacked layers directly fit a desired underlying mapping H(x), we explicitly let these layers fit a residual mapping F(x) := H(x) - x. The original mapping is recast into F(x) + x. We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping. The formulation of F(x) + x can be realized by feedforward neural networks with shortcut connections. Shortcut connections are those skipping one or more layers. In our case, the shortcut connections simply perform identity mapping, and their outputs are added to the outputs of the stacked layers. Identity shortcut connections introduce neither extra parameter nor computation complexity.",
            "3. Experiments on ImageNet\nWe evaluate our method on the ImageNet 2012 classification dataset that consists of 1000 classes. Our extremely deep 152-layer ResNet achieves a 3.57% top-5 error on the ImageNet test set. This result won the 1st place in the ILSVRC 2015 classification competition. We also note that ResNet CNNs generalize reliably on moderate datasets like ImageNet without requiring hundreds of millions of pre-training images."
        ]
    },
    {
        "filename": "deepconvlstm_activity_recognition.pdf",
        "title": "Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition",
        "authors": ["Francisco Javier Ordonez", "Daniel Roggen"],
        "year": 2016,
        "pages": [
            "Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition\n\nFrancisco Javier Ordonez, Daniel Roggen\nDepartment of Informatics, University of Sussex\n\nAbstract\nWearable activity recognition is central to pervasive computing and mobile healthcare. Traditional approaches rely on hand-crafted features extracted from fixed-size sliding windows. In this paper, we propose DeepConvLSTM, a deep learning architecture that combines 1D convolutional layers with recurrent LSTM cells. The convolutional layers act as feature extractors and provide abstract representations of the input sensor data in feature maps. The recurrent layers model the temporal dynamics of the activation of the feature maps. Experiments on the Opportunity and Skidmore benchmark datasets show that recurrence remains essential for continuous temporal sensory streaming, outperforming CNN-only baselines by over 4-6% in macro F1-score.",
            "1. Introduction\nHuman activity recognition (HAR) from body-worn sensors typically relies on triaxial accelerometers and gyroscopes placed on the body. A key challenge is the continuous segmentation and classification of multimodal signals. While recurrent models like LSTMs capture temporal context, they struggle with high-dimensional multichannel raw inputs. Conversely, CNNs excel at local spatial feature extraction across sensor axes but do not explicitly model long-term temporal dependencies between gestures.",
            "2. Architecture\nDeepConvLSTM consists of four 1D convolutional layers with 64 filters each, followed by two LSTM layers with 128 units each, and a final softmax classification layer. We operate on sliding windows of length 2.56 seconds with 50% overlap. This allows the model to capture fine-grained transitions in manipulative and locomotion activities. The recurrence is indispensable for disambiguating activities with similar static postures but different temporal evolutions.",
            "3. Results and Limitations\nOn the Opportunity challenge benchmark, DeepConvLSTM achieves a macro F1-score of 0.895, outperforming pure CNN baselines. However, a primary limitation is the high computational cost and memory footprint of sequential LSTM cell unrolling, which creates challenges for edge deployment on low-power microcontrollers with under 256KB RAM."
        ]
    },
    {
        "filename": "lora_adaptation.pdf",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": ["Edward J. Hu", "Yelong Shen", "Phillip Wallis", "Zeyuan Allen-Zhu"],
        "year": 2021,
        "pages": [
            "LoRA: Low-Rank Adaptation of Large Language Models\n\nEdward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen\nMicrosoft Corporation\n\nAbstract\nAn important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.",
            "1. Introduction and Method\nFull fine-tuning requires creating and storing an entire copy of all model parameters for every downstream task, which is prohibitively expensive for 175B parameter models. We take inspiration from Li et al. (2018) and Aghajanyan et al. (2020) which show that the learned over-parametrized models in fact reside on a low intrinsic dimension. We hypothesize the updates to the weights also have a low 'intrinsic rank' during adaptation. For a pre-trained weight matrix W_0 in R^(d x k), we constrain its update by representing the latter with a low-rank decomposition W_0 + delta_W = W_0 + B * A, where B in R^(d x r), A in R^(r x k), and the rank r << min(d, k). During training, W_0 is frozen and does not receive gradient updates, while A and B contain trainable parameters.",
            "2. Advantages of LoRA\nA prominent advantage is that a deployed model can switch between different tasks by simply loading the small matrices A and B (typically a few megabytes) while keeping the shared 175B base model in memory. Furthermore, LoRA introduces no additional inference latency compared to a fully fine-tuned model because the product B * A can be explicitly added to W_0 upon deployment.",
            "3. Empirical Evaluation\nWe evaluate LoRA on RoBERTa, DeBERTa, and GPT-3. Across GLUE benchmarks and generative tasks, LoRA matches or exceeds the performance of full fine-tuning while using only 0.01% to 0.1% of the trainable parameters. Limitation: LoRA keeps base model weights in full 16-bit precision during training, which still requires significant GPU memory for base weight storage."
        ]
    },
    {
        "filename": "qlora_quantized_llm.pdf",
        "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
        "authors": ["Tim Dettmers", "Artidoro Pagnoni", "Ari Holtzman", "Luke Zettlemoyer"],
        "year": 2023,
        "pages": [
            "QLoRA: Efficient Finetuning of Quantized LLMs\n\nTim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer\nUniversity of Washington\n\nAbstract\nWe present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA). We introduce 4-bit NormalFloat (NF4), a new data type that is information-theoretically optimal for normally distributed weights, Double Quantization to reduce the average memory footprint by quantizing the quantization constants, and Paged Optimizers to manage memory spikes.",
            "1. Method and Innovations\nWhile LoRA reduces trainable parameter memory, the base model still consumes massive VRAM in 16-bit precision. QLoRA solves this by quantizing base weights to 4-bit NormalFloat (NF4). NF4 is constructed by estimating the quantiles of a standard normal distribution N(0, 1) to ensure each quantization bin has an equal number of expected values, outperforming standard INT4 and FP4.\nDouble Quantization (DQ) quantizes the quantization constants themselves, saving an additional 0.37 bits per parameter (about 3GB for a 65B model).\nPaged Optimizers leverage CUDA Unified Memory to automatically page optimizer states to CPU RAM during activation memory spikes, avoiding out-of-memory errors during large batch training.",
            "2. Results and Comparison with LoRA\nOur results demonstrate that QLoRA achieves 99.3% of the performance of 16-bit full fine-tuning on MMLU and Vicuna benchmarks while reducing base model VRAM requirements by over 70%. Limitation: QLoRA incurs a 20-30% slowdown in training throughput compared to 16-bit LoRA due to the on-the-fly dequantization of 4-bit weights to 16-bit during the forward and backward passes."
        ]
    },
    {
        "filename": "dense_passage_retrieval.pdf",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "authors": ["Vladimir Karpukhin", "Barlas Oguz", "Sewon Min", "Patrick Lewis"],
        "year": 2020,
        "pages": [
            "Dense Passage Retrieval for Open-Domain Question Answering\n\nVladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih\nFacebook AI, University of Washington, Princeton University\n\nAbstract\nOpen-domain question answering relies on efficient passage retrieval to select candidate contexts, where traditional systems predominantly use sparse representations such as TF-IDF or BM25. In this work, we show that retrieval can be practically implemented using dense representations alone, where embeddings are learned from a small number of questions and passages by a simple dual-encoder framework. When evaluated on a wide range of open-domain QA datasets, Dense Passage Retrieval (DPR) substantially outperforms strong BM25 baselines by 9-19% absolute gain in top-20 passage retrieval accuracy.",
            "1. Dual-Encoder Architecture\nDPR uses two independent BERT models: a question encoder E_Q and a passage encoder E_P. Given a question q and a passage p, their similarity is defined as the inner product of their vector representations: sim(q, p) = E_Q(q)^T * E_P(p). We index all passage vectors in FAISS using Inner Product search.\nTraining: The dual encoders are trained using a contrastive loss with in-batch negatives. For each positive pair (q_i, p_i^+), passages associated with other questions in the same training mini-batch serve as effective negative examples, supplemented by hard negatives mined via BM25.",
            "2. Comparison with BM25 and Limitations\nDPR achieves 78.4% top-20 retrieval accuracy on Natural Questions compared to 59.1% for BM25. However, BM25 retains an advantage on queries containing rare named entities, serial numbers, or niche vocabulary absent from training distributions. Combining BM25 with DPR via hybrid linear score interpolation yields the highest overall retrieval precision across open-domain benchmarks."
        ]
    },
    {
        "filename": "flash_attention_fast.pdf",
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "authors": ["Tri Dao", "Daniel Y. Fu", "Stefano Ermon", "Atri Rudra", "Christopher Re"],
        "year": 2022,
        "pages": [
            "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness\n\nTri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re\nStanford University\n\nAbstract\nTransformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods have attempted to address this tradeoff by trading off model quality, but often achieve marginal wall-clock speedup. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. FlashAttention trains Transformers faster than existing baselines: 2-4x wall-clock speedup on BERT-large and GPT-2.",
            "1. IO-Aware Design and Tiling\nOn modern GPUs (e.g. A100), compute speed (312 TFLOPs FP16) drastically outpaces memory bandwidth (1.5 TB/s HBM). Standard attention computes S = QK^T in HBM, writes an N x N matrix, reads S to compute P = softmax(S), writes P back to HBM, and reads P again to compute O = PV. This incurs O(N^2) HBM IO operations.\nFlashAttention addresses this by loading blocks of Q, K, and V into fast on-chip SRAM (19 TB/s bandwidth), computing attention locally, and updating the output incrementally using an online softmax rescaling algorithm. It never materializes the N x N attention matrix in global HBM memory.",
            "2. Impact and Results\nFlashAttention achieves exact attention with linear memory complexity in sequence length O(N) rather than quadratic O(N^2). It enables training models with context lengths of up to 64k tokens without approximation."
        ]
    }
]


def create_benchmark_corpus(target_dir: Optional[Path] = None) -> List[Path]:
    """Generate the benchmark research PDFs if none exist."""
    settings = get_settings()
    out_dir = target_dir or settings.papers_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    created_files: List[Path] = []

    for paper in BENCHMARK_PAPERS:
        file_path = out_dir / paper["filename"]
        if file_path.exists() and file_path.stat().st_size > 0:
            created_files.append(file_path)
            continue

        doc = fitz.open()
        for p_idx, text_content in enumerate(paper["pages"], start=1):
            page = doc.new_page(width=612, height=792)  # Standard Letter size

            # Insert Header
            header_rect = fitz.Rect(54, 36, 558, 60)
            page.insert_textbox(header_rect, f"{paper['title']} | Page {p_idx}", fontsize=8, color=(0.4, 0.4, 0.4))

            # Insert Body Text
            body_rect = fitz.Rect(54, 72, 558, 720)
            page.insert_textbox(body_rect, text_content, fontsize=10, fontname="helv", lineheight=1.35)

            # Insert Footer
            footer_rect = fitz.Rect(54, 735, 558, 760)
            page.insert_textbox(footer_rect, f"Research Benchmark Suite — Page {p_idx} of {len(paper['pages'])}", fontsize=8, align=1, color=(0.5, 0.5, 0.5))

        # Set PDF Metadata
        doc.set_metadata({
            "title": paper["title"],
            "author": ", ".join(paper["authors"]),
            "creationDate": f"D:{paper['year']}0101000000",
            "producer": "Research Intelligence Assistant Benchmark Generator"
        })

        doc.save(str(file_path))
        doc.close()
        created_files.append(file_path)
        logger.info(f"Generated benchmark PDF: {file_path.name} ({len(paper['pages'])} pages)")

    return created_files
