# Links'n'Papers
Collection of interesting links and papers on several topics like Robotics, Machine learning, AI and others.

## Years
* [2023](papers_2023.md)
* [2024](papers_2024.md)

## Robotics

* [FlowMap: High-Quality Camera Poses, Intrinsics, and Depth via Gradient Descent](https://arxiv.org/abs/2404.15259)
  * End-to-end differentiable method that solves for precise camera poses, camera intrinsics, and per-frame dense depth of a video sequence. 

## Machine Learning

### Detection

* [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
  * Single-shot detector approach for real-time object detection.
* [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
  * Loss function focusing on "hard" examples to improve standard cross-entropy loss.

### Face Recognition & Identification

* [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://ieeexplore.ieee.org/document/7298682)
  * Introduces Triplet Loss for direct embeddings training - intuitive, yet innnovative concept. Also touches other training pipeline components.

### Speech recognition

* [Project DeepSpeech](https://github.com/mozilla/DeepSpeech)
  * Open-source speech-to-text engine by Mozzila.
* [Whisper](https://github.com/openai/whisper)
  * OpenAI Automatic Speech recognition in many languages with MIT license.

### Convolutional Neural Networks

* [CBAM: Convolutional Block Attention Module](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)
  * Spatial and channel feature attention blocks added to improve convolutions by focusing on features.
* [Mind the Pad -- CNNs can Develop Blind Spots](https://arxiv.org/abs/2010.02178)
  * A bit hidden, but very interesting paper pointing out overlooked blind spots emerging in training convolutional networks.
* [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
  * Rethinking ResNet architecture with knowledge emerging from transformer architectures.

### Transformers
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  * Novel attention based approach to replace convolutional and recurrent networks in NLP. 
* [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
  * Transformers applied to computer vision with aim to replace convolution.
* [Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://arxiv.org/abs/2107.06278)
  * Unified model architecture for semantic and panoptic segmentation.
* [FlashAttention-2: Faster Attention with Better Parallelism](https://tridao.me/publications/flash2/flash2.pdf)
  * Introduces improved FlashAttention by improving both architecture, but also low-level GPU operations to gain performance.
* [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)
  * Teacher-student based transformer (DeiT) architecture with many nice tricks to improve training efficiency.
* [Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models](https://arxiv.org/abs/2409.17146)
  * SOTA performance open weights&data multimodal models.
 
### LLMs

* [LLM Visualization](https://bbycroft.net/llm)
  * Very illustrative interactive GPT LLM visualisation.
* [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
  * Interpreting high level features (multimodal, language agnostic) in LLM autopencoder, even safety relevant.
* [Mixtral of Experts (2024)](https://arxiv.org/abs/2401.04088)
  * Mixture-of-experts paradigm for effective inference surpassing larger models in both speed and performance.
* [NVLM: Open Frontier-Class Multimodal LLMs](https://arxiv.org/abs/2409.11402)
  * Comparison of multimodal approaches, introduction of novel hybrid approach.
 
### RAG

* [Language agents achieve superhuman synthesis of scientific knowledge](https://paperswithcode.com/paper/2409-13740)
  * Scientific knoledge synthesis RAG agent tool supporting local models (version 2). Also introduces LitQA2 benchmark.
  
### Data generation

* [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
  * Reversed corruption process based data generation from noise.
* [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
  * Realistic image synthesis based on text input.

### Reinforced learning

* [ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters](https://arxiv.org/abs/2205.01906)
  * Control policy for athletics learning improved by combining adversal imitation with unsupervised reinforcement learning.

### Deep learning

* [Visualizing and Understanding Convolutional Networks (Zeiler, Fergus)](https://arxiv.org/abs/1311.2901)
  * Novel article on visualisation CNN's feature layers in order to get deeper understanding.
* [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)
  * Model destillation and ensembelling to improve performance.
* [Revisiting ResNets: Improved Training and Scaling Strategies](https://arxiv.org/abs/2103.07579)
  * Applying new training methods to old networks (ResNet) for baseline methods use. Nicely shows, that some improvements were more of training process, then architecture.
* [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385?context=cs)
  * Effective training of very deep neural networks.
* [Dataset Distillation](https://arxiv.org/abs/1811.10959)
  * Distilling datasets to fraction of original size to make learning more effective.
* [NeuralVDB: High-resolution Sparse Volume Representation using Hierarchical Neural Networks](https://arxiv.org/pdf/2208.04448.pdf)
  * Improve storage efficiency of OpenVDB with nerual network architecture.
* [Can I use this publicly available dataset to build commercial AI software? -- A Case Study on Publicly Available Image Datasets](https://arxiv.org/abs/2111.02374)
  * Analysis of licenses of common public datasets and it's implication for commercial use.
* [Andrej Karpathy: A Recipe for Training Neural Networks](
https://karpathy.github.io/2019/04/25/recipe/)
  * Inspiring blog post about with self-explanatory title.
* [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482v3)
  * Mixing model weights (in the soup) from grid search to improve performance.
* [Robust fine-tuning of zero-shot models](https://arxiv.org/abs/2109.01903)
  * Improving model robustness while fine-tuning by simple ensembling the weights of the zero-shot and fine-tuned models (WiSE-FT).

### Data processing

* [Color-to-Grayscale: Does the Method Matter in Image Recognition?](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3254613/)
  * Study on the topic of RGB-grayscale algorithms and their influence on ML algorithms.

### Data
  * [Open Images Dataset V7 and Extensions](https://storage.googleapis.com/openimages/web/index.html)
    * Open source image dataset with various labels (bounding boxes, text labels, segmentation maps,...).
    * [More Inclusive Annotations for People](https://opendatalab.com/MIAP)
      * Extends Open Images wit additional labels for persons to achieve higher ML fairness.

## Artifitial Intelligence

* [Stop using the elbow criterion for k-means and how to choose the number of clusters instead](https://arxiv.org/pdf/2212.12189.pdf)
  * Elbow method k-means clustering critics with proposal of better methods.

## Free Books

 * [Software Engineering at Google](https://abseil.io/resources/swe-book)
   * Collection of good engineering practices for SW development. Not that much about programming, but also covers team leading etc

 * [Free Programming Books](https://ebookfoundation.org/f-p-b.html)
   * Great collection of (4000+) free programming books and courses (2000+).
  
 * [AI Index Report](https://aiindex.stanford.edu/report/)
   * Yearly Stanford University AI index report "Measuring trends in AI".
  
 * [Master RAG](https://www.rungalileo.io/mastering-rag))
   * A Developer's Guide to Enterprise-Grade RAG Systems from Galileo.
  
## Tools & Frameworks

* [Genesis](https://genesis-embodied-ai.github.io)
  * Open-source physics simulation platform designed for general purpose Robotics, Embodied AI, & Physical AI applications.

## Misc

* [Awesome Foundation Models](https://github.com/uncbiag/Awesome-Foundation-Models)
  * List of large scale pretrained foundation models.
* [Blendify](https://github.com/ptrvilya/blendify)
  * Lightweight Python framework that provides a high-level API for creating and rendering scenes with Blender.

## Cyber security

* [A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards](https://arxiv.org/abs/2308.01074)
  * Eavesdropping your computer keyboard with smartphone or from videocall is possible!
 
* [Deep-tempest: Using Deep Learning to Eavesdrop on HDMI from its Unintended Electromagnetic Emanations](https://github.com/emidan19/deep-tempest?tab=readme-ov-file)
  * Pushing forward gr-tempest effort of intercepting HDMI image transfer via electromagnetic emanations.

## Other links

 * [Passphrase generator](https://theworld.com/~reinhold/diceware.html)
   * A Passphrase dice-based generator. One of the best ways how to create password-passphrase.

## Fun stuff

 * [Epigrams in Programming](https://cpsc.yale.edu/epigrams-programming) 
 * [The Tao of Programming](https://www.mit.edu/~xela/tao.html)
 * [xkcd: A webcomic of romance, sarcasm, math, and language.](https://xkcd.com)
