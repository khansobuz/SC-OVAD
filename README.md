SC-OVAD: Scene-Conditioned Open-Vocabulary Decoupled Learning for Video Anomaly Detection

📌 Overview
![Uploading main.jpg…]()

Weakly supervised video anomaly detection aims to identify abnormal events in untrimmed videos using only video-level supervision. However, existing methods often rely on globally shared normality assumptions, fail to capture scene-dependent semantic variations, and show limited generalization to unseen anomaly categories in open-vocabulary settings.

To address these limitations, we propose SC-OVAD (Scene-Conditioned Open-Vocabulary Decoupled Learning), a framework designed for robust and generalizable video anomaly detection across diverse scenes.

🧠 Abstract

We propose SC-OVAD, a Scene-Conditioned Contrastive Open-Vocabulary Decoupling framework for weakly supervised video anomaly detection. Extensive experiments on UCF-Crime, XD-Violence, and UBnormal demonstrate that SC-OVAD achieves state-of-the-art performance and strong generalization to unseen anomaly categories.

⚙️ Implementation Details

We use pre-extracted CLIP features for all experiments.

Backbone: Frozen CLIP (ViT-B/16)
CLIP feature extraction follows:
👉 https://github.com/joos2010kj/CLIP-TSA

📂 Datasets

For the UBnormal dataset, we followed the official implementation and data processing pipeline from the PLOVAD repository (https://github.com/ctX-u/PLOVAD) for consistency and reproducibility.
We evaluate our method on three benchmark datasets:

UCF-Crime

Waqas Sultani, Chen Chen, and Mubarak Shah, Real-World Anomaly Detection in Surveillance Videos, CVPR 2018.

XD-Violence

Peng Wu et al., Not only Look, But Also Listen: Learning Multimodal Violence Detection Under Weak Supervision, ECCV 2020.

UBnormal

Andra Acsintoae et al., UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection, CVPR 2022.

🚀 Running the Code

We provide separate training scripts for each dataset:

# UBnormal dataset
python main_UB.py

# UCF-Crime dataset
python main_UCF.py

# XD-Violence dataset
python main_xd.py

🧪 Features
Scene-conditioned anomaly modeling
Open-vocabulary generalization
CLIP-based semantic alignment
Decoupled scene vs event representation learning

📊 Results

Our method achieves state-of-the-art performance on all evaluated datasets and shows strong generalization to unseen anomaly categories in open-vocabulary settings.



