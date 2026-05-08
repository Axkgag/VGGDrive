<img width="1280" height="300" alt="VGGDrive" src="https://github.com/user-attachments/assets/9976a4f6-51d7-4d2d-aa35-1d9e46bde598" />

<h2 align="center">
✨ VGGDrive: Empowering Vision-Language Models ✨<br>
with Cross-View Geometric Grounding for Autonomous Driving
</h2>

## 📢 News
- **[2026/03/09]** 🌐 Project page is live: [demo](https://WJ-CV.github.io/VGGDrive/)
- **[2026/02/26]** 🚀 Released [VGGDrive NAVSIM v1 weights](#vggdrive-model-zoo) and inference code.
- **[2026/02/24]** 👉 We released our paper on [arXiv](https://arxiv.org/abs/2602.20794).
- **[2026/02/21]** 🎉🎉🎉 Accepted to CVPR 2026.


## 🔬 Project Overview

🧩 Conventional VLMs in autonomous driving “understand language but lack geometric insight.” Even when augmented with constructed Q&A data for auxiliary training, such approaches provide only superficial improvements and fail to address the core limitation in cross-view 3D spatial understanding.

💡 **VGGDrive** moves beyond data-level fixes and **charts a new course** by upgrading the capability structure itself. It introduces a mature 3D foundation model as a geometric backbone for VLMs, establishing a new technical paradigm that empowers Vision-Language Agents (VLAs) with 3D modeling capability and provides a scalable, sustainable pathway for enhancing autonomous driving systems.

🛠️ The core innovation lies in the design of a **plug-and-play Cross-View Geometric Enabler (CVGE)**. Through a hierarchical adaptive injection mechanism, VGGDrive achieves deep coupling between a frozen 3D foundation model and a VLM without altering the original VLM architecture. This mechanism efficiently injects 3D geometric features into the model, enabling genuine cross-view 3D geometric modeling capability for autonomous driving VLAs.

<table>
<tr>
<td width="50%" valign="top">

<p style="text-align: justify;">
📈 Importantly, VGGDrive is not limited to single-task optimization. It consistently improves performance across <strong>five mainstream autonomous driving benchmarks</strong>, covering cross-view risk perception, scene understanding, motion and state prediction, and trajectory planning, thereby enhancing the full pipeline from perception to decision-making.
</p>

</td>
<td width="50%" valign="top" align="center">

<img src="https://github.com/user-attachments/assets/9676c112-8140-4a12-aa02-5145f126d4a5" width="70%" />

</td>
</tr>
</table>

---
## 🏗️ Framework

<img width="3568" height="2208" alt="fig3_2" src="https://github.com/user-attachments/assets/ed54172b-0d78-49b6-940d-db1dea110700" />

<a name="vggdrive-model-zoo"></a>
## 🏛️ Model Zoo
| Model | Dataset | Download | Qwen_json |
|:-----:|:-------:|:--------:|:---------:|
| VGGDrive | NAVSIM | [ckpt](https://huggingface.co/wang-jie825/VGGDrive_model/tree/main) | [train & test](https://huggingface.co/datasets/wang-jie825/VGGDrive_Qwen_json/tree/main/navsim_cache) |
| VGGDrive | NuInstruct | | [train & test](https://huggingface.co/datasets/wang-jie825/VGGDrive_Qwen_json/tree/main/nuScenes_cache)  |
| VGGDrive | DriveLM | [submission.json](https://huggingface.co/datasets/wang-jie825/VGGDrive_Qwen_json/blob/main/DriveLM_submission.json) | [train & test](https://huggingface.co/datasets/wang-jie825/VGGDrive_Qwen_json/tree/main/nuScenes_cache)  |
| VGGDrive | OmniDrive | | [train & test](https://huggingface.co/datasets/wang-jie825/VGGDrive_Qwen_json/tree/main/nuScenes_cache)  |
| VGGDrive | NuScenes | | [train & test](https://huggingface.co/datasets/wang-jie825/VGGDrive_Qwen_json/tree/main/nuScenes_cache)  |
> ⚠️ **Prerequisite:**
> 
> Please download the pretrained VGGT model weights (`model.pt`) from [vggt](https://github.com/facebookresearch/vggt) and place it in the `./vggt` folder.

## 🏁 Quick Start
### 1. Environment

We recommend using Python 3.10+ and CUDA 12.x. The core dependencies used in this project include:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install transformers==4.49.0 accelerate==1.3.0 datasets==3.2.0
pip install vllm==0.7.1 flash-attn==2.7.4.post1 xformers==0.0.28.post3
pip install timm==1.0.14 peft==0.14.0 bitsandbytes==0.45.2
pip install opencv-python==4.11.0.86 pillow==11.1.0
pip install numpy==1.26.4 pandas==2.2.3 scipy==1.15.2 scikit-learn==1.6.1
pip install nuscenes-devkit==1.1.11 pyquaternion==0.9.9 shapely==1.8.5.post1
```
Alternatively, install all dependencies from the provided environment file:
```bash
pip install -r requirements.txt
```
### 2. Run Inference
Run the NAVSIM inference script to generate prediction results:
```bash
bash run_scripts/inference_navsim.sh
```
### 2. Evaluation
Please follow the official NAVSIM v1.1 evaluation protocol:
[NAVSIM-v1.1](https://github.com/autonomousvision/navsim/tree/v1.1)
NAVSIM evaluates end-to-end driving performance with simulation-based metrics such as progress and time-to-collision under a non-reactive simulation setting. For detailed setup, dataset preparation, and metric computation, please refer to the official NAVSIM repository.

## 📌 Citation
```
@article{wang2026vggdrive,
  title={VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving},
  author={Wang, Jie and Li, Guang and Huang, Zhijian and Dang, Chenxu and Ye, Hangjun and Han, Yahong and Chen, Long},
  journal={arXiv preprint arXiv:2602.20794},
  year={2026}
}
```
