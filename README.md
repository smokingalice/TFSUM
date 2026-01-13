# TF-SUM: Training-free Video Summarization via Language-centric Hierarchical Reasoning
> **TF-SUM** is an innovative training-free video summarization framework designed to generate high-quality summaries without relying on any training data or manual annotations. It fully exploits the reasoning capabilities of pre-trained Large Vision-Language Models (LVLMs) and Large Language Models (LLMs). Specifically, an LVLM is first utilized to generate temporal-aware textual descriptions for video frames, where noisy captions are filtered via cross-modal similarity. Subsequently, an LLM performs hierarchical semantic reasoning on these textual summaries to infer both frame-level and segment-level importance scores. Finally, visual similarity mechanisms are applied to eliminate redundant content and refine the importance distribution. Extensive experiments demonstrate that TF-SUM achieves competitive performance compared to state-of-the-art methods.

## Hardware 
- **GPUs**: 2 × NVIDIA Tesla V100-SXM2-32GB  
- **CPU**: Standard compute node (details not specified)  
- **CUDA Version**: 12.2  
- **Operating System**: Ubuntu 20.04.5 LTS  
> Note: TF-SUM is training-free, but efficient inference with LVLMs and LLMs requires GPUs with sufficient memory (24GB+ recommended).

## Getting Started

Clone the repository to your local machine:
   ```bash
   git clone .......
   cd TFSUM
   ```
Set up the conda environment:
   ```bash
   conda create --name TFSUM python=3.10
   conda activate TFSUM
   pip install -r requirements.txt
   ```

##  Datasets

Download the TVSum and SumMe datasets:
- **TVSum**: [https://opendatalab.com/OpenDataLab/TVSum/tree/main] (https://opendatalab.com/OpenDataLab/TVSum)  
- **SumMe**: [https://opendatalab.org.cn/OpenDataLab/SumMe] (https://opendatalab.org.cn/OpenDataLab/SumMe)

Organize the datasets as follows:

- Place the **source videos** into the `/videos` subdirectory under each dataset folder.

- For **TVSum**:  
  Copy `ydata-tvsum50.mat` into `datasets/TVSum/annotations/`

- For **SumMe**:  
  Extract the `GT` folder and place its contents into `datasets/SumMe/annotations/`

Download the **preprocessed HDF5 annotation files**:

- [eccv16_dataset_tvsum_google_pool5.h5](https://www.sendgb.com/upload/?utm_source=igjvxR46m5I) → place into `datasets/TVSum/annotations/`
- [eccv16_dataset_summe_google_pool5.h5](https://www.sendgb.com/upload/?utm_source=igjvxR46m5I) → place into `datasets/SumMe/annotations/`
```
datasets📁/
├── TVSum/
│   ├── videos/
│   │   └── ... (original TVSum video files)
│   └── annotations/
│       ├── ydata-tvsum50.mat
│       ├── eccv16_dataset_tvsum_google_pool5.h5
│       └── ydata-tvsum50-anno.tsv  (auto-included after git clone, do not delete)
│
└── SumMe/
    ├── videos/
    │   └── ... (original SumMe video files)
    └── annotations/
        ├── GT/ (extracted ground truth files)
        ├── eccv16_dataset_summe_google_pool5.h5
        └── ydata-summe50-anno.tsv  (auto-included after git clone, do not delete)
```

> Make sure the folder structure and filenames are correct, as the system relies on these paths for loading data.

## Model Preparation
The large language model used in this project is **LLaMA 2**, specifically the **llama-2-13b-chat** version.

   - You can download it from the official Meta website:  [https://www.llama.com/llama-downloads/](https://www.llama.com/llama-downloads/)
   - Or from Hugging Face:   [https://huggingface.co/meta-llama](https://huggingface.co/meta-llama)

After downloading, place the model folder`llama-2-13b-chat/`into the following path:

   ```
   libs📁/
   └── llama/
       └── llama-2-13b-chat/
   ```
The vision-language models **BLIP-2** and **ImageBind** will be automatically downloaded during runtime.

## Pipeline Execution

To run the full summarization pipeline on the **TVSum** dataset, execute the following scripts in order:

**Extract frame captions and refine them using visual features**:
   ```bash
   bash run_scripts/TVSum/1_extract_frames_tvsum.sh
   bash run_scripts/TVSum/2_frame_caption_tvsum.sh
   bash run_scripts/TVSum/3_make_index_tvsum.sh
   bash run_scripts/TVSum/4_text_refinement_tvsum.sh
   ```
**Generate hierarchical summaries and assign importance scores using the LLM**:
   ```bash
   bash run_scripts/TVSum/5.1_llm_sumscore_tvsum.sh
   bash run_scripts/TVSum/5.2_llm_framescore_tvsum.sh
   bash run_scripts/TVSum/6_make_summary_index_tvsum.sh
   ```
**Refine the hierarchical scores**:
   ```bash
   bash run_scripts/TVSum/7_hierarchical_scores_refine_tvsum.sh
   ```
**Evaluate and view the results**:
   ```bash
   bash run_scripts/TVSum/8_evaluate_tvsum.sh
   ```
The same procedure applies to the **SumMe** dataset (under `run_scripts/SumMe`).  If you want to apply TF-SUM to your **own dataset**, simply modify the file paths in the corresponding scripts, and ensure that your dataset includes annotation files (e.g., `.tsv`, `.mat`, or `.h5`) similar to those used in TVSum or SumMe.
> ⚠️ **Make sure to execute the scripts in the correct order. Running them out of sequence will result in errors.**
