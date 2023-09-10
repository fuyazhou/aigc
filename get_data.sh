pip install anndata scib scvi wandb


git clone https://github.com/bowang-lab/scGPT.git
git clone https://github.com/qiliu-ghddi/singlecell_gpt.git


python /content/singlecell_gpt/data/build_large_scale_data.py --input-dir "/content/singlecell_gpt/data/raw/" --output-dir "/content/"


python /content/singlecell_gpt/data/binning_mask_allcounts.py --data_source "/content/all_counts/"