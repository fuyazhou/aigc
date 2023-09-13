# pip install anndata scib scvi wandb
# git clone https://github.com/bowang-lab/scGPT.git
# git clone https://github.com/qiliu-ghddi/singlecell_gpt.git


cd /home/lushi02/scGPT2/data

python /home/lushi02/scGPT2/singlecell_gpt/data/build_large_scale_data.py --input-dir "/home/lushi02/scGPT2/singlecell_gpt/data/raw/" --output-dir "/home/lushi02/scGPT2/data"


python /home/lushi02/scGPT2/singlecell_gpt/data/binning_mask_allcounts.py --data_source "/home/lushi02/scGPT2/data/all_counts/"