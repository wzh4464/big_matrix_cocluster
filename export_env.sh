conda activate cocluster
conda env export --no-builds | grep -v "prefix" > cocluster.yaml
