# run this script to pre-download the model, so that it is not downloaded anew on each docker build
# make sure you have git and git-lfs installed first
git clone https://huggingface.co/slone/nllb-rus-tyv-v2-extvoc

cd nllb-rus-tyv-v2-extvoc
git lfs install
git lfs pull
cd ..
