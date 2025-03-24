from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="slone/nllb-rus-tyv-v2-extvoc",
    local_dir="nllb-rus-tyv-v2-extvoc",
    local_dir_use_symlinks=False
)