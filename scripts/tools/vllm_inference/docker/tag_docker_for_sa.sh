aif_image="aif/reasoning-lm-tooluse:220825"
dest_image="aif/reasoning-lm-tooluse:220825"

az account set --subscription "MSR LIT"
az acr login -n aifrontiers -g aifrontiers
docker pull aifrontiers.azurecr.io/$aif_image
docker tag aifrontiers.azurecr.io/$aif_image aifrontierssacr.azurecr.io/$dest_image
az account set --subscription "ASG Azure ML"
az acr login -n aifrontierssacr -g ai-frontiers-rg
docker push aifrontierssacr.azurecr.io/$dest_image