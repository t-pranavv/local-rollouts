#!/bin/bash

registry="singularitybase"

# Creates the validation image
validator_image_repo="validations/base/singularity-tests"
validator_image_tag=`az acr manifest list-metadata \
    --registry $registry \
    --name $validator_image_repo \
    --orderby time_desc \
    --query '[].{Tag:tags[0]}' \
    --output tsv \
    --top 1`
validator_image=$registry.azurecr.io/$validator_image_repo:${validator_image_tag%%[[:cntrl:]]}
echo $validator_image

# Creates the installer image
installer_image_repo="installer/base/singularity-installer"
installer_image_tag=`az acr manifest list-metadata \
    --registry $registry \
    --name $installer_image_repo \
    --orderby time_desc \
    --query '[].{Tag:tags[0]}' \
    --output tsv \
    --top 1`
installer_image=$registry.azurecr.io/$installer_image_repo:${installer_image_tag%%[[:cntrl:]]}
echo $installer_image