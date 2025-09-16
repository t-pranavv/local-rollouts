Azure Container Registry (ACR)
==============================

Azure Container Registry (ACR) is used for storing and managing Docker container images in Azure. It allows you to deploy your Docker containers directly to Azure App Service or other Azure services.

Setting up the ACR
------------------

1. **Create an ACR:** Use the following command to create an ACR:

   .. code-block:: bash

      az acr create --resource-group <RESOURCE_GROUP_NAME> --name <ACR_NAME> --sku Basic

2. **Log in to ACR:**

   .. code-block:: bash

      az acr login --name <ACR_NAME>

3. **Build and push Docker images to ACR:**

   - Build your Docker image using Docker Compose:

     .. code-block:: bash

         docker-compose build

   - Tag the image to push to ACR:

     .. code-block:: bash

         docker tag <TAG> <ACR_NAME>.azurecr.io/<TAG>:<VERSION>

   - Push the images to ACR:

     .. code-block:: bash

         docker push <ACR_NAME>.azurecr.io/<TAG>:<VERSION>

Additional resources
--------------------

- `Azure Container Registry Documentation <https://learn.microsoft.com/en-us/azure/container-registry/>`_

- `Docker Tagging and Pushing <https://docs.docker.com/engine/reference/commandline/tag/>`_