Azure App Service
=================

Azure App Service allows you to deploy web applications, REST APIs, and mobile backends to Azure without managing the underlying infrastructure. This guide covers two deployment scenarios: regular Azure App Service for traditional apps and App Service for containers.

App Service
-----------

Deploy your web applications or APIs directly to Azure App Service without managing any servers.

1. **Create an App Service plan:** Create an App Service plan to host your app:

   .. code-block:: bash

      az appservice plan create --name <APP_SERVICE_PLAN_NAME> --resource-group <RESOURCE_GROUP_NAME> --sku B1

2. **Create the Web App:** Once the plan is created, you can deploy your web app:

   .. code-block:: bash

      az webapp create --resource-group <RESOURCE_GROUP_NAME> --plan <APP_SERVICE_PLAN_NAME> --name <WEB_APP_NAME>

3. **Configure environment variables:** If your web app requires environment variables, you can set them via the Azure CLI:

   .. code-block:: bash

      az webapp config appsettings set --name <WEB_APP_NAME> --resource-group <RESOURCE_GROUP_NAME> --settings ENV_VAR_NAME=env_var_value

4. **Deploy your application:** You can use several methods to deploy your app:

   - **Azure DevOps Pipelines** or **GitHub Actions** for CI/CD.

   - **ZIP Deploy** using the Azure CLI:

   .. code-block:: bash

      az webapp deployment source config-zip --resource-group <RESOURCE_GROUP_NAME> --name <WEB_APP_NAME> --src <ZIP_FILE>

5. **Manage and scale:** You can manage your Web App's scaling and performance by adjusting your App Service Plan's pricing tier and instance count.

App Service for Containers
--------------------------

Deploy containerized applications using Docker or Docker Compose to Azure App Service. This guide explains how to set up and deploy containers.

1. **Create an App Service plan for Linux:** Create a Linux-based plan to host containerized applications:

   .. code-block:: bash

      az appservice plan create --name <APP_SERVICE_PLAN_NAME> --resource-group <RESOURCE_GROUP_NAME> --sku B1 --is-linux

2. **Create a Web App for Containers:** Create a Web App with container support using the Docker Compose file:

   .. code-block:: bash

      az webapp create --resource-group <RESOURCE_GROUP_NAME> --plan <APP_SERVICE_PLAN_NAME> --name <WEB_APP_NAME> --multicontainer-config-type compose --multicontainer-config-file docker-compose.yml

3. **Configure environment variables for containers:** You can set environment variables for your containerized app via the Azure CLI:

   .. code-block:: bash

      az webapp config appsettings set --name <WEB_APP_NAME> --resource-group <RESOURCE_GROUP_NAME> --settings ENV_VAR_NAME=env_var_value

4. **Set up Azure Container Registry (ACR) authentication:** If you're using ACR, you'll need to set up authentication for your Web App to pull images from ACR:

   - Assign a managed identity to the Web App:

     .. code-block:: bash

         az webapp identity assign --resource-group <RESOURCE_GROUP_NAME> --name <WEB_APP_NAME>

   - Grant ACR pull permissions:

     .. code-block:: bash

         ACR_ID=$(az acr show --name <ACR_NAME> --query id --output tsv)

         az role assignment create --assignee $(az webapp identity show --name <WEB_APP_NAME> --resource-group <RESOURCE_GROUP_NAME> --query principalId --output tsv) --role "AcrPull" --scope $ACR_ID

5. **Deploy the application:** Ensure that your images are pushed to ACR or any other container registry, and that your ``docker-compose.yml`` is configured correctly.

6. **Restart the Web App:** After setting everything up, restart the Web App:

   .. code-block:: bash

      az webapp restart --name <WEB_APP_NAME> --resource-group <RESOURCE_GROUP_NAME>

Additional resources
--------------------

- `Azure App Service Documentation <https://docs.microsoft.com/en-us/azure/app-service/>`_

- `Deploying Web Apps <https://docs.microsoft.com/en-us/azure/app-service/deploy-overview>`_

- `Azure App Service for Containers <https://docs.microsoft.com/en-us/azure/app-service/containers/>`_

- `Docker Compose for App Service <https://docs.microsoft.com/en-us/azure/app-service/tutorial-multi-container-app>`_