Azure Storage Account
=====================

Accessing Azure storage securely without shared keys ensures better compliance with security policies by leveraging Azure Active Directory (AAD) and Role-Based Access Control (RBAC). This guide explains how to securely access storage resources and transfer files using Azure CLI or AzCopy.

Prerequisites
-------------

* Install the required tools:

  - `Azure CLI <https://learn.microsoft.com/en-us/cli/azure/install-azure-cli>`_

  - `AzCopy <https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10>`_

**Note:** ``az storage copy`` is an alias/replacement for ``azcopy copy``.

Configure access via Azure AD
-----------------------------

1. Use Azure CLI to authenticate:

   .. code-block:: bash

      az group create --name <RESOURCE_GROUP_NAME> --location <LOCATION>

   .. image:: ./img/storage/az_login.png
      :alt: Azure CLI Log-In

   .. image:: ./img/storage/az_login_account.png
      :alt: Azure CLI Log-In Account

   .. image:: ./img/storage/az_login_code.png
      :alt: Azure CLI Log-In Code

   .. image:: ./img/storage/az_login_subscription_select.png
      :alt: Azure CLI Log-In Subscription Select

2. Verify your access:

   .. code-block:: bash

      az storage container list --account-name <storage-account-name> --auth-mode login

   .. image:: ./img/storage/az_storage_list.png
      :alt: Azure CLI Storage List

Accessing storage resources without keys
----------------------------------------

You can interact with Azure Storage resources using Azure CLI commands without specifying account keys:

* List blobs in a container:

  .. code-block:: bash

      az storage blob list --container-name <container-name> --account-name <storage-account-name> --auth-mode login

  .. image:: ./img/storage/az_storage_blob_list.png
      :alt: Azure CLI Storage Blob List

* Upload a file to a container:

  .. code-block:: bash

      az storage copy -s <source-local-folder> -d <destination-remote-folder> --auth-mode login

* Download a file from a container:

  .. code-block:: bash

      az storage copy -s <source-remote-folder> -d <destination-local-folder> --auth-mode login

  .. image:: ./img/storage/az_storage_copy.png
      :alt: Azure CLI Storage Copy

Mounting storage as a file system
---------------------------------

To mount a storage container as a file system using **BlobFuse2**, follow the steps below:

1. **Prepare the configuration file**:

   Create a ``.yaml`` configuration file with the appropriate parameters to define your storage and caching settings.

   .. code-block:: yaml

      allow-other: true

      logging:
         type: syslog
         level: log_debug

      components:
         - libfuse
         - file_cache
         - attr_cache
         - azstorage

      libfuse:
         attribute-expiration-sec: 120
         entry-expiration-sec: 120
         negative-entry-expiration-sec: 240

      file_cache:
         path: <path-to-file-cache>
         timeout-sec: 120
         max-size-mb: 4096

      attr_cache:
         timeout-sec: 7200

      azstorage:
         type: adls
         account-name: <account-name>
         container: <container>
         mode: azcli

2. **Run the mount command**:

   Use the ``blobfuse2`` utility to mount the storage container to a specified mount point.

   .. code-block:: bash

      blobfuse2 mount <path-to-mount> --config-file <yaml-config-file>

   If you plan to use the ``--allow-other`` option, ensure that the system is configured correctly:

   - Open ``/etc/fuse.conf`` with superuser privileges:

     .. code-block:: bash

         sudo nano /etc/fuse.conf

   - Uncomment or add the following line:

     .. code-block:: text

         user_allow_other

   - Save and exit the file.

   - Re-run the mount command with ``--allow-other`` enabled.

Additional resources
--------------------

* `Azure CLI Storage Documentation <https://learn.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-cli>`_

* `AzCopy Documentation <https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10?tabs=dnf>`_

* `Azure Role-Based Access Control (RBAC) <https://learn.microsoft.com/en-us/azure/role-based-access-control/overview>`_