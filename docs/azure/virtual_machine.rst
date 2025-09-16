Azure Virtual Machine (VM)
==========================

Azure VM enable you to deploy, manage, and scale virtualized computing resources in the cloud. This guide provides detailed instructions for creating, managing, and using Azure VMs efficiently.

This tutorial summarizes official Microsoft documentation, and it's recommended to review the original materials before proceeding:

* `Introduction to Azure Virtual Machines <https://learn.microsoft.com/en-us/azure/virtual-machines/>`_

* `Quickstart: Create a Linux VM in Azure <https://learn.microsoft.com/en-us/azure/virtual-machines/linux/quick-create-portal>`_

Regardless of the section below, ensure you have done the following:

1. **Log in to the Azure Portal:** Open the `Azure Portal <https://portal.azure.com>`_ and sign in with your credentials.

2. **Elevate the subscription permission:** Follow the `Elevate permissions with Privileged Identity Management (PIM) <./pim.rst>`_ guide to elevate your permission.

Creating a disk from a snapshot (optional)
------------------------------------------

If you already have a snapshot and want to re-create a virtual machine from it, follow these steps:

1. **Navigate to snapshot:** Click on **Create disk** in the snapshot overview.

   .. image:: ./img/vm/azure_create_disk.png
      :alt: Azure Create Disk

2. **Configure basic settings:** Fill in the required fields and click **Review + Create**.

   .. image:: ./img/vm/azure_disk_form.png
      :alt: Azure Disk Configuration

3. **Review and create:** Validate the configuration and click **Create** to deploy the disk.

   .. image:: ./img/vm/azure_disk_review.png
      :alt: Azure Disk Review

Creating a VM
-------------

To create an Azure VM, follow these steps:

1. **Navigate to Virtual machine:** Click on **Virtual machine** when creating a new resource in the Azure Portal.

   .. image:: ./img/vm/azure_create_vm.png
      :alt: Azure Create Virtual Machine

2. **Configure basic settings:**

   - **Resource group:** Choose your resource group.

   - **Image:** If you created a disk from a snapshot, select the disk from the list.

   - **Public inbound ports:** Select ``None`` for security reasons.

   .. image:: ./img/vm/azure_vm_form_1.png
      :alt: Azure Virtual Machine Configuration (1)

   .. image:: ./img/vm/azure_vm_form_2.png
      :alt: Azure Virtual Machine Configuration (2)

3. **Review and create:**

   - Validate the configuration.

   - Click **Review + Create**, and then click **Create** to deploy the VM.

Connecting to your VM
---------------------

1. **Elevate the subscription permission:** If not done before, follow the `Elevate permissions with Privileged Identity Management (PIM) <./pim.rst>`_ guide to elevate your permission.

2. **Obtain the public IP address:**

   - Go to the **Overview** section of the VM in the Azure Portal.

   - Copy the **Public IP address**.

3. **Create a manual rule on Azure Portal:**

   - Go to the **Network settings** section of the VM in the Azure Portal.

   - Click on **Create port rule**.

   - Add a rule from your IP address to allow traffic on port 22 (SSH).

     .. image:: ./img/vm/azure_vm_port_rule.png
         :alt: Azure Virtual Machine Port Rule

   - Click on **Save**.

   - Connect to the VM using SSH.

     .. code-block:: bash

         ssh azureuser@<public-ip-address>

4. **Create a manual rule on Azure CLI (alternative):**

   - Run the following command to create a rule for port 22:

     .. code-block:: bash

         az network nsg rule create -g <vm-resource-group> --nsg-name <vm-nsg-name> -n AllowSSHInbound --priority 1000 --source-address-prefixes $(curl -4 ifconfig.me) --destination-port-ranges 22 --access Allow --description "Allow from specific IP address ranges on 22."

   - Run the following command to delete the rule:

     .. code-block:: bash

         az network nsg rule delete -g <vm-resource-group> --nsg-name <vm-nsg-name> -n AllowSSHInbound

5. **Create a JIT rule on Azure Portal (alternative):**

   - Select **Connect** on either the overview or the connect tab.

   - Select **More ways to connect** and choose **SSH using Azure CLI**.

     .. image:: ./img/vm/azure_vm_connect.png
         :alt: Azure Virtual Machine Connect

   - Select **Connect from local machine** on the top tab and click on the **Configure** button.

     .. image:: ./img/vm/azure_vm_connect_azure_cli.png
         :alt: Azure Virtual Machine Connect From Azure CLI

   - After configuring the SSH connection, copy the command and run it in your terminal.

     .. code-block:: bash

         az ssh vm --ip <public-ip-address> # standard `ssh` can also be used

   - If the **VM Access** fails for some reason, that happens because Microsoft defines a ``DenyAllInBound`` rule, however, the JIT rule has a higher priority and will allow access.

     .. image:: ./img/vm/azure_vm_access_error.png
         :alt: Azure Virtual Machine Access Error

Additional resources
--------------------

* `Azure Virtual Machines Documentation <https://learn.microsoft.com/en-us/azure/virtual-machines/>`_

* `Azure CLI Documentation for VMs <https://learn.microsoft.com/en-us/cli/azure/vm>`_

* `Azure Resource Management <https://learn.microsoft.com/en-us/azure/azure-resource-manager/>`_

This tutorial equips you with the knowledge to create, manage, and use Azure Virtual Machines. For further assistance, contact `Azure Support <https://azure.microsoft.com/en-us/support/>`_.
