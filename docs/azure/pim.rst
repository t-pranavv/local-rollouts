Elevate permissions with Privileged Identity Management (PIM)
=============================================================

To manage subscriptions and create resources securely, you need to elevate your SC-ALT account permissions using PIM. This guide walks you through the steps to activate an eligible role and manage your permissions.

The topics in this guide were summarized according to the official documentation provided by Microsoft and we recommend reading them before proceeding:

* `Azure PIM - Elevating Permissions <https://microsoft.sharepoint.com/teams/AIRDEPOT/SitePages/azpim-elevation.aspx>`_

Navigate to PIM
---------------

1. Open the `Azure Portal <https://portal.azure.com>`_.

   .. image:: ./img/pim/azure_portal_login.png
      :alt: Azure Portal Log-In

2. In the search bar, type **Privileged Identity Management** and select the service from the results.

   .. image:: ./img/pim/azure_pim_search.png
      :alt: Azure Portal Privileged Identity Management

Search for your subscription
----------------------------

1. Within the PIM interface, find the **My Roles** section.

   .. image:: ./img/pim/azure_pim_roles.png
      :alt: Azure PIM My Roles

2. Click **Azure Resources** to view your eligible assignments.

   .. image:: ./img/pim/azure_pim_resources_list.png
      :alt: Azure PIM Azure Resources

3. Locate the subscription where you need access and click **Activate** on the eligible role record for the desired subscription or resource.

   .. image:: ./img/pim/azure_pim_resources_select.png
      :alt: Azure PIM Subscription

Fill out the activation form
----------------------------

1. **Validate the information:** Check that the **role**, **member account**, and **scope** are correct.

2. **Set activation details:**

   - Use the slider to select the required duration.

   - If needed, specify a custom activation start time by selecting the **Custom activation start time** checkbox.

3. **Provide a justification:** Enter a meaningful and valid reason for activating the role.

4. Click **Activate**.

   .. image:: ./img/pim/azure_pim_resources_activate.png
      :alt: Azure PIM Activate Role

Processing and browser refresh
------------------------------

1. After clicking **Activate**, your request will be processed.

2. Once the process completes, your browser will refresh automatically. No need to sign out and back in.

Validate role assignment
------------------------

1. Navigate to the **Active Assignments** tab in PIM.

2. Check the following details:

   - **Role:** Ensure it matches the one you activated.

   - **Resource:** Confirm the subscription or resource group.

   - **State:** Should display **Activated**.

   - **End Time:** Review the expiration of the active assignment.

   .. image:: ./img/pim/azure_pim_resources_active.png
      :alt: Azure PIM Active Assignments

Deactivating a role
-------------------

1. If you no longer require elevated access, navigate to the **Active Assignments** tab.

2. Select the active role and click **Deactivate**.

  .. image:: ./img/pim/azure_pim_resources_deactivate.png
      :alt: Azure PIM Deactivate Role

Review assignments
------------------

In addition to your active roles, you can check the following tabs:

* **Eligible Assignments:** Roles available for future activation.

* **Expired Assignments:** Roles you previously activated but have since expired.

Additional resources
--------------------

* `Azure AD Privileged Identity Management Documentation <https://learn.microsoft.com/en-us/azure/active-directory/privileged-identity-management>`_

This guide ensures you can securely elevate permissions using your SC-ALT account and manage your assignments effectively.