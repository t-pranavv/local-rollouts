Log-In with SC-ALT
==================

Your SC-ALT account is required for administrative access and secure operations within Azure. This document provides detailed guidance on creating, managing, and using your SC-ALT account effectively.

The topics in this guide were summarized according to the official documentation provided by Microsoft and we recommend reading them before proceeding:

* `How to sign-in with your SC-ALT account <https://microsoft.sharepoint.com/teams/AIRDEPOT/SitePages/azpim-scaltsignin.aspx>`_

* `Unblock or Change Your Smart Card PIN with Microsoft Smart Card Manager (Legacy PIN Tool is deprecated) <https://microsoft.sharepoint.com/sites/Identity_Authentication/SitePages/MicrosoftPINTool/Unblock-Your-Smart-Card-PIN-with-PIN-Tool.aspx>`_

What is an SC-ALT account?
--------------------------

The SC-ALT account (Smartcard Alternate Account) is a secure account used to manage production systems in Microsoft-managed tenants and domains. These accounts:

* **Authenticate via Smart Card only:** Password authentication is disabled for enhanced security.

* **Separate administrative tasks:** Provides isolation between administrative work and daily productivity tasks.

* **Enable privileged access:** Required for Owner, Owner, or other elevated roles in Azure.

Creating an SC-ALT account
--------------------------

To create an SC-ALT account, follow these steps:

1. **Access the CoreIdentity portal:** Go to the `CoreIdentity Portal <https://aka.ms/CoreIdentity>`_.

2. **Initiate account creation:** On the homepage, click on the **Create SC-ALT Account** tile.

3. **Select a domain:**

   - Choose the domain where you need the account (e.g., Redmond by default).

   - You will only see domains where you do not already have an SC-ALT account.

4. **Mail-enable option:** If necessary, select the option to mail-enable your SC-ALT account (each user can only have one mail-enabled SC-ALT).

5. **Provide business justification:** Enter a clear and concise reason for the account creation.

6. **Submit your request:** Wait for the provisioning process (approximately 10-60 minutes).

7. **Smart Card request:** CoreIdentity will automatically notify the Global Security Access Management (GSAM) team to create a smart card for your account. For support with your smart card, visit `GSAM <https://spo.ms/gsam>`_.

Unblocking Your Smart Card PIN
------------------------------

If your Smart Card PIN is blocked:

1. **Use the Microsoft Smart Card Manager Tool:** Download and install the **Microsoft Smart Card Manager (MSCM)** tool.

2. **Follow the instructions:**

   - Insert your smart card into the reader.

   - Launch the MSCM tool and select **Manage PIN** - **Unblock PIN**.

   - Enter your new PIN and confirmation.

3. **Offline unblock (if required):**

   - Use the **Get Challenge** option in MSCM.

   - Contact the helpdesk to provide your Challenge code and receive a Response code.

   - Follow prompts to complete the unblock process.

Signing in to SC-ALT
--------------------

Follow these steps to sign in with your SC-ALT account:

1. **Prepare your Smart Card and reader:** Ensure your Smart Card is inserted into the reader.

2. **Log in to the Azure Portal:**

   - Open `Azure Portal <https://portal.azure.com>`_ in the Edge browser.

   - Select **Sign in with a different account**.

   .. image:: ./img/sc-alt/azure_portal_login.png
      :alt: Azure Portal Log-In

3. **Input SC-ALT credentials:** Use the format: ``<alias>@<domain>.microsoft.com``.

4. **Authenticate with your Smart Card:**

   - Choose **Sign in with PIN or Smart Card**.

   .. image:: ./img/sc-alt/azure_portal_login_certificate.png
      :alt: Azure Portal SC-ALT Certificate

   - Select your SC-ALT certificate when prompted and enter your PIN.

   .. image:: ./img/sc-alt/azure_portal_login_certificate_select.png
      :alt: Azure Portal SC-ALT Certificate Selection

5. **Verify successful login:** Confirm your SC-ALT account is active by checking the profile details in Azure Portal.

Additional resources
--------------------

* `CoreIdentity Portal <https://aka.ms/CoreIdentity>`_

* `SC-ALT Documentation <https://aka.ms/scalt>`_

* `Global Services Access Management (GSAM) <https://spo.ms/gsam>`_

This guide equips you to set up, manage, and use your SC-ALT account securely. For further assistance, contact `Identity Help <https://aka.ms/identityhelp>`_.