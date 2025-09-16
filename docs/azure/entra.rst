Microsoft Entra
===============

Microsoft Entra is used for managing identity and authentication in your application. It includes services like Azure Active Directory (Azure AD) and is essential for creating secure Single-Page Applications (SPA).

Setting up Microsoft Entra for your application
-----------------------------------------------

1. **Access Microsoft Entra:** Go to `Microsoft Entra <https://entra.microsoft.com>`_ and sign in.

2. **Create a new application:**

   - Navigate to **App registrations** from the home dashboard.

   - Select **New registration** to create a new application.

   - Choose **Single-Page Application** and set the **Redirect URI** to ``http://localhost:3000/`` (or any other relevant frontend URL).

   - Make sure to note down the Application (Client) ID, as it will be used in your application configuration.

3. **Configure authentication:** In the **Authentication** section, ensure that **Access tokens (used for implicit flows)** is selected under **Implicit grant and hybrid flows.**

4. **Expose an API (if applicable):**

   - Navigate to **Expose an API** and add a new scope, naming it ``default``.

   - Set **Who can consent?** to ``Admins only``, and fill out the required fields with relevant information.

   - Add your client application using the ``ENTRA_CLIENT_ID``.

Additional resources
--------------------

- `Microsoft Entra Documentation <https://learn.microsoft.com/en-us/azure/active-directory/fundamentals/active-directory-whatis>`_

- `OAuth 2.0 and OpenID Connect <https://learn.microsoft.com/en-us/azure/active-directory/develop/v2-oauth2-implicit-grant-flow>`_