Data generation infrastructure
==============================

The API Gateway and Playground are crucial components of our infrastructure for data generation and model evaluation:

* `API Gateway <http://gateway.phyagi.net>`_: This gateway facilitates interaction with multiple LLM providers, enabling data generation, model evaluation, and more.
* `Playground <http://playground.phyagi.net>`_: A web-based interface for visually interacting with the PhyAGI platform, allowing for data generation, model evaluation, and additional functionalities.

Accessing the API Gateway and Playground
----------------------------------------

To access the API Gateway and Playground, please contact a member of the PhyAGI team with Playground admin access. Reach out to Gustavo de Rosa (gderosa@microsoft.com) or Piero Kauffmann (pkauffmann@microsoft.com) to request access.

Calling the API Gateway
-----------------------

1. Navigate to the Playground `Keys <https://playground.phyagi.net/collections/keys>`_ tab to select an existing API key or create a new one.

2. Use the API key to call the API Gateway with the following cURL command:

   .. code-block:: bash

      curl --request POST \
      --url http://gateway.phyagi.net/api/chat/completions \
      --header 'Authorization: Bearer <my-api-key>' \
      --header 'Content-Type: application/json' \
      --data '{
          "model": "gpt-4o",
          "messages": [
              {"role": "user", "content": "What is 0.5! equal to?"}
          ],
          "temperature": 0
      }'

   Alternatively, use the Python ``requests`` library:

   .. code-block:: python

      import requests

      api_key = "<my-api-key>"
      url = "http://gateway.phyagi.net/api/chat/completions"
      headers = {
          "Authorization": f"Bearer {api_key}",
          "Content-Type": "application/json"
      }

      data = {
          "model": "gpt-4o",
          "messages": [
              {"role": "user", "content": "Explain like I'm five: what is the meaning of life?"},
          ],
          "temperature": 0.0,
      }

      response = requests.post(url, headers=headers, json=data)
      if response.status_code == 200:
          print(response.json())
      else:
          print(response.text)

The Playground provides a complete list of available `models <https://playground.phyagi.net/collections/models>`_ and `endpoints <https://playground.phyagi.net/collections/endpoints>`_.

Phigen
------

Phigen is a Python package that provides a high-level interface for generating complex multi-turn synthetic data using the PhyAGI API Gateway. It is designed to simplify the process of generating synthetic data for training and evaluating language models. To see the full documentation and installation instructions, visit its `repository <https://github.com/technology-and-research/phyagi/tree/main/phigen>`_.

Gateway statistics
------------------

To view global and per-model statistics for the API Gateway, log in your SC-ALT account and check the `dashboard <https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/dashboard/arm/subscriptions/2aac527a-de5a-4fe3-95e9-5c8b9d48ed62/resourceGroups/services/providers/Microsoft.Portal/dashboards/71e7d10d-0769-410b-a4f5-d96c808dab85>`_.
