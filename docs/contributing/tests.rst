Tests
=====

The test suite consists of end-to-end (e2e) tests and unit tests. These tests help ensure the correctness and reliability of our project, enabling us to develop and maintain high-quality code.

E2e tests are full scripts that simulate real-world scenarios, test the entire functionality of the system, and ensure that all components of the system work together as expected in various situations. On the other hand, unit tests focus on testing individual components or functions in isolation, ensuring that each part of the system behaves correctly.

To run the tests, simply execute the following command:

.. code-block:: bash

    pytest tests  # Use --slow or --slowest to enable time-consuming tests

This command will discover and run all the e2e and unit tests in the specified folder, and display the results in the console.

.. important::
    Make sure to run the appropriate tests as needed when developing new features or fixing bugs.