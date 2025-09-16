# Tests

This directory contains the test suite for verifying the correctness and reliability of PhyAGI. It is composed of two main types of tests, each serving a different purpose in ensuring code quality and system integrity:

- **End-to-End (E2E) tests**: Full scripts that simulate real-world scenarios and verify that system components work together as expected.

- **Unit tests**: Validate individual components or functions in isolation to ensure correctness at a granular level.

## Running Tests

To run all tests (both unit and end-to-end):

```bash
pytest ./
```

To include time-consuming tests, use one of the optional flags:

```bash
pytest ./ --slow     # Includes moderately slow tests
pytest ./ --slowest  # Includes all slowest tests
```

For more details on test coverage and guidelines, refer to the [test guide](../docs/contributing/tests.rst).