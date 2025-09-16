Batch tracking
==============

The ``BatchTracker`` utility helps you record and query the sample indices that belong to each dataset contained in a batch. This information is invaluable when you need to:

* Debug data loading issues.
* Compute dataset specific metrics.
* Perform curriculum learning with multiple datasets.

Unlike ad-hoc tensor slicing, ``BatchTracker`` keeps a persistent mapping of dataset identifier - list of sample indices, so you can inspect it at any point during training.

Simply create an instance:

.. code-block:: python

    tracker = BatchTracker()

A new object starts with an empty internal dictionary named tracker that will be filled as you process batches.

Usage
-----

Tracking samples
^^^^^^^^^^^^^^^^

You can update the batch tracker with a new batch of data using the ``update`` method. The batch should be a dictionary containing tensors keyed by ``idx`` for sample indices and ``dataset_idx`` for dataset identifiers.

.. code-block:: python

    # Example batch data
    batch_data = {
        "idx": torch.tensor([0, 1, 2]),
        "dataset_idx": torch.tensor([1, 1, 2])
    }

    # Update the tracker with the batch
    tracker.update(batch_data)

Retrieving sample indices
^^^^^^^^^^^^^^^^^^^^^^^^^

To retrieve the sample indices for each dataset, access the ``samples_idx_per_dataset`` property:

.. code-block:: python

    indices_per_dataset = tracker.samples_idx_per_dataset
    print(indices_per_dataset)
    # Output: {1: [0, 1], 2: [2]}

Retrieving sample counts
^^^^^^^^^^^^^^^^^^^^^^^^

To get the number of samples for each dataset, access the ``n_samples_per_dataset`` property:

.. code-block:: python

    sample_counts = tracker.n_samples_per_dataset
    print(sample_counts)
    # Output: {1: 2, 2: 1}

Resetting the tracker
^^^^^^^^^^^^^^^^^^^^^

You can reset the tracker to its initial state by calling the ``reset`` method:

.. code-block:: python

    tracker.reset()

You typically want to reset the tracker at the beginning of each epoch (or evaluation loop) to avoid mixing information from different passes over the data.

DeepSpeed integration
---------------------

``BatchTracker`` was designed to work seamlessly with the ``DsTrainer`` class used in DeepSpeed training scripts. Internally, ``DsTrainer`` wraps your dataloader with a ``RepeatingLoader`` so that the dataset can be iterated infinitely.

If you pass a ``BatchTracker`` instance to ``RepeatingLoader``, every batch is automatically forwarded to update. This enables:

* Debugging: Easily verify which datasets contribute to gradient updates.
* Logging: Dump per-dataset statistics to disk for post-training analysis.
* Curriculum schedules: Plug the mapping into a scheduler that adjusts sampling probabilities on-the-fly.
