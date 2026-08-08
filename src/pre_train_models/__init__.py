def build_model(kgs, args):
    """Create the backbone used by the training loop.

    Add a model to this package and return it here. The model must implement:

    - ``forward(batch) -> (loss, output, sub_embeddings)``
    - ``get_embeddings() -> entity_embeddings``

    ``output`` must contain ``loss_dic`` and may contain ``weight``.
    ``sub_embeddings`` is the list used by iterative selection.
    """
    raise NotImplementedError(
        "Add a backbone in src/pre_train_models and return it from build_model()."
    )
