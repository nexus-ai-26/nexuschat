from scripts.copy_local_to_neon import foreign_key_order, vector_literal


def test_foreign_key_order_places_parents_first():
    tables = {"kb_topic_message", "message", "kbtopic", "group", "sender"}
    dependencies = [
        ("message", "sender"),
        ("message", "group"),
        ("kbtopic", "group"),
        ("kb_topic_message", "message"),
        ("kb_topic_message", "kbtopic"),
    ]

    ordered = foreign_key_order(tables, dependencies)
    positions = {table: ordered.index(table) for table in ordered}

    assert positions["sender"] < positions["message"]
    assert positions["group"] < positions["message"]
    assert positions["group"] < positions["kbtopic"]
    assert positions["message"] < positions["kb_topic_message"]
    assert positions["kbtopic"] < positions["kb_topic_message"]


def test_vector_literal_preserves_embedding_values():
    assert vector_literal([0.1, -0.2, 3]) == "[0.1,-0.2,3]"
    assert vector_literal(None) is None
