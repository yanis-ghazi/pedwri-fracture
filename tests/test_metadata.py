from pedwri import ao
from pedwri.metadata import parse_filestem, read_yolo_labels


def test_parse_filestem_example_from_paper():
    d = parse_filestem("4229_0020600143_02_WRI-L1_F010")
    assert d["patient_id"] == "4229"
    assert d["study"] == "02"
    assert d["side"] == "L"
    assert d["projection"] == "ap"
    assert d["sex"] == "F"
    assert d["age"] == 10


def test_parse_filestem_rejects_other_names():
    assert parse_filestem("meta") is None


def test_ao_patterns():
    assert ao.patterns("23r-M/3.1; 23u-E/7") == {"metaphyseal complete", "avulsion"}
    assert ao.patterns("23r-E/2.1") == {"physeal (Salter-Harris)"}
    assert ao.patterns("") == set()
    assert ao.detailed("23-M/3.1") == {"radius+ulna M/3"}
    assert ao.detailed("23r-M/2.1; 23u-M/2.1") == {"radius M/2", "ulna M/2"}


def test_polygon_labels_become_boxes(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("5 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n3 0.5 0.5 0.2 0.1\n")
    rows = read_yolo_labels(f)
    assert rows[1] == (3, 0.5, 0.5, 0.2, 0.1)
    cls, cx, cy, w, h = rows[0]
    assert cls == 5 and abs(cx - 0.2) < 1e-9 and abs(cy - 0.3) < 1e-9
    assert abs(w - 0.2) < 1e-9 and abs(h - 0.2) < 1e-9
