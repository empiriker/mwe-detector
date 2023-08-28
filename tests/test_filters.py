import pytest
from spacy import load
from mwe_detector.filters import F1, F2, F3, F4, F5, F6, F7, ExampleType

nlp_en = load("en_core_web_sm")
nlp_fr = load("fr_core_news_sm")


@pytest.fixture
def setup_F1():
    filter = F1()
    doc1 = nlp_en("The quick brown fox jumps over the lazy dog.")
    doc2 = nlp_en("Time flies like an arrow; fruit flies like a banana.")
    return filter, doc1, doc2


def test_F1_add_example(setup_F1):
    filter, doc1, doc2 = setup_F1
    data = filter.default_data()
    example = ExampleType(
        lemma="example1",
        lemmas=["quick", "brown", "fox"],
        example=doc1,
        match_idx=(1, 2, 3),
        pos="VERB",
    )

    filter.add_example(data, example)
    assert len(data) == 1
    assert data[0] == ["ADJ", "ADJ", "NOUN"]


def test_F1_filter(setup_F1):
    filter, doc1, doc2 = setup_F1
    data = filter.default_data()
    example = ExampleType(
        lemma="example1",
        lemmas=["quick", "brown", "fox"],
        example=doc1,
        match_idx=(1, 2, 3),
        pos="VERB",
    )
    filter.add_example(data, example)

    match_idx = (1, 2, 3)
    result = filter.filter(data, doc1, match_idx)
    assert result

    match_idx = (0, 1, 2)
    result = filter.filter(data, doc1, match_idx)
    assert not result

    match_idx = (2, 3, 5)
    result = filter.filter(data, doc2, match_idx)
    assert not result


@pytest.fixture
def f2_doc1():
    return nlp_en("The quick brown fox jumps over the lazy dog.")


@pytest.fixture
def f2_doc2():
    return nlp_en("Time flies like an arrow; fruit flies like a banana.")


@pytest.fixture
def f3_doc1(f2_doc1):
    return f2_doc1


@pytest.fixture
def f3_doc2(f2_doc2):
    return f2_doc2


def test_f2_add_example(f2_doc1):
    f2_filter = F2()
    data = f2_filter.default_data()
    example = ExampleType(
        lemma="example2",
        lemmas=["jumps", "over", "the"],
        example=f2_doc1,
        match_idx=(4, 5, 6),
        pos="VERB",
    )
    f2_filter.add_example(data, example)
    assert len(data) == 1  # type: ignore
    assert data[0] == ["VERB", "ADP", "DET"]  # type: ignore


def test_f2_filter(f2_doc1, f2_doc2):
    f2_filter = F2()
    data = f2_filter.default_data()
    example = ExampleType(
        lemma="example2",
        lemmas=["jumps", "over", "the"],
        example=f2_doc1,
        match_idx=(4, 5, 6),
        pos="VERB",
    )
    f2_filter.add_example(data, example)

    result = f2_filter.filter(data, f2_doc1, (4, 5, 6))
    assert result

    result = f2_filter.filter(data, f2_doc1, (4, 6, 5))
    assert not result

    result = f2_filter.filter(data, f2_doc2, (2, 3, 5))
    assert not result


def test_f3_add_example(f3_doc1):
    f3_filter = F3()
    data = f3_filter.default_data()
    example = ExampleType(
        lemma="example3",
        lemmas=["The", "fox", "jumps"],
        example=f3_doc1,
        match_idx=(0, 3, 4),
        pos="VERB",
    )
    f3_filter.add_example(data, example)
    assert len(data) == 1  # type: ignore
    assert data[0] == ["DET", "ADJ", "ADJ", "NOUN", "VERB"]  # type: ignore


def test_f3_filter(f3_doc1, f3_doc2):
    f3_filter = F3()
    data = f3_filter.default_data()
    example = ExampleType(
        lemma="example3",
        lemmas=["The", "fox", "jumps"],
        example=f3_doc1,
        match_idx=(0, 3, 4),
        pos="VERB",
    )
    f3_filter.add_example(data, example)

    result = f3_filter.filter(data, f3_doc1, (0, 3, 4))
    assert result

    result = f3_filter.filter(data, f3_doc1, (0, 4, 3))
    assert result

    result = f3_filter.filter(data, f3_doc2, (0, 2, 4))
    assert not result


@pytest.fixture
def f4_doc1():
    return nlp_en("The quick brown fox jumps over the lazy dog.")


@pytest.fixture
def f4_doc2():
    return nlp_en("Time flies like an arrow; fruit flies like a banana.")


def test_f4_add_example(f4_doc1):
    f4_filter = F4()
    data = f4_filter.default_data()
    example = ExampleType(
        lemma="example4",
        lemmas=["The", "jumps", "dog"],
        example=f4_doc1,
        match_idx=(0, 4, 8),
        pos="VERB",
    )

    f4_filter.add_example(data, example)
    assert len(data) == 1  # type: ignore
    assert data[0] == 4  # type: ignore


def test_f4_filter(f4_doc1, f4_doc2):
    f4_filter = F4()
    data = f4_filter.default_data()
    example = ExampleType(
        lemma="example4",
        lemmas=["The", "jumps", "dog"],
        example=f4_doc1,
        match_idx=(0, 4, 8),
        pos="VERB",
    )
    f4_filter.add_example(data, example)

    assert f4_filter.filter(data, f4_doc1, (0, 4, 8))
    assert not f4_filter.filter(data, f4_doc1, (0, 3, 8))
    assert not f4_filter.filter(data, f4_doc2, (0, 2, 8))


@pytest.fixture
def f5_doc():
    return nlp_en(
        "Time flies like an arrow; fruit flies like a banana. People like these flies."
    )


def test_f5_filter(f5_doc):
    f5_filter = F5()

    assert not f5_filter.filter(None, f5_doc, (1, 2, 10))  # type: ignore
    assert f5_filter.filter(None, f5_doc, (7, 8, 10))  # type: ignore


@pytest.fixture
def f6_doc():
    return nlp_en("John, who lives in New York, likes apples.")


def test_f6_filter(f6_doc):
    f6_filter = F6()

    assert f6_filter.filter(None, f6_doc, (0, 8))  # type: ignore
    assert not f6_filter.filter(None, f6_doc, (0, 9))  # type: ignore
    assert f6_filter.filter(None, f6_doc, (0, 2))  # type: ignore
    assert not f6_filter.filter(None, f6_doc, (0, 5))  # type: ignore
    assert f6_filter.filter(None, f6_doc, (0, 3, 4, 5, 6))  # type: ignore
    assert f6_filter.filter(None, f6_doc, (0, 8, 9))  # type: ignore
    assert not f6_filter.filter(None, f6_doc, (0, 3, 4, 9))  # type: ignore


@pytest.fixture
def f7_doc_en():
    return nlp_en("The quick brown fox jumps over the lazy dogs.")


@pytest.fixture
def f7_doc_fr():
    return nlp_fr("La vache brune rapide saute par-dessus les chiens paresseux.")


def test_f7_filter_en(f7_doc_en):
    f7_filter = F7()

    assert f7_filter.filter([], f7_doc_en, (1, 2))  # type: ignore
    assert f7_filter.filter([], f7_doc_en, (3, 8))  # type: ignore

    example = ExampleType(
        lemma="test", lemmas=["dogs"], example=f7_doc_en, match_idx=(8,), pos="NOUN"
    )
    noun_morphs = f7_filter.default_data()
    f7_filter.add_example(noun_morphs, example)

    assert f7_filter.filter(noun_morphs, f7_doc_en, (8,))
    assert not f7_filter.filter(noun_morphs, f7_doc_en, (3,))
    assert not f7_filter.filter([], f7_doc_en, (3,))  # type: ignore


def test_f7_filter_fr(f7_doc_fr):
    f7_filter = F7()

    assert f7_filter.filter([], f7_doc_fr, (0, 2))  # type: ignore
    assert f7_filter.filter([], f7_doc_fr, (1, 9))  # type: ignore

    example = ExampleType(
        lemma="test", lemmas=["chiens"], example=f7_doc_fr, match_idx=(9,), pos="NOUN"
    )
    noun_morphs = f7_filter.default_data()
    f7_filter.add_example(noun_morphs, example)

    assert f7_filter.filter(noun_morphs, f7_doc_fr, (9,))
    assert not f7_filter.filter(noun_morphs, f7_doc_fr, (1,))
    assert not f7_filter.filter([], f7_doc_fr, (1,))  # type: ignore
