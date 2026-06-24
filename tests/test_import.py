def test_import_caligo():
    import caligo

    assert caligo is not None


def test_import_submodules():
    from caligo import background
    from caligo import diagnostics
    from caligo import microphysics
    from caligo import optics

    assert background is not None
    assert diagnostics is not None
    assert microphysics is not None
    assert optics is not None
