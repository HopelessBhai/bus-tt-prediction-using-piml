"""Tests for bus_tt.constants."""

from bus_tt.constants import (
    SECTION_LENGTH_M,
    TIME_WINDOW,
    SEQ_INPUT_DIM,
    CTX_DIM,
    TABULAR_INPUT_DIM,
    OUTPUT_DIM,
    DROP_SECTIONS,
    TEST_DATES,
    TRIGGER_THRESHOLD_S,
)


class TestConstants:
    def test_section_length(self):
        assert SECTION_LENGTH_M == 100.0

    def test_time_window(self):
        assert TIME_WINDOW == 2

    def test_seq_input_dim(self):
        assert SEQ_INPUT_DIM == 1

    def test_ctx_dim(self):
        assert CTX_DIM == 4

    def test_tabular_input_dim(self):
        assert TABULAR_INPUT_DIM == 6

    def test_output_dim(self):
        assert OUTPUT_DIM == 1

    def test_drop_sections_count(self):
        assert len(DROP_SECTIONS) == 5
        assert all(s.startswith("Section ") for s in DROP_SECTIONS)

    def test_test_dates_format(self):
        import re
        for d in TEST_DATES:
            assert re.match(r"\d{4}-\d{2}-\d{2}", d)

    def test_trigger_threshold(self):
        assert TRIGGER_THRESHOLD_S == 60.0
