import RESource.utility as utils


def setup_function():
    utils.reset_timings()
    utils.set_timing_context("")


def test_report_lists_only_slow_processes_and_total():
    timings = [("fast", 10.0), ("slow A", 400.0), ("slow B", 900.0)]
    report = utils.build_timing_report(1400.0, timings=timings)
    assert "TIMING REPORT" in report
    assert "Total time: 00h 23m 20s" in report
    assert "fast" not in report
    assert report.index("slow B") < report.index("slow A")


def test_report_none_slow():
    report = utils.build_timing_report(5.0, timings=[("fast", 1.0)])
    assert "none" in report


def test_steps_and_context_recorded():
    utils.set_timing_context("BC/wind")
    utils.start_step("Step 1")
    utils.start_step("Step 2")
    with utils.timed("inner"):
        pass
    utils.end_step()
    report = utils.build_timing_report(0.0, threshold_s=-1)
    assert "[BC/wind] Step 1" in report
    assert "[BC/wind] Step 2" in report
    assert "[BC/wind] inner" in report


def test_plot_layer_impacts_default_and_config():
    from RESource.AttributesParser import AttributesParser

    prop = AttributesParser.plot_layer_impacts
    assert isinstance(prop, property)
