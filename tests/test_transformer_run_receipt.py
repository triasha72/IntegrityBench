from scripts.train_civil_comments_transformer import runtime_environment


class _CudaUnavailable:
    @staticmethod
    def is_available() -> bool:
        return False


class _TorchFixture:
    __version__ = "fixture"
    cuda = _CudaUnavailable()


def test_runtime_receipt_marks_a_cpu_only_run() -> None:
    receipt = runtime_environment(_TorchFixture())
    assert receipt["torch"] == "fixture"
    assert receipt["cuda_available"] is False
    assert receipt["cuda_device_count"] == 0
    assert receipt["cuda_device_name"] is None
