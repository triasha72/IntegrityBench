import pytest

from scripts.train_civil_comments_transformer import require_cuda, runtime_environment


class _CudaUnavailable:
    @staticmethod
    def is_available() -> bool:
        return False


class _TorchFixture:
    __version__ = "fixture"
    cuda = _CudaUnavailable()


class _CudaAvailable:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_name(index: int) -> str:
        assert index == 0
        return "NVIDIA T4"


class _CudaTorchFixture:
    __version__ = "fixture"
    cuda = _CudaAvailable()


def test_runtime_receipt_marks_a_cpu_only_run() -> None:
    receipt = runtime_environment(_TorchFixture())
    assert receipt["torch"] == "fixture"
    assert receipt["cuda_available"] is False
    assert receipt["cuda_device_count"] == 0
    assert receipt["cuda_device_name"] is None
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        require_cuda(receipt)


def test_require_cuda_accepts_a_named_gpu() -> None:
    receipt = runtime_environment(_CudaTorchFixture())
    assert receipt["cuda_available"] is True
    assert receipt["cuda_device_count"] == 1
    assert receipt["cuda_device_name"] == "NVIDIA T4"
    require_cuda(receipt)
